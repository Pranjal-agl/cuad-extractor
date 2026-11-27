"""
src/training/train.py

Fine-tunes DeBERTa-v3-base on CUAD clause extraction as a token-classification
task (BIO tagging): each token is labelled B-<clause>, I-<clause>, or O.

HYPERPARAMETER SEARCH
---------------------
Grid search over HPARAM_GRID on the validation set. The selection metric is
the SAME span-level macro F1 that evaluate.py reports on the test set, so the
val-vs-test comparison is apples to apples. (The first version selected on a
token-level F1 averaged over 8 clause types, 3 of which had no data, and then
compared that number to a span-level test F1 averaged over 5.)
Test data is never touched here.

All runs are logged to MLflow (store configured in src.config).

USAGE
-----
    python -m src.training.train                 # full grid
    python -m src.training.train --configs 3,7   # subset (1-based indices)
    python -m src.training.train --prune         # delete non-best checkpoints
"""

from __future__ import annotations

import argparse
import gc
import json
import shutil
import subprocess
from collections import Counter

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    get_linear_schedule_with_warmup,
)

from src.config import (
    CLAUSE_TYPES,
    DATA_DIR,
    ID2LABEL,
    LABEL2ID,
    MAX_LENGTH,
    MODEL_NAME,
    MODELS_DIR,
    NUM_LABELS,
    ROOT,
    SEED,
    STRIDE,
    setup_mlflow,
)
from src.metrics import set_seed
from src.spans import IGNORE_INDEX, label_tokens, token_level_f1
from src.training.prepare_data import example_spans

# class_weighting:
#   inverse       weight_c = N / (K * count_c). What the first run used. It
#                 fixed the all-"O" collapse but over-corrected: the model
#                 predicted 2.3x more clause tokens than exist (34,335 vs
#                 14,946 on val) and precision trails recall on every clause.
#   sqrt_inverse  square root of the above -- the usual softer variant.
HPARAM_GRID = [
    {"lr": 1e-5, "batch_size": 8,  "epochs": 3, "warmup_ratio": 0.1,  "class_weighting": "inverse"},
    {"lr": 1e-5, "batch_size": 8,  "epochs": 5, "warmup_ratio": 0.1,  "class_weighting": "inverse"},
    {"lr": 2e-5, "batch_size": 8,  "epochs": 3, "warmup_ratio": 0.1,  "class_weighting": "inverse"},
    {"lr": 2e-5, "batch_size": 16, "epochs": 3, "warmup_ratio": 0.1,  "class_weighting": "inverse"},
    {"lr": 5e-6, "batch_size": 8,  "epochs": 5, "warmup_ratio": 0.1,  "class_weighting": "inverse"},
    {"lr": 3e-5, "batch_size": 8,  "epochs": 2, "warmup_ratio": 0.15, "class_weighting": "inverse"},
    {"lr": 2e-5, "batch_size": 8,  "epochs": 3, "warmup_ratio": 0.1,  "class_weighting": "sqrt_inverse"},
]


def load_jsonl(path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def tokenize_and_label(example: dict, tokenizer, max_length: int = MAX_LENGTH,
                       stride: int = STRIDE) -> list[dict]:
    """Tokenize one example into one or more overlapping windows with BIO labels.

    Contexts longer than `max_length` tokens used to be truncated, and a span
    running past the cut lost its labels entirely (the example silently
    became all-"O"). Windows keep every annotated token.
    """
    enc = tokenizer(
        example["context"], max_length=max_length, truncation=True, stride=stride,
        return_overflowing_tokens=True, return_offsets_mapping=True, padding=False,
    )
    spans = example_spans(example)
    windows = []
    for i in range(len(enc["input_ids"])):
        windows.append({
            "input_ids": enc["input_ids"][i],
            "attention_mask": enc["attention_mask"][i],
            "labels": label_tokens(enc["offset_mapping"][i], spans, LABEL2ID),
        })
    return windows


class CUADDataset(Dataset):
    def __init__(self, examples: list[dict], tokenizer, max_length: int = MAX_LENGTH):
        self.encodings = [w for ex in examples for w in tokenize_and_label(ex, tokenizer, max_length)]
        # Validate once here instead of on every batch inside the training loop.
        vocab = len(tokenizer)
        for enc in self.encodings:
            if max(enc["input_ids"]) >= vocab or min(enc["input_ids"]) < 0:
                raise ValueError(f"input_ids out of range for tokenizer vocab size {vocab}")
            if any(lab != IGNORE_INDEX and not 0 <= lab < NUM_LABELS for lab in enc["labels"]):
                raise ValueError("label id out of range")

    def __len__(self):
        return len(self.encodings)

    def __getitem__(self, idx):
        return {k: torch.tensor(v) for k, v in self.encodings[idx].items()}


def class_weights(dataset: CUADDataset, scheme: str) -> torch.Tensor | None:
    if scheme == "none":
        return None
    counts = torch.zeros(NUM_LABELS)
    for enc in dataset.encodings:
        for lab in enc["labels"]:
            if lab != IGNORE_INDEX:
                counts[lab] += 1
    seen = counts > 0
    weights = torch.ones(NUM_LABELS)
    # Labels that never occur get weight 1 (they used to get ~20,000x).
    weights[seen] = counts[seen].sum() / (int(seen.sum()) * counts[seen])
    if scheme == "sqrt_inverse":
        weights = weights.sqrt()
    elif scheme != "inverse":
        raise ValueError(f"unknown class_weighting {scheme!r}")
    return weights


def _git_commit() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True,
                                       stderr=subprocess.DEVNULL).strip()
    except Exception:
        return None


def validate(model, tokenizer, val_examples, val_loader, device) -> dict:
    """Span-level metrics (selection metric) + token-level F1 (diagnostic)."""
    from src.eval.evaluate import run_eval
    from src.inference import ClauseExtractor

    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in val_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            logits = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]).logits
            all_preds.extend(logits.argmax(-1).flatten().tolist())
            all_labels.extend(batch["labels"].flatten().tolist())
    tok = token_level_f1(all_preds, all_labels, ID2LABEL, CLAUSE_TYPES)

    pred_dist = Counter(p for p, lab in zip(all_preds, all_labels) if lab != IGNORE_INDEX and p != 0)
    n_true = sum(1 for lab in all_labels if lab not in (IGNORE_INDEX, 0))
    span = run_eval(ClauseExtractor(model, tokenizer), val_examples, split_name="val", verbose=False)
    return {"span": span, "token": tok,
            "pred_non_o": sum(pred_dist.values()), "true_non_o": n_true}


def train_one_config(config, train_ds, val_examples, val_ds, tokenizer, run_name, mlflow):
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Using device: {device}"
          + (f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""))

    model = AutoModelForTokenClassification.from_pretrained(
        MODEL_NAME, num_labels=NUM_LABELS, id2label=ID2LABEL, label2id=LABEL2ID,
        ignore_mismatched_sizes=True,
    ).to(device)

    collator = DataCollatorForTokenClassification(tokenizer)
    weights = class_weights(train_ds, config.get("class_weighting", "inverse"))
    if weights is not None:
        print(f"  Class weights ({config.get('class_weighting')}): min={weights.min():.3f} "
              f"max={weights.max():.3f} O={weights[LABEL2ID['O']]:.3f}")
        weights = weights.to(device)
    loss_fct = torch.nn.CrossEntropyLoss(weight=weights, ignore_index=IGNORE_INDEX)

    train_loader = DataLoader(train_ds, batch_size=config["batch_size"], shuffle=True, collate_fn=collator,
                              generator=torch.Generator().manual_seed(SEED))
    val_loader = DataLoader(val_ds, batch_size=config["batch_size"] * 2, shuffle=False, collate_fn=collator)

    # eps=1e-6: DeBERTa-v3's gradient-disentangled embedding sharing produces
    # tiny early gradients; with AdamW's default eps=1e-8 the adaptive step
    # can spike and the loss goes NaN after one step.
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=0.01, eps=1e-6)
    total_steps = len(train_loader) * config["epochs"]
    scheduler = get_linear_schedule_with_warmup(optimizer, int(total_steps * config["warmup_ratio"]), total_steps)

    model_path = MODELS_DIR / run_name
    best_val_f1 = -1.0
    best_epoch = None

    with mlflow.start_run(run_name=run_name):
        mlflow.log_params({**config, "seed": SEED, "model": MODEL_NAME, "max_length": MAX_LENGTH,
                           "stride": STRIDE, "n_train_windows": len(train_ds),
                           "git_commit": _git_commit() or "unknown"})

        for epoch in range(config["epochs"]):
            model.train()
            total_loss, n_valid, n_bad_loss, n_bad_grad = 0.0, 0, 0, 0
            for batch in train_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                out = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
                loss = loss_fct(out.logits.view(-1, NUM_LABELS), batch["labels"].view(-1))
                if not torch.isfinite(loss):
                    n_bad_loss += 1
                    optimizer.zero_grad()
                    continue
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                # A finite loss can still produce NaN/Inf gradients; stepping on
                # them would poison the weights for the rest of the run.
                if not torch.isfinite(grad_norm):
                    n_bad_grad += 1
                    optimizer.zero_grad()
                    continue
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                total_loss += loss.item()
                n_valid += 1

            if n_bad_loss or n_bad_grad:
                print(f"  [WARNING] skipped {n_bad_loss + n_bad_grad}/{len(train_loader)} batches "
                      f"({n_bad_loss} non-finite loss, {n_bad_grad} non-finite grad)")
            avg_loss = total_loss / n_valid if n_valid else float("nan")
            mlflow.log_metrics({"train_loss": avg_loss, "n_skipped_loss": n_bad_loss,
                                "n_skipped_grad": n_bad_grad}, step=epoch)
            if n_valid == 0:
                print(f"  [ERROR] zero valid steps in epoch {epoch + 1}; abandoning this config.")
                break

            v = validate(model, tokenizer, val_examples, val_loader, device)
            span_f1 = v["span"]["macro_f1"]
            mlflow.log_metrics({"val_span_macro_f1": span_f1, "val_token_macro_f1": v["token"]["macro"],
                                "val_pred_non_o_tokens": v["pred_non_o"]}, step=epoch)
            for clause, m in v["span"]["per_clause"].items():
                if m["f1"] is not None:
                    mlflow.log_metric(f"val_f1_{clause.replace(' ', '_').replace('-', '_')}", m["f1"], step=epoch)
            print(f"  epoch {epoch + 1}/{config['epochs']}  loss={avg_loss:.4f}  "
                  f"val_span_macro_f1={span_f1:.4f}  val_token_macro_f1={v['token']['macro']:.4f}  "
                  f"non-O tokens pred/true={v['pred_non_o']}/{v['true_non_o']}")

            if span_f1 > best_val_f1:
                best_val_f1, best_epoch = span_f1, epoch + 1
                model.save_pretrained(model_path)
                tokenizer.save_pretrained(model_path)

        if not model_path.exists():  # always leave a loadable checkpoint behind
            model.save_pretrained(model_path)
            tokenizer.save_pretrained(model_path)
        mlflow.log_metric("best_val_span_macro_f1", max(best_val_f1, 0.0))
        mlflow.log_params({"model_path": str(model_path), "best_epoch": best_epoch})

    del model, optimizer, scheduler
    return max(best_val_f1, 0.0), model_path, best_epoch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", default=None,
                        help="comma-separated 1-based indices into HPARAM_GRID (default: all)")
    parser.add_argument("--prune", action="store_true", help="delete non-best run checkpoints at the end")
    args = parser.parse_args()

    from src.training.prepare_data import verify_lock
    verify_lock()

    indices = [int(i) for i in args.configs.split(",")] if args.configs else range(1, len(HPARAM_GRID) + 1)
    MODELS_DIR.mkdir(exist_ok=True)
    train_examples = load_jsonl(DATA_DIR / "train.jsonl")
    val_examples = load_jsonl(DATA_DIR / "val.jsonl")
    print(f"Train: {len(train_examples)} | Val: {len(val_examples)}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    train_ds = CUADDataset(train_examples, tokenizer)
    val_ds = CUADDataset(val_examples, tokenizer)
    print(f"Windows: {len(train_ds)} train / {len(val_ds)} val (max_length={MAX_LENGTH}, stride={STRIDE})")
    mlflow = setup_mlflow()

    results = []
    for i in indices:
        config = HPARAM_GRID[i - 1]
        run_name = (f"run_{i}_lr{config['lr']}_bs{config['batch_size']}_ep{config['epochs']}"
                    f"_{config['class_weighting']}")
        print(f"\n[{i}/{len(HPARAM_GRID)}] {run_name}")
        val_f1, model_path, best_epoch = train_one_config(
            config, train_ds, val_examples, val_ds, tokenizer, run_name, mlflow)
        results.append({"config": config, "val_f1": val_f1, "best_epoch": best_epoch,
                        "model_path": str(model_path.relative_to(ROOT)), "run_name": run_name})
        print(f"  Best val span macro F1: {val_f1:.4f} (epoch {best_epoch})")
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    best = max(results, key=lambda r: r["val_f1"])
    print(f"\nBest config: {best['run_name']}  val_span_macro_f1={best['val_f1']:.4f}")
    selection = {**best, "selection_metric": "val_span_macro_f1", "all_runs": results}
    (MODELS_DIR / "best_config.json").write_text(json.dumps(selection, indent=2))
    print(f"Selection saved -> {MODELS_DIR / 'best_config.json'}")

    if args.prune:
        for r in results:
            if r["run_name"] != best["run_name"]:
                shutil.rmtree(ROOT / r["model_path"], ignore_errors=True)
        print("Pruned non-best checkpoints.")

    summary = sorted(results, key=lambda r: -r["val_f1"])
    print("\nRun summary (val span macro F1):")
    for r in summary:
        print(f"  {r['val_f1']:.4f}  {r['run_name']}")
    print(f"Mean across runs: {np.mean([r['val_f1'] for r in results]):.4f}")


if __name__ == "__main__":
    main()
