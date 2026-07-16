"""
src/training/train.py

Fine-tunes DeBERTa-v3-base on CUAD clause extraction as a token-classification
task (BIO tagging). Each token is labelled B-CLAUSE, I-CLAUSE, or O.

HYPERPARAMETER SEARCH
---------------------
Grid search over 6 configurations on the validation set.
Best configuration selected by macro-averaged F1 on val.
Test set is only evaluated once, on the best configuration.

All runs logged to MLflow under experiment "cuad-extraction".

REPRODUCIBILITY
---------------
Seed is set for Python, NumPy, and PyTorch at the start of every run.
"""

import os
os.environ.setdefault("CUDA_LAUNCH_BLOCKING", "1")

import json
import random
from pathlib import Path

import mlflow
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    get_linear_schedule_with_warmup,
)

ROOT = Path(__file__).parent.parent.parent
DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"
MODELS_DIR.mkdir(exist_ok=True)

MODEL_NAME = "microsoft/deberta-v3-base"
EXPERIMENT_NAME = "cuad-extraction"
SEED = 42

CLAUSE_TYPES = [
    "Governing Law",
    "Termination For Convenience",
    "Renewal Term",
    "Non-Compete",
    "Cap On Liability",
    "Anti-Assignment",
    "Audit Rights",
    "Exclusivity",
]

LABEL2ID = {"O": 0}
for i, clause in enumerate(CLAUSE_TYPES):
    LABEL2ID[f"B-{clause}"] = 2 * i + 1
    LABEL2ID[f"I-{clause}"] = 2 * i + 2
ID2LABEL = {v: k for k, v in LABEL2ID.items()}

HPARAM_GRID = [
    {"lr": 1e-5, "batch_size": 8,  "epochs": 3, "warmup_ratio": 0.1},
    {"lr": 1e-5, "batch_size": 8,  "epochs": 5, "warmup_ratio": 0.1},
    {"lr": 2e-5, "batch_size": 8,  "epochs": 3, "warmup_ratio": 0.1},
    {"lr": 2e-5, "batch_size": 16, "epochs": 3, "warmup_ratio": 0.1},
    {"lr": 5e-6, "batch_size": 8,  "epochs": 5, "warmup_ratio": 0.1},
    {"lr": 3e-5, "batch_size": 8,  "epochs": 2, "warmup_ratio": 0.15},
]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_jsonl(path: Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def tokenize_and_label(example: dict, tokenizer, max_length: int = 512) -> dict:
    encoding = tokenizer(
        example["context"], max_length=max_length,
        truncation=True, padding=False, return_offsets_mapping=True,
    )
    offsets = encoding["offset_mapping"]
    labels = [LABEL2ID["O"]] * len(offsets)
    clause = example["clause"]

    for answer_text in example["answers"]:
        start_char = example["context"].find(answer_text)
        if start_char == -1:
            continue
        end_char = start_char + len(answer_text)
        token_start = None
        token_end = None
        for idx, (tok_start, tok_end) in enumerate(offsets):
            if tok_start <= start_char < tok_end and token_start is None:
                token_start = idx
            if tok_start < end_char <= tok_end:
                token_end = idx
        if token_start is not None and token_end is not None:
            labels[token_start] = LABEL2ID[f"B-{clause}"]
            for i in range(token_start + 1, token_end + 1):
                labels[i] = LABEL2ID[f"I-{clause}"]

    for idx, (tok_start, tok_end) in enumerate(offsets):
        if tok_start == tok_end == 0:
            labels[idx] = -100

    encoding.pop("offset_mapping")
    encoding["labels"] = labels
    return encoding


class CUADDataset(Dataset):
    def __init__(self, examples: list[dict], tokenizer, max_length: int = 512):
        self.encodings = [tokenize_and_label(ex, tokenizer, max_length) for ex in examples]

    def __len__(self):
        return len(self.encodings)

    def __getitem__(self, idx):
        return {k: torch.tensor(v) for k, v in self.encodings[idx].items()}


def compute_f1(preds: list[int], labels: list[int]) -> dict[str, float]:
    from collections import defaultdict
    tp = defaultdict(int)
    fp = defaultdict(int)
    fn = defaultdict(int)

    for pred, label in zip(preds, labels):
        if label == -100:
            continue
        pred_clause = ID2LABEL.get(pred, "O")
        true_clause = ID2LABEL.get(label, "O")
        pred_c = pred_clause.split("-", 1)[-1] if pred_clause != "O" else "O"
        true_c = true_clause.split("-", 1)[-1] if true_clause != "O" else "O"
        if true_c != "O":
            if pred_c == true_c:
                tp[true_c] += 1
            else:
                fn[true_c] += 1
                if pred_c != "O":
                    fp[pred_c] += 1
        elif pred_c != "O":
            fp[pred_c] += 1

    results = {}
    for clause in CLAUSE_TYPES:
        t, f, p_err = tp[clause], fn[clause], fp[clause]
        precision = t / (t + p_err) if (t + p_err) > 0 else 0.0
        recall = t / (t + f) if (t + f) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        results[clause] = round(f1, 4)

    results["macro"] = round(float(np.mean(list(results.values()))), 4)
    return results


def train_one_config(config, train_examples, val_examples, tokenizer, run_name):
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Using device: {device}"
          + (f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""))

    model = AutoModelForTokenClassification.from_pretrained(
        MODEL_NAME, num_labels=len(LABEL2ID),
        id2label=ID2LABEL, label2id=LABEL2ID, ignore_mismatched_sizes=True,
    ).to(device)

    # max_length=384 keeps sequences clear of the 512 boundary where
    # DeBERTa-v2/v3's relative-position bucket lookup in
    # disentangled_attention_bias can index out of range and silently
    # read garbage (NaN) on CUDA instead of raising. This is the actual
    # source of the NaN pattern -- it happens identically at every LR
    # we tried, which rules out an optimizer/precision cause.
    train_ds = CUADDataset(train_examples, tokenizer, max_length=384)
    val_ds = CUADDataset(val_examples, tokenizer, max_length=384)
    collator = DataCollatorForTokenClassification(tokenizer)

    train_loader = DataLoader(train_ds, batch_size=config["batch_size"], shuffle=True, collate_fn=collator)
    val_loader = DataLoader(val_ds, batch_size=config["batch_size"] * 2, shuffle=False, collate_fn=collator)

    optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=0.01)
    total_steps = len(train_loader) * config["epochs"]
    warmup_steps = int(total_steps * config["warmup_ratio"])
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    with mlflow.start_run(run_name=run_name):
        mlflow.log_params({**config, "seed": SEED, "model": MODEL_NAME})
        best_val_f1 = 0.0
        model_path = MODELS_DIR / run_name

        for epoch in range(config["epochs"]):
            model.train()
            total_loss = 0.0
            n_valid_steps = 0
            n_skipped_loss = 0
            n_skipped_grad = 0
            for batch in train_loader:
                batch = {k: v.to(device) for k, v in batch.items()}

                # Sanity-check indices BEFORE the forward pass. An out-of-range
                # input_id (embedding lookup) or label (loss target) causes an
                # async CUDA illegal-memory-access that does not raise cleanly --
                # it silently corrupts the CUDA context, and every op after that
                # returns NaN/garbage for the rest of the process, regardless of
                # LR, batch size, or model version. That matches this run's
                # symptoms exactly, so we check explicitly rather than guess again.
                vocab_size = model.config.vocab_size
                bad_ids = batch["input_ids"][(batch["input_ids"] < 0) | (batch["input_ids"] >= vocab_size)]
                if bad_ids.numel() > 0:
                    raise ValueError(
                        f"input_ids out of range for vocab_size={vocab_size}: "
                        f"found values {bad_ids.unique().tolist()[:10]}"
                    )
                label_vals = batch["labels"]
                valid_label_mask = (label_vals == -100) | ((label_vals >= 0) & (label_vals < len(LABEL2ID)))
                if not torch.all(valid_label_mask):
                    bad_labels = label_vals[~valid_label_mask]
                    raise ValueError(
                        f"labels out of range for num_labels={len(LABEL2ID)}: "
                        f"found values {bad_labels.unique().tolist()[:10]}"
                    )

                out = model(**batch)
                loss = out.loss

                if torch.isnan(loss) or torch.isinf(loss):
                    n_skipped_loss += 1
                    optimizer.zero_grad()
                    continue

                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

                # Critical: DeBERTa-v3's disentangled attention can produce a
                # finite loss but NaN/Inf gradients during backward(). If we
                # call optimizer.step() here, NaN propagates into the model
                # weights permanently -- every subsequent batch becomes NaN
                # from then on, even with a completely different sequence.
                # Skip the step entirely rather than let this happen.
                if torch.isnan(grad_norm) or torch.isinf(grad_norm):
                    n_skipped_grad += 1
                    optimizer.zero_grad()
                    continue

                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                total_loss += loss.item()
                n_valid_steps += 1

            total_skipped = n_skipped_loss + n_skipped_grad
            if total_skipped > 0:
                print(f"  [WARNING] Skipped {total_skipped}/{len(train_loader)} batches "
                      f"({n_skipped_loss} NaN loss, {n_skipped_grad} NaN gradient)")

            avg_loss = total_loss / n_valid_steps if n_valid_steps > 0 else float("nan")
            mlflow.log_metric("train_loss", avg_loss, step=epoch)
            mlflow.log_metric("n_skipped_loss", n_skipped_loss, step=epoch)
            mlflow.log_metric("n_skipped_grad", n_skipped_grad, step=epoch)

            if n_valid_steps == 0:
                print(f"  [ERROR] Zero valid steps in epoch {epoch+1} -- "
                      f"this config is fundamentally unstable, moving to next config.")
                break

            model.eval()
            all_preds, all_labels = [], []
            with torch.no_grad():
                for batch in val_loader:
                    batch = {k: v.to(device) for k, v in batch.items()}
                    out = model(**batch)
                    preds = out.logits.argmax(dim=-1).cpu().numpy().flatten().tolist()
                    labels = batch["labels"].cpu().numpy().flatten().tolist()
                    all_preds.extend(preds)
                    all_labels.extend(labels)

            f1s = compute_f1(all_preds, all_labels)
            mlflow.log_metric("val_macro_f1", f1s["macro"], step=epoch)
            for clause, f1 in f1s.items():
                mlflow.log_metric(f"val_f1_{clause.replace(' ', '_')}", f1, step=epoch)

            print(f"  epoch {epoch+1}/{config['epochs']}  loss={avg_loss:.4f}  val_macro_f1={f1s['macro']:.4f}")

            if f1s["macro"] > best_val_f1:
                best_val_f1 = f1s["macro"]
                model.save_pretrained(model_path)
                tokenizer.save_pretrained(model_path)

        mlflow.log_metric("best_val_macro_f1", best_val_f1)
        mlflow.log_param("model_path", str(model_path))

    return best_val_f1, model_path


def main():
    from src.training.prepare_data import verify_lock
    verify_lock()

    train_examples = load_jsonl(DATA_DIR / "train.jsonl")
    val_examples = load_jsonl(DATA_DIR / "val.jsonl")
    print(f"Train: {len(train_examples)} | Val: {len(val_examples)}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    mlflow.set_experiment(EXPERIMENT_NAME)

    results = []
    for i, config in enumerate(HPARAM_GRID):
        run_name = f"run_{i+1}_lr{config['lr']}_bs{config['batch_size']}_ep{config['epochs']}"
        print(f"\n[{i+1}/{len(HPARAM_GRID)}] {run_name}")
        val_f1, model_path = train_one_config(config, train_examples, val_examples, tokenizer, run_name)
        results.append({"config": config, "val_f1": val_f1, "model_path": str(model_path), "run_name": run_name})
        print(f"  Best val macro F1: {val_f1:.4f}")

        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    best = max(results, key=lambda x: x["val_f1"])
    print(f"\nBest config: {best['run_name']}  val_macro_f1={best['val_f1']:.4f}")
    print(f"Model saved to: {best['model_path']}")

    selection_path = MODELS_DIR / "best_config.json"
    with open(selection_path, "w") as f:
        json.dump(best, f, indent=2)
    print(f"Selection saved -> {selection_path}")


if __name__ == "__main__":
    main()
