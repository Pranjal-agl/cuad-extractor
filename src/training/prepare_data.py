"""
src/training/prepare_data.py

Downloads CUAD from HuggingFace using Parquet-native datasets:
  - dvgodoy/CUAD_v1_Contract_Understanding_clause_classification  (labels)
  - dvgodoy/CUAD_v1_Contract_Understanding_PDF                    (full text)

Joins on file_name, keeps the clause types in src.config.CLAUSE_TYPES,
creates contract-level 70/15/15 train/val/test splits, writes JSONL +
splits.lock.

Example schema (one JSON object per line):
  contract     source file name (split key)
  clause       the clause type this example asks about
  context      a window of contract text
  spans        ALL target-clause annotations overlapping the window, as
               {"clause", "start", "end"} character offsets into `context`
  answers      text of the spans whose clause == `clause`
  has_answer   whether `clause` occurs in the window
  kind         "positive" (built around an annotation) or "negative"
               (random window from a contract that has no `clause` at all)

Changes from the first version, and why:
  * Clause names come from src.config and the script aborts if any of them
    yields zero examples (three names silently matched nothing before).
  * Offsets are exact. The old code .strip()-ed the context and the answer
    without adjusting `start_in_context`, so offsets could be off by a few
    characters.
  * Every annotated span inside a window is labelled, not just the anchor.
    Previously, other clauses (or a second span of the same clause) inside
    the window were labelled "O", teaching the model to ignore real clauses.
  * Negative windows. Every example used to be positive, so the model never
    saw a window that contains no clause of the type it is asked about.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from collections import Counter, defaultdict

from src.config import CLAUSE_TYPES, DATA_DIR, SEED

TRAIN_FRAC = 0.70
VAL_FRAC = 0.15
CONTEXT_CHARS = 512  # characters of context kept on each side of an annotation
NEG_WINDOW_CHARS = 1500  # size of a negative window (~350 tokens)
NEG_RATIO = 0.5  # negatives per positive, per split


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def load_cuad():
    from datasets import load_dataset

    print("Loading clause labels...")
    clauses_ds = load_dataset(
        "dvgodoy/CUAD_v1_Contract_Understanding_clause_classification", split="train"
    )
    print(f"  {len(clauses_ds)} rows, columns: {clauses_ds.column_names}")

    print("Loading full contract text...")
    text_ds = load_dataset("dvgodoy/CUAD_v1_Contract_Understanding_PDF", split="train")
    print(f"  {len(text_ds)} contracts")
    text_lookup = {row["file_name"]: row["text"] for row in text_ds.select_columns(["file_name", "text"])}
    return clauses_ds, text_lookup


def collect_annotations(rows, text_lookup: dict[str, str], clause_types=CLAUSE_TYPES):
    """Group valid target-clause annotations by contract.

    Returns ({contract: [{"clause", "start", "end"}]}, stats).
    """
    targets = set(clause_types)
    by_contract: dict[str, list[dict]] = defaultdict(list)
    stats = Counter()
    for row in rows:
        label = row.get("label", "")
        if label not in targets:
            stats["other_label"] += 1
            continue
        file_name = row.get("file_name", "")
        text = text_lookup.get(file_name)
        if text is None:
            stats["missing_text"] += 1
            continue
        start, end = row.get("start_at"), row.get("end_at")
        if start is None or end is None or not (0 <= start < end <= len(text)):
            stats["bad_offsets"] += 1
            continue
        # Trim whitespace by moving the offsets, never by editing the text.
        while start < end and text[start].isspace():
            start += 1
        while end > start and text[end - 1].isspace():
            end -= 1
        if end <= start:
            stats["empty_span"] += 1
            continue
        clause_text = row.get("clause")
        if isinstance(clause_text, str) and clause_text.strip():
            stats["alignment_checked"] += 1
            if " ".join(clause_text.split())[:40] in " ".join(text[start:end].split()):
                stats["alignment_ok"] += 1
        by_contract[file_name].append({"clause": label, "start": start, "end": end})
        stats[f"kept:{label}"] += 1

    for anns in by_contract.values():
        anns.sort(key=lambda a: (a["start"], a["end"], a["clause"]))
        # drop exact duplicate annotations
        anns[:] = [a for i, a in enumerate(anns) if i == 0 or a != anns[i - 1]]
    return dict(by_contract), stats


def _snap_left(text: str, pos: int) -> int:
    """Move a window start left to the previous whitespace so words aren't cut."""
    pos = max(0, pos)
    while pos > 0 and not text[pos - 1].isspace():
        pos -= 1
    return pos


def _snap_right(text: str, pos: int) -> int:
    pos = min(len(text), pos)
    while pos < len(text) and not text[pos].isspace():
        pos += 1
    return pos


def _make_example(contract, clause, text, win_start, win_end, annotations, kind):
    spans = []
    for a in annotations:
        s, e = max(a["start"], win_start), min(a["end"], win_end)
        if e > s:
            spans.append({"clause": a["clause"], "start": s - win_start, "end": e - win_start})
    context = text[win_start:win_end]
    answers = [context[sp["start"]:sp["end"]] for sp in spans if sp["clause"] == clause]
    return {
        "contract": contract,
        "clause": clause,
        "context": context,
        "spans": spans,
        "answers": answers,
        "has_answer": bool(answers),
        "kind": kind,
    }


def build_examples(
    annotations: dict[str, list[dict]],
    text_lookup: dict[str, str],
    neg_ratio: float = NEG_RATIO,
    context_chars: int = CONTEXT_CHARS,
    seed: int = SEED,
    clause_types=CLAUSE_TYPES,
) -> dict[str, list[dict]]:
    """Build positive + negative examples, grouped by contract."""
    rng = random.Random(seed)
    by_contract: dict[str, list[dict]] = {}
    for contract in sorted(annotations):
        text, anns = text_lookup[contract], annotations[contract]
        examples = []
        for a in anns:
            ws = _snap_left(text, a["start"] - context_chars)
            we = _snap_right(text, a["end"] + context_chars)
            examples.append(_make_example(contract, a["clause"], text, ws, we, anns, "positive"))

        # Negative candidates: clause types this contract does not contain at all.
        # CUAD annotators reviewed every contract for every category, so absence
        # of an annotation means absence of the clause.
        present = {a["clause"] for a in anns}
        absent = [c for c in clause_types if c not in present]
        n_neg = int(round(neg_ratio * len(examples)))
        if absent and n_neg and len(text) > 0:
            for _ in range(n_neg):
                clause = rng.choice(absent)
                ws = _snap_left(text, rng.randrange(0, max(1, len(text) - NEG_WINDOW_CHARS)))
                we = _snap_right(text, ws + NEG_WINDOW_CHARS)
                if text[ws:we].strip():
                    examples.append(_make_example(contract, clause, text, ws, we, anns, "negative"))
        by_contract[contract] = examples
    return by_contract


def split_contracts(by_contract: dict, seed: int = SEED):
    contracts = sorted(by_contract.keys())
    rng = random.Random(seed)
    rng.shuffle(contracts)
    n = len(contracts)
    n_train = int(n * TRAIN_FRAC)
    n_val = int(n * VAL_FRAC)
    train_c = set(contracts[:n_train])
    val_c = set(contracts[n_train:n_train + n_val])
    train, val, test = [], [], []
    for c in sorted(by_contract):
        target = train if c in train_c else val if c in val_c else test
        target.extend(by_contract[c])
    return train, val, test


def contamination_check(train, val, test):
    tc = {e["contract"] for e in train}
    vc = {e["contract"] for e in val}
    ec = {e["contract"] for e in test}
    violations = []
    if tc & vc:
        violations.append(f"train/val overlap: {len(tc & vc)} contracts")
    if tc & ec:
        violations.append(f"train/test overlap: {len(tc & ec)} contracts")
    if vc & ec:
        violations.append(f"val/test overlap: {len(vc & ec)} contracts")
    if violations:
        print("CONTAMINATION DETECTED:")
        for v in violations:
            print(f"  {v}")
        sys.exit(1)
    print(f"Contamination check passed. train={len(tc)}, val={len(vc)}, test={len(ec)} contracts")


def coverage_check(examples: list[dict], clause_types=CLAUSE_TYPES) -> None:
    """Abort if a configured clause type produced no positive examples."""
    counts = Counter(e["clause"] for e in examples if e["has_answer"])
    missing = [c for c in clause_types if counts.get(c, 0) == 0]
    if missing:
        print(f"ERROR: no examples for clause types {missing}. Check the names in "
              "src/config.py against the dataset's `label` values.")
        sys.exit(1)


def example_spans(example: dict) -> list[dict]:
    """Gold spans of an example as {"clause", "start", "end"} dicts.

    Supports both the current schema ("spans") and JSONL files written by
    the first version of this script ("answers" + "start_in_context").
    """
    if "spans" in example:
        return example["spans"]
    spans = []
    for ans in example.get("answers", []):
        start = example.get("start_in_context", -1)
        if start is None or start < 0 or example["context"][start:start + len(ans)] != ans:
            start = example["context"].find(ans)
        if start >= 0:
            spans.append({"clause": example["clause"], "start": start, "end": start + len(ans)})
    return spans


def _examples_hash(examples: list[dict]) -> str:
    return _sha256("\n".join(json.dumps(e, sort_keys=True) for e in examples))


def write_splits(train, val, test, extra_lock: dict | None = None):
    DATA_DIR.mkdir(exist_ok=True)
    for name, split in [("train", train), ("val", val), ("test", test)]:
        path = DATA_DIR / f"{name}.jsonl"
        with open(path, "w") as f:
            for ex in split:
                f.write(json.dumps(ex) + "\n")
        pos = sum(1 for e in split if e["has_answer"])
        print(f"  {name}: {len(split)} examples ({pos} positive) -> {path}")

    test_contracts = sorted({e["contract"] for e in test})
    lock = {
        "test_contracts_hash": _sha256(json.dumps(test_contracts)),
        "test_examples_hash": _examples_hash(test),
        "n_test_contracts": len(test_contracts),
        "n_test_examples": len(test),
        "seed": SEED,
        "clause_types": list(CLAUSE_TYPES),
        **(extra_lock or {}),
    }
    with open(DATA_DIR / "splits.lock", "w") as f:
        json.dump(lock, f, indent=2)
    print(f"  Lock written -> {DATA_DIR / 'splits.lock'}")


def verify_lock() -> bool:
    """Check the test split still matches splits.lock. Exits on mismatch.

    Returns False (with a warning) if there is no lock to check against.
    """
    lock_path = DATA_DIR / "splits.lock"
    test_path = DATA_DIR / "test.jsonl"
    if not lock_path.exists() or not test_path.exists():
        print("WARNING: no splits.lock/test.jsonl found -- test set integrity NOT verified.")
        return False
    with open(lock_path) as f:
        lock = json.load(f)
    with open(test_path) as f:
        test = [json.loads(line) for line in f if line.strip()]
    test_contracts = sorted({e["contract"] for e in test})
    if _sha256(json.dumps(test_contracts)) != lock["test_contracts_hash"]:
        print("TEST SET CHANGED -- splits.lock contract mismatch. Aborting.")
        sys.exit(1)
    if "test_examples_hash" in lock and _examples_hash(test) != lock["test_examples_hash"]:
        print("TEST SET CHANGED -- splits.lock example mismatch. Aborting.")
        sys.exit(1)
    if lock.get("clause_types") and lock["clause_types"] != list(CLAUSE_TYPES):
        print("CLAUSE_TYPES in src/config.py differ from the ones the splits were built "
              "with. Re-run prepare_data. Aborting.")
        sys.exit(1)
    print("Test set lock verified.")
    return True


def print_distribution(name: str, split: list[dict]) -> None:
    pos = Counter(e["clause"] for e in split if e["has_answer"])
    neg = Counter(e["clause"] for e in split if not e["has_answer"])
    print(f"\n{name} clause distribution (positive / negative):")
    for clause in CLAUSE_TYPES:
        print(f"  {clause:<30} {pos.get(clause, 0):5d} / {neg.get(clause, 0):5d}")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--neg-ratio", type=float, default=NEG_RATIO,
                        help="negative windows per positive example (0 disables)")
    parser.add_argument("--context-chars", type=int, default=CONTEXT_CHARS)
    args = parser.parse_args()

    clauses_ds, text_lookup = load_cuad()
    annotations, stats = collect_annotations(clauses_ds, text_lookup)
    print(f"  Skipped: {stats['other_label']} other-label rows, {stats['missing_text']} missing "
          f"text, {stats['bad_offsets']} bad offsets, {stats['empty_span']} empty spans")
    if stats["alignment_checked"]:
        rate = stats["alignment_ok"] / stats["alignment_checked"]
        print(f"  Offset alignment check: {rate:.1%} of annotations match their clause text")

    by_contract = build_examples(annotations, text_lookup, neg_ratio=args.neg_ratio,
                                 context_chars=args.context_chars)
    total = sum(len(v) for v in by_contract.values())
    print(f"Found {len(by_contract)} contracts, {total} examples")

    train, val, test = split_contracts(by_contract)
    print(f"Split: {len(train)} train / {len(val)} val / {len(test)} test")

    contamination_check(train, val, test)
    coverage_check(train + val + test)
    write_splits(train, val, test, extra_lock={"neg_ratio": args.neg_ratio,
                                               "context_chars": args.context_chars})
    for name, split in [("train", train), ("val", val), ("test", test)]:
        print_distribution(name, split)


if __name__ == "__main__":
    main()
