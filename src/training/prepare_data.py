"""
src/training/prepare_data.py

Downloads CUAD from HuggingFace using Parquet-native datasets:
  - dvgodoy/CUAD_v1_Contract_Understanding_clause_classification
  - dvgodoy/CUAD_v1_Contract_Understanding_PDF

Joins on file_name, filters to 8 clause types, creates contract-level
70/15/15 train/val/test splits, writes JSONL + splits.lock.
"""

import hashlib
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent
DATA_DIR = ROOT / "data"
DATA_DIR.mkdir(exist_ok=True)

CLAUSE_TYPES = [
    "Governing Law",
    "Termination For Convenience",
    "Renewal Term",
    "Non-Compete",
    "Confidentiality Of Agreements",
    "Indemnification",
    "Cap On Liability",
    "Assignment",
]

SEED = 42
TRAIN_FRAC = 0.70
VAL_FRAC = 0.15


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def load_cuad():
    from datasets import load_dataset
    print("Loading clause labels...")
    clauses_ds = load_dataset(
        "dvgodoy/CUAD_v1_Contract_Understanding_clause_classification",
        split="train",
    )
    print(f"  {len(clauses_ds)} rows, columns: {clauses_ds.column_names}")

    print("Loading full contract text...")
    text_ds = load_dataset(
        "dvgodoy/CUAD_v1_Contract_Understanding_PDF",
        split="train",
    )
    print(f"  {len(text_ds)} contracts")
    text_lookup = {row["file_name"]: row["text"] for row in text_ds}
    return clauses_ds, text_lookup


def build_examples(clauses_ds, text_lookup: dict) -> dict[str, list[dict]]:
    target_labels = set(CLAUSE_TYPES)
    by_contract: dict[str, list] = {}
    skipped = 0

    for row in clauses_ds:
        label = row.get("label", "")
        if label not in target_labels:
            skipped += 1
            continue

        file_name = row.get("file_name", "")
        if file_name not in text_lookup:
            skipped += 1
            continue

        full_text = text_lookup[file_name]
        start = row.get("start_at") or 0
        end = row.get("end_at") or 0
        has_answer = end > start

        if has_answer:
            answer_text = full_text[start:end].strip()
            if not answer_text:
                has_answer = False
                answer_text = ""
        else:
            answer_text = ""

        ctx_start = max(0, start - 512)
        ctx_end = min(len(full_text), end + 512)
        context = full_text[ctx_start:ctx_end].strip()

        if not context:
            continue

        example = {
            "contract": file_name,
            "clause": label,
            "context": context,
            "answers": [answer_text] if has_answer and answer_text else [],
            "has_answer": has_answer and bool(answer_text),
            "start_in_context": start - ctx_start if has_answer else -1,
        }
        by_contract.setdefault(file_name, []).append(example)

    print(f"  Skipped {skipped} rows (not in target types or missing text)")
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
    for c, examples in by_contract.items():
        if c in train_c:
            train.extend(examples)
        elif c in val_c:
            val.extend(examples)
        else:
            test.extend(examples)
    return train, val, test


def contamination_check(train, val, test):
    tc = {e["contract"] for e in train}
    vc = {e["contract"] for e in val}
    ec = {e["contract"] for e in test}
    violations = []
    if tc & vc: violations.append(f"train/val overlap: {len(tc & vc)} contracts")
    if tc & ec: violations.append(f"train/test overlap: {len(tc & ec)} contracts")
    if vc & ec: violations.append(f"val/test overlap: {len(vc & ec)} contracts")
    if violations:
        print("CONTAMINATION DETECTED:")
        for v in violations: print(f"  {v}")
        sys.exit(1)
    print(f"Contamination check passed. train={len(tc)}, val={len(vc)}, test={len(ec)} contracts")


def write_splits(train, val, test):
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
        "n_test_contracts": len(test_contracts),
        "n_test_examples": len(test),
        "seed": SEED,
        "clause_types": CLAUSE_TYPES,
    }
    with open(DATA_DIR / "splits.lock", "w") as f:
        json.dump(lock, f, indent=2)
    print(f"  Lock written -> {DATA_DIR / 'splits.lock'}")


def verify_lock():
    lock_path = DATA_DIR / "splits.lock"
    test_path = DATA_DIR / "test.jsonl"
    if not lock_path.exists() or not test_path.exists():
        return
    with open(lock_path) as f:
        lock = json.load(f)
    with open(test_path) as f:
        test = [json.loads(l) for l in f if l.strip()]
    test_contracts = sorted({e["contract"] for e in test})
    if _sha256(json.dumps(test_contracts)) != lock["test_contracts_hash"]:
        print("TEST SET CHANGED -- splits.lock mismatch. Aborting.")
        sys.exit(1)
    print("Test set lock verified.")


def main():
    clauses_ds, text_lookup = load_cuad()
    by_contract = build_examples(clauses_ds, text_lookup)
    total = sum(len(v) for v in by_contract.values())
    print(f"Found {len(by_contract)} contracts, {total} examples")

    train, val, test = split_contracts(by_contract)
    print(f"Split: {len(train)} train / {len(val)} val / {len(test)} test")

    contamination_check(train, val, test)
    write_splits(train, val, test)

    from collections import Counter
    for name, split in [("train", train), ("val", val), ("test", test)]:
        counts = Counter(e["clause"] for e in split)
        positives = Counter(e["clause"] for e in split if e["has_answer"])
        print(f"\n{name} clause distribution:")
        for clause in CLAUSE_TYPES:
            n = counts.get(clause, 0)
            p = positives.get(clause, 0)
            if n:
                print(f"  {clause:<35} {p:4d}/{n:4d} ({100*p/n:.0f}%)")


if __name__ == "__main__":
    main()
