"""
src/spans.py

Pure-Python helpers for converting between character spans and BIO token
labels, decoding model predictions back into spans, and scoring spans.

No torch / transformers imports, so everything here is unit-testable in
milliseconds. Spans are always half-open character ranges [start, end) into
the original text, which avoids the `text.find(span)` pattern the original
code used (that silently picks the first occurrence when a phrase repeats).
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Sequence

IGNORE_INDEX = -100


def _is_special(tok_start: int, tok_end: int) -> bool:
    # Fast tokenizers give special tokens ([CLS], [SEP], padding) an empty
    # (0, 0) offset. Any zero-width token carries no text either.
    return tok_end <= tok_start


def label_tokens(
    offsets: Sequence[Sequence[int]],
    spans: Iterable[dict],
    label2id: dict[str, int],
) -> list[int]:
    """Assign a BIO label id to every token.

    A token belongs to a span if their character ranges overlap at all. This
    is more robust than the original exact-boundary match, which dropped the
    whole span whenever the answer started or ended mid-token, and which
    dropped spans that ran past the truncation point.

    Spans whose clause is not in `label2id` are ignored. If a span starts
    before this window (sliding-window case), the first token inside the
    window gets the B- tag.
    """
    labels = [label2id["O"]] * len(offsets)
    for idx, (s, e) in enumerate(offsets):
        if _is_special(s, e):
            labels[idx] = IGNORE_INDEX

    for span in sorted(spans, key=lambda sp: (sp["start"], sp["end"])):
        b_key, i_key = f"B-{span['clause']}", f"I-{span['clause']}"
        if b_key not in label2id:
            continue
        first = True
        for idx, (s, e) in enumerate(offsets):
            if _is_special(s, e):
                continue
            if s < span["end"] and e > span["start"]:
                labels[idx] = label2id[b_key] if first else label2id[i_key]
                first = False
    return labels


def _trim(text: str | None, start: int, end: int) -> tuple[int, int]:
    """Strip leading/trailing whitespace from a span (SentencePiece offsets
    for word-initial pieces can include the preceding space)."""
    if text is None:
        return start, end
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    return start, end


def decode_bio(
    pred_ids: Sequence[int],
    offsets: Sequence[Sequence[int]],
    id2label: dict[int, str],
    text: str | None = None,
    token_scores: Sequence[float] | None = None,
    lenient: bool = True,
) -> list[dict]:
    """Turn per-token label predictions into character spans.

    With `lenient=True` an I-X tag that does not follow B-X/I-X starts a new
    span instead of being thrown away. Token classifiers frequently predict a
    span's first token as I- rather than B-, and strict decoding turns those
    into silent misses.

    Returns a list of {"clause", "start", "end", "score"} dicts, where score is
    the mean probability of the predicted label over the span's tokens (or
    None when token_scores is not given).
    """
    spans: list[dict] = []
    cur: dict | None = None

    def close():
        nonlocal cur
        if cur is not None:
            start, end = _trim(text, cur["start"], cur["end"])
            if end > start:
                scores = cur.pop("_scores")
                cur["score"] = round(sum(scores) / len(scores), 4) if scores else None
                cur["start"], cur["end"] = start, end
                spans.append(cur)
        cur = None

    for idx, (pred, (s, e)) in enumerate(zip(pred_ids, offsets)):
        if _is_special(s, e):
            close()
            continue
        label = id2label.get(int(pred), "O")
        score = float(token_scores[idx]) if token_scores is not None else None
        if label == "O":
            close()
            continue
        tag, clause = label.split("-", 1)
        continues = cur is not None and cur["clause"] == clause
        if tag == "I" and continues:
            cur["end"] = max(cur["end"], e)
            if score is not None:
                cur["_scores"].append(score)
        elif tag == "B" or lenient:
            close()
            cur = {"clause": clause, "start": s, "end": e,
                   "_scores": [score] if score is not None else []}
        else:
            close()
    close()
    return spans


def merge_spans(spans: Iterable[dict]) -> list[dict]:
    """Merge overlapping or touching spans of the same clause.

    Needed when predictions from overlapping sliding windows are combined.
    The merged span keeps the highest score of its parts.
    """
    by_clause: dict[str, list[dict]] = defaultdict(list)
    for sp in spans:
        by_clause[sp["clause"]].append(dict(sp))
    merged: list[dict] = []
    for items in by_clause.values():
        items.sort(key=lambda sp: (sp["start"], sp["end"]))
        current = items[0]
        for sp in items[1:]:
            if sp["start"] <= current["end"]:
                current["end"] = max(current["end"], sp["end"])
                scores = [x for x in (current.get("score"), sp.get("score")) if x is not None]
                current["score"] = max(scores) if scores else None
            else:
                merged.append(current)
                current = sp
        merged.append(current)
    merged.sort(key=lambda sp: (sp["start"], sp["clause"]))
    return merged


def _char_set(spans: Iterable[Sequence[int]]) -> set[int]:
    chars: set[int] = set()
    for start, end in spans:
        chars.update(range(start, end))
    return chars


def span_f1(pred: Iterable[Sequence[int]], gold: Iterable[Sequence[int]]) -> dict[str, float]:
    """Character-overlap precision / recall / F1 between two sets of spans.

    Both empty -> perfect score (correctly predicted "no clause here").
    Exactly one empty -> zero.
    """
    pred_chars, gold_chars = _char_set(pred), _char_set(gold)
    if not pred_chars and not gold_chars:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0}
    if not pred_chars or not gold_chars:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    tp = len(pred_chars & gold_chars)
    precision = tp / len(pred_chars)
    recall = tp / len(gold_chars)
    f1 = 2 * precision * recall / (precision + recall) if tp else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def token_level_f1(
    preds: Sequence[int],
    labels: Sequence[int],
    id2label: dict[int, str],
    clause_types: Sequence[str],
) -> dict:
    """Token-level per-clause F1 (B/I collapsed to the clause).

    `macro` averages only over clauses that actually occur in `labels`. The
    original version averaged over all 8 configured clauses, including the
    three that had no data, which scaled every validation score by 5/8:
    the reported val F1 of 0.4318 is 0.691 x 5/8. That -- not small-sample
    noise -- is most of the "test F1 is higher than val F1" puzzle.
    """
    tp: dict[str, int] = defaultdict(int)
    fp: dict[str, int] = defaultdict(int)
    fn: dict[str, int] = defaultdict(int)
    support: dict[str, int] = defaultdict(int)

    def clause_of(i: int) -> str:
        lab = id2label.get(int(i), "O")
        return lab.split("-", 1)[1] if lab != "O" else "O"

    for p, t in zip(preds, labels):
        if t == IGNORE_INDEX:
            continue
        pc, tc = clause_of(p), clause_of(t)
        if tc != "O":
            support[tc] += 1
            if pc == tc:
                tp[tc] += 1
            else:
                fn[tc] += 1
                if pc != "O":
                    fp[pc] += 1
        elif pc != "O":
            fp[pc] += 1

    per_clause = {}
    for clause in clause_types:
        t, f_n, f_p = tp[clause], fn[clause], fp[clause]
        precision = t / (t + f_p) if t + f_p else 0.0
        recall = t / (t + f_n) if t + f_n else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_clause[clause] = round(f1, 4)

    supported = [per_clause[c] for c in clause_types if support[c] > 0]
    return {
        "per_clause": per_clause,
        "support": {c: support[c] for c in clause_types},
        "macro": round(sum(supported) / len(supported), 4) if supported else 0.0,
    }
