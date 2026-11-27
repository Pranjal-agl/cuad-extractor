import pytest

from src.training import prepare_data as pd

GOV = "This Agreement is governed by the laws of Delaware."
CAP = "Liability shall not exceed the fees paid."
FILLER = "Lorem ipsum dolor sit amet consectetur. " * 60


def _contract(i):
    return FILLER + GOV + " Middle text here. " + CAP + " " + FILLER + f" contract {i}"


@pytest.fixture()
def corpus():
    texts = {f"c{i}.pdf": _contract(i) for i in range(20)}
    rows = []
    for name, text in texts.items():
        g = text.index(GOV)
        c = text.index(CAP)
        # Offsets padded with whitespace on purpose: the old code stripped the
        # text but not the offsets.
        rows.append({"file_name": name, "label": "Governing Law", "start_at": g - 1, "end_at": g + len(GOV) + 1,
                     "clause": GOV})
        rows.append({"file_name": name, "label": "Cap On Liability", "start_at": c, "end_at": c + len(CAP),
                     "clause": CAP})
        rows.append({"file_name": name, "label": "Parties", "start_at": 0, "end_at": 5})
    rows.append({"file_name": "missing.pdf", "label": "Governing Law", "start_at": 0, "end_at": 5})
    return rows, texts


def test_collect_annotations_trims_offsets_and_counts(corpus):
    rows, texts = corpus
    anns, stats = pd.collect_annotations(rows, texts)
    assert len(anns) == 20
    assert stats["missing_text"] == 1 and stats["other_label"] == 20
    a = [x for x in anns["c0.pdf"] if x["clause"] == "Governing Law"][0]
    assert texts["c0.pdf"][a["start"]:a["end"]] == GOV
    assert stats["alignment_ok"] == stats["alignment_checked"]


def test_build_examples_offsets_exact_and_all_spans_labelled(corpus):
    rows, texts = corpus
    anns, _ = pd.collect_annotations(rows, texts)
    by_contract = pd.build_examples(anns, texts, neg_ratio=0.0)
    ex = [e for e in by_contract["c0.pdf"] if e["clause"] == "Governing Law"][0]
    assert ex["answers"] == [GOV]
    # the Cap On Liability sentence sits inside the same window and must be labelled too
    clauses = {s["clause"] for s in ex["spans"]}
    assert clauses == {"Governing Law", "Cap On Liability"}
    for s in ex["spans"]:
        assert ex["context"][s["start"]:s["end"]] in (GOV, CAP)


def test_negatives_never_contain_their_clause(corpus):
    rows, texts = corpus
    anns, _ = pd.collect_annotations(rows, texts)
    by_contract = pd.build_examples(anns, texts, neg_ratio=1.0)
    negs = [e for exs in by_contract.values() for e in exs if e["kind"] == "negative"]
    assert negs
    for e in negs:
        assert not e["has_answer"] and e["answers"] == []
        assert e["clause"] not in {"Governing Law", "Cap On Liability"}
        assert all(s["clause"] != e["clause"] for s in e["spans"])


def test_build_examples_is_deterministic(corpus):
    rows, texts = corpus
    anns, _ = pd.collect_annotations(rows, texts)
    assert pd.build_examples(anns, texts) == pd.build_examples(anns, texts)


def test_split_is_contract_level(corpus):
    rows, texts = corpus
    anns, _ = pd.collect_annotations(rows, texts)
    train, val, test = pd.split_contracts(pd.build_examples(anns, texts))
    sets = [{e["contract"] for e in s} for s in (train, val, test)]
    assert not (sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2])
    assert sum(len(s) for s in sets) == 20
    pd.contamination_check(train, val, test)


def test_coverage_check_fails_on_missing_clause_type(corpus):
    rows, texts = corpus
    anns, _ = pd.collect_annotations(rows, texts)
    examples = [e for exs in pd.build_examples(anns, texts).values() for e in exs]
    with pytest.raises(SystemExit):
        pd.coverage_check(examples)  # only 2 of the 8 configured types present
    pd.coverage_check(examples, clause_types=["Governing Law", "Cap On Liability"])


def test_example_spans_supports_legacy_schema():
    legacy = {"clause": "Governing Law", "context": "xx " + GOV, "answers": [GOV], "start_in_context": 3}
    assert pd.example_spans(legacy) == [{"clause": "Governing Law", "start": 3, "end": 3 + len(GOV)}]
    # wrong legacy offset (the old strip() bug) falls back to search
    legacy["start_in_context"] = 1
    assert pd.example_spans(legacy)[0]["start"] == 3


def test_lock_roundtrip(tmp_path, monkeypatch, corpus):
    rows, texts = corpus
    monkeypatch.setattr(pd, "DATA_DIR", tmp_path)
    anns, _ = pd.collect_annotations(rows, texts)
    train, val, test = pd.split_contracts(pd.build_examples(anns, texts))
    pd.write_splits(train, val, test)
    assert pd.verify_lock() is True
    lines = (tmp_path / "test.jsonl").read_text().splitlines()
    (tmp_path / "test.jsonl").write_text("\n".join(lines[:-1]) + "\n")
    with pytest.raises(SystemExit):
        pd.verify_lock()
