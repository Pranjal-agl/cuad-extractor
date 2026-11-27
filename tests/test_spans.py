from src.config import CLAUSE_TYPES, ID2LABEL, LABEL2ID
from src.spans import IGNORE_INDEX, decode_bio, label_tokens, merge_spans, span_f1, token_level_f1

GOV_B, GOV_I = LABEL2ID["B-Governing Law"], LABEL2ID["I-Governing Law"]
CAP_B, CAP_I = LABEL2ID["B-Cap On Liability"], LABEL2ID["I-Cap On Liability"]

#                      0123456789012345678901234567
TEXT = "Laws of Delaware apply here."
# tokens:  [CLS] Laws  of   Delaware apply here  .  [SEP]
OFFSETS = [(0, 0), (0, 4), (5, 7), (8, 16), (17, 22), (23, 27), (27, 28), (0, 0)]


def test_label_tokens_marks_overlapping_tokens_and_specials():
    labels = label_tokens(OFFSETS, [{"clause": "Governing Law", "start": 0, "end": 16}], LABEL2ID)
    assert labels == [IGNORE_INDEX, GOV_B, GOV_I, GOV_I, 0, 0, 0, IGNORE_INDEX]


def test_label_tokens_partial_token_overlap_is_labelled():
    # span boundary falls inside "Delaware" -- old exact-boundary logic dropped the whole span
    labels = label_tokens(OFFSETS, [{"clause": "Governing Law", "start": 5, "end": 12}], LABEL2ID)
    assert labels[2:4] == [GOV_B, GOV_I]


def test_label_tokens_multiple_clauses_and_unknown_clause_ignored():
    spans = [
        {"clause": "Governing Law", "start": 0, "end": 4},
        {"clause": "Cap On Liability", "start": 17, "end": 27},
        {"clause": "Not A Clause", "start": 8, "end": 16},
    ]
    labels = label_tokens(OFFSETS, spans, LABEL2ID)
    assert labels[1] == GOV_B
    assert labels[3] == 0
    assert labels[4:6] == [CAP_B, CAP_I]


def test_decode_bio_roundtrip():
    labels = label_tokens(OFFSETS, [{"clause": "Governing Law", "start": 0, "end": 16}], LABEL2ID)
    preds = [0 if x == IGNORE_INDEX else x for x in labels]
    spans = decode_bio(preds, OFFSETS, ID2LABEL, text=TEXT)
    assert [(s["clause"], s["start"], s["end"]) for s in spans] == [("Governing Law", 0, 16)]
    assert TEXT[spans[0]["start"]:spans[0]["end"]] == "Laws of Delaware"


def test_decode_bio_lenient_orphan_i_tag():
    preds = [0, GOV_I, GOV_I, 0, 0, 0, 0, 0]
    assert len(decode_bio(preds, OFFSETS, ID2LABEL, lenient=True)) == 1
    assert decode_bio(preds, OFFSETS, ID2LABEL, lenient=False) == []


def test_decode_bio_clause_switch_splits_span_and_scores():
    preds = [0, GOV_B, GOV_I, CAP_I, CAP_I, 0, 0, 0]
    scores = [1, 0.9, 0.7, 0.6, 0.8, 1, 1, 1]
    spans = decode_bio(preds, OFFSETS, ID2LABEL, text=TEXT, token_scores=scores)
    assert [(s["clause"], s["start"], s["end"]) for s in spans] == [
        ("Governing Law", 0, 7), ("Cap On Liability", 8, 22)]
    assert spans[0]["score"] == 0.8 and spans[1]["score"] == 0.7


def test_decode_bio_trims_whitespace_from_offsets():
    text = "a  bc"
    spans = decode_bio([GOV_B, GOV_I], [(1, 3), (3, 5)], ID2LABEL, text=text)
    assert (spans[0]["start"], spans[0]["end"]) == (3, 5)


def test_merge_spans_overlapping_same_clause_only():
    spans = [
        {"clause": "Governing Law", "start": 0, "end": 10, "score": 0.5},
        {"clause": "Governing Law", "start": 8, "end": 20, "score": 0.9},
        {"clause": "Cap On Liability", "start": 5, "end": 9, "score": 0.4},
        {"clause": "Governing Law", "start": 30, "end": 40, "score": 0.3},
    ]
    merged = merge_spans(spans)
    gov = [(s["start"], s["end"], s["score"]) for s in merged if s["clause"] == "Governing Law"]
    assert gov == [(0, 20, 0.9), (30, 40, 0.3)]
    assert len(merged) == 3


def test_span_f1_cases():
    assert span_f1([], [])["f1"] == 1.0
    assert span_f1([(0, 5)], [])["f1"] == 0.0
    assert span_f1([], [(0, 5)])["f1"] == 0.0
    r = span_f1([(0, 10)], [(5, 15)])
    assert r["precision"] == 0.5 and r["recall"] == 0.5 and r["f1"] == 0.5


def test_token_level_macro_ignores_clauses_without_support():
    # perfect predictions on Governing Law; the 7 other clauses have no data
    labels = [IGNORE_INDEX, GOV_B, GOV_I, 0, IGNORE_INDEX]
    res = token_level_f1(labels, labels, ID2LABEL, CLAUSE_TYPES)
    assert res["per_clause"]["Governing Law"] == 1.0
    assert res["macro"] == 1.0  # old implementation: 1.0 * 1/8 = 0.125
