import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from src.config import LABEL2ID  # noqa: E402
from src.eval.evaluate import adversarial_cases, adversarial_eval, run_eval  # noqa: E402
from src.inference import ClauseExtractor, allow_noop_to_for_bnb, load_model  # noqa: E402

LONG = "the supplier shall deliver the products to the delivery address. " * 80


def test_spans_are_valid_slices(tiny_model_dir):
    ex = ClauseExtractor.load(tiny_model_dir, device="cpu")
    out = ex.extract(LONG)
    assert out["n_windows"] > 1
    for sp in out["spans"]:
        assert 0 <= sp["start"] < sp["end"] <= len(LONG)
        assert sp["text"] == LONG[sp["start"]:sp["end"]]
        assert 0.0 <= sp["score"] <= 1.0


def test_sliding_windows_cover_whole_text_and_merge(forced_extractor):
    ex = forced_extractor("I-Governing Law", max_length=64, stride=16)
    out = ex.extract(LONG)
    assert out["n_windows"] > 3
    gov = [s for s in out["spans"] if s["clause"] == "Governing Law"]
    # every window predicts the whole window; overlaps must merge into one span
    assert len(gov) == 1
    assert (gov[0]["start"], gov[0]["end"]) == (0, len(LONG.rstrip()))


def test_clause_filter_and_unknown_clause(forced_extractor):
    ex = forced_extractor("I-Governing Law")
    assert ex.extract("the laws of delaware", clause_types=["Non-Compete"])["spans"] == []
    with pytest.raises(ValueError):
        ex.extract("x", clause_types=["Confidentiality Of Agreements"])


def test_clause_types_read_from_model_config(tiny_model_dir):
    ex = ClauseExtractor.load(tiny_model_dir, device="cpu")
    assert ex.clause_types[0] == "Governing Law" and len(ex.clause_types) == (len(LABEL2ID) - 1) // 2


def test_int8_dynamic_quantization_runs(tiny_model_dir):
    model, tok = load_model(tiny_model_dir, quantize="int8")
    assert any("quantized" in type(m).__module__ for m in model.modules())
    ClauseExtractor(model, tok).extract(LONG[:500])


def test_bnb_patch_allows_only_noop_moves(tiny_model_dir):
    model, _ = load_model(tiny_model_dir, device="cpu")
    model.quantization_method = "bitsandbytes"
    with pytest.raises(ValueError):  # transformers 4.44.2 refuses every move
        model.to("cpu")
    with allow_noop_to_for_bnb():
        assert model.to("cpu") is model
        with pytest.raises(ValueError):
            model.to(torch.float16)  # dtype change is a real move -> still refused
    with pytest.raises(ValueError):  # patch removed afterwards
        model.to("cpu")


def test_run_eval_perfect_and_negative_scores(forced_extractor):
    ex = forced_extractor("I-Governing Law")
    text = "the laws of delaware apply."
    examples = [
        {"clause": "Governing Law", "context": text, "kind": "positive",
         "spans": [{"clause": "Governing Law", "start": 0, "end": len(text)}]},
        {"clause": "Non-Compete", "context": text, "kind": "negative", "spans": []},
    ]
    res = run_eval(ex, examples, verbose=False)
    assert res["per_clause"]["Governing Law"]["f1"] == 1.0
    assert res["per_clause"]["Non-Compete"]["f1"] == 1.0
    assert res["per_clause"]["Non-Compete"]["false_positive_rate_negative"] == 0.0
    assert res["n_clause_types_scored"] == 2
    # calibration looks at every predicted span: the Governing Law span is right in the
    # first window and unannotated (so wrong) in the second
    assert res["calibration"]["n"] == 2
    assert res["calibration"]["bins"][0]["accuracy"] == 0.5


def test_adversarial_cases_well_formed_and_long_case_needs_windows(forced_extractor):
    for c in adversarial_cases():
        for s in c["spans"]:
            assert c["context"][s["start"]:s["end"]] in c["answers"]
    res = adversarial_eval(forced_extractor("O"), verbose=False)
    long_case = [r for r in res["adversarial_cases"] if r["case"] == "clause_beyond_first_window"][0]
    assert long_case["n_windows"] > 1
