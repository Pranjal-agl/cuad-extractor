"""Shared fixtures. The model fixtures build a tiny randomly-initialised BERT
token classifier with the project's label set, so inference / serving code
can be exercised end-to-end on CPU without downloading DeBERTa."""

import string

import pytest

from src.config import ID2LABEL, LABEL2ID

WORDS = (
    "this agreement shall be governed by the laws of state california delaware new york "
    "either party may terminate for convenience upon thirty days written notice to other "
    "supplier deliver products delivery address specified in each purchase order title "
    "and risk loss pass customer inspect within ten business notify any non conforming "
    "payment made invoice receipt terms conditions set forth herein parties agree"
).split()


def _write_vocab(path):
    tokens = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
    tokens += sorted(set(WORDS))
    tokens += list(string.ascii_lowercase + string.digits + string.punctuation)
    tokens += [f"##{c}" for c in string.ascii_lowercase + string.digits]
    path.write_text("\n".join(dict.fromkeys(tokens)) + "\n")


@pytest.fixture(scope="session")
def tiny_tokenizer(tmp_path_factory):
    transformers = pytest.importorskip("transformers")
    d = tmp_path_factory.mktemp("tok")
    _write_vocab(d / "vocab.txt")
    return transformers.BertTokenizerFast(vocab_file=str(d / "vocab.txt"), do_lower_case=True)


def _build_model(tokenizer, force_label: str | None = None):
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    torch.manual_seed(0)
    config = transformers.BertConfig(
        vocab_size=len(tokenizer), hidden_size=32, num_hidden_layers=1, num_attention_heads=2,
        intermediate_size=64, max_position_embeddings=512,
        num_labels=len(LABEL2ID), id2label=ID2LABEL, label2id=LABEL2ID,
    )
    model = transformers.BertForTokenClassification(config).eval()
    if force_label is not None:
        # Make the classifier predict one label everywhere, deterministically.
        with torch.no_grad():
            model.classifier.weight.zero_()
            model.classifier.bias.fill_(-10.0)
            model.classifier.bias[LABEL2ID[force_label]] = 10.0
    return model


@pytest.fixture(scope="session")
def tiny_model_dir(tmp_path_factory, tiny_tokenizer):
    d = tmp_path_factory.mktemp("tiny_model")
    _build_model(tiny_tokenizer).save_pretrained(d)
    tiny_tokenizer.save_pretrained(d)
    return d


@pytest.fixture()
def forced_extractor(tiny_tokenizer):
    """Extractor whose model tags every token I-Governing Law."""
    from src.inference import ClauseExtractor

    def make(label="I-Governing Law", **kwargs):
        return ClauseExtractor(_build_model(tiny_tokenizer, label), tiny_tokenizer, **kwargs)

    return make
