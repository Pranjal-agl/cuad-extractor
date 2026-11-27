"""
src/inference.py

One inference path shared by evaluation, quantization, serving and the
validation step inside training, so every number in the project is produced
the same way.

What it does differently from the original per-module `predict()`:
  * Long text is split into overlapping token windows (MAX_LENGTH / STRIDE
    from src.config) instead of being truncated at 512 tokens, and the spans
    from all windows are merged. Clauses past the first window used to be
    invisible.
  * All clause types come out of ONE forward pass. The original server
    re-ran the full model once per requested clause type (8 passes for a
    default request), even though the model tags every clause at once.
  * Every span carries a confidence score (mean softmax probability of its
    tokens), which feeds calibration, monitoring and client-side thresholds.
  * Window size matches training (384). Inference previously used 512.
"""

from __future__ import annotations

import json
import os
import time
from contextlib import contextmanager
from pathlib import Path

import torch

from src.config import MAX_LENGTH, MODELS_DIR, STRIDE
from src.spans import decode_bio, merge_spans

QUANT_MODES = ("none", "int8", "nf4")


def resolve_model_path(model_path: str | Path | None = None) -> Path:
    """Explicit path > $CUAD_MODEL_PATH > models/best_config.json."""
    if model_path:
        return Path(model_path)
    env = os.environ.get("CUAD_MODEL_PATH")
    if env:
        return Path(env)
    best = MODELS_DIR / "best_config.json"
    if not best.exists():
        raise FileNotFoundError(
            "No model found: pass a path, set CUAD_MODEL_PATH, or run "
            "`python -m src.training.train` to produce models/best_config.json"
        )
    path = Path(json.loads(best.read_text())["model_path"])
    if not path.is_absolute():
        path = MODELS_DIR.parent / path
    if not path.exists():
        # best_config.json written on another machine (e.g. Kaggle) -- fall
        # back to the same run name under the local models/ directory.
        local = MODELS_DIR / path.name
        if local.exists():
            return local
    return path


def _target_device(args, kwargs):
    try:
        device, dtype, _, _ = torch._C._nn._parse_to(*args, **kwargs)
    except Exception:
        return None, True
    return device, dtype is not None


@contextmanager
def allow_noop_to_for_bnb():
    """Work around the 4-bit load crash with the pinned transformers==4.44.2.

    Root cause: accelerate >= 0.33 assumes 4-bit bitsandbytes models can be
    moved (true since bitsandbytes 0.43.2), so for a single-GPU device_map its
    dispatch_model() ends with `model.to(device)`. transformers 4.44.2's
    PreTrainedModel.to() still refuses *every* bitsandbytes model, so loading
    crashes even though the weights are already on that device. Newer
    transformers releases relaxed this check.

    While this context is active, `.to()` on a bitsandbytes model is allowed
    only when it would be a no-op (no dtype change, every parameter already on
    the target device). Real moves still raise, exactly as before.
    """
    from transformers import PreTrainedModel

    original = PreTrainedModel.to

    def patched(self, *args, **kwargs):
        if getattr(self, "quantization_method", None) == "bitsandbytes":
            device, changes_dtype = _target_device(args, kwargs)
            if not changes_dtype:
                if device is None:
                    return self
                if device.type == "cuda" and device.index is None and torch.cuda.is_available():
                    device = torch.device("cuda", torch.cuda.current_device())
                if all(p.device == device for p in self.parameters()):
                    return self
        return original(self, *args, **kwargs)

    PreTrainedModel.to = patched
    try:
        yield
    finally:
        PreTrainedModel.to = original


def load_model(model_path: str | Path | None = None, quantize: str = "none", device: str | None = None):
    """Load (model, tokenizer). quantize: none | int8 (CPU dynamic) | nf4 (GPU, bitsandbytes)."""
    from transformers import AutoModelForTokenClassification, AutoTokenizer

    if quantize not in QUANT_MODES:
        raise ValueError(f"quantize must be one of {QUANT_MODES}, got {quantize!r}")
    path = resolve_model_path(model_path)
    tokenizer = AutoTokenizer.from_pretrained(path)
    if not tokenizer.is_fast:
        raise RuntimeError("A fast tokenizer is required (offset mappings). Install `sentencepiece` and `protobuf`.")

    if quantize == "nf4":
        if not torch.cuda.is_available():
            raise RuntimeError("nf4 quantization needs a CUDA GPU; use quantize='int8' on CPU.")
        from transformers import BitsAndBytesConfig

        bnb = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.float16,
        )
        with allow_noop_to_for_bnb():
            model = AutoModelForTokenClassification.from_pretrained(
                path, quantization_config=bnb, device_map={"": torch.cuda.current_device()},
            )
    elif quantize == "int8":
        # Dynamic INT8 quantization of every nn.Linear: CPU-only, no extra
        # dependencies, applied at load time (takes ~1s), so no separate
        # checkpoint has to be stored.
        import warnings

        model = AutoModelForTokenClassification.from_pretrained(path).to("cpu").eval()
        with warnings.catch_warnings():  # torch >= 2.9 marks torch.ao.quantization as deprecated
            warnings.simplefilter("ignore")
            model = torch.ao.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)
    else:
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        model = AutoModelForTokenClassification.from_pretrained(path).to(device)
    model.eval()
    return model, tokenizer


class ClauseExtractor:
    def __init__(self, model, tokenizer, max_length: int = MAX_LENGTH, stride: int = STRIDE,
                 batch_size: int = 16):
        if stride >= max_length - 2:
            raise ValueError("stride must be smaller than max_length")
        self.model = model
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.stride = stride
        self.batch_size = batch_size
        self.id2label = {int(k): v for k, v in model.config.id2label.items()}
        seen: list[str] = []
        for i in sorted(self.id2label):
            lab = self.id2label[i]
            if lab != "O" and lab.split("-", 1)[1] not in seen:
                seen.append(lab.split("-", 1)[1])
        self.clause_types = seen

    @classmethod
    def load(cls, model_path=None, quantize: str = "none", device: str | None = None, **kwargs):
        model, tokenizer = load_model(model_path, quantize=quantize, device=device)
        return cls(model, tokenizer, **kwargs)

    @property
    def device(self) -> torch.device:
        return next(self.model.parameters()).device

    def _windows(self, text: str):
        enc = self.tokenizer(
            text,
            max_length=self.max_length,
            truncation=True,
            stride=self.stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding="longest",
            return_tensors="pt",
        )
        offsets = enc.pop("offset_mapping").tolist()
        enc.pop("overflow_to_sample_mapping", None)
        return enc, offsets

    @torch.inference_mode()
    def extract(self, text: str, clause_types: list[str] | None = None) -> dict:
        """Return {"spans": [{clause, start, end, text, score}], "latency_ms", "n_windows"}."""
        if clause_types is not None:
            unknown = [c for c in clause_types if c not in self.clause_types]
            if unknown:
                raise ValueError(f"Unknown clause types {unknown}; model knows {self.clause_types}")
        t0 = time.perf_counter()
        enc, offsets = self._windows(text)
        n_windows = enc["input_ids"].shape[0]
        device = self.device
        raw: list[dict] = []
        for b in range(0, n_windows, self.batch_size):
            batch = {k: v[b:b + self.batch_size].to(device) for k, v in enc.items()}
            logits = self.model(**batch).logits.float()
            probs = logits.softmax(-1)
            scores, preds = probs.max(-1)
            for i in range(preds.shape[0]):
                raw.extend(decode_bio(
                    preds[i].tolist(), offsets[b + i], self.id2label,
                    text=text, token_scores=scores[i].tolist(),
                ))
        if device.type == "cuda":
            torch.cuda.synchronize()
        latency_ms = (time.perf_counter() - t0) * 1000

        wanted = set(clause_types) if clause_types is not None else None
        spans = []
        for sp in merge_spans(raw):
            if wanted is None or sp["clause"] in wanted:
                sp["text"] = text[sp["start"]:sp["end"]]
                spans.append(sp)
        return {"spans": spans, "latency_ms": latency_ms, "n_windows": n_windows}
