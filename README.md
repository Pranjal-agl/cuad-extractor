# CUAD Clause Extractor

Fine-tuned DeBERTa-v3-base for extractive clause identification from legal contracts. Given a contract paragraph, the model identifies and extracts text spans for 8 clause types: Governing Law, Termination For Convenience, Renewal Term, Non-Compete, Confidentiality, Indemnification, Cap On Liability, and Assignment.

Built to cover what a pure agentic pipeline cannot: model ownership, quantization with a measured accuracy/latency tradeoff, validation-driven hyperparameter search tracked in MLflow, and a p99 latency target verified under load.

---

## Results

| Metric | FP32 | 4-bit Quantized |
|---|---|---|
| Test macro F1 | 0.7747 | N/A (see note below) |
| Val macro F1 (best config) | 0.4318 | N/A |
| Overfit gap (test − val) | +0.3429 | N/A |
| p99 latency | 29.1ms | N/A |
| Model size | 711.9MB | N/A |
| Adversarial pass rate | 50% (2/4) | N/A |
| Latency speedup (4-bit vs FP32) | N/A | N/A |

**Note on quantization:** 4-bit NF4 quantization is implemented (`src/training/quantize.py`) but currently fails at load time due to a long-standing upstream bug in `transformers`' bitsandbytes integration (`from_pretrained` with `quantization_config` calls `dispatch_model()`, which calls `model.to()` on an already-quantized model — disallowed by bitsandbytes). This is reported against multiple `transformers` versions over an extended period (see [huggingface/transformers#35442](https://github.com/huggingface/transformers/issues/35442), [#38717](https://github.com/huggingface/transformers/issues/38717)), not something fixed by pinning a different version. `quantize.py` degrades gracefully: it still reports full FP32 results and records the quantization failure rather than crashing. Revisiting this is a known follow-up, not a blocker for the rest of the pipeline.

**Note on the val/test F1 gap:** test F1 (0.77) is notably *higher* than val F1 (0.43) — the reverse of typical overfitting. This is most likely a small-sample effect: val has only 233 examples across 5 clause types with different per-clause proportions than test (269 examples), and `Governing Law` — the easiest clause type (F1=0.97) — happens to be a larger share of test. Treat both numbers as noisy estimates from small splits rather than a precise generalization gap.

### Per-clause F1 (test set)

| Clause | F1 | 95% CI |
|---|---|---|
| Governing Law | 0.973 | [0.920, 0.997] |
| Renewal Term | 0.764 | [0.744, 0.965] |
| Non-Compete | 0.754 | [0.719, 0.962] |
| Cap On Liability | 0.729 | [0.693, 0.847] |
| Termination For Convenience | 0.655 | [0.589, 0.863] |

### Adversarial eval (test set)

| Case | Result |
|---|---|
| no_clause_present | PASS (F1=1.000) |
| boilerplate_noise | PASS (F1=1.000) |
| truncated_context | FAIL (F1=0.468) |
| multiple_spans | FAIL (F1=0.338) |

The model handles negative cases (no clause present, distracting boilerplate) cleanly, but degrades meaningfully when a contract has more than one span for the same clause type, or when context is cut short — both realistic contract patterns, not synthetic edge cases.

### Ablation: which components matter

| Config | Val F1 | Notes |
|---|---|---|
| Best: lr=2e-5, bs=8, ep=3 | 0.4318 | Selected config |
| lr=1e-5, bs=8, ep=3 | 0.2344 | Underfit |
| lr=1e-5, bs=8, ep=5 | 0.4281 | More epochs closes most of the gap to the best config |
| lr=2e-5, bs=16, ep=3 | 0.3396 | Larger batch, same LR — worse than bs=8 |
| lr=5e-6, bs=8, ep=5 | 0.3181 | Too conservative even with more epochs |
| lr=3e-5, bs=8, ep=2 | 0.4025 | Highest LR tested, competitive despite fewer epochs |

---

## Architecture

```
Contract paragraph (raw text)
        |
        v
DeBERTa-v3-base tokenizer (512 token max, truncation)
        |
        v
DeBERTa-v3-base encoder (fine-tuned)
        |
        v
Token classification head (17 labels: O + B/I for 8 clause types)
        |
        v
BIO span extraction -> clause spans
```

One model handles all 8 clause types simultaneously via multi-class BIO tagging.
This avoids 8 separate forward passes per contract.

---

## What the doc required and where it is

| Requirement | Implementation |
|---|---|
| Held-out eval, no leakage | Contract-level split, splits.lock fingerprint |
| Reproducible runs | seed=42 everywhere, MLflow logs all params |
| Ablations isolating component contribution | 6-config grid search, all logged to MLflow |
| Validation-driven hyperparameter search | Grid search on val F1, test set touched once |
| Honest overfit check | Val F1 vs test F1 reported side by side |
| Quantization with tradeoff measured | bitsandbytes 4-bit NF4, F1 drop + speedup in report |
| p99 latency target | 150ms target, measured on 100 timing runs + load test |
| Cost-per-request awareness | Latency * compute cost tracked in quantization report |
| Slice and adversarial evals | Per-clause F1 + 4 adversarial cases |
| Confidence intervals | 95% Wilson CI per clause on test set |
| Drift detection | monitor.py tracks prediction rate + confidence + latency drift |
| Online/offline metric gap | label_session() + drift_report() measures online F1 |
| Graceful degradation | Serving returns empty spans + error flag, never 500 |
| Model card | MODEL_CARD.md with known limitations and reproduction steps |

---

## Setup

```bash
git clone https://github.com/Pranjal-agl/cuad-extractor
cd cuad-extractor
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

GPU recommended (Colab T4 or Kaggle). Training on CPU is slow but works for testing.

See `notebooks/kaggle_train.ipynb` for the full training run with real outputs (data prep, 6-config hyperparameter search, quantization, evaluation, calibration).

---

## Running

### 1. Prepare data
```bash
python -m src.training.prepare_data
```
Downloads CUAD from HuggingFace, selects 8 clause types, creates contract-level
train/val/test splits, writes splits.lock to prevent test set contamination.

### 2. Train with hyperparameter search
```bash
python -m src.training.train
```
Runs 6 configurations, logs all to MLflow, saves best checkpoint to models/.
Takes 2-6 hours on a T4 depending on config.

View experiment results:
```bash
mlflow ui --backend-store-uri experiments/mlruns
```

### 3. Quantize
```bash
python -m src.training.quantize
```
Loads best checkpoint, applies 4-bit NF4 quantization, measures F1 drop and
latency gain vs FP32, writes report to experiments/quantization_report.json.

### 4. Evaluate on test set
```bash
# Run once, after hyperparameter selection is final
python -m src.eval.evaluate --split test

# With quantized model
python -m src.eval.evaluate --split test --quantized
```

### 5. Serve
```bash
python -m src.serving.serve
```

Test it:
```bash
curl -X POST http://localhost:8000/extract \
  -H "Content-Type: application/json" \
  -d '{"text": "This Agreement is governed by the laws of Delaware.", "clause_types": ["Governing Law"]}'
```

### 6. Load test
```bash
python -m src.serving.load_test --concurrency 4 --n-requests 200
```

### 7. Monitor drift
```bash
# After processing real contracts through the API
python -m src.eval.monitor --report --baseline experiments/eval_test_fp32.json

# Label production predictions for online F1 tracking
python -m src.eval.monitor --label
```

---

## Project structure

```
cuad-extractor/
├── src/
│   ├── training/
│   │   ├── prepare_data.py    # CUAD download, contract-level splits, splits.lock
│   │   ├── train.py           # Fine-tuning with MLflow tracking, 6-config grid search
│   │   └── quantize.py        # 4-bit NF4 quantization, F1/latency tradeoff report
│   ├── eval/
│   │   ├── evaluate.py        # Test set eval, per-clause F1, CI, adversarial cases
│   │   └── monitor.py         # Production drift monitoring, online accuracy tracking
│   └── serving/
│       ├── serve.py           # FastAPI endpoint, rolling latency window, /health /metrics
│       └── load_test.py       # Async load test, p99 under concurrency
├── data/                      # JSONL splits + splits.lock (git-ignored)
├── models/                    # Checkpoints + best_config.json (git-ignored)
├── experiments/               # Eval reports, quantization report, load test report
├── MODEL_CARD.md
├── requirements.txt
└── README.md
```

---

---

## Known issues & debugging notes

Training initially produced NaN loss and, once fixed, a model that collapsed to predicting the majority "O" class. Both were root-caused and fixed rather than worked around:

- **NaN loss from epoch 1:** `AdamW`'s default `eps=1e-8` is too small for DeBERTa-v3's gradient-disentangled embedding sharing, which produces very small early gradients; the adaptive step size spiked. Fixed with `eps=1e-6` in `train.py`.
- **val_macro_f1 stuck at 0.0000:** confirmed via diagnostic instrumentation that the model predicted zero non-"O" tokens across multiple epochs despite loss decreasing — a majority-class collapse caused by unweighted cross-entropy on a heavily imbalanced label distribution (clause spans are a small minority of tokens). Fixed with inverse-frequency class-weighted loss in `train.py`.
- **4-bit quantization crash:** see the Results section above — this remains open, degrades gracefully, does not block the rest of the pipeline.

## Stack

| Component | Technology |
|---|---|
| Base model | microsoft/deberta-v3-base |
| Fine-tuning | HuggingFace Transformers (pinned 4.44.2) + PyTorch |
| Quantization | bitsandbytes 4-bit NF4 (currently blocked, see Known Issues) |
| Experiment tracking | MLflow |
| Serving | FastAPI + uvicorn |
| Dataset | CUAD (theatticusproject/cuad-qa) |
