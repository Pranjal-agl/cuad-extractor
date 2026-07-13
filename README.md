# CUAD Clause Extractor

Fine-tuned DeBERTa-v3-base for extractive clause identification from legal contracts. Given a contract paragraph, the model identifies and extracts text spans for 8 clause types: Governing Law, Termination For Convenience, Renewal Term, Non-Compete, Confidentiality, Indemnification, Cap On Liability, and Assignment.

Built to cover what a pure agentic pipeline cannot: model ownership, quantization with a measured accuracy/latency tradeoff, validation-driven hyperparameter search tracked in MLflow, and a p99 latency target verified under load.

---

## Results

*Fill in after running eval. See instructions below.*

| Metric | FP32 | 4-bit Quantized |
|---|---|---|
| Test macro F1 | - | - |
| Val macro F1 (best config) | - | - |
| Overfit gap | - | - |
| p99 latency | - | - |
| Model size | ~180MB | ~45MB |
| Adversarial pass rate | - | - |
| Latency speedup (4-bit vs FP32) | - | - |

### Ablation: which components matter

| Config | Val F1 | Notes |
|---|---|---|
| Best (validation-selected) | - | Full model, best hyperparams |
| Lowest LR (1e-5) | - | Underfit |
| Highest LR (5e-5) | - | Unstable |
| Fewest epochs (2) | - | Underfit |

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

## Stack

| Component | Technology |
|---|---|
| Base model | microsoft/deberta-v3-base |
| Fine-tuning | HuggingFace Transformers + PyTorch |
| Quantization | bitsandbytes 4-bit NF4 |
| Experiment tracking | MLflow |
| Serving | FastAPI + uvicorn |
| Dataset | CUAD (theatticusproject/cuad-qa) |
