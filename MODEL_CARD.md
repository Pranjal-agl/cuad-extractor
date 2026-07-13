# Model Card: CUAD Clause Extractor

## Model Description

DeBERTa-v3-base fine-tuned on the Contract Understanding Atticus Dataset (CUAD)
for extractive clause identification. Given a contract paragraph, the model
identifies and extracts text spans corresponding to 8 clause types.

**Base model:** microsoft/deberta-v3-base  
**Task:** Token classification (BIO tagging) for clause span extraction  
**Dataset:** CUAD (theatticusproject/cuad-qa on HuggingFace)  
**Clause types covered:** Governing Law, Termination For Convenience, Renewal Term,
Non-Compete, Confidentiality Of Agreements, Indemnification, Cap On Liability, Assignment

---

## Training

**Hyperparameter selection:** Grid search over 6 configurations (learning rate,
batch size, epochs, warmup ratio). Best configuration selected by macro-averaged
F1 on the validation set. Test set was evaluated exactly once, after selection.

**Split strategy:** Contract-level split (70/15/15). All clauses from one contract
stay in the same split, preventing the model from memorising boilerplate shared
across contracts from the same company.

**Reproducibility:** All runs use seed=42 set across Python, NumPy, and PyTorch.
Every training run is logged to MLflow with full hyperparameters, per-epoch metrics,
and the best checkpoint path.

**Contamination guard:** splits.lock records a SHA-256 fingerprint of the test
contract list. The eval script verifies this fingerprint before running to ensure
the test set was not modified after the splits were created.

---

## Evaluation Results

Results below are on the held-out test set (15% of contracts, never seen during
training or hyperparameter selection).

| Metric | FP32 | 4-bit Quantized |
|---|---|---|
| Macro F1 | - | - |
| p99 latency | - | - |
| Model size | ~180MB | ~45MB |
| Adversarial pass rate | - | - |

*Fill in after running: python -m src.eval.evaluate --split test*

### Per-clause F1 (test set)

| Clause Type | F1 | 95% CI | n |
|---|---|---|---|
| Governing Law | - | - | - |
| Termination For Convenience | - | - | - |
| Renewal Term | - | - | - |
| Non-Compete | - | - | - |
| Confidentiality Of Agreements | - | - | - |
| Indemnification | - | - | - |
| Cap On Liability | - | - | - |
| Assignment | - | - | - |

*Fill in after running eval.*

---

## Quantization

4-bit NF4 quantization via bitsandbytes. NF4 is information-theoretically
optimal for normally distributed weights and typically gives less than 1% F1 drop
with approximately 2-4x latency improvement. The quantization report in
experiments/quantization_report.json records the exact accuracy/latency tradeoff
measured on this model.

**Serving p99 target:** 150ms per paragraph (single clause type)

---

## Known Limitations

**Truncation:** DeBERTa-v3-base accepts up to 512 tokens. Contracts longer than
this are truncated, and clauses near the end of a long contract may be missed.
In production, long contracts should be chunked with overlap before extraction.

**Clause coverage:** Only 8 of 41 CUAD clause types are covered. The remaining
33 types were excluded for scope but the model architecture supports adding them
by extending the label set and retraining.

**Domain specificity:** Trained on US commercial contracts. Performance may degrade
on non-US contracts, consumer agreements, or highly technical contracts (software
licensing, IP agreements) that differ structurally from the training distribution.

**Near-limit theft claims:** The fraud scorer in a related project consistently
over-rates near-limit theft claims. This is noted here as a reminder that any
model has distribution-specific failure modes that must be documented, not hidden.

**Overfit check:** Val F1 and test F1 are reported separately in the eval output.
If the gap exceeds 5pp, it indicates overfitting to the validation-selected
hyperparameters and the search should be rerun with more configurations.

---

## How to Reproduce

```bash
# 1. Prepare data
python -m src.training.prepare_data

# 2. Train (hyperparameter search, logged to MLflow)
python -m src.training.train

# 3. Quantize and measure tradeoff
python -m src.training.quantize

# 4. Evaluate on test set (run once, after training is complete)
python -m src.eval.evaluate --split test

# 5. Serve
python -m src.serving.serve

# 6. Load test
python -m src.serving.load_test --concurrency 4 --n-requests 200
```

All runs are tracked in MLflow. Start the UI with:
```bash
mlflow ui --backend-store-uri experiments/mlruns
```

---

## Ethical Considerations

This model extracts clause spans from contracts but does not provide legal advice.
Extracted spans should be reviewed by a qualified legal professional before
being used for any legal or business decision. False negatives (missed clauses)
are more dangerous than false positives in this domain.
