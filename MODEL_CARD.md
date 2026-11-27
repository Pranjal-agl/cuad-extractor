# Model Card: CUAD Clause Extractor

## Model description

DeBERTa-v3-base fine-tuned on the Contract Understanding Atticus Dataset (CUAD v1) for extractive clause identification. Given contract text of any length, the model tags character spans for 8 clause types and gives each span a confidence score.

- **Base model:** `microsoft/deberta-v3-base` (184M parameters)
- **Task:** token classification with BIO tags (17 labels), sliding-window inference (384 tokens, stride 128)
- **Clause types:** Governing Law, Termination For Convenience, Renewal Term, Non-Compete, Cap On Liability, Anti-Assignment, Audit Rights, Exclusivity
- **Training data:** CUAD v1 (510 US commercial contracts, 41 categories, expert-annotated), loaded from `dvgodoy/CUAD_v1_Contract_Understanding_clause_classification` (labels) and `dvgodoy/CUAD_v1_Contract_Understanding_PDF` (text)
- **Intended use:** first-pass highlighting of candidate clauses for a human reviewer
- **Out of scope:** legal advice, or unsupervised decisions about a contract's legal effect

## Training

- **Examples:** one positive window per CUAD annotation (the clause plus ~512 characters on each side), and negative windows (0.5 per positive) drawn from contracts that contain no clause of the queried type. Every target-clause annotation overlapping a window is labelled.
- **Split:** contract-level 70/15/15 (seed 42). All windows from one contract stay in one split. `data/splits.lock` fingerprints the test contracts, test examples and clause list, and every script verifies it before running.
- **Objective:** class-weighted cross-entropy (inverse or square-root inverse frequency; chosen by the grid), AdamW (eps 1e-6), linear warm-up/decay, gradient clipping at 1.0, non-finite batches skipped.
- **Selection:** grid search (7 configs) on span-level macro F1 on the validation split, the same metric reported on test. Test is evaluated once, after selection.
- **Tracking:** all runs are in MLflow with hyperparameters, per-epoch metrics, git commit and checkpoint path.

## Evaluation

**Metrics**

- **Span F1:** character-overlap F1 between predicted and gold spans of the queried clause, averaged per example. An example with no gold span and no prediction scores 1; otherwise an empty side scores 0.
- **Per-clause results:** F1 with a 95% bootstrap CI; precision, recall and hit rate on positives; false-positive rate on negatives.
- **Macro F1:** averaged over clause types present in the split.
- **Calibration:** reliability table and ECE of span confidence scores.
- **Adversarial cases (6):** short sentence, no clause, two spans, signature-block boilerplate, wrong-clause query, and a clause beyond the first window.

**Results**

Pending a re-run of the current pipeline. The last reported run (27 Jul 2026) used the first version of the code. It had data for only 5 of 8 clause types, no negative examples, and a val metric that wasn't comparable to the test metric. Its test span F1 over those 5 types was 0.775, with per-clause F1 from 0.655 (Termination For Convenience) to 0.973 (Governing Law), and p99 latency of 29 ms on a T4. See the README for details. These numbers are not expected to carry over: negatives make the task harder, while multi-span labels and sliding windows should help.

## Quantization

- **INT8 dynamic (CPU):** `nn.Linear` weights quantized at load time; compared with FP32 on the same CPU.
- **NF4 4-bit (GPU, bitsandbytes):** compared with FP32 on the same GPU.

`experiments/quantization_report.json` records the F1 drop, p99 speedup and size for each, after a run.

**Serving p99 target:** 150 ms per request for a single 384-token window. Longer documents cost roughly one window per ~256 new tokens.

## Known limitations

- **Coverage:** 8 of CUAD's 41 categories. Adding more means extending `CLAUSE_TYPES` in `src/config.py` and retraining.
- **Domain:** US commercial contracts, mostly from SEC filings. Expect degradation on non-US law, consumer terms, or heavily formatted or scanned documents (the text comes from PDF extraction).
- **Span boundaries:** CUAD gold spans are annotator highlights and vary in extent. Character F1 penalises a correct clause found with different boundaries.
- **Class weighting:** inverse-frequency weighting over-predicts clause tokens, so precision trails recall. Use the per-span `score` (or `min_score` in the API) to trade recall for precision. `src/eval/calibration.py` shows how well scores track precision.
- **Small evaluation slices:** some clause types have under 40 test examples, so read the bootstrap CIs, not just the point estimates.

## How to reproduce

```bash
python -m src.training.prepare_data
python -m src.training.train
python -m src.training.quantize
python -m src.eval.evaluate --split test
python -m src.serving.serve
python -m src.serving.load_test --concurrency 4 --n-requests 200
```
