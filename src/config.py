"""
src/config.py

Single source of truth for everything the training, evaluation and serving
code must agree on. Before this module existed, prepare_data.py, train.py,
evaluate.py, serve.py and monitor.py each kept their own CLAUSE_TYPES list,
and they had drifted apart: prepare_data asked for three names that are not
CUAD categories ("Confidentiality Of Agreements", "Indemnification",
"Assignment"), so those clause types silently produced zero examples.

This module has no heavy imports (no torch / transformers), so tests and
light-weight tools can import it freely.
"""

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"
EXPERIMENTS_DIR = ROOT / "experiments"

# Override with CUAD_BASE_MODEL to try another encoder (e.g. microsoft/deberta-v3-large).
MODEL_NAME = os.environ.get("CUAD_BASE_MODEL", "microsoft/deberta-v3-base")
EXPERIMENT_NAME = "cuad-extraction"
SEED = 42

# Exact CUAD v1 category names (CUAD uses Title Case, e.g. "Cap On Liability").
# Every name here must exist in the source dataset -- prepare_data.py now
# fails loudly if any of them yields zero examples.
CLAUSE_TYPES = [
    "Governing Law",
    "Termination For Convenience",
    "Renewal Term",
    "Non-Compete",
    "Cap On Liability",
    "Anti-Assignment",
    "Audit Rights",
    "Exclusivity",
]

LABEL2ID: dict[str, int] = {"O": 0}
for _i, _clause in enumerate(CLAUSE_TYPES):
    LABEL2ID[f"B-{_clause}"] = 2 * _i + 1
    LABEL2ID[f"I-{_clause}"] = 2 * _i + 2
ID2LABEL: dict[int, str] = {v: k for k, v in LABEL2ID.items()}
NUM_LABELS = len(LABEL2ID)

# Token window used for training AND inference. Keeping these identical
# matters: the original evaluate.py/serve.py ran inference at 512 tokens on a
# model trained at 384. Text longer than one window is handled with
# overlapping sliding windows rather than being truncated.
MAX_LENGTH = 384
STRIDE = 128  # tokens of overlap between consecutive windows

P99_LATENCY_TARGET_MS = 150

# MLflow: use one explicit store so `mlflow ui` and the scripts agree.
MLFLOW_TRACKING_URI = os.environ.get(
    "MLFLOW_TRACKING_URI", f"sqlite:///{EXPERIMENTS_DIR / 'mlflow.db'}"
)


def setup_mlflow():
    """Point MLflow at the project store and select the experiment."""
    import mlflow

    EXPERIMENTS_DIR.mkdir(exist_ok=True)
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)
    return mlflow
