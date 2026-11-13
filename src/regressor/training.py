"""Trains the model described in configs/model.json and saves it next to its metadata."""

import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd
import sklearn
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import root_mean_squared_error

from regressor.config import ModelConfig
from regressor.data import load_training_data

DEFAULT_ARTIFACTS_DIR = Path("artifacts")
MODEL_FILE = "model.joblib"
METADATA_FILE = "metadata.json"


def feature_profile(X: pd.DataFrame) -> dict:
    """Range and spread of each feature in the training data, used by inference to flag odd batches."""
    return {
        col: {
            "min": float(X[col].min()),
            "max": float(X[col].max()),
            "mean": float(X[col].mean()),
            "std": float(X[col].std()),
        }
        for col in X.columns
    }


def train(data_path: str | Path, cfg: ModelConfig, artifacts_dir: str | Path = DEFAULT_ARTIFACTS_DIR) -> dict:
    """Fits the model on every row of `data_path` and writes model.joblib + metadata.json."""
    X, y = load_training_data(data_path, cfg.features, cfg.target)

    model = HistGradientBoostingRegressor(**cfg.params)
    model.fit(X, y)

    artifacts_dir = Path(artifacts_dir)
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, artifacts_dir / MODEL_FILE)

    metadata = {
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "training_data": str(data_path),
        "n_rows": len(X),
        "features": cfg.features,
        "target": cfg.target,
        "params": cfg.params,
        "sklearn_version": sklearn.__version__,
        "in_sample_rmse": round(float(root_mean_squared_error(y, model.predict(X))), 4),
        "notebook_validation": cfg.validation,
        "feature_profile": feature_profile(X),
    }
    (artifacts_dir / METADATA_FILE).write_text(json.dumps(metadata, indent=2))

    return metadata
