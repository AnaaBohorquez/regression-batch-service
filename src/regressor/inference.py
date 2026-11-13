"""Loads the trained model and predicts a CSV batch, warning when the batch looks unlike the training data."""

import json
import logging
from pathlib import Path

import joblib
import pandas as pd

from regressor.data import load_batch
from regressor.training import DEFAULT_ARTIFACTS_DIR, METADATA_FILE, MODEL_FILE

logger = logging.getLogger(__name__)

MAX_OUT_OF_RANGE = 0.05
MAX_MEAN_SHIFT = 0.25


def load_artifacts(artifacts_dir: str | Path = DEFAULT_ARTIFACTS_DIR):
    """Returns (model, metadata) saved by training.train."""
    artifacts_dir = Path(artifacts_dir)
    model_path = artifacts_dir / MODEL_FILE
    metadata_path = artifacts_dir / METADATA_FILE

    if not model_path.exists() or not metadata_path.exists():
        raise FileNotFoundError(
            f"no trained model in {artifacts_dir}/, run `python -m regressor train` first"
        )

    return joblib.load(model_path), json.loads(metadata_path.read_text())


def drift_report(X: pd.DataFrame, profile: dict) -> list[str]:
    """Compares each feature of the batch against its training profile.

    Flags a feature when more than 5% of its values fall outside the training range, or when its
    mean moved more than 0.25 training standard deviations.
    """
    issues = []
    for col in X.columns:
        ref = profile[col]
        out_of_range = ((X[col] < ref["min"]) | (X[col] > ref["max"])).mean()
        mean_shift = abs(X[col].mean() - ref["mean"]) / ref["std"]

        if out_of_range > MAX_OUT_OF_RANGE:
            issues.append(f"{col}: {out_of_range:.0%} of values outside the training range")
        if mean_shift > MAX_MEAN_SHIFT:
            issues.append(f"{col}: mean moved {mean_shift:.2f} std from training")
    return issues


def predict(
    input_path: str | Path,
    output_path: str | Path,
    artifacts_dir: str | Path = DEFAULT_ARTIFACTS_DIR,
) -> pd.DataFrame:
    """Writes a CSV with a single `target_pred` column, one row per input row and in the same order."""
    model, metadata = load_artifacts(artifacts_dir)
    X = load_batch(input_path, metadata["features"])

    for issue in drift_report(X, metadata["feature_profile"]):
        logger.warning("possible data drift - %s", issue)

    predictions = pd.DataFrame({"target_pred": model.predict(X)})

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(output_path, index=False)

    return predictions
