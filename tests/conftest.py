"""Shared fixtures: a small synthetic dataset and config, so tests never depend on the challenge CSVs."""

import json

import numpy as np
import pandas as pd
import pytest

FEATURES = ["feature_a", "feature_b", "feature_c"]


def make_frame(n_rows: int, seed: int, with_target: bool = True) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(rng.uniform(0, 10, size=(n_rows, len(FEATURES))), columns=FEATURES)
    df["unused_column"] = rng.uniform(size=n_rows)
    if with_target:
        df["target"] = 2 * df["feature_a"] + (df["feature_b"] - 5) ** 2 + rng.normal(0, 0.5, n_rows)
    return df


@pytest.fixture
def train_csv(tmp_path):
    path = tmp_path / "train.csv"
    make_frame(1000, seed=0).to_csv(path, index=False)
    return path


@pytest.fixture
def batch_csv(tmp_path):
    path = tmp_path / "batch.csv"
    make_frame(50, seed=1, with_target=False).to_csv(path, index=False)
    return path


@pytest.fixture
def config_path(tmp_path):
    path = tmp_path / "model.json"
    path.write_text(json.dumps({
        "model": "HistGradientBoostingRegressor",
        "features": FEATURES,
        "target": "target",
        "params": {"max_iter": 50, "learning_rate": 0.1, "random_state": 0},
        "validation": {"cv_rmse": 1.0},
    }))
    return path
