"""Reads the model config chosen in notebooks/02_modeling.ipynb."""

import json
from dataclasses import dataclass
from pathlib import Path

DEFAULT_CONFIG_PATH = Path("configs/model.json")
SUPPORTED_MODEL = "HistGradientBoostingRegressor"
REQUIRED_KEYS = {"model", "features", "target", "params"}


@dataclass(frozen=True)
class ModelConfig:
    features: list[str]
    target: str
    params: dict
    validation: dict


def load_config(path: str | Path = DEFAULT_CONFIG_PATH) -> ModelConfig:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"config file not found: {path}")

    raw = json.loads(path.read_text())

    missing = REQUIRED_KEYS - raw.keys()
    if missing:
        raise ValueError(f"{path} is missing keys: {sorted(missing)}")

    if raw["model"] != SUPPORTED_MODEL:
        raise ValueError(f"unsupported model '{raw['model']}', expected {SUPPORTED_MODEL}")

    if not raw["features"]:
        raise ValueError(f"{path} has an empty feature list")

    return ModelConfig(
        features=list(raw["features"]),
        target=raw["target"],
        params=dict(raw["params"]),
        validation=dict(raw.get("validation", {})),
    )
