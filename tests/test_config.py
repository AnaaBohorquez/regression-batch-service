import json

import pytest

from regressor.config import load_config


def test_loads_valid_config(config_path):
    cfg = load_config(config_path)
    assert cfg.features == ["feature_a", "feature_b", "feature_c"]
    assert cfg.target == "target"
    assert cfg.params["max_iter"] == 50


def test_real_project_config_is_valid():
    cfg = load_config("configs/model.json")
    assert len(cfg.features) == 5


def test_missing_keys_are_reported(tmp_path):
    path = tmp_path / "model.json"
    path.write_text(json.dumps({"model": "HistGradientBoostingRegressor", "features": ["x"]}))
    with pytest.raises(ValueError, match="missing keys"):
        load_config(path)


def test_unsupported_model_is_rejected(tmp_path, config_path):
    raw = json.loads(config_path.read_text())
    raw["model"] = "LinearRegression"
    path = tmp_path / "other.json"
    path.write_text(json.dumps(raw))
    with pytest.raises(ValueError, match="unsupported model"):
        load_config(path)


def test_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_config(tmp_path / "nope.json")
