import json

import pandas as pd
import pytest

from conftest import make_frame
from regressor.__main__ import main
from regressor.config import load_config
from regressor.inference import drift_report, load_artifacts, predict
from regressor.training import train


@pytest.fixture
def artifacts(tmp_path, train_csv, config_path):
    folder = tmp_path / "artifacts"
    train(train_csv, load_config(config_path), folder)
    return folder


def test_train_writes_model_and_metadata(artifacts):
    assert (artifacts / "model.joblib").exists()
    metadata = json.loads((artifacts / "metadata.json").read_text())
    assert metadata["n_rows"] == 1000
    assert metadata["features"] == ["feature_a", "feature_b", "feature_c"]
    assert set(metadata["feature_profile"]["feature_a"]) == {"min", "max", "mean", "std"}


def test_model_learns_something(artifacts):
    metadata = json.loads((artifacts / "metadata.json").read_text())
    target_std = make_frame(1000, seed=0)["target"].std()
    assert metadata["in_sample_rmse"] < target_std / 2


def test_predict_output_format(tmp_path, artifacts, batch_csv):
    output = tmp_path / "out" / "target_pred.csv"
    predict(batch_csv, output, artifacts)

    result = pd.read_csv(output)
    assert list(result.columns) == ["target_pred"]
    assert len(result) == 50
    assert result["target_pred"].notna().all()


def test_predict_without_trained_model(tmp_path, batch_csv):
    with pytest.raises(FileNotFoundError, match="run `python -m regressor train`"):
        predict(batch_csv, tmp_path / "out.csv", tmp_path / "empty_artifacts")


def test_no_drift_on_similar_batch(artifacts):
    _, metadata = load_artifacts(artifacts)
    batch = make_frame(200, seed=5, with_target=False)[metadata["features"]]
    assert drift_report(batch, metadata["feature_profile"]) == []


def test_drift_on_shifted_batch(artifacts):
    _, metadata = load_artifacts(artifacts)
    batch = make_frame(200, seed=5, with_target=False)[metadata["features"]]
    batch["feature_a"] = batch["feature_a"] + 5
    issues = drift_report(batch, metadata["feature_profile"])
    assert any("feature_a" in issue for issue in issues)
    assert not any("feature_b" in issue for issue in issues)


def test_cli_end_to_end(tmp_path, train_csv, batch_csv, config_path):
    folder = str(tmp_path / "artifacts")
    output = tmp_path / "preds.csv"
    assert main(["train", "--data", str(train_csv), "--config", str(config_path), "--artifacts", folder]) == 0
    assert main(["predict", "--input", str(batch_csv), "--output", str(output), "--artifacts", folder]) == 0
    assert len(pd.read_csv(output)) == 50


def test_cli_reports_bad_input(tmp_path, config_path):
    code = main(["train", "--data", str(tmp_path / "missing.csv"), "--config", str(config_path)])
    assert code == 1
