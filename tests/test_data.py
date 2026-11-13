import pandas as pd
import pytest

from regressor.data import DataValidationError, check_columns, load_batch, read_csv

COLUMNS = ["feature_a", "feature_b"]


def test_keeps_only_requested_columns():
    df = pd.DataFrame({"feature_a": [1.0], "feature_b": [2.0], "extra": [3.0]})
    assert list(check_columns(df, COLUMNS).columns) == COLUMNS


def test_missing_column():
    df = pd.DataFrame({"feature_a": [1.0]})
    with pytest.raises(DataValidationError, match="missing columns"):
        check_columns(df, COLUMNS)


def test_null_values():
    df = pd.DataFrame({"feature_a": [1.0, None], "feature_b": [2.0, 3.0]})
    with pytest.raises(DataValidationError, match="null values"):
        check_columns(df, COLUMNS)


def test_text_values():
    df = pd.DataFrame({"feature_a": ["one"], "feature_b": [2.0]})
    with pytest.raises(DataValidationError, match="non numeric"):
        check_columns(df, COLUMNS)


def test_empty_file(tmp_path):
    path = tmp_path / "empty.csv"
    path.write_text("feature_a,feature_b\n")
    with pytest.raises(DataValidationError, match="no rows"):
        read_csv(path)


def test_batch_without_target(batch_csv):
    X = load_batch(batch_csv, ["feature_a", "feature_b", "feature_c"])
    assert X.shape == (50, 3)
