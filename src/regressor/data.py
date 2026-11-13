"""CSV loading and input checks, shared by training and inference."""

from pathlib import Path

import pandas as pd


class DataValidationError(ValueError):
    """The input file can't be used as is (missing columns, nulls, text values...)."""


def read_csv(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"input file not found: {path}")

    df = pd.read_csv(path)
    if df.empty:
        raise DataValidationError(f"{path} has no rows")
    return df


def check_columns(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Returns only `columns`, after checking they exist, are numeric and have no nulls.

    Extra columns are ignored on purpose: the model only needs its own features.
    """
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise DataValidationError(f"missing columns: {missing}")

    subset = df[columns]

    not_numeric = [c for c in columns if not pd.api.types.is_numeric_dtype(subset[c])]
    if not_numeric:
        raise DataValidationError(f"non numeric columns: {not_numeric}")

    nulls = subset.isna().sum()
    nulls = nulls[nulls > 0]
    if not nulls.empty:
        raise DataValidationError(f"null values found: {nulls.to_dict()}")

    return subset


def load_training_data(path: str | Path, features: list[str], target: str) -> tuple[pd.DataFrame, pd.Series]:
    df = read_csv(path)
    checked = check_columns(df, features + [target])
    return checked[features], checked[target]


def load_batch(path: str | Path, features: list[str]) -> pd.DataFrame:
    return check_columns(read_csv(path), features)
