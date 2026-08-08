"""Dataset loading and exploration helpers shared by the notebooks."""

from __future__ import annotations

import os
from collections.abc import Sequence
from typing import Any

import pandas as pd
from sklearn.model_selection import train_test_split


def load_csv_from_dir(path: str, **read_csv_kwargs: Any) -> pd.DataFrame:
    """Read the first CSV file found in ``path``.

    Raises ``FileNotFoundError`` when the directory holds no CSV file.
    """
    csv_files = sorted(f for f in os.listdir(path) if f.endswith(".csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in the directory: {path}")
    if len(csv_files) > 1:
        print(f"Multiple CSV files found. Using the first one: {csv_files[0]}")
    return pd.read_csv(os.path.join(path, csv_files[0]), **read_csv_kwargs)


def missing_value_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Per-column count and percentage of missing values, worst first."""
    counts = df.isnull().sum()
    return pd.DataFrame(
        {"missing_count": counts, "missing_pct": counts / len(df) * 100}
    ).sort_values("missing_count", ascending=False)


def describe_dataframe(df: pd.DataFrame, head: int = 5) -> None:
    """Print the standard first-look summary: head, info, stats and gaps."""
    print("--- Head ---")
    print(df.head(head).to_string())
    print("\n--- DataFrame Info ---")
    df.info()
    print("\n--- Descriptive Statistics ---")
    print(df.describe(include="all").to_string())
    print("\n--- Missing Values ---")
    print(missing_value_summary(df).to_string())


def split_and_report(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    random_state: int = 42,
    stratify: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """``train_test_split`` plus the shape printout every notebook repeats."""
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y if stratify else None,
    )
    print(f"Training set shape: {X_train.shape}")
    print(f"Testing set shape: {X_test.shape}")
    return X_train, X_test, y_train, y_test


def rename_columns(df: pd.DataFrame, mapping: dict[str, str]) -> pd.DataFrame:
    """Rename columns and report the ones the frame did not contain."""
    unknown: Sequence[str] = [c for c in mapping if c not in df.columns]
    if unknown:
        print(f"Columns not present and therefore not renamed: {list(unknown)}")
    return df.rename(columns=mapping)
