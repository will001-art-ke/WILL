"""Encoding and scaling helpers shared by the notebooks."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Literal

import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler

ScalerName = Literal["standard", "minmax"]

_SCALERS = {"standard": StandardScaler, "minmax": MinMaxScaler}


def encode_labels(values: Iterable) -> tuple[pd.Series, LabelEncoder]:
    """Label-encode a single column, returning the codes and the encoder."""
    encoder = LabelEncoder()
    encoded = encoder.fit_transform(list(values))
    return pd.Series(encoded), encoder


def encode_categorical_columns(
    df: pd.DataFrame, columns: Iterable[str] | None = None
) -> tuple[pd.DataFrame, dict[str, LabelEncoder]]:
    """Label-encode the given (or all object/category) columns in place-safe copy."""
    out = df.copy()
    if columns is None:
        columns = out.select_dtypes(include=["object", "category"]).columns
    encoders: dict[str, LabelEncoder] = {}
    for column in columns:
        encoder = LabelEncoder()
        out[column] = encoder.fit_transform(out[column].astype(str))
        encoders[column] = encoder
    return out, encoders


def scale_features(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame | None = None,
    method: ScalerName = "standard",
):
    """Fit a scaler on the training split and apply it to both splits.

    Returns ``(X_train_scaled, X_test_scaled, scaler)``; ``X_test_scaled`` is
    ``None`` when no test split is given.
    """
    if method not in _SCALERS:
        raise ValueError(f"Unknown scaler: {method!r}. Use one of {list(_SCALERS)}.")
    scaler = _SCALERS[method]()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = None if X_test is None else scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, scaler
