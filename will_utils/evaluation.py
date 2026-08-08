"""Model evaluation helpers shared by the notebooks."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)


def evaluate_classifier(
    y_true,
    y_pred,
    target_names: Sequence[str] | None = None,
    title: str = "Model",
    verbose: bool = True,
) -> dict[str, Any]:
    """Accuracy, classification report and confusion matrix in one call."""
    accuracy = accuracy_score(y_true, y_pred)
    report = classification_report(y_true, y_pred, target_names=target_names)
    matrix = confusion_matrix(y_true, y_pred)
    if verbose:
        print(f"{title} Accuracy: {accuracy:.4f}")
        print("\nClassification Report:")
        print(report)
        print("Confusion Matrix:")
        print(
            pd.DataFrame(matrix, index=target_names, columns=target_names).to_string()
        )
    return {"accuracy": accuracy, "report": report, "confusion_matrix": matrix}


def evaluate_regressor(
    y_true, y_pred, title: str = "Model", verbose: bool = True
) -> dict[str, float]:
    """MSE, RMSE, MAE and R^2 for a regression model."""
    mse = mean_squared_error(y_true, y_pred)
    metrics = {
        "mse": mse,
        "rmse": float(np.sqrt(mse)),
        "mae": mean_absolute_error(y_true, y_pred),
        "r2": r2_score(y_true, y_pred),
    }
    if verbose:
        print(f"--- {title} ---")
        for name, value in metrics.items():
            print(f"{name.upper()}: {value:.4f}")
    return metrics


def feature_importance_frame(
    feature_names: Sequence[str], importances, top_n: int | None = None
) -> pd.DataFrame:
    """Feature/Importance frame sorted descending, optionally truncated."""
    frame = pd.DataFrame(
        {"Feature": list(feature_names), "Importance": np.asarray(importances)}
    ).sort_values(by="Importance", ascending=False, ignore_index=True)
    return frame if top_n is None else frame.head(top_n)


def coefficient_importance_frame(
    feature_names: Sequence[str], coefficients, top_n: int | None = None
) -> pd.DataFrame:
    """Importance frame from linear model coefficients (mean absolute magnitude)."""
    coefficients = np.asarray(coefficients)
    if coefficients.ndim > 1:
        coefficients = np.mean(np.abs(coefficients), axis=0)
    else:
        coefficients = np.abs(coefficients)
    return feature_importance_frame(feature_names, coefficients, top_n=top_n)


def compare_models(results: dict[str, dict[str, float]]) -> pd.DataFrame:
    """Turn ``{model_name: {metric: value}}`` into a comparison table."""
    return pd.DataFrame(results).T.rename_axis("Model").reset_index()
