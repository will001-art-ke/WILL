"""Matplotlib/seaborn plotting helpers shared by the notebooks."""

from __future__ import annotations

from collections.abc import Sequence

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.tree import plot_tree


def styled_figure(
    figsize: tuple[float, float] = (10, 6),
    title: str = "",
    xlabel: str = "",
    ylabel: str = "",
    grid_axis: str | None = "y",
):
    """Create a figure with the title/label/grid styling used across notebooks."""
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if grid_axis:
        ax.grid(axis=grid_axis, linestyle="--", alpha=0.7)
    return fig, ax


def show(fig=None) -> None:
    """Tighten the layout and display the figure."""
    (fig or plt.gcf()).tight_layout()
    plt.show()


def plot_confusion_matrix(
    y_true,
    y_pred,
    labels: Sequence[str] | None = None,
    title: str = "Confusion Matrix",
    cmap: str = "Blues",
    figsize: tuple[float, float] = (10, 8),
    ax=None,
):
    """Heatmap of the confusion matrix with annotated counts."""
    matrix = confusion_matrix(y_true, y_pred)
    if ax is None:
        _, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap=cmap,
        xticklabels=labels if labels is not None else "auto",
        yticklabels=labels if labels is not None else "auto",
        ax=ax,
    )
    ax.set_title(title)
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    return ax


def plot_feature_importance(
    importance_df: pd.DataFrame,
    top_n: int = 15,
    title: str = "Feature Importances",
    palette: str = "magma",
    figsize: tuple[float, float] = (10, 8),
    ax=None,
):
    """Horizontal bar chart of a ``Feature``/``Importance`` frame."""
    data = importance_df.head(top_n)
    if ax is None:
        _, ax = plt.subplots(figsize=figsize)
    sns.barplot(
        data=data,
        x="Importance",
        y="Feature",
        hue="Feature",
        palette=palette,
        legend=False,
        ax=ax,
    )
    ax.set_title(title)
    ax.set_xlabel("Importance Score")
    ax.set_ylabel("Features")
    ax.grid(axis="x", linestyle="--", alpha=0.7)
    return ax


def plot_training_history(
    history,
    metrics: Sequence[str] = ("accuracy", "loss"),
    figsize: tuple[float, float] = (12, 5),
):
    """Train/validation curves for a Keras ``History`` (or its ``.history`` dict)."""
    values = getattr(history, "history", history)
    fig, axes = plt.subplots(1, len(metrics), figsize=figsize, squeeze=False)
    for ax, metric in zip(axes[0], metrics):
        if metric in values:
            ax.plot(values[metric], label="Train")
        val_metric = f"val_{metric}"
        if val_metric in values:
            ax.plot(values[val_metric], label="Validation")
        ax.set_title(f"Model {metric.capitalize()}")
        ax.set_xlabel("Epoch")
        ax.set_ylabel(metric.capitalize())
        ax.legend(loc="upper left")
    fig.tight_layout()
    return fig, axes[0]


def plot_decision_tree(
    model,
    feature_names: Sequence[str],
    class_names: Sequence[str] | None = None,
    figsize: tuple[float, float] = (20, 10),
    title: str = "Decision Tree Structure",
    fontsize: int = 12,
):
    """Render a fitted decision tree with the notebooks' usual styling."""
    _, ax = plt.subplots(figsize=figsize)
    plot_tree(
        model,
        feature_names=list(feature_names),
        class_names=None if class_names is None else list(class_names),
        filled=True,
        rounded=True,
        fontsize=fontsize,
        ax=ax,
    )
    ax.set_title(title)
    return ax


def plot_value_counts(
    series: pd.Series,
    title: str = "",
    top_n: int | None = None,
    figsize: tuple[float, float] = (10, 6),
    rotation: int = 45,
    ax=None,
):
    """Bar chart of category frequencies."""
    counts = series.value_counts()
    if top_n is not None:
        counts = counts.head(top_n)
    if ax is None:
        _, ax = plt.subplots(figsize=figsize)
    sns.barplot(x=counts.index, y=counts.values, hue=counts.index, legend=False, ax=ax)
    ax.set_title(title or f"Distribution of {series.name}")
    ax.set_xlabel(series.name or "")
    ax.set_ylabel("Count")
    ax.tick_params(axis="x", rotation=rotation)
    ax.grid(axis="y", linestyle="--", alpha=0.7)
    return ax


def plot_correlation_heatmap(
    df: pd.DataFrame,
    title: str = "Correlation Matrix",
    figsize: tuple[float, float] = (12, 10),
    cmap: str = "coolwarm",
    ax=None,
):
    """Annotated correlation heatmap of the numeric columns."""
    corr = df.select_dtypes(include="number").corr()
    if ax is None:
        _, ax = plt.subplots(figsize=figsize)
    sns.heatmap(corr, annot=True, fmt=".2f", cmap=cmap, ax=ax)
    ax.set_title(title)
    return ax
