import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import pytest
from sklearn.tree import DecisionTreeClassifier

from will_utils.evaluation import feature_importance_frame
from will_utils.plotting import (
    plot_confusion_matrix,
    plot_correlation_heatmap,
    plot_decision_tree,
    plot_feature_importance,
    plot_training_history,
    plot_value_counts,
    styled_figure,
)


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")


def test_styled_figure_applies_labels():
    _, ax = styled_figure(title="t", xlabel="x", ylabel="y")
    assert (ax.get_title(), ax.get_xlabel(), ax.get_ylabel()) == ("t", "x", "y")


def test_plot_confusion_matrix_labels():
    ax = plot_confusion_matrix([0, 1, 1], [0, 1, 0], labels=["no", "yes"])
    assert ax.get_title() == "Confusion Matrix"
    assert [t.get_text() for t in ax.get_xticklabels()] == ["no", "yes"]


def test_plot_feature_importance_top_n():
    frame = feature_importance_frame(["a", "b", "c"], [0.1, 0.7, 0.2])
    ax = plot_feature_importance(frame, top_n=2, title="Top")
    assert ax.get_title() == "Top"
    assert [t.get_text() for t in ax.get_yticklabels()] == ["b", "c"]


def test_plot_training_history_accepts_dict():
    history = {"accuracy": [0.1, 0.5], "val_accuracy": [0.2, 0.4], "loss": [1.0, 0.5]}
    _, axes = plot_training_history(history)
    assert [ax.get_title() for ax in axes] == ["Model Accuracy", "Model Loss"]
    assert len(axes[0].lines) == 2
    assert len(axes[1].lines) == 1


def test_plot_decision_tree_renders():
    model = DecisionTreeClassifier(max_depth=1).fit([[0], [1]], [0, 1])
    ax = plot_decision_tree(model, ["f"], ["a", "b"], title="Tree")
    assert ax.get_title() == "Tree"


def test_plot_value_counts_orders_by_frequency():
    ax = plot_value_counts(pd.Series(list("aabbbc"), name="letter"))
    assert [t.get_text() for t in ax.get_xticklabels()] == ["b", "a", "c"]
    assert ax.get_ylabel() == "Count"


def test_plot_correlation_heatmap_ignores_non_numeric():
    df = pd.DataFrame({"a": [1, 2, 3], "b": [3, 2, 1], "c": list("xyz")})
    ax = plot_correlation_heatmap(df)
    assert [t.get_text() for t in ax.get_xticklabels()] == ["a", "b"]
