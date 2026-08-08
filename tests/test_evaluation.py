import numpy as np
import pandas as pd

from will_utils.evaluation import (
    coefficient_importance_frame,
    compare_models,
    evaluate_classifier,
    evaluate_regressor,
    feature_importance_frame,
)


def test_evaluate_classifier_metrics(capsys):
    result = evaluate_classifier([0, 1, 1, 0], [0, 1, 0, 0], target_names=["no", "yes"])
    assert result["accuracy"] == 0.75
    assert result["confusion_matrix"].shape == (2, 2)
    assert "Classification Report" in capsys.readouterr().out


def test_evaluate_classifier_quiet(capsys):
    evaluate_classifier([0, 1], [0, 1], verbose=False)
    assert capsys.readouterr().out == ""


def test_evaluate_regressor_metrics():
    metrics = evaluate_regressor([1.0, 2.0, 3.0], [1.0, 2.0, 4.0], verbose=False)
    assert metrics["mse"] == 1 / 3
    assert metrics["rmse"] == np.sqrt(1 / 3)
    assert metrics["mae"] == 1 / 3


def test_feature_importance_frame_sorted_and_truncated():
    frame = feature_importance_frame(["a", "b", "c"], [0.1, 0.7, 0.2], top_n=2)
    assert list(frame["Feature"]) == ["b", "c"]
    assert list(frame.columns) == ["Feature", "Importance"]


def test_coefficient_importance_frame_handles_multiclass():
    coefs = np.array([[-1.0, 0.5], [1.0, 0.1]])
    frame = coefficient_importance_frame(["x", "y"], coefs)
    assert list(frame["Feature"]) == ["x", "y"]
    assert frame.loc[0, "Importance"] == 1.0
    assert frame.loc[1, "Importance"] == 0.3


def test_coefficient_importance_frame_handles_1d():
    frame = coefficient_importance_frame(["x", "y"], np.array([-2.0, 1.0]))
    assert list(frame["Feature"]) == ["x", "y"]
    assert list(frame["Importance"]) == [2.0, 1.0]


def test_compare_models_table():
    table = compare_models({"rf": {"r2": 0.9}, "lr": {"r2": 0.5}})
    assert isinstance(table, pd.DataFrame)
    assert list(table["Model"]) == ["rf", "lr"]
