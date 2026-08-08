"""Unit tests for the data preparation/evaluation helpers of the ML notebooks."""

import numpy as np
import pandas as pd
import pytest

from tests.notebook_loader import load_definitions

SEQUENCES = load_definitions(
    "stock_exchange_data_by_omoke.ipynb", ["create_sequences"], cell_index=16
)
WEATHER = load_definitions(
    "simulated_weather_kenyan_ML_model.ipynb", ["assign_condition"], cell_index=1
)
TRAFFIC = load_definitions(
    "web_analysis_ml,nn.ipynb",
    ["categorize_traffic"],
    cell_index=52,
    inject={"q1_threshold": 10.0, "q3_threshold": 100.0},
)
CLUSTERING = load_definitions(
    "clustering_algrithms_on_kenyan_medical_facilities.ipynb",
    ["calculate_score_no_noise"],
    cell_index=24,
)
REGRESSION = load_definitions(
    "mpesa_transaction_by_omoke.ipynb", ["evaluate_regression"], cell_index=28
)
BENCHMARK = load_definitions(
    "ASSOCIATION_RULE_MINING_PERFORMANCE_COMPARISON_BY_Omoke.ipynb",
    ["monitor_algorithm"],
    cell_index=15,
)


def weather_row(**overrides):
    row = {
        "precip_mm": 0.0,
        "humidity_pct": 50.0,
        "wind_speed_kmh": 5.0,
        "pressure_hpa": 1015.0,
    }
    row.update(overrides)
    return pd.Series(row)


class TestCreateSequences:
    def test_shapes_follow_the_window_size(self):
        data = np.arange(10, dtype=float).reshape(-1, 1)
        X, y = SEQUENCES["create_sequences"](data, 3)
        assert X.shape == (7, 3)
        assert y.shape == (7,)

    def test_windows_and_targets_are_consecutive(self):
        data = np.arange(6, dtype=float).reshape(-1, 1)
        X, y = SEQUENCES["create_sequences"](data, 2)
        assert X.tolist() == [[0, 1], [1, 2], [2, 3], [3, 4]]
        assert y.tolist() == [2, 3, 4, 5]

    def test_window_equal_to_length_yields_no_samples(self):
        data = np.arange(4, dtype=float).reshape(-1, 1)
        X, y = SEQUENCES["create_sequences"](data, 4)
        assert X.size == 0
        assert y.size == 0


class TestAssignCondition:
    @pytest.mark.parametrize(
        "row, expected",
        [
            (weather_row(precip_mm=0.5), "Rainy"),
            (weather_row(humidity_pct=80, pressure_hpa=1005), "Rainy"),
            (weather_row(wind_speed_kmh=25), "Windy"),
            (weather_row(humidity_pct=70), "Cloudy"),
            (weather_row(), "Sunny"),
        ],
    )
    def test_condition_labels(self, row, expected):
        assert WEATHER["assign_condition"](row) == expected

    def test_rain_takes_priority_over_wind(self):
        row = weather_row(precip_mm=1.0, wind_speed_kmh=40)
        assert WEATHER["assign_condition"](row) == "Rainy"

    def test_thresholds_are_exclusive(self):
        # Exactly at the precipitation/wind/humidity thresholds nothing triggers.
        row = weather_row(precip_mm=0.2, wind_speed_kmh=18, humidity_pct=60)
        assert WEATHER["assign_condition"](row) == "Sunny"

    def test_high_humidity_needs_low_pressure_to_be_rainy(self):
        assert WEATHER["assign_condition"](
            weather_row(humidity_pct=80, pressure_hpa=1015)
        ) == "Cloudy"


class TestCategorizeTraffic:
    @pytest.mark.parametrize(
        "views, expected",
        [
            (0, "Bottom Quartile (Low)"),
            (10, "Bottom Quartile (Low)"),
            (11, "Middle Quartiles"),
            (99, "Middle Quartiles"),
            (100, "Top Quartile (High)"),
            (5000, "Top Quartile (High)"),
        ],
    )
    def test_quartile_boundaries(self, views, expected):
        assert TRAFFIC["categorize_traffic"](views) == expected

    def test_applies_elementwise_to_a_series(self):
        series = pd.Series([1, 50, 500])
        assert series.apply(TRAFFIC["categorize_traffic"]).tolist() == [
            "Bottom Quartile (Low)",
            "Middle Quartiles",
            "Top Quartile (High)",
        ]


class TestCalculateScoreNoNoise:
    def test_returns_none_when_a_single_cluster_remains(self):
        X = np.array([[0.0, 0.0], [1.0, 1.0], [5.0, 5.0]])
        labels = np.array([0, 0, -1])
        assert CLUSTERING["calculate_score_no_noise"](X, labels) is None

    def test_returns_none_when_everything_is_noise(self):
        X = np.array([[0.0, 0.0], [1.0, 1.0]])
        assert CLUSTERING["calculate_score_no_noise"](X, np.array([-1, -1])) is None

    def test_well_separated_clusters_score_close_to_one(self):
        X = np.array([[0.0, 0.0], [0.1, 0.0], [10.0, 10.0], [10.1, 10.0]])
        score = CLUSTERING["calculate_score_no_noise"](X, np.array([0, 0, 1, 1]))
        assert score == pytest.approx(1.0, abs=0.05)

    def test_noise_points_are_excluded_from_the_score(self):
        X = np.array([[0.0, 0.0], [0.1, 0.0], [10.0, 10.0], [10.1, 10.0], [5.0, 5.0]])
        labels = np.array([0, 0, 1, 1, -1])
        with_noise = CLUSTERING["calculate_score_no_noise"](X, labels)
        without_noise = CLUSTERING["calculate_score_no_noise"](
            X[:4], np.array([0, 0, 1, 1])
        )
        assert with_noise == pytest.approx(without_noise)


class DummyRegressor:
    def __init__(self, predictions):
        self._predictions = np.asarray(predictions, dtype=float)

    def predict(self, X):
        return self._predictions


class TestEvaluateRegression:
    def test_perfect_predictions(self):
        y_true = np.array([1.0, 2.0, 3.0])
        metrics = REGRESSION["evaluate_regression"](
            DummyRegressor(y_true), None, y_true, "perfect"
        )
        assert metrics["Model"] == "perfect"
        assert metrics["RMSE"] == pytest.approx(0.0)
        assert metrics["MAE"] == pytest.approx(0.0)
        assert metrics["R2"] == pytest.approx(1.0)

    def test_metric_values_for_a_known_error_pattern(self):
        y_true = np.array([0.0, 0.0, 0.0, 0.0])
        metrics = REGRESSION["evaluate_regression"](
            DummyRegressor([1.0, -1.0, 1.0, -1.0]), None, y_true, "constant error"
        )
        assert metrics["RMSE"] == pytest.approx(1.0)
        assert metrics["MAE"] == pytest.approx(1.0)

    def test_predicting_the_mean_gives_zero_r2(self):
        y_true = np.array([1.0, 2.0, 3.0, 4.0])
        metrics = REGRESSION["evaluate_regression"](
            DummyRegressor(np.full(4, y_true.mean())), None, y_true, "mean"
        )
        assert metrics["R2"] == pytest.approx(0.0)


class TestMonitorAlgorithm:
    def test_reports_positive_runtime_and_memory(self):
        stats = BENCHMARK["monitor_algorithm"](lambda n: [0] * n, 100_000)
        assert stats["Execution Time (s)"] >= 0.0
        assert stats["Peak Memory (MB)"] > 0.0

    def test_forwards_positional_and_keyword_arguments(self):
        seen = {}

        def record(*args, **kwargs):
            seen["args"] = args
            seen["kwargs"] = kwargs

        BENCHMARK["monitor_algorithm"](record, 1, 2, flag=True)
        assert seen == {"args": (1, 2), "kwargs": {"flag": True}}

    def test_slow_calls_report_a_larger_runtime(self):
        import time

        fast = BENCHMARK["monitor_algorithm"](lambda: None)
        slow = BENCHMARK["monitor_algorithm"](lambda: time.sleep(0.05))
        assert slow["Execution Time (s)"] > fast["Execution Time (s)"]
