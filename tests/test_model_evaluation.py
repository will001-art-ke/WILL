"""Unit tests for helpers of the notebooks whose code is otherwise untested.

Both notebooks are dominated by heavy dependencies (Keras, PyTorch, Earth
Engine); the helpers below are the parts that can be exercised with stubs.
"""

import numpy as np
from sklearn.metrics import r2_score

from tests.notebook_loader import load_definitions

BOOTSTRAP = load_definitions(
    "last_supervised_learning_ml.ipynb",
    ["get_confidence_interval"],
    cell_index=28,
    inject={"np": np, "r2_score": r2_score},
)
NDVI = load_definitions("GIS_analysis.ipynb", ["calculate_ndvi"], cell_index=42)


class NoisyRegressor:
    """Returns the target plus a deterministic offset, to give a stable R²."""

    def __init__(self, offset=0.0):
        self._offset = offset
        self._rng = np.random.default_rng(0)

    def predict(self, X):
        return np.asarray(X).ravel() + self._offset


class TestGetConfidenceInterval:
    def test_a_perfect_model_has_an_interval_pinned_at_one(self):
        X = np.linspace(0, 1, 200).reshape(-1, 1)
        y = X.ravel()
        lower, upper, mean = BOOTSTRAP["get_confidence_interval"](
            NoisyRegressor(), X, y, n_iterations=5
        )
        assert lower == upper == mean == 1.0

    def test_bounds_bracket_the_mean_score(self):
        X = np.linspace(0, 1, 200).reshape(-1, 1)
        y = X.ravel()
        lower, upper, mean = BOOTSTRAP["get_confidence_interval"](
            NoisyRegressor(offset=0.05), X, y, n_iterations=20
        )
        assert lower <= mean <= upper

    def test_a_wider_confidence_level_widens_the_interval(self):
        X = np.linspace(0, 1, 200).reshape(-1, 1)
        y = X.ravel()
        narrow = BOOTSTRAP["get_confidence_interval"](
            NoisyRegressor(offset=0.05), X, y, confidence=0.5, n_iterations=40
        )
        wide = BOOTSTRAP["get_confidence_interval"](
            NoisyRegressor(offset=0.05), X, y, confidence=0.99, n_iterations=40
        )
        assert (wide[1] - wide[0]) >= (narrow[1] - narrow[0])

    def test_results_are_reproducible_for_a_deterministic_model(self):
        X = np.linspace(0, 1, 100).reshape(-1, 1)
        y = X.ravel()
        first = BOOTSTRAP["get_confidence_interval"](
            NoisyRegressor(offset=0.1), X, y, n_iterations=10
        )
        second = BOOTSTRAP["get_confidence_interval"](
            NoisyRegressor(offset=0.1), X, y, n_iterations=10
        )
        assert np.allclose(first, second, atol=0.2)


class FakeEarthEngineImage:
    """Records the Earth Engine calls made by ``calculate_ndvi``."""

    def __init__(self):
        self.bands = None
        self.name = None

    def normalizedDifference(self, bands):
        self.bands = bands
        return self

    def rename(self, name):
        self.name = name
        return self


class TestCalculateNdvi:
    def test_uses_the_landsat_8_nir_and_red_bands(self):
        image = FakeEarthEngineImage()
        NDVI["calculate_ndvi"](image)
        assert image.bands == ["SR_B5", "SR_B4"]

    def test_renames_the_resulting_band(self):
        image = FakeEarthEngineImage()
        assert NDVI["calculate_ndvi"](image) is image
        assert image.name == "NDVI"
