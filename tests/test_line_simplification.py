"""Unit tests for the Visvalingam-Whyatt helpers of the weather dashboard notebook."""

import numpy as np

from tests.notebook_loader import load_definitions

NOTEBOOK = "simulated_weather_kenyan_ML_model.ipynb"

MASK_VARIANT = load_definitions(NOTEBOOK, ["visvalingam_whyatt"], cell_index=53)
ITERATIVE_VARIANT = load_definitions(
    NOTEBOOK, ["visvalingam_whyatt_simplify"], cell_index=57
)

STRAIGHT_LINE = np.array([[float(i), float(i)] for i in range(6)])


class TestVisvalingamWhyatt:
    def test_short_polylines_are_returned_unchanged(self):
        points = np.array([[0.0, 0.0], [1.0, 1.0]])
        assert np.array_equal(MASK_VARIANT["visvalingam_whyatt"](points, 0.1), points)

    def test_collinear_interior_points_are_dropped(self):
        simplified = MASK_VARIANT["visvalingam_whyatt"](STRAIGHT_LINE, 0.001)
        assert simplified.tolist() == [[0.0, 0.0], [5.0, 5.0]]

    def test_endpoints_are_always_kept(self):
        points = np.array([[0.0, 0.0], [1.0, 5.0], [2.0, 0.0], [3.0, 5.0]])
        simplified = MASK_VARIANT["visvalingam_whyatt"](points, 0.001)
        assert simplified[0].tolist() == [0.0, 0.0]
        assert simplified[-1].tolist() == [3.0, 5.0]

    def test_larger_thresholds_remove_more_points(self):
        points = np.array([[0.0, 0.0], [1.0, 0.2], [2.0, 0.0], [3.0, 5.0], [4.0, 0.0]])
        loose = MASK_VARIANT["visvalingam_whyatt"](points, 5.0)
        tight = MASK_VARIANT["visvalingam_whyatt"](points, 0.001)
        assert len(loose) <= len(tight) <= len(points)


class TestVisvalingamWhyattSimplify:
    def test_short_polylines_are_returned_unchanged(self):
        coords = [[0.0, 0.0], [1.0, 1.0]]
        assert ITERATIVE_VARIANT["visvalingam_whyatt_simplify"](coords, 0.1) == coords

    def test_never_reduces_below_three_points(self):
        simplified = ITERATIVE_VARIANT["visvalingam_whyatt_simplify"](
            STRAIGHT_LINE, 1e9
        )
        assert len(simplified) == 3

    def test_keeps_points_above_the_area_threshold(self):
        points = np.array([[0.0, 0.0], [1.0, 0.01], [2.0, 0.0], [3.0, 8.0], [4.0, 0.0]])
        simplified = ITERATIVE_VARIANT["visvalingam_whyatt_simplify"](points, 0.5)
        assert [3.0, 8.0] in simplified.tolist()
        assert [1.0, 0.01] not in simplified.tolist()

    def test_first_and_last_coordinates_are_preserved(self):
        points = np.array([[0.0, 0.0], [1.0, 0.05], [2.0, 0.02], [3.0, 4.0], [4.0, 1.0]])
        simplified = ITERATIVE_VARIANT["visvalingam_whyatt_simplify"](points, 0.5)
        assert simplified[0].tolist() == [0.0, 0.0]
        assert simplified[-1].tolist() == [4.0, 1.0]
