"""Unit tests for the numerical methods notebooks (RREF, Newton-Raphson, fixed point)."""

import numpy as np
import pytest

from tests.notebook_loader import load_definitions

LINALG = load_definitions(
    "RREF_&_NEWTON_RAPHSON_TECH.ipynb", ["rref"], cell_index=1
)
NEWTON = load_definitions(
    "RREF_&_NEWTON_RAPHSON_TECH.ipynb", ["newton_raphson"], cell_index=6
)
NEWTON_VISUAL = load_definitions(
    "RREF_&_NEWTON_RAPHSON_TECH.ipynb", ["newton_raphson_visual"], cell_index=8
)
FIXED_POINT = load_definitions(
    "Fixed_Point_Iteration_num_technique.ipynb", ["g", "gx", "gy"]
)


class TestRref:
    def test_solves_a_system_with_a_unique_solution(self):
        augmented = np.array([[1, 2, -1, -4], [2, 1, 1, 1], [3, 3, 2, 7]])
        reduced = LINALG["rref"](augmented)
        assert np.allclose(reduced[:, :-1], np.identity(3))
        assert np.allclose(reduced[:, -1], [-3, 2, 5])

    def test_does_not_mutate_the_input_matrix(self):
        augmented = np.array([[2.0, 4.0], [1.0, 3.0]])
        original = augmented.copy()
        LINALG["rref"](augmented)
        assert np.array_equal(augmented, original)

    def test_identity_matrix_is_a_fixed_point(self):
        identity = np.identity(4)
        assert np.allclose(LINALG["rref"](identity), identity)

    def test_dependent_rows_are_zeroed(self):
        # Second row is twice the first, so the system is rank deficient.
        reduced = LINALG["rref"](np.array([[1, 2, 3], [2, 4, 6]]))
        assert np.allclose(reduced[0], [1, 2, 3])
        assert np.allclose(reduced[1], [0, 0, 0])

    @pytest.mark.xfail(
        reason="known bug: a column without a pivot advances the row counter too, "
        "so the first row is skipped and the result is not in RREF",
        strict=True,
    )
    def test_leading_zero_column_is_skipped(self):
        reduced = LINALG["rref"](np.array([[0, 1, 2], [0, 2, 5]]))
        assert np.allclose(reduced, [[0, 1, 0], [0, 0, 1]])

    def test_swaps_rows_when_the_pivot_is_zero(self):
        reduced = LINALG["rref"](np.array([[0, 1], [1, 0]]))
        assert np.allclose(reduced, np.identity(2))


class TestNewtonRaphson:
    def test_finds_the_root_of_a_cubic(self):
        root = NEWTON["newton_raphson"](
            lambda x: x**3 - 2 * x - 5, lambda x: 3 * x**2 - 2, 100
        )
        assert root == pytest.approx(2.0945514815, abs=1e-6)

    def test_finds_the_root_of_cos_x_minus_x(self):
        root = NEWTON["newton_raphson"](
            lambda x: np.cos(x) - x, lambda x: -np.sin(x) - 1, 0.5
        )
        assert root == pytest.approx(0.7390851332, abs=1e-6)

    def test_returns_none_when_the_derivative_vanishes(self):
        assert NEWTON["newton_raphson"](lambda x: x**2 + 1, lambda x: 2 * x, 0.0) is None

    def test_returns_none_when_the_iteration_budget_is_exhausted(self):
        # x^(1/3) diverges under Newton-Raphson, so it never meets the tolerance.
        assert (
            NEWTON["newton_raphson"](
                lambda x: np.cbrt(x), lambda x: 1 / (3 * np.cbrt(x) ** 2), 1.0, max_iter=5
            )
            is None
        )

    def test_tolerance_controls_the_accuracy(self):
        loose = NEWTON["newton_raphson"](
            lambda x: x**2 - 2, lambda x: 2 * x, 1.0, tol=1e-1
        )
        tight = NEWTON["newton_raphson"](
            lambda x: x**2 - 2, lambda x: 2 * x, 1.0, tol=1e-12
        )
        assert abs(tight - np.sqrt(2)) < abs(loose - np.sqrt(2))


class TestNewtonRaphsonVisual:
    def test_history_starts_at_the_initial_guess_and_ends_at_the_root(self):
        history = NEWTON_VISUAL["newton_raphson_visual"](
            lambda x: x**3 - 2 * x - 5, lambda x: 3 * x**2 - 2, 100
        )
        assert history[0] == 100
        assert history[-1] == pytest.approx(2.0945514815, abs=1e-6)

    def test_history_is_monotonically_approaching_the_root(self):
        history = NEWTON_VISUAL["newton_raphson_visual"](
            lambda x: x**2 - 2, lambda x: 2 * x, 5.0
        )
        errors = np.abs(history - np.sqrt(2))
        assert np.all(np.diff(errors) <= 0)

    def test_stops_immediately_on_a_flat_derivative(self):
        history = NEWTON_VISUAL["newton_raphson_visual"](
            lambda x: x + 1, lambda x: 0.0, 3.0
        )
        assert history.tolist() == [3.0]


class TestFixedPointIteration:
    def test_cos_iteration_converges_to_the_dottie_number(self):
        x = 0.3
        for _ in range(200):
            x = FIXED_POINT["g"](x)
        assert x == pytest.approx(0.7390851332, abs=1e-9)

    def test_g_is_cosine(self):
        assert FIXED_POINT["g"](0.0) == pytest.approx(1.0)
        assert FIXED_POINT["g"](np.pi) == pytest.approx(-1.0)

    def test_two_dimensional_map_converges_to_a_fixed_point(self):
        x, y = 0.8, 0.2
        for _ in range(200):
            x, y = FIXED_POINT["gx"](x, y), FIXED_POINT["gy"](x, y)
        assert x == pytest.approx(FIXED_POINT["gx"](x, y), abs=1e-9)
        assert y == pytest.approx(FIXED_POINT["gy"](x, y), abs=1e-9)

    def test_two_dimensional_map_formulas(self):
        assert FIXED_POINT["gx"](1.0, 0.0) == pytest.approx(0.6)
        assert FIXED_POINT["gy"](0.0, 1.0) == pytest.approx(0.1)
