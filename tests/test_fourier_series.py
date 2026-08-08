"""Unit tests for ``fourier_series_and_transform10.ipynb``."""

import numpy as np
import pytest

from tests.notebook_loader import load_definitions

FOURIER = load_definitions(
    "fourier_series_and_transform10.ipynb",
    ["square_wave", "calculate_fourier_coefficients", "square_wave_period",
     "reconstruct_fourier_series"],
)


class TestSquareWave:
    def test_takes_only_plus_and_minus_one(self):
        values = FOURIER["square_wave"](np.linspace(-10, 10, 501))
        assert set(np.unique(values)) == {-1.0, 1.0}

    def test_is_periodic_with_the_given_period(self):
        x = np.linspace(-5, 5, 97)
        assert np.allclose(
            FOURIER["square_wave"](x), FOURIER["square_wave"](x + 2 * np.pi)
        )

    def test_custom_period(self):
        assert FOURIER["square_wave"](np.array([0.5]), period=2.0) == 1.0
        assert FOURIER["square_wave"](np.array([1.5]), period=2.0) == -1.0

    def test_period_restriction_matches_the_sign_of_x(self):
        x = np.array([-2.0, -0.5, 0.0, 0.5, 2.0])
        assert FOURIER["square_wave_period"](x).tolist() == [-1.0, -1.0, 1.0, 1.0, 1.0]


@pytest.fixture(scope="module")
def coefficients():
    return FOURIER["calculate_fourier_coefficients"](
        FOURIER["square_wave_period"], np.pi, 6
    )


class TestFourierCoefficients:
    def test_returns_the_requested_number_of_terms(self, coefficients):
        _, an, bn = coefficients
        assert len(an) == len(bn) == 6

    def test_odd_function_has_no_constant_or_cosine_terms(self, coefficients):
        a0, an, _ = coefficients
        assert a0 == pytest.approx(0.0, abs=1e-9)
        assert np.allclose(an, 0.0, atol=1e-9)

    def test_sine_terms_match_the_analytic_square_wave_series(self, coefficients):
        # b_n = 4 / (n * pi) for odd n and 0 for even n.
        _, _, bn = coefficients
        expected = [4 / (n * np.pi) if n % 2 else 0.0 for n in range(1, 7)]
        assert np.allclose(bn, expected, atol=1e-6)


class TestReconstruction:
    def test_constant_term_only(self):
        x = np.linspace(-1, 1, 11)
        reconstructed = FOURIER["reconstruct_fourier_series"](x, 2.5, [], [], np.pi)
        assert np.allclose(reconstructed, 2.5)

    def test_single_sine_term_is_reproduced(self):
        x = np.linspace(-np.pi, np.pi, 65)
        reconstructed = FOURIER["reconstruct_fourier_series"](
            x, 0.0, [0.0], [1.0], np.pi
        )
        assert np.allclose(reconstructed, np.sin(x))

    def test_series_approximates_the_square_wave_away_from_discontinuities(self):
        a0, an, bn = FOURIER["calculate_fourier_coefficients"](
            FOURIER["square_wave_period"], np.pi, 40
        )
        x = np.linspace(0.4, np.pi - 0.4, 50)
        reconstructed = FOURIER["reconstruct_fourier_series"](x, a0, an, bn, np.pi)
        assert np.max(np.abs(reconstructed - 1.0)) < 0.15
