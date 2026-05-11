"""Tests for BSDE driver functions."""

from __future__ import annotations

import numpy as np
import pytest

from ebsde.bsde.drivers import (
    linear_driver,
    quadratic_driver,
    entropy_driver,
    power_utility_driver,
    make_quadratic_driver,
    make_entropy_driver,
)


SEED = 42


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def lipschitz_check_fd(f_partial, lo: float = -3.0, hi: float = 3.0,
                        n: int = 50) -> float:
    """
    Estimate the Lipschitz constant of f_partial (a 1-D function) via
    finite differences over [lo, hi].  Returns the maximum |f'|.
    """
    xs = np.linspace(lo, hi, n)
    eps = 1e-5
    grads = np.abs((f_partial(xs + eps) - f_partial(xs - eps)) / (2 * eps))
    return float(np.max(grads))


# ------------------------------------------------------------------
# linear_driver
# ------------------------------------------------------------------


class TestLinearDriver:
    def test_zero_coefficients(self):
        x = np.zeros(5)
        y = np.zeros(5)
        z = np.zeros((5, 1))
        out = linear_driver(x, y, z)
        assert np.allclose(out, 0.0)

    def test_a_only(self):
        x = np.zeros(3)
        y = np.zeros(3)
        z = np.zeros((3, 1))
        out = linear_driver(x, y, z, a=2.5)
        assert np.allclose(out, 2.5)

    def test_b_coefficient(self):
        x = np.zeros(4)
        y = np.array([1.0, 2.0, 3.0, 4.0])
        z = np.zeros((4, 1))
        out = linear_driver(x, y, z, b_coeff=3.0)
        assert np.allclose(out, 3.0 * y)

    def test_c_quadratic_term(self):
        x = np.zeros(3)
        y = np.zeros(3)
        z = np.array([[2.0], [0.0], [-3.0]])
        out = linear_driver(x, y, z, c=1.0)
        expected = 0.5 * np.array([4.0, 0.0, 9.0])
        assert np.allclose(out, expected)

    def test_all_coefficients(self):
        x = np.zeros(2)
        y = np.array([1.0, 2.0])
        z = np.array([[1.0], [2.0]])
        out = linear_driver(x, y, z, a=1.0, b_coeff=2.0, c=1.0)
        expected = 1.0 + 2.0 * y + 0.5 * np.array([1.0, 4.0])
        assert np.allclose(out, expected)

    def test_lipschitz_in_y(self):
        x = np.zeros(1)
        z = np.zeros((1, 1))
        f_y = lambda y: linear_driver(x, y, z, b_coeff=5.0)
        L = lipschitz_check_fd(lambda y_arr: np.array([f_y(np.array([y])) for y in y_arr]))
        assert L < 10.0, f"Lipschitz in y too large: {L}"

    def test_lipschitz_in_z_scalar(self):
        x = np.zeros(1)
        y = np.zeros(1)
        f_z = lambda zv: linear_driver(x, y, np.array([[zv]]), c=2.0)
        L = lipschitz_check_fd(lambda zs: np.array([f_z(z) for z in zs]))
        assert L < 30.0, f"Lipschitz in z too large: {L}"

    def test_vectorised(self):
        n = 100
        rng = np.random.default_rng(SEED)
        x = rng.standard_normal(n)
        y = rng.standard_normal(n)
        z = rng.standard_normal((n, 1))
        out = linear_driver(x, y, z, a=1.0, b_coeff=2.0, c=0.5)
        assert out.shape == (n,)


# ------------------------------------------------------------------
# quadratic_driver
# ------------------------------------------------------------------


class TestQuadraticDriver:
    def test_zero_y_z(self):
        x = np.array([1.0, 2.0, 3.0])
        y = np.zeros(3)
        z = np.zeros((3, 1))
        out = quadratic_driver(x, y, z, h_func=lambda x: x**2)
        assert np.allclose(out, x**2), "f(x, 0, 0) should equal h(x)"

    def test_no_h_func(self):
        x = np.zeros(3)
        y = np.zeros(3)
        z = np.zeros((3, 1))
        out = quadratic_driver(x, y, z)
        assert np.allclose(out, 0.0)

    def test_z_penalty(self):
        x = np.zeros(3)
        y = np.zeros(3)
        z = np.array([[1.0], [2.0], [3.0]])
        out = quadratic_driver(x, y, z, risk_aversion=2.0)
        expected = -1.0 * np.array([1.0, 4.0, 9.0])
        assert np.allclose(out, expected)

    def test_y_independence(self):
        """Driver should not depend on y."""
        x = np.zeros(3)
        z = np.zeros((3, 1))
        out0 = quadratic_driver(x, np.zeros(3), z)
        out5 = quadratic_driver(x, 5 * np.ones(3), z)
        assert np.allclose(out0, out5), "Quadratic driver must be independent of y"

    def test_lipschitz_in_z(self):
        x = np.zeros(1)
        y = np.zeros(1)
        f_z = lambda zv: quadratic_driver(x, y, np.array([[zv]]), risk_aversion=1.0)
        L = lipschitz_check_fd(lambda zs: np.array([f_z(z) for z in zs]))
        assert L < 10.0

    def test_factory(self):
        h = lambda x: x**2
        driver = make_quadratic_driver(h, risk_aversion=2.0)
        x = np.array([1.0])
        y = np.zeros(1)
        z = np.zeros((1, 1))
        assert np.isclose(driver(x, y, z), 1.0)


# ------------------------------------------------------------------
# entropy_driver
# ------------------------------------------------------------------


class TestEntropyDriver:
    def test_no_h_zero_z(self):
        x = np.zeros(3)
        y = np.zeros(3)
        z = np.zeros((3, 1))
        out = entropy_driver(x, y, z, penalty=1.0)
        assert np.allclose(out, 0.0)

    def test_with_h(self):
        h = lambda x: 2.0 * np.ones_like(x)
        x = np.zeros(3)
        y = np.zeros(3)
        z = np.zeros((3, 1))
        out = entropy_driver(x, y, z, h_func=h, penalty=1.0)
        assert np.allclose(out, 2.0)

    def test_z_term_positive(self):
        x = np.zeros(3)
        y = np.zeros(3)
        z = np.array([[2.0], [1.0], [0.0]])
        out = entropy_driver(x, y, z, penalty=2.0)
        expected = np.array([1.0, 0.25, 0.0])  # z²/(2η) = z²/4
        assert np.allclose(out, expected)

    def test_invalid_penalty_raises(self):
        with pytest.raises(ValueError):
            entropy_driver(np.zeros(1), np.zeros(1), np.zeros((1, 1)), penalty=-1.0)

    def test_lipschitz_in_z(self):
        x = np.zeros(1)
        y = np.zeros(1)
        f_z = lambda zv: entropy_driver(x, y, np.array([[zv]]), penalty=1.0)
        L = lipschitz_check_fd(lambda zs: np.array([f_z(z) for z in zs]))
        assert L < 10.0

    def test_factory(self):
        h = lambda x: np.ones_like(x)
        driver = make_entropy_driver(h, penalty=2.0)
        x = np.zeros(1)
        y = np.zeros(1)
        z = np.array([[2.0]])
        # h(x) + z²/(2η) = 1 + 4/4 = 2.0
        assert np.isclose(driver(x, y, z), 2.0)


# ------------------------------------------------------------------
# power_utility_driver
# ------------------------------------------------------------------


class TestPowerUtilityDriver:
    def test_merton_formula(self):
        """
        f*(x, y, z) = (μ - r)² / (2γ σ²)
        """
        gamma, r, mu, sigma_mkt = 2.0, 0.02, 0.08, 0.2
        x = np.zeros(3)
        y = np.zeros(3)
        z = np.zeros((3, 1))
        out = power_utility_driver(x, y, z, gamma=gamma, r=r,
                                    mu=mu, sigma_mkt=sigma_mkt)
        expected = (mu - r)**2 / (2 * gamma * sigma_mkt**2)
        assert np.allclose(out, expected)

    def test_independent_of_y(self):
        x = np.zeros(2)
        z = np.zeros((2, 1))
        out0 = power_utility_driver(x, np.zeros(2), z)
        out1 = power_utility_driver(x, 10 * np.ones(2), z)
        assert np.allclose(out0, out1)

    def test_invalid_gamma_raises(self):
        with pytest.raises(ValueError):
            power_utility_driver(np.zeros(1), np.zeros(1), np.zeros((1, 1)), gamma=-1.0)

    def test_invalid_sigma_raises(self):
        with pytest.raises(ValueError):
            power_utility_driver(np.zeros(1), np.zeros(1), np.zeros((1, 1)), sigma_mkt=-0.2)

    def test_output_shape(self):
        n = 50
        rng = np.random.default_rng(SEED)
        x = rng.standard_normal(n)
        y = rng.standard_normal(n)
        z = rng.standard_normal((n, 1))
        out = power_utility_driver(x, y, z)
        assert out.shape == (n,)
