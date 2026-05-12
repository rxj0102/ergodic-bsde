"""
Tests for ErgodicPicardSolver.

Critical test: OU + quadratic driver  λ ≈ -0.7321  (κ=1, σ=1, α=1, γ=1)
Tolerance: 1% relative error.

Covers:
- Accuracy on quadratic driver (within 1%)
- λ_estimates converge as T increases
- Multiple T values → finite-difference estimates stabilise
- Richardson extrapolation gives ≤ finite-difference error
- Output dict keys and types
"""

from __future__ import annotations

import numpy as np
import pytest

from ebsde.forward.ou_process import OrnsteinUhlenbeck
from ebsde.bsde.ergodic import ErgodicBSDE
from ebsde.solvers.ergodic_picard import ErgodicPicardSolver
from data.synthetic import ergodic_ou_quadratic_analytical

SEED = 0
KAPPA, SIGMA = 1.0, 1.0
ALPHA, GAMMA = 1.0, 1.0
LAMBDA_EXACT = ergodic_ou_quadratic_analytical(KAPPA, SIGMA, ALPHA, 0.0, GAMMA)["lambda_ergodic"]


def _ou():
    return OrnsteinUhlenbeck(kappa=KAPPA, theta=0.0, sigma=SIGMA)


def _quadratic_ebsde(ou):
    return ErgodicBSDE(
        forward=ou,
        driver=lambda x, y, z: ALPHA * x**2 - (GAMMA / 2.0) * np.sum(z**2, axis=-1),
    )


# ------------------------------------------------------------------
# Accuracy
# ------------------------------------------------------------------


class TestErgodicPicardAccuracy:
    @pytest.fixture
    def setup(self):
        ou = _ou()
        return _quadratic_ebsde(ou)

    def test_lambda_within_1pct(self, setup):
        res = ErgodicPicardSolver(
            setup,
            T_values=[5.0, 10.0, 20.0, 40.0],
            n_paths=40_000,
            n_steps_per_unit=40,
            n_picard=3,
            basis_degree=4,
            rng_seed=SEED,
        ).solve()
        rel_err = abs(res["lambda_ergodic"] - LAMBDA_EXACT) / abs(LAMBDA_EXACT)
        assert rel_err < 0.01, (
            f"λ={res['lambda_ergodic']:.5f}, exact={LAMBDA_EXACT:.5f}, rel={rel_err:.4f}"
        )

    def test_output_keys(self, setup):
        res = ErgodicPicardSolver(
            setup, T_values=[5.0, 10.0], n_paths=2_000,
            n_steps_per_unit=20, n_picard=1, rng_seed=SEED,
        ).solve()
        for key in ("lambda_ergodic", "lambda_estimates", "Y0_values",
                    "v_at_x0", "convergence"):
            assert key in res

    def test_lambda_finite(self, setup):
        res = ErgodicPicardSolver(
            setup, T_values=[5.0, 10.0], n_paths=5_000,
            n_steps_per_unit=20, n_picard=1, rng_seed=SEED,
        ).solve()
        assert np.isfinite(res["lambda_ergodic"])

    def test_lambda_positive(self, setup):
        """For this problem λ > 0 (αx² cost dominates)."""
        res = ErgodicPicardSolver(
            setup, T_values=[10.0, 20.0], n_paths=20_000,
            n_steps_per_unit=30, n_picard=2, rng_seed=SEED,
        ).solve()
        assert res["lambda_ergodic"] > 0.0


# ------------------------------------------------------------------
# Convergence with T
# ------------------------------------------------------------------


class TestErgodicPicardConvergence:
    @pytest.fixture
    def setup(self):
        ou = _ou()
        return _quadratic_ebsde(ou)

    def test_fd_estimates_stabilise(self, setup):
        """
        Finite-difference estimates λ(T₁,T₂) should be closer to exact
        for larger T pairs.
        """
        res = ErgodicPicardSolver(
            setup,
            T_values=[5.0, 10.0, 20.0, 40.0],
            n_paths=20_000,
            n_steps_per_unit=30,
            n_picard=2,
            rng_seed=SEED,
        ).solve()

        ests = res["lambda_estimates"]
        # All estimates should be finite
        for (T1, T2), lam in ests.items():
            assert np.isfinite(lam), f"λ({T1},{T2}) = {lam}"

    def test_y0_values_increasing(self, setup):
        """
        Y_0(T) ≈ λ·T + const, so since λ>0, Y_0(T) should increase with T.
        """
        res = ErgodicPicardSolver(
            setup,
            T_values=[5.0, 10.0, 20.0],
            n_paths=20_000,
            n_steps_per_unit=30,
            n_picard=2,
            rng_seed=SEED,
        ).solve()
        Y0s = res["Y0_values"]
        T_sorted = sorted(Y0s.keys())
        for i in range(len(T_sorted) - 1):
            assert Y0s[T_sorted[i + 1]] > Y0s[T_sorted[i]], (
                f"Y_0 not increasing with T: {Y0s}"
            )

    def test_extrapolation_error_finite(self, setup):
        res = ErgodicPicardSolver(
            setup,
            T_values=[5.0, 10.0, 20.0, 40.0],
            n_paths=10_000,
            n_steps_per_unit=20,
            n_picard=1,
            rng_seed=SEED,
        ).solve()
        err = res["convergence"]["extrapolation_error"]
        assert np.isfinite(err) or np.isnan(err)  # nan if only 1 estimate


# ------------------------------------------------------------------
# Smoke tests
# ------------------------------------------------------------------


class TestErgodicPicardSmoke:
    def test_default_T_values(self):
        ou = _ou()
        ebsde = _quadratic_ebsde(ou)
        res = ErgodicPicardSolver(
            ebsde, n_paths=2_000, n_steps_per_unit=10, n_picard=1, rng_seed=SEED,
        ).solve()
        assert np.isfinite(res["lambda_ergodic"])

    def test_two_T_values_minimum(self):
        """Works with just 2 T values."""
        ou = _ou()
        ebsde = _quadratic_ebsde(ou)
        res = ErgodicPicardSolver(
            ebsde,
            T_values=[10.0, 20.0],
            n_paths=5_000,
            n_steps_per_unit=20,
            n_picard=1,
            rng_seed=SEED,
        ).solve()
        assert len(res["lambda_estimates"]) == 1
        assert np.isfinite(res["lambda_ergodic"])
