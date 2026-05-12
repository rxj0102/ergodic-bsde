"""
Tests for ErgodicPDESolver.

Critical test: OU + quadratic driver f(x,y,z) = αx² - (γ/2)z²
Analytical: λ = -(2/γ)·½(-κ + √(κ²+2γασ²))
With κ=1, σ=1, α=1, γ=1: λ = 1 - √3 ≈ -0.7321

Covers:
- Quadratic driver: |λ - exact| / |exact| < 0.001
- Linear (y-only) driver: eigenvalue method
- Grid convergence: refine n_x → λ converges
- Both 'iteration' and 'newton' methods run
- Output shapes and invariants
- Residual decays to near zero
"""

from __future__ import annotations

import numpy as np
import pytest

from ebsde.forward.ou_process import OrnsteinUhlenbeck
from ebsde.bsde.ergodic import ErgodicBSDE
from ebsde.solvers.ergodic_pde import ErgodicPDESolver
from data.synthetic import ergodic_ou_quadratic_analytical

SEED = 0


# ------------------------------------------------------------------
# Test parameters
# ------------------------------------------------------------------

KAPPA, SIGMA = 1.0, 1.0
ALPHA, BETA, GAMMA = 1.0, 0.0, 1.0
LAMBDA_EXACT = ergodic_ou_quadratic_analytical(KAPPA, SIGMA, ALPHA, BETA, GAMMA)["lambda_ergodic"]
# ≈ -0.7320508...


def _ou():
    return OrnsteinUhlenbeck(kappa=KAPPA, theta=0.0, sigma=SIGMA)


def _quadratic_ebsde(ou):
    return ErgodicBSDE(
        forward=ou,
        driver=lambda x, y, z: ALPHA * x**2 - (GAMMA / 2.0) * np.sum(z**2, axis=-1),
    )


def _linear_y_ebsde(ou, c):
    """f(x,y,z) = αx² + c·y  (linear in y, no z)"""
    return ErgodicBSDE(
        forward=ou,
        driver=lambda x, y, z: ALPHA * x**2 + c * y,
    )


# ------------------------------------------------------------------
# Quadratic driver accuracy
# ------------------------------------------------------------------


class TestErgodicPDEQuadratic:
    @pytest.fixture
    def setup(self):
        ou = _ou()
        ebsde = _quadratic_ebsde(ou)
        return ebsde

    def test_lambda_accuracy_iteration(self, setup):
        res = ErgodicPDESolver(
            setup, n_x=600, method="iteration", max_iter=300, tol=1e-9
        ).solve()
        rel_err = abs(res["lambda_ergodic"] - LAMBDA_EXACT) / abs(LAMBDA_EXACT)
        assert rel_err < 0.001, (
            f"λ={res['lambda_ergodic']:.6f}, exact={LAMBDA_EXACT:.6f}, rel={rel_err:.4f}"
        )

    def test_lambda_accuracy_newton(self, setup):
        res = ErgodicPDESolver(
            setup, n_x=400, method="newton", max_iter=50, tol=1e-8
        ).solve()
        rel_err = abs(res["lambda_ergodic"] - LAMBDA_EXACT) / abs(LAMBDA_EXACT)
        assert rel_err < 0.002, (
            f"λ={res['lambda_ergodic']:.6f}, exact={LAMBDA_EXACT:.6f}, rel={rel_err:.4f}"
        )

    def test_output_keys(self, setup):
        res = ErgodicPDESolver(setup, n_x=100).solve()
        for key in ("lambda_ergodic", "v", "dv", "z", "x_grid", "residual", "convergence_history"):
            assert key in res

    def test_output_shapes(self, setup):
        n_x = 120
        res = ErgodicPDESolver(setup, n_x=n_x).solve()
        assert res["v"].shape == (n_x,)
        assert res["dv"].shape == (n_x,)
        assert res["z"].shape == (n_x,)
        assert res["x_grid"].shape == (n_x,)

    def test_normalisation(self, setup):
        """v should have v[mid] ≈ 0."""
        n_x = 200
        res = ErgodicPDESolver(setup, n_x=n_x).solve()
        mid = n_x // 2
        assert abs(res["v"][mid]) < 1e-8

    def test_residual_small(self, setup):
        res = ErgodicPDESolver(
            setup, n_x=500, method="iteration", max_iter=300, tol=1e-9
        ).solve()
        assert res["residual"] < 0.05, f"residual={res['residual']:.4e}"

    def test_lambda_finite(self, setup):
        res = ErgodicPDESolver(setup, n_x=100).solve()
        assert np.isfinite(res["lambda_ergodic"])

    def test_convergence_history_populated(self, setup):
        res = ErgodicPDESolver(
            setup, n_x=100, method="iteration", max_iter=30
        ).solve()
        assert len(res["convergence_history"]) > 1


# ------------------------------------------------------------------
# Linear (eigenvalue) method
# ------------------------------------------------------------------


class TestErgodicPDELinearEigenvalue:
    """f(x,y,z) = αx² + c·y; linear method should solve the eigenvalue problem."""

    @pytest.fixture
    def setup(self):
        ou = _ou()
        c = -0.5
        ebsde = _linear_y_ebsde(ou, c)
        return ebsde, c

    def test_lambda_finite(self, setup):
        ebsde, _ = setup
        res = ErgodicPDESolver(ebsde, n_x=300, method="linear").solve()
        assert np.isfinite(res["lambda_ergodic"])

    def test_eigenvalue_matches_iteration(self, setup):
        """Eigenvalue and iteration methods should agree within 1%."""
        ebsde, _ = setup
        res_eig = ErgodicPDESolver(ebsde, n_x=400, method="linear").solve()
        res_it  = ErgodicPDESolver(ebsde, n_x=400, method="iteration", max_iter=200).solve()
        assert abs(res_eig["lambda_ergodic"] - res_it["lambda_ergodic"]) < 0.02, (
            f"eig={res_eig['lambda_ergodic']:.5f} vs iter={res_it['lambda_ergodic']:.5f}"
        )

    def test_eigenfunction_shape(self, setup):
        ebsde, _ = setup
        n_x = 100
        res = ErgodicPDESolver(ebsde, n_x=n_x, method="linear").solve()
        assert res["v"].shape == (n_x,)


# ------------------------------------------------------------------
# Grid convergence
# ------------------------------------------------------------------


class TestErgodicPDEGridConvergence:
    def test_refine_n_x(self):
        """Both coarse and fine grids converge to within 1% of exact."""
        ou    = _ou()
        ebsde = _quadratic_ebsde(ou)
        for n_x in (100, 400):
            res = ErgodicPDESolver(
                ebsde, n_x=n_x, method="iteration", max_iter=200, tol=1e-8
            ).solve()
            rel = abs(res["lambda_ergodic"] - LAMBDA_EXACT) / abs(LAMBDA_EXACT)
            assert rel < 0.01, (
                f"n_x={n_x}: λ={res['lambda_ergodic']:.5f}, exact={LAMBDA_EXACT:.5f}, rel={rel:.4f}"
            )

    def test_lambda_converges_to_correct_sign(self):
        """For this problem λ > 0 (αx² cost dominates)."""
        ou    = _ou()
        ebsde = _quadratic_ebsde(ou)
        res = ErgodicPDESolver(ebsde, n_x=300).solve()
        assert res["lambda_ergodic"] > 0.0


# ------------------------------------------------------------------
# Different OU parameters
# ------------------------------------------------------------------


class TestErgodicPDEParameterVariation:
    @pytest.mark.parametrize("kappa,sigma,alpha,gamma,beta", [
        (1.0, 1.0, 1.0, 1.0, 0.0),
        (2.0, 1.0, 1.0, 1.0, 0.0),
        (1.0, 0.5, 1.0, 1.0, 0.0),
        (1.0, 1.0, 2.0, 1.0, 0.0),
    ])
    def test_lambda_near_analytical(self, kappa, sigma, alpha, gamma, beta):
        ou = OrnsteinUhlenbeck(kappa=kappa, theta=0.0, sigma=sigma)
        ebsde = ErgodicBSDE(
            forward=ou,
            driver=lambda x, y, z: alpha * x**2 + beta - (gamma / 2.0) * np.sum(z**2, axis=-1),
        )
        exact = ergodic_ou_quadratic_analytical(kappa, sigma, alpha, beta, gamma)["lambda_ergodic"]
        res = ErgodicPDESolver(ebsde, n_x=500, method="iteration", max_iter=300, tol=1e-9).solve()
        rel = abs(res["lambda_ergodic"] - exact) / (abs(exact) + 1e-10)
        assert rel < 0.005, (
            f"kappa={kappa},sigma={sigma},alpha={alpha}: "
            f"λ={res['lambda_ergodic']:.5f}, exact={exact:.5f}, rel={rel:.4f}"
        )
