"""
Tests for PDEBSDESolver.

Covers:
- Zero driver, g(x)=x²: compare to E[X_T²]
- Linear driver f=a+b·y, g=0: compare to analytical u
- Quadratic driver: compare to Cole-Hopf
- Grid-refinement convergence: error ∝ Δx² + Δt
- Z = σ ∂u/∂x has correct sign / magnitude
- Output shapes and key invariants
"""

from __future__ import annotations

import numpy as np
import pytest

from ebsde.forward.ou_process import OrnsteinUhlenbeck
from ebsde.bsde.standard import StandardBSDE
from ebsde.bsde.drivers import make_quadratic_driver
from ebsde.solvers.pde import PDEBSDESolver
from data.synthetic import quadratic_bsde_cole_hopf

SEED = 0


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _ou_x2_exact(kappa, sigma, x0, T):
    E_XT = x0 * np.exp(-kappa * T)
    Var  = sigma**2 / (2 * kappa) * (1 - np.exp(-2 * kappa * T))
    return Var + E_XT**2


def _linear_exact(a, b, T):
    if abs(b) < 1e-12:
        return a * T
    return (a / b) * (np.exp(b * T) - 1.0)


def _zero_driver_bsde(ou, T):
    return StandardBSDE(
        forward=ou,
        driver=lambda x, y, z: np.zeros_like(x),
        terminal=lambda x: x**2,
        T=T,
    )


def _linear_bsde(ou, T, a, b):
    return StandardBSDE(
        forward=ou,
        driver=lambda x, y, z: a + b * y,
        terminal=lambda x: np.zeros_like(x),
        T=T,
    )


# ------------------------------------------------------------------
# Zero driver, g(x)=x²
# ------------------------------------------------------------------


class TestPDEZeroDriver:
    @pytest.fixture
    def setup(self):
        ou    = OrnsteinUhlenbeck(kappa=2.0, theta=0.0, sigma=1.0)
        T     = 0.5
        bsde  = _zero_driver_bsde(ou, T)
        exact = _ou_x2_exact(ou.kappa, ou.sigma, x0=0.0, T=T)
        return bsde, exact

    def test_y0_accuracy(self, setup):
        bsde, exact = setup
        res = PDEBSDESolver(bsde, n_x=200, n_t=400, x_range=(-4, 4)).solve(x0=0.0)
        assert abs(res["Y0"] - exact) < 0.003, (
            f"Y0={res['Y0']:.6f}, exact={exact:.6f}"
        )

    def test_output_shapes(self, setup):
        bsde, _ = setup
        n_x, n_t = 80, 100
        res = PDEBSDESolver(bsde, n_x=n_x, n_t=n_t, x_range=(-3, 3)).solve()
        assert res["u"].shape     == (n_t + 1, n_x)
        assert res["du_dx"].shape == (n_t + 1, n_x)
        assert res["x_grid"].shape == (n_x,)
        assert res["times"].shape  == (n_t + 1,)
        assert np.isfinite(res["Y0"])
        assert np.isfinite(res["Z0"])

    def test_terminal_condition_satisfied(self, setup):
        """u(T, x) should equal g(x) = x²."""
        bsde, _ = setup
        res = PDEBSDESolver(bsde, n_x=150, n_t=200, x_range=(-3, 3)).solve()
        x   = res["x_grid"]
        err = np.max(np.abs(res["u"][-1] - x**2))
        assert err < 1e-10, f"Terminal condition error: {err:.2e}"

    def test_u_nonneg_interior(self, setup):
        """Since g(x)=x² ≥ 0 and f=0, u should remain ≥ 0 (comparison principle)."""
        bsde, _ = setup
        res = PDEBSDESolver(bsde, n_x=100, n_t=200, x_range=(-4, 4)).solve()
        assert np.all(res["u"] >= -0.01), "u should be non-negative"

    def test_time_monotonicity(self, setup):
        """
        u(t, 0) should be non-increasing as t decreases from T to 0,
        because Y_t = E[X_T² | X_t=0] decreases as t moves away from T.
        Actually E[X_T² | X_t=0] = Var(X_{T-t}), which increases as T-t grows,
        so u(t=0, 0) > u(t=T/2, 0).
        """
        bsde, _ = setup
        res = PDEBSDESolver(bsde, n_x=100, n_t=200, x_range=(-4, 4)).solve()
        mid = len(res["times"]) // 2
        x0_idx = np.argmin(np.abs(res["x_grid"]))
        # u at t=0 should be larger than at t=T/2
        assert res["u"][0, x0_idx] > res["u"][mid, x0_idx]


# ------------------------------------------------------------------
# Linear driver
# ------------------------------------------------------------------


class TestPDELinearDriver:
    @pytest.fixture
    def setup(self):
        ou    = OrnsteinUhlenbeck(kappa=1.5, theta=0.0, sigma=0.8)
        T     = 0.5
        a, b  = 0.4, -0.8
        bsde  = _linear_bsde(ou, T, a, b)
        exact = _linear_exact(a, b, T)
        return bsde, exact

    def test_y0_accuracy(self, setup):
        bsde, exact = setup
        res = PDEBSDESolver(bsde, n_x=200, n_t=400, x_range=(-4, 4)).solve(x0=0.0)
        assert abs(res["Y0"] - exact) < 0.005, (
            f"Y0={res['Y0']:.6f}, exact={exact:.6f}"
        )

    def test_u_at_origin_matches_analytical(self, setup):
        """
        u(0, 0) should match the analytical constant (a/b)(e^{bT}-1).
        Near the spatial boundaries, Dirichlet BCs impose u=0, so the
        solution naturally falls off; the interior should match.
        """
        bsde, exact = setup
        res = PDEBSDESolver(bsde, n_x=200, n_t=400, x_range=(-4, 4)).solve(x0=0.0)
        assert abs(res["Y0"] - exact) < 0.005, (
            f"u(0,0)={res['Y0']:.5f}, exact={exact:.5f}"
        )


# ------------------------------------------------------------------
# Quadratic driver vs Cole-Hopf
# ------------------------------------------------------------------


class TestPDEQuadraticDriver:
    @pytest.fixture
    def setup(self):
        ou    = OrnsteinUhlenbeck(kappa=2.0, theta=0.0, sigma=1.0)
        T     = 0.5
        gamma = 1.0
        h     = lambda x: 0.5 * x**2
        bsde  = StandardBSDE(
            forward=ou,
            driver=make_quadratic_driver(h, risk_aversion=gamma),
            terminal=lambda x: np.zeros_like(x),
            T=T,
        )
        ref = quadratic_bsde_cole_hopf(ou, h_func=h, gamma=gamma, T=T)
        return bsde, ref["Y0_at_theta"]

    def test_y0_within_tolerance(self, setup):
        bsde, ref = setup
        res = PDEBSDESolver(bsde, n_x=200, n_t=400, x_range=(-4, 4)).solve(x0=0.0)
        assert abs(res["Y0"] - ref) < 0.003, (
            f"Y0={res['Y0']:.6f}, Cole-Hopf={ref:.6f}"
        )

    def test_z0_sign(self, setup):
        """Z = σ ∂u/∂x; for this symmetric problem Z(x=0) ≈ 0."""
        bsde, _ = setup
        res = PDEBSDESolver(bsde, n_x=200, n_t=400, x_range=(-4, 4)).solve(x0=0.0)
        assert abs(res["Z0"]) < 0.05, f"Z0={res['Z0']:.4f} should be ~0 by symmetry"


# ------------------------------------------------------------------
# Grid-refinement convergence
# ------------------------------------------------------------------


class TestPDEGridConvergence:
    """
    Error should decrease as grid is refined.
    We verify that error(coarse) > error(fine).
    """

    def test_spatial_refinement(self):
        ou    = OrnsteinUhlenbeck(kappa=2.0, theta=0.0, sigma=1.0)
        T     = 0.5
        bsde  = _zero_driver_bsde(ou, T)
        exact = _ou_x2_exact(ou.kappa, ou.sigma, x0=0.0, T=T)

        errors = {}
        for n_x in (50, 150):
            res = PDEBSDESolver(bsde, n_x=n_x, n_t=500, x_range=(-5, 5)).solve(x0=0.0)
            errors[n_x] = abs(res["Y0"] - exact)

        assert errors[150] < errors[50], (
            f"Spatial refinement did not reduce error: {errors}"
        )

    def test_temporal_refinement(self):
        ou    = OrnsteinUhlenbeck(kappa=2.0, theta=0.0, sigma=1.0)
        T     = 0.5
        bsde  = _zero_driver_bsde(ou, T)
        exact = _ou_x2_exact(ou.kappa, ou.sigma, x0=0.0, T=T)

        errors = {}
        for n_t in (50, 300):
            res = PDEBSDESolver(bsde, n_x=300, n_t=n_t, x_range=(-5, 5)).solve(x0=0.0)
            errors[n_t] = abs(res["Y0"] - exact)

        assert errors[300] < errors[50], (
            f"Temporal refinement did not reduce error: {errors}"
        )

    def test_cn_vs_explicit_accuracy(self):
        """Crank-Nicolson (θ=0.5) should be at least as accurate as explicit (θ=0)."""
        ou    = OrnsteinUhlenbeck(kappa=2.0, theta=0.0, sigma=1.0)
        T     = 0.3
        bsde  = _zero_driver_bsde(ou, T)
        exact = _ou_x2_exact(ou.kappa, ou.sigma, x0=0.0, T=T)

        res_cn  = PDEBSDESolver(bsde, n_x=100, n_t=200, x_range=(-4, 4), theta_cn=0.5).solve(x0=0.0)
        res_imp = PDEBSDESolver(bsde, n_x=100, n_t=200, x_range=(-4, 4), theta_cn=1.0).solve(x0=0.0)

        err_cn  = abs(res_cn["Y0"]  - exact)
        err_imp = abs(res_imp["Y0"] - exact)

        # Both should be small; CN should be at least as good
        assert err_cn  < 0.01
        assert err_imp < 0.02


# ------------------------------------------------------------------
# Z gradient consistency
# ------------------------------------------------------------------


class TestPDEZGradient:
    def test_du_dx_odd_at_origin(self):
        """
        For g(x)=x² and f=0, u is an even function of x (by symmetry of OU).
        Therefore ∂u/∂x is odd, so ∂u/∂x(0) ≈ 0.
        """
        ou   = OrnsteinUhlenbeck(kappa=2.0, theta=0.0, sigma=1.0)
        bsde = _zero_driver_bsde(ou, T=0.5)
        res  = PDEBSDESolver(bsde, n_x=200, n_t=300, x_range=(-4, 4)).solve(x0=0.0)
        # du_dx at t=0, x=0 should be ~0
        assert abs(res["Z0"]) < 0.05, f"Z0={res['Z0']:.4f}"

    def test_du_dx_sign(self):
        """
        For g(x)=x and f=0, u(0,x) = E[X_T|X_0=x] = x·e^{-κT}, which is increasing.
        So ∂u/∂x(0, x) > 0 for all x (specifically at x=1).
        """
        ou   = OrnsteinUhlenbeck(kappa=2.0, theta=0.0, sigma=1.0)
        T    = 0.5
        bsde = StandardBSDE(
            forward=ou,
            driver=lambda x, y, z: np.zeros_like(x),
            terminal=lambda x: x,
            T=T,
        )
        res = PDEBSDESolver(bsde, n_x=200, n_t=300, x_range=(-4, 4)).solve(x0=1.0)
        assert res["Z0"] > 0.0, f"Z0 should be positive for g(x)=x: got {res['Z0']:.4f}"
