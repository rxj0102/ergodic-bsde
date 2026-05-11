"""
Tests for PicardBSDESolver.

Ground truth:
- Zero driver, g(x)=x²: Y_0 = E[X_T²] = σ²/(2κ)(1-e^{-2κT})
- Linear driver f = a + b·y, g=0: u(0,·) = (a/b)(e^{bT}-1)
- Quadratic driver: compare to Cole-Hopf via data.synthetic
"""

from __future__ import annotations

import numpy as np
import pytest

from ebsde.forward.ou_process import OrnsteinUhlenbeck
from ebsde.bsde.standard import StandardBSDE
from ebsde.bsde.drivers import make_quadratic_driver
from ebsde.solvers.picard import PicardBSDESolver
from data.synthetic import quadratic_bsde_cole_hopf

SEED = 0


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _ou_x2_exact(kappa, sigma, x0, T):
    """E_{x0}[X_T²] for OU(κ, θ=0, σ)."""
    E_XT = x0 * np.exp(-kappa * T)
    Var  = sigma**2 / (2 * kappa) * (1 - np.exp(-2 * kappa * T))
    return Var + E_XT**2


def _linear_exact(a, b, T):
    """u(0,x) = (a/b)(e^{bT}-1) for g=0, f=a+b·y."""
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
# Zero driver, g(x) = x²
# ------------------------------------------------------------------


class TestPicardZeroDriver:
    @pytest.fixture
    def setup(self):
        ou    = OrnsteinUhlenbeck(kappa=2.0, theta=0.0, sigma=1.0)
        T     = 0.5
        bsde  = _zero_driver_bsde(ou, T)
        exact = _ou_x2_exact(ou.kappa, ou.sigma, x0=0.0, T=T)
        return bsde, exact

    def test_y0_accuracy(self, setup):
        bsde, exact = setup
        res = PicardBSDESolver(
            bsde, n_paths=40_000, n_steps=50, n_picard=1,
            basis_degree=4, rng_seed=SEED,
        ).solve()
        assert abs(res["Y0"] - exact) < 0.01, (
            f"Y0={res['Y0']:.5f}, exact={exact:.5f}"
        )

    def test_output_shapes(self, setup):
        bsde, _ = setup
        n_paths, n_steps = 500, 20
        res = PicardBSDESolver(
            bsde, n_paths=n_paths, n_steps=n_steps, n_picard=1, rng_seed=SEED
        ).solve()
        assert res["Y"].shape  == (n_paths, n_steps + 1)
        assert res["Z"].shape  == (n_paths, n_steps, 1)
        assert res["times"].shape == (n_steps + 1,)
        assert len(res["picard_convergence"]) == 1

    def test_terminal_values_nonneg(self, setup):
        bsde, _ = setup
        res = PicardBSDESolver(
            bsde, n_paths=2_000, n_steps=30, n_picard=1, rng_seed=SEED
        ).solve()
        assert np.all(res["Y"][:, -1] >= 0.0), "g(x)=x² ≥ 0 must hold"

    def test_y0_positive(self, setup):
        bsde, _ = setup
        res = PicardBSDESolver(
            bsde, n_paths=5_000, n_steps=30, n_picard=1, rng_seed=SEED
        ).solve()
        assert res["Y0"] > 0.0

    def test_reproducible(self, setup):
        bsde, _ = setup
        y1 = PicardBSDESolver(bsde, n_paths=500, n_steps=20, n_picard=1, rng_seed=7).solve()["Y0"]
        y2 = PicardBSDESolver(bsde, n_paths=500, n_steps=20, n_picard=1, rng_seed=7).solve()["Y0"]
        assert np.isclose(y1, y2)

    def test_picard_convergence_list_length(self, setup):
        bsde, _ = setup
        n_picard = 4
        res = PicardBSDESolver(
            bsde, n_paths=200, n_steps=10, n_picard=n_picard, rng_seed=SEED
        ).solve()
        assert len(res["picard_convergence"]) == n_picard


# ------------------------------------------------------------------
# Linear driver
# ------------------------------------------------------------------


class TestPicardLinearDriver:
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
        res = PicardBSDESolver(
            bsde, n_paths=60_000, n_steps=50, n_picard=1,
            basis_degree=4, rng_seed=SEED,
        ).solve()
        assert abs(res["Y0"] - exact) < 0.02, (
            f"Y0={res['Y0']:.5f}, exact={exact:.5f}"
        )

    def test_one_picard_pass_sufficient(self, setup):
        """Extra iterations must not meaningfully change Y0 for a linear BSDE."""
        bsde, _ = setup
        y1 = PicardBSDESolver(
            bsde, n_paths=20_000, n_steps=40, n_picard=1, rng_seed=SEED
        ).solve()["Y0"]
        y5 = PicardBSDESolver(
            bsde, n_paths=20_000, n_steps=40, n_picard=5, rng_seed=SEED
        ).solve()["Y0"]
        assert abs(y1 - y5) < 5e-4


# ------------------------------------------------------------------
# Quadratic driver: Picard convergence + Cole-Hopf comparison
# ------------------------------------------------------------------


class TestPicardQuadraticDriver:
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
        res = PicardBSDESolver(
            bsde, n_paths=30_000, n_steps=60, n_picard=5,
            basis_degree=4, rng_seed=SEED,
        ).solve()
        assert abs(res["Y0"] - ref) < 0.006, (
            f"Y0={res['Y0']:.5f}, Cole-Hopf={ref:.5f}"
        )

    def test_picard_convergence_decreasing(self, setup):
        bsde, _ = setup
        res = PicardBSDESolver(
            bsde, n_paths=5_000, n_steps=30, n_picard=5, rng_seed=SEED
        ).solve()
        convs = res["picard_convergence"]
        assert convs[-1] <= convs[0], f"Not converging: {convs}"

    def test_picard_residual_shrinks(self, setup):
        """Last residual must be strictly smaller than first."""
        bsde, _ = setup
        res = PicardBSDESolver(
            bsde, n_paths=5_000, n_steps=30, n_picard=4, rng_seed=SEED
        ).solve()
        convs = res["picard_convergence"]
        assert convs[-1] < convs[0], f"Residual did not shrink: {convs}"


# ------------------------------------------------------------------
# Basis-type smoke tests
# ------------------------------------------------------------------


class TestPicardBasisTypes:
    @pytest.fixture
    def bsde(self):
        ou = OrnsteinUhlenbeck(kappa=2.0, theta=0.0, sigma=1.0)
        return _zero_driver_bsde(ou, T=0.3)

    @pytest.mark.parametrize("basis", ["polynomial", "hermite", "rbf"])
    def test_all_basis_types_run(self, bsde, basis):
        res = PicardBSDESolver(
            bsde, n_paths=1_000, n_steps=20, n_picard=1,
            basis_type=basis, basis_degree=3, rng_seed=SEED,
        ).solve()
        assert np.isfinite(res["Y0"])

    def test_invalid_basis_raises(self, bsde):
        with pytest.raises(ValueError):
            PicardBSDESolver(
                bsde, n_paths=100, n_steps=5, basis_type="bogus", rng_seed=SEED
            ).solve()


# ------------------------------------------------------------------
# MC std-error decreases with N
# ------------------------------------------------------------------


class TestPicardMCConvergence:
    def test_std_error_decreases_with_n_paths(self):
        ou   = OrnsteinUhlenbeck(kappa=2.0, theta=0.0, sigma=1.0)
        bsde = _zero_driver_bsde(ou, T=0.3)

        n_reps = 6
        stds = {}
        for n_paths in (2_000, 8_000):
            y0s = [
                PicardBSDESolver(
                    bsde, n_paths=n_paths, n_steps=30, n_picard=1,
                    basis_degree=3, rng_seed=s,
                ).solve()["Y0"]
                for s in range(n_reps)
            ]
            stds[n_paths] = float(np.std(y0s))

        assert stds[8_000] < stds[2_000], (
            f"std did not decrease with N: {stds}"
        )
