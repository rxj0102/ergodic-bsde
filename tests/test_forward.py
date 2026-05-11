"""Tests for forward SDE simulation."""

from __future__ import annotations

import numpy as np
import pytest
from scipy import stats

from ebsde.forward.ou_process import OrnsteinUhlenbeck
from ebsde.forward.cev_process import CEVProcess
from ebsde.forward.multidimensional import MultiDimOU


# ------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------

SEED = 42


@pytest.fixture
def ou_default():
    return OrnsteinUhlenbeck(kappa=1.0, theta=0.5, sigma=0.8)


@pytest.fixture
def rng():
    return np.random.default_rng(SEED)


# ------------------------------------------------------------------
# OrnsteinUhlenbeck: basic
# ------------------------------------------------------------------


class TestOUBasic:
    def test_drift(self, ou_default):
        x = np.array([[2.0]])
        expected = -ou_default.kappa * (2.0 - ou_default.theta)
        assert np.isclose(ou_default.drift(x), expected)

    def test_diffusion_constant(self, ou_default):
        x = np.array([[0.0], [1.0], [5.0]])
        sig = ou_default.diffusion(x)
        assert np.allclose(sig, ou_default.sigma)

    def test_is_ergodic(self, ou_default):
        assert ou_default.is_ergodic()

    def test_negative_kappa_raises(self):
        with pytest.raises(ValueError):
            OrnsteinUhlenbeck(kappa=-0.1)

    def test_zero_kappa_raises(self):
        with pytest.raises(ValueError):
            OrnsteinUhlenbeck(kappa=0.0)

    def test_stationary_distribution(self, ou_default):
        sd = ou_default.stationary_distribution()
        assert np.isclose(sd["mean"], ou_default.theta)
        expected_var = ou_default.sigma**2 / (2 * ou_default.kappa)
        assert np.isclose(sd["variance"], expected_var)

    def test_dimension(self, ou_default):
        assert ou_default.dimension == 1


# ------------------------------------------------------------------
# OrnsteinUhlenbeck: Euler simulation
# ------------------------------------------------------------------


class TestOUSimulation:
    def test_shape(self, ou_default, rng):
        result = ou_default.simulate(
            x0=np.array([0.0]), T=1.0, n_steps=100, n_paths=50, rng=rng
        )
        assert result["paths"].shape == (50, 101, 1)
        assert result["brownian_increments"].shape == (50, 100, 1)
        assert result["times"].shape == (101,)

    def test_initial_condition(self, ou_default, rng):
        x0 = np.array([1.5])
        result = ou_default.simulate(x0=x0, T=1.0, n_steps=100, n_paths=20, rng=rng)
        assert np.allclose(result["paths"][:, 0, 0], 1.5)

    def test_reproducible(self, ou_default):
        r1 = ou_default.simulate(
            x0=np.array([0.0]), T=1.0, n_steps=50, n_paths=10,
            rng=np.random.default_rng(SEED)
        )
        r2 = ou_default.simulate(
            x0=np.array([0.0]), T=1.0, n_steps=50, n_paths=10,
            rng=np.random.default_rng(SEED)
        )
        assert np.allclose(r1["paths"], r2["paths"])

    def test_unknown_scheme_raises(self, ou_default, rng):
        with pytest.raises(ValueError, match="Unknown scheme"):
            ou_default.simulate(
                x0=np.array([0.0]), T=1.0, n_steps=10, n_paths=5,
                scheme="bogus", rng=rng
            )

    def test_times_correct(self, ou_default, rng):
        result = ou_default.simulate(
            x0=np.array([0.0]), T=2.0, n_steps=200, n_paths=5, rng=rng
        )
        assert np.isclose(result["times"][0], 0.0)
        assert np.isclose(result["times"][-1], 2.0)
        assert len(result["times"]) == 201


# ------------------------------------------------------------------
# OU: Euler vs exact simulation match as dt → 0
# ------------------------------------------------------------------


class TestOUEulerVsExact:
    """Euler-Maruyama converges to the exact simulation in L2 as dt → 0."""

    def test_euler_converges_to_exact(self, ou_default):
        """
        For the same Brownian increments, the Euler path and the exact
        path should be close for small dt.  We compare the terminal
        distribution moments: as n_steps → ∞ they match.
        """
        ou = OrnsteinUhlenbeck(kappa=2.0, theta=0.0, sigma=1.0)
        T = 0.5
        n_paths = 4000

        errors = []
        for n_steps in (50, 200, 800):
            euler = ou.simulate(
                x0=np.array([0.0]), T=T, n_steps=n_steps, n_paths=n_paths,
                rng=np.random.default_rng(SEED)
            )
            exact = ou.exact_simulate(
                x0=0.0, T=T, n_steps=n_steps, n_paths=n_paths,
                rng=np.random.default_rng(SEED)
            )
            euler_mean = euler["paths"][:, -1, 0].mean()
            exact_mean = exact["paths"][:, -1, 0].mean()
            errors.append(abs(euler_mean - exact_mean))

        # Error should decrease as n_steps increases
        assert errors[1] < errors[0] or errors[2] < errors[1], (
            f"Expected Euler to converge; errors: {errors}"
        )


# ------------------------------------------------------------------
# OU: Ergodicity — long-run moments match stationary distribution
# ------------------------------------------------------------------


class TestOUErgodicity:
    def test_long_run_mean(self):
        ou = OrnsteinUhlenbeck(kappa=2.0, theta=1.5, sigma=0.5)
        sim = ou.simulate(
            x0=np.array([0.0]), T=30.0, n_steps=30_000, n_paths=1,
            rng=np.random.default_rng(SEED)
        )
        path = sim["paths"][0, 5000:, 0]  # discard burn-in
        assert abs(path.mean() - ou.theta) < 0.15, (
            f"Long-run mean {path.mean():.4f} far from θ={ou.theta}"
        )

    def test_long_run_variance(self):
        ou = OrnsteinUhlenbeck(kappa=2.0, theta=1.5, sigma=0.5)
        sim = ou.simulate(
            x0=np.array([0.0]), T=30.0, n_steps=30_000, n_paths=1,
            rng=np.random.default_rng(SEED)
        )
        path = sim["paths"][0, 5000:, 0]
        target_var = ou.sigma**2 / (2 * ou.kappa)
        assert abs(path.var() - target_var) < 0.05, (
            f"Long-run variance {path.var():.4f} far from target {target_var:.4f}"
        )

    def test_stationary_sample_mean(self):
        ou = OrnsteinUhlenbeck(kappa=1.0, theta=2.0, sigma=1.0)
        samples = ou.simulate_stationary(
            n_samples=2000, burn_in=5000, thin=50, dt=0.01,
            rng=np.random.default_rng(SEED)
        )
        assert samples.shape == (2000, 1)
        assert abs(samples.mean() - ou.theta) < 0.15

    def test_stationary_sample_variance(self):
        ou = OrnsteinUhlenbeck(kappa=1.0, theta=2.0, sigma=1.0)
        samples = ou.simulate_stationary(
            n_samples=2000, burn_in=5000, thin=50, dt=0.01,
            rng=np.random.default_rng(SEED)
        )
        target_var = ou.sigma**2 / (2 * ou.kappa)
        assert abs(samples.var() - target_var) < 0.2


# ------------------------------------------------------------------
# OU: exact simulation
# ------------------------------------------------------------------


class TestOUExact:
    def test_exact_shape(self):
        ou = OrnsteinUhlenbeck(kappa=1.0, theta=0.0, sigma=1.0)
        result = ou.exact_simulate(
            x0=0.0, T=1.0, n_steps=100, n_paths=50,
            rng=np.random.default_rng(SEED)
        )
        assert result["paths"].shape == (50, 101, 1)

    def test_exact_terminal_distribution(self):
        ou = OrnsteinUhlenbeck(kappa=1.0, theta=0.0, sigma=1.0)
        n_paths = 10_000
        T = 5.0
        result = ou.exact_simulate(
            x0=0.0, T=T, n_steps=1, n_paths=n_paths,
            rng=np.random.default_rng(SEED)
        )
        terminal = result["paths"][:, -1, 0]
        sd = ou.stationary_distribution()
        assert abs(terminal.mean() - sd["mean"]) < 0.05
        assert abs(terminal.var() - sd["variance"]) < 0.1

    def test_transition_density_sums_to_one(self):
        ou = OrnsteinUhlenbeck(kappa=1.0, theta=0.0, sigma=1.0)
        y_grid = np.linspace(-5, 5, 2000)
        dy = y_grid[1] - y_grid[0]
        densities = np.array([ou.transition_density(x=0.0, y=y, dt=0.5) for y in y_grid])
        integral = (densities * dy).sum()
        assert abs(integral - 1.0) < 0.01


# ------------------------------------------------------------------
# OU: Milstein scheme
# ------------------------------------------------------------------


class TestOUMilstein:
    def test_milstein_shape(self):
        ou = OrnsteinUhlenbeck(kappa=1.0, theta=0.0, sigma=1.0)
        result = ou.simulate(
            x0=np.array([0.0]), T=1.0, n_steps=100, n_paths=20,
            scheme="milstein", rng=np.random.default_rng(SEED)
        )
        assert result["paths"].shape == (20, 101, 1)


# ------------------------------------------------------------------
# CEV process
# ------------------------------------------------------------------


class TestCEVProcess:
    def test_basic_properties(self):
        cev = CEVProcess(kappa=1.0, theta=1.0, sigma=0.3, gamma=0.5)
        assert cev.dimension == 1
        assert cev.is_ergodic()

    def test_invalid_kappa(self):
        with pytest.raises(ValueError):
            CEVProcess(kappa=-1.0, theta=1.0, sigma=0.3, gamma=0.5)

    def test_paths_stay_positive(self):
        cev = CEVProcess(kappa=1.0, theta=1.0, sigma=0.3, gamma=0.5, x_floor=1e-6)
        result = cev.simulate(
            x0=np.array([1.0]), T=5.0, n_steps=500, n_paths=100,
            rng=np.random.default_rng(SEED)
        )
        assert np.all(result["paths"] >= 0), "CEV paths must stay non-negative"

    def test_shape(self):
        cev = CEVProcess(kappa=1.0, theta=1.0, sigma=0.3, gamma=0.5)
        result = cev.simulate(
            x0=np.array([1.0]), T=2.0, n_steps=200, n_paths=30,
            rng=np.random.default_rng(SEED)
        )
        assert result["paths"].shape == (30, 201, 1)

    def test_diffusion_positive(self):
        cev = CEVProcess(kappa=1.0, theta=1.0, sigma=0.3, gamma=0.5)
        x = np.array([[0.5], [1.0], [2.0]])
        sig = cev.diffusion(x)
        assert np.all(sig > 0)

    def test_reprodicibility(self):
        cev = CEVProcess(kappa=1.0, theta=1.0, sigma=0.3, gamma=0.5)
        r1 = cev.simulate(
            x0=np.array([1.0]), T=1.0, n_steps=50, n_paths=10,
            rng=np.random.default_rng(SEED)
        )
        r2 = cev.simulate(
            x0=np.array([1.0]), T=1.0, n_steps=50, n_paths=10,
            rng=np.random.default_rng(SEED)
        )
        assert np.allclose(r1["paths"], r2["paths"])


# ------------------------------------------------------------------
# Multi-dimensional OU
# ------------------------------------------------------------------


class TestMultiDimOU:
    @pytest.fixture
    def multi_ou(self):
        d = 2
        K = np.array([[2.0, 0.5], [0.0, 1.5]])
        theta = np.array([1.0, -1.0])
        Sigma = np.array([[0.8, 0.0], [0.2, 0.6]])
        return MultiDimOU(K=K, theta=theta, Sigma=Sigma)

    def test_dimension(self, multi_ou):
        assert multi_ou.dimension == 2

    def test_is_ergodic(self, multi_ou):
        assert multi_ou.is_ergodic()

    def test_non_ergodic_raises(self):
        K_bad = np.array([[1.0, 0.0], [0.0, -1.0]])
        theta = np.zeros(2)
        Sigma = np.eye(2)
        with pytest.raises(ValueError, match="positive real parts"):
            MultiDimOU(K=K_bad, theta=theta, Sigma=Sigma)

    def test_shape(self, multi_ou):
        result = multi_ou.simulate(
            x0=np.array([0.0, 0.0]), T=2.0, n_steps=200, n_paths=30,
            rng=np.random.default_rng(SEED)
        )
        assert result["paths"].shape == (30, 201, 2)

    def test_stationary_covariance_lyapunov(self, multi_ou):
        """V should satisfy K V + V K^T = Σ Σ^T."""
        V = multi_ou.stationary_covariance()
        Q = multi_ou.Sigma @ multi_ou.Sigma.T
        residual = multi_ou.K @ V + V @ multi_ou.K.T - Q
        assert np.allclose(residual, 0.0, atol=1e-10)

    def test_stationary_covariance_symmetric_posdef(self, multi_ou):
        V = multi_ou.stationary_covariance()
        assert np.allclose(V, V.T, atol=1e-10)
        assert np.all(np.linalg.eigvalsh(V) > 0)

    def test_long_run_covariance_matches_analytical(self, multi_ou):
        """Sample covariance from a long path ≈ analytical stationary covariance."""
        result = multi_ou.simulate(
            x0=np.zeros(2), T=200.0, n_steps=200_000, n_paths=1,
            rng=np.random.default_rng(SEED)
        )
        path = result["paths"][0, 50_000:, :]  # discard burn-in
        sample_cov = np.cov(path.T)
        V = multi_ou.stationary_covariance()
        assert np.allclose(sample_cov, V, atol=0.15), (
            f"Sample cov:\n{sample_cov}\nAnalytical:\n{V}"
        )

    def test_reproducible(self, multi_ou):
        r1 = multi_ou.simulate(
            x0=np.zeros(2), T=1.0, n_steps=50, n_paths=10,
            rng=np.random.default_rng(SEED)
        )
        r2 = multi_ou.simulate(
            x0=np.zeros(2), T=1.0, n_steps=50, n_paths=10,
            rng=np.random.default_rng(SEED)
        )
        assert np.allclose(r1["paths"], r2["paths"])
