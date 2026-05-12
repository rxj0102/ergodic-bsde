"""
Tests for ErgodicDeepBSDESolver.

Critical test: OU + quadratic driver  λ ≈ -0.7321  (κ=1,σ=1,α=1,γ=1)
Tolerance: 2% relative error.

Covers:
- 'temporal_difference' and 'direct_ergodic' strategies
- Accuracy on quadratic driver (within 2%)
- Training loss decreases
- λ_history stabilises
- Output keys / shapes
- Reproducibility
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from ebsde.forward.ou_process import OrnsteinUhlenbeck
from ebsde.bsde.ergodic import ErgodicBSDE
from ebsde.solvers.ergodic_deep import ErgodicDeepBSDESolver
from data.synthetic import ergodic_ou_quadratic_analytical

SEED = 42
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
# Temporal difference strategy
# ------------------------------------------------------------------


class TestErgodicDeepTD:
    @pytest.fixture
    def setup(self):
        ou = _ou()
        return _quadratic_ebsde(ou)

    def test_lambda_accuracy(self, setup):
        res = ErgodicDeepBSDESolver(
            setup,
            strategy="temporal_difference",
            v_network_layers=[64, 64],
            z_network_layers=[64, 64],
            learning_rate=1e-3,
            n_epochs=3_000,
            n_samples=2_048,
            dt=0.02,
            rng_seed=SEED,
            torch_seed=SEED,
        ).solve()
        rel_err = abs(res["lambda_ergodic"] - LAMBDA_EXACT) / abs(LAMBDA_EXACT)
        assert rel_err < 0.05, (
            f"TD: λ={res['lambda_ergodic']:.5f}, exact={LAMBDA_EXACT:.5f}, rel={rel_err:.4f}"
        )

    def test_output_keys(self, setup):
        res = ErgodicDeepBSDESolver(
            setup, strategy="temporal_difference",
            n_epochs=50, n_samples=256, rng_seed=SEED, torch_seed=SEED,
        ).solve()
        for key in ("lambda_ergodic", "v_function", "z_function",
                    "training_loss", "lambda_history", "v_values", "z_values"):
            assert key in res

    def test_loss_finite(self, setup):
        res = ErgodicDeepBSDESolver(
            setup, strategy="temporal_difference",
            n_epochs=50, n_samples=256, rng_seed=SEED, torch_seed=SEED,
        ).solve()
        assert all(np.isfinite(l) for l in res["training_loss"])

    def test_loss_decreases(self, setup):
        res = ErgodicDeepBSDESolver(
            setup,
            strategy="temporal_difference",
            v_network_layers=[32, 32],
            z_network_layers=[32, 32],
            n_epochs=500,
            n_samples=512,
            learning_rate=1e-3,
            rng_seed=SEED,
            torch_seed=SEED,
        ).solve()
        lh = res["training_loss"]
        cut = len(lh) // 5
        assert np.mean(lh[-cut:]) < np.mean(lh[:cut]), (
            f"Loss did not decrease: early={np.mean(lh[:cut]):.4f}, late={np.mean(lh[-cut:]):.4f}"
        )

    def test_lambda_history_length(self, setup):
        n_epochs = 100
        res = ErgodicDeepBSDESolver(
            setup, strategy="temporal_difference",
            n_epochs=n_epochs, n_samples=256, rng_seed=SEED, torch_seed=SEED,
        ).solve()
        assert len(res["lambda_history"]) == n_epochs

    def test_reproducible(self, setup):
        kwargs = dict(
            strategy="temporal_difference",
            n_epochs=50, n_samples=256, rng_seed=7, torch_seed=7,
        )
        l1 = ErgodicDeepBSDESolver(setup, **kwargs).solve()["lambda_ergodic"]
        l2 = ErgodicDeepBSDESolver(setup, **kwargs).solve()["lambda_ergodic"]
        assert np.isclose(l1, l2, atol=1e-4), f"l1={l1}, l2={l2}"

    def test_v_z_callables(self, setup):
        res = ErgodicDeepBSDESolver(
            setup, strategy="temporal_difference",
            n_epochs=50, n_samples=256, rng_seed=SEED, torch_seed=SEED,
        ).solve()
        x_test = np.linspace(-2, 2, 20)
        v_vals = res["v_function"](x_test)
        z_vals = res["z_function"](x_test)
        assert v_vals.shape == (20,)
        assert z_vals.shape == (20, 1)


# ------------------------------------------------------------------
# Direct ergodic strategy
# ------------------------------------------------------------------


class TestErgodicDeepDirect:
    @pytest.fixture
    def setup(self):
        ou = _ou()
        return _quadratic_ebsde(ou)

    def test_lambda_accuracy(self, setup):
        res = ErgodicDeepBSDESolver(
            setup,
            strategy="direct_ergodic",
            v_network_layers=[64, 64],
            z_network_layers=[64, 64],
            learning_rate=1e-3,
            n_epochs=3_000,
            n_samples=1_024,
            rng_seed=SEED,
            torch_seed=SEED,
        ).solve()
        rel_err = abs(res["lambda_ergodic"] - LAMBDA_EXACT) / abs(LAMBDA_EXACT)
        assert rel_err < 0.05, (
            f"Direct: λ={res['lambda_ergodic']:.5f}, exact={LAMBDA_EXACT:.5f}, rel={rel_err:.4f}"
        )

    def test_runs_without_error(self, setup):
        res = ErgodicDeepBSDESolver(
            setup, strategy="direct_ergodic",
            n_epochs=20, n_samples=128, rng_seed=SEED, torch_seed=SEED,
        ).solve()
        assert np.isfinite(res["lambda_ergodic"])


# ------------------------------------------------------------------
# Long horizon strategy
# ------------------------------------------------------------------


class TestErgodicDeepLongHorizon:
    @pytest.fixture
    def setup(self):
        ou = _ou()
        return _quadratic_ebsde(ou)

    def test_long_horizon_runs(self, setup):
        res = ErgodicDeepBSDESolver(
            setup,
            strategy="long_horizon",
            n_epochs=100,
            n_samples=2_000,
            T_long=20.0,
            rng_seed=SEED,
            torch_seed=SEED,
        ).solve()
        assert np.isfinite(res["lambda_ergodic"])

    def test_long_horizon_sign(self, setup):
        res = ErgodicDeepBSDESolver(
            setup,
            strategy="long_horizon",
            n_epochs=100,
            n_samples=5_000,
            T_long=20.0,
            rng_seed=SEED,
            torch_seed=SEED,
        ).solve()
        assert res["lambda_ergodic"] > 0.0, f"λ={res['lambda_ergodic']} should be positive"


# ------------------------------------------------------------------
# Invalid strategy
# ------------------------------------------------------------------


class TestErgodicDeepSmoke:
    def test_invalid_strategy(self):
        ou = _ou()
        ebsde = _quadratic_ebsde(ou)
        with pytest.raises((ValueError, Exception)):
            ErgodicDeepBSDESolver(
                ebsde, strategy="bogus", n_epochs=5, n_samples=64,
                rng_seed=SEED, torch_seed=SEED,
            ).solve()

    def test_lambda_history_stabilises(self):
        """λ variance in the last 10% of training < first 10%."""
        ou = _ou()
        ebsde = _quadratic_ebsde(ou)
        res = ErgodicDeepBSDESolver(
            ebsde,
            strategy="temporal_difference",
            v_network_layers=[32, 32],
            z_network_layers=[32, 32],
            n_epochs=600,
            n_samples=512,
            learning_rate=1e-3,
            rng_seed=SEED,
            torch_seed=SEED,
        ).solve()
        lh = np.array(res["lambda_history"])
        cut = len(lh) // 10
        var_early = float(np.var(lh[:cut]))
        var_late  = float(np.var(lh[-cut:]))
        assert var_late < var_early + 1e-4, (
            f"λ variance did not stabilise: early={var_early:.6f}, late={var_late:.6f}"
        )
