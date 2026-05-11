"""
Tests for DeepBSDESolver (Han-Jentzen-E 2018).

Covers:
- Zero driver, g(x)=x²: |Y0 - exact| / exact < 0.10
- Linear driver f=b·y, g=0: absolute error < 0.02
- Quadratic driver vs Cole-Hopf: relative error < 0.15
- per_step and shared network types both run and return finite Y0
- Output shape / key invariants
- Training loss decreases over epochs
- Reproducibility with same seed
- d=1 and d=5 (MultiDimOU)
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from ebsde.forward.ou_process import OrnsteinUhlenbeck
from ebsde.forward.multidimensional import MultiDimOU
from ebsde.bsde.standard import StandardBSDE
from ebsde.solvers.deep_bsde import DeepBSDESolver, SubNet
from data.synthetic import quadratic_bsde_cole_hopf

SEED = 42


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


def _make_zero_driver_bsde(ou, T):
    return StandardBSDE(
        forward=ou,
        driver=lambda x, y, z: y * 0.0,
        terminal=lambda x: x**2,
        T=T,
    )


def _make_linear_bsde(ou, T, a, b):
    return StandardBSDE(
        forward=ou,
        driver=lambda x, y, z: a + b * y,
        terminal=lambda x: x * 0.0,
        T=T,
    )


# ------------------------------------------------------------------
# SubNet unit tests
# ------------------------------------------------------------------


class TestSubNet:
    def test_output_shape(self):
        net = SubNet(d_in=3, d_out=3, hidden_layers=[32, 32])
        x = torch.randn(16, 3)
        y = net(x)
        assert y.shape == (16, 3)

    def test_invalid_activation(self):
        with pytest.raises(ValueError):
            SubNet(d_in=1, d_out=1, hidden_layers=[8], activation="bogus")

    def test_xavier_init_reasonable(self):
        net = SubNet(d_in=5, d_out=5, hidden_layers=[32])
        for m in net.modules():
            if isinstance(m, torch.nn.Linear):
                assert m.weight.abs().mean() < 1.0
                assert torch.all(m.bias == 0.0)

    @pytest.mark.parametrize("act", ["relu", "tanh", "silu"])
    def test_activations(self, act):
        net = SubNet(1, 1, [16], activation=act)
        out = net(torch.randn(8, 1))
        assert torch.isfinite(out).all()


# ------------------------------------------------------------------
# Zero driver, g(x)=x²
# ------------------------------------------------------------------


class TestDeepBSDEZeroDriver:
    @pytest.fixture
    def setup(self):
        ou    = OrnsteinUhlenbeck(kappa=2.0, theta=0.0, sigma=1.0)
        T     = 0.5
        bsde  = _make_zero_driver_bsde(ou, T)
        exact = _ou_x2_exact(ou.kappa, ou.sigma, x0=0.0, T=T)
        return bsde, exact

    def test_y0_accuracy(self, setup):
        bsde, exact = setup
        res = DeepBSDESolver(
            bsde,
            n_steps=10,
            n_paths_train=2_048,
            hidden_layers=[32, 32],
            n_epochs=600,
            batch_size=256,
            lr=5e-3,
            rng_seed=SEED,
            torch_seed=SEED,
        ).solve()
        rel_err = abs(res["Y0"] - exact) / (exact + 1e-8)
        assert rel_err < 0.10, (
            f"Y0={res['Y0']:.5f}, exact={exact:.5f}, rel_err={rel_err:.3f}"
        )

    def test_output_keys(self, setup):
        bsde, _ = setup
        res = DeepBSDESolver(
            bsde, n_steps=5, n_paths_train=512, n_epochs=10,
            batch_size=64, rng_seed=SEED, torch_seed=SEED,
        ).solve()
        for key in ("Y0", "Z0", "loss_history", "model"):
            assert key in res
        assert isinstance(res["Y0"], float)
        assert res["Z0"].shape == (1,)
        assert len(res["loss_history"]) == 10

    def test_y0_positive(self, setup):
        bsde, _ = setup
        res = DeepBSDESolver(
            bsde, n_steps=5, n_paths_train=1_024, n_epochs=200,
            batch_size=128, rng_seed=SEED, torch_seed=SEED,
        ).solve()
        assert res["Y0"] > 0.0, f"Y0={res['Y0']} should be > 0"

    def test_loss_finite(self, setup):
        bsde, _ = setup
        res = DeepBSDESolver(
            bsde, n_steps=5, n_paths_train=512, n_epochs=50,
            batch_size=64, rng_seed=SEED, torch_seed=SEED,
        ).solve()
        assert all(np.isfinite(v) for v in res["loss_history"])

    def test_reproducible(self, setup):
        bsde, _ = setup
        kwargs = dict(
            n_steps=5, n_paths_train=512, n_epochs=20,
            batch_size=64, rng_seed=7, torch_seed=7,
        )
        y1 = DeepBSDESolver(bsde, **kwargs).solve()["Y0"]
        y2 = DeepBSDESolver(bsde, **kwargs).solve()["Y0"]
        assert np.isclose(y1, y2, atol=1e-5), f"y1={y1}, y2={y2}"


# ------------------------------------------------------------------
# Linear driver
# ------------------------------------------------------------------


class TestDeepBSDELinearDriver:
    @pytest.fixture
    def setup(self):
        ou    = OrnsteinUhlenbeck(kappa=1.5, theta=0.0, sigma=0.8)
        T     = 0.5
        a, b  = 0.0, -0.5
        bsde  = _make_linear_bsde(ou, T, a, b)
        exact = _linear_exact(a, b, T)
        return bsde, exact

    def test_y0_accuracy(self, setup):
        bsde, exact = setup
        res = DeepBSDESolver(
            bsde,
            n_steps=10,
            n_paths_train=2_048,
            hidden_layers=[32, 32],
            n_epochs=600,
            batch_size=256,
            lr=5e-3,
            rng_seed=SEED,
            torch_seed=SEED,
        ).solve()
        abs_err = abs(res["Y0"] - exact)
        assert abs_err < 0.02, (
            f"Y0={res['Y0']:.5f}, exact={exact:.5f}, abs_err={abs_err:.4f}"
        )


# ------------------------------------------------------------------
# Quadratic driver vs Cole-Hopf
# ------------------------------------------------------------------


class TestDeepBSDEQuadraticDriver:
    @pytest.fixture
    def setup(self):
        ou    = OrnsteinUhlenbeck(kappa=2.0, theta=0.0, sigma=1.0)
        T     = 0.5
        gamma = 1.0
        h     = lambda x: 0.5 * x**2

        def quad_driver(x, y, z):
            if isinstance(z, torch.Tensor):
                z_sq = (z * z).sum(dim=-1)
                hx = 0.5 * (x**2)
            else:
                z_sq = np.sum(z**2, axis=-1) if hasattr(z, 'ndim') and z.ndim > 1 else z**2
                hx = 0.5 * (x**2)
            return hx - (gamma / 2.0) * z_sq

        bsde  = StandardBSDE(
            forward=ou,
            driver=quad_driver,
            terminal=lambda x: x * 0.0,
            T=T,
        )
        ref = quadratic_bsde_cole_hopf(ou, h_func=h, gamma=gamma, T=T)
        return bsde, ref["Y0_at_theta"]

    def test_y0_within_tolerance(self, setup):
        bsde, ref = setup
        res = DeepBSDESolver(
            bsde,
            n_steps=10,
            n_paths_train=2_048,
            hidden_layers=[32, 32],
            n_epochs=800,
            batch_size=256,
            lr=5e-3,
            rng_seed=SEED,
            torch_seed=SEED,
        ).solve()
        rel_err = abs(res["Y0"] - ref) / (abs(ref) + 1e-8)
        assert rel_err < 0.15, (
            f"Y0={res['Y0']:.5f}, Cole-Hopf={ref:.5f}, rel_err={rel_err:.3f}"
        )


# ------------------------------------------------------------------
# Network types: per_step and shared
# ------------------------------------------------------------------


class TestDeepBSDENetworkTypes:
    @pytest.fixture
    def bsde(self):
        ou = OrnsteinUhlenbeck(kappa=2.0, theta=0.0, sigma=1.0)
        return _make_zero_driver_bsde(ou, T=0.3)

    @pytest.mark.parametrize("ntype", ["per_step", "shared"])
    def test_network_type_runs(self, bsde, ntype):
        res = DeepBSDESolver(
            bsde,
            n_steps=5,
            n_paths_train=512,
            n_epochs=20,
            batch_size=64,
            network_type=ntype,
            rng_seed=SEED,
            torch_seed=SEED,
        ).solve()
        assert np.isfinite(res["Y0"]), f"Y0 not finite for {ntype}"

    def test_invalid_network_type(self, bsde):
        with pytest.raises(ValueError):
            DeepBSDESolver(
                bsde,
                n_steps=5,
                network_type="bogus",
                rng_seed=SEED,
                torch_seed=SEED,
            ).solve()


# ------------------------------------------------------------------
# Loss convergence
# ------------------------------------------------------------------


class TestDeepBSDELossConvergence:
    def test_loss_decreases(self):
        """Mean loss in last 20% of epochs should be < first 20%."""
        ou   = OrnsteinUhlenbeck(kappa=2.0, theta=0.0, sigma=1.0)
        bsde = _make_zero_driver_bsde(ou, T=0.3)

        res = DeepBSDESolver(
            bsde,
            n_steps=5,
            n_paths_train=1_024,
            hidden_layers=[32, 32],
            n_epochs=300,
            batch_size=128,
            lr=5e-3,
            rng_seed=SEED,
            torch_seed=SEED,
        ).solve()

        lh = res["loss_history"]
        cut = len(lh) // 5
        loss_early = np.mean(lh[:cut])
        loss_late  = np.mean(lh[-cut:])
        assert loss_late < loss_early, (
            f"Loss did not decrease: early={loss_early:.4f}, late={loss_late:.4f}"
        )

    def test_more_epochs_lower_loss(self):
        """A solver trained for more epochs should end with a lower loss."""
        ou   = OrnsteinUhlenbeck(kappa=2.0, theta=0.0, sigma=1.0)
        bsde = _make_zero_driver_bsde(ou, T=0.3)

        kwargs_base = dict(
            n_steps=5, n_paths_train=1_024,
            hidden_layers=[32, 32], batch_size=128,
            lr=5e-3, rng_seed=SEED, torch_seed=SEED,
        )
        res_short = DeepBSDESolver(bsde, n_epochs=50,  **kwargs_base).solve()
        res_long  = DeepBSDESolver(bsde, n_epochs=300, **kwargs_base).solve()

        assert res_long["loss_history"][-1] < res_short["loss_history"][-1], (
            f"Longer training loss {res_long['loss_history'][-1]:.4f} >= "
            f"shorter {res_short['loss_history'][-1]:.4f}"
        )


# ------------------------------------------------------------------
# Multi-dimensional (d=5)
# ------------------------------------------------------------------


class TestDeepBSDEMultiDim:
    @pytest.fixture
    def setup_d5(self):
        d     = 5
        K     = 2.0 * np.eye(d)
        theta = np.zeros(d)
        Sigma = np.eye(d)
        ou5   = MultiDimOU(K=K, theta=theta, Sigma=Sigma)
        T     = 0.3
        bsde  = StandardBSDE(
            forward=ou5,
            driver=lambda x, y, z: y * 0.0,
            terminal=lambda x: (x**2).sum(-1),
            T=T,
        )
        # E[||X_T||²] for 5 independent OU(kappa=2, theta=0, sigma=1)
        exact = d * (1.0 / (2 * 2.0)) * (1 - np.exp(-2 * 2.0 * T))
        return bsde, exact, d

    def test_output_shapes_d5(self, setup_d5):
        bsde, exact, d = setup_d5
        res = DeepBSDESolver(
            bsde,
            n_steps=5,
            n_paths_train=512,
            hidden_layers=[32, 32],
            n_epochs=20,
            batch_size=64,
            rng_seed=SEED,
            torch_seed=SEED,
        ).solve()
        assert res["Z0"].shape == (d,), f"Z0 shape {res['Z0'].shape} != ({d},)"
        assert np.isfinite(res["Y0"])

    def test_y0_accuracy_d5(self, setup_d5):
        bsde, exact, d = setup_d5
        res = DeepBSDESolver(
            bsde,
            n_steps=8,
            n_paths_train=2_048,
            hidden_layers=[64, 64],
            n_epochs=800,
            batch_size=256,
            lr=5e-3,
            rng_seed=SEED,
            torch_seed=SEED,
        ).solve()
        rel_err = abs(res["Y0"] - exact) / (exact + 1e-8)
        assert rel_err < 0.15, (
            f"d=5: Y0={res['Y0']:.4f}, exact={exact:.4f}, rel_err={rel_err:.3f}"
        )


# ------------------------------------------------------------------
# evaluate() method
# ------------------------------------------------------------------


class TestDeepBSDEEvaluate:
    def test_evaluate_returns_dict(self):
        ou   = OrnsteinUhlenbeck(kappa=2.0, theta=0.0, sigma=1.0)
        bsde = _make_zero_driver_bsde(ou, T=0.3)
        solver = DeepBSDESolver(
            bsde, n_steps=5, n_paths_train=512, n_epochs=20,
            batch_size=64, rng_seed=SEED, torch_seed=SEED,
        )
        solver.solve()
        ev = solver.evaluate(n_eval_paths=200, rng_seed=0)
        for key in ("Y0_mean", "Y0_std", "Y_terminal"):
            assert key in ev
        assert np.isfinite(ev["Y0_mean"])
        assert ev["Y_terminal"].shape == (200,)
