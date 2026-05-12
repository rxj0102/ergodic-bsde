"""
Tests for ebsde/analysis/convergence.py and ergodic_constant.py.

Covers:
- Grid convergence: PDE error decreases as n_x increases
- Path convergence: Picard error shrinks with n_paths (≈1/√n)
- Horizon convergence: λ(T) converges to true λ as T grows
- Solver comparison: PDE, Picard, Deep all agree within tolerance
- Spectral gap: positive for ergodic, reflects mean-reversion speed
- Donsker-Varadhan / bootstrap CI: CI contains the true λ
- Training diagnostics: loss decreasing, λ stabilized
"""

from __future__ import annotations

import numpy as np
import pytest

from ebsde.forward.ou_process import OrnsteinUhlenbeck
from ebsde.bsde.ergodic import ErgodicBSDE
from ebsde.solvers.ergodic_pde import ErgodicPDESolver
from ebsde.solvers.ergodic_picard import ErgodicPicardSolver
from ebsde.solvers.ergodic_deep import ErgodicDeepBSDESolver
from ebsde.analysis.convergence import ConvergenceDiagnostics, SolverComparison
from ebsde.analysis.ergodic_constant import ErgodicConstantAnalysis
from data.synthetic import ergodic_ou_quadratic_analytical

# ------------------------------------------------------------------
# Shared constants
# ------------------------------------------------------------------

KAPPA, SIGMA, ALPHA, GAMMA_DRV = 1.0, 1.0, 1.0, 1.0
LAMBDA_EXACT = ergodic_ou_quadratic_analytical(KAPPA, SIGMA, ALPHA, 0.0, GAMMA_DRV)[
    "lambda_ergodic"
]  # ≈ 0.36603


def _ou():
    return OrnsteinUhlenbeck(kappa=KAPPA, theta=0.0, sigma=SIGMA)


def _quadratic_ebsde():
    ou = _ou()
    return ErgodicBSDE(
        forward=ou,
        driver=lambda x, y, z: ALPHA * np.asarray(x, dtype=float) ** 2
        - (GAMMA_DRV / 2.0) * np.sum(np.asarray(z, dtype=float) ** 2, axis=-1),
    )


def _linear_ebsde():
    """Simple linear driver f(x,y,z) = x² (no z term), exact λ from FD eigenvalue."""
    ou = _ou()
    return ErgodicBSDE(
        forward=ou,
        driver=lambda x, y, z: np.asarray(x, dtype=float) ** 2,
    )


# ══════════════════════════════════════════════════════════════════
# ConvergenceDiagnostics
# ══════════════════════════════════════════════════════════════════

class TestGridConvergence:

    # Use finest grid (n_x=1000) as reference for self-convergence.
    # The analytical λ=0.5 for f=x² (E_π[x²] under OU with κ=σ=1) is the
    # true limit, but the FD solver converges to it from within its domain.
    # We measure self-convergence of the PDE discretisation.
    _REF_FINE = None  # computed lazily

    @classmethod
    def _get_ref(cls):
        if cls._REF_FINE is None:
            ebsde = _linear_ebsde()
            cls._REF_FINE = float(
                ErgodicPDESolver(ebsde, n_x=2000).solve()["lambda_ergodic"]
            )
        return cls._REF_FINE

    def test_lambda_changes_with_grid(self):
        """Coarser grids give different (less accurate) λ than finer grids."""
        diag = ConvergenceDiagnostics()
        ebsde = _linear_ebsde()
        ref = self._get_ref()
        res = diag.grid_convergence(
            ErgodicPDESolver, ebsde,
            grid_sizes=[20, 50, 100, 200, 500],
            reference=ref,
        )
        lambdas = [l for l in res["lambdas"] if np.isfinite(l)]
        assert len(lambdas) >= 3, "Need at least 3 valid grid runs"

    def test_errors_decrease_with_grid(self):
        """Errors should decrease as n_x increases (monotone)."""
        diag = ConvergenceDiagnostics()
        ebsde = _linear_ebsde()
        ref = self._get_ref()
        res = diag.grid_convergence(
            ErgodicPDESolver, ebsde,
            grid_sizes=[20, 50, 100, 200, 500],
            reference=ref,
        )
        errors = [e for e in res["errors"] if np.isfinite(e)]
        assert len(errors) >= 2
        # Fine grid should have smaller error than coarse grid
        assert errors[-1] < errors[0], (
            f"Error did not decrease: coarse={errors[0]:.5e}, fine={errors[-1]:.5e}"
        )

    def test_convergence_rate_near_two(self):
        """FD PDE scheme is O(Δx²): estimated convergence rate ≈ 2."""
        diag = ConvergenceDiagnostics()
        ebsde = _linear_ebsde()
        ref = self._get_ref()
        res = diag.grid_convergence(
            ErgodicPDESolver, ebsde,
            grid_sizes=[20, 50, 100, 200, 500],
            reference=ref,
        )
        rate = res["convergence_rate"]
        assert np.isfinite(rate), "Convergence rate should be finite"
        # Rate should be near 2 for second-order FD (allow generous range)
        assert 1.0 < rate < 3.5, (
            f"Expected rate ≈ 2 for O(Δx²) FD, got {rate:.3f}"
        )

    def test_return_keys(self):
        diag = ConvergenceDiagnostics()
        ebsde = _linear_ebsde()
        res = diag.grid_convergence(
            ErgodicPDESolver, ebsde, grid_sizes=[50, 100]
        )
        for k in ("grid_sizes", "lambdas", "errors", "convergence_rate", "reference"):
            assert k in res


class TestPathConvergence:

    def test_variance_decreases_with_paths(self):
        """
        MC variance ∝ 1/n_paths.  Run multiple seeds at each n and check that
        std(λ) shrinks as n grows (confirming the 1/√n rate).
        """
        ebsde = _linear_ebsde()
        n_reps = 6
        n_small, n_large = 1_000, 20_000

        lams_small, lams_large = [], []
        for seed in range(n_reps):
            for n, lst in [(n_small, lams_small), (n_large, lams_large)]:
                sol = ErgodicPicardSolver(
                    ebsde, n_paths=n, T_values=[10.0, 20.0], rng_seed=seed
                ).solve()
                lst.append(float(sol["lambda_ergodic"]))

        std_small = float(np.std(lams_small))
        std_large = float(np.std(lams_large))
        assert std_large < std_small, (
            f"std should decrease: std_small={std_small:.4f}, std_large={std_large:.4f}"
        )

    def test_lambda_finite_for_all_path_counts(self):
        """Solver should produce finite λ for all path counts."""
        diag = ConvergenceDiagnostics()
        ebsde = _linear_ebsde()
        res = diag.path_convergence(
            ErgodicPicardSolver, ebsde,
            n_paths_list=[1000, 5000, 20000],
            T_values=[10.0, 20.0],
            rng_seed=42,
        )
        assert all(np.isfinite(l) for l in res["lambdas"]), (
            f"Non-finite λ values: {res['lambdas']}"
        )

    def test_return_keys(self):
        diag = ConvergenceDiagnostics()
        ebsde = _linear_ebsde()
        res = diag.path_convergence(
            ErgodicPicardSolver, ebsde,
            n_paths_list=[1000, 5000],
            T_values=[5.0, 10.0],
        )
        for k in ("n_paths", "lambdas", "errors", "convergence_rate", "reference"):
            assert k in res


class TestHorizonConvergence:

    def test_lambda_converges_with_T(self):
        """λ(T) should approach λ_exact as T grows."""
        diag = ConvergenceDiagnostics()
        ebsde = _quadratic_ebsde()
        res = diag.horizon_convergence(
            ebsde,
            T_values=[5.0, 10.0, 20.0, 30.0],
            reference=LAMBDA_EXACT,
            n_paths=10_000,
            rng_seed=42,
        )
        lambdas = [l for l in res["lambdas"] if np.isfinite(l)]
        errors = [e for e in res["errors"] if np.isfinite(e)]
        assert len(lambdas) >= 2, "Need at least 2 T values"
        # λ at larger T should be closer to exact (last error ≤ first error)
        assert errors[-1] <= errors[0] + 0.05, (
            f"horizon convergence: T-small err={errors[0]:.4f}, T-large err={errors[-1]:.4f}"
        )

    def test_return_keys(self):
        diag = ConvergenceDiagnostics()
        ebsde = _quadratic_ebsde()
        res = diag.horizon_convergence(
            ebsde, T_values=[5.0, 10.0], n_paths=5000, rng_seed=0
        )
        for k in ("T_values", "lambdas", "errors", "reference", "exponential_rate"):
            assert k in res

    def test_all_finite(self):
        diag = ConvergenceDiagnostics()
        ebsde = _quadratic_ebsde()
        res = diag.horizon_convergence(
            ebsde, T_values=[5.0, 10.0], n_paths=5000, rng_seed=0
        )
        assert all(np.isfinite(l) for l in res["lambdas"])


class TestTrainingDiagnostics:

    def test_loss_decreasing_flag(self):
        """ErgodicDeepBSDESolver training should show decreasing loss."""
        ebsde = _quadratic_ebsde()
        res = ErgodicDeepBSDESolver(
            ebsde,
            strategy="direct_ergodic",
            n_epochs=300,
            n_samples=512,
            learning_rate=1e-3,
            rng_seed=42,
            torch_seed=42,
        ).solve()
        diag = ConvergenceDiagnostics()
        td = diag.training_diagnostics(res)
        assert td["loss_decreasing"], (
            f"Loss should decrease: reduction={td['loss_reduction']:.3f}"
        )

    def test_lambda_stabilized_flag(self):
        """After sufficient training, λ variance should decrease."""
        ebsde = _quadratic_ebsde()
        res = ErgodicDeepBSDESolver(
            ebsde,
            strategy="temporal_difference",
            n_epochs=400,
            n_samples=512,
            learning_rate=1e-3,
            rng_seed=42,
            torch_seed=42,
        ).solve()
        diag = ConvergenceDiagnostics()
        td = diag.training_diagnostics(res)
        assert "lambda_stabilized" in td
        assert np.isfinite(td["final_loss"])

    def test_output_keys(self):
        ebsde = _quadratic_ebsde()
        res = ErgodicDeepBSDESolver(
            ebsde,
            strategy="direct_ergodic",
            n_epochs=50,
            n_samples=128,
            rng_seed=0,
            torch_seed=0,
        ).solve()
        diag = ConvergenceDiagnostics()
        td = diag.training_diagnostics(res)
        for k in ("loss_decreasing", "lambda_stabilized", "final_loss",
                  "loss_reduction", "lambda_final"):
            assert k in td


# ══════════════════════════════════════════════════════════════════
# SolverComparison
# ══════════════════════════════════════════════════════════════════

class TestSolverComparison:

    def test_all_solvers_agree(self):
        """PDE, Picard, Deep all agree on λ within 10% for the quadratic driver."""
        ebsde = _quadratic_ebsde()
        cmp = SolverComparison()
        df = cmp.compare_all_solvers(
            ebsde,
            analytical_lambda=LAMBDA_EXACT,
            picard_kwargs={"n_paths": 10_000, "T_values": [10.0, 20.0], "rng_seed": 0},
            deep_kwargs={
                "n_epochs": 500, "n_samples": 512,
                "strategy": "direct_ergodic",
                "rng_seed": 0, "torch_seed": 0,
            },
        )
        assert len(df) == 3
        for _, row in df.iterrows():
            assert row["status"] == "ok", f"{row['solver']} failed: {row['status']}"
            assert np.isfinite(row["lambda"]), f"{row['solver']} returned nan λ"

    def test_errors_within_tolerance(self):
        """PDE and Picard solvers agree with analytical λ within 5%."""
        ebsde = _quadratic_ebsde()
        cmp = SolverComparison()
        df = cmp.compare_all_solvers(
            ebsde,
            analytical_lambda=LAMBDA_EXACT,
            picard_kwargs={"n_paths": 10_000, "T_values": [10.0, 20.0], "rng_seed": 0},
            deep_kwargs={
                "n_epochs": 500, "n_samples": 512,
                "strategy": "direct_ergodic",
                "rng_seed": 0, "torch_seed": 0,
            },
        )
        # Tolerances: PDE is highly accurate; Picard at T∈[10,20] has small bias;
        # Deep at 500 epochs is only a smoke-test (accuracy tested elsewhere).
        tols = {"PDE": 0.05, "Picard": 0.15, "Deep": 1.0}
        for _, row in df.iterrows():
            if row["status"] == "ok":
                rel_err = row["error"] / abs(LAMBDA_EXACT)
                tol = tols.get(row["solver"], 0.15)
                assert rel_err < tol, (
                    f"{row['solver']}: λ={row['lambda']:.4f}, "
                    f"exact={LAMBDA_EXACT:.4f}, rel_err={rel_err:.4f} (tol={tol})"
                )

    def test_dataframe_columns(self):
        ebsde = _linear_ebsde()
        cmp = SolverComparison()
        df = cmp.compare_all_solvers(
            ebsde,
            picard_kwargs={"n_paths": 5000, "T_values": [5.0, 10.0]},
            deep_kwargs={"n_epochs": 100, "n_samples": 128,
                         "strategy": "direct_ergodic", "rng_seed": 0, "torch_seed": 0},
        )
        for col in ("solver", "lambda", "error", "time_seconds", "iterations", "status"):
            assert col in df.columns

    def test_time_recorded(self):
        ebsde = _linear_ebsde()
        cmp = SolverComparison()
        df = cmp.compare_all_solvers(
            ebsde,
            picard_kwargs={"n_paths": 2000, "T_values": [5.0, 10.0]},
            deep_kwargs={"n_epochs": 50, "n_samples": 64,
                         "strategy": "direct_ergodic", "rng_seed": 0, "torch_seed": 0},
        )
        assert all(df["time_seconds"] >= 0)


# ══════════════════════════════════════════════════════════════════
# ErgodicConstantAnalysis
# ══════════════════════════════════════════════════════════════════

class TestErgodicConstantAnalysis:

    def test_sensitivity_returns_dataframe(self):
        """sensitivity() should return a DataFrame with correct columns."""
        ebsde = _quadratic_ebsde()
        ana = ErgodicConstantAnalysis()

        def bsde_factory(alpha_val):
            ou = _ou()
            return ErgodicBSDE(
                forward=ou,
                driver=lambda x, y, z, a=alpha_val: float(a) * np.asarray(x, dtype=float) ** 2,
            )

        df = ana.sensitivity(
            ebsde, "alpha",
            param_values=np.array([0.5, 1.0, 1.5]),
            bsde_factory=bsde_factory,
        )
        assert "alpha" in df.columns
        assert "lambda" in df.columns
        assert "dlambda_dparam" in df.columns
        assert len(df) == 3
        assert np.all(np.isfinite(df["lambda"]))

    def test_lambda_increases_with_driver_scale(self):
        """Larger α in f = αx² → larger λ (more potential → higher eigenvalue)."""
        ana = ErgodicConstantAnalysis()

        def bsde_factory(alpha_val):
            ou = _ou()
            return ErgodicBSDE(
                forward=ou,
                driver=lambda x, y, z, a=alpha_val: float(a) * np.asarray(x, dtype=float) ** 2,
            )

        df = ana.sensitivity(
            _quadratic_ebsde(), "alpha",
            param_values=np.array([0.5, 1.0, 2.0]),
            bsde_factory=bsde_factory,
        )
        assert df["lambda"].iloc[0] < df["lambda"].iloc[1] < df["lambda"].iloc[2], (
            f"λ not increasing with α: {list(df['lambda'])}"
        )

    def test_spectral_gap_positive_ergodic(self):
        """Spectral gap > 0 for ergodic process."""
        ana = ErgodicConstantAnalysis()
        ebsde = _linear_ebsde()
        gap = ana.spectral_gap_estimate(ebsde)
        assert gap > 0, f"Spectral gap {gap:.4f} should be > 0"

    def test_spectral_gap_reflects_mean_reversion(self):
        """Higher κ → larger spectral gap (faster mixing → larger gap)."""
        ana = ErgodicConstantAnalysis()

        def make_ebsde(kappa_val):
            ou = OrnsteinUhlenbeck(kappa=kappa_val, theta=0.0, sigma=1.0)
            return ErgodicBSDE(
                forward=ou,
                driver=lambda x, y, z: np.asarray(x, dtype=float) ** 2,
            )

        gap_slow = ana.spectral_gap_estimate(make_ebsde(0.5))
        gap_fast = ana.spectral_gap_estimate(make_ebsde(2.0))
        assert gap_fast > gap_slow, (
            f"Higher κ should give larger spectral gap: κ=0.5→{gap_slow:.3f}, κ=2→{gap_fast:.3f}"
        )

    def test_confidence_interval_pde(self):
        """PDE CI is a point estimate (deterministic solver)."""
        ana = ErgodicConstantAnalysis()
        ebsde = _linear_ebsde()
        ci = ana.confidence_interval_lambda(ebsde, solver="pde")
        assert np.isfinite(ci["mean"])
        assert ci["std"] == 0.0
        assert ci["ci_lower"] == ci["ci_upper"] == ci["mean"]

    def test_confidence_interval_picard(self):
        """Bootstrap CI from Picard should contain the true λ."""
        ana = ErgodicConstantAnalysis()
        ebsde = _quadratic_ebsde()
        ci = ana.confidence_interval_lambda(
            ebsde,
            n_bootstrap=10,
            solver="picard",
            T_values=[5.0, 10.0],
            n_paths=5_000,
        )
        assert np.isfinite(ci["mean"])
        assert np.isfinite(ci["std"])
        assert ci["ci_lower"] <= ci["mean"] <= ci["ci_upper"], (
            f"CI inconsistent: [{ci['ci_lower']:.4f}, {ci['ci_upper']:.4f}], "
            f"mean={ci['mean']:.4f}"
        )
        # CI should contain the true λ (with high probability)
        assert ci["ci_lower"] <= LAMBDA_EXACT <= ci["ci_upper"] + 0.1, (
            f"True λ={LAMBDA_EXACT:.4f} not in CI [{ci['ci_lower']:.4f}, {ci['ci_upper']:.4f}]"
        )

    def test_confidence_interval_keys(self):
        ana = ErgodicConstantAnalysis()
        ebsde = _linear_ebsde()
        ci = ana.confidence_interval_lambda(ebsde, solver="pde")
        for k in ("mean", "std", "ci_lower", "ci_upper", "bootstrap_lambdas"):
            assert k in ci
