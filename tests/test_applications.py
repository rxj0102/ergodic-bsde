"""
Tests for ebsde/applications/.

Covers:
- LongRunRiskPricer: λ changes monotonically with risk aversion γ
- RobustPricer: λ(η) is monotonically decreasing, approaches E[h] as η → ∞
- RiskSensitiveOptimizer: growth rate ≥ 0 for positive Sharpe; hedging demand non-zero
- PrincipalEigenvalueSolver: matches ErgodicPDESolver for linear driver
- Donsker-Varadhan bounds contain the computed λ
- Hansen-Scheinkman: martingale component L_T has E[L_T] ≈ 1
"""

from __future__ import annotations

import numpy as np
import pytest

from ebsde.forward.ou_process import OrnsteinUhlenbeck
from ebsde.applications.long_run_risk import LongRunRiskPricer
from ebsde.applications.robust_pricing import RobustPricer
from ebsde.applications.risk_sensitive import RiskSensitiveOptimizer
from ebsde.applications.principal_eigenvalue import PrincipalEigenvalueSolver


# ------------------------------------------------------------------
# Shared fixture
# ------------------------------------------------------------------

@pytest.fixture
def ou():
    return OrnsteinUhlenbeck(kappa=1.0, theta=0.0, sigma=1.0)


# ══════════════════════════════════════════════════════════════════
# LongRunRiskPricer
# ══════════════════════════════════════════════════════════════════

class TestLongRunRiskPricer:

    def test_returns_finite_lambda(self, ou):
        pricer = LongRunRiskPricer(gamma=5.0, psi=1.5, delta=0.02,
                                   consumption_model=ou)
        res = pricer.compute_risk_adjusted_rate()
        assert np.isfinite(res["lambda"])

    def test_output_keys(self, ou):
        pricer = LongRunRiskPricer(gamma=5.0, psi=1.5, delta=0.02,
                                   consumption_model=ou)
        res = pricer.compute_risk_adjusted_rate()
        for key in ("lambda", "v_function", "risk_premium", "equity_premium"):
            assert key in res

    def test_v_function_callable(self, ou):
        pricer = LongRunRiskPricer(gamma=5.0, psi=1.5, delta=0.02,
                                   consumption_model=ou)
        res = pricer.compute_risk_adjusted_rate()
        x_test = np.linspace(-2, 2, 10)
        v_vals = res["v_function"](x_test)
        assert v_vals.shape == (10,)
        assert np.all(np.isfinite(v_vals))

    def test_lambda_increases_with_risk_aversion(self, ou):
        """Higher risk aversion → higher risk-adjusted rate."""
        gammas = [2.0, 5.0, 8.0]
        lambdas = []
        for g in gammas:
            pricer = LongRunRiskPricer(gamma=g, psi=1.5, delta=0.02,
                                       consumption_model=ou)
            res = pricer.compute_risk_adjusted_rate()
            lambdas.append(res["lambda"])
        assert lambdas[0] < lambdas[1] < lambdas[2], (
            f"λ not monotone in γ: {lambdas}"
        )

    def test_sensitivity_analysis_returns_dataframe(self, ou):
        pricer = LongRunRiskPricer(gamma=5.0, psi=1.5, delta=0.02,
                                   consumption_model=ou)
        df = pricer.sensitivity_analysis("gamma", np.array([2.0, 5.0, 8.0]))
        assert "gamma" in df.columns
        assert "lambda" in df.columns
        assert len(df) == 3
        assert np.all(np.isfinite(df["lambda"]))

    def test_term_structure_converges(self, ou):
        pricer = LongRunRiskPricer(gamma=5.0, psi=1.5, delta=0.02,
                                   consumption_model=ou)
        maturities = np.array([0.5, 1.0, 2.0])
        ts = pricer.term_structure_of_risk(maturities)
        assert "yields" in ts
        assert len(ts["yields"]) == 3
        assert np.isfinite(ts["lambda_limit"])

    def test_invalid_method_raises(self, ou):
        pricer = LongRunRiskPricer(gamma=5.0, psi=1.5, delta=0.02,
                                   consumption_model=ou)
        with pytest.raises((ValueError, Exception)):
            pricer.compute_risk_adjusted_rate(method="bogus")

    def test_non_ergodic_raises(self):
        from ebsde.forward.sde import ForwardSDE

        class NonErgodic(ForwardSDE):
            @property
            def dimension(self): return 1
            def drift(self, x): return np.zeros_like(x)
            def diffusion(self, x): return np.ones_like(x)
            def is_ergodic(self): return False

        with pytest.raises((ValueError, Exception)):
            LongRunRiskPricer(consumption_model=NonErgodic())


# ══════════════════════════════════════════════════════════════════
# RobustPricer
# ══════════════════════════════════════════════════════════════════

class TestRobustPricer:

    def _h(self, x):
        return np.asarray(x, dtype=float) ** 2

    def test_returns_finite_lambda(self, ou):
        pricer = RobustPricer(forward=ou, running_payoff=self._h,
                              uncertainty_penalty=1.0)
        res = pricer.price_robust()
        assert np.isfinite(res["lambda_robust"])

    def test_output_keys(self, ou):
        pricer = RobustPricer(forward=ou, running_payoff=self._h,
                              uncertainty_penalty=1.0)
        res = pricer.price_robust()
        for key in ("lambda_robust", "lambda_physical", "uncertainty_premium",
                    "worst_case_drift", "v_function"):
            assert key in res

    def test_uncertainty_premium_nonneg(self, ou):
        pricer = RobustPricer(forward=ou, running_payoff=self._h,
                              uncertainty_penalty=1.0)
        res = pricer.price_robust()
        assert res["uncertainty_premium"] >= -1e-6, (
            f"uncertainty_premium={res['uncertainty_premium']:.4f} < 0"
        )

    def test_lambda_monotone_decreasing_in_eta(self, ou):
        """λ(η) should decrease as η increases."""
        etas = np.array([0.5, 1.0, 2.0, 5.0])
        df = RobustPricer(
            forward=ou, running_payoff=self._h, uncertainty_penalty=1.0
        ).uncertainty_sensitivity(etas)
        lambdas = df["lambda_robust"].values
        diffs = np.diff(lambdas)
        assert np.all(diffs <= 1e-4), (
            f"λ(η) not monotone decreasing: {list(zip(etas, lambdas))}"
        )

    def test_lambda_approaches_physical_at_large_eta(self, ou):
        """λ(large η) closer to E_π[h] than λ(small η)."""
        res_lo = RobustPricer(forward=ou, running_payoff=self._h,
                              uncertainty_penalty=0.5).price_robust()
        res_hi = RobustPricer(forward=ou, running_payoff=self._h,
                              uncertainty_penalty=50.0).price_robust()
        lam_phys = res_hi["lambda_physical"]
        gap_lo = abs(res_lo["lambda_robust"] - lam_phys)
        gap_hi = abs(res_hi["lambda_robust"] - lam_phys)
        assert gap_hi < gap_lo, (
            f"Large η should approach physical: gap_lo={gap_lo:.4f}, gap_hi={gap_hi:.4f}"
        )

    def test_v_and_wc_drift_callable(self, ou):
        res = RobustPricer(forward=ou, running_payoff=self._h,
                           uncertainty_penalty=1.0).price_robust()
        x_test = np.linspace(-2, 2, 10)
        assert np.all(np.isfinite(res["v_function"](x_test)))
        assert np.all(np.isfinite(res["worst_case_drift"](x_test)))

    def test_negative_eta_raises(self, ou):
        with pytest.raises((ValueError, Exception)):
            RobustPricer(forward=ou, running_payoff=self._h,
                         uncertainty_penalty=-1.0)


# ══════════════════════════════════════════════════════════════════
# RiskSensitiveOptimizer
# ══════════════════════════════════════════════════════════════════

class TestRiskSensitiveOptimizer:

    def test_returns_finite_lambda(self, ou):
        opt = RiskSensitiveOptimizer(market_model=ou, risk_aversion=2.0)
        res = opt.compute_optimal_policy()
        assert np.isfinite(res["lambda"])

    def test_output_keys(self, ou):
        opt = RiskSensitiveOptimizer(market_model=ou, risk_aversion=2.0)
        res = opt.compute_optimal_policy()
        for key in ("lambda", "optimal_portfolio", "myopic_portfolio",
                    "hedging_demand", "v_function", "certainty_equivalent_rate"):
            assert key in res

    def test_optimal_growth_nonneg_positive_sharpe(self, ou):
        """With myopic term μ²/(2γ) ≥ 0, λ should be ≥ 0."""
        opt = RiskSensitiveOptimizer(
            market_model=ou, risk_aversion=2.0,
            mu_func=lambda x: np.asarray(x, dtype=float),
            sigma_func=lambda x: np.ones_like(np.asarray(x, dtype=float)),
        )
        res = opt.compute_optimal_policy()
        assert res["lambda"] >= -1e-6, f"λ={res['lambda']:.4f} should be ≥ 0"

    def test_hedging_demand_nonzero_stochastic_opp_set(self, ou):
        """Stochastic opportunity set → non-trivial hedging demand."""
        opt = RiskSensitiveOptimizer(
            market_model=ou, risk_aversion=2.0,
            mu_func=lambda x: np.asarray(x, dtype=float),
            sigma_func=lambda x: np.ones_like(np.asarray(x, dtype=float)),
        )
        res = opt.compute_optimal_policy()
        x_test = np.linspace(-2, 2, 20)
        hd = res["hedging_demand"](x_test)
        assert np.max(np.abs(hd)) > 1e-4, (
            f"Hedging demand max={np.max(np.abs(hd)):.6f} unexpectedly zero"
        )

    def test_portfolios_callable(self, ou):
        res = RiskSensitiveOptimizer(
            market_model=ou, risk_aversion=2.0
        ).compute_optimal_policy()
        x_test = np.linspace(-2, 2, 10)
        for fn_key in ("optimal_portfolio", "myopic_portfolio", "hedging_demand"):
            vals = res[fn_key](x_test)
            assert len(vals) == 10
            assert np.all(np.isfinite(vals))

    def test_higher_risk_aversion_lower_lambda(self, ou):
        """λ = E[μ²/(2γ)] should decrease as γ increases."""
        lambdas = []
        for g in [1.0, 2.0, 5.0]:
            opt = RiskSensitiveOptimizer(
                market_model=ou, risk_aversion=g,
                mu_func=lambda x: np.asarray(x, dtype=float),
                sigma_func=lambda x: np.ones_like(np.asarray(x, dtype=float)),
            )
            res = opt.compute_optimal_policy()
            lambdas.append(res["lambda"])
        assert lambdas[0] > lambdas[1] > lambdas[2] - 1e-4, (
            f"λ not decreasing with γ: {lambdas}"
        )

    def test_certainty_equivalent_is_annualized(self, ou):
        res = RiskSensitiveOptimizer(
            market_model=ou, risk_aversion=2.0
        ).compute_optimal_policy()
        assert np.isclose(res["certainty_equivalent_rate"], res["lambda"] * 100.0)


# ══════════════════════════════════════════════════════════════════
# PrincipalEigenvalueSolver
# ══════════════════════════════════════════════════════════════════

class TestPrincipalEigenvalueSolver:

    @pytest.fixture
    def solver(self, ou):
        return PrincipalEigenvalueSolver(forward=ou)

    @pytest.fixture
    def grid(self, ou):
        stat = ou.stationary_distribution()
        x_mean = stat["mean"]
        x_std = np.sqrt(stat["variance"])
        return np.linspace(x_mean - 4 * x_std, x_mean + 4 * x_std, 200)

    def test_principal_eigenvalue_nonneg_for_nonneg_h(self, solver, grid):
        h_func = lambda x: np.asarray(x, dtype=float) ** 2
        res = solver.compute_principal_eigenvalue((h_func, 0.0), grid)
        assert res["lambda"] >= -0.1

    def test_eigenfunction_positive(self, solver, grid):
        h_func = lambda x: np.asarray(x, dtype=float) ** 2
        res = solver.compute_principal_eigenvalue((h_func, 0.0), grid)
        phi = res["eigenfunction"]
        assert np.all(phi >= -1e-4), "Eigenfunction should be non-negative"
        assert np.max(phi) > 0.5, "Eigenfunction should be non-trivial"

    def test_matches_ergodic_pde_solver(self, ou, solver, grid):
        """Linear driver: FD eigenvalue ≈ ErgodicPDESolver λ."""
        h_func = lambda x: 0.5 * np.asarray(x, dtype=float) ** 2
        res = solver.verify_against_pde(h_func, c=0.0, grid=grid)
        assert res["absolute_error"] < 0.1, (
            f"eigenvalue={res['lambda_eigenvalue']:.4f}, "
            f"pde={res['lambda_pde']:.4f}, err={res['absolute_error']:.4f}"
        )

    def test_dv_bounds_contain_lambda(self, ou, solver):
        h_func = lambda x: np.asarray(x, dtype=float) ** 2
        bounds = solver.donsker_varadhan_bound(h_func)
        stat = ou.stationary_distribution()
        x_mean = stat["mean"]
        x_std = np.sqrt(stat["variance"])
        grid = np.linspace(x_mean - 4 * x_std, x_mean + 4 * x_std, 200)
        lam = solver.compute_principal_eigenvalue((h_func, 0.0), grid)["lambda"]
        assert bounds["lower_bound"] <= lam + 0.1, (
            f"lower_bound={bounds['lower_bound']:.4f} > λ={lam:.4f}"
        )
        assert lam <= bounds["upper_bound"] + 0.1, (
            f"λ={lam:.4f} > upper_bound={bounds['upper_bound']:.4f}"
        )

    def test_dv_gap_positive(self, ou, solver):
        h_func = lambda x: np.asarray(x, dtype=float) ** 2
        bounds = solver.donsker_varadhan_bound(h_func)
        assert bounds["gap"] > 0, "Gap should be positive for non-constant h"

    def test_hansen_scheinkman_martingale_approx_one(self, ou, solver, grid):
        """E[L_T] ≈ 1 for the martingale component of H-S decomposition."""
        h_func = lambda x: 0.5 * np.asarray(x, dtype=float) ** 2
        res = solver.hansen_scheinkman_decomposition({
            "h_func": h_func,
            "c": 0.0,
            "grid": grid,
            "T": 3.0,
            "n_paths": 3000,
            "x0": 0.0,
            "rng_seed": 42,
        })
        assert np.isfinite(res["lambda"])
        assert np.isfinite(res["L_T_mean"])
        assert 0.2 < res["L_T_mean"] < 5.0, (
            f"E[L_T]={res['L_T_mean']:.4f} unexpectedly far from 1"
        )

    def test_eigenvalue_increases_with_potential_scale(self, solver, grid):
        h1 = lambda x: np.asarray(x, dtype=float) ** 2
        h2 = lambda x: 2.0 * np.asarray(x, dtype=float) ** 2
        lam1 = solver.compute_principal_eigenvalue((h1, 0.0), grid)["lambda"]
        lam2 = solver.compute_principal_eigenvalue((h2, 0.0), grid)["lambda"]
        assert lam2 > lam1, f"λ(2h)={lam2:.4f} should exceed λ(h)={lam1:.4f}"

    def test_non_ergodic_raises(self):
        from ebsde.forward.sde import ForwardSDE

        class NonErgodic(ForwardSDE):
            @property
            def dimension(self): return 1
            def drift(self, x): return np.zeros_like(x)
            def diffusion(self, x): return np.ones_like(x)
            def is_ergodic(self): return False

        with pytest.raises((ValueError, Exception)):
            PrincipalEigenvalueSolver(forward=NonErgodic())
