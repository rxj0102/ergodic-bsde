"""
Long-run risk pricing via ergodic BSDEs.

Bansal-Yaron (2004) style: the ergodic BSDE gives the long-run
risk-adjusted discount rate λ under Epstein-Zin preferences.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Optional

from ebsde.forward.sde import ForwardSDE
from ebsde.forward.ou_process import OrnsteinUhlenbeck
from ebsde.bsde.ergodic import ErgodicBSDE
from ebsde.bsde.standard import StandardBSDE
from ebsde.solvers.ergodic_pde import ErgodicPDESolver
from ebsde.solvers.pde import PDEBSDESolver


class LongRunRiskPricer:
    """
    Long-run risk pricing via ergodic BSDEs.

    In the Bansal-Yaron (2004) long-run risk model the pricing kernel
    involves a risk adjustment that depends on long-horizon expectations.

    The ergodic BSDE provides the natural framework:
    - The state X_t = consumption growth component is ergodic (OU proxy)
    - The driver encodes recursive utility preferences (Epstein-Zin)
    - The ergodic constant λ = long-run risk-adjusted discount rate

    For Epstein-Zin with risk aversion γ and EIS ψ the continuation value
    satisfies the ergodic PDE:

        Lv + f(x, v, σ∇v) = λ

    with driver (log-linearised around the stationary mean):

        f(x, v, z) = δ·θ_ez·x  +  (γ/2)·|z|²

    where θ_ez = (1-γ)/(1-1/ψ) and x proxies log consumption growth
    deviation from its mean.

    Parameters
    ----------
    gamma  : risk aversion
    psi    : elasticity of intertemporal substitution
    delta  : subjective discount rate
    consumption_model : ForwardSDE for the consumption state (default OU)
    """

    def __init__(
        self,
        gamma: float = 10.0,
        psi: float = 1.5,
        delta: float = 0.02,
        consumption_model: Optional[ForwardSDE] = None,
    ) -> None:
        if gamma <= 0:
            raise ValueError("gamma must be positive")
        if psi <= 0:
            raise ValueError("psi must be positive")
        if psi == 1.0:
            raise ValueError("psi == 1 is the log-utility knife-edge case")

        self.gamma = float(gamma)
        self.psi = float(psi)
        self.delta = float(delta)

        # EZ aggregator coefficient
        self.theta_ez = (1.0 - gamma) / (1.0 - 1.0 / psi)

        if consumption_model is None:
            # Default: OU with unit mean-reversion, zero mean, unit vol
            consumption_model = OrnsteinUhlenbeck(kappa=1.0, theta=0.0, sigma=1.0)

        if not consumption_model.is_ergodic():
            raise ValueError("consumption_model must be ergodic")

        self.forward = consumption_model

        # Build ergodic BSDE with EZ driver
        self._ebsde = ErgodicBSDE(
            forward=self.forward,
            driver=self._ez_driver,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _ez_driver(self, x: np.ndarray, y: np.ndarray, z: np.ndarray) -> np.ndarray:
        """
        Log-linearised Epstein-Zin driver:
            f(x, v, z) = δ·θ_ez·x + (γ/2)|z|²

        z is clipped to avoid overflow for large γ.
        """
        x = np.asarray(x, dtype=float)
        z = np.asarray(z, dtype=float)
        z_clipped = np.clip(z, -5.0, 5.0)
        z_sq = np.sum(z_clipped**2, axis=-1)
        return self.delta * self.theta_ez * x + (self.gamma / 2.0) * z_sq

    def _build_pde_solver(self, **kw) -> ErgodicPDESolver:
        return ErgodicPDESolver(self._ebsde, **kw)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compute_risk_adjusted_rate(self, method: str = "pde") -> dict:
        """
        Compute the long-run risk-adjusted rate λ.

        λ determines long-run asset pricing:
            P_t/D_t → exp(-(λ - g)T)  as  T → ∞
        where g = long-run consumption growth rate.

        Parameters
        ----------
        method : 'pde'  (finite-difference PDE solver)

        Returns
        -------
        dict with keys:
            'lambda'           : ergodic constant
            'v_function'       : callable x → v(x)
            'risk_premium'     : λ - E_π[f(X)] under stationary measure
            'equity_premium'   : approximate equity risk premium (annualised %)
        """
        if method != "pde":
            raise ValueError(f"Unknown method '{method}'. Use 'pde'.")

        sol = self._build_pde_solver().solve()
        lam = sol["lambda_ergodic"]

        # Stationary mean of f under physical measure (z=0 proxy)
        stat = self.forward.stationary_distribution()
        x_mean = stat["mean"]
        x_var = stat["variance"]

        # E_π[f(X, 0, 0)] ≈ δ·θ_ez·E[X]  (z=0 gives linear part)
        f_mean_no_z = self.delta * self.theta_ez * x_mean
        risk_premium = float(lam - f_mean_no_z)

        # Rough equity premium: γ × σ_consumption × σ_SDF
        sigma_c = float(np.sqrt(x_var))
        equity_premium = self.gamma * sigma_c**2  # standard approximation

        # Wrap v_grid as callable via linear interpolation
        x_grid = sol["x_grid"]
        v_grid = sol["v"]

        def v_function(x_in: np.ndarray) -> np.ndarray:
            return np.interp(np.asarray(x_in, dtype=float), x_grid, v_grid)

        return {
            "lambda": lam,
            "v_function": v_function,
            "risk_premium": risk_premium,
            "equity_premium": equity_premium,
        }

    def term_structure_of_risk(self, maturities: np.ndarray) -> dict:
        """
        Compute the term structure of risk premia (yield curve).

        For finite maturity T: solve the standard BSDE on [0, T].
        The yield y(T) = -log(E[SDF_T]) / T (proxy via PDE value).

        As T → ∞: y(T) → λ (the ergodic constant).

        Parameters
        ----------
        maturities : 1-D array of T values

        Returns
        -------
        dict with 'maturities', 'yields', 'lambda_limit'
        """
        maturities = np.asarray(maturities, dtype=float)
        if np.any(maturities <= 0):
            raise ValueError("All maturities must be positive")

        # Ergodic (infinite-horizon) limit
        sol_inf = self._build_pde_solver().solve()
        lam = sol_inf["lambda_ergodic"]

        yields = []
        for T in maturities:
            # Finite-horizon BSDE with zero terminal condition
            # The PDE value at x=E[X] proxy for the "yield"
            bsde_T = StandardBSDE(
                forward=self.forward,
                driver=self._ez_driver,
                terminal=lambda x: np.zeros_like(np.asarray(x, dtype=float)),
                T=float(T),
            )
            try:
                pde_sol = PDEBSDESolver(bsde_T, n_x=200, n_t=max(50, int(T / 0.1))).solve(x0=0.0)
                # yield ≈ -Y_0 / T  (since Y_0 = E[∫f dt] approximately)
                y_T = float(-pde_sol["Y0"] / T) if T > 0 else 0.0
            except Exception:
                y_T = float(lam)
            yields.append(y_T)

        return {
            "maturities": maturities,
            "yields": np.array(yields),
            "lambda_limit": float(lam),
        }

    def sensitivity_analysis(
        self, param_name: str, param_values: np.ndarray
    ) -> pd.DataFrame:
        """
        Compute λ as a function of a model parameter.

        Parameters
        ----------
        param_name  : 'gamma', 'psi', 'delta', or 'sigma'
        param_values: 1-D array of values to sweep

        Returns
        -------
        DataFrame with columns [param_name, 'lambda']
        """
        param_values = np.asarray(param_values, dtype=float)
        rows = []
        for val in param_values:
            kwargs = dict(
                gamma=self.gamma,
                psi=self.psi,
                delta=self.delta,
                consumption_model=self.forward,
            )
            if param_name in ("gamma", "psi", "delta"):
                kwargs[param_name] = float(val)
                pricer = LongRunRiskPricer(**kwargs)
            elif param_name == "sigma":
                fwd = OrnsteinUhlenbeck(
                    kappa=getattr(self.forward, "kappa", 1.0),
                    theta=getattr(self.forward, "theta", 0.0),
                    sigma=float(val),
                )
                kwargs["consumption_model"] = fwd
                pricer = LongRunRiskPricer(**kwargs)
            else:
                raise ValueError(f"Unknown param_name '{param_name}'")

            try:
                res = pricer.compute_risk_adjusted_rate()
                lam = res["lambda"]
            except Exception:
                lam = float("nan")
            rows.append({param_name: val, "lambda": lam})

        return pd.DataFrame(rows)
