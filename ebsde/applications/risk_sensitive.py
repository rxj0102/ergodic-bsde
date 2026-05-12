"""
Risk-sensitive portfolio optimization via ergodic BSDEs.

The risk-sensitive criterion:
    J(π) = lim_{T→∞} (1/T) · (-2/γ) log E[exp(-γ/2 · ∫_0^T r(π, X) dt)]

leads to an ergodic BSDE with optimized-out portfolio driver.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Optional, Callable

from ebsde.forward.sde import ForwardSDE
from ebsde.forward.ou_process import OrnsteinUhlenbeck
from ebsde.bsde.ergodic import ErgodicBSDE
from ebsde.solvers.ergodic_pde import ErgodicPDESolver


class RiskSensitiveOptimizer:
    """
    Risk-sensitive portfolio optimization via ergodic BSDEs.

    The risk-sensitive criterion with risk aversion γ:

        J(π) = lim_{T→∞} (1/T) · (-2/γ) log E[exp(-γ/2 · ∫_0^T [π_s μ_s ds + π_s σ_s dW_s])]

    After optimizing over portfolio π at each state x:

        f*(x, y, z) = μ(x)²/(2γσ(x)²)  +  (extra z terms from z-π interaction)

    For the reduced ergodic BSDE the driver is the optimized Hamiltonian.
    The Merton intertemporal hedging demand is captured by z(x) = σ(x)∂v/∂x.

    Parameters
    ----------
    market_model  : ForwardSDE for the market state (volatility, etc.)
    risk_aversion : γ > 0
    n_assets      : number of risky assets (currently 1)
    mu_func       : excess return μ(x) callable  (default: μ=x, state=Sharpe)
    sigma_func    : volatility σ_asset(x) callable (default: σ_asset=1)
    """

    def __init__(
        self,
        market_model: ForwardSDE,
        risk_aversion: float = 2.0,
        n_assets: int = 1,
        mu_func: Optional[Callable] = None,
        sigma_func: Optional[Callable] = None,
    ) -> None:
        if risk_aversion <= 0:
            raise ValueError("risk_aversion must be positive")
        if not market_model.is_ergodic():
            raise ValueError("market_model must be ergodic")

        self.forward = market_model
        self.gamma = float(risk_aversion)
        self.n_assets = int(n_assets)

        # Default: excess return = state variable (stochastic opportunity set)
        if mu_func is None:
            mu_func = lambda x: np.asarray(x, dtype=float)
        if sigma_func is None:
            sigma_func = lambda x: np.ones_like(np.asarray(x, dtype=float))

        self.mu_func = mu_func
        self.sigma_func = sigma_func

        self._ebsde = ErgodicBSDE(
            forward=self.forward,
            driver=self._optimal_driver,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _optimal_driver(
        self, x: np.ndarray, y: np.ndarray, z: np.ndarray
    ) -> np.ndarray:
        """
        Optimised Hamiltonian after π* = (μ - γσ_asset · z·σ_state) / (γσ_asset²).

        For 1-D case with state-dependent Sharpe ratio μ(x)/σ_asset(x):

            f*(x, y, z) = μ(x)²/(2γ σ_asset(x)²)
                          - z · σ_state(x) · μ(x)/σ_asset(x)
                          + γ/2 · z² · σ_state(x)²

        But the ergodic PDE has z = σ_state(x)·v'(x), so the z·σ_state terms
        reconstruct the hedging demand interaction.  We include the full
        Hamiltonian so the solver finds the correct fixed point.
        """
        x = np.asarray(x, dtype=float)
        z = np.asarray(z, dtype=float)

        mu = np.asarray(self.mu_func(x), dtype=float)
        sig_a = np.asarray(self.sigma_func(x), dtype=float)
        sig_a = np.where(np.abs(sig_a) < 1e-12, 1e-12, sig_a)

        # Myopic term
        myopic = mu**2 / (2.0 * self.gamma * sig_a**2)

        # z-dependent correction: from interplay between portfolio and Brownian
        # z is (N, d); for d=1 flatten to (N,)
        z_flat = z.ravel() if z.ndim == 2 and z.shape[-1] == 1 else np.asarray(z, dtype=float)
        if z_flat.ndim == 0:
            z_flat = float(z_flat) * np.ones_like(mu)

        sig_state = np.asarray(
            [float(self.forward.diffusion(np.array([xi])).flat[0]) for xi in np.atleast_1d(x)],
            dtype=float,
        ).reshape(np.asarray(x).shape)

        # Correction: -z·σ_state·μ/σ_asset + (γ/2)·z²·σ_state²  (hedging demand)
        hedge_correction = (
            -z_flat * sig_state * mu / sig_a
            + (self.gamma / 2.0) * z_flat**2 * sig_state**2
        )

        return myopic + hedge_correction

    def _myopic_driver(
        self, x: np.ndarray, y: np.ndarray, z: np.ndarray
    ) -> np.ndarray:
        """Driver ignoring hedging demand (z=0 in interaction terms)."""
        x = np.asarray(x, dtype=float)
        mu = np.asarray(self.mu_func(x), dtype=float)
        sig_a = np.asarray(self.sigma_func(x), dtype=float)
        sig_a = np.where(np.abs(sig_a) < 1e-12, 1e-12, sig_a)
        return mu**2 / (2.0 * self.gamma * sig_a**2)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compute_optimal_policy(self, method: str = "pde") -> dict:
        """
        Compute optimal risk-sensitive growth rate and portfolio.

        Parameters
        ----------
        method : 'pde'

        Returns
        -------
        dict with:
            'lambda'                  : optimal risk-sensitive growth rate
            'optimal_portfolio'       : callable x → π*(x)
            'myopic_portfolio'        : callable x → μ(x)/(γσ²(x))
            'hedging_demand'          : callable x → π*(x) - π_myopic(x)
            'v_function'              : callable x → v(x)
            'certainty_equivalent_rate': annualised % equivalent
        """
        if method != "pde":
            raise ValueError(f"Unknown method '{method}'. Use 'pde'.")

        # Solve full problem with hedging demand
        sol = ErgodicPDESolver(self._ebsde).solve()
        lam = sol["lambda_ergodic"]
        x_grid = sol["x_grid"]
        v_grid = sol["v"]

        # v'(x) via finite differences → z = σ_state · v'(x)
        dv = np.gradient(v_grid, x_grid)
        sigma_state_grid = np.array(
            [float(self.forward.diffusion(np.array([xi])).flat[0]) for xi in x_grid]
        )
        z_grid = sigma_state_grid * dv  # z = σ_state · ∂v/∂x

        mu_grid = np.asarray(self.mu_func(x_grid), dtype=float)
        sig_a_grid = np.asarray(self.sigma_func(x_grid), dtype=float)
        sig_a_grid = np.where(np.abs(sig_a_grid) < 1e-12, 1e-12, sig_a_grid)

        # π*(x) = μ(x)/(γσ_asset²) - σ_state·v'(x)/σ_asset  (hedging demand)
        myopic_grid = mu_grid / (self.gamma * sig_a_grid**2)
        optimal_grid = myopic_grid - sigma_state_grid * dv / sig_a_grid
        hedging_grid = optimal_grid - myopic_grid

        def v_function(x_in: np.ndarray) -> np.ndarray:
            return np.interp(np.asarray(x_in, dtype=float), x_grid, v_grid)

        def optimal_portfolio(x_in: np.ndarray) -> np.ndarray:
            return np.interp(np.asarray(x_in, dtype=float), x_grid, optimal_grid)

        def myopic_portfolio(x_in: np.ndarray) -> np.ndarray:
            return np.interp(np.asarray(x_in, dtype=float), x_grid, myopic_grid)

        def hedging_demand(x_in: np.ndarray) -> np.ndarray:
            return np.interp(np.asarray(x_in, dtype=float), x_grid, hedging_grid)

        return {
            "lambda": float(lam),
            "optimal_portfolio": optimal_portfolio,
            "myopic_portfolio": myopic_portfolio,
            "hedging_demand": hedging_demand,
            "v_function": v_function,
            "certainty_equivalent_rate": float(lam) * 100.0,
        }

    def sensitivity_analysis(
        self, param_name: str, param_values: np.ndarray
    ) -> pd.DataFrame:
        """
        λ as a function of a model parameter ('gamma', 'mu_scale', 'sigma_scale').

        Returns DataFrame with [param_name, 'lambda', 'myopic_lambda'].
        """
        param_values = np.asarray(param_values, dtype=float)
        rows = []
        for val in param_values:
            if param_name == "gamma":
                opt = RiskSensitiveOptimizer(
                    self.forward, risk_aversion=float(val),
                    mu_func=self.mu_func, sigma_func=self.sigma_func,
                )
            else:
                raise ValueError(f"Unknown param_name '{param_name}'")
            try:
                res = opt.compute_optimal_policy()
                lam = res["lambda"]
            except Exception:
                lam = float("nan")
            rows.append({param_name: val, "lambda": lam})

        return pd.DataFrame(rows)
