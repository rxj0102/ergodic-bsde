"""
Robust pricing under model uncertainty via ergodic BSDEs.

The agent faces model uncertainty about the drift of the state process.
The minimax problem leads to an ergodic BSDE with entropic driver:
    f(x, y, z) = h(x) + (1/(2η))|z|²

The ergodic constant λ = worst-case long-run yield.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Callable, Optional

from ebsde.forward.sde import ForwardSDE
from ebsde.bsde.ergodic import ErgodicBSDE
from ebsde.solvers.ergodic_pde import ErgodicPDESolver


class RobustPricer:
    """
    Pricing under model uncertainty via ergodic BSDEs.

    The agent is uncertain about the true drift of the state process.
    Under the worst-case model (minimax), the pricing problem becomes:

        inf_Q { E_Q[payoff] + η · KL(Q || P) }

    This leads to the ergodic BSDE with ENTROPIC driver:

        f(x, y, z) = h(x) + (1/(2η))|z|²

    The ergodic constant λ = worst-case long-run yield.

    Financial interpretation:
    - η → ∞ (no uncertainty): λ → E_π[h(X)]   (physical expectation)
    - η → 0  (max uncertainty): λ → sup_x h(x)  (worst-case catastrophe)
    - Intermediate η: interpolates between neutral and worst-case

    Parameters
    ----------
    forward            : ForwardSDE (must be ergodic)
    running_payoff     : h(x) callable — state-dependent payoff rate
    uncertainty_penalty: η ≥ 0 (larger → less uncertainty aversion)
    """

    def __init__(
        self,
        forward: ForwardSDE,
        running_payoff: Callable,
        uncertainty_penalty: float = 1.0,
    ) -> None:
        if not forward.is_ergodic():
            raise ValueError("forward SDE must be ergodic")
        if uncertainty_penalty < 0:
            raise ValueError("uncertainty_penalty η must be non-negative")

        self.forward = forward
        self.running_payoff = running_payoff
        self.eta = float(uncertainty_penalty)

        self._ebsde = self._build_ebsde(self.eta)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_ebsde(self, eta: float) -> ErgodicBSDE:
        h = self.running_payoff

        def driver(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> np.ndarray:
            x = np.asarray(x, dtype=float)
            z = np.asarray(z, dtype=float)
            h_val = np.asarray(h(x), dtype=float)
            z_clipped = np.clip(z, -10.0, 10.0)
            z_sq = np.sum(z_clipped**2, axis=-1) if z_clipped.ndim >= 2 else z_clipped ** 2
            if eta > 1e-14:
                return h_val + (1.0 / (2.0 * eta)) * z_sq
            else:
                # η = 0 limit: max uncertainty — just h(x) term in PDE, λ → sup h
                return h_val

        return ErgodicBSDE(forward=self.forward, driver=driver)

    def _physical_expectation(self) -> float:
        """E_π[h(X)] under the stationary distribution."""
        try:
            stat = self.forward.simulate_stationary(n_samples=20_000, rng=np.random.default_rng(0))
            x_stat = stat[:, 0] if stat.ndim == 2 else stat
            return float(np.mean(self.running_payoff(x_stat)))
        except Exception:
            stat = self.forward.stationary_distribution()
            return float(self.running_payoff(np.array([stat["mean"]]))[0])

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def price_robust(self, method: str = "pde") -> dict:
        """
        Compute the robust long-run price.

        Parameters
        ----------
        method : 'pde'

        Returns
        -------
        dict with:
            'lambda_robust'       : worst-case long-run yield
            'lambda_physical'     : E_π[h(X)] under physical measure
            'uncertainty_premium' : λ_robust - λ_physical
            'worst_case_drift'    : v(x) the adversarial drift perturbation ∝ ∂v/∂x
            'v_function'          : callable x → v(x)
        """
        if method != "pde":
            raise ValueError(f"Unknown method '{method}'. Use 'pde'.")

        sol = ErgodicPDESolver(self._ebsde).solve()
        lam_robust = sol["lambda_ergodic"]
        lam_physical = self._physical_expectation()

        x_grid = sol["x_grid"]
        v_grid = sol["v"]

        def v_function(x_in: np.ndarray) -> np.ndarray:
            return np.interp(np.asarray(x_in, dtype=float), x_grid, v_grid)

        # Worst-case drift perturbation: η^{-1} σ(x) ∂v/∂x
        dv = np.gradient(v_grid, x_grid)
        sigma_grid = np.asarray(
            [float(self.forward.diffusion(np.array([xi])).flat[0]) for xi in x_grid],
            dtype=float,
        )
        wc_drift = (1.0 / max(self.eta, 1e-14)) * sigma_grid * dv

        def worst_case_drift(x_in: np.ndarray) -> np.ndarray:
            return np.interp(np.asarray(x_in, dtype=float), x_grid, wc_drift)

        return {
            "lambda_robust": float(lam_robust),
            "lambda_physical": float(lam_physical),
            "uncertainty_premium": float(lam_robust - lam_physical),
            "worst_case_drift": worst_case_drift,
            "v_function": v_function,
        }

    def uncertainty_sensitivity(self, eta_values: np.ndarray) -> pd.DataFrame:
        """
        λ(η) as a function of the uncertainty penalty.

        Should be monotonically decreasing in η
        (more penalty → less conservative → smaller worst-case yield).

        Parameters
        ----------
        eta_values : 1-D array of η values (must be > 0)

        Returns
        -------
        DataFrame with columns ['eta', 'lambda_robust', 'lambda_physical']
        """
        eta_values = np.asarray(eta_values, dtype=float)
        if np.any(eta_values <= 0):
            raise ValueError("All eta_values must be strictly positive")

        lam_physical = self._physical_expectation()
        rows = []
        for eta in eta_values:
            pricer = RobustPricer(
                forward=self.forward,
                running_payoff=self.running_payoff,
                uncertainty_penalty=float(eta),
            )
            try:
                res = pricer.price_robust()
                lam = res["lambda_robust"]
            except Exception:
                lam = float("nan")
            rows.append({
                "eta": eta,
                "lambda_robust": lam,
                "lambda_physical": lam_physical,
            })

        return pd.DataFrame(rows)
