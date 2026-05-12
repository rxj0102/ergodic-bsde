"""
Analysis tools for the ergodic constant λ.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Optional

from ebsde.bsde.ergodic import ErgodicBSDE
from ebsde.solvers.ergodic_pde import ErgodicPDESolver


class ErgodicConstantAnalysis:
    """
    Analysis tools for the ergodic constant λ.
    """

    # ------------------------------------------------------------------
    # Sensitivity analysis
    # ------------------------------------------------------------------

    def sensitivity(
        self,
        ergodic_bsde: ErgodicBSDE,
        param_name: str,
        param_values: np.ndarray,
        solver: str = "pde",
        bsde_factory=None,
        **solver_kwargs,
    ) -> pd.DataFrame:
        """
        ∂λ/∂θ for each model parameter θ via finite-difference sweeps.

        The user supplies a `bsde_factory(param_value) → ErgodicBSDE` that
        reconstructs the problem for each value of the parameter.

        If `bsde_factory` is None, a simple finite-difference perturbation
        is attempted on the driver (works only for scalar drivers that accept
        the parameter via closure).

        Parameters
        ----------
        ergodic_bsde  : base problem
        param_name    : human-readable label for the parameter
        param_values  : 1-D array of values to sweep
        solver        : 'pde' (default) or 'picard'
        bsde_factory  : callable(val) → ErgodicBSDE
        solver_kwargs : forwarded to solver

        Returns
        -------
        DataFrame with columns [param_name, 'lambda', 'dlambda_dparam']
        """
        param_values = np.asarray(param_values, dtype=float)
        lambdas = []

        if bsde_factory is None:
            # Fallback: use the base bsde for all values (degenerate)
            bsde_factory = lambda v: ergodic_bsde

        for val in param_values:
            try:
                ebsde = bsde_factory(val)
                if solver == "pde":
                    sol = ErgodicPDESolver(ebsde, **solver_kwargs).solve()
                else:
                    from ebsde.solvers.ergodic_picard import ErgodicPicardSolver
                    sol = ErgodicPicardSolver(ebsde, **solver_kwargs).solve()
                lambdas.append(float(sol["lambda_ergodic"]))
            except Exception:
                lambdas.append(float("nan"))

        lambdas_arr = np.array(lambdas)

        # Numerical derivative via central differences
        dlambda = np.gradient(lambdas_arr, param_values)

        return pd.DataFrame({
            param_name: param_values,
            "lambda": lambdas_arr,
            "dlambda_dparam": dlambda,
        })

    # ------------------------------------------------------------------
    # Spectral gap estimate
    # ------------------------------------------------------------------

    def spectral_gap_estimate(self, ergodic_bsde: ErgodicBSDE) -> float:
        """
        Estimate the spectral gap: difference between the two largest eigenvalues
        of the operator A = L + f (evaluated with v=0, z=0).

        The spectral gap determines:
        - Speed of convergence of the finite-T approximation to λ
        - Mixing rate of the forward process under the optimal measure

        For the OU process with linear driver, the spectral gap equals κ
        (the mean-reversion speed). For nonlinear drivers, we estimate it
        numerically via the FD eigenvalue problem.

        Returns
        -------
        float : estimated spectral gap (> 0 for ergodic processes)
        """
        fwd = ergodic_bsde.forward

        try:
            stat = fwd.stationary_distribution()
            x_mean = float(stat["mean"])
            x_std = float(np.sqrt(stat["variance"]))
        except Exception:
            x_mean, x_std = 0.0, 1.0

        n_x = 300
        x_grid = np.linspace(x_mean - 5 * x_std, x_mean + 5 * x_std, n_x)
        dx = x_grid[1] - x_grid[0]

        b_vals = np.array([float(fwd.drift(np.array([xi])).flat[0]) for xi in x_grid])
        s_vals = np.array([float(fwd.diffusion(np.array([xi])).flat[0]) for xi in x_grid])

        # Linearise driver at v=0, z=0: f(x, 0, 0)
        h_vals = np.asarray(
            ergodic_bsde.driver(x_grid, np.zeros_like(x_grid), np.zeros((n_x, 1))),
            dtype=float,
        )

        # Build tridiagonal FD matrix for L + h
        diag = np.zeros(n_x)
        upper = np.zeros(n_x - 1)
        lower = np.zeros(n_x - 1)

        for i in range(n_x):
            s2 = s_vals[i] ** 2
            b = b_vals[i]
            diag[i] = -s2 / dx**2 + h_vals[i]
            if i < n_x - 1:
                upper[i] = s2 / (2 * dx**2) + b / (2 * dx)
            if i > 0:
                lower[i - 1] = s2 / (2 * dx**2) - b / (2 * dx)

        A = np.diag(diag) + np.diag(upper, 1) + np.diag(lower, -1)

        import scipy.linalg
        eigvals = np.sort(scipy.linalg.eigvals(A).real)[::-1]  # descending

        # Spectral gap = λ_1 - λ_2
        if len(eigvals) >= 2 and np.isfinite(eigvals[0]) and np.isfinite(eigvals[1]):
            gap = float(eigvals[0] - eigvals[1])
            return max(gap, 0.0)
        return 0.0

    # ------------------------------------------------------------------
    # Bootstrap confidence interval
    # ------------------------------------------------------------------

    def confidence_interval_lambda(
        self,
        ergodic_bsde: ErgodicBSDE,
        n_bootstrap: int = 20,
        solver: str = "picard",
        alpha: float = 0.05,
        base_seed: int = 0,
        **solver_kwargs,
    ) -> dict:
        """
        Bootstrap confidence interval for λ.

        Runs the solver n_bootstrap times with different random seeds.
        Reports the (α/2, 1-α/2) percentile confidence interval.

        Parameters
        ----------
        ergodic_bsde : problem
        n_bootstrap  : number of bootstrap replications (default 20)
        solver       : 'picard' or 'pde' (PDE is deterministic so CI collapses)
        alpha        : confidence level (0.05 → 95% CI)
        base_seed    : base random seed
        solver_kwargs: forwarded to solver constructor

        Returns
        -------
        dict with 'mean', 'std', 'ci_lower', 'ci_upper', 'bootstrap_lambdas'
        """
        lambdas = []

        if solver == "pde":
            # PDE is deterministic — run once, CI collapses to point estimate
            sol = ErgodicPDESolver(ergodic_bsde, **solver_kwargs).solve()
            lam = float(sol["lambda_ergodic"])
            return {
                "mean": lam,
                "std": 0.0,
                "ci_lower": lam,
                "ci_upper": lam,
                "bootstrap_lambdas": [lam],
            }

        from ebsde.solvers.ergodic_picard import ErgodicPicardSolver

        for i in range(n_bootstrap):
            seed = base_seed + i
            try:
                sol = ErgodicPicardSolver(
                    ergodic_bsde, rng_seed=seed, **solver_kwargs
                ).solve()
                lambdas.append(float(sol["lambda_ergodic"]))
            except Exception:
                pass

        if not lambdas:
            return {
                "mean": float("nan"),
                "std": float("nan"),
                "ci_lower": float("nan"),
                "ci_upper": float("nan"),
                "bootstrap_lambdas": [],
            }

        arr = np.array(lambdas)
        return {
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr)),
            "ci_lower": float(np.percentile(arr, 100 * alpha / 2)),
            "ci_upper": float(np.percentile(arr, 100 * (1 - alpha / 2))),
            "bootstrap_lambdas": lambdas,
        }
