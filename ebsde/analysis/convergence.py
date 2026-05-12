"""
Convergence diagnostics and solver comparison tools.
"""

from __future__ import annotations

import time
import traceback
import numpy as np
import pandas as pd
from typing import Optional

from ebsde.bsde.ergodic import ErgodicBSDE
from ebsde.solvers.ergodic_pde import ErgodicPDESolver
from ebsde.solvers.ergodic_picard import ErgodicPicardSolver
from ebsde.solvers.ergodic_deep import ErgodicDeepBSDESolver


class ConvergenceDiagnostics:
    """
    Diagnostic tools for BSDE solver convergence.
    """

    # ------------------------------------------------------------------
    # Grid convergence
    # ------------------------------------------------------------------

    def grid_convergence(
        self,
        solver_class,
        bsde: ErgodicBSDE,
        grid_sizes: list,
        reference: Optional[float] = None,
        **solver_kwargs,
    ) -> dict:
        """
        Run an ErgodicPDESolver at multiple spatial grid resolutions.

        Estimates the convergence rate via log-log regression:
            error(n_x) ≈ C · n_x^{-p}

        For a second-order FD scheme p ≈ 2 (O(Δx²)).

        Parameters
        ----------
        solver_class : must be ErgodicPDESolver (the only grid-based solver)
        bsde         : ErgodicBSDE instance
        grid_sizes   : list of n_x values
        reference    : known λ (optional). If None, uses the finest grid as reference.
        solver_kwargs: additional kwargs forwarded to solver_class

        Returns
        -------
        dict with:
            'grid_sizes'       : list of n_x
            'lambdas'          : list of λ estimates
            'errors'           : absolute errors vs reference
            'convergence_rate' : estimated p from log-log fit
            'reference'        : λ_ref used
        """
        grid_sizes = sorted(grid_sizes)
        lambdas = []

        for n_x in grid_sizes:
            try:
                sol = solver_class(bsde, n_x=int(n_x), **solver_kwargs).solve()
                lambdas.append(float(sol["lambda_ergodic"]))
            except Exception:
                lambdas.append(float("nan"))

        # Use finest grid as reference if none given
        lam_ref = reference if reference is not None else lambdas[-1]
        errors = [abs(l - lam_ref) for l in lambdas]

        # Log-log fit to estimate convergence rate (skip NaNs and zeros)
        valid = [
            (n, e) for n, e, l in zip(grid_sizes, errors, lambdas)
            if np.isfinite(e) and e > 1e-14 and np.isfinite(l)
        ]
        rate = float("nan")
        if len(valid) >= 2 and reference is not None:
            ns, es = zip(*valid)
            coeffs = np.polyfit(np.log(ns), np.log(es), 1)
            rate = float(-coeffs[0])  # negative because error decreases with n_x

        return {
            "grid_sizes": grid_sizes,
            "lambdas": lambdas,
            "errors": errors,
            "convergence_rate": rate,
            "reference": lam_ref,
        }

    # ------------------------------------------------------------------
    # Path convergence (MC)
    # ------------------------------------------------------------------

    def path_convergence(
        self,
        solver_class,
        bsde: ErgodicBSDE,
        n_paths_list: list,
        reference: Optional[float] = None,
        **solver_kwargs,
    ) -> dict:
        """
        MC convergence: how does the λ estimate improve with n_paths?

        For Monte Carlo estimators the expected error is O(1/√n_paths).
        Fitting log(error) ∝ -0.5·log(n_paths) should recover rate ≈ 0.5.

        Parameters
        ----------
        solver_class   : ErgodicPicardSolver
        bsde           : ErgodicBSDE
        n_paths_list   : list of path counts to try
        reference      : known λ (uses largest n_paths result if None)
        solver_kwargs  : forwarded to solver_class

        Returns
        -------
        dict with 'n_paths', 'lambdas', 'errors', 'convergence_rate', 'reference'
        """
        n_paths_list = sorted(n_paths_list)
        lambdas = []

        for n in n_paths_list:
            try:
                sol = solver_class(bsde, n_paths=int(n), **solver_kwargs).solve()
                lambdas.append(float(sol["lambda_ergodic"]))
            except Exception:
                lambdas.append(float("nan"))

        lam_ref = reference if reference is not None else lambdas[-1]
        errors = [abs(l - lam_ref) for l in lambdas]

        valid = [
            (n, e) for n, e, l in zip(n_paths_list, errors, lambdas)
            if np.isfinite(e) and e > 1e-14 and np.isfinite(l)
        ]
        rate = float("nan")
        if len(valid) >= 2 and reference is not None:
            ns, es = zip(*valid)
            coeffs = np.polyfit(np.log(ns), np.log(es), 1)
            rate = float(-coeffs[0])

        return {
            "n_paths": n_paths_list,
            "lambdas": lambdas,
            "errors": errors,
            "convergence_rate": rate,
            "reference": lam_ref,
        }

    # ------------------------------------------------------------------
    # Horizon convergence (ergodic)
    # ------------------------------------------------------------------

    def horizon_convergence(
        self,
        ergodic_bsde: ErgodicBSDE,
        T_values: list,
        reference: Optional[float] = None,
        n_paths: int = 20_000,
        rng_seed: Optional[int] = None,
    ) -> dict:
        """
        For ergodic problems: how does the Picard λ(T) estimate depend on T?

        Theory: λ(T) → λ at an exponential rate related to the spectral gap:
            |λ(T) - λ| ≈ C · e^{-gap · T}

        Uses the ErgodicPicardSolver with pairs of consecutive T values.

        Parameters
        ----------
        ergodic_bsde : ErgodicBSDE
        T_values     : sorted list of horizon values
        reference    : true λ (if available)
        n_paths      : MC paths per horizon
        rng_seed     : random seed

        Returns
        -------
        dict with 'T_values', 'lambdas', 'errors', 'reference',
                  'exponential_rate' (fitted decay constant)
        """
        T_values = sorted(T_values)

        # Run one solve with all T values so estimates are comparable
        solver = ErgodicPicardSolver(
            ergodic_bsde,
            T_values=T_values,
            n_paths=n_paths,
            rng_seed=rng_seed,
        )
        sol = solver.solve()
        lambda_estimates = sol.get("lambda_estimates", {})

        # lambda_estimates is keyed by (T1, T2) pairs → pick the last pair for each T
        # Build a per-T series by using the estimate from each pair's larger T
        T_mid = []
        lambdas = []
        for (t1, t2), lam_est in sorted(lambda_estimates.items()):
            T_mid.append(float(t2))
            lambdas.append(float(lam_est))

        lam_ref = reference if reference is not None else lambdas[-1]
        errors = [abs(l - lam_ref) for l in lambdas]

        # Fit exponential decay: log(|error|) ≈ -gap·T + const
        exp_rate = float("nan")
        if reference is not None and len(T_mid) >= 2:
            valid = [(t, e) for t, e, l in zip(T_mid, errors, lambdas)
                     if np.isfinite(e) and e > 1e-10]
            if len(valid) >= 2:
                ts, es = zip(*valid)
                coeffs = np.polyfit(ts, np.log(es), 1)
                exp_rate = float(-coeffs[0])

        return {
            "T_values": T_mid,
            "lambdas": lambdas,
            "errors": errors,
            "reference": lam_ref,
            "exponential_rate": exp_rate,
        }

    # ------------------------------------------------------------------
    # Training diagnostics (deep BSDE)
    # ------------------------------------------------------------------

    def training_diagnostics(self, deep_solver_result: dict) -> dict:
        """
        Analyse the training curves from an ErgodicDeepBSDESolver result.

        Checks:
        - Is the loss decreasing? (last 20% < first 20%)
        - Has λ stabilized? (variance in last 10% < first 10%)
        - What's the terminal loss level?
        - Simple learning rate schedule visualization (placeholder)

        Parameters
        ----------
        deep_solver_result : dict returned by ErgodicDeepBSDESolver.solve()

        Returns
        -------
        dict with:
            'loss_decreasing'    : bool
            'lambda_stabilized'  : bool
            'final_loss'         : float
            'loss_reduction'     : ratio (early/late mean)
            'lambda_final'       : float
            'lambda_variance_early', 'lambda_variance_late'
        """
        loss = np.array(deep_solver_result.get("training_loss", []))
        lam_hist = np.array(deep_solver_result.get("lambda_history", []))

        result = {
            "loss_decreasing": False,
            "lambda_stabilized": False,
            "final_loss": float("nan"),
            "loss_reduction": float("nan"),
            "lambda_final": float(deep_solver_result.get("lambda_ergodic", float("nan"))),
            "lambda_variance_early": float("nan"),
            "lambda_variance_late": float("nan"),
        }

        if len(loss) >= 10:
            cut = max(1, len(loss) // 5)
            early_mean = float(np.mean(loss[:cut]))
            late_mean = float(np.mean(loss[-cut:]))
            result["loss_decreasing"] = late_mean < early_mean
            result["final_loss"] = float(loss[-1])
            result["loss_reduction"] = early_mean / max(late_mean, 1e-14)

        if len(lam_hist) >= 10:
            cut = max(1, len(lam_hist) // 10)
            var_early = float(np.var(lam_hist[:cut]))
            var_late = float(np.var(lam_hist[-cut:]))
            result["lambda_stabilized"] = var_late < var_early + 1e-6
            result["lambda_variance_early"] = var_early
            result["lambda_variance_late"] = var_late

        return result


class SolverComparison:
    """
    Systematic comparison of ergodic BSDE solver performance.
    """

    # ------------------------------------------------------------------
    # Compare all solvers on one problem
    # ------------------------------------------------------------------

    def compare_all_solvers(
        self,
        ergodic_bsde: ErgodicBSDE,
        analytical_lambda: Optional[float] = None,
        pde_kwargs: Optional[dict] = None,
        picard_kwargs: Optional[dict] = None,
        deep_kwargs: Optional[dict] = None,
    ) -> pd.DataFrame:
        """
        Run PDE, Picard, and Deep solvers on the same ergodic BSDE.

        Parameters
        ----------
        ergodic_bsde      : problem to solve
        analytical_lambda : known λ for error computation
        pde_kwargs        : kwargs for ErgodicPDESolver
        picard_kwargs     : kwargs for ErgodicPicardSolver
        deep_kwargs       : kwargs for ErgodicDeepBSDESolver

        Returns
        -------
        DataFrame with columns:
            solver, lambda, error, time_seconds, iterations, status
        """
        pde_kwargs = pde_kwargs or {}
        picard_kwargs = picard_kwargs or {}
        deep_kwargs = deep_kwargs or {
            "n_epochs": 500,
            "n_samples": 512,
            "strategy": "direct_ergodic",
        }

        rows = []

        for name, SolverCls, kwargs in [
            ("PDE", ErgodicPDESolver, pde_kwargs),
            ("Picard", ErgodicPicardSolver, picard_kwargs),
            ("Deep", ErgodicDeepBSDESolver, deep_kwargs),
        ]:
            t0 = time.perf_counter()
            try:
                sol = SolverCls(ergodic_bsde, **kwargs).solve()
                lam = float(sol["lambda_ergodic"])
                err = abs(lam - analytical_lambda) if analytical_lambda is not None else float("nan")
                # Iterations: convergence_history length for PDE, epochs for deep
                iters = (
                    len(sol.get("convergence_history", sol.get("training_loss", [])))
                )
                status = "ok"
            except Exception as exc:
                lam = float("nan")
                err = float("nan")
                iters = 0
                status = f"error: {exc!s:.60s}"

            elapsed = time.perf_counter() - t0
            rows.append({
                "solver": name,
                "lambda": lam,
                "error": err,
                "time_seconds": round(elapsed, 3),
                "iterations": iters,
                "status": status,
            })

        return pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # Scaling with dimension
    # ------------------------------------------------------------------

    def scaling_comparison(
        self,
        bsde_factory,
        dimensions: list = [1, 2, 5],
        **kwargs,
    ) -> pd.DataFrame:
        """
        How do solvers scale with problem dimension d?

        PDE: exponential in d (curse of dimensionality — only feasible for d=1)
        Picard/MC: polynomial in d but with high constant
        Deep: polynomial (the deep learning advantage)

        Parameters
        ----------
        bsde_factory : callable(d) → ErgodicBSDE for that dimension
        dimensions   : list of d values to test
        kwargs       : passed to compare_all_solvers

        Returns
        -------
        DataFrame with columns: d, solver, lambda, time_seconds, status
        """
        rows = []
        for d in dimensions:
            try:
                ebsde = bsde_factory(d)
            except Exception as e:
                for name in ("PDE", "Picard", "Deep"):
                    rows.append({"d": d, "solver": name, "lambda": float("nan"),
                                 "time_seconds": float("nan"), "status": f"factory error: {e}"})
                continue

            # Only run PDE for d=1
            solvers_to_run = []
            if d == 1:
                solvers_to_run.append(("PDE", ErgodicPDESolver, {}))
            solvers_to_run.append(("Picard", ErgodicPicardSolver,
                                   {"n_paths": 5000, "T_values": [5.0, 10.0]}))
            solvers_to_run.append(("Deep", ErgodicDeepBSDESolver,
                                   {"n_epochs": 200, "n_samples": 256, "strategy": "direct_ergodic"}))

            for name, SolverCls, skw in solvers_to_run:
                t0 = time.perf_counter()
                try:
                    sol = SolverCls(ebsde, **skw).solve()
                    lam = float(sol["lambda_ergodic"])
                    status = "ok"
                except Exception as exc:
                    lam = float("nan")
                    status = f"error: {exc!s:.60s}"
                elapsed = time.perf_counter() - t0
                rows.append({
                    "d": d,
                    "solver": name,
                    "lambda": lam,
                    "time_seconds": round(elapsed, 3),
                    "status": status,
                })

        return pd.DataFrame(rows)
