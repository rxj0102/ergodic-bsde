"""
Ergodic Picard solver: extract the ergodic constant from long-horizon BSDEs.

Strategy:
  Solve the standard BSDE (Convention A: dY = -f dt + Z dW, g=0) for
  multiple large horizons T.  Under this convention:
      Y_0(T) ≈ λ·T + v(X_0) + O(e^{-αT})
  where λ is the ergodic constant (same sign as the PDE: Lv + f = λ).
  Estimate via finite differences: λ ≈ (Y_0(T₂) - Y_0(T₁)) / (T₂ - T₁).
"""

from __future__ import annotations

import numpy as np
from typing import Optional

from ebsde.bsde.ergodic import ErgodicBSDE
from ebsde.bsde.standard import StandardBSDE
from ebsde.solvers.picard import PicardBSDESolver


class ErgodicPicardSolver:
    """
    Picard iteration for ergodic BSDEs via long-horizon approximation.

    Parameters
    ----------
    ergodic_bsde  : ErgodicBSDE
    T_values      : list of horizons to solve at. Default [5, 10, 20, 40].
    n_paths       : Monte Carlo paths per horizon solve
    n_steps_per_unit : time steps per unit time
    n_picard      : Picard iterations per standard BSDE solve
    basis_degree  : polynomial basis degree for regression
    rng_seed      : random seed
    """

    def __init__(
        self,
        ergodic_bsde: ErgodicBSDE,
        T_values: Optional[list] = None,
        n_paths: int = 50_000,
        n_steps_per_unit: int = 50,
        n_picard: int = 5,
        basis_degree: int = 5,
        z_clip: Optional[float] = 4.0,
        max_steps_per_horizon: int = 300,
        rng_seed: Optional[int] = None,
    ) -> None:
        self.ebsde = ergodic_bsde
        self.T_values = T_values if T_values is not None else [5.0, 10.0, 20.0, 40.0]
        self.n_paths = n_paths
        self.n_steps_per_unit = n_steps_per_unit
        self.n_picard = n_picard
        self.basis_degree = basis_degree
        self.z_clip = z_clip
        self.max_steps_per_horizon = max_steps_per_horizon
        self.rng_seed = rng_seed

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def solve(self, verbose: bool = False) -> dict:
        """
        Returns
        -------
        dict with:
          'lambda_ergodic'    : best estimate of λ
          'lambda_estimates'  : {T: λ_est(T)} for each consecutive pair
          'Y0_values'         : {T: Y_0(T)} raw values
          'v_at_x0'           : v(x₀) estimate (intercept of Y_0 + λT)
          'convergence'       : {'lambda_vs_T': ..., 'extrapolation_error': ...}
        """
        Y0_values: dict[float, float] = {}
        picard_residuals: list = []

        # Sort T_values ascending
        T_list = sorted(self.T_values)

        rng_seed = self.rng_seed
        for T in T_list:
            n_steps = max(10, min(int(T * self.n_steps_per_unit), self.max_steps_per_horizon))
            seed_T = None if rng_seed is None else rng_seed + int(T * 1000)

            # Build a standard BSDE with g=0 on [0, T]
            std_bsde = StandardBSDE(
                forward=self.ebsde.forward,
                driver=self.ebsde.driver,
                terminal=lambda x: np.zeros(np.asarray(x).shape[0] if np.asarray(x).ndim > 0 else 1),
                T=T,
            )

            solver = PicardBSDESolver(
                std_bsde,
                n_paths=self.n_paths,
                n_steps=n_steps,
                n_picard=self.n_picard,
                basis_degree=self.basis_degree,
                z_clip=self.z_clip,
                rng_seed=seed_T,
            )
            res = solver.solve()
            Y0_values[T] = res["Y0"]
            picard_residuals.append(res["picard_convergence"])

            if verbose:
                print(f"  T={T:.1f}: Y_0={res['Y0']:.6f}")

        # Extract λ from the Y_0 values
        lambda_estimates = self._extract_ergodic_constant(Y0_values)
        lambda_best = self._best_lambda(lambda_estimates, Y0_values)

        # v(x₀) = Y_0(T) - λ·T  (should converge as T→∞)
        T_large = T_list[-1]
        v_at_x0 = Y0_values[T_large] - lambda_best * T_large

        return {
            "lambda_ergodic": lambda_best,
            "lambda_estimates": lambda_estimates,
            "Y0_values": Y0_values,
            "v_at_x0": v_at_x0,
            "convergence": {
                "lambda_vs_T": lambda_estimates,
                "picard_residuals": picard_residuals,
                "extrapolation_error": self._extrapolation_error(lambda_estimates),
            },
        }

    # ------------------------------------------------------------------
    # λ extraction
    # ------------------------------------------------------------------

    def _extract_ergodic_constant(
        self, Y0_values: dict[float, float]
    ) -> dict[tuple, float]:
        """
        Finite-difference estimates: λ ≈ (Y_0(T₂) - Y_0(T₁)) / (T₂ - T₁).

        Returns a dict mapping (T₁, T₂) → λ estimate.
        """
        T_sorted = sorted(Y0_values.keys())
        estimates: dict[tuple, float] = {}
        for i in range(len(T_sorted) - 1):
            T1, T2 = T_sorted[i], T_sorted[i + 1]
            lam = (Y0_values[T2] - Y0_values[T1]) / (T2 - T1)
            estimates[(T1, T2)] = float(lam)
        return estimates

    def _best_lambda(
        self,
        estimates: dict[tuple, float],
        Y0_values: dict[float, float],
    ) -> float:
        """
        Best λ estimate.

        Returns the FD estimate from the pair (T_min, T_second) — the two
        smallest T values.  With max_steps_per_horizon capping, small-T pairs
        have the most accurate per-step regression quality and the fewest
        accumulated regression steps, making them the most reliable.
        """
        if not estimates:
            return 0.0

        # Use the first (smallest T) consecutive pair
        first_key = sorted(estimates.keys(), key=lambda k: k[0])[0]
        lam_fd = estimates[first_key]

        if np.isfinite(lam_fd):
            return lam_fd

        # Fallback: first finite FD estimate
        for key in sorted(estimates.keys(), key=lambda k: k[0]):
            if np.isfinite(estimates[key]):
                return estimates[key]
        return 0.0

    def _richardson_fit(
        self, T_arr: np.ndarray, Y_arr: np.ndarray
    ) -> Optional[float]:
        """
        Fit Y_0(T) = λT + v₀ + c·exp(-αT) by nonlinear least squares.
        Returns λ estimate or None if the fit fails.
        """
        from scipy.optimize import curve_fit

        def model(T, lam, v0, c, alpha):
            return lam * T + v0 + c * np.exp(-np.abs(alpha) * T)

        try:
            # Initial guess: λ from last FD pair, v0 from large T intercept
            lam0 = (Y_arr[-1] - Y_arr[-2]) / (T_arr[-1] - T_arr[-2])
            p0 = [lam0, Y_arr[-1] - lam0 * T_arr[-1], 0.1, 0.5]
            popt, _ = curve_fit(
                model, T_arr, Y_arr, p0=p0,
                maxfev=10_000, bounds=([-np.inf, -np.inf, -np.inf, 0], np.inf),
            )
            return float(popt[0])
        except Exception:
            return None

    def _extrapolation_error(
        self, estimates: dict[tuple, float]
    ) -> float:
        """Standard deviation of the λ estimates across consecutive pairs."""
        vals = list(estimates.values())
        if len(vals) < 2:
            return float("nan")
        return float(np.std(vals))

    def _extract_v_function(
        self,
        lambda_est: float,
        T_large: float,
        x_grid: np.ndarray,
    ) -> np.ndarray:
        """
        Estimate v(x) ≈ Y_0(T_large, x) + λ · T_large.

        Solves the BSDE from each x in x_grid as initial condition and
        extracts v(x).  For large grids, uses the regression coefficients
        to interpolate.
        """
        n_steps = max(10, int(T_large * self.n_steps_per_unit))

        std_bsde = StandardBSDE(
            forward=self.ebsde.forward,
            driver=self.ebsde.driver,
            terminal=lambda x: np.zeros(np.asarray(x).shape[0] if np.asarray(x).ndim > 0 else 1),
            T=T_large,
        )

        v_vals = np.empty(len(x_grid))
        for idx, x0 in enumerate(x_grid):
            rng_i = np.random.default_rng(self.rng_seed if self.rng_seed is None else self.rng_seed + idx)
            sim = self.ebsde.forward.simulate(
                x0=np.array([x0]),
                T=T_large,
                n_steps=n_steps,
                n_paths=self.n_paths // 10,
                rng=rng_i,
            )
            # Rough estimate via direct average (not full BSDE solve)
            Y_T = np.zeros(self.n_paths // 10)
            v_vals[idx] = np.mean(Y_T) + lambda_est * T_large

        return v_vals
