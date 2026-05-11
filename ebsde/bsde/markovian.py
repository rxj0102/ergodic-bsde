"""Markovian BSDE ↔ PDE connection utilities."""

from __future__ import annotations

import numpy as np
from typing import Optional


class MarkovianBSDEConnection:
    """
    The Markovian connection: BSDE ↔ PDE.

    For a Markovian forward-backward system:

        Forward:  dX_t = b(X_t) dt + σ(X_t) dW_t
        Backward: Y_t = g(X_T) + ∫_t^T f(X_s, Y_s, Z_s) ds - ∫_t^T Z_s dW_s

    By the nonlinear Feynman-Kac formula:

        Y_t = u(t, X_t)

    where u satisfies the quasilinear PDE:

        ∂u/∂t + (1/2)σ² ∂²u/∂x² + b ∂u/∂x + f(x, u, σ ∂u/∂x) = 0
        u(T, x) = g(x)

    and Z_t = σ(X_t) ∂u/∂x(t, X_t).

    Ergodic limit (T → ∞)
    ----------------------
    u(t, x) → v(x) - λt  where (λ, v) solves:

        (1/2)σ² v'' + b v' + f(x, v, σv') = λ
    """

    # ------------------------------------------------------------------
    # BSDE → PDE
    # ------------------------------------------------------------------

    @staticmethod
    def bsde_to_pde(bsde, x_grid: np.ndarray, t_grid: np.ndarray) -> dict:
        """
        Convert a StandardBSDE specification to PDE components on a grid.

        Returns
        -------
        dict
            'x_grid', 't_grid',
            'terminal_values' : g evaluated on x_grid,
            'pde_operator'    : callable (t, x, u, u_x, u_xx) → rhs,
        """
        x_grid = np.asarray(x_grid, dtype=float)
        t_grid = np.asarray(t_grid, dtype=float)

        terminal_values = bsde.eval_terminal(x_grid)

        def pde_operator(t, x, u, u_x, u_xx):
            x = np.asarray(x, dtype=float)
            sigma_x = bsde.forward.diffusion(x.reshape(-1, 1)).reshape(x.shape)
            b_x = bsde.forward.drift(x.reshape(-1, 1)).reshape(x.shape)
            z = sigma_x * u_x
            f_val = bsde.eval_driver(x, u, z[..., np.newaxis])
            return 0.5 * sigma_x**2 * u_xx + b_x * u_x + f_val

        return {
            "x_grid": x_grid,
            "t_grid": t_grid,
            "terminal_values": terminal_values,
            "pde_operator": pde_operator,
        }

    # ------------------------------------------------------------------
    # PDE solution → BSDE paths
    # ------------------------------------------------------------------

    @staticmethod
    def pde_solution_to_bsde(
        u_grid: np.ndarray,
        x_grid: np.ndarray,
        t_grid: np.ndarray,
        forward_paths: np.ndarray,
        sigma_func,
    ) -> dict:
        """
        Given PDE solution u(t, x) on a grid, recover (Y, Z) along forward paths.

            Y_t = u(t, X_t)
            Z_t = σ(X_t) · ∂u/∂x(t, X_t)

        Parameters
        ----------
        u_grid : (n_t, n_x) ndarray
            PDE solution.
        x_grid : (n_x,) ndarray
        t_grid : (n_t,) ndarray
        forward_paths : (n_paths, n_t, 1) ndarray
        sigma_func : callable
            σ(x) — diffusion function.

        Returns
        -------
        dict with 'Y_paths' (n_paths, n_t) and 'Z_paths' (n_paths, n_t).
        """
        n_paths, n_t, _ = forward_paths.shape
        Y_paths = np.zeros((n_paths, n_t))
        Z_paths = np.zeros((n_paths, n_t))

        for ti in range(n_t):
            x_t = forward_paths[:, ti, 0]
            # Interpolate u and ∂u/∂x at the path locations
            Y_paths[:, ti] = np.interp(x_t, x_grid, u_grid[ti])
            # Finite-difference derivative of u w.r.t. x
            du_dx = np.gradient(u_grid[ti], x_grid)
            Z_paths[:, ti] = sigma_func(x_t) * np.interp(x_t, x_grid, du_dx)

        return {"Y_paths": Y_paths, "Z_paths": Z_paths}

    # ------------------------------------------------------------------
    # Verification: Y_t ≈ u(t, X_t)
    # ------------------------------------------------------------------

    @staticmethod
    def verify_feynman_kac(
        bsde_Y: np.ndarray,
        pde_Y: np.ndarray,
    ) -> dict:
        """
        Verify that Y_t (from BSDE solver) ≈ u(t, X_t) (from PDE solver).

        Parameters
        ----------
        bsde_Y : (n_paths, n_t) array
        pde_Y  : (n_paths, n_t) array

        Returns
        -------
        dict with 'max_deviation' and 'mean_deviation'.
        """
        diff = np.abs(bsde_Y - pde_Y)
        return {
            "max_deviation": float(np.max(diff)),
            "mean_deviation": float(np.mean(diff)),
        }
