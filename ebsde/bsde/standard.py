"""Standard (finite-horizon) BSDE specification."""

from __future__ import annotations

import numpy as np
from typing import Callable

from ebsde.forward.sde import ForwardSDE


class StandardBSDE:
    """
    A finite-horizon BSDE specification:

        Y_t = g(X_T) + ∫_t^T f(s, X_s, Y_s, Z_s) ds - ∫_t^T Z_s · dW_s

    Components
    ----------
    forward : ForwardSDE
        Forward process dX = b dt + σ dW.
    driver : callable
        f(x, y, z) → scalar.  Time-independent Markovian driver.
    terminal : callable
        g(x) → scalar.  Terminal condition at time T.
    T : float
        Time horizon.

    Markovian representation
    ------------------------
    Y_t = u(t, X_t)  for a function u satisfying the quasilinear PDE:

        ∂u/∂t + (1/2)σ² ∂²u/∂x² + b ∂u/∂x + f(x, u, σ ∂u/∂x) = 0
        u(T, x) = g(x)

    Z_t = σ(X_t) ∂u/∂x(t, X_t)
    """

    def __init__(
        self,
        forward: ForwardSDE,
        driver: Callable,
        terminal: Callable,
        T: float,
    ) -> None:
        self.forward = forward
        self.driver = driver
        self.terminal = terminal
        self.T = float(T)

    @property
    def is_markovian(self) -> bool:
        """All our problems are Markovian by construction."""
        return True

    # ------------------------------------------------------------------
    # Evaluation helpers
    # ------------------------------------------------------------------

    def eval_driver(
        self,
        x: np.ndarray,
        y: np.ndarray,
        z: np.ndarray,
    ) -> np.ndarray:
        """Evaluate f(x, y, z)."""
        return np.asarray(self.driver(x, y, z), dtype=float)

    def eval_terminal(self, x: np.ndarray) -> np.ndarray:
        """Evaluate g(x)."""
        return np.asarray(self.terminal(x), dtype=float)

    # ------------------------------------------------------------------
    # Naive Monte-Carlo Y_0 estimate (Euler discretisation of the BSDE)
    # ------------------------------------------------------------------

    def monte_carlo_y0(
        self,
        x0: np.ndarray,
        n_steps: int,
        n_paths: int,
        scheme: str = "euler",
        rng: np.random.Generator = None,
    ) -> dict:
        """
        Estimate Y_0 via the simple Euler backward scheme:

            Y_T        = g(X_T)
            Y_{t_n}   ≈ Y_{t_{n+1}} + f(X_{t_n}, Y_{t_{n+1}}, Z_{t_n}) Δt

        where Z_{t_n} ≈ E[Y_{t_{n+1}} ΔW_n] / Δt  (Clark-Ocone / regression).

        This provides a crude but unbiased first estimate useful for testing.

        Returns
        -------
        dict with 'Y0_mean', 'Y0_std', 'paths'.
        """
        sim = self.forward.simulate(
            x0=x0, T=self.T, n_steps=n_steps, n_paths=n_paths,
            scheme=scheme, rng=rng,
        )
        paths = sim["paths"]      # (n_paths, n_steps+1, d)
        dW = sim["brownian_increments"]  # (n_paths, n_steps, d)
        dt = self.T / n_steps

        X_T = paths[:, -1, :]    # (n_paths, d)
        Y = self.eval_terminal(X_T.squeeze(-1) if X_T.shape[-1] == 1 else X_T)

        for n in reversed(range(n_steps)):
            X_n = paths[:, n, :]
            x_n = X_n.squeeze(-1) if X_n.shape[-1] == 1 else X_n
            dw_n = dW[:, n, :]
            # Rough Z estimate: (1/dt) E[Y * ΔW]  — use cross-section mean
            z_n = np.mean(Y[:, None] * dw_n, axis=0, keepdims=True).repeat(
                n_paths, axis=0
            ) / dt  # crude; proper regression done in solvers/
            z_n_sq = z_n  # shape (n_paths, d)
            f_val = self.eval_driver(x_n, Y, z_n_sq)
            Y = Y + f_val * dt

        return {
            "Y0_mean": float(np.mean(Y)),
            "Y0_std": float(np.std(Y) / np.sqrt(n_paths)),
            "Y0_paths": Y,
        }

    def __repr__(self) -> str:
        return (
            f"StandardBSDE(forward={self.forward.__class__.__name__}, "
            f"T={self.T})"
        )
