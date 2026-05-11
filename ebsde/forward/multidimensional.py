"""Multi-dimensional forward SDE processes."""

from __future__ import annotations

import numpy as np
from scipy import linalg
from typing import Optional

from ebsde.forward.sde import ForwardSDE


class MultiDimOU(ForwardSDE):
    """
    Multi-dimensional Ornstein-Uhlenbeck process:

        dX_t = -K(X_t - θ) dt + Σ dW_t

    where K is a d×d mean-reversion matrix (positive-definite eigenvalues),
    θ ∈ ℝ^d, and Σ is d×d diffusion matrix.

    Stationary distribution: N(θ, V) where V solves the Lyapunov equation:

        K V + V K^T = Σ Σ^T

    Parameters
    ----------
    K : (d, d) array_like
        Mean-reversion matrix.  All eigenvalues must have positive real parts.
    theta : (d,) array_like
        Long-run mean vector.
    Sigma : (d, d) array_like
        Diffusion matrix.
    """

    def __init__(
        self,
        K: np.ndarray,
        theta: np.ndarray,
        Sigma: np.ndarray,
    ) -> None:
        self.K = np.asarray(K, dtype=float)
        self.theta = np.asarray(theta, dtype=float)
        self.Sigma = np.asarray(Sigma, dtype=float)

        d = self.K.shape[0]
        if self.K.shape != (d, d):
            raise ValueError("K must be square")
        if self.theta.shape != (d,):
            raise ValueError("theta must have length d")
        if self.Sigma.shape != (d, d):
            raise ValueError("Sigma must be (d, d)")

        eigs = np.linalg.eigvals(self.K)
        if not np.all(eigs.real > 0):
            raise ValueError("All eigenvalues of K must have positive real parts")

        self._d = d

    # ------------------------------------------------------------------
    # ForwardSDE interface
    # ------------------------------------------------------------------

    @property
    def dimension(self) -> int:
        return self._d

    def drift(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        # x shape: (..., d)
        return -(x - self.theta) @ self.K.T  # -K(x - θ)

    def diffusion(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        # Constant diffusion: broadcast Σ to match leading dimensions of x
        out = np.broadcast_to(self.Sigma, x.shape[:-1] + (self._d, self._d))
        return out.copy()

    def is_ergodic(self) -> bool:
        return bool(np.all(np.linalg.eigvals(self.K).real > 0))

    # ------------------------------------------------------------------
    # Simulation (multi-dim override — diffusion is a matrix)
    # ------------------------------------------------------------------

    def simulate(
        self,
        x0: np.ndarray,
        T: float,
        n_steps: int,
        n_paths: int,
        scheme: str = "euler",
        rng: Optional[np.random.Generator] = None,
    ) -> dict:
        """
        Euler-Maruyama for multi-dimensional OU:
            X_{n+1} = X_n + b(X_n) Δt + Σ ΔW_n
        """
        if scheme not in ("euler",):
            raise ValueError("Only 'euler' scheme supported for MultiDimOU")

        if rng is None:
            rng = np.random.default_rng()

        d = self._d
        dt = T / n_steps
        sqrt_dt = np.sqrt(dt)
        times = np.linspace(0.0, T, n_steps + 1)

        x0 = np.asarray(x0, dtype=float).reshape(d)
        paths = np.zeros((n_paths, n_steps + 1, d))
        paths[:, 0, :] = x0

        # Brownian increments: (n_paths, n_steps, d)
        dW = rng.standard_normal((n_paths, n_steps, d)) * sqrt_dt

        for n in range(n_steps):
            x_n = paths[:, n, :]              # (n_paths, d)
            b = self.drift(x_n)               # (n_paths, d)
            noise = dW[:, n, :] @ self.Sigma.T  # (n_paths, d)
            paths[:, n + 1, :] = x_n + b * dt + noise

        return {
            "paths": paths,
            "brownian_increments": dW,
            "times": times,
        }

    # ------------------------------------------------------------------
    # Analytical stationary distribution
    # ------------------------------------------------------------------

    def stationary_covariance(self) -> np.ndarray:
        """
        Solve the continuous Lyapunov equation: KV + VK^T = ΣΣ^T.

        Returns
        -------
        V : (d, d) ndarray
            Stationary covariance matrix.
        """
        Q = self.Sigma @ self.Sigma.T
        return linalg.solve_continuous_lyapunov(self.K, Q)

    def stationary_distribution(self) -> dict:
        """
        Analytical stationary distribution N(θ, V).

        Returns
        -------
        dict with keys 'mean' and 'covariance'.
        """
        return {
            "mean": self.theta.copy(),
            "covariance": self.stationary_covariance(),
        }
