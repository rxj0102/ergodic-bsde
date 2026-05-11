"""Abstract base class for forward SDEs."""

from __future__ import annotations

import numpy as np
from abc import ABC, abstractmethod
from typing import Optional


class ForwardSDE(ABC):
    """
    Abstract base for forward stochastic differential equations:

        dX_t = b(X_t) dt + σ(X_t) dW_t

    The forward process X provides the state variable for Markovian BSDEs.
    For ergodic BSDEs, X must be ERGODIC — i.e., it admits a unique
    stationary distribution μ such that the law of X_t → μ as t → ∞.

    This is essential: the ergodic BSDE only makes sense when the
    forward process forgets its initial condition.
    """

    @abstractmethod
    def drift(self, x: np.ndarray) -> np.ndarray:
        """b(x) — drift coefficient."""

    @abstractmethod
    def diffusion(self, x: np.ndarray) -> np.ndarray:
        """σ(x) — diffusion coefficient."""

    @abstractmethod
    def is_ergodic(self) -> bool:
        """Whether the process is ergodic."""

    @property
    @abstractmethod
    def dimension(self) -> int:
        """State space dimension d."""

    # ------------------------------------------------------------------
    # Simulation
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
        Simulate paths of the forward SDE.

        Schemes
        -------
        euler
            Euler-Maruyama:
            X_{n+1} = X_n + b(X_n)Δt + σ(X_n)ΔW_n
        milstein
            Milstein (1-D only):
            X_{n+1} = X_n + b(X_n)Δt + σ(X_n)ΔW_n
                      + (1/2)σ(X_n)σ'(X_n)(ΔW_n² - Δt)

        Returns
        -------
        dict
            paths: (n_paths, n_steps+1, d)
            brownian_increments: (n_paths, n_steps, d)
            times: (n_steps+1,)
        """
        if rng is None:
            rng = np.random.default_rng()

        x0 = np.asarray(x0, dtype=float)
        d = self.dimension
        dt = T / n_steps
        sqrt_dt = np.sqrt(dt)
        times = np.linspace(0.0, T, n_steps + 1)

        paths = np.zeros((n_paths, n_steps + 1, d))
        # broadcast x0 across paths
        paths[:, 0, :] = np.broadcast_to(x0.reshape(-1), (n_paths, d))

        dW = rng.standard_normal((n_paths, n_steps, d)) * sqrt_dt

        if scheme == "euler":
            for n in range(n_steps):
                x_n = paths[:, n, :]                     # (n_paths, d)
                b = self.drift(x_n)                       # (n_paths, d)
                sig = self.diffusion(x_n)                 # (n_paths, d)
                paths[:, n + 1, :] = x_n + b * dt + sig * dW[:, n, :]

        elif scheme == "milstein":
            if d != 1:
                raise ValueError("Milstein scheme is only implemented for d=1")
            eps = 1e-5
            for n in range(n_steps):
                x_n = paths[:, n, :]
                b = self.drift(x_n)
                sig = self.diffusion(x_n)
                # σ'(x) by finite difference
                sig_prime = (
                    self.diffusion(x_n + eps) - self.diffusion(x_n - eps)
                ) / (2 * eps)
                paths[:, n + 1, :] = (
                    x_n
                    + b * dt
                    + sig * dW[:, n, :]
                    + 0.5 * sig * sig_prime * (dW[:, n, :] ** 2 - dt)
                )
        else:
            raise ValueError(f"Unknown scheme '{scheme}'. Use 'euler' or 'milstein'.")

        return {
            "paths": paths,
            "brownian_increments": dW,
            "times": times,
        }

    def simulate_stationary(
        self,
        n_samples: int,
        burn_in: int = 10_000,
        thin: int = 100,
        dt: float = 0.01,
        rng: Optional[np.random.Generator] = None,
    ) -> np.ndarray:
        """
        Sample from the stationary distribution by running a long path
        and subsampling after burn-in.

        Returns
        -------
        np.ndarray of shape (n_samples, d)
            Approximate i.i.d. samples from the stationary distribution μ.
        """
        if rng is None:
            rng = np.random.default_rng()

        d = self.dimension
        total_steps = burn_in + n_samples * thin
        x = np.zeros((1, d))
        sqrt_dt = np.sqrt(dt)

        samples = []
        for i in range(total_steps):
            dw = rng.standard_normal((1, d)) * sqrt_dt
            x = x + self.drift(x) * dt + self.diffusion(x) * dw
            if i >= burn_in and (i - burn_in) % thin == 0:
                samples.append(x.copy())

        return np.vstack(samples)  # (n_samples, d)
