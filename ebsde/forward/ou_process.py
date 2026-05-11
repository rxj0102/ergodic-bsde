"""Ornstein-Uhlenbeck forward process."""

from __future__ import annotations

import numpy as np
from scipy import stats
from typing import Optional

from ebsde.forward.sde import ForwardSDE


class OrnsteinUhlenbeck(ForwardSDE):
    """
    Ornstein-Uhlenbeck process:

        dX_t = -κ(X_t - θ) dt + σ dW_t

    The canonical ergodic process.  Stationary distribution: N(θ, σ²/(2κ)).

    This is the primary test bed for ergodic BSDE solvers because:
    1. The forward process has a known stationary distribution.
    2. Many ergodic BSDE problems on OU have analytical or semi-analytical
       solutions.
    3. The Markovian PDE reduces to a second-order ODE in the stationary case.

    Parameters
    ----------
    kappa : float
        Mean-reversion speed (must be > 0 for ergodicity).
    theta : float
        Long-run mean.
    sigma : float
        Diffusion coefficient (> 0).
    """

    def __init__(
        self,
        kappa: float = 1.0,
        theta: float = 0.0,
        sigma: float = 1.0,
    ) -> None:
        if kappa <= 0:
            raise ValueError("kappa must be positive for ergodicity")
        if sigma <= 0:
            raise ValueError("sigma must be positive")
        self.kappa = float(kappa)
        self.theta = float(theta)
        self.sigma = float(sigma)

    # ------------------------------------------------------------------
    # ForwardSDE interface
    # ------------------------------------------------------------------

    @property
    def dimension(self) -> int:
        return 1

    def drift(self, x: np.ndarray) -> np.ndarray:
        return -self.kappa * (x - self.theta)

    def diffusion(self, x: np.ndarray) -> np.ndarray:
        return np.full_like(np.asarray(x, dtype=float), self.sigma)

    def is_ergodic(self) -> bool:
        return self.kappa > 0

    # ------------------------------------------------------------------
    # Analytical results
    # ------------------------------------------------------------------

    def stationary_distribution(self) -> dict:
        """
        Analytical stationary distribution: N(θ, σ²/(2κ)).

        Returns
        -------
        dict with keys 'mean' and 'variance'.
        """
        return {
            "mean": self.theta,
            "variance": self.sigma**2 / (2.0 * self.kappa),
        }

    def exact_simulate(
        self,
        x0: float,
        T: float,
        n_steps: int,
        n_paths: int,
        rng: Optional[np.random.Generator] = None,
    ) -> dict:
        """
        Exact simulation (no discretisation error).

        The conditional distribution is Gaussian:

            X_{t+Δt} | X_t  ~  N(θ + (X_t - θ) e^{-κΔt},
                                   (σ²/(2κ))(1 - e^{-2κΔt}))

        Returns
        -------
        dict with keys 'paths', 'times'.
            paths: (n_paths, n_steps+1, 1)
        """
        if rng is None:
            rng = np.random.default_rng()

        dt = T / n_steps
        times = np.linspace(0.0, T, n_steps + 1)
        e_kdt = np.exp(-self.kappa * dt)
        cond_var = (self.sigma**2 / (2.0 * self.kappa)) * (1.0 - np.exp(-2.0 * self.kappa * dt))
        cond_std = np.sqrt(cond_var)

        paths = np.zeros((n_paths, n_steps + 1, 1))
        paths[:, 0, 0] = float(x0)

        for n in range(n_steps):
            x_n = paths[:, n, 0]
            cond_mean = self.theta + (x_n - self.theta) * e_kdt
            paths[:, n + 1, 0] = (
                cond_mean + cond_std * rng.standard_normal(n_paths)
            )

        return {"paths": paths, "times": times}

    def transition_density(self, x: float, y: float, dt: float) -> float:
        """
        p(y, t+dt | x, t) evaluated at scalar x, y.

            p = N(y;  θ + (x-θ)e^{-κdt},  (σ²/(2κ))(1 - e^{-2κdt}))
        """
        mean = self.theta + (x - self.theta) * np.exp(-self.kappa * dt)
        var = (self.sigma**2 / (2.0 * self.kappa)) * (
            1.0 - np.exp(-2.0 * self.kappa * dt)
        )
        return float(stats.norm.pdf(y, loc=mean, scale=np.sqrt(var)))
