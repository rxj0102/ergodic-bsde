"""Constant Elasticity of Variance (CEV) forward process."""

from __future__ import annotations

import numpy as np
from typing import Optional

from ebsde.forward.sde import ForwardSDE


class CEVProcess(ForwardSDE):
    """
    Mean-reverting Constant Elasticity of Variance process:

        dX_t = κ(θ - X_t) dt + σ X_t^γ dW_t

    Ergodic for κ > 0 and γ ∈ [0.5, 1] (typically).  The stationary
    distribution is not Gaussian, providing a non-trivial test case.

    Parameters
    ----------
    kappa : float
        Mean-reversion speed (> 0).
    theta : float
        Long-run mean (> 0 so that the process stays positive).
    sigma : float
        Diffusion scale (> 0).
    gamma : float
        Elasticity exponent.  Common choices: 0.5 (CIR-like), 1.0 (lognormal).
    x_floor : float
        Reflection floor used during simulation to keep X positive.
    """

    def __init__(
        self,
        kappa: float = 1.0,
        theta: float = 1.0,
        sigma: float = 0.3,
        gamma: float = 0.5,
        x_floor: float = 1e-6,
    ) -> None:
        if kappa <= 0:
            raise ValueError("kappa must be positive for ergodicity")
        if theta <= 0:
            raise ValueError("theta must be positive to keep X > 0")
        if sigma <= 0:
            raise ValueError("sigma must be positive")
        if gamma < 0:
            raise ValueError("gamma must be non-negative")

        self.kappa = float(kappa)
        self.theta = float(theta)
        self.sigma = float(sigma)
        self.gamma = float(gamma)
        self.x_floor = float(x_floor)

    # ------------------------------------------------------------------
    # ForwardSDE interface
    # ------------------------------------------------------------------

    @property
    def dimension(self) -> int:
        return 1

    def drift(self, x: np.ndarray) -> np.ndarray:
        return self.kappa * (self.theta - x)

    def diffusion(self, x: np.ndarray) -> np.ndarray:
        x_safe = np.maximum(np.asarray(x, dtype=float), self.x_floor)
        return self.sigma * x_safe**self.gamma

    def is_ergodic(self) -> bool:
        return self.kappa > 0

    # ------------------------------------------------------------------
    # Override simulate to apply the positivity floor
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
        """Simulate with a positivity floor applied after each step."""
        result = super().simulate(x0, T, n_steps, n_paths, scheme=scheme, rng=rng)
        result["paths"] = np.maximum(result["paths"], self.x_floor)
        return result
