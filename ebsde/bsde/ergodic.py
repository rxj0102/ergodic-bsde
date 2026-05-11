"""Ergodic BSDE specification."""

from __future__ import annotations

import numpy as np
from typing import Callable, Optional

from ebsde.forward.sde import ForwardSDE


class ErgodicBSDE:
    """
    Ergodic BSDE specification:

        Y_t = Y_T + ∫_t^T [f(X_s, Y_s, Z_s) - λ] ds - ∫_t^T Z_s · dW_s

    for ALL T > t.

    Unknowns
    --------
    λ ∈ ℝ
        The ergodic constant (scalar).
    Y_t
        The value process (determined up to an additive constant).
    Z_t
        The control process.

    Markovian case
    --------------
    If Y_t = v(X_t), the pair (λ, v) satisfies the ERGODIC PDE:

        λ = (1/2)σ(x)² v''(x) + b(x) v'(x) + f(x, v(x), σ(x)v'(x))

    This is a NONLINEAR EIGENVALUE PROBLEM:

        L[v](x) + f(x, v(x), σ(x)v'(x)) = λ

    where L = (1/2)σ² ∂²/∂x² + b ∂/∂x is the generator of X.

    Existence and uniqueness (Fuhrman-Hu-Tessitore 2009)
    -----------------------------------------------------
    Under Lipschitz and monotonicity conditions on f:
    - λ is UNIQUE.
    - v is unique up to an additive constant; normalise by v(x_0) = 0.

    Connection to principal eigenvalue
    -----------------------------------
    For the linear driver f(x, y, z) = h(x) + c·y:

        λ = principal eigenvalue of (L + h + cI)
        v = corresponding eigenfunction
    """

    def __init__(
        self,
        forward: ForwardSDE,
        driver: Callable,
    ) -> None:
        if not forward.is_ergodic():
            raise ValueError(
                "Forward process must be ergodic for an ergodic BSDE. "
                "Check that the mean-reversion speed κ > 0."
            )
        self.forward = forward
        self.driver = driver

    # ------------------------------------------------------------------
    # Classification
    # ------------------------------------------------------------------

    @property
    def is_linear(self) -> bool:
        """
        Heuristic check for driver linearity in (y, z).

        A linear ergodic BSDE reduces to a principal eigenvalue problem.
        """
        try:
            x0 = np.zeros(self.forward.dimension)
            y0 = np.zeros(1)
            z0 = np.zeros(self.forward.dimension)
            f0 = self.driver(x0, y0, z0)
            # Check: f(x, 2y, 2z) ≈ 2 f(x, y, z)  (homogeneity test)
            f_double = self.driver(x0, 2 * y0, 2 * z0)
            return bool(np.allclose(f_double, 2 * f0, rtol=1e-6))
        except Exception:
            return False

    # ------------------------------------------------------------------
    # PDE operator evaluation
    # ------------------------------------------------------------------

    def markovian_pde_operator(
        self,
        x: np.ndarray,
        v: np.ndarray,
        dv: np.ndarray,
        d2v: np.ndarray,
    ) -> np.ndarray:
        """
        Evaluate the ergodic PDE operator:

            F[v](x) = (1/2)σ(x)² v''(x) + b(x) v'(x) + f(x, v(x), σ(x)v'(x))

        At the solution: F[v](x) = λ  for all x.

        Parameters
        ----------
        x : array (...,)
        v : array (...,)   — value
        dv : array (...,)  — first derivative v'
        d2v : array (...,) — second derivative v''
        """
        x = np.asarray(x, dtype=float)
        v = np.asarray(v, dtype=float)
        dv = np.asarray(dv, dtype=float)
        d2v = np.asarray(d2v, dtype=float)

        sigma_x = self.forward.diffusion(x.reshape(-1, 1)).reshape(x.shape)
        b_x = self.forward.drift(x.reshape(-1, 1)).reshape(x.shape)
        # Z = σ v'  (scalar case)
        z = sigma_x * dv
        return (
            0.5 * sigma_x**2 * d2v
            + b_x * dv
            + self.driver(x, v, z[..., np.newaxis])
        )

    # ------------------------------------------------------------------
    # Residual along simulated paths
    # ------------------------------------------------------------------

    def path_residual(
        self,
        x_paths: np.ndarray,
        y_paths: np.ndarray,
        z_paths: np.ndarray,
        lambda_hat: float,
        dt: float,
    ) -> dict:
        """
        Compute the ergodic BSDE residual along simulated paths.

        For the BSDE to hold, we need (roughly):

            ΔY_n  ≈  (f(X_n, Y_n, Z_n) - λ) Δt  - Z_n · ΔW_n

        Parameters
        ----------
        x_paths : (n_paths, n_steps, d)
        y_paths : (n_paths, n_steps)
        z_paths : (n_paths, n_steps, d)
        lambda_hat : float
            Estimated ergodic constant.
        dt : float

        Returns
        -------
        dict with 'mean_residual' and 'std_residual'.
        """
        n_paths, n_steps, d = x_paths.shape
        residuals = []
        for n in range(n_steps - 1):
            x_n = x_paths[:, n, :]
            y_n = y_paths[:, n]
            z_n = z_paths[:, n, :]
            y_next = y_paths[:, n + 1]
            f_n = self.driver(
                x_n.squeeze(-1) if d == 1 else x_n,
                y_n,
                z_n,
            )
            drift_term = (f_n - lambda_hat) * dt
            dy = y_next - y_n
            residuals.append(dy - drift_term)

        res = np.concatenate(residuals)
        return {
            "mean_residual": float(np.mean(np.abs(res))),
            "std_residual": float(np.std(res)),
        }

    def __repr__(self) -> str:
        return (
            f"ErgodicBSDE(forward={self.forward.__class__.__name__}, "
            f"driver={getattr(self.driver, '__name__', type(self.driver).__name__)})"
        )
