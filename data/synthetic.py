"""
Synthetic test problems with KNOWN analytical solutions.

These problems are essential for validating solvers.  All results are
expressed in closed form so that solver output can be compared numerically.
"""

from __future__ import annotations

import numpy as np
from scipy import linalg
from typing import Callable, Optional

from ebsde.forward.ou_process import OrnsteinUhlenbeck


# ------------------------------------------------------------------
# Finite-horizon linear BSDE
# ------------------------------------------------------------------


def linear_bsde_analytical(
    forward: OrnsteinUhlenbeck,
    a: float,
    b_coeff: float,
    T: float,
    terminal_func: Optional[Callable] = None,
) -> dict:
    """
    Analytical solution for a linear BSDE on OU:

        f(x, y, z) = a + b·y     (no z dependence)
        g(x) = terminal_func(x)  (default: g ≡ 0)

    Feynman-Kac solution:

        u(t, x) = E_x[e^{b(T-t)} g(X_T) + ∫_t^T e^{b(s-t)} a ds]
                = e^{b(T-t)} E_x[g(X_T)] + (a/b)(e^{b(T-t)} - 1)   [b≠0]

    For OU and g(x) = 0:
        u(t, x) = (a/b)(e^{b(T-t)} - 1)   if b ≠ 0
        u(t, x) = a(T-t)                   if b = 0

    Parameters
    ----------
    a, b_coeff : float
        Driver coefficients.
    T : float
        Time horizon.
    terminal_func : callable, optional
        g(x).  Defaults to g ≡ 0.

    Returns
    -------
    dict with 'u_func' callable (t, x) → u, 'Y0' at x=θ.
    """
    kappa = forward.kappa
    theta = forward.theta
    sigma = forward.sigma

    if terminal_func is None:
        terminal_func = lambda x: np.zeros_like(np.asarray(x, dtype=float))

    def u_func(t: float, x: float) -> float:
        tau = T - t
        x = np.asarray(x, dtype=float)
        # OU: E_x[X_T] = θ + (x - θ) e^{-κτ}
        # For g ≡ 0, terminal contribution vanishes
        term_exp = np.exp(b_coeff * tau) * terminal_func(
            theta + (x - theta) * np.exp(-kappa * tau)
        )
        if abs(b_coeff) < 1e-12:
            running = a * tau
        else:
            running = (a / b_coeff) * (np.exp(b_coeff * tau) - 1.0)
        return float(term_exp + running)

    Y0 = u_func(0.0, theta)
    return {"u_func": u_func, "Y0_at_theta": Y0, "T": T}


# ------------------------------------------------------------------
# Cole-Hopf transform for quadratic BSDE
# ------------------------------------------------------------------


def quadratic_bsde_cole_hopf(
    forward: OrnsteinUhlenbeck,
    h_func: Callable,
    gamma: float,
    T: float,
    n_quad: int = 1000,
) -> dict:
    """
    For the quadratic driver f(x, y, z) = h(x) - (γ/2)|z|²,
    the Cole-Hopf transform  v = -(2/γ) log(w)  reduces the
    nonlinear BSDE to a LINEAR PDE for w:

        ∂w/∂t + Lw - (γ/2) h(x) w = 0,   w(T) = exp(-(γ/2) g(X_T))

    For g ≡ 0: w(T) = 1  and the PDE is:

        ∂w/∂t + Lw - (γ/2) h(x) w = 0

    This can be solved via the Feynman-Kac representation:

        w(t, x) = E_x[exp(-γ/2 · ∫_t^T h(X_s) ds)]

    We evaluate this expectation by Monte-Carlo.

    Returns
    -------
    dict with 'w_func', 'v_func', 'Y0_at_theta'.
    """
    kappa = forward.kappa
    theta = forward.theta
    sigma = forward.sigma

    def w_mc(t: float, x: float, n_paths: int = 4096, n_steps: int = 200) -> float:
        """E_x[exp(-γ/2 ∫_t^T h(X_s) ds)] via Monte-Carlo."""
        rng = np.random.default_rng(42)
        dt_inner = (T - t) / n_steps
        paths = np.zeros(n_paths)
        paths[:] = float(x)
        integrals = np.zeros(n_paths)
        sqrt_dt = np.sqrt(dt_inner)
        for _ in range(n_steps):
            integrals += h_func(paths) * dt_inner
            paths += -kappa * (paths - theta) * dt_inner + sigma * rng.standard_normal(n_paths) * sqrt_dt
        return float(np.mean(np.exp(-gamma / 2.0 * integrals)))

    def v_func(t: float, x: float) -> float:
        w = w_mc(t, x)
        if w <= 0:
            return np.inf
        return -(2.0 / gamma) * np.log(w)

    Y0 = v_func(0.0, theta)
    return {"v_func": v_func, "Y0_at_theta": Y0, "T": T}


# ------------------------------------------------------------------
# Ergodic constant: OU + quadratic driver (QHO analogy)
# ------------------------------------------------------------------


def ergodic_ou_quadratic_analytical(
    kappa: float,
    sigma: float,
    alpha: float,
    beta: float,
    gamma: float,
) -> dict:
    """
    Analytical ergodic constant for OU + quadratic driver.

    Setting:
        Forward : dX = -κX dt + σ dW   (θ = 0)
        Driver  : f(x, y, z) = αx² + β - (γ/2)z²

    Via Cole-Hopf (v = -(2/γ) log w), the ergodic PDE reduces to a
    quantum harmonic oscillator (QHO) eigenvalue problem:

        (σ²/2) w'' - κx w' + (γ/2)(αx² + β) w = λ_w · w

    The principal eigenvalue of this QHO-type operator is:

        λ_w = (1/2)[ -κ + √(κ² + 2γα σ²) ]   (for α > 0, γ > 0)

    The ergodic constant of the BSDE is:

        λ = -(2/γ) λ_w + β

    Parameters
    ----------
    kappa, sigma : float
        OU parameters (θ=0 for simplicity).
    alpha, beta : float
        Quadratic running reward h(x) = α x² + β.
    gamma : float
        Risk-aversion parameter (> 0).

    Returns
    -------
    dict with 'lambda_ergodic', 'lambda_w', 'parameters'.
    """
    if alpha < 0:
        raise ValueError("alpha must be non-negative for a well-posed problem")
    if gamma <= 0:
        raise ValueError("gamma must be positive")

    discriminant = kappa**2 + 2.0 * gamma * alpha * sigma**2
    lambda_w = 0.5 * (-kappa + np.sqrt(discriminant))
    lambda_ergodic = -(2.0 / gamma) * lambda_w + beta

    return {
        "lambda_ergodic": float(lambda_ergodic),
        "lambda_w": float(lambda_w),
        "parameters": {
            "kappa": kappa,
            "sigma": sigma,
            "alpha": alpha,
            "beta": beta,
            "gamma": gamma,
        },
    }


# ------------------------------------------------------------------
# Ergodic linear BSDE — eigenvalue problem
# ------------------------------------------------------------------


def ergodic_linear_eigenvalue(
    kappa: float,
    sigma: float,
    alpha: float,
    c: float,
    n_grid: int = 500,
    x_range: float = 5.0,
) -> dict:
    """
    Ergodic BSDE with linear driver f(x, y, z) = αx² + c·y.

    The ergodic PDE:

        (σ²/2) v'' - κx v' + αx² + c·v = λ

    Rearranged: L v + c·v + αx² = λ, i.e., eigenvalue problem.

    For θ=0 OU and h(x) = αx², this is again a QHO problem.
    We solve numerically by finite differences on a truncated grid.

    Returns
    -------
    dict with 'lambda_ergodic', 'v_grid', 'x_grid'.
    """
    x = np.linspace(-x_range, x_range, n_grid)
    dx = x[1] - x[0]

    # Build generator matrix L (tridiagonal):
    # L f_i = (σ²/2)(f_{i+1} - 2f_i + f_{i-1})/dx²
    #           - κx_i (f_{i+1} - f_{i-1})/(2dx)
    diag_main = -sigma**2 / dx**2 + c * np.ones(n_grid)
    diag_upper = (sigma**2 / (2 * dx**2)) - kappa * x[:-1] / (2 * dx)
    diag_lower = (sigma**2 / (2 * dx**2)) + kappa * x[1:] / (2 * dx)

    # Add h(x) = α x² term to diagonal
    diag_main += alpha * x**2

    L = (
        np.diag(diag_main)
        + np.diag(diag_upper, k=1)
        + np.diag(diag_lower, k=-1)
    )

    # Principal (largest real part) eigenvalue
    eigenvalues, eigenvectors = linalg.eig(L)
    idx = np.argmax(eigenvalues.real)
    lambda_ergodic = float(eigenvalues[idx].real)
    v_grid = eigenvectors[:, idx].real

    # Normalise: v(0) = 0
    mid = n_grid // 2
    v_grid -= v_grid[mid]

    return {
        "lambda_ergodic": lambda_ergodic,
        "v_grid": v_grid,
        "x_grid": x,
    }
