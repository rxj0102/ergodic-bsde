"""
Driver functions f(x, y, z) for BSDEs.

All drivers follow the signature:
    f(x, y, z) -> np.ndarray of shape broadcast(x, y, z[..., 0]).shape

where x, y are (...,) arrays and z is (..., d) array.
"""

from __future__ import annotations

import numpy as np


# ------------------------------------------------------------------
# Linear / quadratic drivers
# ------------------------------------------------------------------


def linear_driver(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    a: float = 0.0,
    b_coeff: float = 0.0,
    c: float = 0.0,
) -> np.ndarray:
    """
    Linear driver:

        f(x, y, z) = a + b·y + (c/2)|z|²

    c=0  → linear BSDE (explicit Feynman-Kac solution).
    c≠0  → quadratic BSDE (related to exponential utility / entropic risk).

    c > 0: risk-averse  (penalise large |z|)
    c < 0: risk-seeking

    The quadratic term arises in:
    - Entropic risk measure
    - Risk-sensitive control
    - Robust pricing under model uncertainty
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    z = np.asarray(z, dtype=float)
    z_sq = np.sum(z**2, axis=-1)
    return a + b_coeff * y + (c / 2.0) * z_sq


def quadratic_driver(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    h_func=None,
    risk_aversion: float = 1.0,
) -> np.ndarray:
    """
    Quadratic driver for risk-sensitive control:

        f(x, y, z) = h(x) - (γ/2)|z|²

    where h(x) is a running reward and γ is the risk-sensitivity parameter.

    The ergodic BSDE with this driver gives λ = long-run risk-adjusted growth rate.

    Parameters
    ----------
    h_func : callable, optional
        Running reward function h(x).  Defaults to h(x) = 0.
    risk_aversion : float
        γ > 0 (risk-averse), γ < 0 (risk-seeking).
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    z = np.asarray(z, dtype=float)
    h = h_func(x) if h_func is not None else np.zeros_like(x)
    return h - (risk_aversion / 2.0) * np.sum(z**2, axis=-1)


def entropy_driver(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    h_func=None,
    penalty: float = 1.0,
) -> np.ndarray:
    """
    Entropic penalty driver for robust pricing:

        f(x, y, z) = h(x) + (1/(2η))|z|²

    Arises from: inf_Q { E_Q[payoff] + η · KL(Q ‖ P) }
    i.e., worst-case pricing with entropy penalty on measure change.

    The ergodic constant λ = long-run yield under worst-case model.

    Parameters
    ----------
    h_func : callable, optional
        Running cost/reward h(x).
    penalty : float
        η > 0.  Larger η → less ambiguity aversion.
    """
    if penalty <= 0:
        raise ValueError("penalty η must be positive")
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    z = np.asarray(z, dtype=float)
    h = h_func(x) if h_func is not None else np.zeros_like(x)
    return h + (1.0 / (2.0 * penalty)) * np.sum(z**2, axis=-1)


def power_utility_driver(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    gamma: float = 2.0,
    r: float = 0.0,
    mu: float = 0.05,
    sigma_mkt: float = 0.2,
) -> np.ndarray:
    """
    Driver for power-utility portfolio optimisation (Merton problem).

    After optimising over the portfolio weight π:

        f*(x, y, z) = (μ - r)² / (2γ σ²)

    This is the classical Merton optimised driver.  The ergodic constant
    λ = (μ - r)² / (2γ σ²) + r  is the optimal long-run growth rate.

    Parameters
    ----------
    gamma : float
        Relative risk aversion (> 0).
    r : float
        Risk-free rate.
    mu : float
        Expected return on the risky asset.
    sigma_mkt : float
        Volatility of the risky asset.
    """
    if gamma <= 0:
        raise ValueError("gamma must be positive")
    if sigma_mkt <= 0:
        raise ValueError("sigma_mkt must be positive")
    x = np.asarray(x, dtype=float)
    sharpe_sq = ((mu - r) / sigma_mkt) ** 2
    return np.full_like(x, sharpe_sq / (2.0 * gamma))


# ------------------------------------------------------------------
# Convenience: make_quadratic_driver factory
# ------------------------------------------------------------------


def make_quadratic_driver(h_func, risk_aversion: float = 1.0):
    """Return a quadratic driver with fixed h and γ."""

    def _driver(x, y, z):
        return quadratic_driver(x, y, z, h_func=h_func, risk_aversion=risk_aversion)

    _driver.__doc__ = (
        f"Quadratic driver: f(x,y,z) = h(x) - ({risk_aversion}/2)|z|²"
    )
    return _driver


def make_entropy_driver(h_func, penalty: float = 1.0):
    """Return an entropy driver with fixed h and η."""

    def _driver(x, y, z):
        return entropy_driver(x, y, z, h_func=h_func, penalty=penalty)

    return _driver
