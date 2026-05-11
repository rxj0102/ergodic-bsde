"""Shared utility functions."""

from __future__ import annotations

import numpy as np


def finite_difference_grad(f, x: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    """Compute ∇f(x) by central finite differences."""
    x = np.asarray(x, dtype=float)
    grad = np.zeros_like(x)
    for i in range(x.size):
        xp = x.copy(); xp.flat[i] += eps
        xm = x.copy(); xm.flat[i] -= eps
        grad.flat[i] = (f(xp) - f(xm)) / (2.0 * eps)
    return grad


def lipschitz_constant(f, x_samples: np.ndarray, eps: float = 1e-5) -> float:
    """
    Estimate the Lipschitz constant of f w.r.t. its first argument
    over x_samples via finite differences.
    """
    grads = np.array([np.abs(finite_difference_grad(f, x, eps)) for x in x_samples])
    return float(np.max(grads))


def set_seed(seed: int) -> np.random.Generator:
    """Return a seeded Generator for reproducible results."""
    return np.random.default_rng(seed)
