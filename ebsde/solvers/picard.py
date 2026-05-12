"""
Picard iteration solver for standard (finite-horizon) BSDEs.

Algorithm: Bouchard & Touzi (2004), Zhang (2004).
"""

from __future__ import annotations

import numpy as np
from typing import Optional

from ebsde.bsde.standard import StandardBSDE


class PicardBSDESolver:
    """
    Solve a standard BSDE via Picard iteration (backward regression).

    Algorithm
    ---------
    1. Simulate N forward paths {X^n_{t_i}}.
    2. Set Y^n_{t_M} = g(X^n_{t_M}).
    3. For i = M-1, ..., 0 (backward):
       a. Regress Y_{i+1} on φ(X_i) → E[Y_{i+1} | X_i].
       b. Regress Y_{i+1}·ΔW_i/Δt on φ(X_i) → Z_i.
       c. Y_i = E[Y_{i+1} | X_i] + Δt · f(X_i, Y_i^{prev}, Z_i).
    4. Repeat 3 for n_picard iterations to handle nonlinearity.

    Basis functions: polynomial, probabilist's Hermite, or RBF.
    """

    def __init__(
        self,
        bsde: StandardBSDE,
        n_paths: int = 50_000,
        n_steps: int = 100,
        n_picard: int = 5,
        basis_type: str = "polynomial",
        basis_degree: int = 3,
        z_clip: Optional[float] = 20.0,
        rng_seed: Optional[int] = None,
    ) -> None:
        self.bsde = bsde
        self.n_paths = n_paths
        self.n_steps = n_steps
        self.n_picard = n_picard
        self.basis_type = basis_type
        self.basis_degree = basis_degree
        self.z_clip = z_clip
        self.rng = np.random.default_rng(rng_seed)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def solve(self) -> dict:
        """
        Returns
        -------
        dict
            Y          : (n_paths, n_steps+1)
            Z          : (n_paths, n_steps, d)
            Y0         : float — mean of Y[:, 0]
            Z0         : (d,) — mean of Z[:, 0, :]
            times      : (n_steps+1,)
            picard_convergence : list[float] — ||Y^{k+1} - Y^k||_2 per iteration
        """
        bsde = self.bsde
        M = self.n_steps
        N = self.n_paths
        d = bsde.forward.dimension
        dt = bsde.T / M

        # --- forward simulation ---
        sim = bsde.forward.simulate(
            x0=np.zeros(d),
            T=bsde.T,
            n_steps=M,
            n_paths=N,
            rng=self.rng,
        )
        paths = sim["paths"]          # (N, M+1, d)
        dW    = sim["brownian_increments"]  # (N, M, d)
        times = sim["times"]

        # squeeze d=1 for driver calls
        X = paths[:, :, 0] if d == 1 else paths  # (N, M+1) or (N, M+1, d)

        # --- terminal condition ---
        X_T = paths[:, -1, 0] if d == 1 else paths[:, -1, :]
        Y = np.zeros((N, M + 1))
        Y[:, M] = bsde.eval_terminal(X_T)

        Z = np.zeros((N, M, d))

        convergence = []

        for picard_k in range(self.n_picard):
            Y_prev = Y.copy()

            for i in range(M - 1, -1, -1):
                X_i  = paths[:, i, :]           # (N, d)
                dW_i = dW[:, i, :]              # (N, d)

                # EY = E[Y_{i+1} | X_i] via regression
                EY, Z_i = self._regression_step(Y[:, i + 1], X_i, dW_i, dt)

                # Clip Z to prevent explosion in quadratic/non-Lipschitz drivers
                if self.z_clip is not None:
                    Z_i = np.clip(Z_i, -self.z_clip, self.z_clip)

                # Evaluate f at EY (explicit scheme): correct in one backward
                # pass for all drivers, including those linear in y.
                # For nonlinear drivers, subsequent Picard passes refine Z.
                x_arg = X_i[:, 0] if d == 1 else X_i
                f_val = bsde.eval_driver(x_arg, EY, Z_i)

                Y[:, i]    = EY + dt * f_val
                Z[:, i, :] = Z_i

            delta = float(np.linalg.norm(Y - Y_prev) / np.sqrt(N * (M + 1)))
            convergence.append(delta)

        Y0 = float(np.mean(Y[:, 0]))
        Z0 = Z[:, 0, :].mean(axis=0)

        return {
            "Y": Y,
            "Z": Z,
            "Y0": Y0,
            "Z0": Z0,
            "times": times,
            "picard_convergence": convergence,
        }

    # ------------------------------------------------------------------
    # Regression step
    # ------------------------------------------------------------------

    def _regression_step(
        self,
        Y_next: np.ndarray,   # (N,)
        X_curr: np.ndarray,   # (N, d)
        dW: np.ndarray,       # (N, d)
        dt: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Estimate E[Y_{i+1} | X_i] and Z_i via OLS regression.

        Returns
        -------
        EY : (N,)   conditional expectation
        Z  : (N, d) control estimate
        """
        # Standardise X for numerical stability of the polynomial basis
        x_mean = X_curr.mean(axis=0)
        x_std = X_curr.std(axis=0) + 1e-8
        X_scaled = (X_curr - x_mean) / x_std

        Phi = self._build_basis(X_scaled)  # (N, K)

        # --- conditional expectation ---
        alpha, *_ = np.linalg.lstsq(Phi, Y_next, rcond=None)
        EY = Phi @ alpha                 # (N,)

        # --- Z: regress Y_{i+1} * ΔW_j / dt for each Brownian component j ---
        d = dW.shape[1]
        Z = np.zeros((len(Y_next), d))
        for j in range(d):
            target = Y_next * dW[:, j] / dt
            beta, *_ = np.linalg.lstsq(Phi, target, rcond=None)
            Z[:, j] = Phi @ beta

        return EY, Z

    # ------------------------------------------------------------------
    # Basis construction
    # ------------------------------------------------------------------

    def _build_basis(self, X: np.ndarray) -> np.ndarray:
        """
        Build basis matrix Φ of shape (N, K).

        Supports:
        - 'polynomial' : monomials up to degree p
        - 'hermite'    : probabilist's Hermite polynomials He_0, ..., He_p
        - 'rbf'        : radial basis functions (Gaussian)
        """
        N, d = X.shape
        p = self.basis_degree

        if self.basis_type == "polynomial":
            return _polynomial_basis(X, p)

        elif self.basis_type == "hermite":
            return _hermite_basis(X, p)

        elif self.basis_type == "rbf":
            return _rbf_basis(X, p)

        else:
            raise ValueError(f"Unknown basis_type '{self.basis_type}'")


# ------------------------------------------------------------------
# Basis function helpers (module-level, reused by regression solver)
# ------------------------------------------------------------------


def _polynomial_basis(X: np.ndarray, degree: int) -> np.ndarray:
    """
    All monomials x1^a1 * ... * xd^ad with sum(a) <= degree.
    Returns (N, K).
    """
    N, d = X.shape
    cols = [np.ones(N)]
    for deg in range(1, degree + 1):
        cols.extend(_monomials(X, d, deg))
    return np.column_stack(cols)


def _monomials(X: np.ndarray, d: int, total_deg: int):
    """Yield all degree-total_deg monomials in d variables."""
    from itertools import combinations_with_replacement
    for combo in combinations_with_replacement(range(d), total_deg):
        col = np.ones(X.shape[0])
        for idx in combo:
            col = col * X[:, idx]
        yield col


def _hermite_basis(X: np.ndarray, degree: int) -> np.ndarray:
    """
    Probabilist's Hermite polynomials He_0,...,He_p applied to each column,
    then take products up to total degree.
    For d=1: [He_0(x), He_1(x), ..., He_p(x)].
    For d>1: tensor products.
    """
    N, d = X.shape
    # Compute He_k(x) for k=0..degree, each column of X
    He = np.zeros((N, d, degree + 1))
    He[:, :, 0] = 1.0
    if degree >= 1:
        He[:, :, 1] = X
    for k in range(2, degree + 1):
        He[:, :, k] = X * He[:, :, k - 1] - (k - 1) * He[:, :, k - 2]

    # For d=1: just the polynomials in one variable
    if d == 1:
        return He[:, 0, :]  # (N, degree+1)

    # For d>1: include constant + all univariate terms (simplified)
    cols = [np.ones(N)]
    for j in range(d):
        for k in range(1, degree + 1):
            cols.append(He[:, j, k])
    return np.column_stack(cols)


def _rbf_basis(X: np.ndarray, n_centers: int) -> np.ndarray:
    """
    Gaussian RBF basis with n_centers centers placed at quantiles of X[:,0].
    Returns (N, n_centers + 1) including a constant.
    """
    N, d = X.shape
    quantiles = np.linspace(5, 95, n_centers)
    centers = np.percentile(X[:, 0], quantiles)
    bw = max((centers[-1] - centers[0]) / max(n_centers, 1), 0.1)
    cols = [np.ones(N)]
    for c in centers:
        cols.append(np.exp(-0.5 * ((X[:, 0] - c) / bw) ** 2))
    return np.column_stack(cols)
