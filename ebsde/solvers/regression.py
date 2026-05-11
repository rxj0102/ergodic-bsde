"""
Gobet-Lemor-Warin regression-based BSDE solver.

Reference: Gobet, Lemor & Warin (2005) "A regression-based Monte Carlo method
to solve backward stochastic differential equations."
"""

from __future__ import annotations

import numpy as np
from typing import Optional

from ebsde.bsde.standard import StandardBSDE
from ebsde.solvers.picard import _polynomial_basis, _hermite_basis, _rbf_basis


class RegressionBSDESolver:
    """
    Gobet-Lemor-Warin regression-based BSDE solver.

    Key distinction from the Picard solver
    ----------------------------------------
    This solver directly estimates the value function u(t_i, x) by regressing
    Y_{i+1} onto basis functions φ(X_i).  The gradient ∂u/∂x is then recovered
    analytically from the regression coefficients, giving Z_i = σ(X_i) · ∂u/∂x.

    This avoids the separate Z-regression and can reduce estimator variance.

    Algorithm
    ---------
    For i = M-1, ..., 0:
      1. Regress Y_{i+1} on φ(X_i):  α_i = argmin ||Y_{i+1} - Φ α||²
      2. û(t_i, X_i) = Φ α_i
      3. ∂û/∂x(t_i, X_i) = Φ' α_i  (derivative of basis)
      4. Z_i = σ(X_i) · ∂û/∂x
      5. Y_i = û(t_i, X_i) + Δt · f(X_i, Y_i, Z_i)
    """

    def __init__(
        self,
        bsde: StandardBSDE,
        n_paths: int = 50_000,
        n_steps: int = 100,
        basis_type: str = "polynomial",
        basis_degree: int = 5,
        rng_seed: Optional[int] = None,
    ) -> None:
        self.bsde = bsde
        self.n_paths = n_paths
        self.n_steps = n_steps
        self.basis_type = basis_type
        self.basis_degree = basis_degree
        self.rng = np.random.default_rng(rng_seed)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def solve(self) -> dict:
        """
        Returns
        -------
        dict
            Y     : (n_paths, n_steps+1)
            Z     : (n_paths, n_steps, d)
            Y0    : float
            Z0    : (d,)
            times : (n_steps+1,)
            coeffs: list[ndarray] — regression coefficients at each time step
        """
        bsde = self.bsde
        M = self.n_steps
        N = self.n_paths
        d = bsde.forward.dimension
        dt = bsde.T / M

        sim = bsde.forward.simulate(
            x0=np.zeros(d),
            T=bsde.T,
            n_steps=M,
            n_paths=N,
            rng=self.rng,
        )
        paths = sim["paths"]   # (N, M+1, d)
        times = sim["times"]

        X_T = paths[:, -1, 0] if d == 1 else paths[:, -1, :]
        Y = np.zeros((N, M + 1))
        Y[:, M] = bsde.eval_terminal(X_T)

        Z = np.zeros((N, M, d))
        all_coeffs = []

        for i in range(M - 1, -1, -1):
            X_i = paths[:, i, :]      # (N, d)
            Y_next = Y[:, i + 1]      # (N,)

            Phi, dPhi = self._build_basis_and_grad(X_i)  # (N,K), (N,K,d)

            # regression: u(t_i, x) ≈ Φ α
            alpha, *_ = np.linalg.lstsq(Phi, Y_next, rcond=None)
            all_coeffs.append(alpha)

            EY = Phi @ alpha           # (N,) — conditional expectation

            # Z via analytical gradient: Z = σ(X_i) * Φ' α
            Z_i = self._estimate_z_from_coefficients(alpha, dPhi, X_i)  # (N, d)

            x_arg = X_i[:, 0] if d == 1 else X_i
            f_val = bsde.eval_driver(x_arg, EY, Z_i)

            Y[:, i] = EY + dt * f_val
            Z[:, i, :] = Z_i

        Y0 = float(np.mean(Y[:, 0]))
        Z0 = Z[:, 0, :].mean(axis=0)

        return {
            "Y": Y,
            "Z": Z,
            "Y0": Y0,
            "Z0": Z0,
            "times": times,
            "coeffs": list(reversed(all_coeffs)),
        }

    # ------------------------------------------------------------------
    # Z from gradient of basis
    # ------------------------------------------------------------------

    def _estimate_z_from_coefficients(
        self,
        alpha: np.ndarray,    # (K,)
        dPhi: np.ndarray,     # (N, K, d)
        X: np.ndarray,        # (N, d)
    ) -> np.ndarray:
        """
        Z_i = σ(X_i) · ∂û/∂x  where ∂û/∂x = dΦ · α.

        Returns (N, d).
        """
        d = X.shape[1]
        # du_dx: (N, d)
        du_dx = np.einsum("nkj,k->nj", dPhi, alpha)

        # σ(X_i): shape (N, d) for scalar σ broadcast
        sigma_x = self.bsde.forward.diffusion(X)   # (N, d)

        return sigma_x * du_dx

    # ------------------------------------------------------------------
    # Basis and its gradient
    # ------------------------------------------------------------------

    def _build_basis_and_grad(
        self, X: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Build Φ (N, K) and dΦ (N, K, d) — gradient of each basis function.
        """
        if self.basis_type == "polynomial":
            return _poly_basis_and_grad(X, self.basis_degree)
        elif self.basis_type == "hermite":
            return _hermite_basis_and_grad(X, self.basis_degree)
        elif self.basis_type == "rbf":
            return _rbf_basis_and_grad(X, self.basis_degree)
        else:
            raise ValueError(f"Unknown basis_type '{self.basis_type}'")


# ------------------------------------------------------------------
# Basis + gradient implementations
# ------------------------------------------------------------------


def _poly_basis_and_grad(
    X: np.ndarray, degree: int
) -> tuple[np.ndarray, np.ndarray]:
    """
    Polynomial basis and its gradient.

    For d=1:  φ_k(x) = x^k,  φ'_k(x) = k x^{k-1}
    For d>1:  all monomials up to total degree, gradient per variable.
    """
    N, d = X.shape
    # Build list of (exponent tuples, col, grad_cols)
    cols = []
    grad_cols = []  # each element is (N, d)

    # constant
    cols.append(np.ones(N))
    grad_cols.append(np.zeros((N, d)))

    from itertools import combinations_with_replacement
    for deg in range(1, degree + 1):
        for combo in combinations_with_replacement(range(d), deg):
            # monomial: product of X[:, combo[j]]
            col = np.ones(N)
            for idx in combo:
                col = col * X[:, idx]
            cols.append(col)

            # gradient: d/dx_j of product
            gc = np.zeros((N, d))
            for j in range(d):
                count_j = combo.count(j)
                if count_j == 0:
                    continue
                # derivative: count_j * X[:,j]^{count_j-1} * product of others
                deriv = np.ones(N) * count_j
                for idx in combo:
                    if idx == j:
                        if count_j > 1:
                            deriv = deriv * X[:, idx] ** (count_j - 1)
                            break  # handled all j's at once
                    # only do one pass — reconstruct properly
                # Simpler: col / X[:,j] * count_j (with safe division)
                with np.errstate(divide="ignore", invalid="ignore"):
                    gc[:, j] = np.where(
                        X[:, j] != 0,
                        col * count_j / X[:, j],
                        0.0,
                    )
            grad_cols.append(gc)

    Phi = np.column_stack(cols)         # (N, K)
    dPhi = np.stack(grad_cols, axis=1)  # (N, K, d)
    return Phi, dPhi


def _hermite_basis_and_grad(
    X: np.ndarray, degree: int
) -> tuple[np.ndarray, np.ndarray]:
    """
    Probabilist's Hermite polynomials He_k(x) and their derivatives He'_k = k He_{k-1}.
    """
    N, d = X.shape
    # He[j, k]: shape (N,)
    He = np.zeros((N, d, degree + 1))
    He[:, :, 0] = 1.0
    if degree >= 1:
        He[:, :, 1] = X
    for k in range(2, degree + 1):
        He[:, :, k] = X * He[:, :, k - 1] - (k - 1) * He[:, :, k - 2]

    # dHe/dx = k * He_{k-1}
    dHe = np.zeros((N, d, degree + 1))
    for k in range(1, degree + 1):
        dHe[:, :, k] = k * He[:, :, k - 1]

    if d == 1:
        Phi = He[:, 0, :]                   # (N, degree+1)
        dPhi = dHe[:, 0, :, np.newaxis]     # (N, degree+1, 1)
        return Phi, dPhi

    # multi-d: constant + univariate terms
    cols = [np.ones(N)]
    gcols = [np.zeros((N, d))]
    for j in range(d):
        for k in range(1, degree + 1):
            cols.append(He[:, j, k])
            gc = np.zeros((N, d))
            gc[:, j] = dHe[:, j, k]
            gcols.append(gc)

    Phi = np.column_stack(cols)
    dPhi = np.stack(gcols, axis=1)
    return Phi, dPhi


def _rbf_basis_and_grad(
    X: np.ndarray, n_centers: int
) -> tuple[np.ndarray, np.ndarray]:
    """
    Gaussian RBF: φ_c(x) = exp(-||x - c||² / (2 bw²))
    gradient:  dφ_c/dx_j = -(x_j - c_j) / bw² * φ_c(x)
    """
    N, d = X.shape
    quantiles = np.linspace(5, 95, n_centers)
    centers = np.percentile(X[:, 0], quantiles)
    bw = max((centers[-1] - centers[0]) / max(n_centers, 1), 0.1)

    cols = [np.ones(N)]
    gcols = [np.zeros((N, d))]

    for c in centers:
        phi = np.exp(-0.5 * ((X[:, 0] - c) / bw) ** 2)
        cols.append(phi)
        gc = np.zeros((N, d))
        gc[:, 0] = -(X[:, 0] - c) / bw**2 * phi
        gcols.append(gc)

    Phi = np.column_stack(cols)
    dPhi = np.stack(gcols, axis=1)
    return Phi, dPhi
