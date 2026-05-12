"""
Principal eigenvalue of the generator + potential, via ergodic BSDEs.

Donsker-Varadhan theory and Hansen-Scheinkman decomposition.
"""

from __future__ import annotations

import numpy as np
import scipy.linalg
import scipy.sparse
import scipy.sparse.linalg
from typing import Callable, Optional

from ebsde.forward.sde import ForwardSDE
from ebsde.bsde.ergodic import ErgodicBSDE
from ebsde.solvers.ergodic_pde import ErgodicPDESolver


class PrincipalEigenvalueSolver:
    """
    Principal eigenvalue of the operator  A = L + h + c·I.

    For the linear ergodic BSDE with driver f(x, y, z) = h(x) + c·y,
    the ergodic constant λ equals the principal eigenvalue of A.

    Donsker-Varadhan variational formula:
        λ = sup_{φ>0} inf_x [(Aφ)(x) / φ(x)]
          = sup_{μ prob.} [∫ h dμ + c - I(μ)]

    Hansen-Scheinkman (2009) decomposition:
        M_T ∝ exp(-λT) · φ(X_T)/φ(X_0) · L_T

    Parameters
    ----------
    forward : ForwardSDE  (must be ergodic, 1-D for PDE methods)
    """

    def __init__(self, forward: ForwardSDE) -> None:
        if not forward.is_ergodic():
            raise ValueError("forward SDE must be ergodic")
        self.forward = forward

    # ------------------------------------------------------------------
    # Internal: finite-difference operator matrix
    # ------------------------------------------------------------------

    def _build_fd_matrix(
        self,
        h_func: Callable,
        c: float,
        grid: np.ndarray,
    ) -> np.ndarray:
        """
        Build the tridiagonal FD matrix for A = L + h + cI on `grid`.

        L = (σ²/2)d²/dx² + b·d/dx   (finite-difference, interior points only)
        Boundary conditions: φ = 0 at endpoints.
        """
        n = len(grid)
        dx = grid[1] - grid[0]  # assume uniform grid

        b_vals = np.array([float(self.forward.drift(np.array([xi])).flat[0]) for xi in grid])
        s_vals = np.array([float(self.forward.diffusion(np.array([xi])).flat[0]) for xi in grid])
        h_vals = np.asarray(h_func(grid), dtype=float)

        diag = np.zeros(n)
        upper = np.zeros(n - 1)
        lower = np.zeros(n - 1)

        for i in range(n):
            s2 = s_vals[i] ** 2
            b = b_vals[i]
            diag[i] = -s2 / dx**2 + h_vals[i] + c
            if i < n - 1:
                upper[i] = s2 / (2 * dx**2) + b / (2 * dx)
            if i > 0:
                lower[i - 1] = s2 / (2 * dx**2) - b / (2 * dx)

        A = np.diag(diag) + np.diag(upper, 1) + np.diag(lower, -1)
        return A

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compute_principal_eigenvalue(
        self,
        operator_func: Callable,
        grid: np.ndarray,
    ) -> dict:
        """
        Compute the principal eigenvalue λ and positive eigenfunction φ.

        Parameters
        ----------
        operator_func : callable(h_func, c, grid) → matrix A,
                        OR a tuple (h_func, c) for the standard linear case
        grid          : 1-D spatial grid

        Returns
        -------
        dict with:
            'lambda'         : principal eigenvalue (real part of largest eigenvalue)
            'eigenfunction'  : φ > 0 normalised to ‖φ‖_∞ = 1
            'x_grid'         : grid
        """
        grid = np.asarray(grid, dtype=float)

        if isinstance(operator_func, tuple):
            h_func, c = operator_func
            A = self._build_fd_matrix(h_func, c, grid)
        else:
            A = operator_func(grid)

        # Find the eigenvalue with the largest real part
        eigvals, eigvecs = scipy.linalg.eig(A)
        idx = np.argmax(eigvals.real)
        lam = float(eigvals[idx].real)
        phi = eigvecs[:, idx].real

        # Ensure φ > 0 (Perron-Frobenius: unique positive eigenvector)
        if np.mean(phi) < 0:
            phi = -phi
        phi = phi / (np.max(np.abs(phi)) + 1e-14)

        return {
            "lambda": lam,
            "eigenfunction": phi,
            "x_grid": grid,
        }

    def hansen_scheinkman_decomposition(
        self,
        pricing_kernel_params: dict,
    ) -> dict:
        """
        Decompose the pricing kernel using the Hansen-Scheinkman (2009) approach.

        M_T = exp(-λT) · (φ(X_T) / φ(X_0)) · L_T

        where L_T is a martingale component.

        Parameters
        ----------
        pricing_kernel_params : dict with keys:
            'h_func'   : running payoff h(x)
            'c'        : linear drift coefficient in driver f = h(x) + c·y
            'grid'     : spatial grid
            'n_paths'  : number of MC paths for L_T estimation
            'T'        : time horizon
            'x0'       : initial state (default 0.0)
            'rng_seed' : optional random seed

        Returns
        -------
        dict with 'lambda', 'eigenfunction', 'L_T_mean', 'L_T_std',
                  'decomposition_check' (should be ≈ 1)
        """
        h_func = pricing_kernel_params["h_func"]
        c = float(pricing_kernel_params.get("c", 0.0))
        grid = np.asarray(pricing_kernel_params["grid"], dtype=float)
        T = float(pricing_kernel_params.get("T", 5.0))
        n_paths = int(pricing_kernel_params.get("n_paths", 2000))
        x0 = float(pricing_kernel_params.get("x0", 0.0))
        rng = np.random.default_rng(pricing_kernel_params.get("rng_seed", None))

        # Step 1: principal eigenvalue and eigenfunction
        eig_result = self.compute_principal_eigenvalue((h_func, c), grid)
        lam = eig_result["lambda"]
        phi_grid = eig_result["eigenfunction"]
        x_grid_eig = eig_result["x_grid"]

        def phi(x: np.ndarray) -> np.ndarray:
            return np.interp(np.asarray(x, dtype=float), x_grid_eig, phi_grid)

        # Step 2: simulate forward paths
        n_steps = max(50, int(T / 0.05))
        sim = self.forward.simulate(
            x0=np.array([x0]),
            T=T,
            n_steps=n_steps,
            n_paths=n_paths,
            rng=rng,
        )
        paths = sim["paths"][:, :, 0]  # (n_paths, n_steps+1)
        dW = sim["brownian_increments"][:, :, 0]  # (n_paths, n_steps)
        dt = T / n_steps

        x_T = paths[:, -1]  # (n_paths,)
        phi_x0 = float(np.asarray(phi(np.array([x0])), dtype=float).flat[0])
        phi_xT = phi(x_T)  # (n_paths,)

        # M_T proxy: exp(-λT) · φ(X_T)/φ(X_0) · L_T = M_T
        # L_T = M_T / (exp(-λT) · φ(X_T)/φ(X_0))
        # We approximate M_T via exp(-∫h dt) · exp(-c·T) using Euler sum
        h_integral = np.zeros(n_paths)
        for i in range(n_steps):
            h_integral += np.asarray(h_func(paths[:, i]), dtype=float) * dt

        M_T = np.exp(-(h_integral + c * T))  # proxy for exp(-∫(h+c)dt)

        # L_T = M_T · exp(λT) · φ(X_0) / φ(X_T)
        phi_xT_safe = np.where(np.abs(phi_xT) < 1e-14, 1e-14, phi_xT)
        L_T = M_T * np.exp(lam * T) * phi_x0 / phi_xT_safe

        # Martingale check: E[L_T] should be ≈ 1 if decomposition is exact
        L_T_mean = float(np.mean(L_T))
        L_T_std = float(np.std(L_T))

        return {
            "lambda": lam,
            "eigenfunction": phi,
            "L_T_mean": L_T_mean,
            "L_T_std": L_T_std,
            "decomposition_check": L_T_mean,  # ≈ 1 for correct decomposition
        }

    def donsker_varadhan_bound(self, h_func: Callable) -> dict:
        """
        Donsker-Varadhan variational bounds on λ.

        Upper bound: λ ≤ sup_x h(x) + c
        Lower bound: λ ≥ ∫ h dπ + c  (stationary expectation)

        where c=0 here (pure potential case, no linear-y term).

        Parameters
        ----------
        h_func : callable x → h(x)

        Returns
        -------
        dict with 'lower_bound', 'upper_bound', 'gap',
                  'stationary_mean_h', 'sup_h'
        """
        # Evaluate h on a fine grid around the stationary distribution
        try:
            stat = self.forward.stationary_distribution()
            x_mean = float(stat["mean"])
            x_std = float(np.sqrt(stat["variance"]))
        except Exception:
            x_mean, x_std = 0.0, 1.0

        x_grid = np.linspace(x_mean - 5 * x_std, x_mean + 5 * x_std, 1000)
        h_vals = np.asarray(h_func(x_grid), dtype=float)

        # Lower bound: stationary expectation (Gaussian weights)
        from scipy.stats import norm
        weights = norm.pdf(x_grid, loc=x_mean, scale=x_std)
        weights /= weights.sum()
        lower = float(np.sum(h_vals * weights))

        # Upper bound: supremum
        upper = float(np.max(h_vals))

        return {
            "lower_bound": lower,
            "upper_bound": upper,
            "gap": upper - lower,
            "stationary_mean_h": lower,
            "sup_h": upper,
        }

    def verify_against_pde(
        self,
        h_func: Callable,
        c: float = 0.0,
        grid: Optional[np.ndarray] = None,
    ) -> dict:
        """
        Verify principal eigenvalue against the ErgodicPDESolver.

        The linear ergodic BSDE with driver f(x,y,z) = h(x) + c·y
        should give the same λ as the FD eigenvalue solver above.

        Returns dict with 'lambda_eigenvalue', 'lambda_pde', 'absolute_error'.
        """
        if grid is None:
            try:
                stat = self.forward.stationary_distribution()
                x_mean = float(stat["mean"])
                x_std = float(np.sqrt(stat["variance"]))
            except Exception:
                x_mean, x_std = 0.0, 1.0
            grid = np.linspace(x_mean - 5 * x_std, x_mean + 5 * x_std, 300)

        # Eigenvalue computation
        eig_res = self.compute_principal_eigenvalue((h_func, c), grid)
        lam_eig = eig_res["lambda"]

        # PDE solver
        def driver(x, y, z):
            return np.asarray(h_func(np.asarray(x, dtype=float)), dtype=float) + c * np.asarray(y, dtype=float)

        ebsde = ErgodicBSDE(forward=self.forward, driver=driver)
        pde_res = ErgodicPDESolver(ebsde).solve()
        lam_pde = float(pde_res["lambda_ergodic"])

        return {
            "lambda_eigenvalue": float(lam_eig),
            "lambda_pde": lam_pde,
            "absolute_error": abs(lam_eig - lam_pde),
        }
