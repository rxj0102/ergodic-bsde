"""
Ergodic PDE solver for ergodic BSDEs.

Solves the nonlinear eigenvalue problem:
    (σ²/2) v''(x) + b(x) v'(x) + f(x, v(x), σ(x)v'(x)) = λ

with normalisation v(x₀) = 0.
"""

from __future__ import annotations

import numpy as np
from scipy import sparse, linalg
from scipy.sparse.linalg import spsolve, eigs
from typing import Optional

from ebsde.bsde.ergodic import ErgodicBSDE


class ErgodicPDESolver:
    """
    Solve the ergodic PDE eigenvalue problem on a spatial grid.

    Two methods:
    - 'iteration': iterate  λ ← ∫ (Lv + f) μ dx  with fixed-point updates
    - 'newton'   : full Newton on the augmented (v, λ) system
    - 'linear'   : fast eigenvalue method for drivers linear in (y,z)

    For OU forward processes, the domain is auto-set to ±n_std stationary
    standard deviations if x_range is None.
    """

    def __init__(
        self,
        ergodic_bsde: ErgodicBSDE,
        n_x: int = 500,
        x_range: Optional[tuple] = None,
        method: str = "iteration",
        max_iter: int = 200,
        tol: float = 1e-8,
        n_std: float = 4.0,
    ) -> None:
        self.ebsde = ergodic_bsde
        self.n_x = n_x
        self.method = method
        self.max_iter = max_iter
        self.tol = tol

        fwd = ergodic_bsde.forward
        if x_range is None:
            if hasattr(fwd, "stationary_distribution"):
                sd = fwd.stationary_distribution()
                mu_s = sd["mean"]
                std_s = np.sqrt(sd["variance"])
            else:
                mu_s, std_s = 0.0, 1.0
            x_range = (float(mu_s - n_std * std_s), float(mu_s + n_std * std_s))

        x_min, x_max = x_range
        # Full grid including boundary points (Neumann BCs at edges)
        self.x_grid = np.linspace(x_min, x_max, n_x)
        self.dx = self.x_grid[1] - self.x_grid[0]

        self._build_operator()
        self._compute_stationary_weights()

    # ------------------------------------------------------------------
    # Build discretised generator L
    # ------------------------------------------------------------------

    def _build_operator(self) -> None:
        """
        Build tridiagonal matrix A for Lv = (σ²/2)v'' + b v'.
        Uses central differences; Neumann BCs (ghost-point extension).
        """
        n = self.n_x
        dx = self.dx
        x = self.x_grid

        fwd = self.ebsde.forward
        sigma_x = fwd.diffusion(x.reshape(-1, 1)).ravel()
        b_x = fwd.drift(x.reshape(-1, 1)).ravel()

        D = 0.5 * sigma_x**2  # diffusion coeff

        # Interior rows: central differences
        lower = D[1:] / dx**2 - b_x[1:] / (2 * dx)
        main  = -2 * D / dx**2
        upper = D[:-1] / dx**2 + b_x[:-1] / (2 * dx)

        # Neumann BC at left boundary (i=0): ghost point x_{-1}=x_1
        # v'' ≈ 2(v_1 - v_0)/dx²,  v' ≈ 0  (no drift at ghost)
        main[0]  = -2 * D[0] / dx**2
        upper[0] =  2 * D[0] / dx**2   # doubles upper for Neumann

        # Neumann BC at right boundary (i=n-1): ghost point x_{n}=x_{n-2}
        main[-1]   = -2 * D[-1] / dx**2
        lower[-1]  =  2 * D[-1] / dx**2

        self.A = (
            np.diag(main)
            + np.diag(lower, k=-1)
            + np.diag(upper, k=1)
        )
        self.A_sparse = sparse.csr_matrix(self.A)
        self.sigma_x = sigma_x
        self.b_x = b_x

    # ------------------------------------------------------------------
    # Stationary measure weights for ∫·μ dx
    # ------------------------------------------------------------------

    def _compute_stationary_weights(self) -> None:
        """
        Compute normalised weights w_i ∝ μ(x_i) Δx for integration.

        For OU: μ is Gaussian.  For general processes we use the fact that
        the stationary density satisfies (L* μ = 0):
            μ ∝ exp(2 ∫ b(x)/σ(x)² dx)
        For OU(κ, θ, σ): μ ∝ exp(-κ(x-θ)²/σ²)
        """
        fwd = self.ebsde.forward
        x = self.x_grid

        if hasattr(fwd, "stationary_distribution"):
            sd = fwd.stationary_distribution()
            mu_s = sd["mean"]
            var_s = sd["variance"]
            log_density = -0.5 * (x - mu_s) ** 2 / var_s
        else:
            # General: integrate b/σ² by quadrature
            sigma_x = fwd.diffusion(x.reshape(-1, 1)).ravel()
            b_x = fwd.drift(x.reshape(-1, 1)).ravel()
            integrand = b_x / sigma_x**2
            log_density = np.cumsum(integrand) * self.dx

        log_density -= log_density.max()
        density = np.exp(log_density)
        density /= density.sum() * self.dx
        self.mu_weights = density * self.dx  # (n_x,), sums to ≈1

    # ------------------------------------------------------------------
    # Driver evaluation helpers
    # ------------------------------------------------------------------

    def _eval_f(self, v: np.ndarray, dv: np.ndarray) -> np.ndarray:
        """Evaluate f(x, v, σ·dv) on the grid."""
        x = self.x_grid
        z = (self.sigma_x * dv).reshape(-1, 1)
        return self.ebsde.driver(x, v, z)

    def _gradient(self, v: np.ndarray) -> np.ndarray:
        """Central-difference gradient with Neumann BCs."""
        dv = np.empty_like(v)
        dv[1:-1] = (v[2:] - v[:-2]) / (2 * self.dx)
        dv[0]  = (v[1] - v[0]) / self.dx     # forward at left
        dv[-1] = (v[-1] - v[-2]) / self.dx   # backward at right
        return dv

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def solve(self) -> dict:
        fwd = self.ebsde.forward

        # Choose method
        if self.method == "linear":
            return self._solve_linear_eigenvalue()
        elif self.method == "newton":
            return self._solve_newton_augmented()
        else:
            return self._solve_nonlinear_iteration()

    # ------------------------------------------------------------------
    # Method 1: eigenvalue (linear drivers only)
    # ------------------------------------------------------------------

    def _solve_linear_eigenvalue(self) -> dict:
        """
        For f(x,y,z) = h(x) + c·y  (no z dependence), the ergodic PDE is:

            (A + c·I)v - λ·1 = -h,   v[mid] = 0

        Solved as the (n+1)×(n+1) augmented linear system:

            [(A + c·I)  -1 ] [v]   [-h]
            [eₘᵢdᵀ      0  ] [λ] = [0 ]
        """
        x = self.x_grid
        n = self.n_x
        mid = n // 2

        # Extract h(x) = f(x, 0, 0) and c from linearity
        h = self.ebsde.driver(x, np.zeros(n), np.zeros((n, 1)))
        eps = 1e-6
        f_y = self.ebsde.driver(x, eps * np.ones(n), np.zeros((n, 1)))
        c = float(np.mean((f_y - h) / eps))

        # Augmented (n+1) x (n+1) system
        J = np.zeros((n + 1, n + 1))
        J[:n, :n] = self.A + c * np.eye(n)
        J[:n, n] = -1.0   # λ column
        J[n, mid] = 1.0   # normalisation: v[mid] = 0

        rhs = np.zeros(n + 1)
        rhs[:n] = -h

        sol = np.linalg.solve(J, rhs)
        v = sol[:n]
        lambda_ergodic = float(sol[n])

        dv = self._gradient(v)
        z = self.sigma_x * dv
        residual = float(np.max(np.abs(self.A @ v + h + c * v - lambda_ergodic)))

        return {
            "lambda_ergodic": lambda_ergodic,
            "v": v,
            "dv": dv,
            "z": z,
            "x_grid": x,
            "residual": residual,
            "convergence_history": {0: lambda_ergodic},
            "method": "linear_eigenvalue",
        }

    # ------------------------------------------------------------------
    # Method 2: fixed-point iteration on λ
    # ------------------------------------------------------------------

    def _solve_nonlinear_iteration(self) -> dict:
        """
        Iterates:
          1. Solve (A - λᵏ I) v = -f(x, v, σv') with v(x₀) = 0 pinned.
          2. λᵏ⁺¹ = ∫ (Av + f) μ dx
        """
        n = self.n_x
        x = self.x_grid
        mid = n // 2

        # Initialise
        lambda_k = 0.0
        v = np.zeros(n)
        history: dict[int, float] = {0: lambda_k}

        for k in range(1, self.max_iter + 1):
            # --- solve BVP: (A - λI) v = -f, v[mid] = 0 ---
            # Augmented system: add a Lagrange-multiplier row to pin v[mid]=0
            dv = self._gradient(v)
            f_val = self._eval_f(v, dv)

            rhs = -f_val + lambda_k * np.ones(n)

            # Modify A: replace row mid with e_mid (pinning equation)
            A_mod = self.A.copy()
            A_mod[mid, :] = 0.0
            A_mod[mid, mid] = 1.0
            rhs[mid] = 0.0   # v[mid] = 0

            v_new = np.linalg.solve(A_mod, rhs)

            # --- update λ ---
            dv_new = self._gradient(v_new)
            f_new = self._eval_f(v_new, dv_new)
            Lv = self.A @ v_new
            lambda_new = float(np.dot(Lv + f_new, self.mu_weights))

            history[k] = lambda_new

            delta_lambda = abs(lambda_new - lambda_k)
            delta_v = float(np.linalg.norm(v_new - v)) / np.sqrt(n)

            v = v_new
            lambda_k = lambda_new

            if delta_lambda < self.tol and delta_v < self.tol:
                break

        # Normalise v
        v -= v[mid]
        dv = self._gradient(v)
        z = self.sigma_x * dv

        # Compute residual
        f_val = self._eval_f(v, dv)
        Lv = self.A @ v
        residual = float(np.max(np.abs(Lv + f_val - lambda_k)))

        return {
            "lambda_ergodic": lambda_k,
            "v": v,
            "dv": dv,
            "z": z,
            "x_grid": x,
            "residual": residual,
            "convergence_history": history,
            "method": "iteration",
        }

    # ------------------------------------------------------------------
    # Method 3: Newton on augmented system
    # ------------------------------------------------------------------

    def _solve_newton_augmented(self) -> dict:
        """
        Newton on the (n+1)-dimensional system:
          F_i = (Av)_i + f(x_i, v_i, σ_i v'_i) - λ = 0  for i=0..n-1
          F_n = Σ v_i μ_i Δx = 0   (normalisation)
        """
        n = self.n_x
        x = self.x_grid
        mu = self.mu_weights

        # Initial guess: λ=0, v=0
        v = np.zeros(n)
        lambda_k = 0.0
        history: dict[int, float] = {0: lambda_k}

        eps_fd = 1e-5  # finite-difference step for Jacobian

        for k in range(1, self.max_iter + 1):
            dv = self._gradient(v)
            f_val = self._eval_f(v, dv)
            Lv = self.A @ v

            # Residual
            F = np.empty(n + 1)
            F[:n] = Lv + f_val - lambda_k
            F[n] = np.dot(v, mu)

            res_norm = float(np.linalg.norm(F))
            if res_norm < self.tol:
                break

            # Jacobian columns: ∂F_i/∂v_j via finite differences
            J = np.zeros((n + 1, n + 1))

            for j in range(n):
                v_pert = v.copy()
                v_pert[j] += eps_fd
                dv_p = self._gradient(v_pert)
                f_p = self._eval_f(v_pert, dv_p)
                Lv_p = self.A @ v_pert
                J[:n, j] = (Lv_p + f_p - lambda_k - F[:n]) / eps_fd

            # ∂F/∂λ = -1 for all i<n, 0 for i=n
            J[:n, n] = -1.0
            J[n, :n] = mu
            J[n, n] = 0.0

            # Newton step
            try:
                delta = np.linalg.solve(J, -F)
            except np.linalg.LinAlgError:
                delta = np.linalg.lstsq(J, -F, rcond=None)[0]

            v += delta[:n]
            lambda_k += float(delta[n])
            history[k] = lambda_k

        dv = self._gradient(v)
        z = self.sigma_x * dv
        f_val = self._eval_f(v, dv)
        residual = float(np.max(np.abs(self.A @ v + f_val - lambda_k)))

        return {
            "lambda_ergodic": lambda_k,
            "v": v,
            "dv": dv,
            "z": z,
            "x_grid": x,
            "residual": residual,
            "convergence_history": history,
            "method": "newton",
        }
