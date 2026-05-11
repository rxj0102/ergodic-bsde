"""
PDE solver for Markovian BSDEs.

Solves the quasilinear PDE backward in time on a finite spatial grid.
"""

from __future__ import annotations

import numpy as np
from scipy import linalg, sparse
from scipy.sparse.linalg import spsolve
from typing import Optional

from ebsde.bsde.standard import StandardBSDE


class PDEBSDESolver:
    """
    Solve the Markovian BSDE via the associated quasilinear PDE:

        ∂u/∂t + (1/2)σ(x)² ∂²u/∂x² + b(x) ∂u/∂x + f(x, u, σ∂u/∂x) = 0
        u(T, x) = g(x)

    Discretisation
    --------------
    Space : uniform finite-difference grid on [x_min, x_max], n_x interior pts.
    Time  : Crank-Nicolson (θ=1/2) for the diffusion/drift (linear) part;
            explicit treatment of the nonlinear driver f.

    For nonlinear drivers (quadratic / entropy), a Newton sub-iteration is used
    within each time step to handle implicit nonlinearity.

    Boundary conditions
    -------------------
    Dirichlet: u = 0 at x_min and x_max.  Valid when the OU density decays
    fast enough that the PDE solution is negligible at the boundaries.

    After solving u(t, x):
    - Y_0 = u(0, X_0)
    - Z(t, x) = σ(x) · ∂u/∂x  recovered by finite differences.
    """

    def __init__(
        self,
        bsde: StandardBSDE,
        n_x: int = 200,
        n_t: int = 500,
        x_range: tuple[float, float] = (-5.0, 5.0),
        theta_cn: float = 0.5,
        max_newton: int = 10,
        newton_tol: float = 1e-10,
    ) -> None:
        self.bsde = bsde
        self.n_x = n_x
        self.n_t = n_t
        self.x_range = x_range
        self.theta_cn = theta_cn          # Crank-Nicolson parameter (0=explicit,1=implicit)
        self.max_newton = max_newton
        self.newton_tol = newton_tol

        # Build spatial grid (interior points only; Dirichlet BCs at boundaries)
        x_min, x_max = x_range
        self.x_full = np.linspace(x_min, x_max, n_x + 2)   # includes boundary pts
        self.x_grid = self.x_full[1:-1]                      # (n_x,)
        self.dx = self.x_full[1] - self.x_full[0]

        self.dt = bsde.T / n_t
        self.times = np.linspace(0.0, bsde.T, n_t + 1)

        self._build_generator()

    # ------------------------------------------------------------------
    # Build generator matrix A
    # ------------------------------------------------------------------

    def _build_generator(self) -> None:
        """
        Build the tridiagonal generator matrix A for interior grid points:
        A u ≈ (σ²/2) u_xx + b u_x

        Using central differences:
        u_xx ≈ (u_{j+1} - 2u_j + u_{j-1}) / dx²
        u_x  ≈ (u_{j+1} - u_{j-1}) / (2dx)
        """
        n = self.n_x
        dx = self.dx
        x = self.x_grid

        sigma_x = self.bsde.forward.diffusion(x.reshape(-1, 1)).reshape(n)
        b_x     = self.bsde.forward.drift(x.reshape(-1, 1)).reshape(n)

        # Diffusion coefficients
        D = 0.5 * sigma_x**2

        # Sub-, main, super-diagonal of A
        lower = D[1:] / dx**2  - b_x[1:] / (2 * dx)     # (n-1,)
        main  = -2 * D / dx**2                             # (n,)
        upper = D[:-1] / dx**2 + b_x[:-1] / (2 * dx)     # (n-1,)

        self.A = (
            np.diag(main)
            + np.diag(lower, k=-1)
            + np.diag(upper, k=1)
        )
        self.A_sparse = sparse.csr_matrix(self.A)

        # Pre-factor LHS for fully-implicit step (used when theta=1)
        theta = self.theta_cn
        self.LHS = sparse.eye(n) - theta * self.dt * self.A_sparse

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def solve(self, x0: Optional[float] = None) -> dict:
        """
        Solve the PDE backward in time.

        Parameters
        ----------
        x0 : float, optional
            Point at which to evaluate Y_0 = u(0, x0).  Defaults to x_grid mean.

        Returns
        -------
        dict
            u        : (n_t+1, n_x) PDE solution on interior grid
            du_dx    : (n_t+1, n_x) spatial gradient
            Y0       : float — u(0, x0)
            Z0       : float — σ(x0) · ∂u/∂x(0, x0)
            x_grid   : (n_x,) spatial grid
            times    : (n_t+1,) time grid (0 → T)
        """
        n = self.n_x
        M = self.n_t
        bsde = self.bsde

        # Storage: u[k] = u at time times[k]
        u = np.zeros((M + 1, n))

        # Terminal condition at t=T (index M)
        u[M] = bsde.eval_terminal(self.x_grid)

        # Backward in time
        for k in range(M - 1, -1, -1):
            u_next = u[k + 1]
            u[k] = self._time_step(u_next)

        # Gradient
        du_dx = np.gradient(u, self.dx, axis=1)

        # Y0 and Z0
        if x0 is None:
            x0 = float(np.mean(self.x_grid))
        Y0 = float(np.interp(x0, self.x_grid, u[0]))
        du_dx_0 = float(np.interp(x0, self.x_grid, du_dx[0]))
        sigma_x0 = float(
            self.bsde.forward.diffusion(np.array([[x0]])).ravel()[0]
        )
        Z0 = sigma_x0 * du_dx_0

        return {
            "u": u,
            "du_dx": du_dx,
            "Y0": Y0,
            "Z0": Z0,
            "x_grid": self.x_grid,
            "times": self.times,
        }

    # ------------------------------------------------------------------
    # Time-stepping
    # ------------------------------------------------------------------

    def _time_step(self, u_next: np.ndarray) -> np.ndarray:
        """Dispatch to linear or nonlinear solver for one backward step."""
        # Check whether driver is linear by probing dependence on u and du/dx
        # We always use the nonlinear (Newton) solver for generality.
        return self._time_step_nonlinear(u_next)

    def _time_step_linear(self, u_next: np.ndarray, dt: float) -> np.ndarray:
        """
        One backward CN step assuming f(x, u, z) = h(x) (no u/z dependence):

            (I - θ dt A) u_curr = (I + (1-θ) dt A) u_next + dt h(x)
        """
        n = self.n_x
        dt = self.dt
        theta = self.theta_cn
        x = self.x_grid

        # h(x) = f(x, 0, 0)
        h = self.bsde.eval_driver(x, np.zeros(n), np.zeros((n, 1)))

        rhs = (
            u_next
            + (1 - theta) * dt * (self.A @ u_next)
            + dt * h
        )
        u_curr = spsolve(self.LHS, rhs)
        return u_curr

    def _time_step_nonlinear(
        self,
        u_next: np.ndarray,
    ) -> np.ndarray:
        """
        One backward step for a nonlinear driver via Newton iteration.

        Equation to solve for u_curr:
            u_curr - u_next - dt [θ A u_curr + (1-θ) A u_next
                                  + f(x, u_curr, σ ∂u_curr/∂x)] = 0

        Newton: F(u) = 0,  J δu = -F(u),  u ← u + δu
        """
        n = self.n_x
        dt = self.dt
        theta = self.theta_cn
        x = self.x_grid
        dx = self.dx

        # Pre-compute the explicit part
        explicit = u_next + (1 - theta) * dt * (self.A @ u_next)

        # Initial guess
        u = u_next.copy()
        sigma_x = self.bsde.forward.diffusion(x.reshape(-1, 1)).reshape(n)

        for _ in range(self.max_newton):
            # Gradient of u (finite difference, with zero at ghost boundary)
            u_padded = np.concatenate([[0.0], u, [0.0]])
            du_dx = (u_padded[2:] - u_padded[:-2]) / (2 * dx)
            z = sigma_x * du_dx               # (n,)

            f_val = self.bsde.eval_driver(x, u, z[:, np.newaxis])

            # Residual
            F = u - explicit - theta * dt * (self.A @ u) - dt * f_val

            if np.max(np.abs(F)) < self.newton_tol:
                break

            # Jacobian ≈ (I - θ dt A) + small nonlinear correction
            # Use the linear part as Jacobian approximation (quasi-Newton / Picard)
            u = spsolve(self.LHS, explicit + dt * f_val)

        return u
