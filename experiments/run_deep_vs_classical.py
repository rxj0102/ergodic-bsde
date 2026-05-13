"""
Experiment 6: Deep solver vs classical PDE solver.

Compares ErgodicDeepBSDESolver against ErgodicPDESolver for d=1.

Note on curse of dimensionality:
  PDE solvers discretise a d-dimensional spatial grid.  The number of grid
  points grows as n_x^d, making PDE methods infeasible for d > 1 (for d=2
  with n_x=500 each dimension, storage and compute cost becomes 500^2 = 250,000
  points; for d=5 it is 500^5 ~ 3e13 points — completely intractable).
  Deep solvers parametrise v(x) with a neural network: their cost scales
  polynomially (O(n_samples * n_epochs)) regardless of dimension, making them
  the method of choice for high-dimensional ergodic BSDEs.

For d=1: run both solvers and compare lambda, error, and wall-clock time.
"""

import sys
import os
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from ebsde.forward.ou_process import OrnsteinUhlenbeck
from ebsde.bsde.ergodic import ErgodicBSDE
from ebsde.solvers.ergodic_pde import ErgodicPDESolver
from ebsde.solvers.ergodic_deep import ErgodicDeepBSDESolver
from data.synthetic import ergodic_ou_quadratic_analytical

os.makedirs('experiments/figures', exist_ok=True)

print("=== Experiment 6: Deep BSDE vs Classical PDE (d=1) ===\n")

# ---------------------------------------------------------------
# Problem: OU + quadratic driver, d=1
# ---------------------------------------------------------------
ou = OrnsteinUhlenbeck(kappa=1.0, theta=0.0, sigma=1.0)

def quadratic_driver(x, y, z):
    x = np.asarray(x, dtype=float)
    z = np.asarray(z, dtype=float)
    z_sq = np.sum(z**2, axis=-1) if z.ndim >= 2 else z.ravel()**2
    return x**2 - 0.5 * z_sq

ebsde = ErgodicBSDE(forward=ou, driver=quadratic_driver)

ref = ergodic_ou_quadratic_analytical(kappa=1.0, sigma=1.0, alpha=1.0, beta=0.0, gamma=1.0)
lambda_exact = ref['lambda_ergodic']

RNG_SEED = 42
TORCH_SEED = 42

print(f"  Problem: OU(kappa=1, sigma=1, theta=0) + f(x,y,z) = x^2 - 0.5*z^2")
print(f"  lambda_exact = {lambda_exact:.6f}\n")

d = 1
rows = []

# ---------------------------------------------------------------
# Classical PDE solver (feasible for d=1 only)
# ---------------------------------------------------------------
print("--- ErgodicPDESolver (d=1) ---")
try:
    t0 = time.perf_counter()
    pde_sol = ErgodicPDESolver(ebsde, n_x=500, method='iteration').solve()
    elapsed_pde = time.perf_counter() - t0
    lam_pde = pde_sol['lambda_ergodic']
    err_pde = abs(lam_pde - lambda_exact)
    rows.append(('PDE (classical)', d, lam_pde, err_pde, elapsed_pde, 'ok'))
    print(f"  lambda = {lam_pde:.6f}  |  error = {err_pde:.2e}  |  time = {elapsed_pde:.3f}s\n")
except Exception as exc:
    print(f"  FAILED: {exc}\n")
    rows.append(('PDE (classical)', d, float('nan'), float('nan'), float('nan'), f'FAILED: {exc}'))

# ---------------------------------------------------------------
# Deep BSDE solver
# ---------------------------------------------------------------
print("--- ErgodicDeepBSDESolver (d=1, strategy='direct_ergodic', n_epochs=1000) ---")
deep_sol = None
try:
    t0 = time.perf_counter()
    deep_solver = ErgodicDeepBSDESolver(
        ebsde,
        strategy='direct_ergodic',
        n_epochs=1000,
        n_samples=512,
        rng_seed=RNG_SEED,
        torch_seed=TORCH_SEED,
    )
    deep_sol = deep_solver.solve()
    elapsed_deep = time.perf_counter() - t0
    lam_deep = deep_sol['lambda_ergodic']
    err_deep = abs(lam_deep - lambda_exact)
    rows.append(('Deep (direct_ergodic)', d, lam_deep, err_deep, elapsed_deep, 'ok'))
    print(f"  lambda = {lam_deep:.6f}  |  error = {err_deep:.2e}  |  time = {elapsed_deep:.3f}s\n")
except Exception as exc:
    print(f"  FAILED: {exc}\n")
    rows.append(('Deep (direct_ergodic)', d, float('nan'), float('nan'), float('nan'), f'FAILED: {exc}'))

# ---------------------------------------------------------------
# Print comparison table
# ---------------------------------------------------------------
print("--- Comparison Table ---\n")
header = f"{'method':<24} {'d':>3} {'lambda':>12} {'error':>12} {'time_s':>10} {'status':>8}"
sep = '-' * len(header)
print(header)
print(sep)
for method, dim, lam, err, t_s, status in rows:
    print(f"{method:<24} {dim:>3d} {lam:>12.6f} {err:>12.2e} {t_s:>10.3f} {status:>8}")
print()

# ---------------------------------------------------------------
# Dimensionality note
# ---------------------------------------------------------------
print("--- Note: Curse of Dimensionality ---\n")
print("  PDE solvers are only feasible for d=1 in this library.")
print("  Reason: a spatial grid with n_x points per dimension requires n_x^d")
print("  grid points total. For n_x=500 and d=2, this is 250,000 points;")
print("  for d=5 it is ~3e13 points — completely intractable.")
print("  Deep BSDE solvers parametrise v(x) with a neural network, costing")
print("  O(n_samples * n_epochs) regardless of d, enabling d >> 1.\n")

# ---------------------------------------------------------------
# Save comparison plot (v(x) for d=1 only)
# ---------------------------------------------------------------
try:
    if deep_sol is not None and pde_sol is not None:
        x_grid_pde = pde_sol['x_grid']
        v_pde = pde_sol['v']
        # Centre PDE solution to match deep (v(0)=0)
        mid_idx = np.argmin(np.abs(x_grid_pde))
        v_pde_centred = v_pde - v_pde[mid_idx]

        x_grid_deep = deep_sol.get('x_grid', np.linspace(-4, 4, 200))
        v_deep = deep_sol.get('v_values', np.zeros_like(x_grid_deep))

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(x_grid_pde, v_pde_centred, label='PDE (classical)', linewidth=2)
        ax.plot(x_grid_deep, v_deep, '--', label='Deep BSDE', linewidth=2)
        ax.set_xlabel('x')
        ax.set_ylabel('v(x)  [normalised: v(0)=0]')
        ax.set_title('Ergodic Value Function v(x): PDE vs Deep BSDE (d=1)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xlim(-4, 4)

        plt.tight_layout()
        fig_path = 'experiments/figures/deep_vs_classical_v.png'
        plt.savefig(fig_path, dpi=100)
        plt.close(fig)
        print(f"  Value function comparison figure saved to: {fig_path}\n")

except Exception as exc:
    print(f"  Figure FAILED: {exc}\n")

print("Done.")
