"""
Experiment 2: Ergodic BSDE convergence study.

Problem: OU(kappa=1, sigma=1, theta=0) + quadratic driver f(x,y,z) = x^2 - (1/2)z^2
Exact ergodic constant: lambda_exact = 0.36603...

Sweeps:
  - PDE solver:  n_x in [50, 100, 200, 500, 1000]
  - Picard solver: n_paths in [5000, 10000, 50000]
  - Deep solver: strategy='direct_ergodic', n_epochs=2000, n_samples=1024

Prints tables and saves training loss plot.
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
from ebsde.solvers.ergodic_picard import ErgodicPicardSolver
from ebsde.solvers.ergodic_deep import ErgodicDeepBSDESolver
from data.synthetic import ergodic_ou_quadratic_analytical

os.makedirs('experiments/figures', exist_ok=True)

print("=== Experiment 2: Ergodic BSDE Convergence Study ===\n")

# ---------------------------------------------------------------
# Problem setup
# ---------------------------------------------------------------
ou = OrnsteinUhlenbeck(kappa=1.0, theta=0.0, sigma=1.0)

def quadratic_driver(x, y, z):
    x = np.asarray(x, dtype=float)
    z = np.asarray(z, dtype=float)
    z_sq = np.sum(z**2, axis=-1) if z.ndim >= 2 else z.ravel()**2
    return x**2 - 0.5 * z_sq

ebsde = ErgodicBSDE(forward=ou, driver=quadratic_driver)

# Analytical reference: alpha=1, beta=0, gamma=1
ref = ergodic_ou_quadratic_analytical(kappa=1.0, sigma=1.0, alpha=1.0, beta=0.0, gamma=1.0)
lambda_exact = ref['lambda_ergodic']
print(f"  Analytical lambda_exact = {lambda_exact:.6f}\n")

RNG_SEED = 42
TORCH_SEED = 42

# ---------------------------------------------------------------
# Section 1: PDE solver convergence over grid sizes
# ---------------------------------------------------------------
print("--- PDE Solver: grid refinement ---\n")

nx_values = [50, 100, 200, 500, 1000]
rows_pde = []

for n_x in nx_values:
    try:
        t0 = time.perf_counter()
        sol = ErgodicPDESolver(ebsde, n_x=n_x, method='iteration').solve()
        elapsed = time.perf_counter() - t0
        lam = sol['lambda_ergodic']
        err = abs(lam - lambda_exact)
        rows_pde.append((n_x, lam, err, elapsed))
    except Exception as exc:
        print(f"  PDE n_x={n_x} FAILED: {exc}")
        rows_pde.append((n_x, float('nan'), float('nan'), float('nan')))

header = f"{'n_x':>8} {'lambda':>12} {'error':>12} {'time_s':>10}"
sep = '-' * len(header)
print(header)
print(sep)
for n_x, lam, err, t_s in rows_pde:
    print(f"{n_x:>8d} {lam:>12.6f} {err:>12.2e} {t_s:>10.3f}")
print()

# ---------------------------------------------------------------
# Section 2: Picard solver convergence over n_paths
# ---------------------------------------------------------------
print("--- Picard Solver: path count convergence ---\n")

n_paths_values = [5000, 10000, 50000]
rows_picard = []

for n_paths in n_paths_values:
    try:
        t0 = time.perf_counter()
        sol = ErgodicPicardSolver(
            ebsde,
            T_values=[5.0, 10.0],
            n_paths=n_paths,
            n_steps_per_unit=20,
            n_picard=4,
            rng_seed=RNG_SEED,
        ).solve()
        elapsed = time.perf_counter() - t0
        lam = sol['lambda_ergodic']
        err = abs(lam - lambda_exact)
        rows_picard.append((n_paths, lam, err, elapsed))
    except Exception as exc:
        print(f"  Picard n_paths={n_paths} FAILED: {exc}")
        rows_picard.append((n_paths, float('nan'), float('nan'), float('nan')))

header = f"{'n_paths':>10} {'lambda':>12} {'error':>12} {'time_s':>10}"
sep = '-' * len(header)
print(header)
print(sep)
for n_paths, lam, err, t_s in rows_picard:
    print(f"{n_paths:>10d} {lam:>12.6f} {err:>12.2e} {t_s:>10.3f}")
print()

# ---------------------------------------------------------------
# Section 3: Deep solver
# ---------------------------------------------------------------
print("--- Deep Solver: strategy='direct_ergodic', n_epochs=2000 ---\n")

try:
    t0 = time.perf_counter()
    deep_sol = ErgodicDeepBSDESolver(
        ebsde,
        strategy='direct_ergodic',
        n_epochs=2000,
        n_samples=1024,
        rng_seed=RNG_SEED,
        torch_seed=TORCH_SEED,
    ).solve()
    elapsed_deep = time.perf_counter() - t0
    lam_deep = deep_sol['lambda_ergodic']
    err_deep = abs(lam_deep - lambda_exact)

    header = f"{'method':<22} {'lambda':>12} {'error':>12} {'time_s':>10}"
    sep = '-' * len(header)
    print(header)
    print(sep)
    print(f"{'deep_direct_ergodic':<22} {lam_deep:>12.6f} {err_deep:>12.2e} {elapsed_deep:>10.3f}")
    print()

    # Save training loss figure
    loss_history = deep_sol.get('training_loss', [])
    lambda_history = deep_sol.get('lambda_history', [])

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    if loss_history:
        axes[0].semilogy(loss_history)
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss (log scale)')
        axes[0].set_title('Training Loss (direct_ergodic)')
        axes[0].grid(True, alpha=0.3)

    if lambda_history:
        axes[1].plot(lambda_history, label='lambda estimate')
        axes[1].axhline(lambda_exact, color='red', linestyle='--', label=f'exact={lambda_exact:.4f}')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('lambda')
        axes[1].set_title('Lambda Convergence During Training')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    fig_path = 'experiments/figures/ergodic_convergence_loss.png'
    plt.savefig(fig_path, dpi=100)
    plt.close(fig)
    print(f"  Figure saved to: {fig_path}")

except Exception as exc:
    print(f"  Deep solver FAILED: {exc}")

print()
print("Done.")
