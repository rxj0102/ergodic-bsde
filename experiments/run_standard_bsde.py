"""
Experiment 1: Validate standard BSDE solvers (PDE, Picard, Regression).

Tests:
  - Linear BSDE:   f(x,y,z) = -y,  g(x) = x,  T=1.0, OU forward
  - Quadratic BSDE: f(x,y,z) = x^2 - (1/2)z^2, g=0, via Cole-Hopf reference

Prints a formatted error table: solver, Y0_estimate, Y0_reference, abs_error, time_s
"""

import sys
import os
import time

# No figures needed for this script, but keep consistent import style
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import numpy as np

# Add repo root to path so imports work when run from any directory
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from ebsde.forward.ou_process import OrnsteinUhlenbeck
from ebsde.bsde.standard import StandardBSDE
from ebsde.solvers.pde import PDEBSDESolver
from ebsde.solvers.picard import PicardBSDESolver
from ebsde.solvers.regression import RegressionBSDESolver
from data.synthetic import linear_bsde_analytical, quadratic_bsde_cole_hopf

os.makedirs('experiments/figures', exist_ok=True)

print("=== Experiment 1: Standard BSDE Solver Validation ===\n")

# ---------------------------------------------------------------
# Forward process: OU(kappa=1, theta=0, sigma=1)
# ---------------------------------------------------------------
ou = OrnsteinUhlenbeck(kappa=1.0, theta=0.0, sigma=1.0)
T = 1.0
RNG_SEED = 42

# ---------------------------------------------------------------
# Problem A: Linear BSDE   f(x,y,z) = -y,  g(x) = x
# ---------------------------------------------------------------
print("--- Problem A: Linear BSDE  f(x,y,z) = -y,  g(x)=x,  T=1.0 ---\n")

def linear_driver(x, y, z):
    return -np.asarray(y, dtype=float)

def linear_terminal(x):
    return np.asarray(x, dtype=float).ravel()

bsde_linear = StandardBSDE(
    forward=ou,
    driver=linear_driver,
    terminal=linear_terminal,
    T=T,
)

# Analytical reference
ref_linear = linear_bsde_analytical(ou, a=0.0, b_coeff=-1.0, T=T,
                                     terminal_func=lambda x: np.asarray(x, dtype=float))
Y0_ref_linear = ref_linear['Y0_at_theta']
print(f"  Analytical Y0 (at x=theta=0): {Y0_ref_linear:.6f}\n")

rows_linear = []

# PDE solver
try:
    t0 = time.perf_counter()
    sol = PDEBSDESolver(bsde_linear, n_x=200, n_t=500).solve(x0=0.0)
    elapsed = time.perf_counter() - t0
    rows_linear.append(('PDE', sol['Y0'], Y0_ref_linear, abs(sol['Y0'] - Y0_ref_linear), elapsed))
except Exception as exc:
    print(f"  PDE FAILED: {exc}")
    rows_linear.append(('PDE', float('nan'), Y0_ref_linear, float('nan'), float('nan')))

# Picard solver
try:
    t0 = time.perf_counter()
    sol = PicardBSDESolver(bsde_linear, n_paths=20000, n_steps=100,
                            n_picard=5, rng_seed=RNG_SEED).solve()
    elapsed = time.perf_counter() - t0
    rows_linear.append(('Picard', sol['Y0'], Y0_ref_linear, abs(sol['Y0'] - Y0_ref_linear), elapsed))
except Exception as exc:
    print(f"  Picard FAILED: {exc}")
    rows_linear.append(('Picard', float('nan'), Y0_ref_linear, float('nan'), float('nan')))

# Regression solver
try:
    t0 = time.perf_counter()
    sol = RegressionBSDESolver(bsde_linear, n_paths=20000, n_steps=100,
                                rng_seed=RNG_SEED).solve()
    elapsed = time.perf_counter() - t0
    rows_linear.append(('Regression', sol['Y0'], Y0_ref_linear, abs(sol['Y0'] - Y0_ref_linear), elapsed))
except Exception as exc:
    print(f"  Regression FAILED: {exc}")
    rows_linear.append(('Regression', float('nan'), Y0_ref_linear, float('nan'), float('nan')))

# Print table
header = f"{'Solver':<12} {'Y0_estimate':>14} {'Y0_reference':>14} {'abs_error':>12} {'time_s':>10}"
sep = '-' * len(header)
print(header)
print(sep)
for solver, Y0_est, Y0_ref, err, t_s in rows_linear:
    print(f"{solver:<12} {Y0_est:>14.6f} {Y0_ref:>14.6f} {err:>12.2e} {t_s:>10.3f}")
print()

# ---------------------------------------------------------------
# Problem B: Quadratic BSDE  f(x,y,z) = x^2 - (1/2)z^2,  g=0
# ---------------------------------------------------------------
print("--- Problem B: Quadratic BSDE  f(x,y,z) = x^2 - 0.5*z^2,  g=0,  T=1.0 ---\n")

def quad_driver(x, y, z):
    x = np.asarray(x, dtype=float)
    z = np.asarray(z, dtype=float)
    z_sq = np.sum(z**2, axis=-1) if z.ndim >= 2 else z.ravel()**2
    return x**2 - 0.5 * z_sq

def zero_terminal(x):
    return np.zeros(np.asarray(x).shape[0] if np.ndim(x) > 0 else 1)

bsde_quad = StandardBSDE(
    forward=ou,
    driver=quad_driver,
    terminal=zero_terminal,
    T=T,
)

# Cole-Hopf reference: f = h(x) - (gamma/2)|z|^2 with h(x)=x^2, gamma=1
try:
    ref_quad = quadratic_bsde_cole_hopf(ou, h_func=lambda x: x**2, gamma=1.0, T=T, n_quad=1000)
    Y0_ref_quad = ref_quad['Y0_at_theta']
    print(f"  Cole-Hopf reference Y0 (at x=theta=0): {Y0_ref_quad:.6f}\n")
except Exception as exc:
    print(f"  Cole-Hopf reference FAILED: {exc}")
    Y0_ref_quad = float('nan')

rows_quad = []

# PDE solver
try:
    t0 = time.perf_counter()
    sol = PDEBSDESolver(bsde_quad, n_x=200, n_t=500).solve(x0=0.0)
    elapsed = time.perf_counter() - t0
    rows_quad.append(('PDE', sol['Y0'], Y0_ref_quad, abs(sol['Y0'] - Y0_ref_quad), elapsed))
except Exception as exc:
    print(f"  PDE FAILED: {exc}")
    rows_quad.append(('PDE', float('nan'), Y0_ref_quad, float('nan'), float('nan')))

# Picard solver
try:
    t0 = time.perf_counter()
    sol = PicardBSDESolver(bsde_quad, n_paths=20000, n_steps=100,
                            n_picard=5, rng_seed=RNG_SEED).solve()
    elapsed = time.perf_counter() - t0
    rows_quad.append(('Picard', sol['Y0'], Y0_ref_quad, abs(sol['Y0'] - Y0_ref_quad), elapsed))
except Exception as exc:
    print(f"  Picard FAILED: {exc}")
    rows_quad.append(('Picard', float('nan'), Y0_ref_quad, float('nan'), float('nan')))

# Regression solver
try:
    t0 = time.perf_counter()
    sol = RegressionBSDESolver(bsde_quad, n_paths=20000, n_steps=100,
                                rng_seed=RNG_SEED).solve()
    elapsed = time.perf_counter() - t0
    rows_quad.append(('Regression', sol['Y0'], Y0_ref_quad, abs(sol['Y0'] - Y0_ref_quad), elapsed))
except Exception as exc:
    print(f"  Regression FAILED: {exc}")
    rows_quad.append(('Regression', float('nan'), Y0_ref_quad, float('nan'), float('nan')))

header = f"{'Solver':<12} {'Y0_estimate':>14} {'Y0_reference':>14} {'abs_error':>12} {'time_s':>10}"
sep = '-' * len(header)
print(header)
print(sep)
for solver, Y0_est, Y0_ref, err, t_s in rows_quad:
    print(f"{solver:<12} {Y0_est:>14.6f} {Y0_ref:>14.6f} {err:>12.2e} {t_s:>10.3f}")
print()

print("Done.")
