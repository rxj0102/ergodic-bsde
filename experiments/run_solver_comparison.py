"""
Experiment 3: Systematic solver comparison using SolverComparison.

Tests two ergodic BSDE problems:
  1. OU + quadratic driver f(x,y,z) = x^2 - (1/2)z^2   (lambda ~ 0.366)
  2. OU + linear driver   f(x,y,z) = x^2               (lambda ~ 0.5)

Uses SolverComparison().compare_all_solvers() for each problem and prints
a combined DataFrame.
"""

import sys
import os
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from ebsde.forward.ou_process import OrnsteinUhlenbeck
from ebsde.bsde.ergodic import ErgodicBSDE
from ebsde.analysis.convergence import SolverComparison
from data.synthetic import ergodic_ou_quadratic_analytical, ergodic_linear_eigenvalue

os.makedirs('experiments/figures', exist_ok=True)

print("=== Experiment 3: Solver Comparison (PDE vs Picard vs Deep) ===\n")

ou = OrnsteinUhlenbeck(kappa=1.0, theta=0.0, sigma=1.0)
RNG_SEED = 42
TORCH_SEED = 42

# ---------------------------------------------------------------
# Problem 1: Quadratic driver  f = x^2 - (1/2)z^2
# ---------------------------------------------------------------
def quadratic_driver(x, y, z):
    x = np.asarray(x, dtype=float)
    z = np.asarray(z, dtype=float)
    z_sq = np.sum(z**2, axis=-1) if z.ndim >= 2 else z.ravel()**2
    return x**2 - 0.5 * z_sq

ebsde_quad = ErgodicBSDE(forward=ou, driver=quadratic_driver)

ref_quad = ergodic_ou_quadratic_analytical(kappa=1.0, sigma=1.0, alpha=1.0, beta=0.0, gamma=1.0)
lambda_quad = ref_quad['lambda_ergodic']
print(f"Problem 1: Quadratic driver  (lambda_exact = {lambda_quad:.6f})")

# ---------------------------------------------------------------
# Problem 2: Linear driver f = x^2  (no y, no z dependence)
# ---------------------------------------------------------------
def linear_driver_x2(x, y, z):
    x = np.asarray(x, dtype=float)
    return x**2

ebsde_linear = ErgodicBSDE(forward=ou, driver=linear_driver_x2)

# Correct reference: λ = E_π[x²] = σ²/(2κ) = 0.5 for OU(κ=1,σ=1)
# (Poisson equation solvability: λ = ∫h dπ when h has no y/z terms)
lambda_linear = 0.5
print(f"Problem 2: Linear driver f=x^2  (lambda_ref  = {lambda_linear:.6f},  = E_π[x²])\n")

# ---------------------------------------------------------------
# Solver kwargs (small enough for speed)
# ---------------------------------------------------------------
pde_kwargs = dict(n_x=300, method='iteration')
picard_kwargs = dict(
    T_values=[5.0, 10.0],
    n_paths=10000,
    n_steps_per_unit=20,
    n_picard=4,
    rng_seed=RNG_SEED,
)
deep_kwargs = dict(
    strategy='direct_ergodic',
    n_epochs=500,
    n_samples=512,
    rng_seed=RNG_SEED,
    torch_seed=TORCH_SEED,
)

sc = SolverComparison()

all_frames = []

print("--- Running comparison for Problem 1 (quadratic driver) ---")
try:
    df1 = sc.compare_all_solvers(
        ergodic_bsde=ebsde_quad,
        analytical_lambda=lambda_quad,
        pde_kwargs=pde_kwargs,
        picard_kwargs=picard_kwargs,
        deep_kwargs=deep_kwargs,
    )
    df1.insert(0, 'problem', 'quadratic_f=x2-0.5z2')
    df1.insert(1, 'lambda_exact', round(lambda_quad, 6))
    all_frames.append(df1)
    print(df1.to_string(index=False))
    print()
except Exception as exc:
    print(f"  FAILED: {exc}\n")

print("--- Running comparison for Problem 2 (linear driver f=x^2) ---")
try:
    df2 = sc.compare_all_solvers(
        ergodic_bsde=ebsde_linear,
        analytical_lambda=lambda_linear,
        pde_kwargs=pde_kwargs,
        picard_kwargs=picard_kwargs,
        deep_kwargs=deep_kwargs,
    )
    df2.insert(0, 'problem', 'linear_f=x2')
    df2.insert(1, 'lambda_exact', round(lambda_linear, 6))
    all_frames.append(df2)
    print(df2.to_string(index=False))
    print()
except Exception as exc:
    print(f"  FAILED: {exc}\n")

# Combined DataFrame
if all_frames:
    combined = pd.concat(all_frames, ignore_index=True)
    print("--- Combined Results ---\n")
    print(combined.to_string(index=False))
    print()

print("Done.")
