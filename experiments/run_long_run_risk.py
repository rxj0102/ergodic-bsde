"""
Experiment 4: Long-run risk pricing via ergodic BSDEs.

Uses LongRunRiskPricer with Epstein-Zin preferences:
  gamma=5.0, psi=1.5, delta=0.02

Steps:
  1. Compute risk-adjusted rate: lambda, risk_premium, equity_premium
  2. Sensitivity analysis over gamma in [2.0, 5.0, 8.0]
  3. Term structure of risk for maturities [0.5, 1.0, 2.0]

Saves sensitivity plot to experiments/figures/long_run_risk_sensitivity.png
"""

import sys
import os
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from ebsde.applications.long_run_risk import LongRunRiskPricer

os.makedirs('experiments/figures', exist_ok=True)

print("=== Experiment 4: Long-Run Risk Pricing ===\n")

# ---------------------------------------------------------------
# Instantiate pricer
# ---------------------------------------------------------------
pricer = LongRunRiskPricer(gamma=5.0, psi=1.5, delta=0.02)
print(f"  LongRunRiskPricer(gamma=5.0, psi=1.5, delta=0.02)\n")
print(f"  theta_ez = (1-gamma)/(1-1/psi) = {pricer.theta_ez:.4f}\n")

# ---------------------------------------------------------------
# Step 1: Compute risk-adjusted rate
# ---------------------------------------------------------------
print("--- Step 1: Risk-Adjusted Rate ---\n")

try:
    t0 = time.perf_counter()
    result = pricer.compute_risk_adjusted_rate(method='pde')
    elapsed = time.perf_counter() - t0

    lam = result['lambda']
    risk_prem = result['risk_premium']
    eq_prem = result['equity_premium']

    print(f"  {'Quantity':<25} {'Value':>12}")
    print(f"  {'-'*38}")
    print(f"  {'lambda (ergodic rate)':<25} {lam:>12.6f}")
    print(f"  {'risk_premium':<25} {risk_prem:>12.6f}")
    print(f"  {'equity_premium (approx)':<25} {eq_prem:>12.6f}")
    print(f"  {'time_s':<25} {elapsed:>12.3f}")
    print()
except Exception as exc:
    print(f"  FAILED: {exc}\n")

# ---------------------------------------------------------------
# Step 2: Sensitivity analysis over gamma
# ---------------------------------------------------------------
print("--- Step 2: Sensitivity Analysis over gamma ---\n")

gamma_values = np.array([2.0, 5.0, 8.0])

try:
    t0 = time.perf_counter()
    df_sens = pricer.sensitivity_analysis('gamma', gamma_values)
    elapsed = time.perf_counter() - t0

    print(df_sens.to_string(index=False))
    print(f"\n  (computed in {elapsed:.3f}s)\n")

    # Save sensitivity figure
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(df_sens['gamma'], df_sens['lambda'], 'o-', color='steelblue', linewidth=2, markersize=8)
    ax.set_xlabel('Risk aversion gamma')
    ax.set_ylabel('Ergodic constant lambda')
    ax.set_title('Long-Run Risk: lambda vs Risk Aversion (gamma)')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig_path = 'experiments/figures/long_run_risk_sensitivity.png'
    plt.savefig(fig_path, dpi=100)
    plt.close(fig)
    print(f"  Figure saved to: {fig_path}\n")

except Exception as exc:
    print(f"  Sensitivity analysis FAILED: {exc}\n")

# ---------------------------------------------------------------
# Step 3: Term structure of risk
# ---------------------------------------------------------------
print("--- Step 3: Term Structure of Risk ---\n")

maturities = np.array([0.5, 1.0, 2.0])

try:
    t0 = time.perf_counter()
    ts_result = pricer.term_structure_of_risk(maturities)
    elapsed = time.perf_counter() - t0

    print(f"  {'Maturity T':>12} {'Yield y(T)':>14}")
    print(f"  {'-'*28}")
    for T, y in zip(ts_result['maturities'], ts_result['yields']):
        print(f"  {T:>12.2f} {y:>14.6f}")
    print(f"\n  lambda_limit (T->inf) = {ts_result['lambda_limit']:.6f}")
    print(f"  (computed in {elapsed:.3f}s)\n")

except Exception as exc:
    print(f"  Term structure FAILED: {exc}\n")

print("Done.")
