"""
Experiment 5: Robust pricing under model uncertainty.

Setup:
  - Forward: OU(kappa=1, theta=0, sigma=1)
  - Running payoff: h(x) = x^2
  - Uncertainty penalty eta in [0.25, 0.5, 1.0, 2.0, 5.0]

For each eta:
  - Instantiate RobustPricer and call price_robust()
  - Record lambda_robust, lambda_physical, uncertainty_premium

Prints results table and saves lambda_robust vs eta plot.
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
from ebsde.applications.robust_pricing import RobustPricer

os.makedirs('experiments/figures', exist_ok=True)

print("=== Experiment 5: Robust Pricing Under Model Uncertainty ===\n")
print("  Forward: OU(kappa=1, theta=0, sigma=1)")
print("  Payoff:  h(x) = x^2")
print("  Driver:  f(x,y,z) = h(x) + (1/(2*eta))|z|^2\n")

# ---------------------------------------------------------------
# Setup
# ---------------------------------------------------------------
ou = OrnsteinUhlenbeck(kappa=1.0, theta=0.0, sigma=1.0)

def h(x):
    return np.asarray(x, dtype=float)**2

eta_values = [0.25, 0.5, 1.0, 2.0, 5.0]

# ---------------------------------------------------------------
# Run for each eta
# ---------------------------------------------------------------
rows = []
for eta in eta_values:
    try:
        t0 = time.perf_counter()
        pricer = RobustPricer(forward=ou, running_payoff=h, uncertainty_penalty=eta)
        result = pricer.price_robust(method='pde')
        elapsed = time.perf_counter() - t0

        rows.append((
            eta,
            result['lambda_robust'],
            result['lambda_physical'],
            result['uncertainty_premium'],
            elapsed,
            'ok',
        ))
    except Exception as exc:
        print(f"  eta={eta} FAILED: {exc}")
        rows.append((eta, float('nan'), float('nan'), float('nan'), float('nan'), f'FAILED'))

# ---------------------------------------------------------------
# Print table
# ---------------------------------------------------------------
print("Results:")
header = (f"{'eta':>8} {'lambda_robust':>16} {'lambda_physical':>18} "
          f"{'uncertainty_premium':>22} {'time_s':>10} {'status':>8}")
sep = '-' * len(header)
print(header)
print(sep)
for eta, lam_r, lam_p, uprem, t_s, status in rows:
    print(f"{eta:>8.2f} {lam_r:>16.6f} {lam_p:>18.6f} "
          f"{uprem:>22.6f} {t_s:>10.3f} {status:>8}")
print()

# ---------------------------------------------------------------
# Physical interpretation note
# ---------------------------------------------------------------
print("  Interpretation:")
print("  - eta -> inf  (no uncertainty): lambda_robust -> E_pi[h(X)] = lambda_physical")
print("  - eta -> 0    (max uncertainty): lambda_robust -> sup_x h(x) = worst-case")
print("  - Larger eta => less risk aversion => smaller uncertainty premium\n")

# ---------------------------------------------------------------
# Save plot
# ---------------------------------------------------------------
try:
    valid_rows = [(r[0], r[1], r[2], r[3]) for r in rows if r[5] == 'ok']
    if valid_rows:
        etas, lam_robsts, lam_phys, uprems = zip(*valid_rows)

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        # Left: lambda_robust vs eta
        axes[0].plot(etas, lam_robsts, 'o-', color='steelblue', linewidth=2, markersize=8,
                     label='lambda_robust')
        if lam_phys:
            axes[0].axhline(lam_phys[0], color='gray', linestyle='--',
                            label=f'lambda_physical={lam_phys[0]:.4f}')
        axes[0].set_xlabel('Uncertainty penalty eta')
        axes[0].set_ylabel('Ergodic constant lambda')
        axes[0].set_title('Robust Pricing: lambda_robust vs eta')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # Right: uncertainty premium vs eta
        axes[1].plot(etas, uprems, 's-', color='firebrick', linewidth=2, markersize=8)
        axes[1].set_xlabel('Uncertainty penalty eta')
        axes[1].set_ylabel('Uncertainty premium (lambda_robust - lambda_physical)')
        axes[1].set_title('Uncertainty Premium vs eta')
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        fig_path = 'experiments/figures/robust_pricing_eta.png'
        plt.savefig(fig_path, dpi=100)
        plt.close(fig)
        print(f"  Figure saved to: {fig_path}\n")

except Exception as exc:
    print(f"  Figure FAILED: {exc}\n")

print("Done.")
