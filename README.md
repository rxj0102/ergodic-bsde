# ergodic-bsde

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

A research-grade Python library for numerical solvers of **ergodic backward stochastic differential equations (BSDEs)** and their applications to long-run risk pricing, robust portfolio optimization, and risk-sensitive control.

---

## What is an Ergodic BSDE?

A standard finite-horizon BSDE on `[0, T]`:

```
Y_t = ξ + ∫_t^T f(s, Y_s, Z_s) ds − ∫_t^T Z_s · dW_s
```

As `T → ∞` the terminal condition `ξ` vanishes, and one seeks a triple **(λ, v, z)** where **λ** is a scalar *ergodic constant* and `(v, z)` is a stationary solution:

```
Y_t = Y_T + ∫_t^T [f(X_s, Y_s, Z_s) − λ] ds − ∫_t^T Z_s · dW_s   for all T > t
```

The ergodic constant `λ` is unique and satisfies the stationary (ergodic) PDE:

```
Lv(x) + f(x, v(x), σ(x)∇v(x)) = λ
```

where `L = (σ²/2)∂²/∂x² + b·∂/∂x` is the generator of the forward process.

---

## Quickstart

```python
import numpy as np
from ebsde.forward.ou_process import OrnsteinUhlenbeck
from ebsde.bsde.ergodic import ErgodicBSDE
from ebsde.solvers.ergodic_pde import ErgodicPDESolver

# OU forward process: dX = -κ(X − θ) dt + σ dW
ou = OrnsteinUhlenbeck(kappa=1.0, theta=0.0, sigma=1.0)

# Ergodic BSDE with quadratic driver: f(x, y, z) = x² − (1/2)z²
ebsde = ErgodicBSDE(
    forward=ou,
    driver=lambda x, y, z: x**2 - 0.5 * np.sum(z**2, axis=-1),
)

# Solve with the PDE solver (fastest for d=1)
result = ErgodicPDESolver(ebsde).solve()
print(f"Ergodic constant λ = {result['lambda_ergodic']:.4f}")  # ≈ 0.3660
# v(x) — the value function (normalised to E_π[v] = 0)
print(f"v(0) = {np.interp(0.0, result['x_grid'], result['v']):.4f}")
```

---

## Solvers

| Solver | Method | Best for | Complexity |
|--------|--------|----------|------------|
| `ErgodicPDESolver` | Finite-difference eigenvalue | d=1, high accuracy | O(n_x) |
| `ErgodicPicardSolver` | Long-horizon MC + Richardson | Any d, reference solution | O(n_paths·T·n_steps) |
| `ErgodicDeepBSDESolver` | Neural network (3 strategies) | d≥2, scalable | O(n_epochs·n_samples) |
| `PDEBSDESolver` | Crank-Nicolson PDE | Finite-horizon, d=1 | O(n_x·n_t) |
| `PicardBSDESolver` | Picard iteration + regression | Finite-horizon, any d | O(n_paths·n_steps) |
| `DeepBSDESolver` | Han-Jentzen-E (2018) | Finite-horizon, d≥4 | O(n_epochs) |

### ErgodicDeepBSDESolver strategies

| Strategy | Description | When to use |
|----------|-------------|-------------|
| `temporal_difference` | Continuous-time PDE residual via autograd (z=σ∇v) | Accurate, z derived from v |
| `direct_ergodic` | Minimise E[(Lv+f−λ)²] with independent z-network | Flexible, explicit z(x) |
| `long_horizon` | Regress λ from long trajectories | Simple baseline |

---

## Applications

| Class | Financial problem | Key output |
|-------|-------------------|------------|
| `LongRunRiskPricer` | Bansal-Yaron long-run risk (Epstein-Zin) | Risk-adjusted discount rate λ |
| `RobustPricer` | Model uncertainty (Hansen-Sargent minimax) | Worst-case long-run yield |
| `RiskSensitiveOptimizer` | Merton risk-sensitive portfolio | Optimal growth rate + hedging demand |
| `PrincipalEigenvalueSolver` | Generator eigenvalue, Hansen-Scheinkman decomposition | λ, eigenfunction φ, martingale L_T |

```python
# Long-run risk pricing
from ebsde.applications.long_run_risk import LongRunRiskPricer
pricer = LongRunRiskPricer(gamma=10.0, psi=1.5, delta=0.02)
res = pricer.compute_risk_adjusted_rate()
print(f"Long-run rate: {res['lambda']:.4f}, equity premium: {res['equity_premium']:.4f}")

# Robust pricing
from ebsde.applications.robust_pricing import RobustPricer
rp = RobustPricer(forward=ou, running_payoff=lambda x: x**2, uncertainty_penalty=1.0)
res = rp.price_robust()
print(f"Robust yield: {res['lambda_robust']:.4f}  (physical: {res['lambda_physical']:.4f})")

# Risk-sensitive optimization
from ebsde.applications.risk_sensitive import RiskSensitiveOptimizer
opt = RiskSensitiveOptimizer(market_model=ou, risk_aversion=2.0)
res = opt.compute_optimal_policy()
print(f"Growth rate: {res['lambda']:.4f},  CER: {res['certainty_equivalent_rate']:.2f}%")
```

---

## Library Structure

```
ebsde/
├── forward/
│   ├── sde.py              # ForwardSDE abstract base class
│   ├── ou_process.py       # OrnsteinUhlenbeck (exact simulation, stationary dist.)
│   ├── cev_process.py      # CEVProcess
│   └── multidimensional.py # MultiDimOU
├── bsde/
│   ├── standard.py         # StandardBSDE (finite-horizon)
│   ├── ergodic.py          # ErgodicBSDE
│   ├── drivers.py          # linear_driver, quadratic_driver, entropy_driver
│   └── markovian.py        # MarkovianBSDEConnection
├── solvers/
│   ├── pde.py              # PDEBSDESolver (Crank-Nicolson + Newton)
│   ├── picard.py           # PicardBSDESolver (regression-based)
│   ├── regression.py       # RegressionBSDESolver
│   ├── deep_bsde.py        # DeepBSDESolver (Han-Jentzen-E 2018)
│   ├── ergodic_pde.py      # ErgodicPDESolver (FD eigenvalue)
│   ├── ergodic_picard.py   # ErgodicPicardSolver (long-horizon MC)
│   └── ergodic_deep.py     # ErgodicDeepBSDESolver (3 strategies)
├── applications/
│   ├── long_run_risk.py    # LongRunRiskPricer (Bansal-Yaron)
│   ├── robust_pricing.py   # RobustPricer (Hansen-Sargent)
│   ├── risk_sensitive.py   # RiskSensitiveOptimizer (Merton)
│   └── principal_eigenvalue.py  # PrincipalEigenvalueSolver (Hansen-Scheinkman)
└── analysis/
    ├── convergence.py      # ConvergenceDiagnostics, SolverComparison
    └── ergodic_constant.py # ErgodicConstantAnalysis (sensitivity, bootstrap CI)

experiments/
├── run_standard_bsde.py        # Validate standard BSDE solvers
├── run_ergodic_convergence.py  # Convergence study for ergodic solvers
├── run_solver_comparison.py    # Cross-solver comparison table
├── run_long_run_risk.py        # Long-run risk application
├── run_robust_pricing.py       # Robust pricing under uncertainty
└── run_deep_vs_classical.py    # Deep vs PDE comparison

notebooks/
├── 01_bsde_primer.ipynb        # What is a BSDE? Feynman-Kac
├── 02_ergodic_bsde.ipynb       # Ergodic constant emergence as T → ∞
├── 03_pde_connection.ipynb     # PDE view, grid convergence
├── 04_deep_bsde.ipynb          # Deep BSDE training walkthrough
└── 05_long_run_risk.ipynb      # Long-run risk asset pricing

docs/
├── math_background.md          # Mathematical foundations
├── numerical_methods.md        # Solver algorithms and convergence
├── ergodic_theory.md           # Ergodicity, spectral gap, mixing
└── applications_guide.md       # Financial applications how-to
```

---

## Installation

```bash
git clone https://github.com/rxj0102/ergodic-bsde
cd ergodic-bsde
pip install -e ".[dev]"
```

**Dependencies**: `numpy`, `scipy`, `pandas`, `torch`, `matplotlib`, `seaborn`, `pytest`

---

## Running Experiments

```bash
cd ergodic-bsde

# Validate standard BSDE solvers
python experiments/run_standard_bsde.py

# Ergodic convergence study (OU + quadratic, known λ ≈ 0.366)
python experiments/run_ergodic_convergence.py

# Full solver comparison table
python experiments/run_solver_comparison.py

# Long-run risk pricing application
python experiments/run_long_run_risk.py

# Robust pricing under model uncertainty
python experiments/run_robust_pricing.py

# Deep BSDE vs classical (PDE) comparison
python experiments/run_deep_vs_classical.py
```

Figures are saved to `experiments/figures/`.

---

## Running Tests

```bash
pytest tests/ -v          # all 218 tests
pytest tests/ -q          # compact output
pytest tests/test_ergodic_deep.py -v   # deep solver only
```

---

## Documentation

- **[Math Background](docs/math_background.md)**: BSDEs, ergodic BSDEs, Feynman-Kac, Cole-Hopf transform, Donsker-Varadhan, Hansen-Scheinkman
- **[Numerical Methods](docs/numerical_methods.md)**: FD/PDE solver, deep BSDE architecture, convergence rates, practical recommendations
- **[Ergodic Theory](docs/ergodic_theory.md)**: Lyapunov conditions, spectral gap, mixing time, exponential convergence
- **[Applications Guide](docs/applications_guide.md)**: Step-by-step guide for each financial application class

---

## Analytical Benchmark

For the OU process (`κ=σ=1`) with quadratic driver `f(x,y,z) = αx² − (γ/2)|z|²`, the ergodic constant has the closed-form:

```
λ = (1/γ) · λ_w      where λ_w = (1/2)[−κ + √(κ² + 2γασ²)]
```

For `κ=σ=α=γ=1`: `λ = (√3 − 1)/2 ≈ 0.36603`.

| Solver | λ estimate | Error | Time |
|--------|-----------|-------|------|
| PDE (n_x=500) | 0.3660 | < 0.001 | ~1s |
| Picard (n=50k, T=[20,40]) | 0.366 ± 0.002 | < 0.5% | ~60s |
| Deep TD (n_epochs=3000) | 0.366 ± 0.005 | < 1% | ~90s |

---

## References

- **Pardoux & Peng (1990)**: Adapted solution of a backward stochastic differential equation. *Systems & Control Letters*, 14(1), 55–61.
- **El Karoui, Peng & Quenez (1997)**: Backward stochastic differential equations in finance. *Mathematical Finance*, 7(1), 1–71.
- **Fuhrman, Hu & Tessitore (2009)**: Ergodic BSDEs and optimal ergodic control in Banach spaces. *SIAM J. Control Optim.*, 48(3), 1542–1572.
- **Debussche, Hu & Tessitore (2011)**: Ergodic BSDEs under weak dissipative assumptions. *Stochastic Processes and their Applications*, 121(3), 407–426.
- **Hansen & Scheinkman (2009)**: Long-term risk: An operator approach. *Econometrica*, 77(1), 177–234.
- **Bansal & Yaron (2004)**: Risks for the long run: A potential resolution of asset pricing puzzles. *Journal of Finance*, 59(4), 1481–1509.
- **Han, Jentzen & E (2018)**: Solving high-dimensional partial differential equations using deep learning. *PNAS*, 115(34), 8505–8510.
