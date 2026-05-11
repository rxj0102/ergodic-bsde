# ergodic-bsde

A research-grade Python library for numerical solvers of **ergodic backward stochastic differential equations (BSDEs)** and their applications to long-run risk pricing, robust portfolio optimization, and pricing under model uncertainty.

## What is an Ergodic BSDE?

A standard finite-horizon BSDE on [0, T]:

```
Y_t = ξ + ∫_t^T f(s, Y_s, Z_s) ds - ∫_t^T Z_s dW_s
```

As T → ∞ the terminal condition vanishes, and one seeks a triple **(λ, Y, Z)** where **λ** is a scalar ergodic constant and (Y, Z) is a stationary solution:

```
Y_t = Y_T + ∫_t^T [f(X_s, Y_s, Z_s) - λ] ds - ∫_t^T Z_s dW_s   for all T > t
```

## Financial Interpretation

| Context | λ means |
|---|---|
| Long-run portfolio optimization | Optimal long-run growth rate |
| Robust pricing | Long-run yield under worst-case measure |
| Risk-sensitive control | Risk-adjusted growth rate |
| Large deviations | Rate function of portfolio value |

## Installation

```bash
pip install -e ".[dev]"
```

## Library Structure

```
ebsde/
├── forward/        # Forward SDE simulation (OU, CEV, Multi-dim OU)
├── bsde/           # BSDE specifications and drivers
├── solvers/        # Picard, regression, PDE, deep BSDE solvers
├── applications/   # Long-run risk, robust pricing, risk-sensitive control
└── analysis/       # Convergence diagnostics, ergodic constant estimation
```

## Quick Start

```python
import numpy as np
from ebsde.forward.ou_process import OrnsteinUhlenbeck
from ebsde.bsde.ergodic import ErgodicBSDE
from ebsde.bsde.drivers import quadratic_driver

# OU forward process: dX = -κ(X - θ) dt + σ dW
ou = OrnsteinUhlenbeck(kappa=1.0, theta=0.0, sigma=1.0)

# Ergodic BSDE with quadratic driver: f(x, y, z) = x² - (γ/2)z²
gamma = 2.0
driver = lambda x, y, z: x**2 - (gamma / 2) * np.sum(z**2, axis=-1)

ebsde = ErgodicBSDE(forward=ou, driver=driver)

# Simulate forward paths
sim = ou.simulate(x0=np.array([0.0]), T=10.0, n_steps=1000, n_paths=512)
```

## Key References

- Fuhrman, Hu & Tessitore (2009): Ergodic BSDEs and optimal ergodic control
- Debussche, Hu & Tessitore (2011): Ergodic BSDEs under weak dissipative assumptions
- Han, Jentzen & E (2018): Solving high-dimensional PDEs using deep learning
- Hansen & Scheinkman (2009): Long-term risk: An operator approach
- Bansal & Yaron (2004): Risks for the long run
