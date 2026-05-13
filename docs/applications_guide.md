# Applications Guide

How to map financial problems to ergodic BSDEs using the `ebsde` library.

---

## Mapping Financial Problems to Ergodic BSDEs

An ergodic BSDE requires three ingredients:

1. **A forward process** `X_t` that is ergodic (positive-recurrent diffusion)
2. **A driver** `f(x, y, z)` encoding preferences or payoff structure
3. **The ergodic constant** `λ` — the long-run rate you want to compute

The solution triple `(λ, v, z)` satisfies:

```
Lv(x) + f(x, v(x), σ(x)∇v(x)) = λ      (ergodic PDE)
```

where `L = (σ²/2)∂²/∂x² + b·∂/∂x` is the generator of `X_t`.

---

## Parameter Interpretation Table

| Parameter | Symbol | Financial meaning | Typical range |
|-----------|--------|-------------------|---------------|
| Mean-reversion speed | `κ` | Speed at which consumption/volatility reverts to mean | 0.1–5 |
| Long-run mean | `θ` | Unconditional mean of state variable | 0–0.05 |
| State vol | `σ` | Volatility of the state variable | 0.01–1.0 |
| Risk aversion | `γ` | CRRA risk aversion coefficient | 1–20 |
| EIS | `ψ` | Elasticity of intertemporal substitution | 0.5–3.0 |
| Discount rate | `δ` | Subjective time preference | 0.01–0.05 |
| Uncertainty penalty | `η` | Ambiguity tolerance (∞ = full trust in model) | 0.1–100 |
| Portfolio weight | `π` | Fraction of wealth in risky asset | unconstrained |

---

## Long-Run Risk Pricing

**Problem**: Find the long-run risk-adjusted rate `λ` for an agent with
Epstein-Zin preferences facing a stochastic consumption process.

**Setup**:
```python
from ebsde.forward.ou_process import OrnsteinUhlenbeck
from ebsde.applications.long_run_risk import LongRunRiskPricer

# Consumption state: OU proxy for log-consumption deviations
ou = OrnsteinUhlenbeck(kappa=1.0, theta=0.0, sigma=0.1)

pricer = LongRunRiskPricer(
    gamma=10.0,      # risk aversion
    psi=1.5,         # EIS > 1: substitution effect dominates
    delta=0.02,      # 2% annual discount rate
    consumption_model=ou,
)
```

**Driver structure** (log-linearised around stationary mean):
```
f(x, v, z) = δ·θ_ez·x  +  (γ/2)|z|²
```
where `θ_ez = (1-γ)/(1-1/ψ)`.

**Run**:
```python
result = pricer.compute_risk_adjusted_rate()
print(f"Long-run rate λ = {result['lambda']:.4f}")
print(f"Risk premium     = {result['risk_premium']:.4f}")
print(f"Equity premium ≈ {result['equity_premium']:.2%}")
```

**Output interpretation**:
- `lambda`: the risk-adjusted long-run discount rate. Higher γ → higher λ.
- `risk_premium`: `λ - E_π[f(X,0,0)]` — the contribution of risk to the long-run rate.
- `equity_premium`: rough estimate `γ·Var(X)`, in units of the state variable.

**Term structure**:
```python
import numpy as np
ts = pricer.term_structure_of_risk(np.array([0.5, 1.0, 2.0, 5.0]))
# ts['yields'] converges to ts['lambda_limit'] = λ as T → ∞
```

**Sensitivity analysis**:
```python
df = pricer.sensitivity_analysis('gamma', np.array([2.0, 5.0, 8.0]))
# Returns DataFrame with ['gamma', 'lambda'] — λ increases with γ
```

---

## Robust Pricing Under Model Uncertainty

**Problem**: Find the worst-case long-run yield when the agent distrusts
the model drift. The penalty `η` controls how far the adversary can deviate.

**Setup**:
```python
from ebsde.applications.robust_pricing import RobustPricer

ou = OrnsteinUhlenbeck(kappa=1.0, theta=0.0, sigma=1.0)
h = lambda x: x**2   # running payoff rate

pricer = RobustPricer(
    forward=ou,
    running_payoff=h,
    uncertainty_penalty=1.0,   # η: larger = less uncertainty aversion
)
```

**Driver** (entropic / BSDE with penalty):
```
f(x, y, z) = h(x)  +  (1/2η)|z|²
```

**Run**:
```python
result = pricer.price_robust()
print(f"Worst-case λ      = {result['lambda_robust']:.4f}")
print(f"Physical E[h]     = {result['lambda_physical']:.4f}")
print(f"Uncertainty prem  = {result['uncertainty_premium']:.4f}")
```

**Output interpretation**:
- `lambda_robust`: conservative (worst-case) long-run yield. Always ≥ `lambda_physical`.
- `lambda_physical`: `E_π[h(X)]` under the physical measure — the risk-neutral rate.
- `uncertainty_premium`: extra yield demanded due to model uncertainty.
- `worst_case_drift`: the adversarial drift perturbation `η^{-1}σ(x)∂v/∂x`.

**Sensitivity to η**:
```python
etas = np.array([0.25, 0.5, 1.0, 2.0, 5.0, 10.0])
df = pricer.uncertainty_sensitivity(etas)
# df['lambda_robust'] is strictly decreasing in eta
# df['lambda_robust'] → E_π[h] as eta → ∞
```

**Choosing η**:
- Small η (0.1–0.5): high uncertainty aversion, large premium over physical rate
- Medium η (1–5): moderate caution, suitable for calibrated uncertainty sets
- Large η (10–∞): full trust in model, approaches risk-neutral pricing

---

## Risk-Sensitive Portfolio Optimization

**Problem**: Maximise the long-run risk-sensitive growth rate
`J(π) = lim (1/T)(-2/γ) log E[exp(-γ/2 ∫ π_s(μ_s ds + σ_s dW_s))]`.

**Setup**:
```python
from ebsde.applications.risk_sensitive import RiskSensitiveOptimizer

ou = OrnsteinUhlenbeck(kappa=1.0, theta=0.0, sigma=1.0)

optimizer = RiskSensitiveOptimizer(
    market_model=ou,
    risk_aversion=2.0,
    mu_func=lambda x: x,           # Sharpe ratio = state variable
    sigma_func=lambda x: np.ones_like(x),  # constant asset vol
)
```

**Optimised driver** (after maximising over π):
```
f*(x, v, z) = μ(x)²/(2γσ_a²)  -  z·σ_state·μ/σ_a  +  (γ/2)z²σ_state²
```

**Run**:
```python
result = optimizer.compute_optimal_policy()
print(f"Growth rate λ    = {result['lambda']:.4f}")
print(f"CER (annualised) = {result['certainty_equivalent_rate']:.2f}%")
```

**Portfolio functions** (callable on any state grid):
```python
x_test = np.linspace(-3, 3, 100)
pi_star  = result['optimal_portfolio'](x_test)   # full Merton portfolio
pi_myopic = result['myopic_portfolio'](x_test)   # μ/(γσ²) — no hedging
hd = result['hedging_demand'](x_test)             # pi_star - pi_myopic
```

**Hedging demand interpretation**:
When `σ_state > 0` (stochastic opportunity set), the optimal portfolio
differs from the myopic `μ/(γσ_a²)` by the inter-temporal hedging demand
`-σ_state·∂v/∂x / σ_a`. This is the Merton (1973) hedging term: the
agent tilts the portfolio to hedge against changes in future investment
opportunities.

- Positive hedging demand: agent over-invests to hedge against bad future states
- Zero hedging demand: constant investment opportunity set (CAPM)

---

## Principal Eigenvalue and Hansen-Scheinkman

**Problem**: For a linear driver `f(x,y,z) = h(x) + c·y`, compute the
principal eigenvalue and decompose the pricing kernel.

```python
from ebsde.applications.principal_eigenvalue import PrincipalEigenvalueSolver
import numpy as np

ou = OrnsteinUhlenbeck(kappa=1.0, theta=0.0, sigma=1.0)
solver = PrincipalEigenvalueSolver(forward=ou)

# Principal eigenvalue via FD
stat = ou.stationary_distribution()
grid = np.linspace(-4, 4, 300)
h_func = lambda x: 0.5 * x**2

result = solver.compute_principal_eigenvalue((h_func, 0.0), grid)
print(f"Principal eigenvalue λ = {result['lambda']:.4f}")

# Donsker-Varadhan bounds
bounds = solver.donsker_varadhan_bound(h_func)
print(f"DV lower bound = {bounds['lower_bound']:.4f}")  # E_π[h]
print(f"DV upper bound = {bounds['upper_bound']:.4f}")  # sup h

# Hansen-Scheinkman decomposition
hs = solver.hansen_scheinkman_decomposition({
    "h_func": h_func, "c": 0.0, "grid": grid,
    "T": 5.0, "n_paths": 2000, "x0": 0.0, "rng_seed": 42,
})
print(f"E[L_T] ≈ {hs['L_T_mean']:.3f}")  # should be ≈ 1
```

**Decomposition**: The pricing kernel factorises as
`M_T = exp(-λT) · φ(X_T)/φ(X_0) · L_T`
where `L_T` is a martingale component and `φ` is the principal eigenfunction.
`E[L_T] ≈ 1` is a consistency check.

---

## Solver Recommendations

| Dimension | Problem type | Recommended solver | Typical settings |
|-----------|-------------|-------------------|-----------------|
| d = 1 | Any | `ErgodicPDESolver` | `n_x=500`, fast & accurate |
| d = 1 | Cross-validation | `ErgodicPicardSolver` | `n_paths=50000`, `T=[20,40]` |
| d ≥ 2 | Deep learning | `ErgodicDeepBSDESolver` | `strategy='direct_ergodic'`, `n_epochs=3000` |
| d = 1, linear | Eigenvalue | `PrincipalEigenvalueSolver` | `n_x=300` grid |

**Speed guide** (approximate, d=1, CPU):
- `ErgodicPDESolver(n_x=500)`: ~1 second
- `ErgodicPicardSolver(n_paths=50000, T=[20,40])`: ~30–120 seconds
- `ErgodicDeepBSDESolver(n_epochs=3000, n_samples=2048)`: ~60–200 seconds

---

## Troubleshooting

**λ = NaN from PDE solver**: driver has overflow. Clip `z` in the driver:
```python
driver = lambda x, y, z: h(x) + (c/2) * np.sum(np.clip(z, -5, 5)**2, axis=-1)
```

**λ not converging for Picard**: increase `T_values` and `n_paths`:
```python
solver = ErgodicPicardSolver(ebsde, T_values=[20, 40, 80], n_paths=100_000)
```

**Deep solver converges to wrong λ**: try `strategy='temporal_difference'` or
increase `n_epochs`:
```python
solver = ErgodicDeepBSDESolver(ebsde, strategy='temporal_difference',
                               n_epochs=5000, n_samples=4096, learning_rate=5e-4)
```

**Non-ergodic forward process**: `ErgodicBSDE` raises `ValueError`. Check
`forward.is_ergodic()` returns `True`. For OU this requires `kappa > 0`.
