"""
Create all 5 tutorial notebooks for the ergodic-bsde library using nbformat.
Run from the repo root: python scripts/create_notebooks.py
"""

import os
import nbformat

NOTEBOOKS_DIR = os.path.join(os.path.dirname(__file__), "..", "notebooks")
FIGURES_DIR = os.path.join(NOTEBOOKS_DIR, "figures")

os.makedirs(FIGURES_DIR, exist_ok=True)

METADATA = {
    "kernelspec": {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    },
    "language_info": {
        "name": "python",
        "version": "3.11.0",
    },
}


def make_nb(cells):
    nb = nbformat.v4.new_notebook()
    nb["metadata"] = METADATA
    nb["cells"] = cells
    return nb


def md(source):
    return nbformat.v4.new_markdown_cell(source)


def code(source):
    return nbformat.v4.new_code_cell(source)


def save(nb, name):
    path = os.path.join(NOTEBOOKS_DIR, name)
    with open(path, "w") as f:
        nbformat.write(nb, f)
    print(f"  Written: {path}")


# ======================================================================
# Notebook 01 — BSDE Primer
# ======================================================================

nb01_cells = [
    md("""\
# BSDE Primer: From Feynman-Kac to Numerical Solvers

A **Backward Stochastic Differential Equation (BSDE)** is a pair of adapted
processes $(Y, Z)$ satisfying

$$
Y_t = g(X_T) + \\int_t^T f(X_s, Y_s, Z_s)\\,ds - \\int_t^T Z_s\\,dW_s, \\quad 0 \\le t \\le T.
$$

- $X_t$ — the **forward** (state) process, e.g. an Ornstein-Uhlenbeck diffusion.
- $f$ — the **driver** (generator / running cost).
- $g$ — the **terminal condition**.
- $Y_t$ — the **value** process; in the Markovian case $Y_t = u(t, X_t)$.
- $Z_t$ — the **control** process; $Z_t = \\sigma(X_t)\\,\\partial_x u(t, X_t)$.

**Feynman-Kac** links BSDEs to quasilinear PDEs:

$$
\\partial_t u + \\tfrac{1}{2}\\sigma^2 \\partial_{xx} u + b\\,\\partial_x u
+ f(x, u, \\sigma\\,\\partial_x u) = 0, \\quad u(T, x) = g(x).
$$

This notebook illustrates the connection with a concrete linear example and
compares two numerical solvers.
"""),

    code("""\
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
%matplotlib inline

from ebsde.forward.ou_process import OrnsteinUhlenbeck
from ebsde.bsde.standard import StandardBSDE
from ebsde.solvers.pde import PDEBSDESolver
from ebsde.solvers.picard import PicardBSDESolver
"""),

    md("""\
## Linear BSDE: $f(x, y, z) = -y$, $g(x) = x^2$

We consider an Ornstein-Uhlenbeck forward process

$$
dX_t = -\\kappa X_t\\,dt + \\sigma\\,dW_t, \\quad X_0 = x_0,
$$

with $\\kappa = 1$, $\\sigma = 1$, and the linear BSDE

$$
f(x, y, z) = -y, \\quad g(x) = x^2, \\quad T = 1.
$$

The Feynman-Kac solution is

$$
u(t, x) = e^{-(T-t)} \\mathbb{E}[X_T^2 \\mid X_t = x]
         = e^{-(T-t)} \\bigl[\\text{Var}(X_T|X_t) + (\\mathbb{E}[X_T|X_t])^2\\bigr].
$$

For OU:
$$
\\mathbb{E}[X_T|X_t=x] = x\\,e^{-\\kappa(T-t)}, \\quad
\\text{Var}(X_T|X_t) = \\frac{\\sigma^2}{2\\kappa}(1 - e^{-2\\kappa(T-t)}).
$$
"""),

    code("""\
# --- Setup -----------------------------------------------------------
kappa, sigma, T = 1.0, 1.0, 1.0
ou = OrnsteinUhlenbeck(kappa=kappa, theta=0.0, sigma=sigma)

driver   = lambda x, y, z: -y
terminal = lambda x: x**2

bsde = StandardBSDE(forward=ou, driver=driver, terminal=terminal, T=T)

# --- PDE solver ------------------------------------------------------
pde_sol = PDEBSDESolver(bsde, n_x=200, n_t=200).solve()

x_grid = pde_sol['x_grid']
u0     = pde_sol['u'][:, 0]          # u(0, x)

# --- Exact Feynman-Kac solution at t=0 --------------------------------
def exact_u0(x, kappa=kappa, sigma=sigma, T=T):
    tau = T
    mean_XT  = x * np.exp(-kappa * tau)
    var_XT   = (sigma**2 / (2*kappa)) * (1 - np.exp(-2*kappa*tau))
    E_XT2    = var_XT + mean_XT**2
    return np.exp(-tau) * E_XT2

u_exact = exact_u0(x_grid)

# --- Plot ---------------------------------------------------------------
os.makedirs('notebooks/figures', exist_ok=True)

fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(x_grid, u_exact, 'k--', lw=2, label='Exact Feynman-Kac')
ax.plot(x_grid, u0,      'C0',  lw=2, label='PDEBSDESolver')
ax.set_xlabel('x')
ax.set_ylabel('u(0, x)')
ax.set_title('Linear BSDE: PDE solution vs exact at t=0')
ax.legend()
ax.set_xlim(-4, 4)
fig.tight_layout()
fig.savefig('notebooks/figures/01_pde_solution.png', dpi=120)
plt.close(fig)

print(f"Y0 (PDE, x=0) = {pde_sol['Y0']:.6f}")
print(f"Y0 (exact)    = {exact_u0(0.0):.6f}")
print("Figure saved → notebooks/figures/01_pde_solution.png")
"""),

    md("""\
## Three Solvers: PDE, Picard Iteration

The `ebsde` library provides two complementary solvers for standard BSDEs:

| Solver | Method | Best for |
|--------|--------|----------|
| `PDEBSDESolver` | Crank-Nicolson finite difference | Smooth, 1-D Markovian problems |
| `PicardBSDESolver` | Monte-Carlo + regression (Bouchard-Touzi) | Higher dimensions |

We now compare their $Y_0 = u(0, X_0)$ estimates for the same problem.
"""),

    code("""\
import os
import numpy as np

# --- Picard solver -------------------------------------------------------
picard_sol = PicardBSDESolver(
    bsde, n_paths=10000, n_steps=50, n_picard=3
).solve()

# --- PDE solver (already run above) -------------------------------------
Y0_pde    = float(pde_sol['Y0'])
Y0_picard = float(picard_sol['Y0'])
Y0_exact  = float(exact_u0(0.0))
"""),

    code("""\
print("=" * 50)
print(f"{'Solver':<20}  {'Y0':>10}  {'Error':>10}")
print("-" * 50)
print(f"{'Exact':.<20}  {Y0_exact:>10.6f}  {'—':>10}")
print(f"{'PDEBSDESolver':.<20}  {Y0_pde:>10.6f}  {abs(Y0_pde-Y0_exact):>10.2e}")
print(f"{'PicardBSDESolver':.<20}  {Y0_picard:>10.6f}  {abs(Y0_picard-Y0_exact):>10.2e}")
print("=" * 50)
print()
print("Picard residuals:", picard_sol['residuals'])
"""),
]


# ======================================================================
# Notebook 02 — Ergodic BSDEs
# ======================================================================

nb02_cells = [
    md("""\
# Ergodic BSDEs: What Happens as $T \\to \\infty$?

For long time horizons the value $Y_0$ of a BSDE grows linearly:

$$
Y_0(T) \\approx \\lambda \\cdot T + v(X_0) + o(1) \\quad \\text{as } T\\to\\infty.
$$

The constant $\\lambda$ is the **ergodic constant** — the long-run average of the
driver along the stationary distribution of the forward process.

An **Ergodic BSDE** directly characterises $(\\lambda, v)$ without sending $T\\to\\infty$:

$$
Y_t = Y_T + \\int_t^T [f(X_s, Y_s, Z_s) - \\lambda]\\,ds - \\int_t^T Z_s\\,dW_s
\\quad \\forall\\,T > t.
$$

The Markovian ergodic PDE (nonlinear eigenvalue problem) is:

$$
\\mathcal{L}v(x) + f(x, v(x), \\sigma(x)v'(x)) = \\lambda,
$$

where $\\mathcal{L} = \\tfrac{1}{2}\\sigma^2\\partial_{xx} + b\\,\\partial_x$.
"""),

    code("""\
import numpy as np
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
%matplotlib inline

from ebsde.forward.ou_process import OrnsteinUhlenbeck
from ebsde.bsde.standard import StandardBSDE
from ebsde.bsde.ergodic import ErgodicBSDE
from ebsde.solvers.picard import PicardBSDESolver
from ebsde.solvers.ergodic_pde import ErgodicPDESolver
from data.synthetic import ergodic_ou_quadratic_analytical
"""),

    md("""\
## OU + Quadratic Driver: $f(x, y, z) = x^2 - \\tfrac{1}{2}z^2$

This is the canonical benchmark.  Via the **Cole-Hopf transform**
$v = -2\\log\\phi$, the ergodic PDE linearises to a quantum harmonic
oscillator eigenvalue problem.  The exact ergodic constant is

$$
\\lambda^* = \\frac{1}{2}\\left(-\\kappa + \\sqrt{\\kappa^2 + 2\\alpha\\sigma^2}\\right)
\\approx 0.366 \\quad (\\kappa=\\sigma=\\alpha=\\gamma=1).
$$
"""),

    code("""\
# --- Setup ----------------------------------------------------------
ou = OrnsteinUhlenbeck(kappa=1.0, theta=0.0, sigma=1.0)
driver_q = lambda x, y, z: x**2 - 0.5 * z**2
terminal  = lambda x: np.zeros_like(np.asarray(x, dtype=float))

# --- Exact ergodic constant via Cole-Hopf --------------------------
exact = ergodic_ou_quadratic_analytical(
    kappa=1.0, sigma=1.0, alpha=1.0, beta=0.0, gamma=1.0
)
lambda_exact = exact['lambda_ergodic']
print(f"Exact lambda* = {lambda_exact:.6f}")

# --- Y0(T) grows linearly with T -----------------------------------
T_values = [5, 10, 20, 40]
Y0_list  = []
for T in T_values:
    bsde_T = StandardBSDE(forward=ou, driver=driver_q, terminal=terminal, T=float(T))
    sol    = PicardBSDESolver(bsde_T, n_paths=10000, n_steps=50, n_picard=3).solve()
    Y0_list.append(float(sol['Y0']))
    print(f"  T={T:>4}  Y0={sol['Y0']:.4f}  Y0/T={sol['Y0']/T:.4f}")

# --- Plot Y0 vs T with linear fit ----------------------------------
T_arr  = np.array(T_values, dtype=float)
Y0_arr = np.array(Y0_list)
slope  = np.polyfit(T_arr, Y0_arr, 1)[0]
print(f"\\nLinear slope (Y0/T) = {slope:.6f}  (exact lambda* = {lambda_exact:.6f})")

fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(T_arr, Y0_arr, 'o-', lw=2, ms=8, label='Y0(T)')
ax.plot(T_arr, slope * T_arr, '--', color='gray', label=f'slope = {slope:.3f}')
ax.axhline(0, color='k', lw=0.5)
ax.set_xlabel('T')
ax.set_ylabel('Y_0(T)')
ax.set_title('Y_0(T) grows linearly — slope → λ*')
ax.legend()
fig.tight_layout()

os.makedirs('notebooks/figures', exist_ok=True)
fig.savefig('notebooks/figures/02_y0_vs_T.png', dpi=120)
plt.close(fig)
print("Figure saved → notebooks/figures/02_y0_vs_T.png")
"""),

    md("""\
## The Ergodic Constant via `ErgodicPDESolver`

Instead of computing $Y_0(T)$ for large $T$, the `ErgodicPDESolver` directly
solves the nonlinear eigenvalue problem to find $\\lambda$ and the potential
function $v(x)$ (normalised so that $v(0)=0$).
"""),

    code("""\
# --- Solve ergodic PDE ---------------------------------------------
ebsde = ErgodicBSDE(forward=ou, driver=driver_q)
epde  = ErgodicPDESolver(ebsde, n_x=300).solve()

lambda_pde = epde['lambda_ergodic']
v_grid     = epde['v']
x_grid     = epde['x_grid']

print(f"ErgodicPDESolver lambda = {lambda_pde:.6f}")

# --- Plot v(x) ------------------------------------------------------
fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(x_grid, v_grid, 'C1', lw=2)
ax.set_xlabel('x')
ax.set_ylabel('v(x)')
ax.set_title(f'Ergodic value function  (λ = {lambda_pde:.4f})')
ax.axhline(0, color='k', lw=0.5, ls='--')
fig.tight_layout()
fig.savefig('notebooks/figures/02_value_function.png', dpi=120)
plt.close(fig)
print("Figure saved → notebooks/figures/02_value_function.png")
"""),

    code("""\
print("=" * 45)
print(f"{'Method':<30} {'lambda':>10}")
print("-" * 45)
print(f"{'Exact (Cole-Hopf)':<30} {lambda_exact:>10.6f}")
print(f"{'ErgodicPDESolver':<30} {lambda_pde:>10.6f}")
print(f"{'|Error|':<30} {abs(lambda_pde - lambda_exact):>10.2e}")
print("=" * 45)
"""),
]


# ======================================================================
# Notebook 03 — PDE Connection
# ======================================================================

nb03_cells = [
    md("""\
# The PDE Connection: Ergodic BSDE as a Nonlinear Eigenvalue Problem

In the Markovian setting $Y_t = v(X_t)$, the ergodic BSDE is equivalent to
the **nonlinear eigenvalue problem**:

$$
\\underbrace{\\mathcal{L}v(x)}_{\\text{generator}} +
\\underbrace{f(x, v(x), \\sigma(x)v'(x))}_{\\text{driver}} = \\lambda
$$

where $\\mathcal{L} = \\frac{\\sigma^2}{2}\\partial_{xx} + b\\,\\partial_x$
is the infinitesimal generator of $X$.

Key properties:
- $\\lambda$ is **unique** (ergodic constant).
- $v$ is unique **up to an additive constant**; normalised by $v(x_0) = 0$.
- For a **quadratic driver** the problem is equivalent to the quantum harmonic
  oscillator, admitting a closed-form solution via Cole-Hopf.
"""),

    code("""\
import numpy as np
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
%matplotlib inline

from ebsde.forward.ou_process import OrnsteinUhlenbeck
from ebsde.bsde.ergodic import ErgodicBSDE
from ebsde.solvers.ergodic_pde import ErgodicPDESolver
from data.synthetic import ergodic_ou_quadratic_analytical
"""),

    md("""\
## PDE Residual Verification

Once we obtain $(\\lambda, v)$ from the solver we can check the residual

$$
R(x) = \\mathcal{L}v(x) + f(x, v(x), \\sigma(x)v'(x)) - \\lambda
$$

pointwise.  A good solution should have $\\|R\\|_\\infty \\approx 0$.
"""),

    code("""\
# --- Setup ----------------------------------------------------------
ou = OrnsteinUhlenbeck(kappa=1.0, theta=0.0, sigma=1.0)
driver_q = lambda x, y, z: x**2 - 0.5 * z**2

ebsde  = ErgodicBSDE(forward=ou, driver=driver_q)
sol    = ErgodicPDESolver(ebsde, n_x=300).solve()

lambda_pde = sol['lambda_ergodic']
v_grid     = sol['v']
x_grid     = sol['x_grid']

# --- Compute PDE residual via finite differences -------------------
dx   = x_grid[1] - x_grid[0]
dv   = np.gradient(v_grid, dx)
d2v  = np.gradient(dv, dx)
sigma_x = ou.diffusion(x_grid.reshape(-1, 1)).flatten()
b_x     = ou.drift(x_grid.reshape(-1, 1)).flatten()
z_grid  = sigma_x * dv

f_vals  = np.array([
    driver_q(x_grid[i], v_grid[i], z_grid[i:i+1])
    for i in range(len(x_grid))
]).flatten()

residual = 0.5 * sigma_x**2 * d2v + b_x * dv + f_vals - lambda_pde

print(f"lambda = {lambda_pde:.6f}")
print(f"max |residual| (interior) = {np.max(np.abs(residual[10:-10])):.2e}")

# --- Plot v(x) and residual ----------------------------------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))

ax1.plot(x_grid, v_grid, 'C1', lw=2)
ax1.set_xlabel('x'); ax1.set_ylabel('v(x)')
ax1.set_title(f'Ergodic value function  (λ={lambda_pde:.4f})')
ax1.axhline(0, color='k', lw=0.5, ls='--')

ax2.plot(x_grid[10:-10], residual[10:-10], 'C2', lw=1.5)
ax2.axhline(0, color='k', lw=0.5)
ax2.set_xlabel('x'); ax2.set_ylabel('R(x)')
ax2.set_title('PDE Residual: $\\mathcal{L}v + f - \\lambda$')

fig.tight_layout()
os.makedirs('notebooks/figures', exist_ok=True)
fig.savefig('notebooks/figures/03_pde_residual.png', dpi=120)
plt.close(fig)
print("Figure saved → notebooks/figures/03_pde_residual.png")
"""),

    md("""\
## Grid Convergence Study

The `ErgodicPDESolver` uses a finite-difference discretisation.  As $n_x$
increases the error $|\\hat{\\lambda} - \\lambda^*|$ should decrease,
typically as $O(h^2)$ for smooth problems.
"""),

    code("""\
exact = ergodic_ou_quadratic_analytical(
    kappa=1.0, sigma=1.0, alpha=1.0, beta=0.0, gamma=1.0
)
lambda_exact = exact['lambda_ergodic']

grid_sizes = [50, 100, 200, 500]
errors     = []

for nx in grid_sizes:
    s = ErgodicPDESolver(ebsde, n_x=nx).solve()
    err = abs(s['lambda_ergodic'] - lambda_exact)
    errors.append(err)
    print(f"  n_x={nx:>5}  lambda={s['lambda_ergodic']:.7f}  err={err:.2e}")

# --- Convergence plot -----------------------------------------------
fig, ax = plt.subplots(figsize=(6, 4))
ax.loglog(grid_sizes, errors, 'o-', lw=2, ms=8)

# Reference O(h²) line
h_vals = 1.0 / np.array(grid_sizes, dtype=float)
ax.loglog(grid_sizes, errors[0] * (h_vals / h_vals[0])**2,
          '--', color='gray', label='O(h²)')

ax.set_xlabel('n_x (grid points)')
ax.set_ylabel('|λ̂ - λ*|')
ax.set_title('Grid convergence: ErgodicPDESolver')
ax.legend()
fig.tight_layout()
fig.savefig('notebooks/figures/03_grid_convergence.png', dpi=120)
plt.close(fig)
print("\\nFigure saved → notebooks/figures/03_grid_convergence.png")
"""),
]


# ======================================================================
# Notebook 04 — Deep BSDE
# ======================================================================

nb04_cells = [
    md("""\
# Deep BSDE: Solving Ergodic BSDEs with Neural Networks

The **Deep BSDE** approach parameterises the unknown functions with neural
networks and trains them by minimising a loss derived from the BSDE equation.

For ergodic BSDEs the `ErgodicDeepBSDESolver` implements three strategies:

| Strategy | Loss | Notes |
|---|---|---|
| `temporal_difference` | TD error along simulated paths | Fast, simulation-based |
| `direct_ergodic` | PDE residual under stationary measure | Meshless |
| `long_horizon` | Runs standard deep-BSDE at large $T$, extracts $\\lambda$ | High variance |

In this notebook we use the **`direct_ergodic`** strategy, which minimises

$$
\\mathcal{L}(v_\\theta, \\lambda) = \\mathbb{E}_{X\\sim\\pi}\\bigl[
  |\\mathcal{L}v_\\theta(X) + f(X, v_\\theta(X), z_\\theta(X)) - \\lambda|^2
\\bigr] + \\alpha\\,|v_\\theta(x_0)|^2
$$

where $\\pi$ is the stationary distribution and the second term normalises $v$.
"""),

    code("""\
import numpy as np
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
%matplotlib inline

import torch
print("PyTorch version:", torch.__version__)

from ebsde.forward.ou_process import OrnsteinUhlenbeck
from ebsde.bsde.ergodic import ErgodicBSDE
from ebsde.solvers.ergodic_deep import ErgodicDeepBSDESolver
from ebsde.solvers.ergodic_pde import ErgodicPDESolver
from data.synthetic import ergodic_ou_quadratic_analytical
"""),

    md("""\
## Network Architecture

The solver maintains two networks:
- **$v_\\theta$** — maps $x \\in \\mathbb{R}^d$ to $v(x) \\in \\mathbb{R}$
  (a 3-hidden-layer MLP with Tanh activations).
- **$z_\\theta$** — maps $x$ to $z(x) = \\sigma(x)\\nabla_x v(x)$
  (a separate 2-hidden-layer MLP).

$\\lambda$ is a **trainable scalar parameter**, jointly optimised with Adam.

During training the state samples $X \\sim \\mathcal{N}(0, \\sigma^2/(2\\kappa))$
are drawn from the OU stationary distribution at each step.
"""),

    code("""\
# --- Setup ----------------------------------------------------------
ou       = OrnsteinUhlenbeck(kappa=1.0, theta=0.0, sigma=1.0)
driver_q = lambda x, y, z: x**2 - 0.5 * z**2

ebsde = ErgodicBSDE(forward=ou, driver=driver_q)

# --- Train ----------------------------------------------------------
deep_solver = ErgodicDeepBSDESolver(
    ebsde,
    strategy='direct_ergodic',
    n_epochs=500,
    n_samples=512,
    rng_seed=42,
    torch_seed=42,
)
deep_sol = deep_solver.solve()

lambda_deep = deep_sol['lambda_ergodic']
loss_hist   = deep_sol['training_loss']
lam_hist    = deep_sol['lambda_history']

print(f"lambda (deep)  = {lambda_deep:.6f}")

# --- Training loss curve -------------------------------------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))

epochs = np.arange(1, len(loss_hist) + 1)
ax1.semilogy(epochs, loss_hist, 'C0', lw=1.2)
ax1.set_xlabel('Epoch'); ax1.set_ylabel('Loss')
ax1.set_title('Training loss (log scale)')

ax2.plot(np.arange(1, len(lam_hist)+1), lam_hist, 'C1', lw=1.5)
ax2.axhline(lambda_deep, color='k', lw=0.5, ls='--', label=f'final λ={lambda_deep:.4f}')
ax2.set_xlabel('Epoch'); ax2.set_ylabel('λ estimate')
ax2.set_title('Ergodic constant during training')
ax2.legend()

fig.tight_layout()
os.makedirs('notebooks/figures', exist_ok=True)
fig.savefig('notebooks/figures/04_training_loss.png', dpi=120)
plt.close(fig)
print("Figure saved → notebooks/figures/04_training_loss.png")
"""),

    md("""\
## Comparison: Deep vs Classical Solvers

We compare the ergodic constant $\\lambda$ obtained by the deep solver against
the PDE solver and the exact closed-form result.
"""),

    code("""\
# --- Classical PDE solver -------------------------------------------
pde_sol     = ErgodicPDESolver(ebsde, n_x=300).solve()
lambda_pde  = pde_sol['lambda_ergodic']

# --- Exact ----------------------------------------------------------
exact       = ergodic_ou_quadratic_analytical(
    kappa=1.0, sigma=1.0, alpha=1.0, beta=0.0, gamma=1.0
)
lambda_exact = exact['lambda_ergodic']

# --- Comparison table -----------------------------------------------
print("=" * 55)
print(f"{'Method':<30} {'lambda':>10}  {'|Error|':>10}")
print("-" * 55)
print(f"{'Exact (Cole-Hopf)':<30} {lambda_exact:>10.6f}  {'—':>10}")
print(f"{'ErgodicPDESolver (n_x=300)':<30} {lambda_pde:>10.6f}  {abs(lambda_pde-lambda_exact):>10.2e}")
print(f"{'ErgodicDeepBSDESolver':<30} {lambda_deep:>10.6f}  {abs(lambda_deep-lambda_exact):>10.2e}")
print("=" * 55)
"""),
]


# ======================================================================
# Notebook 05 — Long-Run Risk
# ======================================================================

nb05_cells = [
    md("""\
# Long-Run Risk Pricing via Ergodic BSDEs

The **long-run risk** model of Bansal & Yaron (2004) captures the premium
investors demand for exposure to *long-horizon* risks.

Under **Epstein-Zin** recursive preferences with risk aversion $\\gamma$ and
elasticity of intertemporal substitution $\\psi$, the continuation value
satisfies an ergodic BSDE whose **ergodic constant $\\lambda$** equals the
*long-run risk-adjusted discount rate*.

The driver (log-linearised around the stationary mean of the OU proxy $X_t$):

$$
f(x, v, z) = \\delta\\,\\theta_{\\rm EZ}\\,x + \\frac{\\gamma}{2}|z|^2,
\\quad \\theta_{\\rm EZ} = \\frac{1-\\gamma}{1 - 1/\\psi},
$$

where $\\delta$ is the subjective discount rate.

**Pricing implication**: the price-dividend ratio satisfies

$$
P_t / D_t \\longrightarrow e^{-(\\lambda - g)T} \\quad \\text{as } T \\to \\infty,
$$

so a higher $\\lambda$ implies a *lower* long-run price-dividend ratio.
"""),

    code("""\
import numpy as np
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
%matplotlib inline

from ebsde.applications.long_run_risk import LongRunRiskPricer
"""),

    md("""\
## Epstein-Zin Preferences

The Epstein-Zin aggregator coefficient

$$
\\theta_{\\rm EZ} = \\frac{1-\\gamma}{1 - 1/\\psi}
$$

captures how risk aversion and intertemporal substitution interact.

- $\\gamma > 1/\\psi$ (standard calibration): $\\theta_{\\rm EZ} < 0$,
  so agents prefer early resolution of uncertainty → long-run risk premium.
- $\\gamma = 1/\\psi$ (CRRA): $\\theta_{\\rm EZ}$ is degenerate.
"""),

    code("""\
# --- Calibrate & solve ---------------------------------------------
pricer = LongRunRiskPricer(gamma=5.0, psi=1.5, delta=0.02)

print(f"theta_EZ = {pricer.theta_ez:.4f}  "
      f"(gamma={pricer.gamma}, psi={pricer.psi})")

results = pricer.compute_risk_adjusted_rate(method='pde')

print()
print(f"Long-run risk-adjusted rate  lambda  = {results['lambda']:.6f}")
print(f"Risk premium (lambda - E[f])          = {results['risk_premium']:.6f}")
print(f"Equity risk premium (approx annual %) = {100*results['equity_premium']:.2f}%")
"""),

    md("""\
## Sensitivity to Risk Aversion $\\gamma$

Higher risk aversion $\\gamma$ → steeper quadratic penalty on $z$ → larger
ergodic constant $\\lambda$ (higher discount rate = lower equity valuations).
"""),

    code("""\
# --- Sensitivity analysis ------------------------------------------
gamma_values = [2.0, 5.0, 8.0]
df = pricer.sensitivity_analysis('gamma', gamma_values)

print(df.to_string(index=False))

# --- Plot ----------------------------------------------------------
fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(df['gamma'], df['lambda'], 'o-', lw=2, ms=8, color='C3')
ax.set_xlabel('Risk aversion γ')
ax.set_ylabel('λ  (long-run discount rate)')
ax.set_title('Long-run risk premium vs risk aversion')
ax.grid(alpha=0.3)
fig.tight_layout()

os.makedirs('notebooks/figures', exist_ok=True)
fig.savefig('notebooks/figures/05_sensitivity.png', dpi=120)
plt.close(fig)
print("Figure saved → notebooks/figures/05_sensitivity.png")
"""),

    md("""\
## Term Structure of Risk

For finite maturity $T$ the yield $y(T) = -Y_0(T)/T$ traces the term structure.
As $T \\to \\infty$ the yield converges to the ergodic constant $\\lambda$.

A *upward-sloping* term structure (yields rise with $T$) is consistent with a
positive long-run risk premium.
"""),

    code("""\
# --- Term structure ------------------------------------------------
maturities = [0.5, 1.0, 2.0]
ts = pricer.term_structure_of_risk(maturities)

print(f"{'Maturity T':>12}  {'Yield y(T)':>12}")
print("-" * 27)
for T_val, y_val in zip(ts['maturities'], ts['yields']):
    print(f"{T_val:>12.2f}  {y_val:>12.6f}")
print(f"{'lambda (T->inf)':>12}  {ts['lambda_limit']:>12.6f}")
"""),
]


# ======================================================================
# Write all notebooks
# ======================================================================

notebooks = [
    ("01_bsde_primer.ipynb",   nb01_cells),
    ("02_ergodic_bsde.ipynb",  nb02_cells),
    ("03_pde_connection.ipynb", nb03_cells),
    ("04_deep_bsde.ipynb",     nb04_cells),
    ("05_long_run_risk.ipynb", nb05_cells),
]

print("Creating notebooks ...")
for name, cells in notebooks:
    nb = make_nb(cells)
    save(nb, name)

print(f"\nAll notebooks written to: {os.path.abspath(NOTEBOOKS_DIR)}")
print(f"Figures directory:         {os.path.abspath(FIGURES_DIR)}")
