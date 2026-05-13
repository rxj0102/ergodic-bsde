# Numerical Methods

This document describes the algorithms behind each solver in the `ergodic-bsde` library and
gives practical guidance on when to use each one.

---

## 1. Regression-Based BSDE (Gobet–Lemor–Warin)

### 1.1 Algorithm

The regression-based solver in `ebsde/solvers/regression.py` implements the
Gobet–Lemor–Warin (2005) scheme for **standard (finite-horizon) BSDEs**.

The key idea is to approximate the value function $u(t_i, x) \approx \Phi(x)^\top \alpha_i$
using a finite set of basis functions $\{\phi_1, \ldots, \phi_K\}$, and determine the
coefficients $\alpha_i$ by **least-squares regression** at each time step.

**Algorithm** (backward in time from $i = M-1$ down to $0$):

1. Simulate $N$ forward paths $X_0, X_1, \ldots, X_M$ on the grid $t_i = i \cdot \Delta t$.
2. Set $Y_M = g(X_M)$ (terminal condition).
3. For $i = M-1, \ldots, 0$:
   a. Build the design matrix $\Phi_i \in \mathbb{R}^{N \times K}$ with rows $\Phi(X_i^{(n)})$.
   b. Solve $\alpha_i = \arg\min_\alpha \|\Phi_i \alpha - Y_{i+1}\|^2$ via QR decomposition.
   c. Set $\hat{u}(t_i, X_i) = \Phi_i \alpha_i$ (conditional expectation proxy).
   d. Recover $Z_i = \sigma(X_i) \cdot [\nabla \Phi(X_i)]^\top \alpha_i$ analytically from the gradient of the basis.
   e. Update: $Y_i = \hat{u}(t_i, X_i) + \Delta t \cdot f(X_i, Y_i, Z_i)$.

Step (d) is the key advantage: $Z$ is recovered from the **analytical gradient** of the
fitted polynomial/Hermite/RBF expansion, avoiding a separate noisy $Z$-regression.

### 1.2 Basis Functions

| Type | $\phi_k(x)$ | Gradient $\nabla\phi_k$ | Notes |
|---|---|---|---|
| `polynomial` | $x^{k_1} \cdots x^{k_d}$ | $k_j x^{k_j-1} \prod_{i\ne j}(\cdot)$ | All monomials up to total degree $p$ |
| `hermite` | $\mathrm{He}_k(x)$ | $k\, \mathrm{He}_{k-1}(x)$ | Probabilist's Hermite; orthogonal under $\mathcal{N}(0,1)$ |
| `rbf` | $\exp(-\|x-c\|^2/(2\text{bw}^2))$ | $-(x-c)/\text{bw}^2 \cdot \phi$ | Centers placed at empirical quantiles |

### 1.3 Curse of Dimensionality

For total-degree polynomial basis in dimension $d$, the number of basis functions grows as

$$K = \binom{d + p}{p} = O(p^d / d!).$$

For $d = 5$, $p = 5$, this gives $K = 252$.  For $d = 10$ the count exceeds $3000$, making
regression unstable without regularisation.  Deep BSDE methods (Section 4) are preferred for
$d \gtrsim 5$.

### 1.4 Use QR, Not Normal Equations

The normal equations $\Phi^\top \Phi\, \alpha = \Phi^\top Y$ are numerically unstable when
$\Phi$ is nearly column-rank deficient.  The library uses `np.linalg.lstsq` (backed by LAPACK
`dgelsd`), which uses a divide-and-conquer SVD internally, giving robust results at the cost
of $O(NK^2 + K^3)$ per time step.

---

## 2. PDE Methods for the Ergodic Eigenvalue Problem

The primary PDE solver is `ErgodicPDESolver` in `ebsde/solvers/ergodic_pde.py`.  It
discretises the ergodic elliptic PDE

$$\mathcal{L}v(x) + f(x, v(x), \sigma(x) v'(x)) = \lambda$$

on a uniform grid $x_0 < x_1 < \cdots < x_{n-1}$ with spacing $\Delta x$.

### 2.1 Finite-Difference Discretisation of $\mathcal{L}$

The generator $\mathcal{L} = \frac{\sigma^2}{2}\partial_{xx} + b\,\partial_x$ is discretised
by **central differences**:

$$v''(x_i) \approx \frac{v_{i+1} - 2v_i + v_{i-1}}{\Delta x^2}, \qquad
v'(x_i) \approx \frac{v_{i+1} - v_{i-1}}{2\Delta x}.$$

This yields the **tridiagonal system** $A\mathbf{v} \approx \mathcal{L}\mathbf{v}$ where:

$$A_{i,i-1} = \frac{\sigma_i^2}{2\Delta x^2} - \frac{b_i}{2\Delta x}, \quad
A_{i,i} = -\frac{\sigma_i^2}{\Delta x^2}, \quad
A_{i,i+1} = \frac{\sigma_i^2}{2\Delta x^2} + \frac{b_i}{2\Delta x}.$$

**Boundary conditions**: Neumann (zero-flux) at both endpoints, implemented via ghost points:
$v_{-1} = v_1$ and $v_n = v_{n-2}$.

The matrix $A$ is assembled once in `_build_operator()` and cached; all three solve methods
reuse it.

### 2.2 Method 1: Fixed-Point Iteration on $\lambda$

**Algorithm** (implemented in `_solve_nonlinear_iteration`):

1. Initialise $\lambda^{(0)} = 0$, $v^{(0)} = 0$.
2. For $k = 1, 2, \ldots$:
   a. Given $v^{(k-1)}$, evaluate $f_i = f(x_i, v^{(k-1)}_i, \sigma_i (v^{(k-1)})'_i)$.
   b. Solve the **pinned BVP**: modify row $i_{\text{mid}}$ of $A$ to enforce $v_{i_{\text{mid}}} = 0$, then solve the linear system $(A - \lambda^{(k-1)}I)\mathbf{v} = -\mathbf{f}$.
   c. Update: $\lambda^{(k)} = \int (Av^{(k)} + f^{(k)})\, d\mu \approx \sum_i (A\mathbf{v}^{(k)} + \mathbf{f}^{(k)})_i\, \mu_i \Delta x$, where $\mu$ is the stationary density.
3. Stop when $|\lambda^{(k)} - \lambda^{(k-1)}| < \varepsilon$ and $\|v^{(k)} - v^{(k-1)}\|/\sqrt{n} < \varepsilon$.

This is a **fixed-point (Picard) iteration** on the $(v, \lambda)$ pair.  Convergence is
guaranteed under the Lipschitz and dissipativity conditions of Fuhrman–Hu–Tessitore (2009),
with a contraction rate $\rho < 1$ depending on the Lipschitz constant of $f$.

**Complexity**: $O(n)$ per linear solve (Thomas algorithm for tridiagonals), $O(n_{\text{iter}} \cdot n)$ total.

### 2.3 Method 2: Newton Iteration (Full Jacobian)

**Algorithm** (implemented in `_solve_newton_augmented`):

The ergodic PDE with normalisation is cast as the $(n+1)$-dimensional nonlinear system

$$F(v, \lambda) = \begin{pmatrix} Av + f(x, v, \sigma v') - \lambda \mathbf{1} \\ \langle v, \mu \rangle \end{pmatrix} = 0,$$

where the second equation $\langle v, \mu \rangle = \mathbb{E}_\pi[v(X)] = 0$ is the **gauge
condition** fixing the additive constant in $v$.

Each Newton step solves

$$J^{(k)} \begin{pmatrix}\Delta v \\ \Delta\lambda\end{pmatrix} = -F(v^{(k)}, \lambda^{(k)}),$$

where the Jacobian $J \in \mathbb{R}^{(n+1)\times(n+1)}$ is computed by **finite differences**:

$$J_{ij} = \frac{\partial F_i}{\partial v_j} \approx \frac{F_i(v + \varepsilon e_j, \lambda) - F_i(v, \lambda)}{\varepsilon}, \quad \varepsilon = 10^{-5}.$$

The last column is $\partial F/\partial\lambda = -\mathbf{1}$ (exact).

Newton converges **quadratically** near the solution, requiring far fewer iterations than
fixed-point for nonlinear drivers.  The cost per iteration is $O(n^3)$ for the dense $(n+1)\times(n+1)$ solve, which limits this method to $n_x \lesssim 2000$.  Use Method 1 or deep methods for larger grids.

### 2.4 Method 3: Linear Eigenvalue (Drivers Linear in $y$, No $z$)

For $f(x, y, z) = h(x) + c \cdot y$, the ergodic PDE is the linear system

$$\begin{pmatrix} A + cI & -\mathbf{1} \\ e_{i_0}^\top & 0 \end{pmatrix}
\begin{pmatrix} v \\ \lambda \end{pmatrix}
= \begin{pmatrix} -h \\ 0 \end{pmatrix},$$

where $e_{i_0}$ enforces $v(x_0) = 0$.  This $(n+1) \times (n+1)$ system is solved directly
with `np.linalg.solve`, giving the exact FD solution in $O(n^3)$ (or $O(n)$ if the
tridiagonal structure is exploited via the Thomas algorithm).

### 2.5 Stationary Weights and the Normalisation Integral

The integral $\mathbb{E}_\pi[g(X)] = \int g\, d\pi$ is approximated by

$$\sum_{i=0}^{n-1} g(x_i)\, \mu_i\, \Delta x, \qquad \mu_i = \frac{e^{S_i}}{\sum_j e^{S_j}\,\Delta x},$$

where $S_i = \int_0^{x_i} 2b(x)/\sigma(x)^2\, dx$ is the log-density accumulated by
quadrature.  For OU processes the density is Gaussian and is computed analytically.

### 2.6 Convergence Rates for the PDE Method

- **Spatial discretisation**: central differences give $O(\Delta x^2)$ accuracy in $v$ and $\lambda$.
- **Newton iteration**: quadratic convergence once within the basin of attraction.
- **Fixed-point iteration**: linear convergence; typical rate $\rho \sim 0.1$–$0.5$ for
  drivers with moderate Lipschitz constant.

---

## 3. Deep BSDE (Han–Jentzen–E 2018)

### 3.1 Architecture

The Deep BSDE solver (`ebsde/solvers/deep_bsde.py`) parameterises the solution to a
**standard (finite-horizon)** BSDE.  At each time step $t_i$:

- The initial value $Y_0 \in \mathbb{R}$ is a **trainable scalar parameter**.
- The gradient process is approximated by a **sub-network** $Z_i(x; \theta_i) \in \mathbb{R}^d$
  (one `SubNet` per time step for `per_step` mode, or a single shared network for `shared` mode).
- Sub-network architecture: `Linear → BatchNorm → Act` repeated $L$ times, then `Linear`.

### 3.2 Forward Pass and Loss

The **Euler–Maruyama discretisation** of the BSDE is run forward:

$$Y_{i+1} = Y_i - f(X_i, Y_i, Z_i)\,\Delta t + Z_i \cdot \Delta W_i, \quad i = 0, \ldots, M-1,$$

where $\Delta W_i = W_{t_{i+1}} - W_{t_i} \sim \mathcal{N}(0, \Delta t\, I)$.  The loss is

$$\mathcal{L}(\theta) = \mathbb{E}\bigl[|Y_M - g(X_M)|^2\bigr] \approx \frac{1}{N}\sum_{n=1}^N |Y_M^{(n)} - g(X_M^{(n)})|^2.$$

Minimising this loss drives the network to satisfy the BSDE at the terminal time, and by the
Markovian structure the entire trajectory converges.

### 3.3 Complexity

| Component | Cost |
|---|---|
| Forward pass, 1 path | $O(M \cdot W^2)$ where $W$ = hidden width |
| Gradient (backprop) | $O(M \cdot W^2)$ |
| Full batch, 1 epoch | $O(N \cdot M \cdot W^2)$ |
| Total | $O(N_{\text{epochs}} \cdot N \cdot M \cdot W^2)$ |

Per-step networks have $M \times$ more parameters than the shared network, but are more
expressive for time-inhomogeneous problems.

---

## 4. Ergodic Deep BSDE (Three Strategies)

The ergodic solver `ErgodicDeepBSDESolver` in `ebsde/solvers/ergodic_deep.py` targets the
**stationary** ergodic PDE directly.  All three strategies learn a network $v_\theta : \mathbb{R}^d \to \mathbb{R}$
and a scalar parameter $\lambda_\theta$, with normalisation enforced by penalising
$|\mathbb{E}_\mu[v_\theta(X)]|^2$.

### 4.1 Strategy 1: `temporal_difference`

**Loss** (continuous-time PDE residual via autograd):

$$\mathcal{L}_{\mathrm{TD}}(\theta, \lambda) =
  \mathbb{E}_\mu\!\left[\bigl(\mathcal{L}v_\theta(X) + f(X, v_\theta(X), \sigma(X)\nabla v_\theta(X)) - \lambda\bigr)^2\right]
  + \alpha\,\bigl(\mathbb{E}_\mu[v_\theta(X)]\bigr)^2.$$

The generator $\mathcal{L}v$ is computed **exactly** via double backpropagation:

$$\mathcal{L}v = b \cdot \nabla v + \frac{1}{2}\sum_{j=1}^d \sigma_j^2\, \partial^2_{jj} v,$$

where $\partial^2_{jj} v$ is the $j$-th diagonal of the Hessian, extracted by differentiating
$\partial_j v$ with respect to $x_j$ using `torch.autograd.grad(..., create_graph=True)`.

$Z$ is tied to the gradient: $Z = \sigma(X)\nabla v_\theta(X)$, so no separate $Z$-network
is needed.  This enforces the martingale representation and selects the unique ergodic
solution (up to gauge).

### 4.2 Strategy 2: `direct_ergodic`

**Loss** (separate $v$-net and $z$-net with consistency penalty):

$$\mathcal{L}_{\mathrm{dir}}(\theta, \phi, \lambda) =
  \mathbb{E}_\mu\!\left[|\mathcal{L}v_\theta + f(X, v_\theta, z_\phi) - \lambda|^2\right]
  + \alpha\,(\mathbb{E}_\mu[v_\theta])^2
  + \beta\,\mathbb{E}_\mu\!\left[|z_\phi(X) - \sigma(X)\nabla v_\theta(X)|^2\right].$$

The third term enforces consistency between the separate $z$-network and the gradient of $v$.
The benefit over `temporal_difference` is that $z_\phi$ can have a different inductive bias
from $\nabla v_\theta$ (important when $Z$ is discontinuous or rapidly varying).

### 4.3 Strategy 3: `long_horizon`

Delegates to `ErgodicPicardSolver` with a sequence of increasing horizons
$T_1 < T_2 < T_3$ (e.g.\ $T/4, T/2, T$) and extracts $\lambda$ from the slope of
$Y_0(T)/T$ vs $T$.  This avoids the double-autograd cost but introduces the long-horizon
approximation error $O(e^{-\text{gap}\cdot T})$.

### 4.4 Training Details

- **Optimiser**: Adam with cosine annealing learning rate schedule ($\eta_{\min} = 0.01\,\eta_0$).
- **Gradient clipping**: global norm clipped to 1.0 to prevent exploding gradients.
- **Stationary sampling**: for OU processes, samples drawn analytically from $\mathcal{N}(\theta, \sigma^2/(2\kappa))$; for general processes, a long burn-in path is sub-sampled.
- **Pool refresh**: the stationary sample pool is refreshed every $\max(100, N_{\text{epochs}}/10)$ steps.

---

## 5. Convergence Rates Summary

| Method | Error in $\lambda$ | Dominant cost | Notes |
|---|---|---|---|
| PDE fixed-point | $O(\Delta x^2)$ | $O(n_x \cdot n_{\text{iter}})$ | $n_{\text{iter}} \sim 50$–$200$ |
| PDE Newton | $O(\Delta x^2)$ | $O(n_x^3 \cdot n_{\text{Newton}})$ | $n_{\text{Newton}} \sim 5$–$20$; best accuracy per solve |
| Regression BSDE | $O(\Delta x^2) + O(N^{-1/2})$ | $O(N \cdot M \cdot K^2)$ | Curse of dimensionality in $K$ |
| MC / Picard | $O(N^{-1/2})$ | $O(N \cdot M)$ | Unbiased for large $N$ |
| Deep (TD/direct) | $O(\varepsilon_{\text{opt}})$ | $O(N_{\text{ep}} \cdot N \cdot W^2)$ | $\varepsilon_{\text{opt}}$ = optimisation gap |

For the PDE method the spatial error satisfies $|\lambda_h - \lambda| = O(\Delta x^2)$ by
standard finite-difference theory (second-order central differences on a smooth solution).

For Monte Carlo methods, the MSE satisfies $\mathrm{MSE}(\hat\lambda) = O(N^{-1})$, so the
RMSE is $O(N^{-1/2})$.

For deep methods, convergence is empirically $O(1/\text{epoch}^p)$ with $p \approx 0.5$–$1$,
but has no sharp theoretical rate yet (active research area).

---

## 6. Practical Recommendations

| Scenario | Recommended Solver | Reason |
|---|---|---|
| 1-D, smooth driver, high accuracy needed | `ErgodicPDESolver(method='newton')` | $O(\Delta x^2)$, fast Newton convergence |
| 1-D, moderate accuracy | `ErgodicPDESolver(method='iteration')` | Simpler, robust for nonlinear drivers |
| Linear driver in $y$ | `ErgodicPDESolver(method='linear')` | Direct solve, exact up to FD error |
| 2–5-D | `ErgodicDeepBSDESolver(strategy='temporal_difference')` | Avoids FD grid; noise-free PDE residual |
| $d \ge 5$, smooth $v$ | `ErgodicDeepBSDESolver(strategy='temporal_difference')` | Best variance reduction via autograd |
| $d \ge 5$, discontinuous $Z$ | `ErgodicDeepBSDESolver(strategy='direct_ergodic')` | Separate $Z$-net more flexible |
| Quick $\lambda$ estimate, any $d$ | `ErgodicPicardSolver` | Fast regression-based, no neural nets |
| Production, $d \ge 10$ | `ErgodicDeepBSDESolver(strategy='temporal_difference')` | Scales to 100-D with sufficient width |
