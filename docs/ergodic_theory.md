# Ergodic Theory of Diffusion Processes

This document covers the probabilistic and spectral foundations that guarantee the
well-posedness of ergodic BSDEs and determine the convergence behaviour of numerical solvers.

---

## 1. Ergodicity of Diffusion Processes

### 1.1 Setting

Let $X = (X_t)_{t \ge 0}$ be the solution to the SDE

$$dX_t = b(X_t)\, dt + \sigma(X_t)\, dW_t, \quad X_0 = x,$$

with $b : \mathbb{R}^d \to \mathbb{R}^d$ and $\sigma : \mathbb{R}^d \to \mathbb{R}^{d \times d}$.
Throughout we assume the SDE has a unique strong solution and $\sigma\sigma^\top$ is uniformly
elliptic on compact sets (so the process is non-degenerate).

### 1.2 Recurrence and Positive Recurrence

- **Irreducibility**: $X$ is **(topologically) irreducible** if for any open set $U$ and any
  starting point $x$, the first hitting time $\tau_U = \inf\{t \ge 0 : X_t \in U\}$ is
  finite a.s.  Uniform ellipticity guarantees this.

- **Positive recurrence**: $X$ is **positive recurrent** if there exists a probability measure
  $\pi$ on $\mathbb{R}^d$ such that for every bounded measurable $g$,

  $$\frac{1}{T}\int_0^T g(X_t)\, dt \xrightarrow[T\to\infty]{\text{a.s.}} \int g\, d\pi.$$

  The measure $\pi$ is the **stationary (invariant) distribution**.

Positive recurrence is the minimal hypothesis needed for the ergodic BSDE to have a finite
ergodic constant $\lambda$.  Without it, $\lambda = \pm\infty$ and the ergodic equation has
no solution.

### 1.3 Lyapunov Conditions

A classical sufficient condition for positive recurrence uses a **Lyapunov function**.

**Foster–Lyapunov criterion**: Suppose there exist $V \in C^2(\mathbb{R}^d)$, $V \ge 1$, a
compact set $\mathcal{K}$, and constants $\alpha > 0$, $\beta < \infty$ such that

$$\mathcal{L}V(x) \le -\alpha\, V(x) + \beta\, \mathbf{1}_\mathcal{K}(x), \quad \forall x \in \mathbb{R}^d.$$

Then $X$ is positive recurrent with a unique stationary distribution $\pi$ satisfying
$\int V\, d\pi < \infty$.

**Interpretation**: $\mathcal{L}V \le -\alpha V$ outside the compact set $\mathcal{K}$ means
the process is "pulled inward" at rate $\alpha$ in the Lyapunov sense.  Inside $\mathcal{K}$,
the drift is bounded by $\beta$.

**Example — OU process**: $dX_t = -\kappa(X_t - \theta)\, dt + \sigma\, dW_t$ with $\kappa > 0$.
Take $V(x) = e^{c(x-\theta)^2}$ for small $c > 0$.  Then

$$\mathcal{L}V = V\bigl[(-2c\kappa)(x-\theta)^2 + 2c\sigma^2 (1 + 2c(x-\theta)^2) \cdot \tfrac{1}{2}\bigr],$$

which is $\le -\alpha V$ for large $|x|$ as long as $\kappa > c\sigma^2$.  So any
$\kappa > 0$ suffices (with $c$ chosen small), confirming ergodicity.  Implemented in
`OrnsteinUhlenbeck.is_ergodic()` as the check `kappa > 0`.

---

## 2. Spectral Gap

### 2.1 Definition

The **spectral gap** of the generator $\mathcal{L}$ is

$$\mathrm{gap}(\mathcal{L}) = \inf\left\{
  \frac{-\int v\, \mathcal{L}v\, d\pi}{\mathrm{Var}_\pi(v)}
  : v \in \mathrm{Dom}(\mathcal{L}),\ \mathrm{Var}_\pi(v) > 0
\right\} = \lambda_0 - \lambda_1,$$

where $\lambda_0 = 0 \ge \lambda_1 \ge \lambda_2 \ge \cdots$ are the eigenvalues of $\mathcal{L}$
in $L^2(\pi)$, ordered in decreasing order.  (The eigenvalue $\lambda_0 = 0$ corresponds to
the constant eigenfunction.)

A positive spectral gap is equivalent to the **Poincaré inequality**: there exists $c > 0$
such that for all $f$ with $\mathbb{E}_\pi[f] = 0$,

$$\mathrm{Var}_\pi(f) \le c\, \mathbb{E}_\pi[|\sigma^\top \nabla f|^2].$$

### 2.2 Mixing Time

The spectral gap controls how quickly the process forgets its initial condition.  For the
semigroup $P_t g(x) = \mathbb{E}_x[g(X_t)]$, the $L^2(\pi)$-mixing satisfies

$$\|P_t g - \mathbb{E}_\pi[g]\|_{L^2(\pi)} \le e^{-\mathrm{gap}\cdot t}\, \|g - \mathbb{E}_\pi[g]\|_{L^2(\pi)}.$$

**Mixing time** $\tau_{\text{mix}} \sim 1/\mathrm{gap}$.  A larger spectral gap means faster
mixing and faster convergence of time averages to their ergodic limits.

### 2.3 Spectral Gap for OU

For the OU process $dX_t = -\kappa(X_t - \theta)\, dt + \sigma\, dW_t$:

$$\mathrm{gap}(\mathcal{L}_{\mathrm{OU}}) = \kappa.$$

The eigenfunctions are Hermite polynomials $H_n(x)$ with eigenvalues $-n\kappa$, so the gap
between the top eigenvalue $0$ and the next one $-\kappa$ is exactly $\kappa$.

The library estimates the spectral gap numerically via
`ErgodicConstantAnalysis.spectral_gap_estimate`, which builds the FD generator matrix and
computes $\lambda_0 - \lambda_1$ via `scipy.linalg.eigvals`.

---

## 3. How Forward Process Ergodicity Determines BSDE Structure

### 3.1 Solvability Condition

The ergodic BSDE

$$Y_t = Y_T + \int_t^T [f(X_s, Y_s, Z_s) - \lambda]\, ds - \int_t^T Z_s \cdot dW_s$$

has a finite ergodic constant $\lambda$ if and only if the **running average of the driver
under the stationary measure is finite**:

$$\lambda \approx \mathbb{E}_\pi[f(X, 0, 0)] < \infty.$$

More precisely, $\lambda$ is uniquely determined by the condition that $(Y, Z)$ is a stationary
process, which requires the martingale problem to be well-posed.  If $f$ grows too rapidly
(e.g., $|f(x, 0, 0)| \sim e^{|x|^2}$) and the tails of $\pi$ are heavy, then no finite
$\lambda$ exists.

For OU processes with $\pi = \mathcal{N}(\theta, \sigma^2/(2\kappa))$, the condition
$\mathbb{E}_\pi[f(X, 0, 0)] < \infty$ is satisfied for any polynomially growing driver
$|f(x, 0, 0)| \le C(1 + |x|^p)$ for any $p < \infty$, since Gaussian tails decay faster than
any polynomial.

### 3.2 The Markovian Reduction

When $X$ is a Markov diffusion and $f$ is Markovian ($f = f(x, y, z)$), the process
$(Y_t, Z_t) = (v(X_t), \sigma(X_t)\nabla v(X_t))$ with $Y_t = v(X_t)$ **stationary** requires
that $v$ satisfies the ergodic PDE

$$\mathcal{L}v(x) + f(x, v(x), \sigma(x)\nabla v(x)) = \lambda \quad \forall x.$$

This reduction is exact: the BSDE is stationary if and only if $v$ solves the ergodic PDE.

---

## 4. Stationary Distribution and Gauge Fixing

### 4.1 Non-Uniqueness of $v$

If $(v, \lambda)$ solves the ergodic PDE, so does $(v + c, \lambda)$ for any constant $c$.
This one-dimensional family of solutions corresponds to adding a constant to $Y_t$, which
does not affect the driver evaluation (since $f$ is independent of $y$ in most applications,
or the $y$-dependence is dissipative and killed by the constant shift in the limit).

### 4.2 Gauge Conditions

To select a canonical representative, one imposes a **normalisation (gauge) condition**:

| Condition | Mathematical form | Library implementation |
|---|---|---|
| Point normalisation | $v(x_0) = 0$ | `v[mid] = 0` pinned in all PDE solvers |
| Mean normalisation | $\mathbb{E}_\pi[v(X)] = 0$ | Newton solver uses $\langle v, \mu \rangle = 0$ |
| Deep ergodic | $\mathbb{E}_\mu[v_\theta(X)] \approx 0$ | Penalty term $\alpha(\mathbb{E}_\mu[v])^2$ in loss |

Point normalisation at the midpoint of the grid is the default.  Mean normalisation (used in
the Newton solver) is more natural but requires the stationary weights.

### 4.3 Computing $\lambda$ from $v$

Once $v$ is computed with any normalisation, $\lambda$ is recovered by integrating the PDE
residual against the stationary measure:

$$\lambda = \int \bigl[\mathcal{L}v(x) + f(x, v(x), \sigma(x)\nabla v(x))\bigr]\, \pi(dx)
           \approx \sum_i \bigl[(\mathcal{L}_h\mathbf{v})_i + f_i\bigr]\, \mu_i\, \Delta x.$$

This is implemented in `_solve_nonlinear_iteration` step 2(c).

---

## 5. Exponential Convergence of Finite-$T$ Approximation

### 5.1 The Finite-Horizon BSDE

For a fixed $T$, define the finite-horizon BSDE solution $(Y^T_t, Z^T_t)$ with terminal
condition $g(X_T) = 0$ (for simplicity).  Denote $\lambda(T) = -Y^T_0 / T$, the ergodic
constant estimate from the finite-horizon problem.

### 5.2 Exponential Rate

Under the assumptions of Fuhrman–Hu–Tessitore (2009), with spectral gap $\mathrm{gap} > 0$:

$$|\lambda(T) - \lambda| \le C\, e^{-\mathrm{gap}\cdot T},$$

where $C > 0$ depends on the initial condition and the norm of $g$.

**Proof sketch**: Write $Y^T_0 = -\lambda T + R_T$ where $R_T$ captures the boundary-layer
correction.  By the Feynman–Kac connection, $R_T = u(0, x_0)$ where $u$ solves the parabolic
PDE with the ergodic shift.  The boundary layer decays as $e^{-\mathrm{gap}\cdot T}$ by
spectral theory (the semigroup contracts with rate $e^{-\mathrm{gap}\cdot t}$).

**Practical consequence**: for the OU process with $\kappa = 1$ and gap $= 1$, a horizon
$T = 10$ gives residual error $e^{-10} \approx 4.5 \times 10^{-5}$, which is negligible for
most applications.  The `long_horizon` strategy in `ErgodicDeepBSDESolver` exploits this by
using $T \ge 10$.

### 5.3 Role of the Spectral Gap in Solver Design

| Quantity | Effect of larger gap |
|---|---|
| Mixing time $1/\mathrm{gap}$ | Faster; fewer samples needed for stationarity |
| Finite-$T$ error $Ce^{-\mathrm{gap}\cdot T}$ | Smaller; shorter horizon suffices |
| Picard iteration rate | Faster contraction (smaller Lipschitz-to-gap ratio) |
| Bootstrap CI width | Narrower (less variance in $\hat\lambda$) |

For an OU process, $\mathrm{gap} = \kappa$.  Increasing $\kappa$ improves solver convergence
at the cost of a less persistent state process (which may not match the financial application).

---

## 6. Summary: Ergodicity Checklist

Before constructing an `ErgodicBSDE`, verify:

1. **Is $X$ positive recurrent?** Check `forward.is_ergodic()`.  For `OrnsteinUhlenbeck`
   this requires $\kappa > 0$.  For custom processes, verify the Foster–Lyapunov condition.

2. **Is the driver integrable?** $\mathbb{E}_\pi[|f(X, 0, 0)|] < \infty$.  For polynomial
   drivers on OU this is automatic.

3. **What is the spectral gap?** Use `ErgodicConstantAnalysis.spectral_gap_estimate`.
   A gap $\ge 0.5$ is comfortable; gap $< 0.1$ may require $T \ge 50$ for the long-horizon
   strategy.

4. **Is the gauge condition consistent?** All solvers use `v(x_{\mathrm{mid}}) = 0` by
   default; the Newton solver alternatively uses $\mathbb{E}_\pi[v] = 0$.  Both are valid;
   the resulting $\lambda$ is the same.
