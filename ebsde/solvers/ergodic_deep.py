"""
Deep learning solver for ergodic BSDEs.

Three strategies:
  'temporal_difference' (default): minimise E[δ_n²] where
      δ_n = v(X_{n+1}) - v(X_n) - [f(X_n,v(X_n),z(X_n)) - λ]Δt - z(X_n)·ΔW_n

  'direct_ergodic': minimise the ergodic PDE residual
      E_μ[|Lv + f(x,v,z) - λ|²] + normalisation penalty

  'long_horizon': delegate to DeepBSDESolver on [0,T_large] and extract λ.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from typing import Optional

from ebsde.bsde.ergodic import ErgodicBSDE


class _MLP(nn.Module):
    """Plain MLP (no BatchNorm): Linear → Tanh × L → Linear. Supports
    create_graph=True autograd (BN breaks second-order differentiation)."""

    def __init__(self, d_in: int, d_out: int, hidden: list[int]) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        prev = d_in
        for w in hidden:
            layers += [nn.Linear(prev, w), nn.Tanh()]
            prev = w
        layers.append(nn.Linear(prev, d_out))
        self.net = nn.Sequential(*layers)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ErgodicDeepBSDESolver:
    """
    Deep learning solver for ergodic BSDEs.

    Parameters
    ----------
    ergodic_bsde       : ErgodicBSDE
    strategy           : 'temporal_difference' | 'direct_ergodic' | 'long_horizon'
    v_network_layers   : hidden widths for v_net
    z_network_layers   : hidden widths for z_net
    learning_rate      : Adam LR
    n_epochs           : training iterations
    n_samples          : stationary samples per gradient step
    norm_penalty       : weight α for normalisation penalty (direct_ergodic only)
    consist_penalty    : weight β for z–∇v consistency (direct_ergodic only)
    dt                 : time step for TD simulation
    T_long             : horizon for 'long_horizon' strategy
    device             : 'cpu' or 'cuda'
    rng_seed           : numpy seed for sampling
    torch_seed         : torch seed for weight initialisation
    """

    def __init__(
        self,
        ergodic_bsde: ErgodicBSDE,
        strategy: str = "temporal_difference",
        v_network_layers: list = (64, 64, 64),
        z_network_layers: list = (64, 64),
        learning_rate: float = 1e-3,
        n_epochs: int = 5_000,
        n_samples: int = 4_096,
        norm_penalty: float = 1.0,
        consist_penalty: float = 0.1,
        dt: float = 0.01,
        T_long: float = 20.0,
        device: str = "cpu",
        rng_seed: Optional[int] = None,
        torch_seed: Optional[int] = None,
    ) -> None:
        self.ebsde = ergodic_bsde
        self.strategy = strategy
        self.v_network_layers = list(v_network_layers)
        self.z_network_layers = list(z_network_layers)
        self.lr = learning_rate
        self.n_epochs = n_epochs
        self.n_samples = n_samples
        self.norm_penalty = norm_penalty
        self.consist_penalty = consist_penalty
        self.dt = dt
        self.T_long = T_long
        self.device = torch.device(device)
        self.rng_seed = rng_seed

        if torch_seed is not None:
            torch.manual_seed(torch_seed)

        self.d = ergodic_bsde.forward.dimension
        self.rng = np.random.default_rng(rng_seed)

        self._build_networks()

    # ------------------------------------------------------------------
    # Network construction
    # ------------------------------------------------------------------

    def _build_networks(self) -> None:
        d = self.d
        self.lambda_param = nn.Parameter(
            torch.tensor(0.0, device=self.device)
        )
        # Plain MLP (no BatchNorm) so create_graph=True works for second-order
        # autograd (needed for z = σ∇v in TD and Hessian in direct).
        self.v_net = _MLP(d, 1, self.v_network_layers).to(self.device)
        self.z_net = _MLP(d, d, self.z_network_layers).to(self.device)

    def _all_params(self, include_z_net: bool = True):
        params = list(self.v_net.parameters()) + [self.lambda_param]
        if include_z_net:
            params = params + list(self.z_net.parameters())
        return params

    # ------------------------------------------------------------------
    # Stationary sampling
    # ------------------------------------------------------------------

    def _sample_stationary(self, n: int) -> np.ndarray:
        """
        Sample from stationary distribution.
        Uses analytical OU Gaussian if available, else long path subsampling.
        """
        fwd = self.ebsde.forward
        if hasattr(fwd, "stationary_distribution"):
            sd = fwd.stationary_distribution()
            mu_s = sd["mean"]
            std_s = float(np.sqrt(sd["variance"]))
            x = self.rng.normal(mu_s, std_s, size=(n, self.d))
        else:
            T_burn = 100.0
            n_steps = int(T_burn / self.dt)
            sim = fwd.simulate(
                x0=np.zeros(self.d), T=T_burn,
                n_steps=n_steps, n_paths=1, rng=self.rng,
            )
            paths = sim["paths"][0, :, :]  # (n_steps+1, d)
            idx = self.rng.integers(n_steps // 2, n_steps + 1, size=n)
            x = paths[idx]
        return x  # (n, d)

    def _simulate_transitions(
        self, x_curr: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        One Euler step: X_{n+1} = X_n + b(X_n)Δt + σ(X_n)ΔW_n.
        Returns (x_next, dW).
        """
        fwd = self.ebsde.forward
        dt = self.dt
        n = x_curr.shape[0]
        dW = self.rng.standard_normal((n, self.d)) * np.sqrt(dt)
        sigma_x = fwd.diffusion(x_curr)   # (n, d) or scalar
        b_x     = fwd.drift(x_curr)       # (n, d)
        x_next  = x_curr + b_x * dt + sigma_x * dW
        return x_next, dW

    # ------------------------------------------------------------------
    # Loss functions
    # ------------------------------------------------------------------

    def _td_loss(self, x: torch.Tensor) -> torch.Tensor:
        """
        Continuous-time TD (= exact PDE residual) with z = σ∂v/∂x.

        The finite-step TD estimator (v_{n+1}-v_n)/dt approximates Lv with
        O(dt^{-1/2}) noise per sample; at dt=0.02 and N=2048 this gives
        SNR≈0.6, far too noisy to converge.  The dt→0 limit of the squared
        TD residual is exactly E[(Lv + f(x,v,σ∇v) - λ)²], which we compute
        directly and noise-free via autograd.

        z is tied to σ∂v/∂x (no separate z_net), enforcing the martingale
        representation and selecting the unique ergodic solution.
        """
        x = x.requires_grad_(True)
        v  = self.v_net(x).squeeze(-1)            # (N,)

        grad_v = torch.autograd.grad(
            v.sum(), x, create_graph=True
        )[0]                                       # (N, d)

        fwd    = self.ebsde.forward
        x_np   = x.detach().cpu().numpy()
        sigma_np = fwd.diffusion(x_np)             # (N, d)
        b_np     = fwd.drift(x_np)                 # (N, d)
        sigma_t  = torch.tensor(sigma_np.astype(np.float32), device=x.device)
        b_t      = torch.tensor(b_np.astype(np.float32),     device=x.device)

        z = sigma_t * grad_v                       # (N, d)

        # Infinitesimal generator: Lv = b·∇v + (σ²/2)·tr(Hessian v)
        Lv = (b_t * grad_v).sum(-1)
        for j in range(self.d):
            d2v_j = torch.autograd.grad(
                grad_v[:, j].sum(), x, create_graph=True
            )[0][:, j]
            Lv = Lv + 0.5 * sigma_t[:, j] ** 2 * d2v_j

        x_arg = x[:, 0] if self.d == 1 else x
        f_val = self._eval_f_torch(x_arg, v, z)   # (N,)

        pde_residual = (Lv + f_val - self.lambda_param).pow(2).mean()

        # Gauge: pin E_stat[v] = 0  (v is unique up to a constant)
        norm_loss = v.mean().pow(2)
        return pde_residual + self.norm_penalty * norm_loss

    def _direct_loss(self, x: torch.Tensor) -> torch.Tensor:
        """
        Direct ergodic loss:
            E_μ[|Lv + f(x,v,z) - λ|²]
          + α · (E_μ[v(X)])²
          + β · E_μ[|z - σ(X)∇v|²]

        Lv computed via double autograd.
        """
        x = x.requires_grad_(True)
        v = self.v_net(x)           # (N, 1)
        v_sq = v.squeeze(-1)        # (N,)

        # ∂v/∂x via autograd
        grad_v = torch.autograd.grad(
            v.sum(), x, create_graph=True
        )[0]  # (N, d)

        # Lv = (σ²/2) Σ_j ∂²v/∂x_j²  + b·∇v
        fwd = self.ebsde.forward
        x_np = x.detach().cpu().numpy()
        sigma_np = fwd.diffusion(x_np)   # (N, d)
        b_np = fwd.drift(x_np)           # (N, d)

        sigma_t = torch.tensor(sigma_np, dtype=x.dtype, device=x.device)
        b_t = torch.tensor(b_np, dtype=x.dtype, device=x.device)

        # Lv: drift part
        Lv = (b_t * grad_v).sum(-1)  # (N,)

        # Lv: diffusion part — trace of Hessian weighted by σ²/2
        for j in range(self.d):
            dv_j = grad_v[:, j]
            d2v_j = torch.autograd.grad(
                dv_j.sum(), x, create_graph=True
            )[0][:, j]  # (N,) — diagonal of Hessian
            Lv = Lv + 0.5 * sigma_t[:, j]**2 * d2v_j

        z_curr = self.z_net(x)  # (N, d)
        x_arg = x[:, 0] if self.d == 1 else x
        f_val = self._eval_f_torch(x_arg, v_sq, z_curr)

        pde_residual = (Lv + f_val - self.lambda_param).pow(2).mean()

        # Normalisation: pin gauge by penalising non-zero batch mean of v
        norm_loss = v_sq.mean().pow(2)

        # z–∇v consistency
        z_ref = sigma_t * grad_v
        consist_loss = (z_curr - z_ref).pow(2).mean()

        return pde_residual + self.norm_penalty * norm_loss + self.consist_penalty * consist_loss

    # ------------------------------------------------------------------
    # Driver evaluation (torch)
    # ------------------------------------------------------------------

    def _eval_f_torch(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        z: torch.Tensor,
    ) -> torch.Tensor:
        """
        Evaluate f(x, y, z) with autograd preserved through z.

        Monkey-patches numpy reduction functions (sum, mean, etc.) to dispatch
        to torch equivalents when called with torch tensors, so that drivers
        written with numpy operations work transparently under autograd.
        """
        import numpy as _np

        _saved: dict = {}

        def _torch_dispatch(np_name: str, torch_fn):
            orig = getattr(_np, np_name)
            _saved[np_name] = orig

            def _wrapped(a, axis=None, **kw):
                if isinstance(a, torch.Tensor):
                    kw.pop("out", None)
                    kw.pop("keepdims", None)
                    kw.pop("where", None)
                    if axis is not None:
                        return torch_fn(a, dim=axis)
                    return torch_fn(a)
                return orig(a, axis=axis, **kw)

            setattr(_np, np_name, _wrapped)

        for _name, _tfn in [
            ("sum",  torch.sum),
            ("mean", torch.mean),
        ]:
            _torch_dispatch(_name, _tfn)

        try:
            result = self.ebsde.driver(x, y, z)
            if not isinstance(result, torch.Tensor):
                result = torch.as_tensor(np.asarray(result), dtype=y.dtype, device=y.device)
        except Exception:
            result = None
        finally:
            for _name, _orig in _saved.items():
                setattr(_np, _name, _orig)

        if result is not None:
            return result

        # Fallback: estimate z-quadratic coefficient using a fixed test z,
        # then return a differentiable approximation h(x,y) - (γ/2)|z|².
        x_np = x.detach().cpu().numpy()
        y_np = y.detach().cpu().numpy()
        z_np = z.detach().cpu().numpy().reshape(len(y_np), self.d)

        h_np = self.ebsde.driver(x_np, y_np, np.zeros_like(z_np))
        h_t  = torch.tensor(h_np.astype(np.float32), dtype=y.dtype, device=y.device)

        z_test    = np.ones_like(z_np) * 0.5
        f_test_np = self.ebsde.driver(x_np, y_np, z_test)
        correction    = float(np.mean(f_test_np - h_np))
        z_test_sq_mean = float(np.mean((z_test ** 2).sum(axis=-1)))
        gamma_half = -correction / z_test_sq_mean if z_test_sq_mean > 1e-12 else 0.0

        return h_t - gamma_half * (z * z).sum(-1)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def solve(self, verbose: bool = False) -> dict:
        """
        Train and return results.

        Returns
        -------
        dict:
          'lambda_ergodic'  : learned λ
          'v_function'      : callable x → v(x)  (numpy in, numpy out)
          'z_function'      : callable x → z(x)
          'training_loss'   : list[float]
          'lambda_history'  : list[float]
          'v_values'        : v evaluated on a grid
          'z_values'        : z evaluated on a grid
        """
        if self.strategy == "long_horizon":
            return self._solve_long_horizon()

        # TD strategy trains only v_net + lambda (z derived from ∇v_net).
        # Direct strategy trains v_net + z_net + lambda jointly.
        use_z_net = (self.strategy != "temporal_difference")
        optimizer = torch.optim.Adam(self._all_params(include_z_net=use_z_net), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.n_epochs, eta_min=self.lr * 0.01
        )

        loss_history: list[float] = []
        lambda_history: list[float] = []

        # Pre-sample stationary pool
        pool_size = self.n_samples * 20
        x_pool = self._sample_stationary(pool_size)

        refresh_every = max(100, self.n_epochs // 10)

        for epoch in range(self.n_epochs):
            if epoch > 0 and epoch % refresh_every == 0:
                x_pool = self._sample_stationary(pool_size)

            # Mini-batch
            idx = self.rng.integers(0, pool_size, size=self.n_samples)
            x_np = x_pool[idx].astype(np.float32)

            if self.strategy == "temporal_difference":
                x_t  = torch.tensor(x_np, device=self.device)
                loss = self._td_loss(x_t)

            elif self.strategy == "direct_ergodic":
                x_t = torch.tensor(x_np, device=self.device)
                loss = self._direct_loss(x_t)

            else:
                raise ValueError(f"Unknown strategy '{self.strategy}'")

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self._all_params(include_z_net=use_z_net), 1.0)
            optimizer.step()
            scheduler.step()

            loss_history.append(float(loss.item()))
            lambda_history.append(float(self.lambda_param.item()))

            if verbose and epoch % 500 == 0:
                print(f"  epoch {epoch}: loss={loss.item():.6f}, λ={self.lambda_param.item():.6f}")

        lambda_ergodic = float(self.lambda_param.item())

        # Evaluate on grid for output
        fwd = self.ebsde.forward
        if hasattr(fwd, "stationary_distribution"):
            sd = fwd.stationary_distribution()
            mu_s = sd["mean"]
            std_s = float(np.sqrt(sd["variance"]))
            x_eval = np.linspace(mu_s - 4 * std_s, mu_s + 4 * std_s, 200)
        else:
            x_eval = np.linspace(-4, 4, 200)

        x_eval_np = x_eval.reshape(-1, 1).astype(np.float32)
        x_eval_t  = torch.tensor(x_eval_np, device=self.device)

        self.v_net.eval()
        self.z_net.eval()

        with torch.no_grad():
            v_vals = self.v_net(x_eval_t).squeeze(-1).cpu().numpy()

        # z grid: TD strategy uses σ∇v_net; direct strategy uses z_net
        if self.strategy == "temporal_difference":
            x_eval_g = x_eval_t.requires_grad_(True)
            v_eval = self.v_net(x_eval_g).squeeze(-1)
            grad_eval = torch.autograd.grad(v_eval.sum(), x_eval_g)[0]
            sigma_eval = self.ebsde.forward.diffusion(x_eval_np)
            sigma_et = torch.tensor(sigma_eval.astype(np.float32), device=self.device)
            z_vals = (sigma_et * grad_eval).detach().cpu().numpy()
        else:
            with torch.no_grad():
                z_vals = self.z_net(x_eval_t).cpu().numpy()

        # Normalise v: v(x_0) = 0 (zero at midpoint)
        mid = len(v_vals) // 2
        v_vals = v_vals - v_vals[mid]

        def v_function(x_in: np.ndarray) -> np.ndarray:
            self.v_net.eval()
            x_t = torch.tensor(
                np.asarray(x_in, dtype=np.float32).reshape(-1, self.d),
                device=self.device,
            )
            with torch.no_grad():
                return self.v_net(x_t).squeeze(-1).cpu().numpy()

        def z_function(x_in: np.ndarray) -> np.ndarray:
            self.v_net.eval()
            x_t = torch.tensor(
                np.asarray(x_in, dtype=np.float32).reshape(-1, self.d),
                device=self.device,
                requires_grad=True,
            )
            v_t = self.v_net(x_t).squeeze(-1)
            grad_v = torch.autograd.grad(v_t.sum(), x_t)[0]           # (N, d)
            sigma_np2 = self.ebsde.forward.diffusion(x_t.detach().cpu().numpy())
            sigma_t2  = torch.tensor(sigma_np2.astype(np.float32), device=self.device)
            if self.strategy == "temporal_difference":
                return (sigma_t2 * grad_v).detach().cpu().numpy()     # σ∇v
            # direct: return z_net output
            with torch.no_grad():
                self.z_net.eval()
                x_t2 = torch.tensor(
                    np.asarray(x_in, dtype=np.float32).reshape(-1, self.d),
                    device=self.device,
                )
                return self.z_net(x_t2).cpu().numpy()

        return {
            "lambda_ergodic": lambda_ergodic,
            "v_function": v_function,
            "z_function": z_function,
            "training_loss": loss_history,
            "lambda_history": lambda_history,
            "v_values": v_vals,
            "z_values": z_vals,
            "x_grid": x_eval,
        }

    # ------------------------------------------------------------------
    # Long-horizon strategy
    # ------------------------------------------------------------------

    def _solve_long_horizon(self) -> dict:
        """
        Delegate to ErgodicPicardSolver (deep version) but using
        DeepBSDESolver on [0, T_long] and extracting λ.
        """
        from ebsde.solvers.ergodic_picard import ErgodicPicardSolver

        T_vals = [self.T_long / 4, self.T_long / 2, self.T_long]
        picard = ErgodicPicardSolver(
            self.ebsde,
            T_values=T_vals,
            n_paths=self.n_samples,
            n_steps_per_unit=20,
            n_picard=3,
            basis_degree=4,
            rng_seed=self.rng_seed,
        )
        res = picard.solve()

        loss_history: list[float] = []
        lambda_history: list[float] = [res["lambda_ergodic"]]

        def v_function(x_in):
            return np.zeros(np.asarray(x_in).shape[0])

        def z_function(x_in):
            return np.zeros((np.asarray(x_in).shape[0], self.d))

        return {
            "lambda_ergodic": res["lambda_ergodic"],
            "v_function": v_function,
            "z_function": z_function,
            "training_loss": loss_history,
            "lambda_history": lambda_history,
            "v_values": np.array([res["v_at_x0"]]),
            "z_values": np.zeros((1, self.d)),
            "x_grid": np.array([0.0]),
        }
