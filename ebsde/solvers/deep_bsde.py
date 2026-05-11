"""
Deep BSDE solver following Han, Jentzen & E (2018).

Reference: "Solving high-dimensional partial differential equations using
deep learning", PNAS 2018.

Architecture
------------
- Y_0 ∈ ℝ is a trainable scalar parameter.
- Z_i = SubNet_i(X_i) ∈ ℝ^d, one sub-network per time step (per_step)
  or one shared sub-network (shared).
- Forward pass: Y_{i+1} = Y_i − f(X_i, Y_i, Z_i)·Δt + Z_i·ΔW_i
- Loss: E[|Y_M − g(X_M)|²]

Training
--------
Adam optimizer with warm cosine learning-rate schedule.
Paths are pre-simulated in a pool and refreshed periodically.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from typing import Optional, Callable

from ebsde.bsde.standard import StandardBSDE


# ------------------------------------------------------------------
# SubNet
# ------------------------------------------------------------------


class SubNet(nn.Module):
    """
    Fully-connected sub-network for approximating Z_i(x).

    Architecture: Linear → BN → Act → [Linear → BN → Act] × (L-1) → Linear

    Parameters
    ----------
    d_in        : input dimension (= d)
    d_out       : output dimension (= d)
    hidden_layers : list of hidden layer widths
    activation  : 'relu' | 'tanh' | 'silu'
    """

    def __init__(
        self,
        d_in: int,
        d_out: int,
        hidden_layers: list[int],
        activation: str = "relu",
    ) -> None:
        super().__init__()

        act_fn = {
            "relu": nn.ReLU,
            "tanh": nn.Tanh,
            "silu": nn.SiLU,
        }.get(activation)
        if act_fn is None:
            raise ValueError(f"Unknown activation '{activation}'")

        layers: list[nn.Module] = []
        prev = d_in
        for width in hidden_layers:
            layers.append(nn.Linear(prev, width))
            layers.append(nn.BatchNorm1d(width))
            layers.append(act_fn())
            prev = width
        layers.append(nn.Linear(prev, d_out))

        self.net = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ------------------------------------------------------------------
# DeepBSDESolver
# ------------------------------------------------------------------


class DeepBSDESolver:
    """
    Deep BSDE solver (Han-Jentzen-E 2018).

    Parameters
    ----------
    bsde          : StandardBSDE
    n_steps       : time discretisation steps M
    n_paths_train : MC paths per training batch
    hidden_layers : sub-network hidden widths
    activation    : activation function name
    network_type  : 'per_step' (one SubNet per time step) or
                    'shared' (one SubNet for all time steps, augmented with t)
    n_epochs      : gradient-descent iterations
    batch_size    : paths per gradient step
    lr            : initial learning rate
    grad_clip     : gradient clipping norm (None to disable)
    rng_seed      : numpy random seed for path simulation
    torch_seed    : torch seed for network initialisation
    torch_driver  : optional differentiable driver callable
                    f(x: Tensor, y: Tensor, z: Tensor) → Tensor
                    If None, falls back to bsde.driver with numpy-detach.
    torch_terminal: optional differentiable terminal callable
                    g(x: Tensor) → Tensor
                    If None, falls back to bsde.terminal via numpy.
    device        : torch device string ('cpu' by default)
    """

    def __init__(
        self,
        bsde: StandardBSDE,
        n_steps: int = 20,
        n_paths_train: int = 8_192,
        hidden_layers: list[int] = (64, 64),
        activation: str = "relu",
        network_type: str = "per_step",
        n_epochs: int = 2_000,
        batch_size: int = 256,
        lr: float = 1e-3,
        grad_clip: Optional[float] = 1.0,
        rng_seed: Optional[int] = None,
        torch_seed: Optional[int] = None,
        torch_driver: Optional[Callable] = None,
        torch_terminal: Optional[Callable] = None,
        device: str = "cpu",
    ) -> None:
        self.bsde = bsde
        self.n_steps = n_steps
        self.n_paths_train = n_paths_train
        self.hidden_layers = list(hidden_layers)
        self.activation = activation
        self.network_type = network_type
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.lr = lr
        self.grad_clip = grad_clip
        self.rng_seed = rng_seed
        self.torch_seed = torch_seed
        self.torch_driver = torch_driver
        self.torch_terminal = torch_terminal
        self.device = torch.device(device)

        self.rng = np.random.default_rng(rng_seed)
        if torch_seed is not None:
            torch.manual_seed(torch_seed)

        self.d = bsde.forward.dimension
        self.dt = bsde.T / n_steps
        self.times = np.linspace(0.0, bsde.T, n_steps + 1)

        self._build_model()

    # ------------------------------------------------------------------
    # Model construction
    # ------------------------------------------------------------------

    def _build_model(self) -> None:
        d = self.d
        h = self.hidden_layers
        act = self.activation

        if self.network_type == "per_step":
            self.z_nets = nn.ModuleList(
                [SubNet(d, d, h, act) for _ in range(self.n_steps)]
            ).to(self.device)

        elif self.network_type == "shared":
            # Shared net takes (x, t) as input → augmented input dim = d + 1
            self.z_nets = SubNet(d + 1, d, h, act).to(self.device)

        else:
            raise ValueError(f"Unknown network_type '{self.network_type}'")

        self.Y0_param = nn.Parameter(
            torch.zeros(1, device=self.device)
        )

    def _get_z(self, step: int, x: torch.Tensor) -> torch.Tensor:
        """Evaluate Z-network at time step `step` and state `x` (N, d)."""
        if self.network_type == "per_step":
            return self.z_nets[step](x)
        else:
            t_val = self.times[step]
            t_col = torch.full(
                (x.shape[0], 1), t_val,
                dtype=x.dtype, device=x.device,
            )
            return self.z_nets(torch.cat([x, t_col], dim=-1))

    # ------------------------------------------------------------------
    # Driver / terminal evaluation (torch-compatible)
    # ------------------------------------------------------------------

    def _eval_driver_torch(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        z: torch.Tensor,
    ) -> torch.Tensor:
        if self.torch_driver is not None:
            return self.torch_driver(x, y, z)

        # Try calling the numpy driver directly with tensors — works for many
        # simple lambdas (arithmetic, comparisons, etc.)
        try:
            result = self.bsde.driver(x, y, z)
            if isinstance(result, torch.Tensor):
                return result
            return torch.as_tensor(result, dtype=y.dtype, device=y.device)
        except Exception:
            # Fallback: numpy evaluation (no gradient through driver w.r.t. z)
            x_np = x.detach().cpu().numpy()
            y_np = y.detach().cpu().numpy()
            z_np = z.detach().cpu().numpy()
            # Squeeze spatial dims for 1-d case
            if self.d == 1:
                x_np = x_np[:, 0]
            f_np = self.bsde.eval_driver(x_np, y_np, z_np)
            return torch.tensor(f_np, dtype=y.dtype, device=y.device)

    def _eval_terminal_numpy(self, x_paths: np.ndarray) -> np.ndarray:
        """Evaluate terminal g(X_T) in numpy — used to compute target."""
        x_T = x_paths[:, -1, 0] if self.d == 1 else x_paths[:, -1, :]
        return self.bsde.eval_terminal(x_T)

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def _forward_pass(
        self,
        paths: np.ndarray,   # (N, M+1, d)
        dW: np.ndarray,      # (N, M, d)
    ) -> torch.Tensor:
        """
        Run the deep BSDE forward pass on a batch.

        Returns Y_M : (N,) tensor.
        """
        N = paths.shape[0]
        dtype = torch.float32

        # Y initialised from the learned scalar Y_0
        Y = self.Y0_param.expand(N)  # (N,)

        for i in range(self.n_steps):
            x_i_np = paths[:, i, :]               # (N, d)
            dW_i_np = dW[:, i, :]                  # (N, d)

            x_i  = torch.tensor(x_i_np, dtype=dtype, device=self.device)
            dW_i = torch.tensor(dW_i_np, dtype=dtype, device=self.device)

            Z_i = self._get_z(i, x_i)             # (N, d)

            x_arg = x_i[:, 0] if self.d == 1 else x_i
            f_val = self._eval_driver_torch(x_arg, Y, Z_i)   # (N,)

            Y = Y - f_val * self.dt + (Z_i * dW_i).sum(dim=-1)

        return Y   # (N,)

    # ------------------------------------------------------------------
    # Path simulation helpers
    # ------------------------------------------------------------------

    def _simulate_pool(self) -> tuple[np.ndarray, np.ndarray]:
        """Simulate a fresh pool of training paths."""
        bsde = self.bsde
        sim = bsde.forward.simulate(
            x0=np.zeros(self.d),
            T=bsde.T,
            n_steps=self.n_steps,
            n_paths=self.n_paths_train,
            rng=self.rng,
        )
        return sim["paths"], sim["brownian_increments"]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def solve(self) -> dict:
        """
        Train the deep BSDE model and return results.

        Returns
        -------
        dict
            Y0          : float — learned Y_0
            Z0          : (d,) — mean Z estimate at t=0
            loss_history: list[float] — training loss per epoch
            model       : self (for further evaluation)
        """
        refresh_every = max(50, self.n_epochs // 20)

        # Collect all parameters
        if self.network_type == "per_step":
            z_params = list(self.z_nets.parameters())
        else:
            z_params = list(self.z_nets.parameters())

        optimizer = torch.optim.Adam(
            [self.Y0_param] + z_params, lr=self.lr
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.n_epochs, eta_min=self.lr * 0.01
        )

        # Initial path pool
        paths, dW = self._simulate_pool()
        N_pool = paths.shape[0]

        loss_history: list[float] = []

        for epoch in range(self.n_epochs):
            # Refresh path pool periodically
            if epoch > 0 and epoch % refresh_every == 0:
                paths, dW = self._simulate_pool()
                N_pool = paths.shape[0]

            # Sample mini-batch
            idx = self.rng.integers(0, N_pool, size=self.batch_size)
            batch_paths = paths[idx]   # (B, M+1, d)
            batch_dW    = dW[idx]      # (B, M, d)

            # Pre-compute targets (numpy, no grad needed)
            g_vals = self._eval_terminal_numpy(batch_paths)   # (B,)
            target = torch.tensor(g_vals, dtype=torch.float32, device=self.device)

            # Forward pass
            # Set nets to train mode
            if self.network_type == "per_step":
                self.z_nets.train()
            else:
                self.z_nets.train()

            Y_M = self._forward_pass(batch_paths, batch_dW)  # (B,)

            loss = torch.mean((Y_M - target) ** 2)

            optimizer.zero_grad()
            loss.backward()

            if self.grad_clip is not None:
                nn.utils.clip_grad_norm_(
                    [self.Y0_param] + z_params, self.grad_clip
                )

            optimizer.step()
            scheduler.step()

            loss_history.append(float(loss.item()))

        # --- Post-training evaluation ---
        if self.network_type == "per_step":
            self.z_nets.eval()
        else:
            self.z_nets.eval()

        Y0 = float(self.Y0_param.item())

        # Estimate Z0 = mean Z_0(X_0) over a fresh evaluation batch
        eval_paths, _ = self._simulate_pool()
        x0_batch = eval_paths[:, 0, :]                       # (N_pool, d)
        x0_t = torch.tensor(x0_batch, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            Z0_t = self._get_z(0, x0_t)                      # (N_pool, d)
        Z0 = Z0_t.cpu().numpy().mean(axis=0)                 # (d,)

        return {
            "Y0": Y0,
            "Z0": Z0,
            "loss_history": loss_history,
            "model": self,
        }

    def evaluate(
        self,
        n_eval_paths: int = 10_000,
        rng_seed: Optional[int] = None,
    ) -> dict:
        """
        Post-training Monte-Carlo evaluation of Y_0.

        Uses the learned Z-networks on fresh paths.

        Returns
        -------
        dict with 'Y0_mean', 'Y0_std', 'Y_terminal'.
        """
        rng = np.random.default_rng(rng_seed)
        bsde = self.bsde
        sim = bsde.forward.simulate(
            x0=np.zeros(self.d),
            T=bsde.T,
            n_steps=self.n_steps,
            n_paths=n_eval_paths,
            rng=rng,
        )
        paths = sim["paths"]
        dW    = sim["brownian_increments"]

        if self.network_type == "per_step":
            self.z_nets.eval()
        else:
            self.z_nets.eval()

        with torch.no_grad():
            Y_M = self._forward_pass(paths, dW)

        Y_M_np = Y_M.detach().cpu().numpy()

        Y0_paths = float(self.Y0_param.item())
        return {
            "Y0_mean": Y0_paths,
            "Y0_std": float(np.std(Y_M_np - self._eval_terminal_numpy(paths))),
            "Y_terminal": Y_M_np,
        }
