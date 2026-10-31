"""Flow-matching diffusion process over continuous token embeddings.

Supports both whole-sequence diffusion and block diffusion, where timesteps
are per-position: clean prefix tokens carry t=0 and only the current block is
denoised.
"""

from __future__ import annotations

import torch
from torch.nn import functional as F

from diffusion_llm.model import DiffusionLLM


class FlowMatching:
    """Rectified-flow path between clean token embeddings and isotropic noise.

    Interpolation (for t in [0, 1]):

        x_t = (1 - sigma_min) * (1 - t) * x_0 + t * eps

    and the model learns the velocity field v(x_t, t) = d x_t / dt.
    """

    def __init__(self, sigma_min: float = 1e-4) -> None:
        self.sigma_min = sigma_min

    def sample_timesteps(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return torch.rand(batch_size, device=device)

    def interpolate(self, x0: torch.Tensor, noise: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        t = t.view(-1, 1, 1) if t.dim() == 1 else t.unsqueeze(-1)
        s = self.sigma_min
        return (1.0 - s) * (1.0 - t) * x0 + t * noise

    def velocity_target(self, x0: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        return noise - (1.0 - self.sigma_min) * x0

    def compute_loss(
        self,
        model: DiffusionLLM,
        x0: torch.Tensor,
        t: torch.Tensor,
        loss_mask: torch.Tensor | None = None,
        attn_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Flow-matching MSE. `loss_mask` (B, T) selects block positions to count;
        `attn_mask` is the causal/block attention mask passed to the model."""
        noise = torch.randn_like(x0)
        x_t = self.interpolate(x0, noise, t)
        velocity_pred = model(x_t, t, attn_mask=attn_mask)
        velocity_target = self.velocity_target(x0, noise)
        if loss_mask is None:
            return F.mse_loss(velocity_pred, velocity_target)
        diff = (velocity_pred - velocity_target).pow(2).mean(dim=-1)  # (B, T)
        return (diff * loss_mask).sum() / loss_mask.sum().clamp(min=1.0)

    @torch.no_grad()
    def sample(
        self,
        model: DiffusionLLM,
        batch_size: int,
        seq_len: int,
        steps: int = 16,
    ) -> torch.Tensor:
        """Generate embeddings for a whole sequence from pure noise (pure diffusion).

        Uses the midpoint (2nd-order Heun) rule.
        """
        device = next(model.parameters()).device
        x = torch.randn(batch_size, seq_len, model.cfg.d_model, device=device)
        dt = 1.0 / steps
        for i in range(steps):
            t_cur = torch.full((batch_size,), 1.0 - i * dt, device=device)
            v1 = model(x, t_cur)
            t_mid = torch.full((batch_size,), 1.0 - (i + 0.5) * dt, device=device)
            x_mid = x - 0.5 * dt * v1
            v2 = model(x_mid, t_mid)
            x = x - dt * v2
        return x
