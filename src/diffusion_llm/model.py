"""Transformer backbone for continuous-embedding diffusion LLMs (attention shape set by masks).

The model takes a *noisy* sequence of token embeddings plus per-position
timesteps and predicts the velocity field of the flow-matching diffusion
process. It supports two conditioning modes:

- pure diffusion: one timestep for the whole sequence (t of shape (B,)),
- block diffusion: clean prefix tokens get t=0, the denoised block gets a
  sampled timestep (t of shape (B, T)).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

# Fixed inverse temperature for cosine-similarity logits.
LOGIT_SCALE = 12.0


@dataclass(frozen=True)
class ModelConfig:
    vocab_size: int = 8192  # BPE target size; the trained tokenizer's real size is used
    d_model: int = 256
    n_layers: int = 6
    n_heads: int = 8
    ffn_mult: int = 3
    max_seq_len: int = 2048
    rope_base: float = 10_000.0
    qk_norm: bool = True

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_heads


def timestep_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    """Sinusoidal timestep embedding. Accepts t of shape (B,) or (B, T)."""
    if dim % 2 != 0:
        raise ValueError("dim must be even")
    half = dim // 2
    freqs = torch.exp(
        -math.log(10_000.0) * torch.arange(half, dtype=torch.float32, device=t.device) / half
    )
    args = t.to(torch.float32).unsqueeze(-1) * freqs
    return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).sqrt()
        return x / rms * self.weight


class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, max_seq_len: int, base: float = 10_000.0) -> None:
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        positions = torch.arange(max_seq_len, dtype=torch.float32)
        freqs = torch.outer(positions, inv_freq)  # (max_seq_len, dim / 2)
        emb = torch.cat([freqs, freqs], dim=-1)  # (max_seq_len, dim)
        self.register_buffer("cos_cached", emb.cos().unsqueeze(0))
        self.register_buffer("sin_cached", emb.sin().unsqueeze(0))

    def forward(self, x: torch.Tensor, seq_start: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
        seq_len = x.size(1)
        cos = self.cos_cached[:, seq_start : seq_start + seq_len].unsqueeze(1)  # (1, 1, T, dim)
        sin = self.sin_cached[:, seq_start : seq_start + seq_len].unsqueeze(1)
        return cos, sin


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    q = (q * cos) + (_rotate_half(q) * sin)
    k = (k * cos) + (_rotate_half(k) * sin)
    return q, k


class Attention(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.n_heads = cfg.n_heads
        self.head_dim = cfg.head_dim
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=False)
        self.proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        if cfg.qk_norm:
            self.q_norm = RMSNorm(cfg.head_dim)
            self.k_norm = RMSNorm(cfg.head_dim)
        else:
            self.q_norm = None
            self.k_norm = None

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
        kv_cache: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        batch, seq, _ = x.shape
        qkv = self.qkv(x).view(batch, seq, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.unbind(2)  # each (B, T, H, Dh)
        q = q.transpose(1, 2)  # -> (B, H, T, Dh) for RoPE + attention
        k = k.transpose(1, 2)
        if self.q_norm is not None and self.k_norm is not None:
            q = self.q_norm(q)
            k = self.k_norm(k)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        v = v.transpose(1, 2)
        cache_row = (k, v)  # this block's own keys/values (post-RoPE)
        if kv_cache is not None:
            k_cache, v_cache = kv_cache
            k = torch.cat([k_cache, k], dim=2)
            v = torch.cat([v_cache, v], dim=2)
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        y = y.transpose(1, 2).contiguous().view(batch, seq, -1)
        return self.proj(y), cache_row


class SwiGLU(nn.Module):
    def __init__(self, dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(dim, hidden_dim, bias=False)
        self.w3 = nn.Linear(hidden_dim, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w3(F.silu(self.w1(x)) * self.w2(x))


class Block(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.ln1 = RMSNorm(cfg.d_model)
        self.attn = Attention(cfg)
        self.ln2 = RMSNorm(cfg.d_model)
        self.ffn = SwiGLU(cfg.d_model, cfg.d_model * cfg.ffn_mult)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
        kv_cache: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        attn_out, cache_row = self.attn(self.ln1(x), cos, sin, attn_mask, kv_cache)
        x = x + attn_out
        x = x + self.ffn(self.ln2(x))
        return x, cache_row


class DiffusionLLM(nn.Module):
    """Continuous-diffusion language model.

    - ``forward(x_t, t)`` returns the velocity field (same shape as input embeddings).
    - ``decode_logits(x)`` maps a denoised embedding sequence to vocabulary logits
      through the tied token-embedding matrix (cosine similarity).
    """

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.token_embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.time_embed = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_model * 4),
            nn.GELU(),
            nn.Linear(cfg.d_model * 4, cfg.d_model),
        )
        self.rope = RotaryEmbedding(cfg.head_dim, cfg.max_seq_len, cfg.rope_base)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layers)])
        self.norm_f = RMSNorm(cfg.d_model)
        self.vel_head = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        with torch.no_grad():
            if isinstance(module, nn.Linear):
                module.weight.normal_(0.0, 0.02)
                if module.bias is not None:
                    module.bias.zero_()
            elif isinstance(module, nn.Embedding):
                module.weight.normal_(0.0, 0.02)

    def embed_tokens(self, ids: torch.Tensor) -> torch.Tensor:
        return self.token_embed(ids)

    def forward(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
        cache: list[tuple[torch.Tensor, torch.Tensor]] | None = None,
        cache_out: list[tuple[torch.Tensor, torch.Tensor]] | None = None,
        seq_start: int = 0,
    ) -> torch.Tensor:
        t_emb = self.time_embed(timestep_embedding(t, self.cfg.d_model))
        if t_emb.dim() == 2:  # t was (B,) -> (B, 1, D)
            t_emb = t_emb.unsqueeze(1)
        h = x_t + t_emb
        cos, sin = self.rope(h, seq_start)
        for index, block in enumerate(self.blocks):
            kv = cache[index] if cache is not None else None
            h, cache_row = block(h, cos, sin, attn_mask, kv)
            if cache_out is not None:
                cache_out.append(cache_row)
        h = self.norm_f(h)
        return self.vel_head(h)

    def decode_logits(self, x: torch.Tensor) -> torch.Tensor:
        # Cosine-similarity decoding against the tied embedding matrix: removes
        # embedding-norm bias that otherwise collapses parallel argmax decoding.
        x_norm = F.normalize(x, dim=-1)
        w_norm = F.normalize(self.token_embed.weight, dim=-1)
        return x_norm @ w_norm.T * LOGIT_SCALE
