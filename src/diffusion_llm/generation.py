"""Block-wise parallel decoding for diffusion LLMs.

Generation is autoregressive across blocks:
- previously generated tokens form a clean context (t = 0), served from a KV
  cache or recomputed every step,
- the current block starts from pure noise and is denoised conditioned on that
  context (optional classifier-free guidance),
- after each block, low-confidence tokens can be refined by re-noising and
  denoising them again while high-confidence tokens stay anchored (adaptive
  decoding / Mercury-style refinement).
"""

from __future__ import annotations

from collections.abc import Callable

import torch
from torch.nn import functional as F

from diffusion_llm.model import DiffusionLLM

KVCache = list[tuple[torch.Tensor, torch.Tensor]]
VelocityFn = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


def make_causal_mask(size: int, device: torch.device) -> torch.Tensor:
    return torch.tril(torch.ones(size, size, dtype=torch.bool, device=device))


def make_block_mask(prefix_len: int, block_len: int, device: torch.device) -> torch.Tensor:
    """Mask for block queries: row i (block position i) may attend to columns
    j <= prefix_len + i (prefix + earlier positions in the current block)."""
    cols = torch.arange(prefix_len + block_len, device=device)
    rows = torch.arange(block_len, device=device).unsqueeze(1)
    return rows + prefix_len >= cols


@torch.no_grad()
def decode_tokens(logits: torch.Tensor, temperature: float) -> tuple[list[int], torch.Tensor]:
    """Sample token ids and return per-position confidence (max softmax prob)."""
    if temperature > 0.0:
        probs = F.softmax(logits / temperature, dim=-1)
        ids = torch.multinomial(probs[0], num_samples=1).squeeze(-1).tolist()
    else:
        probs = F.softmax(logits, dim=-1)
        ids = logits.argmax(dim=-1)[0].tolist()
    confidence = probs.max(dim=-1).values[0]  # (N,)
    return ids, confidence


@torch.no_grad()
def prefill_cache(model: DiffusionLLM, ids: list[int], device: torch.device) -> KVCache:
    """Run the clean prefix (t = 0) once and return its per-layer K/V."""
    emb = model.embed_tokens(torch.tensor([ids], dtype=torch.long, device=device))
    cache: KVCache = []
    model(
        emb,
        torch.zeros(1, len(ids), device=device),
        attn_mask=make_causal_mask(len(ids), device),
        cache_out=cache,
    )
    return cache


def _make_velocity_fn(
    model: DiffusionLLM,
    prefix: list[int],
    cache: KVCache | None,
    n_new: int,
    guidance_scale: float,
    device: torch.device,
) -> VelocityFn:
    """Velocity of the current block conditioned on `prefix` (cached or recomputed)."""
    prefix_len = len(prefix)
    block_only_mask = make_causal_mask(n_new, device)

    def unconditional(x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return model(x, t, attn_mask=block_only_mask)

    if prefix_len == 0:
        conditional = unconditional
    elif cache is not None:
        block_mask = make_block_mask(prefix_len, n_new, device)

        def conditional(x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
            return model(x, t, attn_mask=block_mask, cache=cache, seq_start=prefix_len)
    else:
        prefix_emb = model.embed_tokens(torch.tensor([prefix], dtype=torch.long, device=device))
        prefix_t = torch.zeros(1, prefix_len, device=device)
        full_mask = make_causal_mask(prefix_len + n_new, device)

        def conditional(x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
            h = torch.cat([prefix_emb, x], dim=1)
            t_full = torch.cat([prefix_t, t], dim=1)
            return model(h, t_full, attn_mask=full_mask)[:, prefix_len:]

    if guidance_scale <= 0.0 or prefix_len == 0:
        return conditional

    def guided(x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        v_uncond = unconditional(x, t)
        return v_uncond + guidance_scale * (conditional(x, t) - v_uncond)

    return guided


@torch.no_grad()
def _denoise(
    velocity: VelocityFn,
    x: torch.Tensor,
    steps: int,
    update: torch.Tensor | None = None,
) -> torch.Tensor:
    """Euler-integrate from t=1 to t=0; `update` restricts which positions move."""
    dt = 1.0 / steps
    for i in range(steps):
        t = torch.full(x.shape[:2], 1.0 - i * dt, device=x.device)
        v = velocity(x, t)
        if update is None:
            x = x - dt * v
        else:
            x[:, update] = x[:, update] - dt * v[:, update]
    return x


@torch.no_grad()
def _refine_block(
    model: DiffusionLLM,
    velocity: VelocityFn,
    ids: list[int],
    confidence: torch.Tensor,
    *,
    steps: int,
    temperature: float,
    threshold: float,
    rounds: int,
) -> list[int]:
    """Adaptive decoding: re-noise and denoise low-confidence positions while
    keeping confident tokens anchored at their clean embeddings."""
    device = confidence.device
    for _ in range(rounds):
        keep = confidence >= threshold
        if keep.all():
            break
        anchored = model.embed_tokens(torch.tensor([ids], dtype=torch.long, device=device))
        x = torch.randn_like(anchored)
        x[:, keep] = anchored[:, keep]
        x = _denoise(velocity, x, steps, update=~keep)
        ids, confidence = decode_tokens(model.decode_logits(x), temperature)
    return ids


@torch.no_grad()
def generate_blocks(
    model: DiffusionLLM,
    *,
    block_len: int,
    length: int,
    steps: int,
    temperature: float,
    guidance_scale: float,
    device: torch.device,
    use_cache: bool = True,
    refine_rounds: int = 2,
    refine_steps: int = 4,
    confidence_threshold: float = 0.7,
) -> list[int]:
    """Generate `length` tokens block by block."""
    model.eval()
    generated: list[int] = []
    cache: KVCache | None = None

    while len(generated) < length:
        n_new = min(block_len, length - len(generated))
        prefix = generated[-(model.cfg.max_seq_len - n_new) :]

        if use_cache and prefix:
            cached_len = cache[0][0].size(2) if cache is not None else 0
            if cached_len != len(prefix):
                # Window slid (or first block with context): RoPE positions changed, rebuild.
                cache = prefill_cache(model, prefix, device)

        velocity = _make_velocity_fn(
            model, prefix, cache if use_cache else None, n_new, guidance_scale, device
        )
        x = _denoise(velocity, torch.randn(1, n_new, model.cfg.d_model, device=device), steps)
        ids, confidence = decode_tokens(model.decode_logits(x), temperature)

        if refine_rounds > 0:
            ids = _refine_block(
                model,
                velocity,
                ids,
                confidence,
                steps=refine_steps,
                temperature=temperature,
                threshold=confidence_threshold,
                rounds=refine_rounds,
            )
        generated.extend(ids)

        if use_cache and cache is not None:
            # Append the finalized block's clean K/V so the next block can reuse them.
            new_rows: KVCache = []
            model(
                model.embed_tokens(torch.tensor([ids], dtype=torch.long, device=device)),
                torch.zeros(1, n_new, device=device),
                attn_mask=make_block_mask(len(prefix), n_new, device),
                cache=cache,
                cache_out=new_rows,
                seq_start=len(prefix),
            )
            cache = [
                (torch.cat([k_old, k_new], dim=2), torch.cat([v_old, v_new], dim=2))
                for (k_old, v_old), (k_new, v_new) in zip(cache, new_rows, strict=True)
            ]

    model.train()
    return generated
