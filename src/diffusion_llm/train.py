"""Train a block-diffusion LLM (continuous flow matching) on text."""

from __future__ import annotations

import argparse
import math
import random
import time
from dataclasses import asdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from diffusion_llm.dataset import TextDataset, line_iterator, load_corpus
from diffusion_llm.diffusion import FlowMatching
from diffusion_llm.generation import generate_blocks, make_causal_mask
from diffusion_llm.model import DiffusionLLM, ModelConfig
from diffusion_llm.tokenizer import BPETokenizer


class EMA:
    """Exponential moving average of model weights (standard for diffusion models)."""

    def __init__(self, model: DiffusionLLM, decay: float = 0.999) -> None:
        self.decay = decay
        self.shadow: dict[str, torch.Tensor] = {
            key: value.detach().clone() for key, value in model.state_dict().items()
        }

    @torch.no_grad()
    def update(self, model: DiffusionLLM) -> None:
        for key, value in model.state_dict().items():
            self.shadow[key].mul_(self.decay).add_(value.detach(), alpha=1.0 - self.decay)

    def state_dict(self) -> dict[str, torch.Tensor]:
        return {key: value.clone() for key, value in self.shadow.items()}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a block-diffusion LLM.")
    parser.add_argument(
        "--dataset",
        type=str,
        default="tinystories",
        choices=["tinystories", "wikitext-2", "tiny-shakespeare"],
    )
    parser.add_argument(
        "--max-docs",
        type=int,
        default=15000,
        help="Cap on corpus documents (tinystories streaming).",
    )
    parser.add_argument(
        "--data-path", type=Path, default=None, help="Override dataset with a plain-text file."
    )
    parser.add_argument("--out-dir", type=Path, default=Path("checkpoints"))
    parser.add_argument(
        "--window", type=int, default=320, help="Context window length in tokens (prefix + block)."
    )
    parser.add_argument(
        "--block-len",
        type=int,
        default=64,
        help="Block size L' (set == window for pure diffusion).",
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--steps", type=int, default=1500)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--vocab-size", type=int, default=8192)
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--n-layers", type=int, default=6)
    parser.add_argument("--n-heads", type=int, default=8)
    parser.add_argument("--ffn-mult", type=int, default=3)
    parser.add_argument(
        "--prefix-dropout",
        type=float,
        default=0.1,
        help="Drop prefix (zero embeddings) with this probability for CFG.",
    )
    parser.add_argument("--ema-decay", type=float, default=0.999)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--sample-every", type=int, default=500)
    parser.add_argument("--save-every", type=int, default=500)
    parser.add_argument("--sample-block-len", type=int, default=64)
    parser.add_argument("--sample-length", type=int, default=256)
    parser.add_argument("--sample-steps", type=int, default=32)
    parser.add_argument("--sample-guidance", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="auto", help="auto | cpu | cuda | mps")
    parser.add_argument(
        "--amp",
        action="store_true",
        help="Use bfloat16 autocast for compute (1.5-2.5x faster on MPS/CUDA).",
    )
    return parser.parse_args()


def resolve_device(preference: str) -> torch.device:
    if preference != "auto":
        return torch.device(preference)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)


def learning_rate_at_step(step: int, total: int, warmup: int, base_lr: float) -> float:
    if warmup > 0 and step < warmup:
        return base_lr * (step + 1) / warmup
    progress = (step - warmup) / max(1, total - warmup)
    return base_lr * 0.5 * (1.0 + math.cos(math.pi * progress))


def save_checkpoint(
    path: Path,
    model: DiffusionLLM,
    tokenizer: BPETokenizer,
    cfg: ModelConfig,
    step: int,
    ema: EMA,
    optimizer: torch.optim.Optimizer | None = None,
) -> None:
    payload: dict[str, object] = {
        "step": step,
        "cfg": asdict(cfg),
        "tokenizer_json": tokenizer.to_json(),
        "model_state": ema.state_dict(),
        "optimizer_state": optimizer.state_dict() if optimizer is not None else None,
    }
    torch.save(payload, path)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = resolve_device(args.device)

    # Corpus + BPE tokenizer.
    text = load_corpus(args.data_path, args.dataset, args.max_docs)
    tokenizer = BPETokenizer(vocab_size=args.vocab_size)
    tokenizer.train(line_iterator(text))
    tokens = tokenizer.encode(text)
    vocab_size = tokenizer.backend.get_vocab_size()

    dataset = TextDataset(tokens, args.window)
    batch_size = min(args.batch_size, len(dataset))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    cfg = ModelConfig(
        vocab_size=vocab_size,
        d_model=args.d_model,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        ffn_mult=args.ffn_mult,
        max_seq_len=max(args.window * 2, 2048),
    )
    model = DiffusionLLM(cfg).to(device)
    flow = FlowMatching()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.01
    )
    ema = EMA(model, decay=args.ema_decay)

    n_params = sum(p.numel() for p in model.parameters())
    print(
        f"[config] device={device} dataset={args.dataset} vocab={vocab_size} params={n_params:,}",
        flush=True,
    )
    print(
        f"[data] chars={len(text)} tokens={len(tokens)} windows={len(dataset)} "
        f"batch={batch_size} block={args.block_len}",
        flush=True,
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    causal_mask = make_causal_mask(args.window, device)
    start_time = time.time()
    step = 0
    model.train()
    while step < args.steps:
        for batch_ids in loader:
            step += 1
            ids = batch_ids.to(device)
            batch, window = ids.shape
            block_len = min(args.block_len, window)
            block_start = random.randint(0, window - block_len)
            block_end = block_start + block_len

            t_full = torch.zeros(batch, window, device=device)
            t_block = flow.sample_timesteps(batch, device)
            t_full[:, block_start:block_end] = t_block.unsqueeze(1)
            block_mask = torch.zeros(batch, window, device=device)
            block_mask[:, block_start:block_end] = 1.0

            x0 = model.embed_tokens(ids)
            if (
                args.prefix_dropout > 0.0
                and block_start > 0
                and random.random() < args.prefix_dropout
            ):
                x0[:, :block_start] = 0.0

            if args.amp:
                with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=True):
                    loss = flow.compute_loss(model, x0, t_full, block_mask, causal_mask)
            else:
                loss = flow.compute_loss(model, x0, t_full, block_mask, causal_mask)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            lr = learning_rate_at_step(step, args.steps, args.warmup_steps, args.lr)
            for group in optimizer.param_groups:
                group["lr"] = lr
            optimizer.step()
            ema.update(model)

            if step % args.log_every == 0:
                elapsed = time.time() - start_time
                toks_per_s = batch * window * step / max(elapsed, 1e-6)
                print(
                    f"step {step}/{args.steps} loss={loss.item():.4f} lr={lr:.2e} "
                    f"tok/s={toks_per_s:.0f}",
                    flush=True,
                )

            if step % args.sample_every == 0:
                generated = generate_blocks(
                    model,
                    block_len=args.sample_block_len,
                    length=args.sample_length,
                    steps=args.sample_steps,
                    temperature=0.9,
                    guidance_scale=args.sample_guidance,
                    device=device,
                )
                print(f"[sample {step}] {tokenizer.decode(generated)!r}", flush=True)

            if step % args.save_every == 0:
                ckpt = args.out_dir / f"step-{step}.pt"
                save_checkpoint(ckpt, model, tokenizer, cfg, step, ema, optimizer)
                print(f"[saved] {ckpt}", flush=True)

            if step >= args.steps:
                break

    final = args.out_dir / "final.pt"
    save_checkpoint(final, model, tokenizer, cfg, step, ema, optimizer)
    print(
        f"[done] checkpoint saved to {final} in {time.time() - start_time:.1f}s",
        flush=True,
    )


if __name__ == "__main__":
    main()
