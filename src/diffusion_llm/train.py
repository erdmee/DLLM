"""Train a block-diffusion LLM (continuous flow matching) on text.

dllm-train configs/tinystories.yaml [key=value ...]
"""

from __future__ import annotations

import math
import random
import sys
import time
from dataclasses import asdict, replace
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from diffusion_llm.config import Config, load_config, split_cli, to_dict, to_yaml
from diffusion_llm.dataset import TextDataset, line_iterator, load_corpus
from diffusion_llm.diffusion import FlowMatching
from diffusion_llm.generation import generate_blocks, make_causal_mask
from diffusion_llm.model import DiffusionLLM
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


def resolve_device(preference: str) -> torch.device:
    if preference != "auto":
        return torch.device(preference)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def learning_rate_at_step(step: int, total: int, warmup: int, base_lr: float) -> float:
    if warmup > 0 and step < warmup:
        return base_lr * (step + 1) / warmup
    progress = (step - warmup) / max(1, total - warmup)
    return base_lr * 0.5 * (1.0 + math.cos(math.pi * progress))


def save_checkpoint(
    path: Path,
    cfg: Config,
    tokenizer: BPETokenizer,
    step: int,
    ema: EMA,
    optimizer: torch.optim.Optimizer,
) -> None:
    payload: dict[str, object] = {
        "step": step,
        "cfg": asdict(cfg.model),
        "config": to_dict(cfg),
        "tokenizer_json": tokenizer.to_json(),
        "model_state": ema.state_dict(),
        "optimizer_state": optimizer.state_dict(),
    }
    torch.save(payload, path)


def sample_text(
    model: DiffusionLLM, tokenizer: BPETokenizer, cfg: Config, device: torch.device
) -> str:
    s = cfg.sample
    ids = generate_blocks(
        model,
        block_len=s.block_len,
        length=s.length,
        steps=s.steps,
        temperature=s.temperature,
        guidance_scale=s.guidance_scale,
        device=device,
        use_cache=s.use_cache,
        refine_rounds=s.refine_rounds,
        refine_steps=s.refine_steps,
        confidence_threshold=s.confidence_threshold,
    )
    return tokenizer.decode(ids)


def train(cfg: Config) -> Path:
    """Run training and return the final checkpoint path."""
    random.seed(cfg.run.seed)
    torch.manual_seed(cfg.run.seed)
    device = resolve_device(cfg.run.device)
    out_dir = Path(cfg.run.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    text = load_corpus(cfg.data.data_path, cfg.data.dataset, cfg.data.max_docs)
    tokenizer = BPETokenizer(vocab_size=cfg.model.vocab_size)
    tokenizer.train(line_iterator(text))
    tokens = tokenizer.encode(text)
    # The tokenizer can come out smaller than requested on small corpora.
    cfg = replace(cfg, model=replace(cfg.model, vocab_size=tokenizer.backend.get_vocab_size()))
    if cfg.data.window > cfg.model.max_seq_len:
        raise ValueError("data.window must be <= model.max_seq_len")
    (out_dir / "config.yaml").write_text(to_yaml(cfg))

    dataset = TextDataset(tokens, cfg.data.window)
    batch_size = min(cfg.optim.batch_size, len(dataset))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    model = DiffusionLLM(cfg.model).to(device)
    flow = FlowMatching()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.optim.lr,
        betas=(0.9, 0.95),
        weight_decay=cfg.optim.weight_decay,
    )
    ema = EMA(model, decay=cfg.optim.ema_decay)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"[config] device={device} out={out_dir} params={n_params:,}", flush=True)
    print(
        f"[data] {cfg.data.dataset} tokens={len(tokens)} windows={len(dataset)} "
        f"vocab={cfg.model.vocab_size} batch={batch_size} block={cfg.data.block_len}",
        flush=True,
    )

    total = cfg.optim.steps
    window = cfg.data.window
    block_len = min(cfg.data.block_len, window)
    causal_mask = make_causal_mask(window, device)
    start_time = time.time()
    step = 0
    model.train()
    while step < total:
        for batch_ids in loader:
            step += 1
            ids = batch_ids.to(device)
            batch = ids.size(0)
            block_start = random.randint(0, window - block_len)
            block_end = block_start + block_len

            t_full = torch.zeros(batch, window, device=device)
            t_full[:, block_start:block_end] = flow.sample_timesteps(batch, device).unsqueeze(1)
            block_mask = torch.zeros(batch, window, device=device)
            block_mask[:, block_start:block_end] = 1.0

            x0 = model.embed_tokens(ids)
            if block_start > 0 and random.random() < cfg.optim.prefix_dropout:
                x0[:, :block_start] = 0.0

            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=cfg.optim.amp):
                loss = flow.compute_loss(model, x0, t_full, block_mask, causal_mask)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.optim.grad_clip)

            lr = learning_rate_at_step(step, total, cfg.optim.warmup_steps, cfg.optim.lr)
            for group in optimizer.param_groups:
                group["lr"] = lr
            optimizer.step()
            ema.update(model)

            if step % cfg.run.log_every == 0:
                toks_per_s = batch * window * step / max(time.time() - start_time, 1e-6)
                print(
                    f"step {step}/{total} loss={loss.item():.4f} lr={lr:.2e} "
                    f"tok/s={toks_per_s:.0f}",
                    flush=True,
                )
            if cfg.run.sample_every and step % cfg.run.sample_every == 0:
                print(f"[sample {step}] {sample_text(model, tokenizer, cfg, device)!r}", flush=True)
            if cfg.run.save_every and step % cfg.run.save_every == 0:
                ckpt = out_dir / f"step-{step}.pt"
                save_checkpoint(ckpt, cfg, tokenizer, step, ema, optimizer)
                print(f"[saved] {ckpt}", flush=True)
            if step >= total:
                break

    final = out_dir / "final.pt"
    save_checkpoint(final, cfg, tokenizer, step, ema, optimizer)
    print(f"[done] {final} in {time.time() - start_time:.1f}s", flush=True)
    return final


def main() -> None:
    argv = sys.argv[1:]
    if {"-h", "--help"} & set(argv):
        print(__doc__.strip(), "\n\n# defaults\n" + to_yaml(load_config()), sep="")
        return
    paths, overrides, rest = split_cli(argv)
    if rest:
        raise SystemExit(f"unexpected arguments: {rest} (use key=value)")
    train(load_config(paths, overrides))


if __name__ == "__main__":
    main()
