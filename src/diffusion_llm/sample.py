"""Sample text from a trained checkpoint via block-wise decoding.

    dllm-sample runs/tinystories/final.pt [steps=32 guidance_scale=1.0 ...]

Sampling settings start from the config saved in the checkpoint; bare keys
(`steps=32`) are shorthand for `sample.steps=32`.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

from diffusion_llm.config import Config, load_config, split_cli
from diffusion_llm.model import DiffusionLLM, ModelConfig
from diffusion_llm.tokenizer import BPETokenizer
from diffusion_llm.train import resolve_device, sample_text


def load_checkpoint(
    path: str | Path, device: torch.device
) -> tuple[DiffusionLLM, BPETokenizer, dict | None]:
    """Return the EMA model, its tokenizer and the saved experiment config (if any)."""
    payload = torch.load(path, map_location=device)
    model = DiffusionLLM(ModelConfig(**payload["cfg"])).to(device)
    model.load_state_dict(payload["model_state"])
    model.eval()
    tokenizer = BPETokenizer.from_json(payload["tokenizer_json"])
    return model, tokenizer, payload.get("config")


def sample_checkpoint(ckpt: str | Path, overrides: list[str] | None = None) -> list[str]:
    device = resolve_device(load_config(overrides=overrides).run.device)
    model, tokenizer, saved = load_checkpoint(ckpt, device)
    cfg: Config = load_config(base=saved, overrides=overrides)
    torch.manual_seed(cfg.run.seed)
    return [sample_text(model, tokenizer, cfg, device) for _ in range(cfg.sample.num_samples)]


def main() -> None:
    argv = sys.argv[1:]
    paths, overrides, rest = split_cli(argv)
    if {"-h", "--help"} & set(argv) or len(rest) != 1 or paths:
        print(__doc__.strip())
        return
    overrides = [o if "." in o.split("=", 1)[0] else f"sample.{o}" for o in overrides]
    for i, text in enumerate(sample_checkpoint(rest[0], overrides), start=1):
        print(f"\n--- sample {i} ---\n{text}")


if __name__ == "__main__":
    main()
