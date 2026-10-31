"""Sample text from a trained diffusion LLM checkpoint via block-wise decoding."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from diffusion_llm.generation import generate_blocks
from diffusion_llm.model import DiffusionLLM, ModelConfig
from diffusion_llm.tokenizer import BPETokenizer
from diffusion_llm.train import resolve_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate text with a diffusion LLM checkpoint.")
    parser.add_argument("--ckpt", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=32, help="Diffusion steps per block.")
    parser.add_argument("--block-len", type=int, default=64, help="Tokens generated per block.")
    parser.add_argument("--length", type=int, default=256, help="Total tokens to generate.")
    parser.add_argument("--num-samples", type=int, default=4)
    parser.add_argument(
        "--temperature", type=float, default=0.9, help="<=0 means greedy argmax decoding."
    )
    parser.add_argument(
        "--guidance-scale",
        type=float,
        default=0.0,
        help="Classifier-free guidance strength (>0 only if trained with prefix dropout).",
    )
    parser.add_argument(
        "--no-cache",
        action="store_false",
        dest="use_cache",
        help="Disable KV caching across blocks (recompute prefix every step).",
    )
    parser.add_argument(
        "--refine-rounds", type=int, default=2, help="Adaptive refinement rounds per block."
    )
    parser.add_argument(
        "--refine-steps", type=int, default=4, help="Diffusion steps per refinement round."
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.7,
        help="Keep tokens with confidence >= threshold during refine.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="auto")
    return parser.parse_args()


def load_checkpoint(path: Path, device: torch.device) -> tuple[DiffusionLLM, BPETokenizer]:
    payload = torch.load(path, map_location=device)
    cfg = ModelConfig(**payload["cfg"])
    tokenizer = BPETokenizer.from_json(payload["tokenizer_json"])
    model = DiffusionLLM(cfg).to(device)
    model.load_state_dict(payload["model_state"])
    model.eval()
    return model, tokenizer


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    device = resolve_device(args.device)

    model, tokenizer = load_checkpoint(args.ckpt, device)

    print(
        f"[loaded] {args.ckpt} (d_model={model.cfg.d_model}, layers={model.cfg.n_layers}, "
        f"vocab={model.cfg.vocab_size})"
    )
    for i in range(args.num_samples):
        generated = generate_blocks(
            model,
            block_len=args.block_len,
            length=args.length,
            steps=args.steps,
            temperature=args.temperature,
            guidance_scale=args.guidance_scale,
            device=device,
            use_cache=args.use_cache,
            refine_rounds=args.refine_rounds,
            refine_steps=args.refine_steps,
            confidence_threshold=args.confidence_threshold,
        )
        print(f"\n--- sample {i + 1} ({args.length} tokens, {args.steps} steps/block) ---")
        print(tokenizer.decode(generated))


if __name__ == "__main__":
    main()
