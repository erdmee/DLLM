"""Fast CPU smoke tests: config merging, model shapes, decoding paths, train -> sample."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from omegaconf.errors import ConfigKeyError

from diffusion_llm.config import load_config
from diffusion_llm.generation import generate_blocks
from diffusion_llm.model import DiffusionLLM, ModelConfig
from diffusion_llm.sample import sample_checkpoint
from diffusion_llm.train import train

DEBUG_YAML = str(Path(__file__).parents[1] / "configs" / "debug.yaml")
TINY = ModelConfig(vocab_size=64, d_model=32, n_layers=2, n_heads=4, max_seq_len=64)


def test_config_precedence() -> None:
    cfg = load_config([DEBUG_YAML], ["optim.lr=0.01", "model.d_model=48"])
    assert cfg.optim.lr == 0.01  # CLI override wins
    assert cfg.model.d_model == 48
    assert cfg.data.dataset == "builtin"  # YAML beats schema default
    with pytest.raises(ConfigKeyError):
        load_config(overrides=["optim.not_a_key=1"])


def test_forward_shape() -> None:
    model = DiffusionLLM(TINY)
    x = torch.randn(2, 10, TINY.d_model)
    assert model(x, torch.rand(2, 10)).shape == x.shape
    assert model.decode_logits(x).shape == (2, 10, TINY.vocab_size)


@pytest.mark.parametrize("max_seq_len", [64, 24])  # 24 forces the context window to slide
def test_cache_matches_recompute(max_seq_len: int) -> None:
    torch.manual_seed(0)
    model = DiffusionLLM(ModelConfig(**{**TINY.__dict__, "max_seq_len": max_seq_len}))
    outputs = []
    for use_cache in (True, False):
        torch.manual_seed(1)
        outputs.append(
            generate_blocks(
                model,
                block_len=8,
                length=40,
                steps=3,
                temperature=0.0,
                guidance_scale=0.5,
                device=torch.device("cpu"),
                use_cache=use_cache,
                refine_rounds=1,
            )
        )
    assert outputs[0] == outputs[1]


def test_train_then_sample(tmp_path: Path) -> None:
    ckpt = train(load_config([DEBUG_YAML], [f"run.out_dir={tmp_path}"]))
    assert (tmp_path / "config.yaml").exists()
    texts = sample_checkpoint(ckpt, ["sample.num_samples=2", "run.device=cpu"])
    assert len(texts) == 2 and all(isinstance(t, str) for t in texts)
