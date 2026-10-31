"""Experiment configuration: typed dataclass schema + YAML presets + dot-key overrides.

    dllm-train configs/tinystories.yaml optim.lr=1e-3 run.out_dir=runs/lr1e-3

Later sources win: schema defaults < YAML files (in order) < `key=value` overrides.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import cast

from omegaconf import OmegaConf

from diffusion_llm.model import ModelConfig


@dataclass
class DataConfig:
    dataset: str = "tinystories"  # tinystories | wikitext-2 | tiny-shakespeare | builtin
    data_path: str | None = None  # plain-text file; overrides `dataset`
    max_docs: int = 15_000
    window: int = 320  # prefix + block, in tokens
    block_len: int = 64  # set == window for pure diffusion


@dataclass
class OptimConfig:
    steps: int = 6000
    batch_size: int = 8
    lr: float = 3e-4
    warmup_steps: int = 100
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    ema_decay: float = 0.999
    prefix_dropout: float = 0.1  # trains the unconditional path for CFG
    amp: bool = False  # bf16 autocast


@dataclass
class SampleConfig:
    block_len: int = 64
    length: int = 128
    steps: int = 64  # diffusion steps per block
    temperature: float = 0.9  # <= 0 means greedy
    guidance_scale: float = 0.0
    use_cache: bool = True
    refine_rounds: int = 2
    refine_steps: int = 4
    confidence_threshold: float = 0.7
    num_samples: int = 3


@dataclass
class RunConfig:
    out_dir: str = "runs/default"
    seed: int = 0
    device: str = "auto"  # auto | cpu | cuda | mps
    log_every: int = 100
    sample_every: int = 1000  # 0 disables in-training samples
    save_every: int = 2000  # 0 saves only the final checkpoint


@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    optim: OptimConfig = field(default_factory=OptimConfig)
    sample: SampleConfig = field(default_factory=SampleConfig)
    run: RunConfig = field(default_factory=RunConfig)


def load_config(
    paths: list[str] | None = None,
    overrides: list[str] | None = None,
    base: dict | None = None,
) -> Config:
    """Merge defaults, an optional saved dict, YAML files and dot-list overrides."""
    layers = [OmegaConf.structured(Config)]
    if base is not None:
        layers.append(OmegaConf.create(base))
    layers += [OmegaConf.load(path) for path in paths or []]
    layers.append(OmegaConf.from_dotlist(overrides or []))
    return cast(Config, OmegaConf.to_object(OmegaConf.merge(*layers)))


def split_cli(argv: list[str]) -> tuple[list[str], list[str], list[str]]:
    """Split argv into (yaml paths, key=value overrides, other positionals)."""
    paths = [a for a in argv if a.endswith((".yaml", ".yml"))]
    overrides = [a for a in argv if "=" in a and a not in paths]
    rest = [a for a in argv if a not in paths and a not in overrides]
    return paths, overrides, rest


def to_yaml(cfg: Config) -> str:
    return OmegaConf.to_yaml(OmegaConf.structured(cfg))


def to_dict(cfg: Config) -> dict:
    return cast(dict, OmegaConf.to_container(OmegaConf.structured(cfg)))
