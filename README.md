# DLLM

Minimal **block-diffusion LLM** in ~1k lines of PyTorch. It runs flow matching over continuous token embeddings, generating autoregressively across blocks and in parallel within each block.

| | |
| --- | --- |
| **Diffusion** | Rectified flow `x_t = (1-σ)(1-t)·x₀ + t·ε`, velocity MSE on the noisy block only (BD3-LM style) |
| **Backbone** | RMSNorm · RoPE · QK-norm · SwiGLU · block-causal SDPA · per-position timestep embedding |
| **Sampling** | Cosine logits vs tied embeddings · KV cache · CFG (prefix dropout) · Mercury-style adaptive refine |
| **Training** | AdamW · warmup + cosine LR · EMA 0.999 · bf16 · ByteLevel BPE 8k |

## Usage

```bash
uv sync && uv run pytest                      # offline smoke tests (~1 s)
uv run dllm-train configs/tinystories.yaml    # or configs/debug.yaml for a CPU sanity run
uv run dllm-sample runs/tinystories/final.pt steps=32 guidance_scale=1.0
```

To override any setting, append `section.key=value` (e.g. `optim.lr=1e-3 run.out_dir=runs/exp1`). Each run saves its resolved `config.yaml`, and `dllm-train --help` lists every key.

## Sample

7.9M params, 6k steps on TinyStories (15k docs), 64 steps/block:

> okay knewMmm gone take soap Tim swings Her swings wind cry safari memb owl harm goOkay swing King makes Mia even birdhouse …

The model has learned the vocabulary but not yet the grammar.

## Layout

```
├── configs/             # YAML presets: debug, tinystories
├── tests/               # smoke tests
└── src/diffusion_llm/
    ├── config.py        # dataclass schema + YAML / CLI merging
    ├── model.py         # Transformer backbone, cosine decode
    ├── diffusion.py     # flow-matching path, loss
    ├── generation.py    # block decoding, KV cache, CFG, refine
    ├── train.py         # training loop, EMA, checkpoints
    ├── sample.py        # sampling CLI
    ├── tokenizer.py     # BPE tokenizer
    └── dataset.py       # corpora, window dataset
```
