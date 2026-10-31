"""Data loading: famous corpora (WikiText-2 / Tiny Shakespeare) + block-window datasets."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import torch
from torch.utils.data import Dataset

# Tiny fallback corpus (used only if no --data-path and dataset download fails).
_FALLBACK = """
Diffusion language models denoise continuous token embeddings instead of next-token prediction.
This allows many tokens to be sampled in parallel, making decoding much faster.
Block diffusion interpolates between autoregressive and diffusion language models.
"""


def line_iterator(text: str) -> Iterator[str]:
    yield from (line for line in text.splitlines() if line.strip())


def load_famous_corpus(name: str, max_docs: int) -> str:
    """Load a well-known text corpus.

    - tinystories: stream from roneneldan/TinyStories (capped at max_docs docs).
    - tiny-shakespeare: single raw text file from the canonical Karpathy repo.
    - wikitext-2: attempt namespaced HuggingFace variants.
    """
    if name == "tinystories":
        from datasets import load_dataset  # lazy import

        ds = load_dataset("roneneldan/TinyStories", split="train", streaming=True)
        docs: list[str] = []
        for index, row in enumerate(ds):
            if index >= max_docs:
                break
            text = str(row["text"]).strip()
            if text:
                docs.append(text)
        return "\n".join(docs)

    if name == "tiny-shakespeare":
        import urllib.request

        url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
        with urllib.request.urlopen(url, timeout=60) as response:
            return response.read().decode("utf-8")

    if name == "wikitext-2":
        from datasets import load_dataset  # lazy import

        attempts: list[tuple[str, str | None]] = [
            ("Salesforce/wikitext", "wikitext-2-raw-v1"),
            ("wikitext/wikitext-2-raw-v1", None),
            ("wikitext", "wikitext-2-raw-v1"),
        ]
        last_error: Exception | None = None
        for dataset_id, config in attempts:
            try:
                if config:
                    ds = load_dataset(dataset_id, config, split="train")
                else:
                    ds = load_dataset(dataset_id, split="train")
                return "\n".join(str(x) for x in ds["text"] if str(x).strip())
            except Exception as exc:  # noqa: BLE001 - try next candidate id
                last_error = exc
        assert last_error is not None
        raise last_error

    raise ValueError(f"unknown dataset name: {name}")


def load_corpus(data_path: Path | None, dataset: str, max_docs: int) -> str:
    """Return corpus text from a file, a named dataset, or the fallback."""
    if data_path is not None:
        file = Path(data_path)
        if not file.exists():
            raise FileNotFoundError(f"corpus file not found: {file}")
        return file.read_text(encoding="utf-8")
    try:
        return load_famous_corpus(dataset, max_docs)
    except Exception as exc:  # noqa: BLE001 - network/cache issues should not crash
        print(f"[warn] failed to load dataset '{dataset}' ({exc}); using tiny fallback corpus")
        return _FALLBACK


class TextDataset(Dataset[torch.Tensor]):
    """Fixed-length windows of token ids for block-diffusion training."""

    def __init__(self, tokens: list[int], window_size: int) -> None:
        if not tokens:
            raise ValueError("token list is empty")
        if window_size <= 0:
            raise ValueError("window_size must be positive")
        self.window_size = window_size
        if len(tokens) < window_size:
            reps = (window_size + len(tokens) - 1) // len(tokens)
            tokens = tokens * reps
        self.tokens = tokens
        self.length = len(tokens) - window_size + 1

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int) -> torch.Tensor:
        window = self.tokens[index : index + self.window_size]
        return torch.tensor(window, dtype=torch.long)
