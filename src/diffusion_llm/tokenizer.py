"""Byte-Pair-Encoding (BPE) tokenizer built on the HuggingFace `tokenizers` backend.

A fresh BPE vocabulary is trained on the corpus, with a byte-level fallback so
no token is ever out of vocabulary. Encoding/decoding round-trip exactly.
"""

from __future__ import annotations

from collections.abc import Iterable

from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

_PAD = "<|pad|>"
_UNK = "<|unk|>"


class BPETokenizer:
    def __init__(self, vocab_size: int = 8192) -> None:
        self.vocab_size = vocab_size
        self.backend = Tokenizer(models.BPE(unk_token=_UNK, byte_fallback=True))
        self.backend.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        self.backend.decoder = decoders.ByteLevel()

    def train(self, lines: Iterable[str]) -> None:
        """Train the BPE vocabulary on an iterable of text lines."""
        trainer = trainers.BpeTrainer(
            vocab_size=self.vocab_size,
            special_tokens=[_PAD, _UNK],
            min_frequency=1,
        )
        self.backend.train_from_iterator(lines, trainer=trainer)

    def encode(self, text: str) -> list[int]:
        return self.backend.encode(text).ids

    def decode(self, ids: list[int]) -> str:
        return self.backend.decode(ids)

    def to_json(self) -> str:
        return self.backend.to_str()

    @classmethod
    def from_json(cls, payload: str) -> BPETokenizer:
        tokenizer = cls.__new__(cls)
        tokenizer.backend = Tokenizer.from_str(payload)
        tokenizer.vocab_size = tokenizer.backend.get_vocab_size()
        return tokenizer
