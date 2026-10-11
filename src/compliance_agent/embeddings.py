"""Embedders. The OpenAI one is used for real runs; the hashing one needs no network
and exists so the retriever can be tested and demoed offline."""

from __future__ import annotations

import hashlib
import re
from typing import Protocol

import numpy as np

_TOKEN = re.compile(r"[a-z0-9]+")


class Embedder(Protocol):
    name: str

    def embed(self, texts: list[str]) -> np.ndarray:
        """Return one L2-normalised row per text."""
        ...


def _normalise(m: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(m, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return m / norms


class HashingEmbedder:
    """Feature hashing over unigrams and bigrams. Lexical only, but deterministic."""

    def __init__(self, dim: int = 2048) -> None:
        self.dim = dim
        self.name = f"hashing-{dim}"

    def _features(self, text: str) -> list[str]:
        words = _TOKEN.findall(text.lower())
        return words + [f"{a}_{b}" for a, b in zip(words, words[1:], strict=False)]

    def embed(self, texts: list[str]) -> np.ndarray:
        m = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, text in enumerate(texts):
            for f in self._features(text):
                h = int.from_bytes(hashlib.blake2b(f.encode(), digest_size=8).digest(), "big")
                m[i, h % self.dim] += 1.0 if (h >> 63) & 1 else -1.0
        return _normalise(m)


class OpenAIEmbedder:
    def __init__(self, client, model: str) -> None:
        self.client = client
        self.model = model
        self.name = f"openai-{model}"

    def embed(self, texts: list[str]) -> np.ndarray:
        resp = self.client.embeddings.create(model=self.model, input=texts)
        rows = [d.embedding for d in sorted(resp.data, key=lambda d: d.index)]
        return _normalise(np.asarray(rows, dtype=np.float32))
