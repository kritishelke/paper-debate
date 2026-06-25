from __future__ import annotations

import hashlib
import math
import re
from collections import Counter
from typing import Iterable

import numpy as np


def normalize_answer(answer: str) -> str:
    text = re.sub(r"\s+", " ", answer.strip().lower())
    return re.sub(r"[^a-z0-9 .:_-]", "", text)


def stable_embedding(text: str, dim: int = 384) -> np.ndarray:
    """Small deterministic fallback embedding for offline tests and no-key demos."""
    tokens = re.findall(r"[a-zA-Z0-9_]+", text.lower())
    if not tokens:
        return np.zeros(dim, dtype=np.float32)
    vec = np.zeros(dim, dtype=np.float32)
    counts = Counter(tokens)
    for token, count in counts.items():
        digest = hashlib.sha256(token.encode("utf-8")).digest()
        idx = int.from_bytes(digest[:4], "big") % dim
        sign = 1.0 if digest[4] % 2 == 0 else -1.0
        vec[idx] += sign * (1.0 + math.log(count))
    norm = np.linalg.norm(vec)
    if norm:
        vec /= norm
    return vec


def cosine_similarity(a: str | np.ndarray, b: str | np.ndarray) -> float:
    av = stable_embedding(a) if isinstance(a, str) else a
    bv = stable_embedding(b) if isinstance(b, str) else b
    denom = float(np.linalg.norm(av) * np.linalg.norm(bv))
    if denom == 0:
        return 0.0
    return float(np.dot(av, bv) / denom)


def mean_pairwise_diversity(texts: Iterable[str]) -> float:
    items = list(texts)
    if len(items) < 2:
        return 1.0
    sims: list[float] = []
    for i, left in enumerate(items):
        for right in items[i + 1 :]:
            sims.append(cosine_similarity(left, right))
    return max(0.0, min(1.0, 1.0 - (sum(sims) / len(sims))))


class Embedder:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        self.model_name = model_name
        self._model = None
        try:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(model_name)
        except Exception:
            self._model = None

    def encode(self, texts: list[str]) -> np.ndarray:
        if self._model is not None:
            return np.asarray(self._model.encode(texts, normalize_embeddings=True), dtype=np.float32)
        return np.vstack([stable_embedding(text) for text in texts])

