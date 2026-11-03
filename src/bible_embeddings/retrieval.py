"""BM25 plus optional embeddings from the SAME jointly trained evidence model."""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi

from .corpus import Passage, corpus_sha256, overlap

_WORDS = re.compile(r"[a-z0-9']+", re.I)


def tokens(text: str) -> list[str]:
    return _WORDS.findall(text.lower())


def normalize(vectors: np.ndarray) -> np.ndarray:
    return vectors / np.maximum(np.linalg.norm(vectors, axis=-1, keepdims=True), 1e-12)


class Retriever:
    def __init__(self, passages: list[Passage], encoder=None,
                 cache: str | Path | None = None, corpus_path: str | Path | None = None):
        if not passages:
            raise ValueError("Corpus is empty")
        self.passages = passages
        self.bm25 = BM25Okapi([tokens(p.text) for p in passages])
        self.encoder = encoder
        self.embeddings = None
        if encoder is not None:
            if cache and not corpus_path:
                raise ValueError("A corpus path is required to safely cache embeddings")
            cached = Path(cache) if cache else None
            corpus_hash = corpus_sha256(corpus_path) if corpus_path else ""
            widths = tuple(sorted({p.end - p.start + 1 for p in passages}))
            fingerprint = encoder.fingerprint
            if cached and cached.exists():
                with np.load(cached, allow_pickle=False) as data:
                    if (str(data["model"]) != fingerprint or
                            str(data["corpus"]) != corpus_hash or
                            tuple(data["widths"].tolist()) != widths):
                        raise ValueError("Stale embedding cache; rebuild using a new --cache path")
                    self.embeddings = data["embeddings"].astype(np.float32)
                if len(self.embeddings) != len(passages):
                    raise ValueError("Embedding cache has incorrect passage count")
            else:
                # The model judges and embeds with the exact same shared weights.
                # Encode in chunks to avoid allocating a 116k-input token batch.
                arrays = []
                for start in range(0, len(passages), 512):
                    arrays.append(np.asarray(
                        encoder.embed([p.text for p in passages[start:start + 512]]),
                        dtype=np.float32,
                    ))
                self.embeddings = normalize(np.concatenate(arrays))
                if cached:
                    cached.parent.mkdir(parents=True, exist_ok=True)
                    np.savez_compressed(cached, embeddings=self.embeddings,
                                        model=fingerprint, corpus=corpus_hash, widths=widths)

    def search(self, claim: str, limit: int = 40, pool: int = 300,
               remove_overlap: bool = True) -> list[tuple[Passage, float]]:
        if not claim.strip():
            raise ValueError("A claim is required")
        if limit < 1 or pool < 1:
            raise ValueError("limit and pool must be positive")

        lexical = self.bm25.get_scores(tokens(claim))
        lexical_order = (np.argsort(-lexical, kind="stable")[:pool]
                         if np.max(lexical) > 0 else [])
        orders = [lexical_order]
        if self.embeddings is not None:
            query = normalize(np.asarray(self.encoder.embed([claim]), dtype=np.float32))[0]
            sims = self.embeddings @ query
            orders.append(np.argsort(-sims, kind="stable")[:pool])

        scores: dict[int, float] = {}
        for order in orders:
            for rank, index in enumerate(order):
                i = int(index)
                if order is lexical_order and lexical[i] <= 0:
                    continue
                scores[i] = scores.get(i, 0.0) + 1.0 / (61 + rank)
        ranking = sorted(scores, key=lambda i: (-scores[i], i))
        selected: list[tuple[Passage, float]] = []
        for i in ranking:
            passage = self.passages[i]
            if remove_overlap and any(overlap(passage, previous) for previous, _ in selected):
                continue
            selected.append((passage, scores[i]))
            if len(selected) == limit:
                break
        return selected
