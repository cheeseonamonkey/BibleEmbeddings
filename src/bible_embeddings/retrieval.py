"""Hybrid BM25/dense retrieval with reciprocal rank fusion and overlap control."""
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
    def __init__(self, passages: list[Passage], model_name: str | None = None,
                 cache: str | Path | None = None, corpus_path: str | Path | None = None):
        if not passages:
            raise ValueError("Corpus is empty")
        self.passages = passages
        self.bm25 = BM25Okapi([tokens(p.text) for p in passages])
        self.encoder = None
        self.embeddings = None
        if model_name:
            from sentence_transformers import SentenceTransformer
            self.encoder = SentenceTransformer(model_name)
            cached = Path(cache) if cache else None
            fingerprint = corpus_sha256(corpus_path) if corpus_path else ""
            widths = tuple(sorted({p.end-p.start+1 for p in passages}))
            if cached and cached.exists():
                with np.load(cached, allow_pickle=False) as data:
                    if (str(data["model"]) != model_name or
                            str(data["corpus"]) != fingerprint or
                            tuple(data["widths"].tolist()) != widths):
                        raise ValueError("Stale embedding cache")
                    self.embeddings = data["embeddings"].astype(np.float32)
                if len(self.embeddings) != len(passages):
                    raise ValueError("Embedding cache has incorrect passage count")
            else:
                embeddings = self.encoder.encode(
                    [p.text for p in passages], batch_size=64,
                    show_progress_bar=True, convert_to_numpy=True,
                )
                self.embeddings = normalize(np.asarray(embeddings, dtype=np.float32))
                if cached:
                    cached.parent.mkdir(parents=True, exist_ok=True)
                    np.savez_compressed(cached, embeddings=self.embeddings, model=model_name,
                                        corpus=fingerprint, widths=widths)

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
        if self.encoder is not None:
            q = np.asarray(self.encoder.encode([claim], convert_to_numpy=True), np.float32)
            sims = self.embeddings @ normalize(q)[0]
            orders.append(np.argsort(-sims, kind="stable")[:pool])

        scores: dict[int, float] = {}
        for order in orders:
            for rank, index in enumerate(order):
                i = int(index)
                if order is lexical_order and lexical[i] <= 0:
                    continue
                scores[i] = scores.get(i, 0) + 1 / (61 + rank)
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
