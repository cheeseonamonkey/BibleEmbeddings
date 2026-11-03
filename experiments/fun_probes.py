"""Playful adversarial probes for the actual trained shared ModernBERT checkpoint.

Only a smoke-test report. Labels and model scores are not theological judgments.
Run: PYTHONPATH=src .venv-eval/bin/python experiments/fun_probes.py
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import torch

from bible_embeddings.corpus import load_passages
from bible_embeddings.judge import NLIJudge, top_evidence
from bible_embeddings.retrieval import Retriever

torch.set_num_threads(4)

ROOT = Path(__file__).resolve().parents[1]
CORPUS = ROOT / "data/en_kjv.json"
CHECKPOINT = ROOT / "out/domain-v1/verse-domain-v1/model"
RESULTS = ROOT / "out/fun_probes"
RESULTS.mkdir(parents=True, exist_ok=True)

# Includes literal opposites, an apparent same-chapter contradiction,
# absurd modern claims and conspicuously misleading prooftexts.
DIRECT = [
    ("The moon is made of cheese.", "Ecclesiastes 10:19", "neutral"),
    ("Bush did 9/11.", "Proverbs 18:13", "neutral"),
    ("God prefers cats over dogs.", "Genesis 1:25", "neutral"),
    ("The Earth is flat.", "Isaiah 40:22", "neutral"),
    ("The Bible predicted TikTok.", "Ecclesiastes 1:9", "neutral"),
    ("God hates Mondays.", "Genesis 1:5", "neutral"),
    ("An ass spoke to Balaam.", "Numbers 22:28", "entailment"),
    ("All donkeys speak English.", "Numbers 22:28", "neutral"),
    ("It is sinful to be bald.", "2 Kings 2:23", "neutral"),
    ("Feed a hungry enemy.", "Proverbs 25:21", "entailment"),
    ("Never give food to your enemies.", "Proverbs 25:21", "contradiction"),
    ("Answer a fool according to his folly.", "Proverbs 26:4", "contradiction"),
    ("Answer a fool according to his folly.", "Proverbs 26:5", "entailment"),
    ("Answer a fool according to his folly.", "Proverbs 26:4–5", "neutral"),
    ("Both answering and not answering a fool are instructed.", "Proverbs 26:4–5", "entailment"),
    ("Christians shared their possessions.", "Acts 2:44–45", "entailment"),
    ("Modern governments must abolish private property.", "Acts 2:44–45", "neutral"),
    ("There is a time to dance.", "Ecclesiastes 3:4", "entailment"),
    ("All dancing is prohibited.", "Ecclesiastes 3:4", "contradiction"),
    ("Love your enemies.", "Matthew 5:44", "entailment"),
    ("Hate your enemies.", "Matthew 5:44", "contradiction"),
    ("Money answers all things.", "Ecclesiastes 10:19", "entailment"),
    ("Tax rates must be exactly 17 percent.", "Mark 12:17", "neutral"),
    ("God commands the use of smartphones.", "John 1:1", "neutral"),
]

SEARCH = [
    "Answer a fool according to his folly.",
    "Never feed an enemy who is hungry.",
    "You should always love your enemies.",
    "A wealthy man easily enters the kingdom of God.",
    "Money answers all things.",
    "There is a time to dance.",
    "Long hair on a man is a shame.",
    "The Sabbath was made for man.",
    "It is better to give than receive.",
    "Every donkey can talk.",
    "God prefers cats over dogs.",
    "The Bible predicted TikTok.",
    "Bush did 9/11.",
]


def main():
    begun = time.monotonic()
    print("Loading corpus...", flush=True)
    passages = load_passages(CORPUS)
    references = {p.reference: p for p in passages}
    print("Passages:", len(passages), flush=True)
    print("Loading one checkpoint...", flush=True)
    judge = NLIJudge(str(CHECKPOINT), batch_size=12, device="cpu")
    print("Checkpoint loaded", round(time.monotonic() - begun, 1), "seconds", flush=True)
    missing = [ref for _, ref, _ in DIRECT if ref not in references]
    assert not missing, missing

    print("Direct pair tests...", flush=True)
    predictions = judge.predict_pairs(
        [references[reference].text for _, reference, _ in DIRECT],
        [claim for claim, _, _ in DIRECT],
    )
    direct = []
    for (claim, reference, expected), scores in zip(DIRECT, predictions):
        actual = max(scores, key=scores.get)
        item = {"claim": claim, "reference": reference,
                "expected": expected, "predicted": actual, "scores": scores}
        direct.append(item)
        print(("OK " if actual == expected else "BAD"),
              reference, expected, "->", actual,
              f"{scores[actual]:.3f}", claim, flush=True)

    (RESULTS / "direct.json").write_text(json.dumps(direct, indent=2) + "\n")
    print("Direct accuracy", sum(x["predicted"] == x["expected"] for x in direct),
          "/", len(direct), flush=True)

    print("Preparing BM25 retrieval...", flush=True)
    retriever = Retriever(passages)
    print("Testing full-corpus retrieval + contextual NLI...", flush=True)
    search = []
    for claim in SEARCH:
        t = time.monotonic()
        hits = retriever.search(claim, limit=14, remove_overlap=False)
        evaluated = judge.evaluate(hits, claim)
        supports = top_evidence(evaluated, "entailment", n=3, threshold=.4)
        contradicts = top_evidence(evaluated, "contradiction", n=3, threshold=.4)
        item = {
            "claim": claim, "retrieval_hits": len(hits),
            "candidate_references": [p.reference for p, _ in hits],
            "for": [{"reference": e.passage.reference,
                     "score": e.contextual["entailment"],
                     "passage": e.passage.text[:240]} for e in supports],
            "against": [{"reference": e.passage.reference,
                         "score": e.contextual["contradiction"],
                         "passage": e.passage.text[:240]} for e in contradicts],
            "runtime_seconds": round(time.monotonic() - t, 2),
        }
        search.append(item)
        print("CLAIM", json.dumps(item, ensure_ascii=False), flush=True)
        (RESULTS / "search.json").write_text(json.dumps(search, indent=2) + "\n")

    print("DONE", round(time.monotonic()-begun, 1), "seconds", flush=True)


if __name__ == "__main__":
    main()
