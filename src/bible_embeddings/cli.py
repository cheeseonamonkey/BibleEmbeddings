"""Search and score biblical passages with one optional fine-tuned neural model."""
from __future__ import annotations

import argparse
import json

from .corpus import load_passages
from .retrieval import Retriever


def main() -> None:
    parser = argparse.ArgumentParser(description="Find biblical passages for and against a claim")
    parser.add_argument("--data", required=True, help="KJV JSON corpus")
    parser.add_argument("--claim", help="Any proposition")
    parser.add_argument("--model", help="A fine-tuned shared ModernBERT checkpoint")
    parser.add_argument("--semantic", action="store_true",
                        help="Use the same model for embedding retrieval (slower first index)")
    parser.add_argument("--cache", help="Dense embeddings cache (.npz)")
    parser.add_argument("--limit", type=int, default=40)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    if args.semantic and not args.model:
        parser.error("--semantic requires --model")
    if args.cache and not args.semantic:
        parser.error("--cache requires --semantic")

    passages = load_passages(args.data)
    if not args.claim:
        print(f"Indexed {len(passages):,} single/multi-verse passages")
        return

    from .judge import NLIJudge, top_evidence
    judge = NLIJudge(args.model) if args.model else None
    retriever = Retriever(
        passages, encoder=judge if args.semantic else None,
        cache=args.cache, corpus_path=args.data,
    )
    hits = retriever.search(args.claim, limit=args.limit, remove_overlap=judge is None)
    if judge is None:
        result = {
            "status": "retrieval_only",
            "claim": args.claim,
            "candidates": [{
                "reference": p.reference, "passage": p.text,
                "retrieval_score": round(score, 5),
            } for p, score in hits],
        }
    else:
        scored = judge.evaluate(hits, args.claim)
        result = {
            "status": "model_evidence_signals_not_truth_probabilities",
            "claim": args.claim,
            "for": [e.as_dict() for e in top_evidence(scored, "entailment")],
            "against": [e.as_dict() for e in top_evidence(scored, "contradiction")],
            "inspected": len(scored),
        }

    if args.json:
        print(json.dumps(result, indent=2))
        return
    print(f"Claim: {args.claim} ({result['status']})")
    if "candidates" in result:
        for row in result["candidates"][:10]:
            print(f"  {row['reference']}: {row['passage'][:160]}")
    else:
        for side, label in (("for", "entailment"), ("against", "contradiction")):
            print(f"\n{side.upper()}:")
            for row in result[side]:
                print(f"  {row['reference']}: {row['contextual'][label]:.2f} — {row['passage'][:160]}")
        if not result["for"] and not result["against"]:
            print("No passage met the evidence threshold.")
