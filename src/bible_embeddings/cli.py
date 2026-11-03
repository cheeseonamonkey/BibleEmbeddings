"""CLI: index/search the Bible and optionally score textual evidence."""
from __future__ import annotations

import argparse
import json

from .corpus import load_passages
from .retrieval import Retriever


def main() -> None:
    parser = argparse.ArgumentParser(description="Find scriptural arguments for and against a claim")
    parser.add_argument("--data", required=True, help="KJV JSON book/chapter/verse source")
    parser.add_argument("--claim", help="Any proposition")
    parser.add_argument("--embed-model", help="Optional SentenceTransformer identifier")
    parser.add_argument("--cache", help="Embedding cache .npz (optional)")
    parser.add_argument("--nli-model", help="Trained NLI checkpoint (optional)")
    parser.add_argument("--limit", type=int, default=40)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    passages = load_passages(args.data)
    retriever = Retriever(passages, args.embed_model, args.cache, args.data)
    if not args.claim:
        print(f"Indexed {len(passages):,} single/multi-verse passages")
        return
    hits = retriever.search(args.claim, limit=args.limit, remove_overlap=not bool(args.nli_model))
    if not args.nli_model:
        result = {
            "status": "retrieval_only",
            "claim": args.claim,
            "candidates": [{
                "reference": p.reference, "passage": p.text,
                "retrieval_score": round(score, 5)
            } for p, score in hits],
        }
    else:
        from .judge import NLIJudge, top_evidence
        scored = NLIJudge(args.nli_model).evaluate(hits, args.claim)
        result = {
            "status": "model_evidence_signals_not_truth_probabilities",
            "claim": args.claim,
            "for": [e.as_dict() for e in top_evidence(scored, "entailment")],
            "against": [e.as_dict() for e in top_evidence(scored, "contradiction")],
            "inspected": len(scored)
        }

    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print(f"Claim: {args.claim} ({result['status']})")
        if "candidates" in result:
            for row in result["candidates"][:10]:
                print(f"  {row['reference']}: {row['passage'][:160]}")
        else:
            for side in ("for", "against"):
                print(f"\n{side.upper()}:")
                for row in result[side]:
                    signal = row["contextual"]["entailment" if side == "for" else "contradiction"]
                    print(f"  {row['reference']}: {signal:.2f} — {row['passage'][:160]}")
            if not result["for"] and not result["against"]:
                print("No passage met the evidence threshold.")
