"""Smoke-evaluate a trained NLI checkpoint on tentative hand-labeled KJV cases."""
import argparse
import json
from collections import Counter
from pathlib import Path

from bible_embeddings.corpus import load_passages
from bible_embeddings.judge import NLIJudge

LABELS = ("entailment", "neutral", "contradiction")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="KJV JSON corpus")
    ap.add_argument("--model", required=True, help="NLI checkpoint directory")
    ap.add_argument("--seed", default="evaluation/seed.jsonl")
    ap.add_argument("--out", default="out/bible-eval.json")
    args = ap.parse_args()
    refs = {p.reference: p for p in load_passages(args.data)}
    rows = [json.loads(line) for line in Path(args.seed).read_text().splitlines() if line]
    judge = NLIJudge(args.model)
    confusion = {label: Counter() for label in LABELS}
    correct = 0
    details = []
    for row in rows:
        if row["reference"] not in refs:
            raise ValueError(f"Unknown reference: {row['reference']}")
    claims = [row.get("hypothesis", row.get("claim")) for row in rows]
    if any(not isinstance(claim, str) for claim in claims):
        raise ValueError("Missing hypothesis or claim in evaluation rows")
    scores = judge.predict_pairs([refs[row["reference"]].text for row in rows], claims)
    for row, claim, probabilities in zip(rows, claims, scores):
        predicted = max(probabilities, key=probabilities.get)
        raw_label = row["label"]
        expected = LABELS[raw_label] if isinstance(raw_label, int) else raw_label
        if expected not in LABELS:
            raise ValueError(f"Invalid label {raw_label}")
        confusion[expected][predicted] += 1
        correct += predicted == expected
        details.append({"reference": row["reference"], "claim": claim,
                        "label": expected, "predicted": predicted,
                        "scores": probabilities})

    f1 = {}
    for label in LABELS:
        true_positive = confusion[label][label]
        false_positive = sum(confusion[other][label] for other in LABELS if other != label)
        false_negative = sum(confusion[label][other] for other in LABELS if other != label)
        denom = 2*true_positive + false_positive + false_negative
        f1[label] = 2*true_positive / denom if denom else 0.0
    result = {
        "samples": len(rows),
        "accuracy": correct / len(rows),
        "macro_f1": sum(f1.values()) / len(f1),
        "class_f1": f1,
        "confusion": {label: dict(confusion[label]) for label in LABELS},
        "warning": "Small provisional evaluation. Not a calibrated or independent scientific benchmark.",
        "details": details,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))
    print(json.dumps({k: v for k, v in result.items() if k != "details"}, indent=2))


if __name__ == "__main__":
    main()
