"""Compare isolated passages with surrounding context using a calibrated NLI head."""
from __future__ import annotations

from dataclasses import asdict, dataclass

from .corpus import Passage, overlap


@dataclass
class Evidence:
    passage: Passage
    retrieval_score: float
    isolated: dict[str, float]
    contextual: dict[str, float]

    def as_dict(self) -> dict:
        return {
            "reference": self.passage.reference,
            "passage": self.passage.text,
            "context": self.passage.context,
            "retrieval_score": round(self.retrieval_score, 5),
            "isolated": self.isolated,
            "contextual": self.contextual,
            "cherry_pick_drop": round(max(0., self.isolated["entailment"] -
                                          self.contextual["entailment"]), 4),
        }


class NLIJudge:
    LABELS = ("entailment", "neutral", "contradiction")

    def __init__(self, model_path: str, batch_size: int = 8, device: str | None = None):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path).to(self.device).eval()
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.batch_size = batch_size
        labels = {int(k): str(v).lower() for k, v in self.model.config.id2label.items()}
        if set(labels.values()) != set(self.LABELS):
            raise ValueError(f"Expected NLI labels, got {labels}")
        self.ids = {name: i for i, name in labels.items()}

    def predict(self, premises: list[str], hypothesis: str) -> list[dict[str, float]]:
        results = []
        for start in range(0, len(premises), self.batch_size):
            batch = premises[start:start + self.batch_size]
            inputs = self.tokenizer(batch, [hypothesis] * len(batch),
                                    padding=True, truncation=True, max_length=512,
                                    return_tensors="pt").to(self.device)
            with self.torch.inference_mode():
                probabilities = self.torch.softmax(self.model(**inputs).logits, -1).cpu().tolist()
            results.extend({
                label: round(float(row[self.ids[label]]), 4)
                for label in self.LABELS
            } for row in probabilities)
        return results

    def evaluate(self, candidates: list[tuple[Passage, float]], claim: str) -> list[Evidence]:
        isolated = self.predict([p.text for p, _ in candidates], claim)
        contextual = self.predict([p.context for p, _ in candidates], claim)
        return [Evidence(p, float(score), a, b) for (p, score), a, b in
                zip(candidates, isolated, contextual)]


def top_evidence(evidence: list[Evidence], label: str, n: int = 5,
                 threshold: float = .4) -> list[Evidence]:
    if label not in ("entailment", "contradiction"):
        raise ValueError(f"Unsupported evidence label {label}")
    ordered = sorted(
        (e for e in evidence if e.contextual[label] >= threshold),
        key=lambda e: (-e.contextual[label], -e.isolated[label], -e.retrieval_score)
    )
    # Deduplicate within each stance, never across FOR and AGAINST.
    # Adjacent verses can explicitly disagree (e.g., Proverbs 26:4–5).
    selected = []
    for evidence_item in ordered:
        if not any(overlap(evidence_item.passage, previous.passage) for previous in selected):
            selected.append(evidence_item)
            if len(selected) == n:
                break
    return selected
