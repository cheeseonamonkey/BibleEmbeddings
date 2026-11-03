"""Dataset provenance and label safeguards (no full KJV download required for CI)."""
import hashlib
import json
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def test_domain_data_is_reproducible_and_balanced():
    path = ROOT / "training/domain_pairs.jsonl"
    data = load_rows(path)
    manifest = json.loads((ROOT / "training/domain_manifest.json").read_text())
    assert hashlib.sha256(path.read_bytes()).hexdigest() == manifest["data_sha256"]
    assert len(data) == manifest["count"] >= 300
    assert {row["label"] for row in data} == {0, 1, 2}
    labels = Counter(row["label"] for row in data)
    assert min(labels.values()) >= 100
    assert len({(row["premise"], row["hypothesis"]) for row in data}) == len(data)


def test_curated_training_does_not_reuse_existing_dev_passages():
    domain = load_rows(ROOT / "training/domain_pairs.jsonl")
    dev = load_rows(ROOT / "evaluation/seed_pairs.jsonl")
    assert len(dev) == 30
    assert not ({row["premise"] for row in domain} &
                {row["premise"] for row in dev})


def test_curated_references_span_multiple_verses():
    rows = load_rows(ROOT / "training/domain_pairs.jsonl")
    assert sum("–" in row["reference"] for row in rows) >= 20
    assert any(row["kind"] == "hard_neutral" for row in rows)
