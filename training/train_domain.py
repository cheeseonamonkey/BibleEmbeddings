"""Continue the SAME 149M ModernBERT with curated Bible NLI + dual-side retrieval.

The separate adapter/model approach is intentionally excluded.
The 30-case Bible development set is *never* used as training data.
"""
from __future__ import annotations

import json
import math
import os
import random
from collections import Counter
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

LABELS = ("entailment", "neutral", "contradiction")


def main(
    checkpoint: str,
    domain_jsonl: str,
    bible_eval_jsonl: str,
    output: str = "/kaggle/working/verse-domain-v1",
    domain_repeats: int = 4,
    general_samples: int = 2200,
    epochs: int = 2,
    batch_size: int = 8,
    accumulation: int = 2,
    retrieval_weight: float = .08,
    seed: int = 2026,
):
    import numpy as np
    import torch
    import torch.nn.functional as F
    from datasets import load_dataset
    from torch.utils.data import DataLoader
    from transformers import (
        AutoModelForSequenceClassification, AutoTokenizer,
        get_linear_schedule_with_warmup,
    )

    if not torch.cuda.is_available():
        raise RuntimeError("GPU not allocated")
    print(f"GPU: {torch.cuda.get_device_name(0)}; torch={torch.__version__}", flush=True)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    domain = [json.loads(line) for line in Path(domain_jsonl).read_text().splitlines() if line]
    bible_dev = [json.loads(line) for line in Path(bible_eval_jsonl).read_text().splitlines() if line]
    if not domain or not bible_dev:
        raise ValueError("Empty domain training or Bible development set")
    # Fail closed if an exact premise from development has leaked into training.
    if {r["premise"] for r in domain} & {r["premise"] for r in bible_dev}:
        raise ValueError("Bible evaluation premise leaked into domain training")
    print("Bible training cases", len(domain), "development", len(bible_dev), flush=True)

    source = Path(checkpoint)
    if not (source / "config.json").exists() or not list(source.glob("*.safetensors")):
        raise FileNotFoundError(f"No checkpoint at {checkpoint}")
    model = AutoModelForSequenceClassification.from_pretrained(
        checkpoint, reference_compile=False, attn_implementation="eager",
    ).cuda()
    model.config.reference_compile = False
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    if tuple(model.config.id2label[i].lower() for i in range(3)) != LABELS:
        raise ValueError(f"Unexpected label mapping: {model.config.id2label}")
    print("Parameters", sum(x.numel() for x in model.parameters()), flush=True)

    general_train = load_dataset("nyu-mll/multi_nli", split="train").shuffle(seed=seed)
    general_dev = load_dataset("nyu-mll/multi_nli", split="validation_matched").shuffle(seed=42)
    if tuple(general_train.features["label"].names) != LABELS:
        raise ValueError("MNLI label order changed")
    general_train = general_train.select(range(min(general_samples, len(general_train))))
    general_dev = general_dev.select(range(1000))
    generic_rows = [
        {k: row[k] for k in ("premise", "hypothesis", "label")}
        for row in general_train
    ]
    train = generic_rows + domain * domain_repeats
    rng = random.Random(seed)
    rng.shuffle(train)
    print("Training", len(train), "generic", len(generic_rows),
          "curated", len(domain), "repeat", domain_repeats,
          "class balance", dict(Counter(x["label"] for x in train)), flush=True)

    def collate(rows):
        return {key: [r[key] for r in rows] for key in ("premise", "hypothesis", "label")}

    loader = DataLoader(train, batch_size=batch_size, shuffle=True, collate_fn=collate,
                        generator=torch.Generator().manual_seed(seed))
    general_loader = DataLoader(general_dev, batch_size=16, shuffle=False, collate_fn=collate)
    bible_loader = DataLoader(bible_dev, batch_size=8, shuffle=False, collate_fn=collate)

    def encode(first, second=None):
        return tokenizer(first, second, padding=True, truncation=True,
                         max_length=256, return_tensors="pt").to("cuda")

    def embed(texts):
        batch = encode(texts)
        states = model.base_model(**batch).last_hidden_state
        mask = batch["attention_mask"].unsqueeze(-1)
        return F.normalize(((states * mask).sum(1) / mask.sum(1).clamp(min=1)).float(),
                           dim=-1)

    @torch.no_grad()
    def classification_metrics(data_loader):
        model.eval()
        counts = torch.zeros((3, 3), dtype=torch.long)
        errors = []
        for rows in data_loader:
            with torch.autocast("cuda", dtype=torch.float16):
                logits = model(**encode(rows["premise"], rows["hypothesis"])).logits
            predictions = logits.argmax(-1).cpu().tolist()
            for premise, claim, gold, predicted in zip(
                    rows["premise"], rows["hypothesis"], rows["label"], predictions):
                counts[gold, predicted] += 1
                if gold != predicted and len(errors) < 12:
                    errors.append({"claim": claim, "expected": LABELS[gold],
                                   "predicted": LABELS[predicted],
                                   "passage_excerpt": premise[:180]})
        f1 = []
        for i in range(3):
            tp = int(counts[i, i])
            denominator = int(counts[i, :].sum() + counts[:, i].sum())
            f1.append(2 * tp / denominator if denominator else 0)
        return {"samples": int(counts.sum()),
                "accuracy": float(counts.trace() / counts.sum()),
                "macro_f1": sum(f1) / 3,
                "per_class_f1": dict(zip(LABELS, f1)),
                "confusion": counts.tolist(),
                "first_errors": errors}

    @torch.no_grad()
    def retrieval_probe():
        # A question about a passage should retrieve that passage whether
        # the passage SUPPORTS or CONTRADICTS the question.
        model.eval()
        examples = {"entailment": [], "contradiction": []}
        unique = set()
        for row in general_dev:
            if row["premise"] in unique:
                continue
            label = LABELS[row["label"]]
            if label in examples and len(examples[label]) < 128:
                examples[label].append((row["hypothesis"], row["premise"]))
                unique.add(row["premise"])
            if all(len(v) >= 128 for v in examples.values()):
                break
        results = {}
        for label, pairs in examples.items():
            with torch.autocast("cuda", dtype=torch.float16):
                q = torch.cat([embed([x[0] for x in pairs[i:i+16]])
                               for i in range(0, len(pairs), 16)])
                p = torch.cat([embed([x[1] for x in pairs[i:i+16]])
                               for i in range(0, len(pairs), 16)])
            ranks = (q @ p.T).argsort(-1, descending=True)
            positions = (ranks == torch.arange(len(pairs), device="cuda")[:, None]).int().argmax(-1) + 1
            results[label] = {
                "recall_at_1": float((positions <= 1).float().mean()),
                "recall_at_10": float((positions <= 10).float().mean()),
                "mrr": float((1 / positions.float()).mean()),
            }
        return results

    baseline = {
        "general": classification_metrics(general_loader),
        "bible_dev": classification_metrics(bible_loader),
        "retrieval": retrieval_probe(),
    }
    print("BASELINE", json.dumps(baseline), flush=True)

    steps = epochs * math.ceil(len(loader) / accumulation)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.25e-5, weight_decay=0.01)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=max(1, round(steps * .05)),
        num_training_steps=steps)
    scaler = torch.amp.GradScaler("cuda")
    optimizer.zero_grad(set_to_none=True)
    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    best_score = -1.
    best_epoch = 0
    history = []

    for epoch in range(1, epochs + 1):
        model.train()
        total_nli = total_retrieval = 0.0
        for i, rows in enumerate(loader, 1):
            y = torch.tensor(rows["label"], dtype=torch.long, device="cuda")
            with torch.autocast("cuda", dtype=torch.float16):
                logits = model(**encode(rows["premise"], rows["hypothesis"])).logits
                nli = F.cross_entropy(logits, y)
                # Both FOR and AGAINST passages are relevant retrieval targets.
                # Neutral statements remain NLI hard negatives, not positive
                # semantic retrieval pairs.
                positives = (y != 1).nonzero(as_tuple=True)[0].tolist()
                retrieval = logits.sum() * 0
                if len(positives) > 1:
                    claims = [rows["hypothesis"][j] for j in positives]
                    premises = [rows["premise"][j] for j in positives]
                    q, p = embed(claims), embed(premises)
                    similarity = q @ p.T / 0.07
                    # If two examples cite the same passage, both are positives.
                    # Avoid forcing same-passage examples apart.
                    positives_mask = torch.tensor([
                        [a == b for b in premises] for a in premises
                    ], dtype=torch.bool, device="cuda")
                    log_all = torch.logsumexp(similarity, dim=1)
                    log_positives = torch.logsumexp(
                        similarity.masked_fill(~positives_mask, float("-inf")), dim=1)
                    retrieval = (log_all - log_positives).mean()
                loss = nli + retrieval_weight * retrieval
            scaler.scale(loss / accumulation).backward()
            total_nli += float(nli.detach())
            total_retrieval += float(retrieval.detach())
            if i % accumulation == 0 or i == len(loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
            if i % 90 == 0:
                print(f"epoch {epoch}, batch {i}/{len(loader)}, "
                      f"NLI={total_nli / i:.3f}, retrieval={total_retrieval / i:.3f}",
                      flush=True)
        record = {
            "epoch": epoch,
            "general": classification_metrics(general_loader),
            "bible_dev": classification_metrics(bible_loader),
            "retrieval": retrieval_probe(),
        }
        history.append(record)
        print("EVALUATION", json.dumps(record), flush=True)
        # Small development set: don't choose an apparent Bible gain if
        # it destroys the general-language classifier.
        score = .65 * record["bible_dev"]["macro_f1"] + .35 * record["general"]["macro_f1"]
        if score > best_score:
            best_score, best_epoch = score, epoch
            directory = out / "model"
            directory.mkdir(exist_ok=True)
            model.save_pretrained(directory, safe_serialization=True)
            tokenizer.save_pretrained(directory)
            print("Saved best checkpoint", directory, flush=True)

    metadata = {
        "base_checkpoint": checkpoint,
        "parameters": sum(x.numel() for x in model.parameters()),
        "train_domain_examples": len(domain),
        "training_total_after_repeats": len(train),
        "domain_repeats": domain_repeats,
        "general_train": len(generic_rows),
        "bible_dev_examples": len(bible_dev),
        "epochs": epochs, "seed": seed,
        "baseline": baseline,
        "history": history, "best_epoch": best_epoch,
        "best_score": best_score,
        "domain_note": "Curated proposition labels are not independently verified.",
        "eval_note": "30-case Bible dev set already inspected; not sealed holdout.",
    }
    (out / "metrics.json").write_text(json.dumps(metadata, indent=2))
    print("FINISHED", json.dumps(metadata), flush=True)
    return metadata


if __name__ == "__main__":
    raise SystemExit("Run via Kaggle notebook with checkpoint and embedded datasets.")
