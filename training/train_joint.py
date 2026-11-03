"""Single-backbone ModernBERT training: NLI + entailment-pair contrastive retrieval.

No second neural model, adapters, artificial Bible labels, or generative judge.
"""
from __future__ import annotations

import json
import os
import random
from pathlib import Path

# One Kaggle T4 avoids DataParallel and keeps the checkpoint easy to reproduce.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

BASE = "answerdotai/ModernBERT-base"
LABELS = ("entailment", "neutral", "contradiction")


def main(output_dir="/kaggle/working/verse-joint-v1", train_samples=6000,
         eval_samples=1000, epochs=1, batch_size=8, accum_steps=2,
         retrieval_weight=0.15, seed=42):
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
        raise RuntimeError("No GPU available; refusing accidental CPU training")
    print("GPU", torch.cuda.get_device_name(0), "torch", torch.__version__, flush=True)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    tokenizer = AutoTokenizer.from_pretrained(BASE)
    model = AutoModelForSequenceClassification.from_pretrained(
        BASE, num_labels=3, reference_compile=False, attn_implementation="eager",
        id2label={i: v for i, v in enumerate(LABELS)},
        label2id={v: i for i, v in enumerate(LABELS)},
    ).cuda()
    assert model.config.hidden_size > 0
    model.config.reference_compile = False

    training = load_dataset("nyu-mll/multi_nli", split="train").shuffle(seed=seed)
    validation = load_dataset("nyu-mll/multi_nli", split="validation_matched").shuffle(seed=seed)
    assert tuple(training.features["label"].names) == LABELS
    training = training.select(range(min(len(training), train_samples)))
    validation = validation.select(range(min(len(validation), eval_samples)))
    print("Train/eval", len(training), len(validation), flush=True)

    def batch_rows(batch):
        return {key: [row[key] for row in batch] for key in ("premise", "hypothesis", "label")}

    loader = DataLoader(training, shuffle=True, batch_size=batch_size,
                        collate_fn=batch_rows, generator=torch.Generator().manual_seed(seed))
    val_loader = DataLoader(validation, shuffle=False, batch_size=batch_size,
                            collate_fn=batch_rows)

    def tokenize(a, b=None):
        return tokenizer(a, b, padding=True, truncation=True, max_length=192,
                         return_tensors="pt").to("cuda")

    def pool(texts):
        batch = tokenize(texts)
        hidden = model.base_model(**batch).last_hidden_state
        mask = batch["attention_mask"].unsqueeze(-1)
        mean = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1)
        return F.normalize(mean.float(), dim=-1)

    @torch.no_grad()
    def evaluate():
        model.eval()
        count, correct = 0, 0
        table = torch.zeros((3, 3), dtype=torch.int64)
        for rows in val_loader:
            gold = torch.tensor(rows["label"], dtype=torch.long, device="cuda")
            with torch.autocast("cuda", dtype=torch.float16):
                pred = model(**tokenize(rows["premise"], rows["hypothesis"])).logits.argmax(-1)
            count += len(gold)
            correct += int((pred == gold).sum())
            for g, p in zip(gold.tolist(), pred.tolist()):
                table[g, p] += 1
        f1s = []
        for c in range(3):
            tp = int(table[c, c])
            denom = int(table[c, :].sum() + table[:, c].sum())
            f1s.append(2 * tp / denom if denom else 0.0)
        return {"accuracy": correct / max(1, count),
                "macro_f1": sum(f1s) / len(f1s),
                "confusion_matrix": table.tolist()}

    @torch.no_grad()
    def retrieval_probe():
        """Held-out entailment pairs, one example per distinct premise."""
        model.eval()
        pairs, seen = [], set()
        for row in validation:
            if row["label"] == 0 and row["premise"] not in seen:
                pairs.append((row["hypothesis"], row["premise"]))
                seen.add(row["premise"])
            if len(pairs) == 128:
                break
        if len(pairs) < 10:
            return {"error": "Insufficient held-out entailments"}
        with torch.autocast("cuda", dtype=torch.float16):
            queries = torch.cat([pool([x[0] for x in pairs[i:i+8]])
                                 for i in range(0, len(pairs), 8)])
            passages = torch.cat([pool([x[1] for x in pairs[i:i+8]])
                                  for i in range(0, len(pairs), 8)])
        scores = queries @ passages.T
        rank = scores.argsort(dim=1, descending=True)
        correct_indices = torch.arange(len(pairs), device="cuda")
        positions = (rank == correct_indices[:, None]).int().argmax(dim=1) + 1
        return {"pairs": len(pairs), "recall_at_1": float((positions <= 1).float().mean()),
                "recall_at_10": float((positions <= 10).float().mean()),
                "mrr": float((1 / positions.float()).mean())}

    before = retrieval_probe()
    print("Base retrieval", json.dumps(before), flush=True)
    steps_per_epoch = (len(loader) + accum_steps - 1) // accum_steps
    total_steps = steps_per_epoch * epochs
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-5, weight_decay=0.01)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=max(1, round(total_steps * 0.06)),
        num_training_steps=total_steps)
    scaler = torch.amp.GradScaler("cuda")
    optimizer.zero_grad(set_to_none=True)
    history = []

    for epoch in range(epochs):
        model.train()
        running_nli, running_retrieval = 0.0, 0.0
        for step, rows in enumerate(loader, start=1):
            labels = torch.tensor(rows["label"], device="cuda", dtype=torch.long)
            with torch.autocast("cuda", dtype=torch.float16):
                logits = model(**tokenize(rows["premise"], rows["hypothesis"])).logits
                nli_loss = F.cross_entropy(logits, labels)
                positives = (labels == 0).nonzero(as_tuple=True)[0].tolist()
                retrieval_loss = logits.sum() * 0  # retains correct device/dtype
                if len(positives) >= 2:
                    query = pool([rows["hypothesis"][i] for i in positives])
                    passage = pool([rows["premise"][i] for i in positives])
                    similarities = query @ passage.T / 0.07
                    identity = torch.arange(len(positives), device="cuda")
                    retrieval_loss = (
                        F.cross_entropy(similarities, identity) +
                        F.cross_entropy(similarities.T, identity)
                    ) / 2
                loss = nli_loss + retrieval_weight * retrieval_loss
            scaler.scale(loss / accum_steps).backward()
            running_nli += float(nli_loss.detach())
            running_retrieval += float(retrieval_loss.detach())
            if step % accum_steps == 0 or step == len(loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
            if step % 100 == 0:
                print(f"epoch {epoch+1} step {step}/{len(loader)} "
                      f"nli={running_nli/step:.3f} retrieve={running_retrieval/step:.3f}",
                      flush=True)
        classification = evaluate()
        retrieval = retrieval_probe()
        history.append({"epoch": epoch+1, "nli": classification,
                        "retrieval": retrieval})
        print("Evaluation", json.dumps(history[-1]), flush=True)

    out = Path(output_dir)
    checkpoint = out / "model"
    checkpoint.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(checkpoint, safe_serialization=True)
    tokenizer.save_pretrained(checkpoint)
    metrics = {
        "base_model": BASE, "parameters": sum(x.numel() for x in model.parameters()),
        "train_samples": len(training), "eval_samples": len(validation),
        "epochs": epochs, "batch_size": batch_size, "accum_steps": accum_steps,
        "retrieval_weight": retrieval_weight, "seed": seed,
        "base_retrieval": before, "history": history,
        "note": "ONE shared encoder trained for NLI and contrastive retrieval; "
                "MNLI-only proof of concept, not biblical-domain specialization.",
    }
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print("Saved", str(checkpoint), json.dumps(metrics), flush=True)
    return metrics


if __name__ == "__main__":
    main()
