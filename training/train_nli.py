"""Small reproducible ModernBERT MNLI baseline; no generated biblical labels yet."""
from __future__ import annotations

import json
from pathlib import Path


MODEL = "answerdotai/ModernBERT-base"
LABELS = ["entailment", "neutral", "contradiction"]


def main(output_dir="out/modernbert-mnli", train_samples=6000,
         eval_samples=1000, epochs=1.0, seed=42):
    import numpy as np
    import torch
    from datasets import load_dataset
    from sklearn.metrics import accuracy_score, f1_score
    from transformers import (
        AutoModelForSequenceClassification, AutoTokenizer,
        DataCollatorWithPadding, Trainer, TrainingArguments,
    )

    if not torch.cuda.is_available():
        raise RuntimeError("Expected a Kaggle GPU; refusing accidental slow CPU training")
    print("GPU:", torch.cuda.get_device_name(0))
    torch.manual_seed(seed)
    base = load_dataset("nyu-mll/multi_nli", split="train")
    matched = load_dataset("nyu-mll/multi_nli", split="validation_matched")
    train_data = base.shuffle(seed=seed).select(range(min(train_samples, len(base))))
    eval_data = matched.shuffle(seed=seed).select(range(min(eval_samples, len(matched))))
    print("train", len(train_data), "eval", len(eval_data), "labels", train_data.features["label"])
    assert list(train_data.features["label"].names) == LABELS

    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    def encode(rows):
        return tokenizer(rows["premise"], rows["hypothesis"], truncation=True,
                         max_length=256)
    train_data = train_data.map(encode, batched=True,
                                remove_columns=[c for c in train_data.column_names if c != "label"])
    eval_data = eval_data.map(encode, batched=True,
                              remove_columns=[c for c in eval_data.column_names if c != "label"])

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL, num_labels=3,
        id2label={i: label for i, label in enumerate(LABELS)},
        label2id={label: i for i, label in enumerate(LABELS)},
    )
    def metrics(pred):
        guesses = np.argmax(pred.predictions, axis=-1)
        return {"macro_f1": f1_score(pred.label_ids, guesses, average="macro"),
                "accuracy": accuracy_score(pred.label_ids, guesses)}

    out = Path(output_dir)
    args = TrainingArguments(
        output_dir=str(out / "checkpoints"),
        per_device_train_batch_size=8,
        per_device_eval_batch_size=16,
        gradient_accumulation_steps=2,
        learning_rate=3e-5, num_train_epochs=epochs,
        warmup_ratio=0.06, weight_decay=0.01,
        fp16=True, bf16=False, optim="adamw_torch",
        eval_strategy="epoch", save_strategy="epoch",
        load_best_model_at_end=True, metric_for_best_model="macro_f1",
        greater_is_better=True, save_total_limit=1,
        logging_steps=40, report_to="none",
        dataloader_num_workers=2, seed=seed,
    )
    trainer = Trainer(
        model=model, args=args,
        train_dataset=train_data, eval_dataset=eval_data,
        data_collator=DataCollatorWithPadding(tokenizer), processing_class=tokenizer,
        compute_metrics=metrics,
    )
    trainer.train()
    result = trainer.evaluate()
    checkpoint = out / "model"
    trainer.save_model(str(checkpoint))
    tokenizer.save_pretrained(str(checkpoint))
    metadata = {
        "base_model": MODEL, "training_data": "nyu-mll/multi_nli",
        "training_samples": len(train_data), "eval_samples": len(eval_data),
        "seed": seed, "epochs": epochs, "metrics": result,
        "note": "General MNLI baseline, NOT fine-tuned on biblical interpretation.",
    }
    out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").write_text(json.dumps(metadata, indent=2))
    print(json.dumps(metadata, indent=2))
    return metadata


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", default="out/modernbert-mnli")
    ap.add_argument("--train-samples", type=int, default=6000)
    ap.add_argument("--eval-samples", type=int, default=1000)
    ap.add_argument("--epochs", type=float, default=1.)
    args = ap.parse_args()
    main(args.output_dir, args.train_samples, args.eval_samples, args.epochs)
