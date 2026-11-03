# BibleEmbeddings — Verse Lawyer

**Find biblical arguments FOR and AGAINST any proposition.** A research experiment in
prooftexting, not a predictor of what is true.

## One neural model, two jobs

- **Retrieve:** BM25 lexical search, optionally fused with dense passage vectors
  produced by **the same ModernBERT checkpoint**. No separate embedding model.
- **Judge:** the same ~149M-parameter ModernBERT with a 3-class NLI head scores
  entailment, contradiction, and neutral evidence. No generative agent required.
- **Context:** score each isolated passage and its surrounding verses separately.
  Show possible cherry-picking, without collapsing conflicting adjacent verses.
- **Passages:** 1-, 2-, 3-, and 5-verse overlapping windows, with exact references.
  Longer pericopes and cross-chapter sections can be added later.

The neural checkpoint is fine-tuned jointly on **NLI cross-entropy** plus a modest
**contrastive retrieval objective** from entailment pairs. This trains one shared
backbone, not two independent models. The first run uses MultiNLI to test the
pipeline; biblical adaptation requires better labeled data.

## Run

Get the KJV JSON from [thiagobodruk/bible](https://github.com/thiagobodruk/bible)
(`json/en_kjv.json`) and place it at `data/en_kjv.json`.

```sh
pip install -e .
verse-lawyer --data data/en_kjv.json --claim "answer a fool"

# After training and downloading the one checkpoint:
pip install -e '.[models]'
verse-lawyer --data data/en_kjv.json --claim "answer a fool" \
  --model out/verse-joint-v1/model --json

# Optional dense retrieval using that SAME model (slow on first indexing):
verse-lawyer --data data/en_kjv.json --claim "answer a fool" \
  --model out/verse-joint-v1/model --semantic --cache out/kjv-joint.npz
```

BM25-only output is retrieval candidates, not evidence. Scores are heuristic
model signals, **not probabilities a biblical or historical proposition is true**.

## Kaggle training

```sh
python training/build_notebook.py
kaggle kernels push -p kaggle --accelerator NvidiaTeslaT4
kaggle kernels status ffatty/verse-lawyer-modernbert-149m-joint
```

The notebook runs on one T4 to avoid DataParallel compilation conflicts. It
loads ModernBERT with `reference_compile=False`, trains on 6,000 MNLI pairs,
validates on 1,000 held-out examples, and reports accuracy, macro-F1, and
held-out retrieval Recall@1/10. Saves a single checkpoint and metrics.

The 30 provisional Bible-specific pairs in `evaluation/seed.jsonl` are a
smoke test only, not training data or a reliable benchmark:

```sh
python evaluation/evaluate.py --data data/en_kjv.json \
  --model out/verse-joint-v1/model
```

**Next:** create 200–500 carefully labeled claims with hard negatives and
multiple interpretations. Group test splits by related passages and proposition
families. Compare with lexical-only retrieval and general-NLI baselines before
claiming improved scriptural reasoning.

The model cannot establish historical facts from verses about unrelated events.
