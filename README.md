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

## Bible-specific continuation (v2)

The first shared-encoder model has 149,607,171 parameters. Its general NLI
accuracy was 76.9%, with held-out entailment retrieval Recall@1 of 87.5%.
On the **small and previously inspected** Bible development set it got 26/30
right; its four errors were all unsupported claims misclassified as contradicted.

To target that issue, `training/build_domain_data.py` assembles 363 authored
claim/passage pairs across 109 KJV references: direct support, explicit
contradictions, topical non sequiturs, absurd modern claims, and multi-verse
examples. These are curated **provisional labels**, not independent expert
annotations. The original 30 Bible cases remain outside training, as do
adjacent verses. Training uses the KJV text pinned by SHA-256 in the manifest.

```sh
python training/build_domain_data.py
python training/build_domain_notebook.py
kaggle kernels push -p kaggle-domain --accelerator NvidiaTeslaT4
kaggle kernels status ffatty/verse-lawyer-domain-v1-149m
```

This notebook continues from the v1 Kaggle checkpoint (not from scratch), mixes
small repeated domain data with general MultiNLI examples, and optimizes NLI
plus evidence retrieval. **Both entailment and contradiction are relevant
retrieval targets**; neutral examples are NLI hard negatives. The only neural
network remains the one ModernBERT encoder with an NLI head.

Report the Bible development set, general NLI, and retrieval scores before
promoting a new checkpoint. Since the Bible set is tiny and already examined,
its accuracy is not a definitive measure of generalization. Train/test splits
must separate proposition families and overlapping scriptural passages.

**Next:** create 200–500 carefully labeled claims with hard negatives and
multiple interpretations. Group test splits by related passages and proposition
families. Compare with lexical-only retrieval and general-NLI baselines before
claiming improved scriptural reasoning.

The model cannot establish historical facts from verses about unrelated events.
