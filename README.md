# BibleEmbeddings — Verse Lawyer

## Public web demo

[Verse Lawyer](https://mansion.041700.xyz/verse-lawyer/) runs the
saved 149.6M-parameter checkpoint on CPU with 14 BM25 candidates and contextual
NLI. It intentionally retains the measured failures (including the burning
bush example). The [paper (PDF)](https://mansion.041700.xyz/verse-lawyer/paper.pdf)
is linked only in the page footer. This is not a historical fact-checker.

The other public experiments use the same IPv4-only hostname:
[Alex Voice](https://mansion.041700.xyz/alexvoice/) and
[Typability](https://mansion.041700.xyz/typability/).
The former VPS-hostname public demo paths issue scoped, permanent **308 redirects**;
management/MCP routes are intentionally not redirected.

The page, API and Nginx configuration live under `web/` and `deploy/`;
model weights and the Bible corpus live outside the public static root.
The paper source and its ReportLab builder live under `paper/`.

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

Always report Bible development, independent holdout, general NLI, and retrieval
scores before promoting a checkpoint. The Bible datasets are small and
provisionally annotated; results are not a definitive generalization measure.

**Next:** create 200–500 carefully labeled claims with hard negatives and
multiple interpretations. Group test splits by related passages and proposition
families. Compare with lexical-only retrieval and general-NLI baselines before
claiming improved scriptural reasoning.

## Completed domain-tuning results

The [Bible-specific continuation](https://www.kaggle.com/code/ffatty/verse-lawyer-domain-v1-149m)
finished successfully (best checkpoint from epoch 1). On a **96-example fresh
holdout** across 32 unused passages, accuracy rose from **85.4% to 93.8%**;
neutral F1 rose from **83.9% to 98.4%**. General MultiNLI accuracy increased
from 76.9% to 79.5%. Contradiction retrieval Recall@1 slipped slightly, from
71.1% to 69.5%, so semantic retrieval is not uniformly improved.

The ~598 MB checkpoint is in the Kaggle notebook outputs, not GitHub:

```sh
kaggle kernels output ffatty/verse-lawyer-domain-v1-149m -p out -o
verse-lawyer --data data/en_kjv.json --model out/verse-domain-v1/model \
  --claim "the moon is made of cheese" --json
```

See [the measured results and limitations](results/README.md), including the
paired holdout comparison, model checksum, mistakes, and remaining work.

The model cannot establish historical facts from verses about unrelated events.
