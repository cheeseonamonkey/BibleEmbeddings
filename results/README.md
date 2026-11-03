# Verse Lawyer — Bible-specific fine-tuning v1

[Training notebook](https://www.kaggle.com/code/ffatty/verse-lawyer-domain-v1-149m)
· [Original joint baseline](https://www.kaggle.com/code/ffatty/verse-lawyer-modernbert-149m-joint)

One **149,607,171-parameter ModernBERT** checkpoint: pooled embeddings for
retrieval and a 3-way NLI head for evidence. BM25 is a non-neural fallback.

## Training

- Continued the original v1 Kaggle checkpoint (not a second neural model).
- 363 authored, provisionally labeled Bible pairs from 109 cited passages.
- Included difficult unrelated claims and multi-verse cases; no examples near
  the existing 30-case evaluation passages.
- Mixed with 2,200 general MultiNLI cases, repeating domain cases four times.
- Joint objective: NLI classification + contrastive retrieval for **both**
  support and contradiction (neither is a neutral retrieval target).
- Two epochs, selected **epoch 1** via development performance. Epoch 2
  performed slightly worse on the Bible development set.
- Saved one model checkpoint (~598 MB), tokenizer, and metrics.

The model weights are in the Kaggle notebook outputs, **not GitHub**.
Model SHA-256: `d3280a40243750a1cee529dce25a1f665e58d090607fd517d803f1b4dc80e3fc`.

## Evaluation

| Evaluation | Baseline | Domain-tuned |
|---|---:|---:|
| General NLI accuracy (1,000 examples) | 76.9% | **79.5%** |
| Bible development accuracy (30 examples, previously inspected) | 86.7% | **93.3%** |
| **Fresh Bible holdout accuracy (96 examples)** | **85.4%** | **93.8%** |
| Fresh Bible macro-F1 | 85.4% | **93.8%** |
| Fresh support F1 | 86.7% | **92.3%** |
| Fresh contradiction F1 | 85.7% | **90.6%** |
| Fresh neutral F1 | 83.9% | **98.4%** |

On the **96 fresh cases**, nine old errors were corrected and one formerly
correct prediction regressed. The holdout comprises 32 unseen passage groups,
three claims per group. It was created after the training submission and was
never an input to the training or checkpoint-selection notebook.

### Retrieval probe (128 pairs per label)

| Recall | Original | Domain-tuned |
|---|---:|---:|
| Entailment @1 | 87.5% | 89.1% |
| Entailment @10 | 96.9% | 98.4% |
| Contradiction @1 | 71.1% | **69.5%** |
| Contradiction @10 | 90.6% | 93.0% |

Contradiction Recall@1 slipped slightly; **do not claim retrieval improved
uniformly**. These are small MultiNLI probes, not whole-Bible search benchmarks.

### Illustrative development probes

- "George Bush orchestrated 9/11" + Proverbs 18:13: **neutral**.
- "God prefers cats to dogs" + Genesis 1:25: **neutral**.
- "The moon is made of cheese" + Ecclesiastes 10:19: **neutral**.
- "Same-sex civil marriage should be prohibited by civil law" +
  Matthew 19:5: **neutral**.
- "The Earth is flat" + Isaiah 40:22: **incorrectly contradiction**.
- "You should always answer a fool" + Proverbs 26:4–5: **contradiction**;
  this example's provisional "neutral" annotation is arguably disputable.

## Limits

These are curated, small, correlated proposition examples. The original 30
cases were already examined, and even the 96 new examples have not been
independently adjudicated. Classifications reflect linguistic evidence, **not
historical truth, doctrinal consensus, or reliable probability calibration**.

Next: independent annotation of a larger claim/passage dataset; explicit
passage-level ranking benchmarks, especially contradiction retrieval; expanded
multi-verse pericopes and original-language/translation checks where justified.

Raw metrics and the paired holdout comparison are in this directory.
