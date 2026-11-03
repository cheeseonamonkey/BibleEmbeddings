# Verse Lawyer — adversarial smoke tests

**Date:** 2026-11-03
**Checkpoint:** domain-tuned ModernBERT (149,607,171 parameters)
**Weights SHA-256:** d3280a40243750a1cee529dce25a1f665e58d090607fd517d803f1b4dc80e3fc

To reproduce locally, download the KJV corpus and domain-tuned checkpoint,
then run PYTHONPATH=src python experiments/fun_probes.py from the repository.

- **Direct:** 24 authored claim–passage pairs, using the cited text as premise.
- **End-to-end:** 13 propositions, whole-Bible BM25 across 116,091 overlapping
  one-/two-/three-/five-verse windows, 14 candidates per claim; isolated and
  expanded-context NLI evaluations. The dense retrieval index was NOT tested.

Machine-readable outputs: [direct.json](results/direct.json) and
[search.json](results/search.json). These are model judgments, not historical truth.

## Observations

**Direct:** 21/24 provisional expected labels (87.5%). Three errors include
Isaiah 40:22 confidently contradicting a flat Earth, Proverbs 26:4 misreading
"answer not a fool", and Proverbs 26:4–5 marking a debatable neutral label as
contradiction.

**End-to-end:** all 13 queries returned at least one FOR/AGAINST citation.
That includes several unsupported modern assertions, so the pipeline currently
fails to abstain reliably. No end-to-end accuracy has been measured because
retrieved candidates lack independent relevance labels.

| Claim | Returned citation | Model signal | Failure |
|---|---|---:|---|
| Bush did 9/11. | Exodus 3:2–4 | 0.42 FOR | Burning bush ≠ George Bush |
| The Bible predicted TikTok. | Ezra 4:9 | 0.59 FOR | Unrelated genealogy |
| God prefers cats over dogs. | Psalm 22:16 | 0.59 AGAINST | Mentions dogs, not preferences |
| Money answers all things. | Ecclesiastes 10:19 | 0.51 AGAINST | Verse literally says it |
| You should always love your enemies. | Matthew 5:43–47 | 0.84 AGAINST | Missed corrective quotation |
| It is better to give than receive. | Genesis 29:19 | 0.96 FOR | Laban's bargain ≠ general ethics |
| The Sabbath was made for man. | John 5:9 | 0.94 FOR | Sabbath story but wrong proof |
| There is a time to dance. | Ecclesiastes 3:3–5 | 0.64 AGAINST | Verse explicitly permits time to dance |

Scores are raw model confidences, not calibrated truth probabilities.

## What broke

1. **Retrieval relevance:** lexical matching confuses names and generic words.
   A candidate mentioning a word does not support the proposition.
2. **Context inversion:** Proverbs 26:5 in isolation supports "answer a fool"
   at 0.999, but the surrounding context labels it contradiction at 0.757.
   Ecclesiastes 10:19 similarly flips 0.971 entailment to 0.515 contradiction.
   Current ranking relies on the surrounding-context label alone.
3. **Weak abstention:** a 0.40 selection threshold allows very weak evidence,
   but threshold increases alone cannot fix high-confidence mistakes.
4. **Sliding windows:** overlapping passages disagree about identical claims.
5. **Benchmark mismatch:** the earlier 93.8% Bible holdout was a passage
   classification test, not proof of end-to-end retrieval quality.

## Next investigations

- Create a small gold retrieval benchmark including no-evidence claims:
  report recall, precision, wrong-sided citations, and abstention.
- Treat isolated/context disagreements as a warning or abstention rather than
  silently reversing evidence. Validate the policy on held-out queries.
- Evaluate larger candidate pools, stopword filtering, and the shared-model
  dense retrieval path. Remain **one neural model**, with BM25 optional.
- Label the distinction between descriptive narrative, moral injunction,
  and modern extrapolation. Preserve exact verse references.

This is a deliberately adversarial smoke test, not formal benchmark evidence
or authoritative biblical interpretation.
