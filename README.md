# BibleEmbeddings — Verse Lawyer

**An experimental argument engine:** enter any proposition; retrieve scriptural passages
that might support or oppose it. Distinguish persuasive prooftexting from legitimate
textual evidence. This is not a fact-checker or theological truth detector.

## Architecture

1. **Hybrid retrieval:** BM25 + optional MiniLM embeddings; reciprocal rank fusion.
2. **Multi-verse passages:** overlapping 1-, 3- and 5-verse windows. Preserve book,
   chapter and verse ranges; never merge chapters. Overlapping retrieval hits are
   deduplicated to encourage independent evidence.
3. **Specialized judge:** fine-tuned ModernBERT-base (~149M parameters). NLI labels
   are *entailment*, *contradiction*, *neutral*. The claim is always the hypothesis.
4. **Context audit:** score isolated passage and surrounding context separately.
   The difference can indicate cherry-picking.
5. **Outputs:** ranked FOR/AGAINST with raw confidence signals and exact references.
   No forced winner or fabricated evidence.

## Quickstart

The corpus is not committed. Obtain KJV JSON from
[thiagobodruk/bible](https://github.com/thiagobodruk/bible), save
`json/en_kjv.json` to `data/en_kjv.json`.

```sh
python -m pip install -e .
verse-lawyer --data data/en_kjv.json --claim "money buys happiness"

# Dense retrieval (optional; model downloaded once):
python -m pip install -e '.[models]'
verse-lawyer --data data/en_kjv.json --claim "money buys happiness" \
  --embed-model sentence-transformers/all-MiniLM-L6-v2 --cache out/kjv-minilm.npz

# After an NLI model has been trained and downloaded:
verse-lawyer --data data/en_kjv.json --claim "money buys happiness" \
  --nli-model out/modernbert-mnli/model --json
```

Retrieval-only output contains *candidates*, not evidence judgments.

## Initial GPU experiment

```sh
python training/build_notebook.py
kaggle kernels push -p kaggle --accelerator NvidiaTeslaT4
kaggle kernels status ffatty/verse-lawyer-modernbert-149m-nli-baseline
```

The notebook fine-tunes ModernBERT on a seeded subset of MultiNLI (6,000 train /
1,000 held-out evaluation pairs), reporting macro-F1 and accuracy, and exports
the best checkpoint. **This is a generic NLI baseline, not biblical adaptation.**

Next: curate 200–500 gold biblical evidence pairs, including counterfactuals,
paraphrases, translation-sensitive examples, and misleading isolated verses.
Group held-out evaluation by proposition family *and* overlapping passage to
prevent near-duplicate leakage. Only then add domain fine-tuning.

## Limitations

- NLI confidence is not a probability of truth. Scores require calibration.
- Biblical metaphor or analogy is not historical evidence for modern events.
- Different traditions disagree on interpretations; "neutral" can be appropriate.
- Current windows are limited to 5 verses within a single chapter. Longer
  pericopes and cross-chapter spans can be retrieved in a later stage.
- Corpora and models must be versioned for comparable experiments.
