# Bible-specific NLI smoke set

`seed.jsonl` contains **30 provisional human-selected proposition/passage pairs**,
including contradictory adjacent verses, full two-verse cases, literal instructions,
modern claims outside the Bible's scope, and translation-sensitive examples.

Labels follow the *literal evidence relationship between the quoted passage and
proposition*, not doctrinal consensus or the actual truth of the proposition.

Run after downloading a trained NLI checkpoint:

```sh
python evaluation/evaluate.py --data data/en_kjv.json \
    --model out/modernbert-mnli/model
```

This dataset is far too small for substantive research and **must not be used for
fine-tuning**. Its purpose is to reveal obvious failure modes before collecting
a properly annotated, held-out evaluation set with multiple raters.

## Fresh holdout (not used for training or model selection)

`fresh_holdout.jsonl` contains **96 independently authored cases from 32
references**. No passage overlaps the 363-item domain training set or the
original 30-item development set (including adjacent +/-2 verses). It was
prepared after the domain training notebook was submitted and is **not**
embedded in that notebook.

```sh
python evaluation/evaluate.py --data data/en_kjv.json \
    --model out/verse-domain-v1/model \
    --seed evaluation/fresh_holdout.jsonl \
    --out out/holdout.json
```

The labels are provisional, not validated by outside annotators; this is a
small independent development check, not a population-wide accuracy estimate.
Do not tune the model using its error cases and then claim this is still a
sealed holdout.
