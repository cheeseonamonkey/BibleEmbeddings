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
