# Exhibit A: The Burning Bush
### A 149M-Parameter Experiment in Biblical Evidence Retrieval and Confident Nonsense

**Verse Lawyer / BibleEmbeddings**
Research note · 3 November 2026 · Reproducible experimental prototype
[Source and experiments](https://github.com/cheeseonamonkey/BibleEmbeddings) · [Model and training notebook](https://www.kaggle.com/code/ffatty/verse-lawyer-domain-v1-149m)

## Abstract

Can a small language model find Bible passages that argue *for* and *against* an arbitrary proposition, rather than simply retrieving verses that contain similar words? We built Verse Lawyer around a single 149,607,171-parameter ModernBERT encoder. It produces retrieval embeddings and classifies passage–claim pairs as entailment, contradiction, or neutral. The corpus consists of 31,102 King James Version verses indexed as 116,091 overlapping one-, two-, three-, and five-verse passages. A simple BM25 candidate retriever supplies passages for the public experiment.

After general and Bible-specific fine-tuning, the model classified 90 of 96 authored, previously unused Bible proposition–passage pairs correctly (93.75%), compared with 82 of 96 (85.42%) before Bible adaptation. Neutral-case F1 improved from 83.9% to 98.4%. But in unrestricted retrieval, the system gave an apparent supporting citation from *Exodus 3:2–4*, describing a **burning bush**, for the proposition **"Bush did 9/11."** Other unsupported propositions received similarly confident-looking citations. The distinction is the result: a model can be fairly good at judging supplied evidence and still be quite bad at deciding what deserves to be called evidence.

**Keywords:** retrieval-augmented inference, natural-language inference, evidence attribution, prooftexting, biblical corpus, adversarial evaluation, ModernBERT.

## 1. Motivation

In 2023, an earlier BibleEmbeddings project used sentence embeddings to search scripture; one diagnostic query was "I pooped my pants." By 2026, an updated version could produce competing scriptural arguments about almost anything. The ambition was deliberately modest: not to resolve theology, historical allegations, or factual disputes, but to see whether a compact neural system could distinguish genuine textual support from a convenient quotation.

Scripture is a useful stress test. It is rich in metaphor, quotation, contradiction in apparent surface form, narrative reporting, and passages whose meaning depends on adjacent verses. *Proverbs 26:4–5* is a particularly tidy example: one verse says not to answer a fool according to his folly; the next says to answer him. A one-sentence hypothesis may be supported by one verse and opposed by the next, while a combined window requires contextual interpretation.

The experiment retained an important boundary: **a biblical passage can support a reading of scripture without establishing a modern historical fact.** Verse Lawyer's outputs are model judgments about textual relationships, not evidence for real-world conspiracies.

## 2. System and corpus

The underlying corpus is a structured King James Version JSON dataset: **66 books and 31,102 verses**. Each verse retains a canonical reference. We construct contiguous passage windows of lengths 1, 2, 3, and 5 verses, sliding by one verse within each chapter. Windows do not cross chapter boundaries. This produces **116,091 candidate passages**, including overlapping formulations of the same local context.

Retrieval begins with **BM25**, a lexical ranker. It finds candidates; it does not decide truth or support. The same jointly trained ModernBERT encoder can also produce mean-pooled, normalized embeddings for optional dense retrieval. **No second neural encoder** is used. The public experiment deliberately uses the lexical path so that the observed retrieval failures remain inspectable and reproducible.

A single **ModernBERT-base** checkpoint (149,607,171 parameters) handles NLI classification. For each candidate passage, we compute predictions twice: first from the isolated passage, then from the passage expanded by up to two adjacent verses on each side. Each pass emits model softmax scores for *entailment*, *neutral*, and *contradiction*. The demonstrated interface ranks up to three passages per side using contextual scores with a permissive threshold of 0.40.

This approach is architecturally simple: **one backbone, one classification head, one corpus, one candidate retriever**. It also exposes two distinct failure points: what is retrieved, and what the model infers from it.

## 3. Fine-tuning

The first run trained ModernBERT on **6,000 MultiNLI** examples, with **1,000 held-out validation** pairs. Training combined three-way NLI cross-entropy with a contrastive retrieval objective derived from entailment pairs. This yielded 76.9% validation accuracy and 76.85% macro-F1. On a 128-pair entailment retrieval probe, Recall@1 increased from **71.1% before fine-tuning to 87.5% afterward**, and Recall@10 from 80.5% to 96.1%.

The second run continued the **same** checkpoint. We prepared **363 provisional Bible-specific pairs** spanning **109 references**, including literal support, explicit opposition, topical non sequiturs, longer passages, and deliberately absurd modern claims. Their labels were authored for this experiment, not independently adjudicated. The continuation mixed the domain pairs (repeated four times) with **2,200 additional general MultiNLI examples**, using a lower learning rate and a smaller retrieval-loss weight. Both supporting and contradicting passages counted as relevant retrieval targets; only the classification head assigned the stance.

The continuation ran for two epochs. We selected **epoch 1** using a combined score on the existing 30-item Bible development set and general NLI validation; epoch 2 scored lower on the Bible development set. The resulting checkpoint remains 149.6M parameters. Training and evaluation were conducted on Kaggle GPU; CPU inference uses the same saved weights.

## 4. Evaluation: supplied passages

**Table 1. Passage-level classification (provisional authored labels).**

| Metric | General-NLI model | Bible-tuned model |
| --- | ---: | ---: |
| General MultiNLI validation accuracy (1,000) | 76.9% | 79.5% |
| Bible development accuracy (30) | 86.7% | 93.3% |
| Fresh Bible holdout accuracy (96) | 85.4% | **93.8%** |
| Fresh Bible holdout macro-F1 | 85.4% | **93.8%** |
| Fresh holdout entailment F1 | 86.7% | 92.3% |
| Fresh holdout neutral F1 | 83.9% | **98.4%** |
| Fresh holdout contradiction F1 | 85.7% | 90.6% |

The 96-item set covers **32 previously unused passage groups**, each paired with a supporting, opposing, and neutral claim. These passages were excluded from the training notebook and from the inspected 30-item development set; nearby verses were also screened for overlap. The domain-tuned model corrected **nine** cases the baseline missed and regressed on **one** formerly correct case.

These are encouraging but limited measurements. The labels were not independently verified, the examples are deliberately structured, and several claims associated with one passage are correlated. The original 30-case development set had already been inspected before the second run. Consequently, **93.8% is not a claim about real-world theological reasoning or full-system accuracy**.

An additional retrieval probe indicated nonuniform changes: contradiction Recall@1 on a 128-pair general-language test **fell from 71.1% to 69.5%** following domain adaptation, even as contradiction Recall@10 rose from 90.6% to 93.0%. No single score establishes overall retrieval quality.

## 5. Evaluation: a full Bible search

We next ran **24 direct claim–passage challenges** and **13 end-to-end searches**. Direct classification matched the experiment's provisional labels on **21/24** cases (87.5%). The unrestricted search task was more revealing: for each user claim, BM25 searched all 116,091 overlapping passages, retained **14 candidates**, and the model assigned contextual NLI scores.

**Table 2. Representative retrieved arguments from the full pipeline.** These are *incorrect or unsupported model outputs*, not factual claims.

| User proposition | Returned passage | Apparent model stance |
| --- | --- | ---: |
| "Bush did 9/11." | Exodus 3:2–4 (burning bush) | 42% FOR |
| "The Bible predicted TikTok." | Ezra 4:9 | 59% FOR |
| "God prefers cats over dogs." | Psalm 22:16 (dogs) | 59% AGAINST |
| "Money answers all things." | Ecclesiastes 10:19 | 51% AGAINST |
| "You should always love your enemies." | Matthew 5:43–47 | 84% AGAINST |
| "It is better to give than receive." | Genesis 29:19 | 96% FOR |
| "There is a time to dance." | Ecclesiastes 3:3–5 | 64% AGAINST |

The **burning bush** example is a straightforward lexical collision: a plant and a person's surname share a word. The citation is not evidence for the modern allegation. *Genesis 29:19* is subtler: Laban says it is better to give his daughter to Jacob than to another man. The model rates this a strong argument for a general moral claim about giving. The surface grammar is similar, but the semantic scope is wrong.

The system returned at least one FOR or AGAINST citation on **all 13 searches**, including intentionally ungrounded claims. This is not a success rate: it indicates that the present selection threshold and ranking do not reliably admit **no evidence** as an answer.

## 6. Failure analysis

**6.1. Relevant words are not relevant evidence.** BM25 can locate a word like *bush*, *give*, or *dogs*, but cannot determine the intended referent or entailment. If NLI is only applied to the first 14 lexical matches, a good supporting passage may never be considered. The search results prove the distinction between *finding a phrase* and *finding an argument*.

**6.2. More context can reverse the judgment.** *Proverbs 26:5* scored **99.9% entailment** for "Answer a fool according to his folly" in isolation, then **75.7% contradiction** when expanded context included the preceding contrary instruction. Similarly, the isolated *Ecclesiastes 10:19* passage scored 97.1% support for "Money answers all things," while the expanded context scored 51.5% contradiction. The current pipeline ranks by contextual score alone, sometimes suppressing a clearer isolated passage.

**6.3. Confidence is not calibration.** A softmax score is a model preference among labels for the supplied text pair. It does not measure the truth of the user's proposition, the theological legitimacy of an interpretation, or a statistically calibrated probability that the passage genuinely supports the claim. Raising a 40% threshold cannot remove 96% wrong answers.

**6.4. Some examples are genuinely ambiguous.** Narrative description is not prescription; historical events cannot automatically be generalized into contemporary law. Apparent tensions such as *Proverbs 26:4–5* admit more than one good reading. A useful benchmark needs multiple annotators, permitted ambiguity, and a category for legitimate scope disagreement.

## 7. What should be measured next?

A better evaluation would require human-labeled **retrieval gold passages and no-evidence queries**, separated by book, passage overlap, and proposition family. It should report retrieval Recall@k, right-versus-wrong side placement, false citation rate, abstention, and evidence calibration separately from NLI macro-F1. It should also compare contextual scoring policies: isolated-only, expanded-only, and abstention on disagreement.

A larger candidate pool and the **same encoder's dense retrieval** may improve search breadth without introducing a second model. But any improvement should be judged on unseen, manually reviewed queries. The odd results should remain preserved as regression fixtures.

## 8. Conclusion

Verse Lawyer did learn something about entailment. It also learned how to make remarkably plausible-looking errors when retrieval supplied the wrong passages or contextual aggregation flipped a correct reading. The most informative output was not the clean 93.8% classification score; it was *Exodus 3:2–4* presented as evidence for the wrong kind of Bush.

The experiment is intentionally left imperfect. A small model can put on a convincing legal argument. Whether it has a case is another matter.

## References and reproducibility

1. **BibleEmbeddings / Verse Lawyer.** Source, data construction, tests, pretrained and fine-tuned experiment documentation: https://github.com/cheeseonamonkey/BibleEmbeddings
2. **Kaggle continuation.** Saved 149.6M-parameter domain-tuned checkpoint, training notebook and metrics: https://www.kaggle.com/code/ffatty/verse-lawyer-domain-v1-149m
3. **Kaggle general-NLI baseline.** https://www.kaggle.com/code/ffatty/verse-lawyer-modernbert-149m-joint
4. **ModernBERT-base.** https://huggingface.co/answerdotai/ModernBERT-base
5. **MultiNLI.** Williams, Nangia and Bowman, 2018. https://cims.nyu.edu/~sbowman/multinli/
6. **King James Version structured JSON.** https://github.com/thiagobodruk/bible
7. **Robertson et al. / BM25.** Robertson and Zaragoza (2009), *The Probabilistic Relevance Framework: BM25 and Beyond*, Foundations and Trends in Information Retrieval.

The trained weights are provided separately through Kaggle; the project repository contains reproducible training scripts, dataset manifests, evaluation reports, and the 24 + 13 adversarial probes. Model statements involving real individuals or modern events are **not historical assertions**.
