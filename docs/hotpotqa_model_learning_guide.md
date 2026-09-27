# HotpotQA model learning guide

This document explains what is already in this repository, what can be learned from the two
reference projects, and how to select a new model for HotpotQA through controlled experiments.
It is a learning and implementation guide, not a claim that the final model has already been
trained.

## 1. The project today

The repository is a small Python project (`pyproject.toml`) with `datasets`, `pandas`, and
`pyarrow`. The current preparation entry point is
`src/groupproject/prepare_hotpotqa.py`. It downloads the `hotpotqa/hotpot_qa` dataset using the
`distractor` configuration and writes one JSON Lines file and one Parquet file per split under
`data/hotpotqa/`.

Each raw example has ten context paragraphs: two paragraphs containing annotated supporting
facts and distractor paragraphs. The preparation script currently flattens each question into:

| Field | Meaning |
|---|---|
| `id` | Stable HotpotQA example identifier |
| `question` | The multi-hop question |
| `answer` | Gold short answer |
| `type` | `bridge` or `comparison` |
| `level` | HotpotQA difficulty level |
| `supporting_context` | Supporting sentences concatenated together |
| `full_context` | All paragraphs, with titles as headers |
| `num_context_paragraphs` | Number of supplied paragraphs |
| `num_supporting_facts` | Number of supporting-fact annotations |

The prepared files are useful for analysis and prototyping, but they are not yet training
examples for a neural model. A model-training layer still needs to define tokenization,
targets, question-level splits, batching, checkpoints, and evaluation.

### First inspection commands

From the repository root:

```powershell
uv sync
uv run python -m groupproject.prepare_hotpotqa
uv run python -c "import pandas as pd; print(pd.read_parquet('data/hotpotqa/distractor_train.parquet').head())"
```

Do not shuffle individual derived rows after one HotpotQA question has produced several rows.
All rows derived from one question must remain in the same split.

## 2. What the reference projects teach us

### 2.1 Nano-Jev: small typed decisions

Nano-Jev is a 33M-parameter MiniLM cross-encoder. It is not a text generator. It scores each
`(question, option, context)` pair independently, groups the logits for one decision, and
applies softmax. Its useful design choices are:

1. **Turn dataset structure into explicit decisions.** HotpotQA creates relevance examples
   (irrelevant, partially relevant, directly answers) and sufficiency examples (enough versus
   not enough evidence).
2. **Keep option scoring independent.** The same model can score a new option set without
   changing its output head.
3. **Construct hard negatives carefully.** Positive and negative sufficiency examples contain
   the same number of paragraphs, so paragraph count cannot become a shortcut.
4. **Calibrate after training.** Temperature scaling is fitted on a calibration split and is
   not mixed with the final test measurement.
5. **Evaluate out of distribution.** Nano-Jev reports both in-domain results and held-out
   datasets, exposing the accuracy drop on unseen reasoning and writing styles.

Nano-Jev is the best first baseline when the project goal is fast evidence selection or a
reliable `sufficient`/`relevant` signal. It is not, by itself, an answer generator or a full
multi-hop reader.

### 2.2 Rizzo-Flow: prompt-aligned answer-token fine-tuning

Rizzo-Flow uses a causal Spark model, but does not train free-form answer generation. It compiles
the exact prompt used by serving, runs one forward pass, and supervises only the logits of
allowed answer letters at the final prompt position. Its reusable lessons are:

1. **Training and serving must share the exact input contract.** Prompt compilation and
   tokenization are reused rather than approximated in a separate training script.
2. **A distribution can be the target.** Soft cross-entropy supports probabilistic labels and
   avoids pretending that uncertain judgments are perfectly certain.
3. **LoRA makes a large base model practical.** Only selected attention/MLP projections are
   updated, reducing trainable parameters and storage.
4. **Do not materialize unnecessary vocabulary logits.** Only answer-token rows are needed for
   the decision objective.
5. **Guard against contamination.** Evaluation text is explicitly excluded from training data,
   and long examples are dropped rather than silently truncated.

This approach is attractive when the final interface should answer several typed questions from
one state. It is more expensive and operationally complex than Nano-Jev, and it does not
automatically solve HotpotQA's span extraction problem.

### 2.3 Comparison

| Property | Nano-Jev-style cross-encoder | Rizzo-Flow-style causal model | Hybrid reader + selector |
|---|---|---|---|
| Primary output | Class probabilities | Typed answer-token probabilities | Evidence ranking plus answer span |
| Best use | Relevance/sufficiency gate | Multiple decisions over structured state | End-to-end HotpotQA answering |
| Model size/cost | Small and fast | Larger; LoRA reduces training cost | Configurable; usually two smaller heads |
| Native answer text | No | Only if a generator is added | Yes, via extractive span |
| Multi-hop evidence | Indirectly learned | Prompt-dependent | Explicitly modeled |
| First experiment value | Very high | High if a causal base is available | Highest task alignment |
| Main risk | Labels are proxies | Hallucination and prompt sensitivity | More components and more engineering |

The project should compare all three rather than call one “novel” in advance. Novelty should come
from the final architecture, evidence supervision, or training objective demonstrated by an
ablation—not from renaming a baseline.

## 3. Formulating HotpotQA as learning tasks

HotpotQA supplies several complementary targets:

### Task A: paragraph relevance

Input: question and one paragraph. Target:

* `0` — distractor/irrelevant;
* `1` — supporting but not directly containing the answer;
* `2` — supporting and containing the answer.

The answer-string heuristic must be treated as weak supervision: aliases, yes/no answers,
coreference, and paraphrases can make the label imperfect. Keep the original supporting-fact
annotation so this heuristic can be audited.

### Task B: evidence sufficiency

Input: question and a set of paragraphs. Target: enough evidence or insufficient evidence.
Construct positive examples from the gold supporting paragraphs and hard negatives by replacing
one gold paragraph with a distractor. Keep the number and approximate length of paragraphs
balanced.

### Task C: answer extraction

Input: question and all or selected paragraphs. Target: the character/token span of `answer`
in the evidence, when an exact span exists. For yes/no answers, use a small classification
head. When the answer does not occur verbatim, record the example separately rather than
silently creating an invalid span.

### Task D: supporting-fact prediction

Input: question and each paragraph/sentence. Target: whether the paragraph or sentence is part
of the annotated reasoning chain. This is the most direct way to make multi-hop behavior
visible to the model and is the key additional supervision for the hybrid candidate.

### Recommended first model objective

Use a multi-task objective:

```text
loss = answer_loss
     + 0.5 * supporting_fact_loss
     + 0.25 * paragraph_relevance_loss
     + 0.25 * sufficiency_loss
```

The coefficients are starting points, not final claims. Tune them only on a development split.
The answer head should be extractive; the model can select evidence before predicting a span.

## 4. Three baseline experiments

Use the same question-level train/dev/test split for every model. Keep the official validation
set for final reporting or split it once into calibration and test; never tune repeatedly on the
final test portion.

### Baseline 1: Nano-Jev-style cross-encoder

Start with `microsoft/MiniLM-L12-H384-uncased` or another documented encoder. Create JSONL
examples with `question`, `options`, `state`, `label`, and `source`, following the reference
format. Train separate grouped logits for relevance and sufficiency. Fit temperature scaling on
calibration data.

This establishes a fast evidence gate and provides a meaningful baseline for the hybrid model.
It does not need to predict the final answer.

### Baseline 2: Rizzo-Flow-style causal readout

Use a small causal model available in the environment. Serialize the question and context into a
stable prompt and map candidate answers to a fixed set of answer tokens or labels. Train with
LoRA and soft cross-entropy over the allowed labels. Measure both answer accuracy and whether
the model's selected evidence agrees with supporting-fact labels.

Do not compare its free-form text quality against an extractive model without separating answer
selection from generation. If the model generates text, add exact-match and normalization rules
before interpreting results.

### Baseline 3: hybrid multi-hop reader

Build the proposed task-specific model in stages:

1. encode the question and ten paragraph candidates;
2. score paragraphs and sentences with a relevance head;
3. aggregate selected evidence with attention or a lightweight cross-attention block;
4. predict an answer span from the selected evidence;
5. optionally predict `yes`/`no` for non-span answers.

Train first with gold supporting-fact supervision, then repeat with predicted evidence. The
difference measures how much performance depends on oracle evidence.

## 5. Evaluation protocol

Report metrics by question type (`bridge`, `comparison`) and difficulty (`easy`, `medium`,
`hard`) as well as overall:

| Area | Metrics |
|---|---|
| Answering | Exact Match, token-level F1 |
| Evidence | paragraph/sentence precision, recall, F1 |
| Retrieval quality | Recall@1, Recall@2, Recall@k, MRR |
| Sufficiency | accuracy, macro F1, NLL, Brier score, ECE |
| Reliability | calibration curve and confidence-stratified accuracy |
| Efficiency | parameter count, peak memory, examples/second, latency |

Use the official HotpotQA evaluation script where possible. Normalize answers consistently
(lowercase, punctuation/articles/whitespace rules) and keep the raw prediction beside the score.
For evidence, report both paragraph-level and sentence-level results; paragraph-only scores can
hide incorrect supporting sentences.

Required comparisons:

* random or lexical paragraph ranking;
* a pretrained encoder without fine-tuning;
* the three trained candidates;
* hybrid model with oracle evidence versus predicted evidence;
* ablations removing relevance, sufficiency, and supporting-fact losses.

For every result, record seed, base checkpoint revision, tokenizer, maximum sequence length,
training examples, optimizer, learning rate, batch/accumulation settings, and hardware.

## 6. Data hygiene and reproducibility

1. Split by `id` before generating derived relevance/sufficiency/span rows.
2. Preserve the raw question and annotations beside every derived example.
3. Never use the test answers, supporting facts, or evaluation text to select hyperparameters.
4. Keep a manifest with source revision, counts, label distribution, rejected examples, and seed.
5. Fail loudly on missing spans, malformed context, or overlong records; do not silently truncate
   evidence.
6. Save model configuration, tokenizer, calibration parameters, and the exact evaluation
   command with each run.
7. Use Parquet for analysis and JSONL for streaming training examples.

The existing `prepare_hotpotqa.py` is a good raw-data export step. The next data script should
consume those exports (or the original records with annotations retained) and create
question-grouped training examples. It should not infer supporting facts from the flattened
`supporting_context` string alone.

## 7. Suggested implementation order

1. Add a statistics/audit script: counts by split, type, level, paragraph count, answer type,
   supporting-fact count, and exact-answer coverage.
2. Add a deterministic question-level split manifest.
3. Implement Nano-Jev-style relevance and sufficiency examples and train the fast baseline.
4. Implement the causal answer-token baseline only if a suitable local checkpoint and GPU budget
   are available.
5. Implement the hybrid reader with paragraph/sentence supervision and extractive answers.
6. Run the common evaluation protocol and ablations.
7. Select the final model using answer F1/EM, evidence F1, calibration, and latency together.
8. Package the chosen model with a small inference API and a model card describing limitations.

### Definition of a successful “novel model”

The final model should demonstrate at least one measurable contribution beyond copying a
reference pipeline, for example:

* explicit sentence-level multi-hop evidence aggregation;
* a shared encoder with calibrated relevance, sufficiency, and span heads;
* a training objective that improves evidence quality without reducing answer F1;
* an efficient cascade that matches a larger reader while using less compute.

The contribution must be supported by ablations and an unchanged held-out test set.

## 8. Immediate next command set

After the learning document is reviewed, the first implementation milestone should be:

```powershell
uv run python -c "import pandas as pd; df = pd.read_parquet('data/hotpotqa/distractor_train.parquet'); print(df[['type','level','num_context_paragraphs','num_supporting_facts']].describe(include='all'))"
```

Then add the audit and split-manifest scripts before training. This keeps the project focused on
the actual HotpotQA distribution and prevents a model from appearing strong because of leakage,
answer-string shortcuts, or inconsistent preprocessing.
