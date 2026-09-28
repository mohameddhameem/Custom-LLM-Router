# Reference projects

Two existing codebases that this project borrows from. Neither is vendored here.

## nano-jev

A 33M-parameter cross-encoder (`microsoft/MiniLM-L12-H384-uncased`, MIT licence) that returns
a probability distribution over a fixed set of options. It does not generate text.

| Decision | Question template | Options |
|---|---|---|
| `relevance` | "How relevant is this passage to the query: {query}" | irrelevant / partially relevant / directly answers |
| `sufficient` | "Does the context contain enough information to answer: {query}" | yes / no |
| `grounded` | "Is this claim supported by the context? Claim: {claim}" | yes / no |

### Scoring

Each option is scored as its own pair, `"question: {question} option: {opt}"` + `{context}`,
truncated to 512 tokens, giving one logit per option. The logits for one decision are grouped,
divided by a per-decision temperature `T`, and softmaxed. Option order does not matter, and a new
option set needs no new output head.

### Training data (`nanojev.data.build_all()`)

| Source | Decisions | Default size |
|---|---|---|
| HotpotQA distractor | `relevance`, `sufficient` | 6,000 train questions → ~24,000 examples |
| SQuAD 2.0 | `sufficient` | 6,000 |
| MultiNLI | `grounded` | 8,000 |

Each JSONL row: `{decision, question, options, state, label, source}`.

HotpotQA examples per question:

- **Relevance (4):** each gold paragraph is `directly answers` (2) if it contains the answer
  string, otherwise `partially relevant` (1); two random distractors are `irrelevant` (0).
  Yes/no answers never count as "contained".
- **Sufficiency (2):** both gold + 1 distractor → `yes`; 1 gold + 2 distractors → `no`. Both sets
  have three passages, so passage count is not a shortcut.

Splits: HotpotQA validation[0:500] → `calib.jsonl`, validation[500:1000] → `test.jsonl`, split
by question. MuSiQue dev and VitaminC test are held out entirely.

### Training and calibration

AdamW, lr 5e-5 (script default 3e-5), weight decay 0.01, batch 16, 2–3 epochs keeping the best
dev NLL, 6% warmup then linear decay, bf16, grad-clip 1.0. About 13.5 min/epoch on an RTX 3060.
Loss is cross-entropy over the grouped logits.

After training, one temperature per decision is fitted on `calib.jsonl` with L-BFGS, then
accuracy, macro F1, NLL, Brier and ECE are reported on `test.jsonl`. v1.0 temperatures:
relevance 1.295, sufficient 1.347, grounded 1.381 (raw model slightly overconfident).

Latency: ~1–4 ms per decision on GPU, ~17 ms on CPU.

### Implications for this project

- nano-jev has seen 6,000 HotpotQA **train** questions and HotpotQA validation[0:1000]. Router
  features computed with it on those questions will be better than on unseen ones.
- Its sufficiency head was trained on 3-passage sets. Ten distractor-setting paragraphs are
  ~1–1.5k tokens and will be truncated at 512.

## rizzo-flow

LoRA fine-tuning of a causal model ("Spark") that supervises only the logits of the allowed
answer tokens at the final prompt position, instead of free-form generation.

- The exact serving prompt is compiled and tokenized for training; there is no separate
  training-only prompt format.
- The target can be a distribution: soft cross-entropy over the allowed answer tokens.
- LoRA on selected attention/MLP projections keeps trainable parameters small.
- Only the answer-token rows of the vocabulary projection are computed.
- Evaluation text is excluded from training data, and overlong examples are dropped rather than
  truncated.

This is the same pattern as RouteLLM's causal-LLM router: give each expert a label token and
train on soft per-expert success targets.
