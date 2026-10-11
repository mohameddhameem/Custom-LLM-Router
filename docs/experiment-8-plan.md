# Experiment 8 (plan): cheap retrieval + System One routing on the AgentTTS setting

**Status:** phase 0 done on 2026-10-11 (CPU, local). Results are below: **go** for phase 1. Phases
1–2 run on the omega GPU, the default for anything heavy. It extends the central research question ([system-one-router.md](system-one-router.md))
to a published benchmark, and joins three threads:

- our System One judge (experiment 7);
- AgentTTS (Wang et al., NeurIPS 2025, arXiv:2508.00890), a test-time compute allocator whose
  HotpotQA result is 0.74;
- the team's fixed-policy "ruler" on AgentTTS's models (`reference/Lenient Accuracy and Cost
  Comparison_V1.pdf`).

**Goal:** match or beat AgentTTS's HotpotQA accuracy (74.6 lenient, as re-scored by the team) on
their test questions, at a small fraction of its compute.

## Idea

In AgentTTS's two-stage setting, an LLM retriever selects passages from 100 candidates and an LLM
answers from them. Retrieval dominates the cost: at the paper's per-pass prices (its Table 5, as
verified by the team), one 72B retrieval pass costs **814** of the **929** units of "strong
everything" (72B retrieval + 70B answering). The unit is one 3B pass on the cheapest task.

Our retriever, nano-jev (33M), ranks 100 candidates for roughly **1 unit**. That is an estimate,
about 18k tokens × 2 × 33.4M parameters against ~1,150 GFLOPs per unit, and phase 0 measures it. If
nano-jev's ranking holds up on AgentTTS's pools, retrieval becomes nearly free. The System One judge
can then spend the budget per question where it matters: on answering.

## Research question and hypotheses

> **RQ8.** Can nano-jev retrieval plus a System One judge reach AgentTTS's accuracy on their own
> HotpotQA test questions at ≤ 25% of the strong-everything cost?

| Hypothesis | Measured by | Phase |
|---|---|---|
| **H1.** nano-jev's top-k contains both gold passages about as often as an LLM retriever: within 5 points of the 7B LLM retriever's recall | Recall@2/3/5 of the gold pair on the 100-candidate pools. BM25 order as the floor; the team's LLM retrieval outputs, once checked in | 0 |
| **H2.** Llama-3.1-70B reading nano-jev's top-k is within 2 points of 72B retrieval + 70B answering (72.2 lenient), at ~12% of its cost | Lenient accuracy and cost of the 70B on nano-jev evidence | 1 |
| **H3.** A judge choosing per question among answer options reaches ≥ 74.6 lenient at ≤ 25% of the strong-everything cost. Options: Llama-3.2-3B ×1, 3B ×10 + fusion, 70B ×1 | AIQ and the operating point against the fixed-policy hull and the AgentTTS recipe | 1–2 |

## Design

**Questions.**

| Split | Questions | Use |
|---|---|---|
| Test | HotpotQA dev (distractor) items **500–999** in file order: AgentTTS's test questions, as its `retrieval_QA_data_generator.py` selects them (`start = sample_number`). Pending the authors' confirmation | Final numbers |
| Judge training | 3,000 of our `router_train` questions | Fit the judge |
| Calibration | 1,000 of our `calib` questions | Temperature and operating point |

The training and calibration questions come from HotpotQA train: hard only, nano-jev's training
questions excluded, as in experiments 1–7. Test questions nano-jev saw in its own validation are
reported separately, as the clean subset.

**Candidate pools,** rebuilt as in AgentTTS:

1. The gold paragraphs (supporting facts), deduplicated by text.
2. Then the BM25 top 100 for the question.
3. Then a shuffle, with a **fixed seed** (AgentTTS's had none).

The BM25 corpus is the paragraphs of the question's own split: dev for test, train for training and
calibration. AgentTTS used Elasticsearch `multi_match` over title and text, with default BM25
(k1 = 1.2, b = 0.75) and the standard analyzer. We use the same formula over `title + text` with
lowercase word tokens. That is an approximation, since gold is always forced in and only the
distractors can differ.

**Retrieval options.**

- nano-jev relevance ranking, then top-k for k = 2, 3, 5;
- BM25 order, the floor;
- for comparison, the team's Qwen2.5-7B / 72B (AWQ) LLM retrieval, if its outputs can be reused.

**Answer options.**

- **Llama-3.2-3B-Instruct**, K = 1 (deterministic).
- **3B with K = 10 samples at temperature 0.9 + one fusion call by the same model:** AgentTTS's
  repeated sampling with fusion.
- **Llama-3.1-70B-Instruct (AWQ)**, K = 1.
- **Optional:** Qwen2.5 1.5B / 7B / 14B, for continuity with experiments 1–7.

**Policies compared.**

- the team's fixed ladder: every always-X policy and their upper hull, including random mixtures;
- the AgentTTS recipe (72B retrieval + 40 sampled 3B answers + fusion; 74.6 lenient, re-scored);
- the oracle;
- the small-model entropy threshold;
- logistic regression routers;
- **the System One judge**, a multi-option typed question over the answer options above.

**Metrics.**

- **Accuracy:** lenient accuracy as the primary metric, using the team's scorer, imported as shared
  code. Also strict EM and F1 (official normalisation).
- **Cost:** two scales. AgentTTS's per-pass prices (7B retrieval 77, 72B 814, 3B answer 3, 70B 115)
  for comparability, and measured token-based GFLOPs (2 × params × tokens). A fusion call is costed
  at its actual prompt length, because it reads every sampled answer.
- **Comparison:** AIQ against the fixed-policy hull, the operating point chosen on calib, and a
  paired bootstrap (1,000 resamples).

## Phases

| Phase | Where | What | Output | Effort |
|---|---|---|---|---|
| **0** | CPU, local (under a memory cap) | Build the pools for test and judge-training questions. nano-jev relevance on the 500 test pools. Recall@k (H1) and the measured nano-jev retrieval cost | `runs/exp8/pools/`, `runs/exp8/recall.json` | ~1 day; nano-jev scoring ~2 h on CPU |
| 1 | **GPU (omega)** | nano-jev scoring of the 4,000 training/calibration pools. Answer runs on nano-jev top-k evidence: 3B (K = 1; K = 10 + fusion), 70B-AWQ (K = 1). Questions: 500 test + 4,000 train/calib | Per-question caches as in `results/`, with lenient correctness and token counts | ~3–6 GPU-hours; the 70B may need 2 GPUs |
| 2 | **GPU (omega)**, short | Train multi-option System One judges (inputs qp / qpa) on the training caches | Judge scores on calib/test | < 1 h |
| 3 | CPU | Ladder, hull, AIQ, bootstrap, transfer; write-up and the final-report section | `docs/experiments/…-experiment-8.md` | ~1 day |

**Go/no-go after phase 0:** if nano-jev's recall@5 is more than 10 points below the 7B LLM retriever,
phase 1 switches to a two-step retrieval instead: nano-jev shortlists about 20 passages, and the 7B
LLM retriever picks from that shortlist.

## Phase 0 results (2026-10-11)

**Pools.**

| Pools | Questions | Corpus | Mean candidates | Gold per pool |
|---|---|---|---|---|
| test | 500 (dev items 500–999) | 66,635 distinct dev paragraphs | 100.1 | 2 in every pool |
| router_train | 3,000 | train paragraphs | 100.3 | 2 in every pool |
| calib | 1,000 | train paragraphs | 100.3 | 2 in every pool |

- **Seeded and rebuildable:** the pools are built with seed 0 and rebuild identically with
  `pools build`. At 132 MB they are not tracked.
- **Contamination:** 62 of the 500 test questions were in nano-jev's own validation set. The clean
  subset has 438.

**nano-jev ranking of the 500 test pools (CPU)** (`results/exp8-phase0/recall-test.json`).

Each cell shows the share of questions whose top k holds **both** gold paragraphs, then the mean
share of gold found:

| Ranker | top 2 | top 3 | top 5 | top 10 | top 20 |
|---|---|---|---|---|---|
| **nano-jev** | **0.544** / 0.740 | 0.652 / 0.806 | **0.772** / 0.874 | 0.860 / 0.927 | 0.926 / 0.963 |
| BM25 order (floor) | 0.244 / 0.570 | 0.338 / 0.631 | 0.460 / 0.707 | 0.602 / 0.795 | 0.734 / 0.863 |

- **By question type:** comparison questions are nearly solved (both gold in the top 2 for 0.948,
  in the top 5 for 1.000). Bridge questions are the weak spot (0.447 in the top 2, 0.717 in the
  top 5). Their second-hop paragraph often shares few words with the question.
- **Clean subset (438):** within 0.01 of the full set at every k.

**Against AgentTTS's LLM retrievers.** AgentTTS reports retrieval F1 (Ret_F1: precision/recall F1
of the selected set) for HotpotQA. Read off its Figure 10a, so approximate, and measured on its own
unseeded pools:

| Retriever | Ret_F1 | Cost (AgentTTS units per question) |
|---|---|---|
| Qwen2.5-7B | ~0.42 | 77 |
| Qwen2.5-32B | ~0.70 | |
| Qwen2.5-72B | ~0.89 | 814 |
| **nano-jev top 2** | **0.740** | **2.9** |

nano-jev's top 2 has Ret_F1 = 0.740, because with exactly 2 picked, precision = recall = the gold
share.

**Measured nano-jev cost.**

- **Tokens:** 49,649 per question. Relevance scores 3 options per candidate, so ~300 sequences.
  That is above the plan's estimate of ~1 unit.
- **Compute:** 3,317 GFLOPs = **2.9 AgentTTS units** per question, which is 283× cheaper than the
  72B retrieval pass and 27× cheaper than the 7B.
- **Time:** 12.4 s per question on a 20-thread CPU; minutes on the L40S.
- **Truncation:** 0.5% of pairs exceed 512 tokens.

**H1 verdict: supported, and stronger than hypothesised.** nano-jev does not just come within 5
points of the 7B LLM retriever. Its top-2 retrieval F1 (0.74) is far above the 7B's (~0.42) and
slightly above the 32B's (~0.70), at a fraction of their cost. Only the 72B (~0.89) is better, at
283× the cost. **Go for phase 1** with nano-jev top-k retrieval. The two-step fallback is not
needed.

**Implications for phase 1.**

- **Context size:** give the answerer the top 3 or top 5 rather than the top 2. The top 5 holds both
  gold paragraphs for 77%, and for 72% of bridge questions.
- **Cost:** nano-jev (2.9) + a 70B answer is about 13% of strong-everything (929).
- **Optional variant: hop-2 re-scoring.** Re-rank with the question + the top-1 passage, to recover
  bridge questions. It costs one more nano-jev pass, minutes on the GPU.

## Dependencies

- **Team (shared code):**
  - the lenient scorer;
  - the per-cell outputs and token counts (`meta_*.jsonl`);
  - the exact question slices (the "first-500" slice and the 1,481-question test split);
  - how the fusion call is costed.
- **AgentTTS authors:** `evaluation_2wiki.py` (their EM definition) and `tts_test_2_samples500.json`
  (to confirm the test questions). The questions are listed in the email draft.

Neither blocks phase 0.

## Risks

- **Pool difficulty:** nano-jev was trained on sets of 2 gold + 8 distractor paragraphs, while 100
  BM25 candidates are harder. Phase 0 measures this directly.
- **Model size:** the 70B-AWQ on one L40S (46 GB) is tight. Two GPUs, the per-user limit, need a
  multi-device model load (a small change to `HFExpert`), or vLLM.
- **Statistical power:** 500 test questions give ±3.8 points of sampling noise at 74%. A clear
  accuracy win needs about 78.5. Equal accuracy at a much lower cost is the stronger and more likely
  claim.
- **Comparability:** AgentTTS's EM definition is unpublished. Lenient scoring follows the team's
  re-scoring (0.74 → 74.6), and both lenient and strict numbers are reported.
