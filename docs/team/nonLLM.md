# team-member-1: `nonLLM` (snapshot `da4f08f`, 2026-10-17)

Two lines of work: (1) cheap **non-LLM routers** that pick weak or strong models per pipeline stage, and (2) a large
**answer-sampling experiment** that runs AgentTTS's HotpotQA pipeline with 1–40 answer samples per question.
"Non-LLM" refers to the routing decision model; retrieval and answering are still LLMs. Runs on the member's own GPU
servers (L40S, RTX A5000), not on OMEGA.

## 1. Non-LLM routing baselines

| | |
|---|---|
| Question | Can cheap classifiers choose, per question and per stage, between a weak (W) and strong (S) model better than a fixed policy? |
| Pipeline | Two stages, retrieval then answering, each W or S: outcomes WW / WS / SW / SS (retrieval letter first) |
| Data | The team's stored grid `router_data_lenient.jsonl.gz` (measured outcomes per cell) and a frozen split train 4,814 / validation 1,110 / test 1,481 |
| Routers | Majority; TF-IDF (word 1–2-grams) + logistic regression, linear SVM, multinomial NB; random forest on 20 numeric text features. Three classifiers per family: retrieval, QA after weak retrieval, QA after strong retrieval. Tuned on validation balanced accuracy |
| Cost | Pipeline units 2.0 / 5.7 / 5.7 / 9.4 for WW / WS / SW / SS; router CPU cost excluded |
| Metric | Lenient (gold contained) and strict accuracy |

**Result on the stored grid (test, 1,481):** fixed policies WW 31.8% / WS 43.6% / SW 41.7% / SS 53.5% lenient; the
routers reach 37.1–41.7% lenient at 35–51% of SS's cost, **+0.85 to +1.30 points** over the cost-matched mixture of
fixed policies. Stage classifiers are near chance (balanced accuracy 50–57%). Lenient oracle 61.2%.
([results/measured_v1/REPORT.md](https://github.com/mohameddhameem/STAIR/blob/da4f08f/nonLLM/results/measured_v1/REPORT.md))

**Rerun with real models** (Qwen2.5 1.5B / 7B Base, GGUF Q4_K_M, all 1,481 test questions, official EM/F1): always-SS
EM 18.2 / F1 27.1, the best routers EM 10–12, oracle EM 25.7. Low absolute scores; the member notes prompts and
quantisation differ from the stored grid.
([results/full_pipeline_test_v1/REPORT.md](https://github.com/mohameddhameem/STAIR/blob/da4f08f/nonLLM/results/full_pipeline_test_v1/REPORT.md);
a 20-question validation pilot: [live_qa_val20_v1](https://github.com/mohameddhameem/STAIR/blob/da4f08f/nonLLM/results/live_qa_val20_v1/REPORT.md))

## 2. Adaptive answer sampling (AgentTTS pipeline)

| | |
|---|---|
| Question | Per question, which W/S pair and how many answer samples k (1–40, fused) are needed? Does a per-question budget beat AgentTTS's single global choice? |
| Pipeline | AgentTTS's public HotpotQA code (pinned commit `3f0c2df`), prompts executed unchanged: retrieve → Merge → 40 QA samples → fusion of every prefix k = 1..40. Temperatures 0.9 / 0.7 / 0.3, 512 new tokens |
| Models | Qwen2.5 1.5B (W) and 7B (S); first Base GGUF, now **Instruct GGUF Q4_K_M** (llama.cpp) |
| Scale | All 7,405 questions × 4 pairs × 40 samples × 40 fusions (2.4M calls); same 65/15/20 split |
| Cost | Model-weighted token work (a proxy, not FLOPs) |
| Metric | Official HotpotQA EM/F1 after a gold-independent answer extractor ("evaluation v2": a grounded reader for replies over six words) |

**Status and findings so far:**

- **The Base-model run failed** (completed 2026-10-14): test EM 0% for every pair and budget, F1 under 1.1%; outputs
  repeated instructions or invented content. Archived as invalid, not as evidence against sampling.
  ([adaptive_sampling40_base](https://github.com/mohameddhameem/STAIR/blob/da4f08f/nonLLM/results/adaptive_sampling40_base/README.md))
- **Instruct GGUF rerun** started 2026-10-15 on one server's four L40S GPUs (one GPU had an ECC error and was repaired).
- **12-question Strong pilot** (training questions): with extraction, strict-EM hits out of 12 at k = 1 / 40 are
  WW 2 / 4, WS 5 / 3, SW 2 / 2, SS 4 / 3; more samples do not reliably help.
  ([priority_strong_12](https://github.com/mohameddhameem/STAIR/blob/da4f08f/nonLLM/results/adaptive_sampling40_instruct_gguf/priority_strong_12/README.md))
- **500-question cohort, completed 2026-10-17:** a frozen cohort of training questions for the pair-and-budget
  comparison. All 500 questions have every pair and every k = 1..40 (80,000 reader records). After it, the full
  Strong run resumed on all four GPUs.
  ([COHORT_500.md](https://github.com/mohameddhameem/STAIR/blob/da4f08f/nonLLM/experiments/adaptive_sampling/COHORT_500.md);
  results in [cohort_500/](https://github.com/mohameddhameem/STAIR/tree/da4f08f/nonLLM/results/adaptive_sampling40_instruct_gguf/cohort_500))
- **Queued:** 40 retrieval samples per model, with retrieval prefixes, after the current run.
  ([FOLLOWUP_RETRIEVAL40.md](https://github.com/mohameddhameem/STAIR/blob/da4f08f/nonLLM/experiments/adaptive_sampling/FOLLOWUP_RETRIEVAL40.md))

**500-question cohort results.** W = 1.5B and S = 7B. The first letter is the retrieval model and the second the
answer/fusion model.

| Pair | Any k correct (strict EM) | Best single k (strict hits) | Any k correct (lenient EM) | Never correct (lenient) |
|---|---|---|---|---|
| WW | 225 | k = 13: 111 | 283 | 217 |
| WS | 279 | k = 1: 145 | 342 | 158 |
| SW | 257 | k = 3: 142 | 313 | 187 |
| SS | 288 | k = 8: 168 | 346 | 154 |
| any pair | | | 414 | 86 |

([table.md](https://github.com/mohameddhameem/STAIR/blob/da4f08f/nonLLM/results/adaptive_sampling40_instruct_gguf/cohort_500/table.md),
[lenient_ranges/table.md](https://github.com/mohameddhameem/STAIR/blob/da4f08f/nonLLM/results/adaptive_sampling40_instruct_gguf/cohort_500/lenient_ranges/table.md))

- **Most successes come early.** For SS under lenient EM, 215 questions are first right at k = 1, and only 20 first
  at k > 10. Success sets are often non-contiguous: more samples can turn a right answer wrong.
- **Oracle against SS×40** (lenient EM; cost proxy = billions of parameters × tokens):

  | Policy | Correct / 500 | Cost reduction | Mean k |
  |---|---|---|---|
  | SS×40 | 189 (37.8%) | 0% | 40 |
  | oracle: smallest k, then cheapest | 414 (82.8%) | 45.6% | 9.0 |
  | oracle: cheapest, then smallest k | 414 (82.8%) | 57.8% | 11.6 |

  ([oracle_cost/comparison.md](https://github.com/mohameddhameem/STAIR/blob/da4f08f/nonLLM/results/adaptive_sampling40_instruct_gguf/cohort_500/oracle_cost/comparison.md))

  The oracle picks the pair and k after seeing the gold answer, so it is a retrospective upper bound, not a
  policy. The member labels the cohort "descriptive": it is chosen by availability from training questions, not
  held-out. New scripts: `gguf_cohort_budget_ranges.py`, `gguf_cohort_oracle_cost.py`, `plot_cohort_budget_report.py`
  (with an English PDF/PNG report).

## Cautions the member records

- All sampling numbers so far are on **training** questions; validation/test is untouched.
- Oracles use gold answers and are retrospective upper bounds, not deployable policies.
- Strict EM rejects some equivalent answers ("Australia" vs "Australian"); the extractor can abstain or err.
- Models, quantisation and serving differ from the AgentTTS paper; it is prompt-aligned, not a reproduction.

## Key files

- [nonLLM/README.md](https://github.com/mohameddhameem/STAIR/blob/da4f08f/nonLLM/README.md), `baselines.py`,
  `report_baselines.py`
- [experiments/adaptive_sampling/](https://github.com/mohameddhameem/STAIR/tree/da4f08f/nonLLM/experiments/adaptive_sampling):
  README, EXPERIMENT_PLAN, GGUF_RUN, STRONG_PILOT, COHORT_500, REPAIR
- [deployment/](https://github.com/mohameddhameem/STAIR/tree/da4f08f/nonLLM/deployment): live and full-pipeline evaluation scripts
- [results/](https://github.com/mohameddhameem/STAIR/tree/da4f08f/nonLLM/results)
