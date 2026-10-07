# Experiments

Headline numbers for every executed run, next to published HotpotQA results. Each run has its
own write-up in [experiments/](experiments/).

All runs use the **HotpotQA distractor** setting (the question plus 10 paragraphs: 2 gold, 8
distractors). Scores are **answer EM and F1** in %, computed with the official
`hotpot_evaluate_v1.py` normalisation, on the full **dev (validation) set of 7,405 questions**.
Cost is GFLOPs per question.

## Our runs

| Run | Date | Method | EM | F1 | GFLOPs | Sent to large | AIQ |
|---|---|---|---|---|---|---|---|
| [1](experiments/2026-10-05-omega-full-run.md) | 2026-10-05 | Always small: Qwen2.5-1.5B, top-2 nano-jev paragraphs | 48.3 | 61.3 | 1,051 | 0% | |
| | | Always large: Qwen2.5-7B, all 10 paragraphs | 53.0 | 66.9 | 22,762 | 100% | |
| | | Router, evidence / large-helps | 52.9 | 66.7 | 8,503 | 34% | 0.663 |
| | | Router, question+evidence / large-helps | 52.5 | 66.0 | 5,933 | 23% | 0.665 |
| | | Small-entropy threshold (baseline) | | | | | 0.662 |
| | | Random routing (baseline) | | | | | 0.641 |
| | | Oracle router (upper bound) | 63.1 | 77.2 | 5,638 | 21% | 0.755 |
| [2](experiments/2026-10-06-cascade-routers.md) | 2026-10-06 | Cascade router, evidence+small / small-fails | 54.1 | 67.6 | 12,657 | 50% | 0.666 |
| | | Cascade router, question+evidence+small / large-helps | 52.9 | 66.5 | 6,340 | 23% | **0.672** |
| [3a](experiments/2026-10-06-experiments-3-5.md) | 2026-10-06 | Always large: 7B on top-3 nano-jev paragraphs | 55.5 | 69.2 | 7,292 | 100% | |
| | | Router, question+evidence+small / large-helps | 54.8 | 68.2 | 5,007 | 54% | 0.669 |
| [3b](experiments/2026-10-06-experiments-3-5.md) | 2026-10-06 | Always large: 7B on top-5 nano-jev paragraphs | 55.8 | **70.0** | 11,598 | 100% | |
| | | Router, question+evidence+small / large-helps | 54.5 | 68.1 | 4,958 | 34% | 0.682 |
| [4](experiments/2026-10-06-experiments-3-5.md) | 2026-10-06 | Always large: Qwen2.5-14B, all 10 paragraphs | 57.1 | **71.9** | 43,906 | 100% | |
| | | Router, question+evidence+small / large-helps | 57.2 | 71.3 | 23,043 | 50% | 0.697 |
| | | Oracle router (1.5B / 14B) | 64.8 | 79.4 | 11,174 | 24% | 0.772 |
| [5](experiments/2026-10-06-experiments-3-5.md) | 2026-10-06 | Always small: Qwen2.5-3B, top-2 nano-jev paragraphs | 49.3 | 61.8 | 2,105 | 0% | |
| | | Router, evidence+small / large-helps (7B on all 10) | 53.6 | 67.1 | 6,384 | 19% | 0.681 |
| [6](experiments/2026-10-06-multi-stage-cascade.md) | 2026-10-06 | Cascade 1.5B (top 2) → 7B (top 3) → 14B (all 10) | **58.4** | **72.6** | 14,243 | 15% to 14B | 0.721 |
| | | Cascade 1.5B (top 2) → 14B (all 10), same procedure | 57.8 | 72.3 | 35,202 | 78% | 0.695 |

Both experts are instruction-tuned and not fine-tuned on HotpotQA, prompted with two
answer-format examples. Routers are logistic regression, with τ = 0.8.
Run 2 reuses run 1's expert outputs. Its cascade routers decide after the small expert has
answered, and the bootstrap shows they are the only routers significantly better than the entropy
baseline. AIQ 95% CIs are about ±0.008.
Runs 3–5 each change one expert of run 1 (rows not shown are unchanged). Reading only nano-jev's
top 3 or 5 paragraphs makes the 7B both better and cheaper. AIQ depends on each run's own expert
pair, so compare it only within a run. The per-run outputs are in [results/](../results/).
Run 6 chains cached experts from runs 1–4 on CPU. Its AIQ is over 1,051–43,906 GFLOPs (1.5B to 14B),
so it is comparable only within run 6.

**Planned:** experiment 7, a System One router: a fine-tuned nano-jev judge as the small/large router. It is
the final report's central research question ([system-one-router.md](system-one-router.md), job sheet
[gpu-jobs-system-one.md](gpu-jobs-system-one.md), report plan [final-report-plan.md](final-report-plan.md)).

## Published reference points (answer EM / F1)

| System | Type | Split | EM | F1 | Source |
|---|---|---|---|---|---|
| HotpotQA baseline (Yang et al., 2018) | Fine-tuned, trained from scratch | dev | 44.4 | 58.3 | [paper, Table 4][hotpot] |
| HotpotQA baseline (Yang et al., 2018) | Fine-tuned, trained from scratch | test | 45.5 | 59.0 | [paper, Table 4][hotpot] |
| Llama-2-13B | Few-shot chain-of-thought | dev | 30.9 | 45.8 | [Bhuiya et al., 2024, Table 4][distractors] |
| Llama-2-70B | Few-shot chain-of-thought | dev | 54.1 | 67.7 | [Bhuiya et al., 2024][distractors] |
| Mixtral-8x7B-Instruct | Few-shot chain-of-thought | dev | 50.4 | 68.1 | [Bhuiya et al., 2024][distractors] |
| GPT-3.5 | Few-shot chain-of-thought | dev | 63.4 | 77.2 | [Bhuiya et al., 2024][distractors] |
| Longformer | Fine-tuned | dev | 71.5 | 82.1 | [Bhuiya et al., 2024][distractors] |
| Beam Retrieval (2023, leaderboard #1) | Fine-tuned | test | 72.7 | 85.0 | [leaderboard][lb] |
| Human (one crowd worker) | | 1,000 dev/test samples | 83.6 | 91.4 | [paper, Table 8][hotpot] |
| Human upper bound | | 1,000 dev/test samples | 96.8 | 98.8 | [paper, Table 8][hotpot] |

## How the runs compare

- **Against the original baseline:** both experts beat it without any HotpotQA training. The
  1.5B on 2 paragraphs scores +3.0 F1, and the 7B on 10 paragraphs +8.6 F1, over the baseline's
  58.3 dev F1.
- **Against other LLMs:** the 7B reader (66.9 F1) is close to Llama-2-70B and Mixtral-8x7B
  (67.7 and 68.1 F1), which used few-shot chain-of-thought. Reading the top 5 paragraphs (70.0)
  or using the 14B (71.9) moves past them, but both remain 5–7 F1 below GPT-3.5.
- **Against fine-tuned systems:** our best run (the 14B, 71.9) is 10–13 F1 below fine-tuned readers (82–85 F1),
  and far below humans (91.4 F1).
- **The oracle router** (77.2 F1) matches GPT-3.5 using only the 1.5B and 7B experts. With the
  14B, it reaches 79.4 F1. That is the ceiling a better router could reach with these experts.

## Comparability caveats

- **Answer only.** The leaderboard also ranks supporting-fact and joint EM/F1. Our runs do not
  predict supporting facts.
- **Dev, not test.** Our numbers are on the public dev set. The leaderboard uses the hidden test
  set, though dev and test scores were within 1 F1 for the original baseline.
- **Different setups.** The LLM rows use few-shot chain-of-thought. Our experts answer directly,
  and the small expert reads only nano-jev's top 2 paragraphs.
- **Cost is ours only.** Published results do not report cost, so GFLOPs and AIQ compare only
  our own runs.

[hotpot]: https://arxiv.org/abs/1809.09600
[distractors]: https://arxiv.org/abs/2409.05197
[lb]: https://hotpotqa.github.io/
