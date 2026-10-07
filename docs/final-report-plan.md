# Final report plan

The skeleton of the final report, and which document or result feeds each section. Keep this
file current as experiments finish, so the report can be written from it.

**Central research question** (from [system-one-router.md](system-one-router.md)):

> Can a small typed-decision judge (System One), which reads the question, the evidence and
> optionally the small model's answer, route between a small and a large LLM better than
> feature-based routers and uncertainty thresholds, once the judge's own cost is counted?

Experiments 1–6 build the setting, the baselines and the motivation for this question.
Experiment 7 answers it.

| Report section | Content | Sources |
|---|---|---|
| 1. Introduction | Cost-aware routing between a small and a large LLM. The central RQ and sub-questions RQ1–RQ6. The System One pattern (Jev AI guide) as the design idea | [system-one-router.md](system-one-router.md) |
| 2. Related work | HotpotQA readers and leaderboard; LLM-era routing (RouteLLM, Adaptive-RAG, entropy baselines); nano-jev | [research-report.md](research-report.md), [reference-projects.md](reference-projects.md) |
| 3. Setup | Data and splits (hard-only, nano-jev exclusions, clean test set); experts; cost in GFLOPs; AIQ, CPT, the operating point, calibration, the paired bootstrap | [experiment-protocol.md](experiment-protocol.md), [experiments/2026-10-05-omega-full-run.md](experiments/2026-10-05-omega-full-run.md) |
| 4.1 Feature-based routing | Evidence beats question-only routing. `large-helps` beats `small-fails`. Before the small model runs, no router significantly beats entropy | Experiments 1 and 2 |
| 4.2 Cascade routers | Using the small model's answer signals beats entropy significantly. τ is insensitive | [Experiment 2](experiments/2026-10-06-cascade-routers.md) |
| 4.3 The experts matter | Distractors hurt the 7B: top 3 and top 5 are better and cheaper. A 14B widens the gap. A 3B small model is hardly better | [Experiments 3–5](experiments/2026-10-06-experiments-3-5.md) |
| 4.4 Multi-stage cascade | 1.5B → 7B (top 3) → 14B beats always-14B at 32% of its cost | [Experiment 6](experiments/2026-10-06-multi-stage-cascade.md) |
| **4.5 System One router (main result)** | RQ1 zero-shot judges; RQ2 input ablation; RQ3 label; RQ4 pretraining; RQ5 transfer; RQ6 judge cost | Experiment 7: design in [system-one-router.md](system-one-router.md), results in [experiments/2026-10-07-system-one-router.md](experiments/2026-10-07-system-one-router.md) and `results/system-one/` |
| 5. Comparison with published results | Answer EM/F1 against the HotpotQA baseline, LLM few-shot CoT and fine-tuned readers, with the comparability caveats | [experiments.md](experiments.md) |
| 6. Discussion and limitations | Oracle headroom; HotpotQA shortcuts; label proxy (τ); contamination handling; cost accounting (scorer, separate calls); the Jev guide is not peer-reviewed | All write-ups' caveat sections |
| 7. Future work | Fine-tuned readers or chain-of-thought to pass GPT-3.5; MuSiQue / 2Wiki transfer; conformal risk control | [research-report.md](research-report.md) |

## Headline numbers so far (test, 7,405 questions)

| Claim | Number | Source |
|---|---|---|
| Best feature-based cascade router (1.5B / 7B), AIQ | 0.672, entropy 0.662 (difference CI [+0.006, +0.015]) | Experiment 2 |
| Best single-call answer quality | 7B on top 5: F1 0.700 at 51% of the 7B-all-10 cost | Experiment 3b |
| Best system | 3-stage cascade: EM 58.4 / F1 72.6 at 14,243 GFLOPs (always-14B: 57.1 / 71.9 at 43,906) | Experiment 6 |
| **System One router (main result, pair A)** | Judge qp, deciding before the small model: AIQ 0.673. It beats entropy (0.662, CI [+0.006, +0.016]) and the matched "before" LR router (0.665, CI [+0.004, +0.011]), and ties the best cascade LR router (0.672) at 3% of the small model's cost. 18% cheaper than that LR router at the same F1. Lowest CPT 80% (5,380 GFLOPs) | [Experiment 7](experiments/2026-10-07-system-one-router.md) |
| RQ answers | RQ1: zero-shot ties entropy. RQ2: the passages carry the signal (qp ≈ qpa ≫ q, qa). RQ3: large-helps > small-fails. RQ4: nano-jev pretraining > MS MARCO (+0.013). RQ5: qp transfers across pairs within 0.004. RQ6: 3% cost, AIQ −0.0001 | [Experiment 7](experiments/2026-10-07-system-one-router.md) |
| Against published results | Beats the HotpotQA baseline (58.3 F1), Llama-2-70B and Mixtral few-shot CoT (67.7 / 68.1). Below GPT-3.5 CoT (77.2) | [experiments.md](experiments.md) |
