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
| 4.6 On the team's common standard; larger judges; more options | Experiments 7 and 9 on the shared splits, lenient labels and AgentTTS units, with each judge's own cost. Experiment 8: choosing among five answer options (3B with 1 to 40 samples fused, 70B) | [experiments/2026-10-18-common-redo.md](experiments/2026-10-18-common-redo.md), `results/common*/`, `results/exp8-common/` |
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

## Headline numbers on the golden set (`slr-test`, 1,481 questions, lenient, Table 5 prices)

Since 2026-10-21 the HotpotQA headline follows Jonathan's Stage-Level Routing setup
([write-up](experiments/2026-10-21-slr-standard.md)). Our rows land here as experiments 10, 7/9 and 8 move.

| Claim | Number | Source |
|---|---|---|
| Menu A ladder | WW 0.503 at 80, WS 0.640 at 192, SW 0.618 at 817, always-strong 0.768 at 929; oracle 0.815 at 205.5 | [Golden set](experiments/2026-10-21-slr-standard.md) |
| Jonathan's SLR routers (reference) | SFT (label) 76.70 at 834.8; SFT (label) + reward 75.96 at 664.7. Both reproduced exactly, with thresholds re-selected on val | [Golden set](experiments/2026-10-21-slr-standard.md) |

## Headline numbers on the common standard (AgentTTS slice: test = dev[500:1000], 500 questions, lenient, AgentTTS units)

Kept for the comparison with the AgentTTS paper on its own 500 questions.

| Claim | Number | Source |
|---|---|---|
| Decider qpa judge (2B or E4B, one seed), with its own cost | AIQ 0.724. Beats entropy (0.697; all six seeds' intervals above 0, e.g. [+0.009, +0.051]) and the nano-jev judges (about +0.03). Ties the best LR cascade router (0.706; 5 of 6 intervals touch 0) | [Common redo](experiments/2026-10-18-common-redo.md) |
| nano-jev judges (experiment 7) | qp 0.696, qpa 0.698: tie entropy and the LR routers on 500 questions | [Common redo](experiments/2026-10-18-common-redo.md) |
| Mean of 3 seeds vs one judge | The E4B mean scores 0.740 before cost, but 0.708 with its three passes (ties entropy). One judge is the better deployment | [Common redo](experiments/2026-10-18-common-redo.md) |
| Experiment 8, cheapest strong option | 3B ×40 fused: lenient 0.746 (equal to AgentTTS) at 71 units under AgentTTS Eq. 8, below the 70B (0.670, 94 units). AgentTTS's recipe costs at least 302 units (97% of strong-everything, 311), so this is its accuracy at under a quarter of its cost | [Common redo](experiments/2026-10-18-common-redo.md) |
| Experiment 8, option judge | nano-jev qpa 0.714: ties the best fixed mix (0.716), beats entropy [+0.008, +0.044]. On the LLM grade it beats the fixed mix but ties entropy. Deciders don't help with five options | [Common redo](experiments/2026-10-18-common-redo.md) |
