# Experiments 7, 8 and 9 redone on the team's common standard

**Draft.** Experiments 7 and 9 are complete. Experiment 8 is waiting on jobs 42832–42835.

These are the original experiments, rerun with the STAIR `common/` package (`stair_common` at STAIR `main` `052ebe4`),
so that our numbers can be compared directly with the rest of the team's. Original reports:
- [experiment 7](2026-10-07-system-one-router.md);
- [experiment 8](2026-10-14-experiment-8.md);
- [experiment 9](2026-10-17-experiment-9.md).

Job sheet: [gpu-jobs-common.md](../gpu-jobs-common.md). Run on OMEGA on 2026-10-17 and 2026-10-18.

**Result:**

- **Decider judges that read the small model's answer (qpa) are the only routers that beat the baselines, even
  after paying for themselves.**
  - The fine-tuned E4B qpa reaches AIQ **0.740** and the 2B qpa **0.734**; with their own cost they are 0.729 each.
  - Paired with cost, the E4B beats the entropy threshold by **[+0.012, +0.051]** and the best LR cascade router by
    **[+0.003, +0.040]**. The 2B beats them by [+0.010, +0.054] and [+0.002, +0.042].
  - Both also beat the nano-jev judges by about +0.01 to +0.05.
  - The same order holds on the clean subset and with F1 labels.
- **The 33M nano-jev judges no longer beat the baselines.** qp scores 0.696 and qpa 0.698, which ties entropy (0.697)
  and the LR routers. In the original experiment 7, qp was +0.011 over entropy on 7,405 questions. On 500 questions
  that margin is inside the noise.
- **Without the answer, the deciders tie everything.** The qp deciders score 0.706. The zero-shot E4B (0.726) beats
  entropy and nano-jev, but ties the LR cascade router once its cost is paid ([−0.008, +0.028]).
- **One epoch is enough.** Every judge's best validation loss is at epoch 1, except nano-jev qa at epoch 2.
  Validation loss rises after that, so the extra epochs (3 for nano-jev, 2 for the deciders) cost GPU time but don't
  change any result (see [Loss curves](#loss-curves)).

**For the report:** typed-decision routing works at about 2B parameters when the judge reads the small model's
answer. The 33M judge doesn't get there on this test set.

## What changed from the original runs

| | Original (experiments 7 and 9) | Common standard |
|---|---|---|
| Test set | HotpotQA dev, all 7,405 questions | HotpotQA dev[500:1000], 500 questions (AgentTTS's test slice, `stair_common.splits`) |
| Calibration (picks the operating point) | Our own calib split | dev[0:500], the team's val split |
| Judge training data | Our own split | The team's train draw from the HotpotQA train file |
| Correctness labels and headline metric | F1 | Lenient accuracy (`stair_common.score`); F1 rerun as a check |
| Cost unit | GFLOPs | AgentTTS units (`stair_common.cost`, 1 unit = 1,152 GFLOPs); samples from one prompt pay it once (AgentTTS Eq. 8), which matters only for experiment 8's sampled cells |
| Bootstrap | Paired, AIQ without the judge's cost | Paired, also **with each router's own cost** |
| Models | Pair A: Qwen2.5-1.5B (small) / 7B (large), nano-jev v1.0 retriever | Unchanged |

Baselines on the 500 test questions (lenient):
- always-small scores **0.654** at 0.91 units a question;
- always-large scores **0.674** at 19.5 units;
- the oracle scores 0.782 at 3.2 units, sending 12.8% of questions to the large model.

## Experiments 7 and 9: AIQ, 500 test questions, lenient, AgentTTS units

AIQ is the area under the accuracy-vs-cost curve. The "with own cost" column adds the judge's compute, and its 95%
interval comes from the with-cost bootstrap (1,000 resamples). For fine-tuned judges, the figure pools 3 seeds.

| Router | Decides | AIQ [95% CI] | With own cost [95% CI] | Original (7,405 q, F1, GFLOPs) |
|---|---|---|---|---|
| Oracle | | 0.774 [0.738, 0.809] | | 0.755 |
| Decider E4B fine-tuned, **qpa** | after | **0.740** [0.703, 0.777] | **0.729** [0.693, 0.765] | 0.713 (0.700 with cost) |
| Decider 2B fine-tuned, **qpa** | after | **0.734** [0.699, 0.770] | **0.729** [0.694, 0.766] | 0.708 (0.702) |
| Decider E4B zero-shot, `grounded` | after | 0.726 [0.693, 0.763] | 0.716 [0.683, 0.754] | 0.692 |
| Decider E4B zero-shot, `correct` | after | 0.725 [0.695, 0.763] | 0.716 [0.685, 0.753] | 0.693 |
| Decider 2B zero-shot, `correct` | after | 0.715 [0.679, 0.751] | 0.711 [0.675, 0.746] | 0.675 |
| nano-jev zero-shot, `correct` | after | 0.707 [0.674, 0.746] | 0.707 | 0.664 |
| Best LR cascade router (question+evidence+small) | after | 0.706 [0.672, 0.745] | | 0.672 |
| Decider 2B fine-tuned, qp | before | 0.706 [0.672, 0.743] | 0.702 [0.669, 0.740] | 0.680 |
| Decider E4B fine-tuned, qp | before | 0.706 [0.673, 0.748] | 0.699 [0.667, 0.740] | 0.681 |
| Best LR router before the small model (question+evidence) | before | 0.700 [0.667, 0.740] | | 0.665 |
| nano-jev qpa (experiment 7) | after | 0.698 [0.667, 0.738] | 0.698 | 0.673 |
| Small-entropy threshold | after | 0.697 [0.661, 0.735] | | 0.662 |
| nano-jev qp (experiment 7) | before | 0.696 [0.663, 0.736] | 0.696 | 0.673 |
| nano-jev qa | after | 0.681 [0.647, 0.719] | | 0.649 |
| nano-jev q | before | 0.676 [0.641, 0.711] | | 0.647 |
| Random | | 0.664 [0.630, 0.699] | | 0.641 |

The "with own cost" figures are the point estimates in `report.json` (`aiq_with_judge_cost`); the intervals are the
with-cost bootstrap. nano-jev's own cost (33M parameters) changes AIQ by less than 0.001.

Absolute AIQs are not comparable across the last column and the rest: the test set, labels and cost unit all
changed. The **order** is what carries over:
- the deciders' qpa judges come first, as in experiment 9;
- nano-jev drops from "best cheap router" to level with entropy.

### Paired differences with each router's own cost (95% interval, 1,000 resamples)

| Pair | Difference | Verdict |
|---|---|---|
| E4B qpa − entropy | [+0.012, +0.051] | better |
| E4B qpa − best LR cascade router | [+0.003, +0.040] | better |
| E4B qpa − nano-jev qp | [+0.013, +0.047] | better |
| E4B qpa − nano-jev qpa | [+0.011, +0.046] | better |
| 2B qpa − entropy | [+0.010, +0.054] | better |
| 2B qpa − best LR cascade router | [+0.002, +0.042] | better |
| 2B qpa − nano-jev qpa | [+0.011, +0.047] | better |
| E4B zero-shot `correct` − entropy | [+0.001, +0.041] | better, just |
| E4B zero-shot `correct` − best LR cascade router | [−0.005, +0.028] | tie |
| 2B / E4B qp − entropy, LR routers, nano-jev | all straddle 0 | tie |
| nano-jev qp − entropy | [−0.021, +0.022] | tie |
| nano-jev qpa − entropy | [−0.019, +0.024] | tie |
| nano-jev qp − best LR router before the small model | [−0.016, +0.009] | tie |
| nano-jev q − entropy | [−0.041, −0.004] | worse |

Checks:
- **Clean subset** (438 questions without nano-jev's own validation questions): E4B qpa 0.737, 2B qpa 0.730, best LR
  cascade router 0.700, nano-jev qpa 0.698, entropy 0.696. Same order.
- **F1 labels** (`runs/common-judges-f1`): 2B qpa 0.727, best LR cascade router 0.702, nano-jev qp 0.701, entropy
  0.697. Same order. The E4B wasn't retrained on F1 labels.

## Loss curves

`results/common/loss-curves.png` and `training-history.csv` hold 30 histories (10 judges × 3 seeds), in the
`stair_common.training` layout.

- Best validation epoch: 1 for every judge except nano-jev qa (2). Training loss keeps falling and validation loss
  rises after the best epoch, a clear case of overfitting.
- The saved checkpoint is the best validation epoch, so the reported results come from epoch 1.
- Lowest validation NLL: E4B qpa about 0.28–0.30, 2B qpa about 0.30, nano-jev qpa about 0.40. This matches the
  routing order.
- The seeds agree closely, so the stopping epoch is stable.

For the team: stop early on validation loss rather than fixing the epoch count. Here, one epoch would have saved most
of the 4 hours the E4B job took.

## Experiment 8

*Pending: jobs 42843 (rescore), 42844 (judge), 42845 (graded judge) and 42846 (decider judges). The grader validation
passed (42841).*

Setup: 1,000 router_train questions (a third of the original 3,000) and 500 val questions. The test set is dev[500:1000],
as in the original. The answers are complete: 3B ×10/×20/×40 fused, 3B top-5 and 70B top-5 (jobs 42818–42829).

**Cost rule corrected (2026-10-18).** AgentTTS counts S samples from one prompt as prompt + S × decode (App. A.2,
Eq. 8: the samples share the prompt's prefill). Our answer files counted the prompt once per sample. `pool-answer
recost` (job 42842) rewrote them from their stored token counts, with no new model calls, and kept the old figures as
`units_prompt_per_sample`. Mean AgentTTS units a question on test, each including nano-jev's ranking:

| Option | Prompt per sample (old) | Eq. 8 (now) |
|---|---|---|
| 3B, one answer (top 5) | 7.3 | 7.3 |
| 3B, 10 samples fused | 64.7 | 30.4 |
| 3B, 20 samples fused | 116.3 | 43.9 |
| 3B, 40 samples fused | 219.8 | 71.1 |
| 70B, one answer (top 5) | 93.6 | 93.6 |

The fused cells now all cost less than the 70B, so the options are reordered (3B → ×10 → ×20 → ×40 → 70B). The judges,
whose label is the cheapest option that answered correctly, are retrained (42844–42846). The judge trained on the old
order (42833) is kept in `runs/exp8-common/judge-5opt-prompt-per-sample`. The original experiment 8 reports
(2026-10-14 and 15) used the old rule, so their fused-cell costs are overstated by the same factors.

## Caveats

- **500 test questions instead of 7,405.** The intervals are about four times wider than in the original experiment
  7. That is why nano-jev's original +0.011 over entropy no longer shows. "Tie" here means "not detectable on 500
  questions", not "equal".
- **Test-set overlap with the judges' training.** The clean-subset check is for this: it drops 62 questions, and the
  order holds.
- **Lenient labels make "the large model helps" rarer:** the oracle escalates 12.8% (lenient) against 15.8% (F1).
  The F1 check gives the same order.
- **Seeds.** Fine-tuned AIQs pool 3 seeds. The single-seed spread is small next to the gaps: E4B qpa 0.734 ± 0.006, 2B qpa
  0.729 ± 0.002, nano-jev qp 0.694 ± 0.002, nano-jev qpa 0.696 ± 0.004 (mean ± sd).

## Where the numbers come from

All paths are on OMEGA under `~/Custom-LLM-Router`; the job IDs are logged in `runs/common-jobs.txt`.

| What | Job(s) | File |
|---|---|---|
| Splits, pools, expert answers | 42803, 42804 / 42811 | `runs/common/{calib,test}.parquet`, `config.toml` |
| Zero-shot judges | 42812 | `runs/common-judges/zeroshot/` |
| nano-jev judges (experiment 7) | 42813 | `runs/common-judges/judges/*nanojev*/seed{0,1,2}/` |
| 2B / E4B decider judges (experiment 9) | 42814, 42815 | `runs/common-judges/judges/*strands-decider*/` |
| F1-label judges | 42816 | `runs/common-judges-f1/` |
| Evaluation, with-cost bootstrap | 42817, 42830 | `runs/common-judges/eval/A/report.json`, `report-clean.json` |
| Team-layout export, loss curves | 42831 | `results/common/systemOne-hotpotqa-test.csv`, `training-history.csv`, `loss-curves.png` |
| Experiment 8 answers | 42818–42823, 42828, 42829 | `runs/exp8-common/answers/` |
