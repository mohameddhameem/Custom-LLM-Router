# Experiment 7: a System One router

The router itself is a small typed-decision judge. The plan and research questions are in
[system-one-router.md](../system-one-router.md), and the omega job sheet is
[gpu-jobs-system-one.md](../gpu-jobs-system-one.md). It ran on omega on 2026-10-07 and reuses the cached
outputs of experiments 1, 4 and 5. **No LLM was rerun.**

**Result:**
- **A judge fine-tuned to read the question and the small model's passages routes better than
  the best feature-based router that decides at the same point.** It does so before the small
  model has answered and costs about 3% of that model (AIQ 0.673 against 0.665).
- **Reading the small model's answer adds nothing once the passages are read.** That judge ties
  the best cascade router.
- **The zero-shot judges tie the entropy baseline.** The judge pretrained for relevance only
  (MS MARCO) is worse than nano-jev, so its decision pretraining matters.

## Setup

| | |
|---|---|
| Judge | nano-jev v1.0 (33M MiniLM cross-encoder, revision `ffee345e`). Typed question: *Which model should answer this question: {question}*, options *small model* / *large model*. Each option is scored as a (question + option, state) pair and the two scores are softmaxed |
| State | **q** the question; **qp** + the small model's top-2 passages; **qpa** + its proposed answer; **qa** the answer without the passages |
| Training | 12,000 router-train questions of pair A (10% held out to pick the epoch by dev NLL), label `large-helps` (F1 ≥ 0.8, τ = 0.8), lr 5e-5, weight decay 0.01, 6% warmup, 16 questions per batch, bf16, up to 3 epochs, **3 seeds**. A policy's score is the mean probability over the 3 seeds |
| Pairs | **A** 1.5B (top 2) / 7B (all 10), the main pair; **B** 1.5B / 14B; **C** 3B (top 2) / 7B. Caches from `results/gpu`, `results/gpu-large-14b`, `results/gpu-small-3b` |
| Judges | 12 trained configurations × 3 seeds = 36 trainings, plus 3 zero-shot judges. See the table in [system-one-router.md](../system-one-router.md#design) |
| Evaluation | As in experiments 1–6: AIQ, the operating point chosen on calib, a 1,000-sample paired bootstrap, on the 7,405 test questions and the 6,405 without nano-jev's validation questions |
| Baselines | Always small/large, random, oracle, the small-entropy threshold and the 10 logistic regression (LR) routers of experiment 2 |

## Execution

| | |
|---|---|
| Code | `master` at `f40b6d4` |
| Hardware | omega `a-11`: 1 × NVIDIA L40S, 8 CPUs, 64 GB RAM |
| Software | torch 2.14.1+cu126, transformers 5.17.0 |
| Pilot | PBS job 42314: one judge (`qpa-large-helps-nanojev-A`), 3 seeds. Exit 0 in 7 min 51 s, with a dev NLL of 0.431–0.443 at the best epoch, below the 0.45 bar |
| Full run | PBS job 42315, backup job 42316 (`afternotok`). **Exit 0 on the first attempt in 1 h 20 min.** No errors and no retries, so the backup never ran. The pilot's 3 seeds were skipped as already done |
| Preparation | The MS MARCO MiniLM-L12 model (`cross-encoder/ms-marco-MiniLM-L12-v2`, revision `7b023523`) was downloaded on the login node. The other inputs already existed from earlier runs |
| Logs | `results/system-one/logs/42314.log`, `42315.log` |

All 36 judges have a `done.json` (the pilot's 3 plus 33 from the full job), and all six reports
(`report.json`, `report-clean.json` for pairs A, B and C) exist.

**Timing:** the 36 trainings took 58 minutes in total (the `q` and `qa` judges about 37 s each, the
`qp` and `qpa` judges about 107–109 s). Scoring takes 2.1 ms per question for `qp`/`qpa` and 0.4
ms for `q` on the L40S. Zero-shot scoring takes 4.2 ms per question for both zero-shot judges.
The rest of the 1 h 20 min was scoring, evaluation and the bootstrap.

**Training curves:** the best epoch is 1 or 2 for every training (20 and 16 of the 36). The best dev
NLL of the `large-helps` judges is 0.421–0.472, against about 0.45 for predicting the label rate
alone, so the judges learn a ranking and not much calibration. Epoch 3 overfits: in the pilot the dev
NLL rose to 0.50–0.55. Epoch selection on the dev hold-out therefore matters. (The `small-fails`
judges reach 0.63–0.65, but their label rate is 46%.)

## Results

Test set, 7,405 questions, pair A (1.5B / 7B). AIQ is the area under the F1-vs-GFLOPs hull, with
95% bootstrap intervals. For judges it is the 3-seed ensemble, and "per seed" is the mean of the
three single-seed AIQs. The "judge cost" column is GFLOPs per question (the small model costs
1,051).

| Policy | Decides | AIQ [95% CI] | Per seed | Judge cost |
|---|---|---|---|---|
| Random | | 0.641 [0.633, 0.649] | | |
| Small-entropy threshold | after | 0.662 [0.653, 0.671] | | |
| Best LR before the small model (question+evidence / large-helps) | before | 0.665 [0.657, 0.675] | | |
| Best LR after the small model (question+evidence+small / large-helps) | after | 0.672 [0.665, 0.682] | | |
| zero-shot `sufficient` | before | 0.661 [0.653, 0.671] | | 0 |
| zero-shot `grounded` | after | 0.661 [0.652, 0.670] | | 34 |
| zero-shot "is the answer correct?" | after | 0.664 [0.656, 0.674] | | 33 |
| judge **q** | before | 0.647 [0.639, 0.656] | 0.647 ± 0.001 | 7 |
| judge **qp** | before | **0.673** [0.665, 0.682] | 0.671 ± 0.000 | 32 |
| judge **qpa** | after | **0.673** [0.665, 0.682] | 0.669 ± 0.003 | 33 |
| judge **qa** | after | 0.649 [0.641, 0.658] | 0.648 ± 0.001 | 6 |
| judge qp, label `small-fails` | before | 0.667 [0.658, 0.675] | 0.665 ± 0.001 | 32 |
| judge qpa, label `small-fails` | after | 0.668 [0.660, 0.677] | 0.666 ± 0.004 | 33 |
| judge qp, MS MARCO start | before | 0.660 [0.652, 0.668] | 0.658 ± 0.001 | 32 |
| judge qpa, MS MARCO start | after | 0.659 [0.651, 0.668] | 0.656 ± 0.003 | 33 |
| Oracle | | 0.755 [0.746, 0.763] | | |

Paired bootstrap differences in AIQ (95% interval; the "p" is the share of resamples ≤ 0):

| Judge | Against the entropy baseline | Against the best LR router that decides at the same point |
|---|---|---|
| qp | [+0.006, +0.016], p < 0.001 | [+0.004, +0.011], p < 0.001 |
| qpa | [+0.006, +0.016], p < 0.001 | [−0.003, +0.004], p = 0.45 (a tie) |
| zero-shot correct | [−0.003, +0.008], p = 0.17 | [−0.012, −0.004] |
| qp, MS MARCO | [−0.007, +0.002], p = 0.85 | [−0.010, −0.002] |
| q | [−0.019, −0.010] | [−0.022, −0.014] |

On the clean test set (6,405 questions) the picture is the same: qp 0.675 and qpa 0.674 against
entropy 0.662, and qp beats the best "before" LR router by [+0.004, +0.011].

**Operating points** (chosen on calib, pair A; always-large is F1 0.669 at 22,762 GFLOPs):
judge qp, F1 0.660 at 4,882 GFLOPs with 18% of questions sent to the 7B; judge qpa, F1 0.665 at
5,995 GFLOPs with 22%; the best cascade LR router (experiment 2), F1 0.665 at 6,340 GFLOPs with
23%. At these points no judge matches always-large's F1.

### Answers to the research questions

- **RQ1, zero-shot:** none of the three zero-shot judges beats the entropy threshold
  significantly. "Is the answer correct?" is the best (0.664) and `sufficient` and `grounded`
  match it. nano-jev's built-in decisions do not transfer to routing without fine-tuning.
- **RQ2, what the judge reads:** the passages carry the signal. With them the AIQ is 0.673; without
  them 0.647–0.649, which is little above random (0.641) and below the entropy baseline.
  Adding the small model's answer to the passages (qpa) changes nothing (0.673 against 0.673).
  A judge that decides before the small model runs therefore matches one that waits for it.
- **RQ3, label:** `large-helps` scores higher than `small-fails` (qp 0.673 against 0.667, qpa
  0.673 against 0.668), as it did for LR. These are point differences. I did not test them with
  the bootstrap.
- **RQ4, decision pretraining:** starting from the MS MARCO relevance model costs about 0.013 AIQ
  against nano-jev (0.660 against 0.673) and leaves it no better than the entropy baseline.
- **RQ5, transfer:** a judge trained on pair A transfers to pair B (small model unchanged) with
  almost no loss: qp 0.6971 against 0.6973 for a judge trained on B, and qpa 0.6963 against
  0.7000. On pair C (small model changed) the gap is small: qp 0.6757 against 0.6770, qpa 0.6752
  against 0.6787. On pair C, a judge trained on A is not significantly better than entropy
  (qp [−0.001, +0.010], p = 0.052), whereas the C-trained judge is (qpa [+0.002, +0.012]).
- **RQ6, cost:** the fine-tuned qp/qpa judges cost 32–33 GFLOPs per question, 3.1% of the 1.5B
  model. Adding that cost lowers every AIQ by about 0.0001 at most, so the ranking does not change.

### Other pairs

| Pair | Entropy | Best LR before / after | Judge qp (trained on A) | Judge qpa (trained on A) | Best judge (any training pair) |
|---|---|---|---|---|---|
| A: 1.5B / 7B | 0.662 | 0.665 / 0.672 | 0.673 | 0.673 | 0.674 (qpa, B-trained) |
| B: 1.5B / 14B | 0.687 | 0.691 / 0.697 | 0.697 | 0.696 | 0.700 (qpa, B-trained) |
| C: 3B / 7B | 0.671 | 0.672 / 0.681 | 0.676 | 0.675 | 0.679 (qpa, C-trained) |

On pair B, the best judge beats the best cascade LR router by [+0.000, +0.006] (p = 0.022). On
pair C, no judge beats entropy by a clear margin except the C-trained ones, and the best LR
router (0.681) is above every judge.

## Findings

1. **A 33M judge fine-tuned to read the question and the small model's passages is the best router
   that decides before the small model runs.** It beats the best LR router of that type by +0.008
   AIQ and the entropy threshold by +0.011, and matches the best cascade LR router that waits for
   the small model, at about 3% extra cost.
2. **A "before" router could also skip the small model** for questions it sends straight to the
   large one. The curves here count the small model's cost for every question, so that saving is
   not measured.
3. **The judge's gain comes from reading the passages, not from the label or the answer.** The
   answer adds nothing once the passages are read. The question alone is barely better than
   random.
4. **nano-jev's typed-decision pretraining matters.** The MS MARCO model of the same size is
   worse by about 0.013 AIQ, though I cannot say which part of the pretraining matters.
5. **The judge does not need retraining when the large model changes.** It is weaker when the
   small model changes, but the loss is within the bootstrap noise.

## Cautions

- **The judge does not beat always-large.** Like every router so far, it reaches 0.660 F1 at
  its operating point against 0.669, so the gain is in cost, not accuracy.
- **Ensemble against single seeds.** The headline AIQ is the 3-seed ensemble. A single seed scores
  lower, 0.671 for qp and 0.669 for qpa, which is still above the best "before" LR router (0.665).
  The qpa judge varies more between seeds (±0.003) than qp (±0.000).
- **qpa against the best cascade LR router is a tie,** not a win: the interval includes zero.
- **The top row in the pair A log is a B-trained judge.** The best AIQ on pair A overall (0.674) comes
  from a judge trained on pair B and scored on A, not from an A-trained judge. The in-pair
  best (0.673) is the A-trained qp judge.
- **Little gain in NLL:** about 17% of questions have `large-helps`, so predicting the label rate
  alone gives a dev NLL of about 0.45. The judges' best 0.42–0.44 is only slightly below it, so
  their value is in ranking questions, which is what AIQ measures.
- **Not tested here:** judge against judge differences (RQ3), other τ, and other datasets.

## Reproducing

```bash
uv run python -c "from huggingface_hub import snapshot_download as d; d('sdmlai/nano-jev', revision='v1.0'); d('cross-encoder/ms-marco-MiniLM-L12-v2')"
qsub -v STEP=judge-pilot scripts/omega.pbs      # optional, ~8 min
J1=$(qsub -v STEP=judge scripts/omega.pbs)      # ~1 h 20 min on one L40S
qsub -W depend=afternotok:$J1 -v STEP=judge scripts/omega.pbs
```

The results are in [results/system-one/](../../results/system-one/): `eval/<pair>/` has the
reports, curves and every judge's per-question scores (enough to redo the evaluation on CPU),
`judges/<judge>/seed<k>/done.json` has each training's curve, dev NLL and timing, and `zeroshot/`
has the zero-shot records. The per-seed score files and saved models are not tracked.
