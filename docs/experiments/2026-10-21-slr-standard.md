# HotpotQA on the golden set: Jonathan's Stage-Level Routing setup

**In progress.**
- **Done:** the golden set is imported and verified. Experiment 10 with nano-jev judges is done. Experiments 7 and 9
  are done with nano-jev, MS MARCO and zero-shot judges.
- **Running:** experiment 8's answers on the golden set, and the H2 diagnosis.
- **Queued:** the fine-tuned deciders.

See [the job sheet](../gpu-jobs-slr.md).

**Decision (2026-10-21):** on HotpotQA the team follows Jonathan's Stage-Level Routing (SLR) setup. It is on STAIR
branch `jonathan` at `5b69cce` and is now in `stair_common` 0.3.0 (STAIR PR #8). Our numbers now report against his,
on the same terms:

| | Before (common standard, [2026-10-18](2026-10-18-common-redo.md)) | Now (golden set) |
|---|---|---|
| Test | dev[500:1000], AgentTTS's 500 | `slr-test`, 1,481 of the 7,405 dev questions |
| Train / tune | HotpotQA train draw / dev[0:500] | `slr-train` (4,814) / `slr-val` (1,110), both from dev |
| Ruler | `lenient` (`stair_common.score`) | unchanged |
| Cost | AgentTTS units from measured tokens | AgentTTS Table 5 prices per call, weak-first, every call that ran counted. Strong-everything = 929. Measured units are kept beside them |
| Router cost | the judge's own cost, always | not charged in the golden set's rows. We also report ours with the judge's cost, which our central question needs |

The common-standard write-up stays as the AgentTTS-slice history: it is still the comparison with the paper's own
500 questions.

## The golden set, verified

`slr all` (`src/stair/slr.py`, [configs/slr.toml](../../configs/slr.toml)) reads his files at the pinned commit,
sha256-checked. Output: [results/slr](../../results/slr/report.json).

- **Ruler:** `stair_common.score.lenient` gives his verdict on all 29,620 question × path cells (7,405 × 4).
- **Split:** his `splits.json`, the grid's split column and `stair_common`'s `slr-*` parts agree on every question.
- **His certified rows reproduce exactly.** For each router, the thresholds are re-selected on `slr-val` from his val
  scores with his own rule, and then replayed on `slr-test`:

| His row | His report | Recomputed | Thresholds re-selected on val (his rule) |
|---|---|---|---|
| Always-strong | 76.77 at 929 | 76.77 at 929.0 | |
| SLR SFT (label) | 76.70 at 835 | 76.70 at 834.8 | 0.95 / 1.00 / 1.00 (peak) |
| SLR SFT (label) + reward | 75.96 at 665 | 75.96 at 664.7 | 0.80 / 0.55 / 0.60 (parity) |

**His selection rules** (`configb_pair_sweep.py`):
- Each stage's threshold takes 21 values (0, 0.05, …, 1), so 9,261 points are swept on val.
- **Peak** is the most accurate point. Ties go to the dearer point, as Python's `max` over his tuples orders them.
- **Parity** is the first cheapest point at or above always-strong's val accuracy.

Experiment 10 selects its operating points with the same two rules.

## Menu A on the golden set

Every path at Table 5 prices: a fixed policy runs only its own retrieval and answer. The oracle takes the cheapest
correct path, else the cheapest path. Measured units are AgentTTS units from his token counts, a different scale
(§3 of the `stair_common` README).

**`slr-test` (1,481 questions):**

| Path | Retrieval → answer | Lenient | Price | % of 929 | Measured units |
|---|---|---|---|---|---|
| WW | Qwen2.5-7B → Llama-3.2-3B | 0.503 | 80 | 8.6% | 175 |
| WS | Qwen2.5-7B → Llama-3.1-70B | 0.640 | 192 | 20.7% | 216 |
| SW | Qwen2.5-72B → Llama-3.2-3B | 0.618 | 817 | 87.9% | 1,779 |
| SS | Qwen2.5-72B → Llama-3.1-70B | **0.768** | 929 | 100% | 1,821 |
| Oracle | cheapest correct path | 0.815 | 205.5 | 22.1% | 408 |

- **On `slr-val` (1,110):** WW 0.523, WS 0.650, SW 0.632, SS 0.768, oracle 0.815 at 199.9.
- **Clean test (1,303, without the 178 questions nano-jev saw):** WW 0.500, WS 0.645, SW 0.619, SS 0.771, oracle
  0.821.
- **Where the room is:**
  - Always-strong is 0.768. The oracle reaches 0.815 at 22% of the price.
  - His label router stays level with always-strong (76.70) at 90% of the price. His reward router gives up 0.8
    points for 72%.
  - Experiment 10 asks whether a System One judge reaches further toward the oracle.
- **Retrieval is the price:**
  - 72B retrieval is 814 of strong-everything's 929, and measured, its prompt is the whole 100-passage pool
    (about 14k tokens).
  - The weak paths that skip it (WW, WS) cost 9–21%.

## Experiment 10: the System One judge on menu A, against Jonathan's routers

**Setup.** One typed-decision judge answers two yes/no questions per question, weak-first like his routers:
- **retrieval:** do the 7B's selected passages hold what the answer needs?
- **answer:** under the chosen retrieval, will the weak answer do? It reads that branch's passages (qp), or those plus
  the weak answer (qpa).

The labels are his (`ret_labels`, `qa_labels`). It trains on `slr-train`, holding out 10% by question for the best
epoch. Thresholds come from `slr-val` with his sweep and his two rules (peak, parity), and are reported once on
`slr-test`. Code: `src/stair/exp10_slr.py`, [configs/exp10-slr.toml](../../configs/exp10-slr.toml). Jobs 43081
(training, 11 minutes) and 43082 (eval). Output: [results/exp10-slr](../../results/exp10-slr/eval/report.json).

**nano-jev judges (33M), `slr-test`, 1,481 questions.** AIQ is over weak-first prices from 80 (all WW) to 1,009 (all
SS), with the paired bootstrap.

| Router | AIQ [95% CI] | One seed | Peak point (val rule → test) | Parity point |
|---|---|---|---|---|
| SLR SFT (label) + reward (his) | **0.726** [0.705, 0.747] | | 75.96 at 679.3 | 75.96 at 664.7 |
| SLR SFT (label) (his) | 0.724 [0.704, 0.745] | | 76.70 at 834.8 | 75.35 at 670.3 |
| nano-jev qpa | 0.719 [0.698, 0.739] | 0.717, 0.717, 0.717 | 76.77 at 1,009 (all SS) | 76.77 at 1,009 |
| nano-jev qp | 0.718 [0.697, 0.739] | 0.716, 0.715, 0.716 | 76.77 at 1,009 | 76.77 at 1,009 |
| Fixed paths (hull) | 0.693 [0.672, 0.714] | | | |

- **His routers win, narrowly but clearly:**
  - qpa − his label router is [−0.011, −0.0004];
  - qpa − his reward router is [−0.013, −0.002];
  - qp is worse than both too.
  - Both judges beat the fixed paths: qp by [+0.021, +0.029], qpa by [+0.022, +0.030].
- **At his operating points, our judges collapse to escalating everything.** On val, no threshold reaches always-strong's
  accuracy at a lower price, so peak and parity both pick all-SS: 76.77 at 1,009 (weak-first). His label router
  keeps that accuracy (76.70) at 834.8, and his reward router gives up 0.8 points for 664.7.
- **Clean test** (1,303 questions): the same order. The judges score 0.721 and 0.722, his routers 0.727 and 0.729.
- **The judge's own cost doesn't change this.** A nano-jev call prices at 0 under `budget()`, since it is cheaper than
  one 3B sample.
- **Why, most likely:** his routers are Qwen2.5-1.5B with LoRA, about 45 times nano-jev's size. The 2B Strands
  Decider (job sheet row 10.3) is the size-matched comparison.

## Experiments 7 and 9 on the golden set (pair A)

**Setup.** Experiment 7's pair A is not menu A: Qwen2.5-1.5B reads the top 2 of the 10 distractor passages, and
Qwen2.5-7B reads all 10. Its caches are cut from experiment 1's full-dev run onto `slr-train` / `slr-val` /
`slr-test`, with no model calls (job 43083). The judges and LR routers retrain on `slr-train` (43084, 43085), and are
evaluated on `slr-test` (43097).

**Prices:**
- The 7B pays 10 (Table 5). The 1.5B pays 0.5: the same formula at its nominal size, as Table 5 doesn't list it.
- A router that reads the small answer pays both calls when it escalates.
- A judge's own calls are priced with `budget()` at their measured tokens: nano-jev 0, the 2B decider about 1.2, the
  E4B about 4.5.

Output: [results/slr-judges](../../results/slr-judges/eval/A/report.json).

**`slr-test`, 1,481 questions** (always-small 0.609 at 0.5, always-large 0.666 at 10, oracle 0.772 at 2.05):

| Router | AIQ [95% CI] | With its own price |
|---|---|---|
| Oracle | 0.758 | |
| Decider E4B, zero-shot (`grounded`) | **0.696** [0.675, 0.717] | 0.650 |
| LR evidence + small answer | 0.681 [0.661, 0.702] | |
| Decider 2B, zero-shot (`correct`) | 0.678 [0.656, 0.698] | 0.667 |
| nano-jev qpa (mean of 3 seeds) | 0.677 [0.658, 0.699] | 0.677 (one seed: 0.669–0.678) |
| LR cascade (question + evidence + small answer) | 0.676 [0.655, 0.698] | |
| nano-jev qp | 0.674 [0.654, 0.698] | 0.674 |
| Entropy threshold | 0.665 [0.645, 0.686] | |
| MS MARCO cross-encoder, qp | 0.659 [0.639, 0.680] | |
| nano-jev qa | 0.646 | |
| nano-jev q | 0.641 | |
| Random | 0.638 | |

**Paired differences:**

| Comparison | Before its price | With its price |
|---|---|---|
| nano-jev qpa − entropy | [−0.000, +0.025] | the same (its calls price at 0) |
| nano-jev qpa − LR cascade | [−0.007, +0.011] | the same |
| Zero-shot E4B − entropy | [+0.019, +0.043] | **[−0.030, −0.003]** |
| Zero-shot E4B − LR cascade | [+0.009, +0.030] | **[−0.039, −0.015]** |
| Zero-shot 2B − LR cascade | [−0.009, +0.012] | [−0.019, −0.0001] |
| MS MARCO qp − nano-jev qp | [−0.024, −0.007] | |
| MS MARCO qpa − nano-jev qpa | [−0.031, −0.014] | |

- **The research questions on the golden set:**
  - **RQ2:** the passages carry the signal. q (0.641) and qa (0.646) are far below qp (0.674) and qpa (0.677).
  - **RQ3:** the label doesn't matter here. large-helps and small-fails give 0.674 against 0.674 (qp), and 0.677
    against 0.675 (qpa).
  - **RQ4:** nano-jev's pretraining beats MS MARCO's, by +0.007 to +0.031.
- **The best router before cost is the zero-shot E4B decider.** But at Table 5 prices its 4.5 a call is nearly half
  the 7B's answer, so it pays for itself out of the lead. Every LR router and entropy beat it with its price.
- **The 33M nano-jev judge ties entropy and the LR cascade,** at no price.
- **Clean test** (1,303): the same order. E4B zero-shot 0.694, LR 0.676, nano-jev qpa 0.672, entropy 0.661.
- **Fine-tuned deciders:** queued (job sheet rows 7.5, 7.6).
- **Caveat on the prices:** Table 5 prices each answering call at the paper's nominal lengths (256 + 64 tokens), while
  the 7B here reads all 10 passages. A judge has no Table 5 entry, so it is priced at its measured tokens. Relative to
  the experts, judges are therefore dearer here than in measured AgentTTS units. There, the E4B zero-shot with its cost
  tied the LR cascade router ([2026-10-18](2026-10-18-common-redo.md)).
