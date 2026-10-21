# HotpotQA on the golden set: Jonathan's Stage-Level Routing setup

**In progress.** The golden set is imported and verified. Experiment 10 and the moves of experiments 7/9 and 8 are
queued: see [the job sheet](../gpu-jobs-slr.md).

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
