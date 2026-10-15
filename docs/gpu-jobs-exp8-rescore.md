# GPU jobs: re-score experiment 8 with an LLM grader

**Audience: a Claude Code session running on the omega login node.** Follow this file step by step. It is
written as instructions to you. Do not skip the checks, and stop where it says STOP. The human (the
project owner) reads your final report and the committed write-up. They do not watch the jobs.

## Why this job exists (read first)

Experiment 8's accuracies used a placeholder *lenient* scorer: an answer counted as correct when the gold
answer appeared anywhere in it. A local check on 2026-10-15 showed that this scorer rewards long answers.
See the "Metric caveat" at the top of [experiments/2026-10-14-experiment-8.md](experiments/2026-10-14-experiment-8.md):

- **No model beats every model.** Pasting nano-jev's top-5 passages as the "answer", with no model, scores
  0.798 lenient, above every model.
- **The ranking reverses with length fixed.** Counting only the first 40 words or the first sentence, the
  70B beats the 40-sample fused 3B.
- **Many credited hits are wrong.** Most of those that appear only after the first sentence are hedges, or
  wrong first answers that later quote a passage.

This job re-scores every cached answer two ways:

1. **An LLM grader** (`Qwen/Qwen2.5-14B-Instruct`, bf16, one answer per forward pass). It reads (question,
   gold, response) and returns P(CORRECT) from the exact log-likelihoods of the two verdict words. This
   becomes the primary metric, `correct` = P ≥ 0.5.
2. **Short-answer extraction** by the same model, then the official HotpotQA EM/F1. That is the secondary
   metric, comparable with experiments 1–7.

Then the five-option System One judge is retrained and re-evaluated on the grader's labels.

No answers are regenerated. The code is in `src/stair/rescore.py`; read its docstring.

## Rules for you (the cluster Claude)

- **MUST** work on branch `feat/exp8-rescore` (`git fetch && git switch feat/exp8-rescore && git pull`).
- **MUST** run `exp8-rescore-validate` first, and **STOP** if `runs/exp8/rescore/validation.json` has
  `"passed": false`. If that happens:
  1. Report the numbers and the first 20 rows of `runs/exp8/rescore/validation-disagreements.csv`.
  2. Do not continue to `exp8-rescore`.
  3. Do not change the thresholds or the grader prompt to make it pass.
- **MUST NOT** edit the cached answers (`runs/exp8/answers/**`, `results/exp8-phase1/**`), the prompts in
  `rescore.py`, the validation thresholds (`MIN_AGREE_EM1`, `MIN_AGREE_F10`, `MAX_DUMP_CORRECT`), or
  `configs/exp8.toml`.
- **MUST NOT** tune anything on the test split. The operating points are chosen on calib by the existing
  code.
- **MUST NOT** delete or soften the "Metric caveat" blocks. You may add a line under them pointing to the
  re-scored write-up (step 6).
- **If a job fails:**
  - **Twice for the same reason:** stop and report the error and the log path.
  - **Trivial fixes are allowed:** a path, an out-of-memory error (lower `--extract-batch`), a missing
    download. Describe every fix in the write-up's Execution section.
  - **Do not rewrite the logic.**
- **Report numbers as they come out,** with their intervals. Do not round in a way that changes a
  comparison. Write "not significant" when an interval includes zero.
- **The model must be downloaded on the login node** (step 1). The GPU jobs run with `HF_HUB_OFFLINE=1`.
- **MUST NOT run compute on the login node.** The login node is for git, downloads, reading files and `qsub`
  (SMU OMEGA user guide; [omega-cluster.md](omega-cluster.md), "Good practice on OMEGA"). Everything else runs as a
  job:
  - GPU steps use `scripts/omega.pbs`;
  - CPU-only steps (building pools, the summary table) use `scripts/omega-cpu.pbs` (queue `short`).

  Short `uv run python` one-liners that only read result files (the checks below) are fine on the login
  node. Anything that loads a model, builds pools or trains is not.
- **MUST give every `qsub` the walltime shown in this file.** `scripts/omega.pbs` defaults to 4 h. Ask for
  no more than a step needs.

## Step 1: update and download (login node)

```bash
cd ~/Custom-LLM-Router
git fetch && git switch feat/exp8-rescore && git pull
uv sync --extra llm
uv run python -c "from huggingface_hub import snapshot_download as d; print(d('Qwen/Qwen2.5-14B-Instruct'))"   # ~30 GB, ungated
ls runs/exp8/answers/test runs/exp8/answers/calib runs/exp8/answers/router_train
ls runs/exp8/pools/test.parquet runs/exp8/scores/test.parquet results/gpu/test.parquet
project_summary                                  # check the CPU/GPU-hour allocation before a multi-hour run
```

**Check:** the three `answers/` folders list the cells below, and the other files exist. If
`runs/exp8/answers` is missing, copy it from `results/exp8-phase1/answers` (`cp -r`). If the pools are
missing, rebuild them with a CPU job, `qsub -v STEP=exp8-pools scripts/omega-cpu.pbs` (about 5 min): they rebuild
identically. Do not run that step on the login node.

| Split | Cells (parquet files) | Rows each |
|---|---|---|
| test | llama3b-top3, llama3b-top5, llama3b-s10fuse-top5, llama3b-s20fuse-top5, llama3b-s40fuse-top5, llama70b-top3, llama70b-top5 | 500 |
| calib | the same, without llama70b-top3 | 1,000 |
| router_train | the same, without llama70b-top3 | 3,000 |

## Step 2: validate the grader (GPU, about 10 min)

```bash
qsub -l walltime=01:00:00 -v STEP=exp8-rescore-validate scripts/omega.pbs
```

When it finishes (`qstat -u $USER` no longer lists it), read `runs/exp8/rescore/validation.json`. It
reports:

| Field | Meaning | Must be |
|---|---|---|
| `agree_em1_correct` | Share of experiment 1's 7B short answers with official EM = 1 that the grader calls CORRECT | ≥ 0.90 |
| `agree_f10_incorrect` | Share of those with F1 = 0 that it calls INCORRECT | ≥ 0.90 |
| `dump_graded_correct` | Share of "responses" made of nano-jev's top-5 passages pasted together that it calls CORRECT (the lenient scorer gave these 0.798) | ≤ 0.10 |
| `passed` | All three hold | `true` |

Also read 10 rows of `runs/exp8/rescore/validation-disagreements.csv` yourself and note, in one line each,
whether the grader or the official EM looks right. EM is strict, so some "EM = 0 but grader says correct"
cases are fine: aliases and formats. **If `passed` is false: STOP** (see Rules).

## Step 3: re-score every answer (GPU, about 2–3 h; two jobs in parallel)

```bash
J1=$(qsub -l walltime=03:00:00 -v STEP=exp8-rescore,SPLITS=test scripts/omega.pbs)                    # 3,500 answers
J2=$(qsub -l walltime=10:00:00 -v "STEP=exp8-rescore,SPLITS=calib router_train" scripts/omega.pbs)   # 24,000 answers
```

- **Resuming:** each answer file is scored in resumable chunks of 250 into
  `runs/exp8/rescore/<split>/<cell>.parts/`, then merged into `runs/exp8/rescore/<split>/<cell>.parquet`.
  If a job dies, resubmit the same line.
- **Summary:** each job ends with `rescore summary`, which writes `runs/exp8/rescore/summary.csv`, one row
  per split and cell, with `lenient`, `graded_correct`, `ext_em`, `ext_f1` and `units`.
- **After both jobs:** each job's summary only saw the files done by then, so rewrite the full table
  with a short CPU job: `qsub -l walltime=00:15:00 -v STEP=exp8-rescore-summary scripts/omega-cpu.pbs`.

**Checks** (write the results down for the write-up):

1. **Completeness:** there are 19 files under `runs/exp8/rescore/{test,calib,router_train}/` (7 + 6 + 6),
   each with the row count above, unique ids, and no missing `p_correct`.
2. **Consistency with the lenient scorer:** `graded_correct` ≤ `lenient` + 0.05 for most cells. The grader
   is stricter but also accepts format variants. A cell where `graded_correct` is far *above* `lenient` is
   suspicious: inspect 5 of its rows.
3. **The length question:** on test, compare `graded_correct` for `llama70b-top5` with
   `llama3b-s40fuse-top5`. This is the key result. Report both, and their paired difference with a
   bootstrap interval:

   ```bash
   uv run python - <<'EOF'
   import numpy as np, pandas as pd
   a = pd.read_parquet("runs/exp8/rescore/test/llama70b-top5.parquet").set_index("id")["correct"]
   b = pd.read_parquet("runs/exp8/rescore/test/llama3b-s40fuse-top5.parquet").set_index("id").loc[a.index, "correct"]
   rng = np.random.default_rng(0); n = len(a); d = []
   for _ in range(10000):
       i = rng.integers(0, n, n); d.append(b.to_numpy()[i].mean() - a.to_numpy()[i].mean())
   print(f"70B {a.mean():.3f}  fused-40 {b.mean():.3f}  diff {b.mean()-a.mean():+.3f}  95% CI [{np.percentile(d,2.5):+.3f}, {np.percentile(d,97.5):+.3f}]")
   EOF
   ```

4. **Spot-check:** read 5 random test rows where `lenient = 1` but `correct = 0`, and 5 where `lenient = 0`
   but `correct = 1`. Quote one example of each in the write-up.

## Step 4: retrain and evaluate the judge on the grader's labels (GPU, about 20 min)

```bash
qsub -l select=1:ncpus=8:ngpus=1:mem=32gb -l walltime=01:00:00 -v STEP=exp8-judge-graded scripts/omega.pbs
```

This uses [`configs/exp8-judge-graded.toml`](../configs/exp8-judge-graded.toml):

- the same five options, seeds and training as the lenient run;
- `metric = "correct"`, read from `runs/exp8/rescore`;
- output in `runs/exp8/judge-5opt-graded/`.

The lenient run's outputs in `runs/exp8/judge-5opt` are left untouched.

**Check:** `runs/exp8/judge-5opt-graded/eval/report.json` exists with `"metric": "correct"`, and the
same for `report-clean.json`. There are 6 `done.json` files (qp and qpa × 3 seeds).

## Step 5: copy results into git

```bash
mkdir -p results/exp8-rescore
cp runs/exp8/rescore/validation.json runs/exp8/rescore/validation-disagreements.csv runs/exp8/rescore/summary.csv results/exp8-rescore/
rsync -a --exclude='*.parts/' --include='*/' --include='*.parquet' --exclude='*' runs/exp8/rescore/ results/exp8-rescore/graded/
mkdir -p results/exp8-phase2c && rsync -a --exclude='*.parts/' runs/exp8/judge-5opt-graded/ results/exp8-phase2c/
cp configs/exp8-judge-graded.toml results/exp8-phase2c/plan.toml
mkdir -p results/exp8-rescore/logs && cp runs/logs/<every job id of steps 2-4>.log results/exp8-rescore/logs/
du -sh results/exp8-rescore results/exp8-phase2c    # expect a few MB
```

## Step 6: write it up

Create `docs/experiments/<YYYY-MM-DD>-experiment-8-rescored.md`, using today's date, with these sections in
this order. Be factual. Every number comes from the files above.

1. **Title and result bullets (3–5).** The main questions to answer:
   - Under the grader, which answering option is most accurate on test, and is the fused 3B still ahead of
     the 70B? Give the step 3 check 3 interval.
   - Does the judge still beat the fixed-policy hull, the entropy threshold and logistic regression? Give
     the paired AIQ intervals from `report.json` → `bootstrap`.
   - What is the judge's accuracy and cost at its operating points?
2. **Setup:** the grader (model, revision from `validation.json`, bf16, one answer per pass), the prompt
   (point to `rescore.py`), extraction, `correct` = P ≥ 0.5, and the judge settings (unchanged except the
   labels).
3. **Validation:** the table of step 2, plus your notes on the disagreements.
4. **Execution:** job ids, durations, exit codes, any fix you made, and the checks of steps 3–4.
5. **Results.**
   - **Per-cell table on test:** lenient, graded, extracted EM, extracted F1, units and % of 929. Put the
     passage-dump control in a row. Do the same in short for calib and router_train, to show whether test
     is representative.
   - **Policy table:** fixed hull, entropy, logistic regression before and after, judge qp and qpa at the
     budget point and the near-best point, and the oracle, with AIQ and intervals.
   - **What-the-judge-picks table**, as in the earlier write-ups.
6. **Hypotheses, re-assessed under the grader:**
   - **H2:** the 70B on nano-jev's top-k against 72.2. Note that 72.2 was a lenient number from the team's
     grid, so it is not directly comparable.
   - **H3:** state plainly that AgentTTS's 74.6 is *lenient* and therefore **not comparable**. A fair
     comparison needs AgentTTS's own answers graded by this same grader. The team's reproduction has them;
     note this as the next step. Do not write "H3 met" or "H3 not met" against 74.6.
7. **Findings and cautions:** 500 test questions give about ±4 points; the grader's own error rate (from
   validation); the 4-bit 70B.
8. **Reproducing:** the commands of steps 2–4.

Then edit these two documents, and nothing else in them:

- **`docs/experiments.md`:** under the "Metric caveat" in the Experiment 8 section, add one line, "Re-scored
  with an LLM grader on <date>: see experiments/<file>.md", and a short graded-results table (cell, graded
  accuracy, units).
- **`docs/experiment-8-plan.md`:** add the same one line under its caveat.

## Step 7: commit and report

```bash
git add src configs docs results/exp8-rescore results/exp8-phase2c
git commit -m "Re-score experiment 8 with an LLM grader; retrain the judge on graded labels

<3-6 lines: validation outcome, the 70B vs fused-40 result with its interval, the judge result>

Co-Authored-By: Claude <noreply@anthropic.com>"
git push origin feat/exp8-rescore
```

**Do not open or merge a PR.** The project owner reviews it from their local session. Finish with a short
chat report to the human:

- the validation pass/fail and its three numbers;
- the step 3 check 3 result;
- the judge's AIQ against the hull, with its interval;
- the paths you committed.
