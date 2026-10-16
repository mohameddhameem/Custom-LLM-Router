# GPU jobs: experiment 9, Amazon's Strands Decider as the System One judge

**Audience: whoever drives OMEGA** (the project owner's session over SSH, or a Claude Code session on the login
node). Follow it step by step and stop where it says STOP. Plan and hypotheses:
[experiment-9-plan.md](experiment-9-plan.md).

## What this runs

Two Strands Decider checkpoints (github.com/strands-labs/strands-decider, Apache-2.0) replace nano-jev as the judge,
zero-shot and fine-tuned (LoRA + pointer head, nano-jev's protocol):

| Model | Torso | Params per pass |
|---|---|---|
| `StrandsAgents/strands-decider-2B-hobson-v21` | `Qwen/Qwen3.5-2B-Base` | 2B |
| `StrandsAgents/strands-decider-E4B-gemma4-v1-2610` | `google/gemma-4-E4B-it` | ~4B active |

- **Experiment 7, pair A** (`configs/system-one-decider.toml`):
  - zero-shot (nano-jev's two questions) and fine-tuned (qp and qpa, 3 seeds each);
  - next to experiment 7's nano-jev judges, which are reused, not retrained;
  - reports in `runs/system-one/eval-decider/A/`.
- **Experiment 8, five options, graded labels** (`configs/exp8-judge-decider-{2b,e4b}-{zs,ft}.toml`):
  - each plan's report compares pairwise against the nano-jev judges of `runs/exp8/judge-5opt-graded`.

The code is `src/stair/decider.py` (read its docstring). No answers are generated and no LLM answerer runs.

## Rules

- **Branch:** work on branch `feat/exp9-decider`. Never open or merge a PR.
- **STOP if `exp9-smoke` fails** (step 2). It checks that our wrapper scores exactly like the package's own engine.
  If it fails, report `runs/exp9/smoke.json` and stop.
- **Do not change** the configs' settings, the label definitions or anything in `results/`. Tune nothing on test;
  the operating points are chosen on calib by the existing code.
- **Login node:** git, `uv sync`, downloads, `qsub`, `qstat` and reading files only. Every model load runs as a
  job. Give every `qsub` the walltime below.
- **If a job fails:**
  - Twice for the same reason: stop and report it.
  - Trivial fixes allowed:
    - an out-of-memory error: lower `micro_batch` in the judge's `train` table, or set `--score-batch` lower, and
      record it;
    - a missing download.
  - Never change the logic.

## Step 1: update and download (login node)

```bash
cd ~/Custom-LLM-Router
git fetch && git switch feat/exp9-decider && git pull
uv sync --extra llm --extra decider
uv run decider-smoke --download-only StrandsAgents/strands-decider-2B-hobson-v21 StrandsAgents/strands-decider-E4B-gemma4-v1-2610
ls runs/system-one/judges/qp-large-helps-nanojev-A/seed0/A-test.parquet runs/exp8/judge-5opt-graded/qp/seed0/test.parquet
ls runs/exp8/rescore/test results/gpu/test.parquet data/hotpotqa/nanojev_validation_ids.txt
project_summary
```

The download is about 4.5 GB for the Qwen torso and about 16 GB for Gemma 4 E4B.
- **If the Gemma download is refused (a gated model):** accept its licence on huggingface.co with the account
  logged in here, then rerun.
- **If either nano-jev reference folder is missing:**
  - **Experiment 7:** rebuild it from experiment 7's job sheet ([gpu-jobs-system-one.md](gpu-jobs-system-one.md),
    a few minutes).
  - **Experiment 8:** copy it from `results/exp8-phase2c` into `runs/exp8/judge-5opt-graded`.

## Step 2: smoke test (GPU, about 10 min). STOP if it fails

```bash
qsub -l walltime=00:45:00 -v STEP=exp9-smoke,ATTEMPTS=1 scripts/omega.pbs
```

`runs/exp9/smoke.json` must have `"passed": true` for both models. That means:
- `max_abs_gap_vs_engine` is below 0.02 (our batched wrapper against the package engine, on three questions);
- every probability row sums to 1.

Also note each model's `choice_temperature`, `window` and `tokens`.

## Step 3: experiment 7, pair A (GPU; three jobs, at most two at a time)

```bash
J1=$(qsub -l walltime=03:00:00 -v STEP=exp9-judge7,J7STEP=zeroshot scripts/omega.pbs)
J2=$(qsub -l walltime=12:00:00 -v "STEP=exp9-judge7,J7STEP=train,ONLY=--only qp-large-helps-strands-decider-2b-hobson-v21-A --only qpa-large-helps-strands-decider-2b-hobson-v21-A" scripts/omega.pbs)
# after J1 finishes:
J3=$(qsub -l walltime=24:00:00 -v "STEP=exp9-judge7,J7STEP=train,ONLY=--only qp-large-helps-strands-decider-e4b-gemma4-v1-2610-A --only qpa-large-helps-strands-decider-e4b-gemma4-v1-2610-A" scripts/omega.pbs)
```

- **Estimates:** about 1 h per 2B seed and 2 h per E4B seed (10.8k training questions, 2 epochs), plus about
  20 min of scoring per seed. Each seed writes its own `done.json`, so a resubmitted job resumes.
- **Then evaluate,** after all three jobs: `qsub -l walltime=02:00:00 -v STEP=exp9-judge7,J7STEP=eval scripts/omega.pbs`.
- **Check:**
  - 12 `done.json` files under `runs/system-one/judges/*strands-decider*`;
  - 3 zero-shot folders under `runs/system-one/zeroshot/`;
  - `runs/system-one/eval-decider/A/report.json` and `report-clean.json`.

## Step 4: experiment 8, five options (GPU, about 6–10 h)

```bash
qsub -l walltime=12:00:00 -v STEP=exp9-judge8 scripts/omega.pbs
```

It runs the four plans in turn: 2B zero-shot, 2B fine-tuned, E4B zero-shot, E4B fine-tuned. To split them over
two GPUs, pass `PLANS="configs/exp8-judge-decider-e4b-zs.toml configs/exp8-judge-decider-e4b-ft.toml"` and the
2B pair in a second job.

**Check:** each `runs/exp8/judge-decider-*/eval/report.json` exists and has `nano-jev qp` in `judges`. Fine-tuned
runs have 6 `done.json` files; zero-shot runs have 2.

## Step 5: copy results into git

```bash
mkdir -p results/exp9/{system-one,exp8,logs}
cp runs/exp9/smoke.json results/exp9/
rsync -a runs/system-one/eval-decider/ results/exp9/system-one/eval-decider/
rsync -a --include='*/' --include='done.json' --exclude='*' runs/system-one/judges/ results/exp9/system-one/judges/
rsync -a --include='*/' --include='done.json' --include='*.parquet' --exclude='*' runs/system-one/zeroshot/ results/exp9/system-one/zeroshot/
for d in runs/exp8/judge-decider-*; do rsync -a --exclude='*.parts/' "$d/" "results/exp9/exp8/$(basename "$d")/"; done
cp runs/logs/<every job id of steps 2-4>.log results/exp9/logs/
du -sh results/exp9    # expect a few tens of MB
```

## Step 6: write it up

Create `docs/experiments/<YYYY-MM-DD>-experiment-9.md`. Every number comes from the files above.
1. **Result bullets (3–5):**
   - H9a, H9b and H9c from the plan, each met or not met, with its interval;
   - whether a decider's AIQ, judge cost included, beats nano-jev's;
   - what the experiment 8 judge picks at its operating point.
2. **Setup:** the models, revisions (`done.json` `init_revision`), zero-shot questions, fine-tune settings
   (2 epochs, lr 1e-4, LoRA + head), and the window: the decider keeps its own 4,096-token window, while nano-jev
   cut the state at 512.
3. **Execution:** job ids, durations, exit codes, any fix, and the step checks.
4. **Results:**
   - an experiment 7 table: AIQ, AIQ with judge cost, per-seed spread, and paired differences against nano-jev qp
     (`report.json` → `aiq_diff`);
   - an experiment 8 table: the same, with the operating points and what the judge picks.
5. **Cautions:**
   - a judge costs about 60× (2B) or 120× (E4B) nano-jev per question;
   - one pair;
   - the 500-question test in experiment 8.

Commit `docs`, `results/exp9`, and nothing in `runs/`. Push. Report the smoke-test numbers, the two AIQ tables'
headline rows and the job ids to the owner.
