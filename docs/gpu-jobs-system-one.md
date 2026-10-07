# GPU jobs: experiment 7, the System One router

The omega job for [system-one-router.md](system-one-router.md). One job runs everything in
[`configs/system-one.toml`](../configs/system-one.toml):

1. **Zero-shot judges** (nano-jev's `grounded` question and a custom "is the proposed answer
   correct?" question) on pairs A, B and C.
2. **36 fine-tunings** (12 judge configurations × 3 seeds). Each judge is scored on the calib
   and test splits of all three pairs.
3. **Evaluation** of every pair, on the full and the clean test set, with logistic regression
   routers retrained for comparison and a 1,000-sample bootstrap.

**No LLM runs.** The pairs are the tracked caches in `results/gpu`, `results/gpu-large-14b` and
`results/gpu-small-3b`. Only the 33M judge uses the GPU. Cluster setup, monitoring and recovery
are in [omega-cluster.md](omega-cluster.md).

| Step | What | Estimated time |
|---|---|---|
| Pilot (optional) | One judge (`qpa-large-helps-nanojev-A`), 3 seeds | ~15 min |
| Full | Everything above. The pilot's 3 seeds are skipped as done | ~3 h |

Every step skips finished work: zero-shot pairs, seeds with a `done.json`, and so on. A failed
or killed job is simply resubmitted. The estimates come from a CPU dry run scaled to the L40S,
so treat them as rough.

Pairs A and B share the 1.5B's answers, so every judge's inputs on B are identical to A's. The
question-only and passage judges have identical inputs on all three pairs. Identical inputs are
scored once and reused, which the training records show as `reused`.

**Memory.** Training uses gradient accumulation (`micro_batch`), so the batch of 16 questions is
the same on any machine and only the memory changes:

| Machine | `micro_batch` | Peak memory |
|---|---|---|
| L40S (default) | 16 | small |
| CPU (default) | 2 | about 3.5 GB of RAM |

Without accumulation, one batch needs more than 10 GB of RAM on CPU (float32). That crashed a
15 GB WSL machine on 2026-10-07.

## 1. Update the code (login node)

```bash
cd ~/Custom-LLM-Router
git fetch && git switch master && git pull      # after the PR is merged; until then:
# git switch feat/system-one-router && git pull
uv sync --extra llm
ls results/gpu/test.parquet results/gpu-large-14b/test.parquet results/gpu-small-3b/test.parquet
ls data/hotpotqa/distractor_train.parquet data/hotpotqa/distractor_validation.parquet data/hotpotqa/nanojev_validation_ids.txt
```

The judges rebuild the small expert's passages from `data/hotpotqa/*.parquet`, the files that
experiment 1 used on omega.

## 2. Download the judge models (login node, once)

```bash
uv run python -c "from huggingface_hub import snapshot_download as d; d('sdmlai/nano-jev', revision='v1.0'); d('cross-encoder/ms-marco-MiniLM-L12-v2')"
```

nano-jev v1.0 is already cached from experiment 1. The MS MARCO MiniLM-L12 cross-encoder is the
same architecture without typed-decision pretraining, for RQ4. nano-jev's plain base cannot learn
this task (see [system-one-router.md](system-one-router.md)).

## 3. Optional pilot (about 15 min)

```bash
qsub -v STEP=judge-pilot scripts/omega.pbs
```

Its log, `runs/logs/<jobid>.log`, should show three trainings with one line per epoch, such as
`epoch 1: train loss 0.43 dev NLL 0.42`. The dev NLL should stay below about 0.45, which is
what predicting the label rate alone (about 17% `large-helps`) gives. Each seed writes
`runs/system-one/judges/qpa-large-helps-nanojev-A/seed<k>/done.json`, which includes the time it
took. Multiply that time by 36 for the full run.

## 4. Submit the full job

```bash
J1=$(qsub -v STEP=judge scripts/omega.pbs)
J2=$(qsub -W depend=afternotok:$J1 -v STEP=judge scripts/omega.pbs)   # backup, resumes
```

## 5. Check the job

```bash
grep -c "dev NLL" runs/logs/<jobid>.log                     # 36 judges x 3 epochs = 108 (fewer if the pilot ran first)
ls runs/system-one/judges/*/seed*/done.json | wc -l         # 36
ls runs/system-one/eval/{A,B,C}/report.json runs/system-one/eval/{A,B,C}/report-clean.json
grep -A 20 "^eval A:" runs/logs/<jobid>.log                 # AIQ of every judge on the main pair
grep -iE "error|traceback" runs/logs/<jobid>.log            # should print nothing
```

## 6. Bring the results back

Copy the evaluation reports, the merged judge scores and the training records. Skip the
per-seed score files and any saved models:

```bash
mkdir -p results/system-one
rsync -a --include='*/' --include='eval/**' --include='done.json' --include='plan.toml' \
      --include='provenance.jsonl' --exclude='*' runs/system-one/ results/system-one/
mkdir -p results/system-one/logs && cp runs/logs/<jobid>.log results/system-one/logs/
du -sh results/system-one                                    # about 10-15 MB
```

`results/system-one/eval/<pair>/` then holds:
- `report.json`, `report-clean.json`;
- `curves.csv`, `curves-clean.csv`;
- `scores-calib.parquet`, `scores-test.parquet`: every judge's score per question, which is
  enough to redo the evaluation on CPU.

`results/system-one/judges/<judge>/seed<k>/done.json` holds each judge's training curve, dev
NLL, timing and model revision. Commit `results/system-one/` and the analysis can continue
locally.

## Running on CPU instead

The full plan is not practical on CPU. Each training takes about 4.3 h on a 20-thread CPU, so 36
trainings take about 155 h. [`configs/system-one-cpu.toml`](../configs/system-one-cpu.toml)
is a reduced preview of about 4–5 h:

- all zero-shot judges on all three pairs;
- the RQ2 input ablation (q, qp, qpa, qa), with 1 seed, 1 epoch and 3,000 training questions.

It also previews transfer, because each judge is scored on pairs B and C too. RQ3–RQ5 and the
seed spread need the GPU plan, which should provide the final numbers.

```bash
uv run prepare-hotpotqa --out data/hotpotqa      # the current data format (context_titles / context_sentences)
uv run nanojev-ids --out data/hotpotqa
systemd-run --user --scope -p MemoryMax=8G -p MemorySwapMax=0 \
  uv run judge all --plan configs/system-one-cpu.toml --out runs/system-one-cpu --device cpu \
  --exclude data/hotpotqa/nanojev_validation_ids.txt
```

- **Memory cap:** `systemd-run ... MemoryMax=8G` kills only this command if it ever exceeds 8 GB,
  instead of letting it take WSL down. The measured peak is about 3.5 GB.
- **Resuming:** the run resumes like the GPU one. Rerun the same command after an interruption.
- **Results:** they go to `runs/system-one-cpu/`. Bring them into `results/system-one-cpu/` with
  the rsync line in step 6, replacing `system-one` with `system-one-cpu`.

## What to look at first

`report.json` → `judges`: for every judge, its AIQ, its AIQ with the judge's own cost added, the
per-seed spread and its GFLOPs per question. `bootstrap.aiq_diff` holds each judge against:
- `small-entropy (post)`;
- `random`;
- the logistic regression routers `question+evidence/large-helps` (before the small model) and
  `question+evidence+small/large-helps` (after it).

These numbers answer RQ1–RQ6 in [system-one-router.md](system-one-router.md).
