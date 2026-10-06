# GPU jobs: experiments 3–5

Four omega jobs that follow up [experiment 2](experiments/2026-10-06-cascade-routers.md). Each
changes one expert and reuses everything else from experiment 1's `runs/gpu` (`REUSE=runs/gpu`),
so only one model runs per job. Cluster setup, monitoring and recovery are in
[omega-cluster.md](omega-cluster.md).

**Status:** all four ran on 2026-10-06. See [the results](experiments/2026-10-06-experiments-3-5.md).

| Exp. | Config | Run directory | What changes | GPU stage that runs | Estimated time |
|---|---|---|---|---|---|
| 3a | `configs/gpu-large-top3.toml` | `runs/gpu-large-top3` | 7B reads nano-jev's top 3 paragraphs, not all 10 | large | ~35 min |
| 3b | `configs/gpu-large-top5.toml` | `runs/gpu-large-top5` | 7B reads the top 5 | large | ~50 min |
| 4 | `configs/gpu-large-14b.toml` | `runs/gpu-large-14b` | Qwen2.5-14B replaces the 7B (all 10 paragraphs) | large | ~3 h |
| 5 | `configs/gpu-small-3b.toml` | `runs/gpu-small-3b` | Qwen2.5-3B replaces the 1.5B (top 2) | small | ~20 min |

Each estimate includes about 10 minutes of router training and bootstrap evaluation on the
job's CPUs. The times are scaled from experiment 1's per-question times by prompt length or
model size, so treat them as rough.

## Why these four

- **3a/3b. Is the large expert hurt by distractors?** In experiment 1, the oracle beats
  always-large by 0.10 F1 because the 1.5B with 2 good paragraphs often beats the 7B with all 10.
  nano-jev's top 3 and top 5 contain both gold paragraphs for 87.6% and 95.2% of validation
  questions, against 75.4% for the top 2. A 7B on the top 3 reads about 480 prompt tokens instead
  of 1,490, a third of the cost. If it also scores higher, always-large itself gets cheaper and
  better, and the router's best partner changes.
- **4. Does routing pay more when the gap is wider?** A 14B costs about twice the 7B per token.
  It shows whether the cascade routers' gain over the entropy baseline (+0.010 AIQ) grows as the
  small-large gap widens.
- **5. Does a stronger small expert remove the need to route?** A 3B on the top 2 costs about
  twice the 1.5B but is still cheap. It shows whether the small model's own signals (the
  strongest router features in experiment 2) get better or worse with model size.

All four share experiment 1's questions and splits. Once back, their caches can also be combined
on CPU, for example into a three-expert cascade: 1.5B (top 2) → 7B (top 3) → 7B or 14B (all 10).

## 1. Update the code (login node)

```bash
cd ~/Custom-LLM-Router
git fetch && git switch master && git pull     # after this branch is merged; until then:
# git switch feat/gpu-experiments-3-5 && git pull
uv sync --extra llm
ls runs/gpu/router_train.parts runs/gpu/test.parts | head    # experiment 1's chunks must still be here
```

The reuse needs experiment 1's chunk folders (`runs/gpu/*.parts/`) and `runs/gpu/config.toml`
on the cluster. If they are gone, the jobs still run, but recompute every stage (about 2.5 h
more each).

## 2. Download the new models (login node, once)

```bash
CONFIG=configs/gpu-large-14b.toml bash scripts/run_cluster.sh prepare   # Qwen2.5-14B, ~30 GB
CONFIG=configs/gpu-small-3b.toml  bash scripts/run_cluster.sh prepare   # Qwen2.5-3B, ~6 GB
```

`prepare` skips the data and splits, which already exist, and caches the models in
`~/.cache/huggingface`. The top-3 and top-5 configs use the models experiment 1 already
downloaded.

## 3. Optional: check the 14B fits (about 10 min)

```bash
qsub -v STEP=pilot,CONFIG=configs/gpu-large-14b.toml,RUN=runs/gpu-large-14b,PILOT=100 scripts/omega.pbs
```

The pilot writes to `runs/gpu-large-14b-pilot`, not to the real run. If the log shows CUDA out
of memory, lower `batch_size` under the large expert in `configs/gpu-large-14b.toml` to 2. A run
directory accepts a changed `batch_size`.

## 4. Submit the four jobs

Submit from the repo root:

```bash
qsub -v STEP=full,CONFIG=configs/gpu-large-top3.toml,RUN=runs/gpu-large-top3,REUSE=runs/gpu scripts/omega.pbs
qsub -v STEP=full,CONFIG=configs/gpu-small-3b.toml,RUN=runs/gpu-small-3b,REUSE=runs/gpu scripts/omega.pbs
qsub -v STEP=full,CONFIG=configs/gpu-large-top5.toml,RUN=runs/gpu-large-top5,REUSE=runs/gpu scripts/omega.pbs
qsub -v STEP=full,CONFIG=configs/gpu-large-14b.toml,RUN=runs/gpu-large-14b,REUSE=runs/gpu scripts/omega.pbs
```

With the 2-GPU-per-user limit, two jobs run at a time and the others wait in the queue. Every
job resumes, so if one fails, submit the same line again. For backup jobs, see
[omega-cluster.md](omega-cluster.md#recovering-from-failures).

## 5. Check each job

The log is at `runs/logs/<jobid>.log`. `qstat -u $USER` lists the job ids.

```bash
grep -- "--reuse-from" runs/logs/<jobid>.log | sort | uniq -c   # expect "copied N evidence/small chunks" (or large for exp. 5)
grep -E "large: EM|small: EM" runs/logs/<jobid>.log             # F1 of each expert per split
grep -iE "error|traceback" runs/logs/<jobid>.log                 # should print nothing
ls runs/gpu-large-top3/                                          # report.json, report-clean.json, curves*.csv
```

In each job's log, the expert that did **not** change must show `copied` for all three splits.
For experiments 3 and 4 that is the 24 + 4 + 15 = 43 evidence and small chunks; experiment 5
copies evidence and large instead. If it says `settings differ ... recomputing` for the stage
that should have been copied, the config differs from `runs/gpu/config.toml` in more than the
intended expert. Stop the job with `qdel` and compare the two configs.

## 6. Bring the results back

Copy each run directory **without** its `.parts` folders (the merged parquet files hold
everything), plus the logs:

```bash
# from your own machine, in the repo root
for r in gpu-large-top3 gpu-large-top5 gpu-large-14b gpu-small-3b; do
  rsync -av --exclude '*.parts' <user>@omega:~/Custom-LLM-Router/runs/$r runs/
done
rsync -av <user>@omega:~/Custom-LLM-Router/runs/logs runs/
```

On Windows without rsync, use WinSCP or `scp -r` and delete the `.parts` folders afterwards.
Each run directory has `config.toml`, `provenance.jsonl`, `router_train.parquet`,
`calib.parquet`, `test.parquet`, `routers.pkl`, `report.json`, `report-clean.json` and the
curves. The analysis and the experiment write-ups (one file in `experiments/` per experiment)
are then CPU-only and can be done locally.
