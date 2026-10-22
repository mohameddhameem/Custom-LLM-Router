# Job sheet: the HotpotQA golden set (Jonathan's SLR setup) and experiment 10

Since 2026-10-21 the HotpotQA headline follows Jonathan's Stage-Level Routing setup, through `stair_common` 0.3.0:
- the `slr-train` / `slr-val` / `slr-test` split;
- the standard ruler;
- Table 5 prices, weak-first.

Write-up: [experiments/2026-10-21-slr-standard.md](experiments/2026-10-21-slr-standard.md).

Run every command from the repo root on OMEGA, after `git pull` and `uv sync --extra llm --extra decider` on the login
node. A bare `uv sync` removes torch. Every `qsub` has a walltime; log each job ID in `runs/slr-jobs.txt`. Every step
resumes.

| # | Command | Where | Time | Output |
|---|---|---|---|---|
| S0 | `bash scripts/run_cluster.sh slr-fetch` | login node (download only) | seconds | `runs/slr/jonathan/...` (his pinned files, sha256-checked) |
| S1 | `qsub -l walltime=00:30:00 -v STEP=slr-ladder scripts/omega-cpu.pbs` | CPU | minutes | `results/slr/report.json`, `paths-{val,test}.parquet`. It fails unless the ruler, the split and his certified rows all check out |
| 10.1 | `qsub -l walltime=04:00:00 -v STEP=exp10-train,JUDGE=nano-jev scripts/omega.pbs` | GPU | ~1–2 h | `runs/exp10-slr/judge-nanojev/{qp,qpa}/seed{0,1,2}/` |
| 10.2 | `qsub -l walltime=01:00:00 -W depend=afterok:<10.1> -v STEP=exp10-eval scripts/omega-cpu.pbs` | CPU | ~10 min | `runs/exp10-slr/eval/report{,-clean}.json` |
| 10.3 | `qsub -l walltime=08:00:00 -v STEP=exp10-train,JUDGE=decider-2b scripts/omega.pbs` | GPU | ~3–5 h | `runs/exp10-slr/judge-decider-2b/` |
| 10.4 | `qsub -l walltime=20:00:00 -v STEP=exp10-train,JUDGE=decider-e4b scripts/omega.pbs` | GPU | ~8–12 h | `runs/exp10-slr/judge-decider-e4b/`. Micro-batch 2, as experiment 9's E4B: at 4 it ran out of GPU memory (43099) |
| 10.5 | rerun 10.2 after 10.3 and 10.4 | CPU | ~15 min | the reports with the deciders |

**Experiments 7 and 9 on the golden set** (`configs/system-one-slr.toml`). Pair A (Qwen2.5-1.5B on the top 2 /
Qwen2.5-7B on all 10 distractor passages) is cut from `runs/gpu`'s cache of all 7,405 dev questions, so no expert
reruns:

| # | Command | Where | Time | Output |
|---|---|---|---|---|
| 7.1 | `qsub -l walltime=00:30:00 -v STEP=slr7-prepare scripts/omega-cpu.pbs` | CPU | minutes | `data/hotpotqa/splits-slr.json`, `runs/slr-pairA/{router_train,calib,test}.parquet` |
| 7.2 | `qsub -l walltime=04:00:00 -W depend=afterok:<7.1> -v "STEP=slr7-judges,ONLY=--only q-large-helps-nanojev-A --only qp-large-helps-nanojev-A --only qpa-large-helps-nanojev-A --only qa-large-helps-nanojev-A --only qp-small-fails-nanojev-A --only qpa-small-fails-nanojev-A --only qp-large-helps-ms-marco-minilm-l12-v2-A --only qpa-large-helps-ms-marco-minilm-l12-v2-A" scripts/omega.pbs` | GPU | ~1–2 h | experiment 7's judges, 3 seeds each |
| 7.3 | `qsub -l walltime=02:00:00 -W depend=afterok:<7.1> -v STEP=slr7-judges,J=zeroshot scripts/omega.pbs` | GPU | <1 h | the zero-shot judges |
| 7.4 | `qsub -l walltime=01:00:00 -W depend=afterok:<7.2>:<7.3> -v STEP=slr7-eval scripts/omega-cpu.pbs` | CPU | ~20 min | `runs/slr-judges/eval/A/report{,-clean}.json` |
| 7.5 | `qsub -l walltime=08:00:00 -v "STEP=slr7-judges,ONLY=--only qp-large-helps-strands-decider-2b-hobson-v21-A --only qpa-large-helps-strands-decider-2b-hobson-v21-A" scripts/omega.pbs` | GPU | ~2–3 h | experiment 9, 2B (with the deciders, step 5 of the queue) |
| 7.6 | `qsub -l walltime=12:00:00 -v "STEP=slr7-judges,ONLY=--only qp-large-helps-strands-decider-e4b-gemma4-v1-2610-A --only qpa-large-helps-strands-decider-e4b-gemma4-v1-2610-A" scripts/omega.pbs` | GPU | ~5–6 h | experiment 9, E4B; then rerun 7.4 |

- **Cost (`cost = "table5"`):** each answering call pays its golden-set price: Qwen2.5-7B 10 (Table 5), and
  Qwen2.5-1.5B 0.5 (the same formula; Table 5 doesn't list it).
- **Escalation:** a judge that reads the small answer (qpa) pays both calls when it escalates, weak-first.
- **The judges' own calls:** priced with `budget()` at their measured tokens, in `aiq_with_judge_cost`.

**Experiment 8 and the H2 diagnosis on the golden set** (`configs/exp8-slr.toml`, `configs/exp8-judge-slr.toml`).
- **Split:** router_train is 1,000 of `slr-train`, calib 500 of `slr-val`, and test all 1,481 of `slr-test`.
- **Reuse:** questions already answered in `runs/exp8-common` (dev[0:1000]) are drawn first, and their answers are
  reused by ID (`reuse_answers`). Their pools rebuild exactly.
- **New questions:** 1,282 test, 348 train and 351 val, so about 1,981 per answering cell.
- **Times:** from experiment 8's logs. The fused cells take 10–17 s a question; the 70B 3.5 s on 2 GPUs.
- **GPU cap:** the GPU queue runs at most 2 GPUs per user (`max_run_res.ngpus = [u:PBS_GENERIC=2]`), on one 4-GPU
  node (a-11). So a 2-GPU job (H2, 8.7) starts only when none of our 1-GPU jobs is running.

| # | Command | Where | Time | Output |
|---|---|---|---|---|
| 8.1 | `qsub -l walltime=01:00:00 -v STEP=exp8s-pools scripts/omega-cpu.pbs` | CPU | ~20 min | `data/hotpotqa/splits-exp8-slr.json`, `runs/exp8-slr/pools/` |
| 8.2 | `qsub -l walltime=02:00:00 -W depend=afterok:<8.1> -v STEP=exp8-score,EXP8_OUT=runs/exp8-slr scripts/omega.pbs` | GPU | ~30 min | `runs/exp8-slr/scores/`, recall on test |
| H2 | `qsub -l select=1:ncpus=8:ngpus=2:mem=96gb -l walltime=05:00:00 -W depend=afterok:<8.2> -v "STEP=exp8-answer,PLAN=configs/exp8-slr.toml,SPLITS=test,CELLS=llama70b-slrprompt-top5 llama70b-slrprompt-72bsel" scripts/omega.pbs` | 2 GPUs | ~1–2 h | the 70B with his prompt on nano-jev's top 5 and on his 72B selections; `answers/summary.csv` (`lenient_std`) |
| 8.3 | `qsub -l walltime=02:00:00 -W depend=afterok:<8.2> -v "STEP=exp8-answer,PLAN=configs/exp8-slr.toml,CELLS=llama3b-top5" scripts/omega.pbs` | GPU | ~30 min | 3B, one answer |
| 8.4 | `... -l walltime=10:00:00 ... CELLS=llama3b-s10fuse-top5` | GPU | ~6 h | 3B ×10 fused |
| 8.5 | `... -l walltime=14:00:00 ... CELLS=llama3b-s20fuse-top5` | GPU | ~9 h | 3B ×20 fused |
| 8.6 | `... -l walltime=12:00:00 ... CELLS=llama3b-s40fuse-top5` | GPU | ~8 h | 3B ×40 fused |
| 8.6b | `qsub -l walltime=04:00:00 -v "STEP=exp8-answer,PLAN=configs/exp8-slr.toml,SPLITS=router_train,CELLS=llama3b-s40fuse-top5" scripts/omega.pbs` | GPU | ~1.5 h | ×40's router_train in parallel, added 2026-10-22: OMEGA's login outage stalled 8.6 by about 3 h. 8.6 reaches router_train last and skips the chunks already written |
| 8.7 | `qsub -l select=1:ncpus=8:ngpus=2:mem=96gb -l walltime=05:00:00 -W depend=afterok:<8.2> -v "STEP=exp8-answer,PLAN=configs/exp8-slr.toml,CELLS=llama70b-top5" scripts/omega.pbs` | 2 GPUs | ~2 h | 70B, one answer |
| 8.8 | `qsub -l walltime=03:00:00 -W depend=afterok:<8.3>:<8.4>:<8.5>:<8.6>:<8.7> -v STEP=exp8-judge,PLAN=configs/exp8-judge-slr.toml scripts/omega.pbs` | GPU | ~1 h | nano-jev option judges and their reports at Table 5 prices |

- **Prices** (`cost = "table5"`):
  - 3B 3; ×10 / ×20 / ×40 fused 12 / 22 / 42 (Eq. 8, Tables 6–9), plus a fusion call at 3;
  - 70B 115;
  - nano-jev's ranking call by `budget()` at its measured tokens: about 49k tokens for the 100-passage pool, 6.6 a
    question. Every option's price includes it.
- **A qpa judge** pays the cheapest option it read (fixed 2026-10-21), and every policy that runs two options pays the
  ranking once (fixed 2026-10-22, `dd5b76f`, re-evaluated by 43116).

- **What experiment 10 trains:** one typed-decision judge per (judge set, inputs, seed). Each judge answers two yes/no
  questions:
  - retrieval: do the 7B's selected passages hold what is needed?
  - answer: will the weak answer do, under the chosen retrieval?

  The labels are Jonathan's (`ret_labels`, `qa_labels`). It trains on `slr-train`, holding out 10% by question for the
  best epoch, so `slr-val` stays clean for the thresholds.
- **What the eval does:**
  - It sweeps his 9,261 thresholds on `slr-val` for every router (each seed, the seeds' mean, and his two routers),
    and takes his peak and parity points onto `slr-test`.
  - It computes AIQ over the val-Pareto thresholds, with paired bootstrap intervals.
  - It does this twice: on all 1,481 test questions, and on the 1,303 that nano-jev did not see (`report-clean.json`).
- **Router cost:** none, as in his rows (`aiq`, `price`). Beside it, each judge call is priced at its measured tokens
  (`aiq_with_judge_price`, `judge_price`). nano-jev's calls price at 0.
- **Queue order** (agreed 2026-10-21):
  1. Experiment 10, nano-jev (10.1, 10.2).
  2. Experiments 7/9 on the new split.
  3. The H2 diagnosis.
  4. Experiment 8's generation.
  5. The deciders (10.3, 10.4 and experiments 7/9's).
  6. Experiment 8's judges.

  Steps 2–6 are added to this sheet as their code lands. selfLearnt's smoke test waits until these finish.
