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
| 10.4 | `qsub -l walltime=20:00:00 -v STEP=exp10-train,JUDGE=decider-e4b scripts/omega.pbs` | GPU | ~8–12 h | `runs/exp10-slr/judge-decider-e4b/` |
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
