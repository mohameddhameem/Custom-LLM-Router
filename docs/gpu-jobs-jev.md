# GPU jobs: the Jev novelty programme (E0–E9)

The plan and hypotheses are in [the pre-registration](experiments/2026-10-24-jev-preregistration.md). Run everything
from `~/Custom-LLM-Router` on branch `feat/jev-novelty` after `uv sync --extra llm --extra decider`.
- Each `qsub` has its own walltime.
- Job IDs go to `runs/jev-jobs.txt`.
- Login-node steps only download; every other step is a PBS job.

## E1: decision pretraining × label budget (HotpotQA golden set, pair A)

| # | Command | Where | Time | Output |
|---|---|---|---|---|
| 1.1 | `bash scripts/run_cluster.sh nanojev-fetch` | login node | minutes | `runs/nanojev-pretrain/nano-jev-src` (nano-jev's recipe at `25b4655`); SQuAD 2.0, MultiNLI and the bases cached |
| 1.2 | `qsub -l walltime=02:00:00 -v STEP=nanojev-pretrain scripts/omega.pbs` | GPU | ~40 min | `runs/nanojev-pretrain/{repro,nohotpot}/model`, their `eval-own.json` / `eval-v1-test.json`, and v1.0's in `v1.0/` |
| 1.3 | gate: repro's `eval-v1-test.json` within about 0.02 of v1.0's on every decision | — | — | otherwise stop and look |
| 1.4 | `qsub -l walltime=12:00:00 -W depend=afterok:<1.2> -v "STEP=e1-judges,MATCH=qp-large" scripts/omega.pbs` | GPU | ~5 h | qp: 550 fine-tunes in `runs/e1-label-curve/judges/` |
| 1.5 | `qsub -l walltime=12:00:00 -W depend=afterok:<1.2> -v "STEP=e1-judges,MATCH=qpa-large" scripts/omega.pbs` | GPU | ~5 h | qpa: the other 550 |
| 1.6 | `qsub -l walltime=04:00:00 -W depend=afterany:<1.4>:<1.5> -v STEP=e1-curve scripts/omega-cpu.pbs` | CPU | ~1 h | `runs/e1-label-curve/label-curve/{report,report-clean}.json`, `cells*.csv`, `curves.png`: H1a–H1d |

## E2: 2WikiMultiHopQA and MuSiQue, pair A

| # | Command | Where | Time | Output |
|---|---|---|---|---|
| 2.1 | `DATASET=2wikimultihopqa bash scripts/run_cluster.sh multihop-prepare`, the same with `DATASET=musique` | login node | minutes | `data/<dataset>/{train,dev}.parquet`, `splits.json` (hashes in the pre-registration) |
| 2.2 | `qsub -l walltime=01:00:00 -v STEP=multihop-pilot,DATASET=musique scripts/omega.pbs`, then the same for `2wikimultihopqa` | GPU | ~15 min each | `runs/<dataset>-pilot/`: timing, prompt lengths |
| 2.3 | `qsub -l walltime=12:00:00 -W depend=afterok:<2.2> -v STEP=multihop-experts,DATASET=<dataset> scripts/omega.pbs` | GPU | ~4–6 h each | `runs/<dataset>-pairA/{router_train,calib,test[,test_ext]}.parquet` |

## E3: the weak model's own signals

| # | Command | Where | Time | Output |
|---|---|---|---|---|
| 3.1 | `qsub -l walltime=04:00:00 -v STEP=e3-signals,DATASET=hotpotqa scripts/omega.pbs` | GPU | ~1 h | `runs/slr-pairA/signals/{router_train,calib,test}.parquet` and `-hidden.npy` |
| 3.2 | the same with `DATASET=musique` / `2wikimultihopqa`, `-W depend=afterok:<2.3 of that dataset>` | GPU | ~1 h each | `runs/<dataset>-pairA/signals/` |

## E5: transfer across model pairs (HotpotQA golden set)

| # | Command | Where | Time | Output |
|---|---|---|---|---|
| 5.1 | `qsub -l walltime=12:00:00 -v STEP=pairD-experts scripts/omega.pbs` | GPU | ~4–6 h | `runs/gpu-pairD/test.parquet`: Llama-3.2-3B / 3.1-8B on all 7,405 dev questions (evidence copied from `runs/gpu`) |
| 5.2 | `qsub -l walltime=00:30:00 -W depend=afterok:<5.1> -v STEP=e5-prepare scripts/omega-cpu.pbs` | CPU | minutes | `runs/slr-pair{B,C,D}/` |
| 5.3 | `qsub -l walltime=04:00:00 -W depend=afterok:<5.2> -v STEP=e5-judges scripts/omega.pbs` | GPU | ~1–2 h | `runs/e5-transfer/judges/`: qp / qpa trained on each pair, scored on every pair |
| 5.4 | `qsub -l walltime=02:00:00 -W depend=afterok:<5.3> -v STEP=e5-eval scripts/omega-cpu.pbs` | CPU | ~30 min | `runs/e5-transfer/eval/<pair>/report{,-clean}.json` |

- **Budgets** are nested and stratified (`judge.budget_split`). A budget counts every label the judge sees, including
  its 10% hold-out (at least 10) that picks the epoch.
- **Small budgets** train for at least 150 optimizer steps (`min_steps`), the same for every starting point.
- **Resuming:** every fine-tune writes its own `done.json`, so a rerun skips finished ones.
