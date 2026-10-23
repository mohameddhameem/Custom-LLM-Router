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

## E2: 2WikiMultiHopQA and MuSiQue, pair A

| # | Command | Where | Time | Output |
|---|---|---|---|---|
| 2.1 | `DATASET=2wikimultihopqa bash scripts/run_cluster.sh multihop-prepare`, the same with `DATASET=musique` | login node | minutes | `data/<dataset>/{train,dev}.parquet`, `splits.json` (hashes in the pre-registration) |
| 2.2 | `qsub -l walltime=01:00:00 -v STEP=multihop-pilot,DATASET=musique scripts/omega.pbs`, then the same for `2wikimultihopqa` | GPU | ~15 min each | `runs/<dataset>-pilot/`: timing, prompt lengths |
| 2.3 | `qsub -l walltime=12:00:00 -W depend=afterok:<2.2> -v STEP=multihop-experts,DATASET=<dataset> scripts/omega.pbs` | GPU | ~4–6 h each | `runs/<dataset>-pairA/{router_train,calib,test[,test_ext]}.parquet` |

- **Budgets** are nested and stratified (`judge.budget_split`). A budget counts every label the judge sees, including
  its 10% hold-out (at least 10) that picks the epoch.
- **Small budgets** train for at least 150 optimizer steps (`min_steps`), the same for every starting point.
- **Resuming:** every fine-tune writes its own `done.json`, so a rerun skips finished ones.
