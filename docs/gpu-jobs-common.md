# GPU job sheet: experiments 7 + 9 under the team's common setup

The team's shared setup is STAIR `common/` (`stair-common`, pinned in `pyproject.toml`):
- **Splits:** router_train = the common 12k HotpotQA train draw minus nano-jev's training questions; calib = dev[0:500];
  test = dev[500:1000], AgentTTS's 500.
- **Metric:** the standard lenient ruler.
- **Cost:** AgentTTS units.

Experiment 8 follows in a later sheet.

calib and test come from `runs/gpu`'s cache of all 7,405 dev questions. It was made with the same experts and model
revisions, which `subset-cache` checks, so only router_train needs new model calls.

Run every command from the repo root on OMEGA, after `git pull` and `uv sync --extra llm --extra decider` on the login
node. A bare `uv sync` removes torch, and the GPU jobs then fail at once. Each step resumes.

| # | Command | Where | Time | Output |
|---|---|---|---|---|
| 1 | `qsub -l walltime=00:30:00 -v STEP=common-prepare scripts/omega-cpu.pbs` | CPU | minutes | `data/hotpotqa/splits-common.json`, `runs/common/{calib,test}.parquet` |
| 2 | `qsub -l walltime=03:00:00 -v STEP=common-experts scripts/omega.pbs` | GPU | ~1 h | `runs/common/router_train.parquet` |
| 3 | `qsub -l walltime=02:00:00 -v STEP=common-judges,J=zeroshot scripts/omega.pbs` | GPU | <1 h | `runs/common-judges/zeroshot/` |
| 4a | `qsub -l walltime=03:00:00 -v "STEP=common-judges,ONLY=--only q-large-helps-nanojev-A --only qp-large-helps-nanojev-A --only qpa-large-helps-nanojev-A --only qa-large-helps-nanojev-A --only qp-small-fails-nanojev-A --only qpa-small-fails-nanojev-A --only qp-large-helps-ms-marco-MiniLM-L12-v2-A --only qpa-large-helps-ms-marco-MiniLM-L12-v2-A" scripts/omega.pbs` | GPU | ~1 h | experiment 7's judges, 3 seeds each |
| 4b | `qsub -l walltime=04:00:00 -v "STEP=common-judges,ONLY=--only qp-large-helps-strands-decider-2b-hobson-v21-A --only qpa-large-helps-strands-decider-2b-hobson-v21-A" scripts/omega.pbs` | GPU | ~2–3 h | experiment 9, 2B |
| 4c | `qsub -l walltime=08:00:00 -v "STEP=common-judges,ONLY=--only qp-large-helps-strands-decider-e4b-gemma4-v1-2610-A --only qpa-large-helps-strands-decider-e4b-gemma4-v1-2610-A" scripts/omega.pbs` | GPU | ~5–6 h | experiment 9, E4B |
| 5 | `qsub -l walltime=03:00:00 -v STEP=common-judges,PLAN=configs/system-one-common-f1.toml,JOUT=runs/common-judges-f1 scripts/omega.pbs` | GPU | ~2 h | the F1-label comparison judges |
| 6 | `qsub -l walltime=01:00:00 -v STEP=common-eval scripts/omega-cpu.pbs` | CPU | ~20 min | `runs/common-judges*/eval/A/report{,-clean}.json`, `results/common/systemOne-hotpotqa-test.csv` |

- **Order:** step 2 must finish before steps 3–5; 4a–4c and 5 can run in parallel. The weights are saved with
  `--save-model`.
- **Reports:**
  - `report.json` covers all 500 test questions.
  - `report-clean.json` drops the 62 that nano-jev saw in its own calibration and test (the standard asks for both).
  - `"metric"` and `"cost_unit"` name the ruler and the unit.
- **Cost:** the judge's own cost is included in `aiq_with_judge_cost`, in AgentTTS units.

## Experiment 8 (runs/exp8-common)

- **Test:** dev[500:1000], unchanged. Its pools, nano-jev scores, answers and grades are copied from `runs/exp8`
  (same questions, same pool seed).
- **router_train:** 1,000 questions from the common train run. This is a third of the old 3,000, the user's choice to
  keep the GPU time to about a day.
- **calib:** dev[0:500].

`E` stands for `EXP8_OUT=runs/exp8-common,PLAN=configs/exp8-common.toml,SPLITS=calib router_train`.

| # | Command | Where | Time |
|---|---|---|---|
| 8.1 | `qsub -l walltime=01:00:00 -v STEP=exp8c-pools,EXP8_TRAIN=1000 scripts/omega-cpu.pbs` (after `common-experts`) | CPU | minutes |
| 8.2 | `qsub -l walltime=01:00:00 -v "STEP=exp8-score,$E" scripts/omega.pbs` | GPU | minutes |
| 8.3 | `qsub -v "STEP=exp8-answer,$E,CELLS=<cells>" scripts/omega.pbs`, one job per cell group: `llama3b-top5 llama70b-top5` (4 h); `llama3b-s10fuse-top5` (8 h); `llama3b-s20fuse-top5` (12 h); `llama3b-s40fuse-top5` (12 h) | GPU | ~45 GPU-s a question in all |
| 8.4 | `qsub -l walltime=12:00:00 -v "STEP=exp8-rescore,EXP8_OUT=runs/exp8-common,SPLITS=calib router_train" scripts/omega.pbs` | GPU | hours |
| 8.5 | `qsub -l walltime=03:00:00 -v STEP=exp8-judge,PLAN=configs/exp8-judge-common.toml scripts/omega.pbs` (the standard ruler; doesn't need 8.4) | GPU | <1 h |
| 8.6 | `qsub -l walltime=03:00:00 -v STEP=exp8-judge-graded,PLAN=configs/exp8-judge-graded-common.toml scripts/omega.pbs` (after 8.4) | GPU | <1 h |
| 8.7 | `qsub -l walltime=10:00:00 -v "STEP=exp9-judge8,PLANS=configs/exp8-judge-decider-2b-ft-common.toml configs/exp8-judge-decider-e4b-ft-common.toml configs/exp8-judge-decider-2b-zs-common.toml configs/exp8-judge-decider-e4b-zs-common.toml" scripts/omega.pbs` (after 8.4) | GPU | hours |
