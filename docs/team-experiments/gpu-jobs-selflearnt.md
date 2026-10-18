# selfLearnt (AL-01 to AL-06) on OMEGA

Andy's pipeline is written for SLURM (SMU Origami). Here it runs one stage per PBS job through
[`scripts/omega-selflearnt.pbs`](../../scripts/omega-selflearnt.pbs). The queue order is your choice of 2026-10-18:
1. finish experiment 8;
2. the smoke test;
3. AL-02, then AL-01, at seed 0;
4. AL-03 to AL-06 at seed 0;
5. seeds 1–2 only for the headline method.

**Wait for experiment 8's jobs to finish before submitting anything here** (the per-user limit is 2 GPUs).

Rules, as for our own jobs:
- downloads only on the login node;
- every `qsub` has a walltime;
- never push to Andy's branch: the job sheet sets `STAIR_STATUS_PUSH=0`;
- no token in `qsub -v`.

## 0. Once, on the login node (downloads only)

```bash
mkdir -p ~/team/AL && cd ~/team/AL
git clone https://github.com/mohameddhameem/STAIR.git && cd STAIR && git switch selfLearnt
git checkout <code.commit from the entries>      # bd00667 now; ask Andy to bump it (see below)
cd selfLearnt/pipeline
~/.local/bin/uv venv ~/team/AL/venv --python 3.11
VIRTUAL_ENV=~/team/AL/venv ~/.local/bin/uv pip install "vllm>=0.10.2"   # first: it brings its own PyTorch/CUDA
VIRTUAL_ENV=~/team/AL/venv ~/.local/bin/uv pip install -r requirements.txt -e ../../common
printf 'include: smoke.yaml\ndownload_models: false\n' > configs/omega-smoke.yaml     # local only, never committed
printf 'include: default.yaml\ndownload_models: false\n' > configs/omega.yaml
export STAIR_PROJECT_DIR=~/team/AL/data STAIR_STATUS_PUSH=0
~/team/AL/venv/bin/python run.py --config configs/smoke.yaml --stage setup         # the smoke run's models + HotpotQA
```

The full menu's `setup` (`--config configs/default.yaml`) downloads about 700 GB into `~/team/AL/data`, and
`/storage` has 12 TB free. Several models are already in our Hugging Face cache at the pinned revisions:
- Llama-3.1-70B-Instruct (132 GB: AL-04's QLoRA copy);
- Llama-3.2-3B;
- Qwen2.5-7B.

Linking those snapshots into `~/team/AL/data/models/<name>` before `setup` would save about 160 GB of downloads.

## 1. Smoke test (1 GPU)

```bash
qsub -l walltime=04:00:00 -v "CONFIG=configs/omega-smoke.yaml,STAGE=all" scripts/omega-selflearnt.pbs
```

It runs every stage of AL-01, AL-02, AL-03 and AL-05 on small models and a few questions. It passes when the log
ends with the results tables written. Then validate them with `stair_common.schema.validate` and `validate_history`,
as `jobs/smoke.sbatch` does. It also tells us whether vLLM's wheels (CUDA 12.8) run on OMEGA's driver 555
(CUDA 12.5).

## 2. AL-02, then AL-01, at seed 0 (2 GPUs)

`G2` stands for `-l select=1:ncpus=8:ngpus=2:mem=120gb`. Every stage resumes, so a job that hits its walltime is
resubmitted unchanged.

| Step | Command | Time |
|---|---|---|
| setup (login node) | `run.py --config configs/default.yaml --stage setup` | download |
| data | `qsub -q short -l walltime=02:00:00 -l select=1:ncpus=8:mem=32gb -v "CONFIG=configs/omega.yaml,EXP=AL-02,SEED=0,STAGE=data" scripts/omega-selflearnt.pbs` (BM25 candidates; CPU) | <2 h |
| AL-02 test | `qsub $G2 -l walltime=06:00:00 -v "CONFIG=configs/omega.yaml,EXP=AL-02,SEED=0,STAGE=ws1_test" scripts/omega-selflearnt.pbs` | ~4 h |
| AL-01 grid | `qsub $G2 -l walltime=24:00:00 -v "CONFIG=configs/omega.yaml,EXP=AL-01,SEED=0,STAGE=ws1_grid" scripts/omega-selflearnt.pbs` (or one job per `PART=retrieve-q7`, `answer-l70`, …) | the longest |
| AL-01 search | `... EXP=AL-01,SEED=0,STAGE=ws1_search ...` with `$G2`, 6 h | planner once |
| AL-01 test | `... EXP=AL-01,SEED=0,STAGE=ws1_test ...` with `$G2`, 8 h | |
| export | `qsub -q short -l walltime=00:30:00 -l select=1:ncpus=2:mem=8gb -v "CONFIG=configs/omega.yaml,EXP=AL-01,SEED=0,STAGE=export" scripts/omega-selflearnt.pbs` | minutes |

Log the job IDs in `runs/common-jobs.txt`, and the status in the `SHEET.md` table.

## Questions for Andy (before step 1)

- **The commit:** the entries pin `bd00667`. The branch has about ten cluster fixes since then (up to `5080d0f`:
  FlashInfer sampler, HF token, dataset fallback). Bump `code.commit` to the commit you want run.
- **Epochs:** AL-04 to AL-06 say up to 2–3 epochs. TRAINING.md's recipe is 1 pass for 1.5B and up (2 for 1B and
  smaller), and systemOne's judges all peaked at epoch 1. Your patience stops early anyway. Would `max_epochs` at
  the recipe (1, or 2 for 1B and smaller) be acceptable? It would roughly halve the ~55 h of training.
- **Seeds:** seed 0 for every entry first, and seeds 1–2 only for the headline method (AL-04 or AL-05). With a 2-GPU
  limit, three seeds of everything is about 4 weeks of cluster time.
- **Cost rule:** confirmed. Samples share their prompt (AgentTTS Eq. 8); `stair_common` now says so (PR #6).
- **Training copies:** pinned in `models.toml [training.qlora]` (PR #6). The 70B one is already on OMEGA.
- **nano-jev's 66 val questions:** AL-05 tunes on val, so report AL-05 also on the 438 unseen test questions.
