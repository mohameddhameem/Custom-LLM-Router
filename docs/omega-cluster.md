# Running the experiment on the omega cluster

How to run `scripts/run_cluster.sh` on omega's GPU node. Everything below was checked on
2026-10-05 with test jobs on the `GPU` queue.

## What the cluster has

| | |
|---|---|
| Login node | `omega`: no GPU. Use it to edit, submit and download only |
| Scheduler | **PBS Pro** (`qsub`, `qstat`, `qdel`, `pbsnodes`). SLURM is not in use, so `sbatch` will not work |
| GPU node | `a-11` only: **4 × NVIDIA L40S, 46 GB each**, 96 CPUs, 503 GB RAM. Shared with other users |
| GPU queue | `GPU`. **At most 2 GPUs per user** at a time. No walltime cap is configured on the queue |
| CPU queues | `short` (default), `long`, `g4dn-8xl-short`, `g4dn-8xl-long`. Nodes `a-4`…`a-10`, no GPUs |
| Driver / CUDA | Driver 555.58, **CUDA 12.5**, compute capability 8.9 (Ada: bfloat16 works) |
| Storage | Home (`/storage/home/<user>`) is shared with every node, 12 TB free. `a-11` also has local `/local` |
| Internet | Available on the GPU node too, so `HF_HUB_OFFLINE` is not needed |
| Modules | `cuda/12.5`, `python/3.11.4`, `python/3.9.13`, `apptainer/1.3.3` (the project uses `uv` instead) |

An L40S has 46 GB, so `configs/gpu.toml` and `configs/gpu-vllm.toml` (Qwen2.5-1.5B and 7B in
bfloat16) both fit on a single GPU. One GPU per job is enough, because the pipeline loads one model at a time.

## One-time setup

```bash
cd ~/Custom-LLM-Router
uv sync --extra llm
```

`pyproject.toml` pins `torch` to the PyTorch **cu126** index. The default wheels are built for
CUDA 13, which needs driver ≥ 580. On omega's driver 555 they print *"The NVIDIA driver on your
system is too old"* and `torch.cuda.is_available()` returns `False`. The cu126 build (`torch
2.14.1+cu126`) runs on the 12.5 driver through CUDA minor-version compatibility. Don't install
torch any other way, because a plain `pip install torch` brings back a CUDA 13 build.

**vLLM** (`configs/gpu-vllm.toml`) is not a project dependency, and it has not been tested on
omega. Its wheels pin their own torch and CUDA version. Use `configs/gpu.toml` (transformers),
which is the default.

To check the GPU setup alone, save this as `gpucheck.pbs` and run `qsub gpucheck.pbs`:

```bash
#!/bin/bash
#PBS -N gpucheck
#PBS -q GPU
#PBS -l select=1:ncpus=2:ngpus=1:mem=8gb
#PBS -l walltime=00:05:00
#PBS -j oe
cd "$PBS_O_WORKDIR"
nvidia-smi --query-gpu=name,memory.total --format=csv
.venv/bin/python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

The output file `gpucheck.o<jobid>` should end with `2.14.1+cu126 True`.

## Running the pipeline

Steps are the same as in the README: `prepare` → `pilot` → `full` (`eval` runs inside `full`).
Every step resumes, so if a job dies or runs out of walltime, submit it again.

1. **Prepare** (data, splits, model downloads; CPU and network only). Run it on the login node:

   ```bash
   cd ~/Custom-LLM-Router
   bash scripts/run_cluster.sh prepare
   ```

   Models go to `~/.cache/huggingface` (shared with `a-11`). Set `HF_HOME` to put them elsewhere.

2. **Job script**: [`scripts/omega.pbs`](../scripts/omega.pbs) requests one L40S, 8 CPUs, 64 GB
   and 24 h. It stops early if torch cannot use the GPU, then runs
   `scripts/run_cluster.sh $STEP`. The log is written live to `runs/logs/<jobid>.log`. PBS passes
   no environment variables by default: pass `STEP` (and `CONFIG`, `RUN`, `PILOT` and so on, if
   needed) with `-v`. Always submit from the repo root.

3. **Pilot**: 500 questions. It prints both experts' F1 and an estimate of total GPU hours:

   ```bash
   qsub -v STEP=pilot scripts/omega.pbs
   ```

   Use the estimate to choose the walltime for the full run.

4. **Full run**:

   ```bash
   qsub -v STEP=full scripts/omega.pbs
   qsub -v STEP=full,CONFIG=configs/gpu-vllm.toml,RUN=runs/gpu-vllm scripts/omega.pbs   # once vLLM works
   ```

   With the 2-GPU limit you can run two independent jobs at once, for example the `gpu.toml`
   and `gpu-vllm.toml` runs (use different `RUN` directories). Do not start two jobs on the same
   `RUN`, because both would write the same cache files.

5. **Eval only** (CPU, seconds): `bash scripts/run_cluster.sh eval` on the login node.
   `full` already runs it at the end.

## Watching and managing jobs

| | |
|---|---|
| `qstat -u $USER` | your jobs (`Q` queued, `R` running) |
| `qstat -x -u $USER` | also finished jobs (`F`) |
| `qstat -f <jobid>` | full detail: node, resources used, exit status |
| `tail -f runs/logs/<jobid>.log` | live log of a `scripts/omega.pbs` job |
| `qdel <jobid>` | cancel |
| `pbsnodes -aSj` | free CPUs and GPUs per node (`ngpus f/t` on `a-11`) |
| `qsub -I -q GPU -l select=1:ncpus=4:ngpus=1:mem=32gb -l walltime=01:00:00` | interactive GPU shell for debugging |

A submitted job keeps running if your SSH (PuTTY) session drops. You don't need `tmux` or
`nohup` with `qsub`, only for interactive sessions.

## Etiquette

`a-11` is the only GPU node and other people use it. Request one GPU per job and a walltime
close to the pilot's estimate. Use `qdel` on jobs you no longer need, and don't run GPU work on
the login node.
