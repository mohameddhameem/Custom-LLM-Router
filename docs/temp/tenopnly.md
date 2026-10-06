2. Download the new models on the login node:
CONFIG=configs/gpu-large-14b.toml bash scripts/run_cluster.sh prepare
CONFIG=configs/gpu-small-3b.toml bash scripts/run_cluster.sh prepare
3. Submit the jobs. The full set of four qsub lines is in the doc; here are the first two:
qsub -v STEP=full,CONFIG=configs/gpu-large-top3.toml,RUN=runs/gpu-large-top3,REUSE=runs/gpu scripts/omega.pbs
qsub -v STEP=full,CONFIG=configs/gpu-small-3b.toml,RUN=runs/gpu-small-3b,REUSE=runs/gpu scripts/omega.pbs
   With your 2-GPU limit, two run at a time and the rest wait in the queue.

Two things to check:
- runs/gpu/*.parts/ must still be on the cluster. The reuse copies from those folders. Without them every stage reruns, adding about 2.5 h per job.
- Each log should show copied lines for the models that didn't change. The doc has the exact grep commands.

Bringing results back

Copy each runs/gpu-* folder without its .parts subfolders, plus runs/logs. Everything after that runs on CPU here, including:
- comparing the large model on its top 3 against all 10 paragraphs;
- a three-model cascade: 1.5B on top 2 → 7B on top 3 → 7B or 14B on all 10.