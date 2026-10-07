1. Update the code (login node)

cd ~/Custom-LLM-Router
git fetch && git switch master && git pull
uv sync --extra llm
ls results/gpu/test.parquet results/gpu-large-14b/test.parquet results/gpu-small-3b/test.parquet
ls data/hotpotqa/distractor_train.parquet data/hotpotqa/distractor_validation.parquet data/hotpotqa/nanojev_validation_ids.txt
All of these files should already exist on omega from your earlier runs.

2. Download the two judge models (login node, once)

uv run python -c "from huggingface_hub import snapshot_download as d; d('sdmlai/nano-jev', revision='v1.0'); d('cross-encoder/ms-marco-MiniLM-L12-v2')"

3. Run a pilot first (recommended, about 15 min)

qsub -v STEP=judge-pilot scripts/omega.pbs
This trains one judge (qpa-large-helps-nanojev-A) with 3 seeds. Then check the log:
grep "dev NLL" runs/logs/<jobid>.log            # 3 seeds × 3 epochs; dev NLL should fall below ~0.45
cat runs/system-one/judges/qpa-large-helps-nanojev-A/seed0/done.json | grep train_seconds
train_seconds × 36 gives the real time for the full run. The full run skips these pilot seeds.

4. Submit the full job (about 3 h), with a backup that resumes if it fails

J1=$(qsub -v STEP=judge scripts/omega.pbs)
qsub -W depend=afternotok:$J1 -v STEP=judge scripts/omega.pbs

5. Check it finished

ls runs/system-one/judges/*/seed*/done.json | wc -l                          # 36
ls runs/system-one/eval/{A,B,C}/report.json runs/system-one/eval/{A,B,C}/report-clean.json
grep -iE "error|traceback" runs/logs/<jobid>.log                              # should print nothing
grep -A 25 "^eval A:" runs/logs/<jobid>.log                                   # AIQ of every judge, main pair

6. Bring the results back, about 10–15 MB

mkdir -p results/system-one/logs
rsync -a --include='*/' --include='eval/**' --include='done.json' --include='plan.toml' \
      --include='provenance.jsonl' --exclude='*' runs/system-one/ results/system-one/
cp runs/logs/<pilot-jobid>.log runs/logs/<full-jobid>.log results/system-one/logs/
git add results/system-one && git commit -m "Run experiment 7 (System One router) on omega" && git push