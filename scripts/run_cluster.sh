#!/usr/bin/env bash
# The whole experiment on a GPU machine reached over SSH (e.g. PuTTY through the university VPN).
#
#   bash scripts/run_cluster.sh prepare   # data, splits and model downloads; needs internet (login node)
#   bash scripts/run_cluster.sh pilot     # PILOT random router_train questions, prints F1 and a time estimate
#   bash scripts/run_cluster.sh full      # router_train, calib and all 7,405 validation questions
#   bash scripts/run_cluster.sh eval      # routers and reports only (CPU, seconds)
#   bash scripts/run_cluster.sh judge-pilot  # experiment 7: one System One judge, 3 seeds (GPU, ~15 min)
#   bash scripts/run_cluster.sh judge        # experiment 7: everything in configs/system-one.toml (GPU, ~3 h)
#
# Every step resumes: after a disconnect or a killed job, run the same command again.
# An SSH disconnect kills foreground jobs, so start long steps inside tmux/screen, or with
#   nohup bash scripts/run_cluster.sh full > full.log 2>&1 &
# On omega (PBS), submit the GPU steps as jobs instead: qsub -v STEP=full scripts/omega.pbs (docs/omega-cluster.md).
# GPU nodes without internet: run `prepare` on the login node, then export HF_HUB_OFFLINE=1.
# REUSE=runs/gpu copies the stages whose settings match that run (e.g. evidence and small when only
# the large expert changed) instead of recomputing them: see run-experts --reuse-from.
set -euo pipefail

STEP=${1:?usage: run_cluster.sh prepare|pilot|full|eval|judge-pilot|judge}
CONFIG=${CONFIG:-configs/gpu.toml}
RUN=${RUN:-runs/gpu}
DATA=${DATA:-data/hotpotqa}
ROUTER_SIZE=${ROUTER_SIZE:-12000}
CALIB_SIZE=${CALIB_SIZE:-2000}
PILOT=${PILOT:-500}
TAU=${TAU:-0.8}
BIN=${BIN-uv run}  # "uv run", or BIN="" with the project venv activated
REUSE=${REUSE:-}

experts() {  # experts <run> <name> <data file> [run-experts options...]; one model on the GPU per command
  local run=$1 name=$2 data=$3
  shift 3
  for stage in evidence small large merge; do
    $BIN run-experts --config "$CONFIG" --data "$data" --run "$run" --name "$name" --stage "$stage" \
      ${REUSE:+--reuse-from "$REUSE"} "$@"
  done
}

case $STEP in
  prepare)
    [ -f "$DATA/distractor_validation.parquet" ] || $BIN prepare-hotpotqa --out "$DATA"
    [ -f "$DATA/nanojev_validation_ids.txt" ] || $BIN nanojev-ids --out "$DATA"
    [ -f "$DATA/splits.json" ] || $BIN make-splits --train "$DATA/distractor_train.parquet" \
      --exclude "$DATA/nanojev_train_ids.txt" --router-size "$ROUTER_SIZE" --calib-size "$CALIB_SIZE" \
      --out "$DATA/splits.json"
    $BIN prefetch-models --config "$CONFIG"
    ;;
  pilot)
    experts "$RUN-pilot" pilot "$DATA/distractor_train.parquet" \
      --splits "$DATA/splits.json" --split router_train --limit "$PILOT"
    RUN_DIR="$RUN-pilot" N=$((ROUTER_SIZE + CALIB_SIZE + 7405)) $BIN python - <<'EOF'
import os
import pandas as pd

df = pd.read_parquet(os.path.join(os.environ["RUN_DIR"], "pilot.parquet"))
print(df[["answer", "small_pred", "large_pred"]].head(15).to_string())
n = int(os.environ["N"])
hours = n * (df["small_seconds"].mean() + df["large_seconds"].mean()) / 3600
print(f"\nsmall F1 {df.small_f1.mean():.3f} | large F1 {df.large_f1.mean():.3f} | "
      f"gold pair in top-2 {df.gold_pair_in_top2.mean():.3f}")
print(f"estimated GPU time for all {n:,} questions: {hours:.1f} h")
EOF
    ;;
  full)
    experts "$RUN" router_train "$DATA/distractor_train.parquet" --splits "$DATA/splits.json" --split router_train
    experts "$RUN" calib "$DATA/distractor_train.parquet" --splits "$DATA/splits.json" --split calib
    experts "$RUN" test "$DATA/distractor_validation.parquet"
    bash "$0" eval
    ;;
  eval)
    $BIN train-router --run "$RUN" --tau "$TAU"
    $BIN eval-routing --run "$RUN"
    $BIN eval-routing --run "$RUN" --exclude "$DATA/nanojev_validation_ids.txt"
    ;;
  judge-pilot)
    $BIN judge train --plan "${PLAN:-configs/system-one.toml}" --data-dir "$DATA" --out "${JUDGE_OUT:-runs/system-one}" \
      --only qpa-large-helps-nanojev-A
    ;;
  judge)
    $BIN judge all --plan "${PLAN:-configs/system-one.toml}" --data-dir "$DATA" --out "${JUDGE_OUT:-runs/system-one}" \
      --exclude "$DATA/nanojev_validation_ids.txt"
    ;;
  *)
    echo "unknown step: $STEP (prepare|pilot|full|eval|judge-pilot|judge)" >&2
    exit 2
    ;;
esac
