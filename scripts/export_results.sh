#!/usr/bin/env bash
# Copy a finished run into the git-tracked results/<run name>/ folder:
#
#   bash scripts/export_results.sh runs/gpu-large-top3 runs/logs/42217.log
#
# Copies the reports, curves, per-question caches (*.parquet), config and provenance, plus any job
# logs given after the run directory. Skips routers.pkl (train-router rebuilds it from the caches in
# minutes, and pickles are unsafe to load from a shared repo) and the *.parts chunk folders.
set -euo pipefail

RUN=${1:?usage: export_results.sh <run dir> [job log ...]}
shift
OUT=results/$(basename "$RUN")

mkdir -p "$OUT"
find "$RUN" -maxdepth 1 -type f ! -name routers.pkl -exec cp {} "$OUT"/ \;
if [ $# -gt 0 ]; then
  mkdir -p "$OUT/logs"
  cp "$@" "$OUT/logs/"
fi
du -sh "$OUT"
