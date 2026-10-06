# Results

Outputs of every executed run, copied from `runs/` (which git ignores) with
`scripts/export_results.sh`. The write-ups are in [docs/experiments/](../docs/experiments/) and the
headline numbers are in [docs/experiments.md](../docs/experiments.md).

| Folder | Experiment | Config |
|---|---|---|
| `gpu` | 1: 1.5B (top 2) / 7B (all 10) | `configs/gpu.toml` |
| `gpu-cascade` | 2: cascade routers, bootstrap, τ sweep on experiment 1's caches | `configs/gpu.toml` |
| `gpu-large-top3` | 3a: 7B reads the top 3 | `configs/gpu-large-top3.toml` |
| `gpu-large-top5` | 3b: 7B reads the top 5 | `configs/gpu-large-top5.toml` |
| `gpu-large-14b` | 4: 14B replaces the 7B | `configs/gpu-large-14b.toml` |
| `gpu-small-3b` | 5: 3B replaces the 1.5B | `configs/gpu-small-3b.toml` |
| `cascade` | 6: multi-stage cascades over the caches above (CPU) | `cascade` command, see `provenance.jsonl` |

Each folder has:

| File | Contents |
|---|---|
| `report.json`, `report-clean.json` | All metrics on the 7,405 test questions, and on the 6,405 without nano-jev's validation questions |
| `curves.csv`, `curves-clean.csv` | F1-vs-cost curve points for every router and baseline |
| `router_train.parquet`, `calib.parquet`, `test.parquet` | Per-question caches: predictions, F1/EM, token counts, entropies, nano-jev scores |
| `config.toml`, `provenance.jsonl` | The exact config, and every command with its git commit, package versions and model revisions |
| `logs/` | The PBS job log |
| `tau-sweep.csv`, `tau-sweep.json` | `gpu-cascade` only |
| `run.log` | `cascade` only; it has no caches, config or `routers.pkl` of its own |

`routers.pkl` is left out: it is about 8 MB, loading a shared pickle is unsafe, and the caches
rebuild it in minutes. To rerun any CPU analysis, copy a folder back into `runs/`:

```bash
cp -r results/gpu-large-top5 runs/
uv run train-router --run runs/gpu-large-top5 --tau 0.8
uv run eval-routing --run runs/gpu-large-top5
```
