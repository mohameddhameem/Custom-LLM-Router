# Custom LLM Router

Cost-aware routing between question-answering models on HotpotQA (distractor setting). The
question under test: does an evidence-sufficiency signal from a small cross-encoder route better
than a matched question-only router?

## Setup

Requires Python 3.12 and [uv](https://docs.astral.sh/uv/).

```sh
uv sync
uv run prepare-hotpotqa   # writes data/hotpotqa/distractor_{train,validation}.parquet
uv run nanojev-ids        # writes data/hotpotqa/nanojev_{train,validation}_ids.txt
uv run make-splits --exclude data/hotpotqa/nanojev_train_ids.txt   # writes splits.json
uv run pytest
```

`prepare-hotpotqa --sample 500` exports a random subset per split; `--jsonl` also writes JSON
Lines. `nanojev-ids` lists the questions nano-jev v1.0 was trained or tuned on, so router data
and the clean test report can leave them out.

## Pipeline

Two experts (`small`, `large`) and one decision: answer with small or escalate to large. A run
directory holds cached expert outputs, trained routers and the report.

```sh
uv sync --extra llm       # torch + transformers; not needed for configs/smoke.toml
R=runs/cpu-pilot
uv run run-experts --config configs/cpu.toml --data data/hotpotqa/distractor_train.parquet \
    --split router_train --limit 50 --run $R --name router_train
uv run run-experts --config configs/cpu.toml --data data/hotpotqa/distractor_train.parquet \
    --split calib --limit 20 --run $R --name calib
uv run run-experts --config configs/cpu.toml --data data/hotpotqa/distractor_validation.parquet \
    --limit 20 --run $R --name test
uv run train-router --run $R --tau 0.8
uv run eval-routing --run $R   # writes $R/report.json and $R/curves.csv
uv run eval-routing --run $R --exclude data/hotpotqa/nanojev_validation_ids.txt   # report-clean.json
```

| Config | Scorer | Experts | Use |
|---|---|---|---|
| `configs/smoke.toml` | lexical | heuristic stand-ins, no model | checking the pipeline |
| `configs/cpu.toml` | nano-jev | Qwen2.5-0.5B (top-2) / 1.5B (all 10) | CPU pilot, tens of questions |
| `configs/gpu.toml` | nano-jev | Qwen2.5-1.5B (top-2) / 7B (all 10) | the minimal first experiment |
| `configs/t4-vllm.toml` | nano-jev | vLLM: Qwen2.5-1.5B (top-2) / 7B-AWQ (all 10) | free Colab T4 |
| `configs/t4-hf.toml` | nano-jev | transformers: 1.5B / 7B in 4-bit | T4 fallback if vLLM fails |

nano-jev is loaded from the Hub (`sdmlai/nano-jev@v1.0`) unless `scorer.path` is a local folder;
`kind = "reranker"` with an MS MARCO cross-encoder still works as a relevance-only stand-in.

Routers `question` and `question+evidence` share model, labels and data; only the input
differs. Each is trained on two labels, `small-fails` and `large-helps` (small wrong and large
right), giving four routers named `<inputs>/<label>`. `eval-routing` compares them with random,
oracle and a small-model entropy threshold on F1-vs-cost curves (AIQ), and picks an operating
point on `calib` (≤1% F1 below always-large). Cost defaults to GFLOPs, 2 × `params` × tokens,
with `params` (billions) set per expert in the config; `--cost tokens` and `--cost seconds` are
the alternatives.

`run-experts` works in stages (`--stage evidence|small|large|merge`, default all) and saves
chunks of 500 questions under `<run>/<name>.parts/`. Rerunning a command skips finished chunks.
On a GPU, run one stage per command so only one model holds GPU memory.

### Google Colab (free T4)

Open [`notebooks/colab_t4.ipynb`](notebooks/colab_t4.ipynb) in Colab, select a T4 runtime and
**Run all**. It installs into its own environment, keeps data and results on Google Drive,
runs a 500-question pilot with a time estimate, then the full run. After a disconnect, run all
again to resume.

## Docs

- [docs/research-report.md](docs/research-report.md): the plan and the prior work behind it.
- [docs/experiment-protocol.md](docs/experiment-protocol.md): splits, leakage rules, scoring.
- [docs/reference-projects.md](docs/reference-projects.md): nano-jev and rizzo-flow.
- [docs/research-notes/](docs/research-notes/): source notes with citations. Items marked
  unverified or `[bg]` must be checked before citing.
