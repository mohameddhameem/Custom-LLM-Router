# Custom LLM Router

Cost-aware routing between question-answering models on HotpotQA (distractor setting). The
question under test: does an evidence-sufficiency signal from a small cross-encoder route better
than a matched question-only router?

## Setup

Requires Python 3.12 and [uv](https://docs.astral.sh/uv/).

```sh
uv sync
uv run prepare-hotpotqa   # writes data/hotpotqa/distractor_{train,validation}.parquet
uv run make-splits        # writes data/hotpotqa/splits.json (router_train, calib)
uv run pytest
```

`prepare-hotpotqa --sample 500` exports a random subset per split; `--jsonl` also writes JSON
Lines. `make-splits --exclude ids.txt` drops questions another component was trained on.

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
```

| Config | Scorer | Experts | Use |
|---|---|---|---|
| `configs/smoke.toml` | lexical | heuristic stand-ins, no model | checking the pipeline |
| `configs/cpu.toml` | MS MARCO reranker | Qwen2.5-0.5B (top-2) / 1.5B (all 10) | CPU pilot, tens of questions |
| `configs/gpu.toml` | nano-jev | Qwen2.5-1.5B (top-2) / 7B (all 10) | the minimal first experiment |

Routers `question` and `question+evidence` share model, labels and data; only the input
differs. `eval-routing` compares them with random, oracle and a small-model entropy threshold
on F1-vs-cost curves (AIQ), and picks an operating point on `calib` (≤1% F1 below always-large).

`HFExpert` answers one question at a time with `transformers`. That is fine for pilots; for the
full 20k-question label set, add a vLLM expert.

## Docs

- [docs/research-report.md](docs/research-report.md): the plan and the prior work behind it.
- [docs/experiment-protocol.md](docs/experiment-protocol.md): splits, leakage rules, scoring.
- [docs/reference-projects.md](docs/reference-projects.md): nano-jev and rizzo-flow.
- [docs/research-notes/](docs/research-notes/): source notes with citations. Items marked
  unverified or `[bg]` must be checked before citing.
