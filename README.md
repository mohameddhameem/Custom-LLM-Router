# Custom LLM Router

Cost-aware routing between question-answering models on HotpotQA (distractor setting). The
question under test: does an evidence-sufficiency signal from a small cross-encoder route better
than a matched question-only router?

## Setup

Requires Python 3.12 and [uv](https://docs.astral.sh/uv/).

```sh
uv sync
uv run prepare-hotpotqa   # writes data/hotpotqa/distractor_{train,validation}.{json,parquet}
```

Set `SAMPLE` in `src/groupproject/prepare_hotpotqa.py` to export a random subset per split.

## Docs

- [docs/research-report.md](docs/research-report.md): the plan and the prior work behind it.
- [docs/experiment-protocol.md](docs/experiment-protocol.md): splits, leakage rules, scoring.
- [docs/reference-projects.md](docs/reference-projects.md): nano-jev and rizzo-flow.
- [docs/research-notes/](docs/research-notes/): source notes with citations. Items marked
  unverified or `[bg]` must be checked before citing.
