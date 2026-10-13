# GPU jobs: experiment 8, phase 1

**Status:** phase 1 ran on 2026-10-12 to 2026-10-14 (seven jobs, all exit 0; the 70B needs 2 GPUs).
Results: [experiment-8-plan.md](experiment-8-plan.md#phase-1-results-2026-10-12-to-2026-10-14), files in
`results/exp8-phase1/`.

The omega jobs for phase 1 of [experiment-8-plan.md](experiment-8-plan.md). Phase 0 found that nano-jev's
top 2 picks AgentTTS's gold passages better than its 7B and 32B LLM retrievers (retrieval F1 0.74, at 2.9
AgentTTS units per question). Phase 1 answers from nano-jev's passages with AgentTTS's answer models and
prompts. The answering cells are in [`configs/exp8.toml`](../configs/exp8.toml):

| Cell | Model | Passages | Samples | AgentTTS's equivalent |
|---|---|---|---|---|
| `llama3b-top3`, `llama3b-top5` | Llama-3.2-3B-Instruct | nano-jev top 3 / top 5 | 1, temperature 0.3 | weak answerer |
| `llama3b-s10fuse-top5` | Llama-3.2-3B-Instruct | top 5 | 10 at 0.9, then one fusion call at 0.3 | its repeated sampling + fusion (it used 40) |
| `llama70b-top3`, `llama70b-top5` | Llama-3.1-70B-Instruct, 4-bit NF4 | top 3 / top 5 | 1, temperature 0.3 | strong answerer |

**Splits:**
- `test`: AgentTTS's 500 test questions, dev items 500–999.
- `calib` (1,000) and `router_train` (3,000): from HotpotQA train, for phase 2's System One judge.

Everything resumes from finished chunks, so a failed job is simply resubmitted.

## 0. Prerequisites (login node, once)

1. **Llama licences.** On huggingface.co, accept the licences for `meta-llama/Llama-3.2-3B-Instruct` and
   `meta-llama/Llama-3.1-70B-Instruct`. Then log in on omega with a token:

   ```bash
   uv run hf auth login        # older huggingface_hub: uv run huggingface-cli login
   ```

   Or put a read token in `~/.bashrc` as `export HF_TOKEN="hf_..."` (with `export`, or subprocesses do not
   see it). Hugging Face rejects a revoked or mistyped token with "Invalid user token", so check it first:
   `uv run python -c "from huggingface_hub import HfApi; print(HfApi().whoami()['name'])"`.
   The download skips each repo's `original/` folder, a second copy of the weights (141 GB for the 70B) that
   transformers does not load; the 70B needs about 141 GB and the 3B about 13 GB. If a download is interrupted, rerun
   `exp8-prepare`: it resumes. To clean up afterwards, delete `*.incomplete` files in the model's `blobs/` folder.
   The token is only needed on the login node, to download. The PBS jobs run with `HF_HUB_OFFLINE=1`
   (set in `scripts/omega.pbs` for the `exp8-*` steps) and read the cache, so the token never reaches a
   compute node or `qstat -f`.

2. **Code, pools and models:**

   ```bash
   cd ~/Custom-LLM-Router
   git fetch && git switch master && git pull   # after the PR is merged; until then: git switch feat/experiment-8-phase0
   uv sync --extra llm                           # now also installs bitsandbytes and accelerate
   bash scripts/run_cluster.sh exp8-prepare      # pools for test/calib/router_train (~3 min), nano-jev ids, models
   ```

   The pools rebuild exactly as in phase 0 (seeded). The 70B download is about 141 GB, and the
   Hugging Face cache needs room for it (`HF_HOME`).

## 1. nano-jev scores (GPU, minutes)

```bash
qsub -v STEP=exp8-score scripts/omega.pbs
```

This scores 4,500 pools of ~100 candidates and rewrites `runs/exp8/recall-test.json`. The CPU run took
12.4 s per question. Check that the GPU recall matches phase 0 (`results/exp8-phase0/recall-test.json`,
top 2: 0.544, top 5: 0.772) to within float noise.

## 2. Pilot (GPU, about 30–60 min)

```bash
qsub -v STEP=exp8-pilot scripts/omega.pbs
```

This runs every cell on the first 20 test questions (into `runs/exp8/answers/test-limit20/`) and prints a
summary table. Check three things:

- **No CUDA out-of-memory for the 70B.** If there is one, lower its `batch_size` to 2, or give it 2 GPUs (below).
- **Answers look like answers:** `uv run python -c "import pandas as pd; print(pd.read_parquet('runs/exp8/answers/test-limit20/llama70b-top5.parquet')[['pred','lenient']].head())"`.
- **Speed:** `s_per_q` per cell × 4,500 questions gives each cell's GPU time for step 3.

## 3. Answer runs (GPU)

Submit one job per cell, so that two run at once (2 GPUs per user):

```bash
for c in llama3b-top5 llama3b-top3 llama3b-s10fuse-top5 llama70b-top5 llama70b-top3; do
  qsub -v STEP=exp8-answer,CELLS=$c scripts/omega.pbs
done
```

- **Order:** the 3B cells take minutes to about 2 h. The 70B cells are the long ones, an estimated
  3–5 h each; the pilot gives the real figure.
- **70B on two GPUs**, if one L40S runs out of memory: `qsub -l select=1:ncpus=8:ngpus=2:mem=96gb -v STEP=exp8-answer,CELLS=llama70b-top5 scripts/omega.pbs`.
  The model then spreads over both GPUs by itself.
- **Test only first:** add `SPLITS=test` to run only the 500 test questions, for a quick look at the
  head-to-head with AgentTTS.

Every job ends with `pool-answer summary`, which writes `runs/exp8/answers/summary.csv`. Per split and
cell it holds lenient accuracy, EM/F1, cost in AgentTTS units and its % of AgentTTS's strong-everything
budget (929), plus an oracle over the cells.

## 4. Check and bring back

```bash
grep -iE "error|traceback|out of memory" runs/logs/<jobid>.log      # should print nothing
column -s, -t runs/exp8/answers/summary.csv | head -30
mkdir -p results/exp8-phase1
cp runs/exp8/answers/summary.csv runs/exp8/recall-test.json runs/exp8/provenance.jsonl results/exp8-phase1/
rsync -a --exclude='*.parts/' --include='*/' --include='*.parquet' --exclude='*' runs/exp8/answers/ results/exp8-phase1/answers/
cp runs/logs/<jobid>.log results/exp8-phase1/   # every phase-1 job log
```

The per-question answer files (one parquet per split and cell, with samples) are what phase 2 trains the
judge on. They are about 10–50 MB in total. The pools themselves are not copied: they rebuild
identically.

## Known differences from AgentTTS's runs

- **Retrieval:** nano-jev top-k replaces the LLM retriever, which is the point of the experiment.
- **Fusion:** it sees only the answerer's prompt (question + top-k passages) and the samples. AgentTTS's
  fuser repeats the whole 100-candidate user message twice.
- **Samples:** 10 rather than AgentTTS's 40, to keep the GPU time reasonable. More can be added later as
  another cell.
- **The 70B is 4-bit NF4.** The team's reproduction used AWQ for the 72B retriever. Costs count full
  parameters, as the paper's Table 5 does.
- **Lenient scoring** is a placeholder (gold answer contained in the response, whole words) until the
  team's scorer is shared.

## 5. Phase 2: the option-choosing System One judge (GPU, under an hour)

**Status:** ran on 2026-10-14 (job 42609, 6 min 38 s, exit 0). Results: [the write-up](experiments/2026-10-14-experiment-8.md)
and `results/exp8-phase2/`.

After step 4, with all of `llama3b-top5`, `llama3b-s10fuse-top5` and `llama70b-top5` answered on all three splits:

```bash
qsub -v STEP=exp8-judge scripts/omega.pbs
```

This runs `exp8-judge all` ([`configs/exp8-judge.toml`](../configs/exp8-judge.toml)): 2 judge inputs (`qp`, `qpa`) × 3
seeds, trained on `router_train`, then the evaluation on `calib` and `test` with the baselines (fixed-policy hull, oracle,
entropy threshold, logistic regression) and a 1,000-sample paired bootstrap. It writes
`runs/exp8/judge/{qp,qpa}/seed<k>/`, `runs/exp8/judge/eval/report.json`, `report-clean.json` and `curves.csv`. To redo only
the evaluation, on CPU: `uv run exp8-judge eval --plan configs/exp8-judge.toml --exclude runs/exp8/nanojev_validation_ids.txt`.

The judge picks the cheapest option whose cumulative probability reaches a threshold `t`, so one judge gives a whole
cost-accuracy curve. Its label is the cheapest option that answered correctly (the cheapest option when none did).
Bring `runs/exp8/judge/` back without the per-seed score files you do not need: `report*.json`, `curves*.csv`,
`*/seed*/done.json` and `*/seed*/{calib,test}.parquet`.

