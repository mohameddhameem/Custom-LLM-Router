# team-member-2: `selfLearnt*` (snapshots `dcaa60b`, `badb9ba`, `69161f5`, 2026-10-16)

**"No router: the models learn their own work by fine-tuning, and a rulebook learnt from experiments decides, per
question, when to escalate to a bigger model."** The cheapest retriever always runs; rules escalate retrieval and
answering when signals say so; every answer gets a calibrated Sure / Unsure flag.

Three branches:

- `selfLearnt`: the method and the full-scale pipeline for the SMU **Origami** cluster (Slurm, vLLM).
- `selfLearnt-colab-dev`: the development history (Google Colab, one A100, 28 Sep – 13 Oct; notebooks 00–28,
  the experiment log).
- `selfLearnt-runs`: live status page of the cluster run. At this snapshot: **"No cluster run has started yet."**

## Method

| Layer | What it is |
|---|---|
| Models | Ladders of retrievers and answerers, fine-tuned with LoRA (QLoRA for 32B / 70B / 72B, on 2,000 questions): retrievers on HotpotQA gold-passage labels, answerers on gold answers |
| Soft rules | `IF signal < threshold THEN escalate` (next size, a named size, or skip sizes), ensemble settings. Tuned by micro-experiments on 1,000 dev questions; a planner LLM proposes changes. A change is kept only if it wins on both dev halves, beats escalating the same number of random questions, and breaks no hard rule. 4 rounds, then frozen |
| Hard rules H1–H8 | Locked guardrails, e.g. H1 escalate retrieval when the first retriever's confidence is in its lowest 10%; H3 at most 60% of questions reach the largest model; H5 "Sure" needs calibrated ≥ 0.85; H8 accuracy may cost ≤ 10% extra compute per point |

Signals include retriever confidence, shortlist agreement, answer words missing from the passages, answer token
confidence, and (workstream 3) hidden-state probes.

## Five workstreams (cluster), all on AgentTTS's 500 test questions

Test set: HotpotQA dev [500:1000], gold paragraphs + top-100 BM25 each, AgentTTS's metric (gold contained in the
response) and budget formula, plus FLOPs.

| # | Workstream | Models | Purpose |
|---|---|---|---|
| 1 | AgentTTS as in its code (planner search on 50 questions, budget 900; planner Qwen2.5 72B instead of o3-mini) | Qwen2.5 7B / 32B / 72B retrievers; Llama 3.2 3B / 3.1 8B / 3.1 70B answerers | Baseline |
| 2 | Strongest models | Qwen2.5 72B → Llama 3.1 70B | Baseline |
| 3 | Hidden-state probe + calibrated rules, no fine-tuning | AgentTTS's | Savings without training? |
| 4 | The method on AgentTTS's models | same, fine-tuned | Like-for-like with 1–2 |
| 5 | The method on full ladders | retrievers from Qwen2.5 0.5B up to 72B; Llama answerers | More sizes to switch between |

Also runs AgentTTS's reported best setting (Qwen 72B ×1 → Llama 3B ×40 + fusion, 0.742) as a check. 72 Slurm jobs,
about 4.5 days on 2 GPUs. Differences from AgentTTS forced by availability: Python BM25 instead of Elasticsearch, a
4-bit AWQ 70B, their scorer file is not public, so containment is used.

## Colab development results (400 test questions, 20 candidate passages)

"Corrected label" = exact match, or token F1 ≥ 0.5, or gold contained; yes/no must match.

| Pipeline | Corrected accuracy | EM | Cost index |
|---|---|---|---|
| Untrained Qwen 7B → Qwen 32B (AgentTTS-style stand-in) | 76.8% | 57.5% | 100 |
| Fine-tuned Qwen 3B → Qwen 32B | 84.5% | 64.2% | 57.6 |
| **Fine-tuned Qwen3.5 0.8B → Qwen 32B** | **84.0%** | 63.5% | **43.5** |
| Fine-tuned 0.5B → Qwen 32B | 79.5% | 60.0% | 41.7 |
| Fine-tuned 0.8B → Llama 8B | 77.2% | 54.8% | 16.9 |
| Cascade 0.5B → 0.6B → 0.8B retrievers, Qwen 32B | 83.5% | 63.2% | 44.9 |
| Gold passages → Qwen 32B (ceiling) | 84.5% | | |

([selfLearnt/README.md](https://github.com/mohameddhameem/STAIR/blob/dcaa60b/selfLearnt/README.md),
[colab_dev/README.md](https://github.com/mohameddhameem/STAIR/blob/badb9ba/selfLearnt/colab_dev/README.md))

**What the Colab sets found** (from the archive's summary):

- Sets 1–5: per-question routers that read the question (text, topics, embeddings) were at chance. A hidden-state
  probe on Llama 8B's answer matched AgentTTS's pick at −27% compute.
- Set 3: the gap is mostly retrieval; with the right passages a 3B with chain of thought answers as well as a 7B.
- Set 7: fine-tuning the retriever is the big win, reaching the gold-passage ceiling at −42% compute; sub-1B
  retrievers do as well (set 7d).
- Set 9: learned routing collapsed to the majority label; Llama's failures could not be predicted before it answered
  (separation 0.54–0.59, where 0.5 is useless). The Sure flag is honest (91.5% right when Sure).
- Set 10: the rulebook was built and tested on saved outputs (+4.6 points over random escalation), not run end to end.
  It moved to the cluster pipeline.

## Cautions

- The Colab numbers (400 questions, 20 passages, corrected label) are **not comparable** with the cluster runs
  (500 questions, 100 passages, containment), as the member notes.
- The scorer is lenient (gold contained) for comparability with AgentTTS.
- Under about 1.5 points, differences on 400 questions are not reliable.

## Key files

- [selfLearnt/README.md](https://github.com/mohameddhameem/STAIR/blob/dcaa60b/selfLearnt/README.md)
- [pipeline/configs/default.yaml](https://github.com/mohameddhameem/STAIR/blob/dcaa60b/selfLearnt/pipeline/configs/default.yaml)
  (models, rules, guardrails); `pipeline/stair/` (agenttts.py, train.py, probe.py, rulebook.py, report.py)
- [colab_dev/docs/Experiment_Plan_and_Log.md](https://github.com/mohameddhameem/STAIR/blob/badb9ba/selfLearnt/colab_dev/docs/Experiment_Plan_and_Log.md)
  (full design and log)
- [selfLearnt-runs README](https://github.com/mohameddhameem/STAIR/blob/69161f5/README.md) (run status)
