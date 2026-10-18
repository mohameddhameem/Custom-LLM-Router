# Code worth adopting from teammates' branches

Proposals only. Dhameem decides; anything adopted is a separate PR, credited to its owner.

## selfLearnt (Andy), branch `selfLearnt`, reviewed at `5080d0f` (2026-10-18)

| What | Where | Better than ours because | Proposal |
|---|---|---|---|
| Cost of sampled answers | `stair/agenttts.py`, `stair/rulebook.py` | It counts the prompt once for S samples, as AgentTTS does (App. A.2, Eq. 8). Our `pool_answer` counted it once per sample, so the 40-fused cell cost 220 units instead of 71 | **Adopted** (2026-10-18): `pool-answer recost`, the experiment 8 options reordered, and the rule written into `stair_common` (STAIR PR #6) |
| Early stopping | `stair/train.py` (`train`) | Evaluates val loss every half epoch, stops after `patience` evaluations without improvement, and keeps a resumable state (`resume_at`, saves every N questions). Ours (`judge.fine_tune`) runs a fixed number of epochs and keeps the best, with no stopping and no resume | Use it for the next judge retrain: our judges peak at epoch 1, so patience 1 would have saved the 2nd and 3rd epochs (about half of the E4B decider's 4 h) |
| Hidden-state probes | `stair/probe.py` | A logistic probe on the last-token hidden state (read through vLLM's pooling runner), 5-fold cross-fitted on val | A systemOne baseline beside small-model entropy: a probe on the 1.5B's own hidden state, against our 33M and 2B judges |
| Results export | `stair/export.py` | Validates with `stair_common.schema` before writing, one CSV per experiment and seed | Same as our `export_common`; nothing to adopt |
| Downloads offline | `configs/default.yaml` `download_models: false` | Setup on the login node, jobs read the files | Same as our `exp8-prepare` + `HF_HUB_OFFLINE=1`; nothing to adopt |

Not adopted: their own `lenient()` in `stair/common.py` ("our corrected label": EM, or F1 ≥ 0.5, or gold contained). Their
headline column comes from `stair_common.score` through `scores()`, which is right. Their `lenient()` is an internal
label, and it must never appear as "lenient" in a team table.
