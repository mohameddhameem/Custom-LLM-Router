# Code worth adopting from teammates' branches

Proposals only. Dhameem decides; anything adopted is a separate PR, credited to its owner.

## jonathan (Jonathan), branch `jonathan`, reviewed at `5b69cce` (2026-10-21)

Decision (2026-10-21): the team follows his Stage-Level Routing setup as the HotpotQA golden set.

| What | Where | Better than ours because | Proposal |
|---|---|---|---|
| The split | `lab_package/splits.json` | Frozen before any training, and stratified on type, level and label. Every one of its 7,405 dev questions has all four menu A paths | **Adopted:** `slr-train` / `slr-val` / `slr-test` in `stair_common` (STAIR PR #8) |
| Table 5 prices, weak-first | `lab_package/configb_test_readout.py` | Counts every call that ran, at the paper's own prices | **Adopted:** `SLR_PATH_PRICES`, `STRONG_EVERYTHING` and `budget()` in `stair_common` |
| Config B grid | `runs/config_b/` (`qa_*`, `retr_*`, `meta_*`, `router_data_standard.jsonl`) | All four menu A paths on all 7,405 questions, with measured tokens. Scored with the standard ruler: `stair_common.score.lenient` gives identical verdicts | **Reused**, read at the pinned commit: experiment 10 and the always-* ladder |
| Labels | `runs/config_b/ret_labels/`, `qa_labels/` | His rule, so that our judge against his router is like for like | **Reused** for experiment 10 |
| Sealed-test readout | `configb_test_readout.py`, `runs/config_b/test_scores/` | Thresholds frozen on val. Reproduces his three certified rows exactly from the committed files | **Reused** as the head-to-head baseline |
| Reward-shaped router training | `configb_reward_train_qa.py`, `configb_reward_train_ret.py` | A cost-aware reward on top of the labels: 75.96 at 665 against 76.70 at 835 for labels alone | A candidate for our judges after experiment 10 |

Not adopted: Config A's cost family. It is normalised to its own cheapest call, so it is neither AgentTTS units nor
Table 5 prices.

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
