# exp8-rescore: grader v3 validation (FAILED: the extraction step)

Job 42689 (`exp8-rescore-validate`, `-l walltime=01:00:00`, exit 0, 10 min 20 s), Qwen/Qwen2.5-14B-Instruct at
revision `cf98f3b3`, code at `d8a2557`. Method: extract the short answer, then grade it. Gate slice: dev items
1000-1499; dumps from the calib pools.

| Field | Result | Required | |
|---|---|---|---|
| `extract_faithful` | **0.758** | >= 0.90 | **fail** |
| `agree_em1_correct` (n = 269) | **0.844** | >= 0.95 | **fail** |
| `agree_f10_incorrect` (n = 109) | 0.908 | >= 0.80 | pass |
| `dump_graded_correct` (n = 1,000) | **0.414** | <= 0.10 | **fail** |
| `written_agreement` | 0.994 | >= 0.95 | pass |
| `verdict_mass_mean` | 0.999 | >= 0.90 | pass |

Not gating: `known_extracted_none` 0.238, `dump_extracted_none` 0.522. Diagnostic slice (dev 500-999, test-pool
dumps): extract_faithful 0.748, agree_em1 0.794, dumps 0.454.

`passed` is false, so `exp8-rescore` and `exp8-judge-graded` were not run. Prompts, thresholds and code unchanged.

**Why.** The extractor returns "none" for about a quarter of already-short, correct answers (for example "Man Booker
Prize for Fiction", "David Dobkin", "Shut Up", "no"), and "none" is graded wrong. Dumps that open with a sentence
stating the gold still grade CORRECT 41% of the time.

| Version | Short answers | Pasted-passage control |
|---|---|---|
| v1 (`../README.md`) | identical answers graded wrong | 0.662 |
| v2 (`../v2/README.md`) | good (0.996 / 0.917) | 0.619 (fail) |
| v3 (this folder) | extractor drops 24% of correct short answers | 0.414 (fail) |
