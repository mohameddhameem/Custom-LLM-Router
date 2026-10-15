# exp8-rescore: grader v2 validation (FAILED on the pasted-passage control)

Job 42685 (`exp8-rescore-validate`, `-l walltime=01:00:00`, exit 0, 10 min 20 s), Qwen/Qwen2.5-14B-Instruct at
revision `cf98f3b3`, code at `55ce4a2`. Gate slice: dev items 1000-1499; dumps from the calib pools.

| Field | Result | Required | |
|---|---|---|---|
| `agree_em1_correct` (n = 269) | 0.996 | >= 0.95 | pass |
| `agree_f10_incorrect` (n = 109) | 0.917 | >= 0.80 | pass |
| `dump_graded_correct` (n = 1,000) | **0.619** | <= 0.05 | **fail** |
| `written_agreement` | 0.991 | >= 0.95 | pass |
| `verdict_mass_mean` | 0.999 | >= 0.90 | pass |

`diagnostic_not_gating` (dev 500-999, dumps from test pools): 0.993 / 0.875, dumps 0.666.

`passed` is false, so `exp8-rescore` and `exp8-judge-graded` were not run. Prompt, thresholds and code unchanged.

Reading 10 rows of `validation-disagreements.csv`: the grader is right or fair on short answers (aliases, plurals,
"UDC" for "University of the District of Columbia"). Two are grader errors: "no" against a long gold string
(P = 0.986) and an identical "Benedict of Nursia" (P = 0.38). The failure is concentrated on long, passage-style
responses, which contain the gold string.
