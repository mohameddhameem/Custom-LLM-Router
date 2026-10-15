# exp8-rescore: grader validation (FAILED)

Job 42683 (`exp8-rescore-validate`, exit 0, 3 min 37 s), grader Qwen/Qwen2.5-14B-Instruct at revision `cf98f3b3`.

| Check | Result | Required |
|---|---|---|
| Agrees with EM = 1 answers (n = 267) | 0.850 | >= 0.90 |
| Agrees with F1 = 0 answers (n = 112) | 0.795 | >= 0.90 |
| Pasted-passage control graded correct (n = 500) | 0.662 | <= 0.10 |

`passed` is false, so `exp8-rescore` and `exp8-judge-graded` were not run. Many verdicts are extreme and wrong in both
directions (identical answers with P(correct) around 1e-5; wrong answers with P around 1), see
`validation-disagreements.csv`. Cause not yet investigated; the prompt, thresholds and code are unchanged.
