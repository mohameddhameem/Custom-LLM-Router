# team-member-3: `jonathan/` on `main` (snapshot `e3ff28b`, 2026-10-17)

GitHub handle `dohyungjahn-alt`. Their work is a folder on `main`, not a branch. They describe it as "Stage-Level
Routing (RightRoute)".

- `final/` holds validated scripts: "each file was reviewed line-by-line before running and ties to reported
  numbers".
- `wip/` holds work in progress, not to be used for reported numbers. It is empty at this snapshot.

([jonathan/README.md](https://github.com/mohameddhameem/STAIR/blob/e3ff28b/jonathan/README.md))

## The team's standard scorer: `final/lenient_match_agenttts.py`

The file is called the "standard answer-scoring ruler behind ALL headline numbers (ours and the AgentTTS
comparisons)", and is "certified 13 Oct 2026: reproduces the published AgentTTS 0.746". `LEN(answer, gold_answers)`
returns 1 when, for some gold answer, either test passes:

- **Exact match:** the normalised answer equals the gold. Normalising lowercases, strips punctuation and removes
  articles.
- **Containment either way:** one side's words occur as a contiguous run inside the other's.
  - The run must not follow a quantifier word ("more", "about", "nearly", …), so the answer "about 300" does not match the gold "300".

The file is not to be modified; variants go in new files.
([lenient_match_agenttts.py](https://github.com/mohameddhameem/STAIR/blob/e3ff28b/jonathan/final/lenient_match_agenttts.py))

## Reported numbers (README, scored with that file)

| | Lenient accuracy | Cost |
|---|---|---|
| Always-strong | 76.77 | 100% |
| Router | 75.96 | 71.6% of always-strong |
| AgentTTS's published method (reproduced 16 Oct) | 74.60 | |

These are on a "sealed test, n = 1,481", which is the same size as the team's frozen test split that nonLLM uses. The
repo does not yet say which router, models or cost unit these numbers use. The README warns that strict EM gives much
lower values and is not comparable.

## Why it matters to us

- **A common grader for the AgentTTS comparison.** Our final report's section 5.1 is waiting for one.
  - Our experiment 8 found that containment overstates long fused answers. Scoring our experiment 8 answers with
    this file, next to our LLM grader, would put our numbers on the team's scale.
- **It differs from nonLLM's lenient EM.** nonLLM's version is a "guarded bidirectional word-boundary containment"
  that it defines locally and does not certify as identical. Check which scorer a teammate's number used before
  comparing it with another's.
