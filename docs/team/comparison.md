# How the three lines of work compare

Ours is this repo (experiments 7–9, final report sections 4.5–4.7 and 5.1). The teammates' lines are summarised in
[nonLLM.md](nonLLM.md) and [selfLearnt.md](selfLearnt.md). Snapshot 2026-10-16/17.

| | **Ours: System One judge** | **team-member-1: nonLLM** | **team-member-2: selfLearnt** |
|---|---|---|---|
| Question | Can a small typed decision model (nano-jev 33M; Strands Decider 2B / 4B) decide per question when the large model is needed, cheaply? | Can cheap non-LLM classifiers route each stage W/S? How many answer samples does each question need? | Can fine-tuned model ladders plus a learnt rulebook replace a router? |
| Routing unit | Whole question (answer stage), small vs large; 5 options in experiment 8 | Per stage (retrieval, answering), W/S; plus pair × sample budget k | Per stage, escalation up a ladder of sizes, by rules |
| Decision model | Fine-tuned / zero-shot cross-encoder or decider reading question + passages (+ small answer) | TF-IDF LR / SVM / NB, random forest | Thresholded signals (confidence, agreement, probes), tuned by micro-experiments |
| Models | Qwen 1.5B / 7B (pair A); Llama 3B ×1–40 fused vs Llama 70B (experiment 8) | Qwen2.5 1.5B / 7B (stored grid; GGUF Base, then Instruct) | Qwen2.5 0.5B–72B retrievers, Llama 3B / 8B / 70B answerers; Colab: Qwen 7B / 32B |
| Test set | 7,405 HotpotQA questions (experiment 7); AgentTTS's 500 (experiment 8) | Team split: test 1,481 of 7,405 | AgentTTS's 500 (cluster); 400 (Colab) |
| Metric | F1 / EM (experiment 7); LLM-graded correctness of a committed short answer (experiment 8) | Lenient + strict on the stored grid; official EM/F1 with an extractor in the sampling run | Containment (AgentTTS's), plus FLOPs; Colab: corrected label |
| Cost | GFLOPs (2 × params × tokens), judge cost included; AgentTTS units in experiment 8 | Pipeline units 2.0 / 5.7 / 5.7 / 9.4; token-work proxy | AgentTTS budget formula + FLOPs; Colab: cost index |
| Best result so far | Experiment 7: AIQ 0.673 (nano-jev) → 0.708 (2B decider, qpa), +0.030 to +0.039 paired. Experiment 8: 0.748 graded at 10.3% of strong-everything, but judge only ties the hull | Routers +0.9 to +1.3 points over cost-matched fixed policies (stored grid, lenient) | Colab: fine-tuned 0.8B retriever → Qwen 32B 84.0% at 43.5% of the stand-in's compute |
| Status | Done (experiments 7–9 merged); report drafted | Instruct GGUF sampling running; 500-question cohort tables pending | Cluster run not started at the snapshot |

## Where the numbers can and cannot be compared

- **Metrics differ.** Our experiment 8 found that lenient containment overstates long fused answers by up to 0.14
  (graded vs lenient); team-member-2's cluster results and AgentTTS's 74.2 / 74.6 are containment. Comparing them with
  our graded 0.748 needs the same grader on their answers (`stair.rescore` in this repo).
- **Test sets differ.** Experiment 8 and selfLearnt's cluster workstreams share AgentTTS's 500 questions, the only
  shared test set. nonLLM uses the team's 1,481-question test split; our experiment 7 uses all 7,405 with its own
  calib/test split.
- **Model menus differ.** Experiment 8 uses AgentTTS's Llama 3B / 70B answerers, as selfLearnt's workstreams 1, 2 and
  4 do; nonLLM and our experiment 7 use Qwen 1.5B / 7B.
- **Common findings.** All three lines find that question-only signals are weak: nonLLM's stage classifiers are near
  chance; selfLearnt's question-reading routers were at chance and learned routes collapsed to the majority; our
  judges gain most when they read the small model's answer (qpa). Retrieval quality is the bigger lever in selfLearnt
  (fine-tuned retrievers) and in our experiment 8 (72B retrieval at 2.9 vs 814 units).

## What to watch as the branches evolve

- nonLLM: the 500-question cohort tables (pair × budget), then a validation-selected budget on test.
- selfLearnt: workstreams 1–2 on the 500 questions (AgentTTS reproduced, strongest models), then 4–5. Workstream 1's
  answers are what our H3 needs, graded with our grader.
- Any change to the shared split or metric.
