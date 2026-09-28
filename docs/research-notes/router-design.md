# Designing, Labelling, Training and Evaluating a Router on HotpotQA (Distractor Setting)

Scope: implementation-level guidance for a student team with one consumer GPU (RTX 3060/4090 or Colab/Kaggle T4/P100). The team already exports HotpotQA distractor (train ~90k, validation ~7.4k) and has two reference codebases: nano-jev (MiniLM-L12-H384 cross-encoder, grouped softmax CE and temperature scaling) and rizzo-flow (LoRA on a small causal LM, soft CE over allowed answer-token logits).

---

## Q1. Which expert/solver pool makes sense, and what HotpotQA EM/F1 do small/open models get?

### Takeaway
A three-tier pool fits the budget: (1) a fine-tuned extractive reader (DeBERTa-v3-base class; cheapest, strong on span answers but weak or missing on yes/no), (2) a small open instruction LLM (Qwen2.5/Llama-3.x, 1–8B) that reads all 10 paragraphs, with or without CoT, and (3) either a larger LLM/API model or a "gold-paragraph + LLM" pipeline as the expensive tier. Published numbers show large gaps between tiers and a big gain from paragraph selection (roughly 17–21 F1 for Llama 3.1). That gap is what the router exploits.

### Cited Findings

**Dataset facts that shape the pool**
- HF `hotpotqa/hotpot_qa` distractor config: train 90,447 rows and validation 7,405 rows. Fields are `id, question, answer, type, level, supporting_facts, context`. License CC BY-SA 4.0. — [HF dataset card](https://huggingface.co/datasets/hotpotqa/hotpot_qa)
- Distractor setting: 8 distractor paragraphs are retrieved by bigram TF-IDF using the question as the query and mixed with the 2 gold paragraphs. — [Yang et al. 2018, HotpotQA (EMNLP)](https://aclanthology.org/D18-1259.pdf)
- Answer-type distribution in a sample: Person 30%, Group/Org 13%, Location 10%, Date 9%, Number 8%, Artwork 8%, **Yes/No 6%**, Adjective 4%, Event 1%, Other proper noun 6%. — [Yang et al. 2018](https://aclanthology.org/D18-1259.pdf)
- Reasoning-type sample from train-medium + train-hard: Type I (bridge-entity chain) 38%, Type II 29%, Comparison 20%, Other 7%, Type III 2%, single-hop 2%, unanswerable 2%. — [Yang et al. 2018](https://aclanthology.org/D18-1259.pdf)
- Original baseline (distractor): dev answer EM 44.44 / F1 58.28, test EM 45.46 / F1 58.99. Human performance on 1,000 samples: answer EM 83.60 / F1 91.40, with an annotator upper bound of EM 96.80 / F1 98.77. — [Yang et al. 2018](https://aclanthology.org/D18-1259.pdf)

**(a) Extractive / encoder readers**
- Public HF checkpoint `MhoOmm/HotPotQA_DEBERT` is built on `microsoft/deberta-v3-base` (about 0.2B). It was fine-tuned on 84,959 train and 4,687 validation examples and reports **EM 60.53 / F1 74.21** on its validation set. **It does not support yes/no questions**, which were removed in preprocessing, and it expects context assembled by an external retriever. — [HF model card](https://huggingface.co/MhoOmm/HotPotQA_DEBERT)
- A HF hub filter on the HotpotQA dataset shows few reader checkpoints: `MhoOmm/HotPotQA_DEBERT` (QA), `fwp/BART-base-HotpotQA-finetune` and `fwp/BART-large-HotpotQA-finetune` (generative), `xiuyul/Lloco-7b-hqa`, plus question-generation models. The page listed 31 models in total. — [HF models filtered by hotpot_qa](https://huggingface.co/models?dataset=dataset:hotpotqa/hotpot_qa&sort=downloads)
- General SQuAD2.0 readers such as `deepset/deberta-v3-base-squad2` and `deepset/deberta-v3-large-squad2` exist and could serve as a zero-shot extractive baseline. They are not HotpotQA-tuned. — [deepset/deberta-v3-large-squad2](https://huggingface.co/deepset/deberta-v3-large-squad2)
- Strong supervised pipelines (selector + reader), FE2H, HotpotQA distractor **test** results: ELECTRA-large reader answer EM 69.54 / F1 82.69; ALBERT-xxlarge answer EM 71.89 / F1 84.44, joint EM 50.04 / F1 76.54. The ELECTRA-large reader trained at batch 16 on 2×A100, about 2 h/epoch and about 6.5 h total. The ALBERT-xxlarge reader took about 25 h on 3×A100. The document selector reaches about 98 F1 on dev. — [FE2H, Li et al. 2022](https://arxiv.org/pdf/2205.11729)
- Beam Retrieval (NAACL 2024, DeBERTa backbone) reports SOTA on HotpotQA. It is cited elsewhere as 90.09 supporting-fact F1, 85.04 answer F1 and 77.54 joint F1. — [Beam Retrieval repo](https://github.com/canghongjian/beam_retriever); numbers as quoted by [Bactrainus, arXiv 2501.06286](https://arxiv.org/html/2501.06286)
- Quark: a simple pipeline of BERT sentence selection → BERT span reader → support selection. It beats graph-based models of its time and is "very close to a RoBERTa model". — [Groeneveld et al. 2020, arXiv 2004.06753](https://arxiv.org/abs/2004.06753)

**(b) Small/open LLMs**
- Llama 3.1 8B Instruct, zero-shot, **gold supporting facts only**: EM 60.11 / F1 74.52. With **all candidate paragraphs** (distractor-style) F1 falls to 57.20, a drop of 17.32. Llama 3.1 70B: gold EM 65.60 / F1 80.04, all-candidates F1 58.61 (a drop of 21.43). Llama 3.1 405B gold: EM 67.46 / F1 82.56. GPT-4o gold: EM 67.54 / F1 83.44. Claude 3.5 Sonnet gold: EM 67.12 / F1 83.07. — [Bactrainus, arXiv 2501.06286](https://arxiv.org/html/2501.06286)
- **CoT hurt** Llama 3.1 8B on gold facts. At 0 shots: direct F1 74.52 vs CoT 73.41. Across 0/1/2/4/8 shots, CoT was 0.9–3.9 F1 lower in every setting. — [Bactrainus](https://arxiv.org/html/2501.06286)
- LoRA fine-tuning helps a lot. On gold facts, Llama 3.1 8B + LoRA reached EM 74.02 / F1 86.46 and the 70B + LoRA reached EM 75.73 / F1 90.01. An "all-in-one" Llama 3.1 8B system got answer F1 83.31 / joint F1 75.96. — [Bactrainus](https://arxiv.org/html/2501.06286)
- Older few-shot CoT numbers on the HotpotQA dev (original) set: Llama-2-13B EM 30.9 / F1 45.8, Mixtral-8x7B-Instruct EM 50.4 / F1 68.1, Llama-2-70B EM 54.1 / F1 67.7, GPT-3.5 EM 63.4 / F1 77.2. A fine-tuned Longformer got EM 71.5 / F1 82.1. — [Bhuiya et al. 2024, "Seemingly Plausible Distractors", arXiv 2409.05197](https://arxiv.org/pdf/2409.05197)
- The same paper shows that **plausible distractors** built from alternate reasoning chains cut F1 by up to about 24 points for GPT-3.5 (77.2 → 52.7). GPT-4 lost about 14 F1 in the strongest attack. — [arXiv 2409.05197](https://arxiv.org/pdf/2409.05197)
- FLAN-T5-XL (3B) on HotpotQA in the open-retrieval Adaptive-RAG setting (not distractor): no-retrieval EM 16.60 / F1 22.71, single-step EM 34.40 / F1 46.15, multi-step EM 44.60 / F1 56.54. — [Adaptive-RAG, Jeong et al. NAACL 2024](https://arxiv.org/html/2403.14403v2)
- A GitHub project reports Qwen2.5-7B-Instruct at EM 53.2 / F1 61.6 on 5,000 distractor-dev questions with an agentic RAG pipeline, vs EM 43.1 / F1 54.0 for single-pass dense retrieval. This is a personal repo, not peer-reviewed, so treat it with caution. — [hammas159/mcp-lab](https://github.com/hammas159/mcp-lab/tree/main/projects/02_hotpotqa_multihop_rag)

### Inferences
- A **recommended pool for the course project**, cheapest to most expensive:
  - E0: the nano-jev-style cross-encoder, or a heuristic, for yes/no only. It is very cheap.
  - E1: an extractive DeBERTa-v3-base reader over the top-k paragraphs (self-trained, or `MhoOmm/HotPotQA_DEBERT` as a start). It needs a yes/no head or a fallback, because the public checkpoint drops yes/no.
  - E2: a small LLM (Qwen2.5-1.5B/3B-Instruct or Llama-3.2-3B) answering directly over the top-2/top-4 paragraphs chosen by the cross-encoder.
  - E3: a 7–8B LLM (Qwen2.5-7B / Llama-3.1-8B, 4-bit) over all 10 paragraphs, optionally with CoT or two-hop decomposition.
  - E4 (optional): an API model.
- The 17–21 F1 drop from gold-only to all-candidates context in Bactrainus suggests that "paragraph selection with the existing nano-jev cross-encoder, then a small LLM" is itself a powerful mid-cost expert. It also means routing can include *context-size decisions* (top-2 vs all 10), not just model choice.
- Because CoT did not help Llama 3.1 8B on gold context, "CoT vs no-CoT" may not be worth being its own expert. Measure it on a 500-question pilot before paying CoT generation cost across the whole label set.
- rizzo-flow's LoRA recipe maps directly onto a "LoRA-tuned small LLM reader" expert. Bactrainus shows that LoRA lifts an 8B from 74.5 to 86.5 F1 on gold context.

### Gaps
- No peer-reviewed source was found for **Qwen2.5/Qwen3 0.5B–3B, Phi or Gemma zero-shot EM/F1 on the HotpotQA distractor setting**. The team must measure these itself; plan a 500–1,000-question pilot.
- No HF checkpoint was found for a HotpotQA-tuned *large* extractive reader (ELECTRA-large/DeBERTa-v3-large with yes/no and supporting-fact heads). FE2H/Beam Retrieval code exists; downloadable weights unconfirmed.
- The `MhoOmm/HotPotQA_DEBERT` validation split (4,687) differs from the official 7,405 dev set, so its 60.5/74.2 is not directly comparable.

---

## Q2. How should routing labels be derived from HotpotQA, and what are the pitfalls?

### Takeaway
The standard recipe is to run every expert on every training question, score each answer with the official EM/F1, and label each question with **"the cheapest expert that is correct"** (Adaptive-RAG style). Keep the full per-expert correctness vector for multi-label training as well. Soft labels come from sampling several answers per expert (Hybrid-LLM style). HotpotQA's own `level` field is a trap: dev/test contain only "hard" questions.

### Cited Findings
- **Adaptive-RAG labelling:** it runs no-retrieval (A), single-step (B) and multi-step (C) strategies. When several succeed, the simplest correct one wins: "if both single-step and multi-step approaches produce the same correct answer while the non-retrieval-based approach fails, we assign label 'B'". Queries no strategy answers fall back to the **dataset's inductive bias** (single-hop datasets → B, multi-hop datasets → C). — [Adaptive-RAG paper](https://arxiv.org/html/2403.14403v2); code combines "silver" and "binary" labels via `concat_binary_silver_train.py` — [starsuzi/Adaptive-RAG](https://github.com/starsuzi/Adaptive-RAG)
- Adaptive-RAG trained its classifier on only **400 sampled queries per dataset**, with no overlap with the test set. Classifier accuracy was just **54.52%** overall (No-retrieval 30.52%, One-step 66.28%, Multi-step 65.45%). Even so, the router gave a useful cost trade-off on HotpotQA: Adaptive-RAG F1 53.82 at 3.55 steps / 5.99 relative time, vs multi-step F1 56.54 at 5.53 steps / 9.38 time. The **oracle** router reached F1 64.00 at 1.59 steps / 2.77 time. — [Adaptive-RAG](https://arxiv.org/html/2403.14403v2)
- **Hybrid LLM (Ding et al., ICLR 2024)** defines three label types:
  - deterministic: y = 1[q(small) ≥ q(large)], from one response per model;
  - probabilistic: Pr[q(small) ≥ q(large)], estimated from **10 sampled responses per model**;
  - transformed: Pr[H(x) ≥ −t], with a relaxation margin t tuned by grid search.

  The soft and transformed labels gave better routers under large quality gaps. — [Hybrid LLM, arXiv 2404.14618](https://arxiv.org/html/2404.14618)
- **RouteLLM** label sources: human preference (win_strong / tie / win_weak), "golden-labeled" data built from about 1,500 MMLU validation questions by checking correctness, and about 120K GPT-4-judge labels costing about $700. Golden-labelled augmentation (correctness against a reference, which is what HotpotQA provides for free) helped substantially on MMLU. — [RouteLLM paper, arXiv 2406.18665](https://arxiv.org/html/2406.18665)
- **HotpotQA `level` field:** train-easy (18,089) is "mostly single-hop". Train-medium (56,814) holds multi-hop questions that a baseline answered correctly with high confidence under 3-fold cross-validation. The remaining hard examples were split into train-hard (15,661), **dev (7,405), test-distractor and test-fullwiki, which are all "hard"**. — [Yang et al. 2018](https://aclanthology.org/D18-1259.pdf)
- About 6% of sampled questions can be answered from one of the two paragraphs, and about 2% are unanswerable. — [Yang et al. 2018](https://aclanthology.org/D18-1259.pdf)
- The official metric treats yes/no strictly. If either normalised prediction or gold is 'yes', 'no' or 'noanswer' and they differ, F1 = 0, so no partial credit. — [hotpot_evaluate_v1.py](https://raw.githubusercontent.com/hotpotqa/hotpot/master/hotpot_evaluate_v1.py)

### Inferences
- **Recommended label schema.** Per question q and expert e, store `em[q,e]`, `f1[q,e]`, `cost[q,e]` (tokens/latency) and optionally `p_correct[q,e]` from k = 3–5 samples at temperature ~0.7. From these, derive:
  1. `y_cheapest` = argmin cost over experts with F1 ≥ τ, with τ = 0.8 or EM = 1. If no expert is correct, use label "none" or the most expensive expert (Adaptive-RAG fallback). This is the multi-class target.
  2. `y_multi[e]` = 1[F1 ≥ τ], giving a per-expert success vector for BCE.
  3. `y_soft[e]` = mean correctness across samples (Hybrid-LLM style).
- **The level field.** Since validation is 100% "hard", **do not** train a router on `level` and evaluate on validation. The label is constant there, and it was defined by a 2018 baseline's confidence, not by your experts. Use `level` only as a train-set feature or for a per-level breakdown *within a held-out slice of train*. `type` (bridge/comparison) and "gold answer ∈ {yes, no}" are clean and present in all splits. They are good auxiliary labels or type-specific experts, but using gold-answer yes/no at *test time* is leakage. At test time it has to be predicted from the question.
- **Leakage checklist:**
  - never feed `supporting_facts` or gold-paragraph identity into the router at test time (distractor order also carries no gold signal);
  - build router train/calibration/test splits from *train* questions that were not used to fine-tune E1/E2, or use cross-fitting (k-fold) so that expert correctness on router-training data is not inflated by memorisation;
  - keep the official 7,405 validation set untouched for the final report.
- **Label noise:** HotpotQA's EM is strict and LLMs paraphrase, which produces false negatives. Use F1 ≥ 0.8 or a normalised-containment check as the success criterion, or add an LLM-judge pass on disagreements. Report sensitivity to τ. The low Adaptive-RAG classifier accuracy (54.5%) together with its useful end-to-end gains suggests that labels are noisy but routers still help. Judge the router by the cost–quality curve, not classification accuracy.

### Gaps
- No HotpotQA-specific study was found that quantifies how often small-LLM answers are marked wrong by EM but are semantically correct. Estimating this on a 200-question manual audit is recommended.

---

## Q3. What should the router see as input?

### Takeaway
There are three design points of increasing cost and power:
1. **Question-only** (pre-retrieval, as in Adaptive-RAG and RouteLLM).
2. **Question + cheap context signals**: cross-encoder paragraph scores from nano-jev, such as the top-1/top-2 margin and the entropy over the 10 paragraphs.
3. **Cascade / post-hoc**: the cheap expert's own answer confidence (token-level uncertainty) decides whether to escalate.

Features from option 3 are known to be strong, but they pay for the cheap expert on every query.

### Cited Findings
- Adaptive-RAG routes on the **query alone** with a T5-Large (770M) classifier. — [Adaptive-RAG](https://arxiv.org/html/2403.14403v2)
- RouteLLM routers take the prompt only: a BERT-base classifier, matrix factorisation over prompt embeddings, similarity-weighted Elo, and a Llama-3-8B classifier. — [RouteLLM paper](https://arxiv.org/html/2406.18665); [RouteLLM repo](https://github.com/lm-sys/RouteLLM)
- RouterBench "predictive routers" use **KNN and MLP over query embeddings** to predict each LLM's performance, then pick the best predicted quality minus λ·cost. These often failed to beat the "zero router" on some tasks. For example, on one benchmark the Zero router AIQ was 0.763 vs KNN 0.773 and MLP 0.769, and on another the zero router scored 0.660 vs KNN 0.644 and MLP 0.642. On a RAG dataset, however, all routers significantly beat the zero router. — [RouterBench, arXiv 2403.12031](https://arxiv.org/pdf/2403.12031)
- **Cascades / token-level uncertainty:** Gupta et al. (ICLR 2024) show that simple aggregates of a small LM's token uncertainty (e.g., sequence probability) are sub-optimal for deferral. Learned post-hoc deferral rules over the **per-token uncertainty vector** do significantly better, and adding the small model's embeddings gives a further boost. — [Language Model Cascades: Token-level uncertainty and beyond](https://arxiv.org/abs/2404.10136)
- RouterBench cascades: with a perfect (0-error) judge, the cascading router AIQ was 0.901 vs 0.763 for the zero router on MMLU. Performance degrades as judge error rises (tested at 0.0/0.01/0.05/0.1/0.2/0.4). — [RouterBench](https://arxiv.org/pdf/2403.12031)
- Distractor sensitivity is large. Llama 3.1 8B F1 falls from 74.52 (gold) to 57.20 (all candidates). — [Bactrainus](https://arxiv.org/html/2501.06286)

### Inferences
- **Practical feature set for HotpotQA** (all computable without gold labels):
  - (i) question embedding: MiniLM/DeBERTa CLS, or fine-tune the encoder end-to-end;
  - (ii) lexical cues: starts with is/are/was/did/does → likely yes/no; "which … or …" / "both" / "same" → comparison; question length; number of named entities;
  - (iii) nano-jev cross-encoder scores over the 10 paragraphs: top-2 sum, top-2 vs third-score margin, softmax entropy. A low margin suggests ambiguous evidence and so an expensive expert;
  - (iv) optional cascade features: extractive-reader start/end max-prob and yes/no-head prob, or the small LLM's mean/min token log-prob over the answer span.
- The existing nano-jev cross-encoder is a natural **router backbone**. Encode `[question] [SEP] [expert descriptor / option]` and score each expert as an "option" with grouped softmax. This is exactly its existing (question, option) formulation, and temperature scaling then carries over directly.

### Gaps
- No study was found on HotpotQA that directly compares question-only routers with routers that also see paragraph-relevance scores. This is a good ablation for the project.

---

## Q4. Which training objectives, and how should thresholds be chosen?

### Takeaway
Train a **per-expert success predictor** (multi-label BCE, or soft BCE on sampled labels). Then pick the expert that maximises `p̂_success(e) − λ·cost(e)`, or a cascade threshold, and sweep λ (or the threshold) on a held-out calibration split to trace the cost–quality curve. Multi-class CE on "cheapest-correct" is a simpler baseline. RouteLLM-style pairwise "strong-wins" probability is the two-expert special case.

### Cited Findings
- **Hybrid LLM:** DeBERTa-v3-large (300M) router trained with **BCE** on deterministic, probabilistic or transformed labels. Setup: 10k training examples, 5 epochs, 1×A100-80GB. Thresholds come from grid search on **500 validation samples** to maximise cost advantage subject to ≤1% quality drop, and they generalised to test. Router latency is 0.036 s, about 10× faster than the fastest LLM. Results: 22% fewer large-model calls with <1% quality drop (Llama-2-13B vs GPT-3.5); 40% cost advantage with a 10.3% quality drop under a large gap. — [Hybrid LLM](https://arxiv.org/html/2404.14618)
- **RouteLLM:** each router outputs P(strong model wins | q) and routes to the strong model when it exceeds a threshold α. The threshold is calibrated so that a target fraction of queries goes to the strong model (e.g., "For 50.0% strong model calls for mf, threshold = 0.11593"). The BERT router is BERT-base with full fine-tuning, lr 1e-5, batch 16, about 2,000 steps, weight decay 0.01, max length 512, on 2×L4. The matrix-factorisation router uses P(win_s|q) = σ(δ(M_s,q) − δ(M_w,q)) with a bilinear score. The causal-LLM router is Llama-3-8B predicting label tokens. — [RouteLLM paper](https://arxiv.org/html/2406.18665); [RouteLLM repo](https://github.com/lm-sys/RouteLLM)
- **Adaptive-RAG:** multi-class CE with T5-Large, AdamW, lr 3e-5, best validation checkpoint within 100 iterations. — [Adaptive-RAG](https://arxiv.org/html/2403.14403v2)
- **RouterBench predictive router:** predict per-LLM performance P̂(x) with KNN or MLP, then select argmax of P̂ − λ·cost, sweeping λ. — [RouterBench](https://arxiv.org/pdf/2403.12031)

### Inferences
- **Mapping to existing code:**
  - nano-jev's grouped softmax CE over options is a multi-class router that treats the experts as options. Adding a sigmoid-per-option head gives multi-label BCE. Its post-hoc temperature scaling is directly useful, because utility-based routing (`p̂ − λ·cost`) needs calibrated probabilities.
  - rizzo-flow's "supervise only allowed answer-token logits with soft CE" is exactly RouteLLM's causal-LLM router pattern. Each expert gets a label token such as `A`/`B`/`C`, and the soft targets are the normalised `y_soft` from Q2.
- **Recommended objectives, ranked for a student project:**
  1. Multi-label BCE on `y_multi` or `y_soft` (one sigmoid per expert). This is the most flexible, because any cost vector or budget can be applied afterwards without retraining.
  2. Multi-class CE on `y_cheapest`, which is easy to explain.
  3. Optional cost-sensitive CE: weight or soft-target the classes by utility `u(e) = F1(e) − λ·cost(e)` and apply a softmax over u/T as the target.
  4. Pairwise ranking between adjacent experts, which is RouteLLM's special case.
- **Threshold/λ selection:** split router data into train / calibration (for example, 2k questions) / test. On calibration, fit the temperature, then sweep λ (or the escalate threshold) over a grid. Report the operating point that meets "≤1% F1 below the best single expert" (the Hybrid-LLM convention) and the one that meets "≤50% of expensive-expert calls" (the RouteLLM CPT convention).

### Gaps
- No source was found reporting explicit utility-maximising, cost-sensitive loss results on HotpotQA. That part of the recommendation is design inference.

---

## Q5. How should the router be evaluated?

### Takeaway
Plot **answer F1 (and EM) against average cost** for every router operating point, each single expert, a random-mixture baseline and the oracle router. Summarise with the area under the non-decreasing convex hull (RouterBench AIQ) and/or APGR/CPT (RouteLLM). Break results down by `type`, yes/no vs span, and (train-slice only) `level`. Score answers with the official `hotpot_evaluate_v1.py`.

### Cited Findings
- **Official scorer:** `python hotpot_evaluate_v1.py <prediction_file> <gold_file>`. The prediction JSON has an `answer` dict (id → string) and an `sp` dict (id → list of [title, sent_id]). — [hotpotqa/hotpot README](https://github.com/hotpotqa/hotpot)
- Normalisation drops articles and punctuation, lowercases, and collapses whitespace. yes/no/noanswer mismatches get zero F1. The script outputs em, f1, prec, recall, sp_em, sp_f1, sp_prec, sp_recall and joint_* metrics. Joint scores multiply the two parts, e.g. `joint_em = em * sp_em` and `joint_prec = prec * sp_prec`. — [hotpot_evaluate_v1.py](https://raw.githubusercontent.com/hotpotqa/hotpot/master/hotpot_evaluate_v1.py)
- **RouteLLM metrics:**
  - PGR = (r(router) − r(weak)) / (r(strong) − r(weak));
  - APGR = ∫ PGR d(cost), approximated by averaging PGR at 10 call-rate points;
  - CPT(x%) = the minimum % of strong-model calls needed to reach x% PGR.

  Example numbers: MT-Bench CPT(50%) 23.21% for MF+augmentation vs 49.03% for random; MMLU CPT(50%) 35.40% vs 50.07% random. — [RouteLLM paper](https://arxiv.org/html/2406.18665)
- **RouterBench metrics:** router points are plotted on the cost–quality plane. Linear interpolation between routers (randomly choosing A or B) is allowed, and the **non-decreasing convex hull (NDCH)** keeps the Pareto-optimal points. **AIQ** is the area under the NDCH, normalised by (c_max − c_min). The **Zero router** is the best random mixture of single models, i.e. the NDCH of the individual LLMs. The **Oracle router** picks the cheapest model that answers correctly and "achieves near-optimal performance at a low cost". — [RouterBench, arXiv 2403.12031](https://arxiv.org/pdf/2403.12031)
- **Adaptive-RAG** reports an efficiency axis of **retrieval steps and relative time per query** alongside EM/F1/Acc. Its HotpotQA oracle (F1 64.00 at 2.77 relative time) beats every fixed strategy on both axes. — [Adaptive-RAG](https://arxiv.org/html/2403.14403v2)

### Inferences
- **Recommended evaluation table and plot:**
  - x-axis: average cost per question in (a) generated plus prompt tokens, (b) wall-clock ms on the team GPU, and optionally (c) $ at API prices. Pick one as primary and report the others in an appendix.
  - Lines: each single expert (points); the Zero router (convex hull of the singles); a random router at matched call rates; the trained routers (curves from the λ sweep); the Oracle router (cheapest-correct per question); and the Adaptive-RAG-style question-only classifier as a baseline.
  - Scalars: AIQ over a common cost range; APGR / CPT(50%, 80%) for the cheapest-vs-most-expensive pair; F1 at matched cost to the best single expert.
  - Breakdowns: bridge vs comparison; yes/no-gold vs span-gold; answer-type buckets; `level` only on a held-out train slice.
- **Calibration diagnostics:** report the ECE/reliability diagram of per-expert success probabilities before and after temperature scaling, which reuses the nano-jev tooling.

### Gaps
- There is no published HotpotQA-distractor router leaderboard or standard cost unit, so absolute comparisons with the literature will be loose.

---

## Q6. Practical tooling and realistic compute on one consumer GPU

### Takeaway
Use HF `datasets`/`transformers`/`peft` for readers and routers, and **vLLM offline batch generation** for LLM experts. On an RTX 4090, 1.5–3B models sustain on the order of 10 requests/s in vLLM offline mode, and 7–8B models run at roughly 4–6 req/s. Labelling all ~90k training questions with several experts is feasible but slow. Subsample 15–25k training questions for router labels plus the full 7,405 dev set.

### Cited Findings
- Throughput on RTX 4090 with vLLM offline, 300 requests, 100 input / 600 output tokens:

  | Model | Requests/s | Total tokens/s |
  |---|---|---|
  | Qwen2.5-3B-Instruct | 10.31 | 7,214 |
  | DeepSeek-R1-Distill-Qwen-1.5B | 13.85 | 9,696 |
  | DeepSeek-R1-Distill-Qwen-7B | 5.96 | 4,173 |
  | DeepSeek-R1-Distill-Llama-8B | 3.96 | 2,769 |

  The source describes the 4090 as "best suited for LLMs under 8B". — [DatabaseMart RTX4090 vLLM benchmark](https://www.databasemart.com/blog/vllm-gpu-benchmark-rtx4090) (vendor blog; vLLM version not stated)
- Router training cost reference points:
  - RouteLLM BERT-base: about 2,000 steps at batch 16 on 2×L4. — [RouteLLM](https://arxiv.org/html/2406.18665)
  - Hybrid-LLM DeBERTa-v3-large: 10k examples, 5 epochs, on an A100. — [Hybrid LLM](https://arxiv.org/html/2404.14618)
  - Adaptive-RAG T5-Large: 400 queries per dataset. — [Adaptive-RAG](https://arxiv.org/html/2403.14403v2)
- Reader training reference: an ELECTRA-large HotpotQA reader took about 6.5 h on 2×A100 at batch 16. — [FE2H](https://arxiv.org/pdf/2205.11729)
- Label-generation cost reference: RouteLLM's 120K GPT-4-judge labels cost about $700. — [RouteLLM](https://arxiv.org/html/2406.18665)

### Inferences
- **Label-generation time budget** (my estimate, to verify with a pilot).
  - The benchmark workload (100 in / 600 out) is decode-heavy. HotpotQA direct answering is prefill-heavy: 10 paragraphs, likely around 1–1.5k tokens (measure this), with about 5–20 output tokens. Req/s for direct answering should therefore be the same order or better than the table.
  - At a conservative 5 req/s for a 3B model, the full 7,405 dev set takes about 25 min and 20k train questions take about 1.1 h. All 90k takes about 5 h per expert per sample.
  - A 7–8B model at about 2–4 req/s roughly doubles this.
  - CoT (about 150–300 output tokens) could cost 5–10× more. Budget for CoT only on a subsample.
  - T4/P100 (16 GB, no bf16 on P100, older kernels) will be several times slower. Use 4-bit/AWQ 7B or stick to ≤3B there.
- **Suggested subsampling plan:**
  - Router-train: 20k train questions, stratified by type and yes/no. Exclude the questions used to fine-tune E1/E2, or use 2-fold cross-fitting.
  - Router-calibration: 2k train questions.
  - Final test: the full official 7,405 dev set.
  - For soft labels, draw k = 3–5 samples only for the LLM experts and only on the 20k subset.
- **Tooling:**
  - `datasets.load_dataset("hotpotqa/hotpot_qa", "distractor")` for data;
  - `transformers` `AutoModelForQuestionAnswering` for E1, adding a 3-way yes/no/span classifier head (HotpotQA readers commonly do this);
  - vLLM `LLM.generate` with `SamplingParams(temperature=0, max_tokens=32, logprobs=5)` to capture token log-probs for cascade features;
  - `peft` LoRA for the router-as-causal-LM variant (rizzo-flow) or for a LoRA-tuned LLM expert;
  - the nano-jev MiniLM cross-encoder for the router and for paragraph scoring. It is 33M params and trains quickly on any listed GPU.
- **A realistic 3–4-week plan:**
  - Week 1: experts E1–E3 inference on dev plus a 20k train subset, cached to parquet (id, expert, prediction, em, f1, tokens, latency, logprobs).
  - Week 2: labels, the question-only router (MiniLM/DeBERTa-v3-small BCE), and the Adaptive-RAG-style CE baseline.
  - Week 3: the router with cross-encoder features plus the cascade variant, temperature scaling, and the λ sweep.
  - Week 4: cost–quality curves, AIQ/APGR, per-type breakdowns, and the oracle/random baselines.

### Gaps
- No authoritative per-token or per-question vLLM throughput for HotpotQA-length prompts (about 1–1.5k input and short output) on T4/P100/3060. The team should benchmark this on 500 questions.
- The average token length of a HotpotQA distractor context was not confirmed from a primary source. Compute it with the chosen tokenizer.
- The RTX 4090 throughput figures come from a vendor blog with unspecified vLLM settings, so treat them as order-of-magnitude.
