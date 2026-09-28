# Routing Techniques for LLM / ML Systems (with emphasis on HotpotQA / multi-hop QA)

Compiled 2026-09-27. Source status: items marked **[unverified]** are from recall, not a fetched source. Check them before citing. Most other numbers come from summaries of arXiv HTML pages; one such summary (FrugalGPT) was wrong and was corrected against the PDF, so spot-check any number you quote.

---

## Q1. Model routing and cascades (FrugalGPT, RouteLLM, Hybrid LLM, AutoMix, RouterBench, Router-R1, RAGRouter, etc.): designs, how labels are made, metrics

### Takeaway
Almost all model routers use the same recipe. (1) Run every candidate LLM on a training set of queries. (2) Turn the outcomes into labels: "did the cheap model answer correctly / as well as the expensive one?" (binary, soft, or preference win-rate). (3) Train a small encoder classifier or regressor (DistilBERT, BERT-base, DeBERTa-v3-large, MF embeddings, kNN) on the query text, or on query+answer for cascades. (4) Sweep a threshold to trace a cost-quality curve and summarize it with APGR, CPT(x%), AIQ, or IBC. The 2026 "Routing Plateau" analysis finds that most learned routers land in a narrow accuracy band far below the oracle.

### Cited Findings

**FrugalGPT (Chen, Zaharia, Zou; arXiv May 9 2023): LLM cascade**
- Proposes three cost-saving strategies: prompt adaptation, LLM approximation, and LLM cascade. FrugalGPT is the cascade: it "learns which combinations of LLMs to use for different queries" — [arXiv 2305.05176](https://arxiv.org/abs/2305.05176)
- Cascade = (i) a generation scoring function g(q, a) → [0,1] that gives a reliability score for a query plus an LLM's answer, and (ii) an LLM router that picks an ordered list of m APIs. Each API in the list is queried in turn. Its answer is returned if g ≥ threshold τ_i, otherwise the next API is called — [FrugalGPT PDF](https://arxiv.org/pdf/2305.05176)
- The scorer is "a simple regression model that learns whether a generation is correct from the query and a generated answer". The case study uses **DistilBERT tailored to regression**. The list L and thresholds τ are learned by constrained optimization: maximize expected reward subject to expected cost ≤ budget b — [FrugalGPT PDF](https://arxiv.org/pdf/2305.05176)
- Setup: 12 APIs from 5 providers (OpenAI, AI21, CoHere, Textsynth, ForeFrontAI), cascade length 3, datasets HEADLINES, OVERRULING, and COQA, each split randomly into train/test. On HEADLINES with a $6.5 budget (1/5 of GPT-4 cost), the learned cascade is GPT-J (accept if score > 0.96) → J1-L (accept if > 0.37) → GPT-4 — [FrugalGPT PDF](https://arxiv.org/pdf/2305.05176)
- Results: cost savings of 50%–98% at matched accuracy. It can match GPT-4 with up to 98% cost reduction, or beat GPT-4 by up to 4% at the same cost. **No HotpotQA evaluation.** — [arXiv 2305.05176](https://arxiv.org/abs/2305.05176); [PDF](https://arxiv.org/pdf/2305.05176)

**RouteLLM (Ong et al., LMSYS; arXiv 2406.18665, ICLR 2025): preference-data routers for a strong/weak pair**
- Label/target: learn P_θ(wins | q), the probability that the strong model beats the weak model on query q. Route to the strong model if P ≥ α, else to the weak model. Sweeping α traces the cost-quality curve — [arXiv 2406.18665 HTML](https://arxiv.org/html/2406.18665v4)
- Four routers:
  (1) **Similarity-weighted (SW) ranking**: a Bradley-Terry model whose training battles are weighted by cosine similarity of query embeddings to the test query. Solved at inference time with BCE.
  (2) **Matrix factorization**: model embedding v_m and query embedding v_q, score δ(M,q) = w₂ᵀ(v_m ⊙ (W₁ᵀ v_q + b)). About 10 epochs on an 8 GB GPU.
  (3) **BERT classifier**: BERT-base [CLS] → logistic head, full fine-tune for about 2000 steps on 2×L4.
  (4) **Causal-LLM classifier**: Llama-3-8B, instruction-style, softmax over label tokens, about 2000 steps on 8×A100-80GB.
  — [arXiv 2406.18665 HTML](https://arxiv.org/html/2406.18665v4)
- Training data: Chatbot Arena, about 80k battles → about 65k comparisons after pruning. Augmentations: about 1,500 "golden-label" MMLU validation questions, and about 120K GPT-4-as-judge samples (cost ≈ $700) — [arXiv 2406.18665 HTML](https://arxiv.org/html/2406.18665v4)
- Metrics:
  - PGR = (r_router − r_weak) / (r_strong − r_weak).
  - APGR = PGR integrated over cost budgets, approximated as the mean over 10 thresholds.
  - CPT(x%) = the minimum % of strong-model calls needed to reach x% PGR.
  — [arXiv 2406.18665 HTML](https://arxiv.org/html/2406.18665v4)
- Results: more than 2× cost savings. MF on MT-Bench nearly halves CPT(80%) compared with random routing. Table 6 reports up to 3.66× savings on MT-Bench at 95% quality. With golden-label augmentation on MMLU, routers need about 20% fewer GPT-4 calls than random at CPT(50%). **No HotpotQA evaluation.** — [arXiv 2406.18665 HTML](https://arxiv.org/html/2406.18665v4)
- Repo: lm-sys/RouteLLM [unverified].

**Hybrid LLM (Ding et al., ICLR 2024; arXiv 2404.14618): difficulty-aware router, small vs large model**
- Router: **DeBERTa-v3-large (about 300M params)**, single forward pass, about 0.036 s latency (about 10× faster than the smallest LLM tested) — [arXiv 2404.14618 HTML](https://arxiv.org/html/2404.14618v1)
- Three label schemes, all trained with BCE — [arXiv 2404.14618 HTML](https://arxiv.org/html/2404.14618v1):
  - r_det: y = 1[q(S(x)) ≥ q(L(x))] from one sample per model.
  - r_prob: soft label = Pr[q(S(x)) ≥ q(L(x))], estimated from 10 samples per model.
  - r_trans: y(t) = Pr[q(S(x)) > q(L(x)) − t], with a relaxation t* chosen by grid search to maximize label separation. This handles the case where the small model is almost never strictly better.
- Quality q = BART score. Data: MixInstruct, with 10k train / 5k val / 5k test examples — [arXiv 2404.14618 HTML](https://arxiv.org/html/2404.14618v1)
- Result: "up to 40% fewer calls to the large model, with no drop in response quality" — [arXiv 2404.14618](https://arxiv.org/abs/2404.14618)

**AutoMix (Aggarwal/Madaan et al., NeurIPS 2024; arXiv 2310.12963): self-verification + POMDP cascade**
- The small model answers first. Then **few-shot (4-shot) entailment-style self-verification**: the same small model is asked whether the answer is supported by the context. It samples k=8 times at temperature 1.0, and the verification probability is the fraction of "Correct" judgments — [arXiv 2310.12963 HTML](https://arxiv.org/html/2310.12963v5)
- Meta-verifier/router options:
  - Thresholding: escalate if v < t.
  - POMDP: states ⟨i, Perf_LM1..N⟩, verifier outputs treated as noisy observations, P(o|s) estimated by kernel density estimation on training data, reward R = P − λC.
  — [arXiv 2310.12963 HTML](https://arxiv.org/html/2310.12963v5)
- Metric IBC (incremental benefit per cost) = (P_M − P_SLM)/(C_M − C_SLM). ΔIBC is the relative lift over a baseline — [arXiv 2310.12963 HTML](https://arxiv.org/html/2310.12963v5)
- Datasets: QASPER, QuALITY, CoQA, MuTual, Diplomat, all context-grounded. Small models: Llama2-13B, Mistral-7B, GPT-3.5. Large model: GPT-4, with cost ratios from 1:60 to 1:200. Reports more than 50% cost reduction at comparable performance. **The paper does not evaluate on HotpotQA.** — [arXiv 2310.12963 HTML](https://arxiv.org/html/2310.12963v5); [abs](https://arxiv.org/abs/2310.12963)

**RouterBench (Hu et al., Martian; arXiv 2403.12031, 2024)**
- More than 405k precomputed inference records across 11 LLMs, covering commonsense reasoning, QA, conversation, math, coding, and RAG. Routers can be evaluated offline — [RouterBench arXiv](https://arxiv.org/pdf/2403.12031); [GitHub withmartian/routerbench](https://github.com/withmartian/routerbench)
- Metric AIQ (Average Improvement in Quality) = normalized area under the router's cost-quality curve within a cost range. Convex-hull evaluation is also used — [emergentmind summary of RouterBench](https://www.emergentmind.com/topics/routerbench-dataset)

**Router-R1 (Zhang/Feng et al., UIUC; NeurIPS 2025; arXiv 2506.09033): the router is itself an LLM trained with RL, multi-round**
- Policy LLM: Qwen2.5-3B-Instruct or Llama-3.2-3B-Instruct. It interleaves `<think>`, calls to a candidate LLM (the `<search>` tag), and `<answer>`, with up to 4 routing steps. The pool has 6 candidate LLMs, each described by price, latency, and capability descriptors. The descriptors let it generalize to unseen LLMs — [arXiv 2506.09033 HTML](https://arxiv.org/html/2506.09033)
- Reward: r = Format (−1 if malformed, else 0) + (1−α)·EM outcome + α·Cost reward, where the cost reward is inversely related to model size × output tokens. Trained with PPO (batch 64, converges within about 100 steps) on **14K samples: 7K NQ + 7K HotpotQA** — [arXiv 2506.09033 HTML](https://arxiv.org/html/2506.09033); [search summary](https://arxiv.org/abs/2506.09033)
- **HotpotQA EM**: Router-R1-Qwen 0.352, Prompt-LLM 0.268, GraphRouter 0.244, Search-R1 0.236. Average EM over 7 QA benchmarks: 0.416 (Qwen) / 0.409 (Llama). It makes more LLM calls on multi-hop sets (HotpotQA, 2Wiki, MuSiQue, Bamboogle) than on single-hop ones — [arXiv 2506.09033 HTML](https://arxiv.org/html/2506.09033); [GitHub ulab-uiuc/Router-R1](https://github.com/ulab-uiuc/Router-R1)

**RAGRouter (Zhang et al., NeurIPS 2025; arXiv 2505.23052): routing among retrieval-augmented LLMs**
- Motivation: retrieved documents shift what an LLM "knows", so routers based on static model embeddings are suboptimal under RAG — [arXiv 2505.23052](https://arxiv.org/abs/2505.23052)
- Architecture:
  - Query/document encoder all-mpnet-base-v2 (shared weights) and cross-encoder ms-marco-MiniLM-L12-v2.
  - Learnable per-LLM knowledge embeddings and "RAG-capability" embeddings.
  - Attention fusion v_f; updated model vector v_k' = v_k + v_f; route by sim(v_q, v_k').
  - Loss: contrastive loss (cross-setting RAG vs non-RAG, plus intra-setting across LLMs) + binary classification loss (λ=2.0).
  — [arXiv 2505.23052 HTML](https://arxiv.org/html/2505.23052)
- 15 candidate LLMs (0.5B–72B plus GPT-4o etc.). Main tasks: PopQA, MedMCQA, NQ, WebQuestions, TriviaQA. **HotpotQA is used only for cross-domain evaluation.** Results: +3.61 pts over the best single LLM (64.46% vs 60.85%); beats GraphRouter by 3.29, MF by 4.21, KNN by 4.02, RouterDC by 7.75 — [arXiv 2505.23052 HTML](https://arxiv.org/html/2505.23052)

**Router comparisons and benchmarks, 2025–2026**
- RouterArena (ICLR 2026) compares CARROT, RouterDC, GraphRouter, MIRT-BERT, NIRT-BERT, and RouteLLM on Arena, Cost-ratio, Optimal-acc, Latency, and Robustness scores. CARROT is strong on Arena and Latency; RouterDC is best on Cost-ratio — [RouterArena arXiv 2510.00202](https://arxiv.org/html/2510.00202v1); [ICLR 2026 PDF](https://proceedings.iclr.cc/paper_files/paper/2026/file/4987bb24bc53c198785922d1bd9e18cf-Paper-Conference.pdf)
- RouterDC trains a query encoder with dual contrastive losses (sample-LLM and sample-sample) that pull query embeddings toward the embeddings of top-performing LLMs. CARROT/SPROUT estimates each model's cost and performance and introduces the SPROUT dataset — [RouterArena arXiv 2510.00202](https://arxiv.org/html/2510.00202v1); [CARROT arXiv 2502.03261](https://arxiv.org/pdf/2502.03261)
- "The Routing Plateau" (Lu et al., arXiv 2606.07587, May 27 2026), 21 methods × 5 benchmarks:
  - "many methods, including kNN, achieve very similar accuracy and converge to a narrow performance range that remains far below the oracle router".
  - Routers "mainly learn global averaged model-performance trends rather than fine-grained query-specific routing signals".
  - Larger training sets, stronger encoders, and end-to-end fine-tuning help.
  — [arXiv 2606.07587](https://arxiv.org/abs/2606.07587)
- LLMRouter (Feng et al., arXiv 2608.06867, Aug 7 2026): unified library with 16+ routers, framed as context encoder + model encoder + scoring function + decision rule + learning signal. Introduces the xRouteBench benchmark. Learned routers beat the strongest fixed model by 14.6% (relative) — [arXiv 2608.06867](https://arxiv.org/abs/2608.06867)
- Survey "Doing More with Less: A Survey on Routing Strategies for Resource Optimisation in LLM-Based Systems" (Varangot-Reille et al., arXiv 2502.00409, Feb 1 2025; v3 Jul 21 2025). Taxonomy: pre-generation vs post-generation (cascade) routing; similarity-based, supervised, RL-based, and generative routers — [arXiv 2502.00409](https://arxiv.org/abs/2502.00409)

**Other named routers [unverified]**
- LLM-Blender (Jiang et al., ACL 2023): PairRanker (pairwise cross-encoder) + GenFuser; MixInstruct dataset — [arXiv 2306.02561](https://arxiv.org/abs/2306.02561)
- Zooter (Lu et al., 2023): reward-model-distilled query→expert router — [arXiv 2311.08692](https://arxiv.org/abs/2311.08692)
- Tryage (Hu et al., 2023): perceptive router predicting per-model loss — [arXiv 2308.11601](https://arxiv.org/abs/2308.11601)
- RouterDC (Chen et al., NeurIPS 2024) — [arXiv 2409.19886](https://arxiv.org/abs/2409.19886)
- GraphRouter (Feng et al., ICLR 2025): heterogeneous task–query–LLM graph with a GNN edge predictor — [arXiv 2410.03834](https://arxiv.org/abs/2410.03834)
- EmbedLLM (Zhuang et al., ICLR 2025): learns compact LLM embeddings for correctness prediction — [arXiv 2410.02223](https://arxiv.org/abs/2410.02223)
- MixLLM (Wang et al., NAACL 2025): contextual-bandit routing with online updates — [arXiv 2502.18482](https://arxiv.org/abs/2502.18482)
- Smoothie (Guha et al., NeurIPS 2024): label-free routing via weak supervision over output embeddings — [arXiv 2412.04692](https://arxiv.org/abs/2412.04692)
- Arch-Router (Katanemo, 2025): 1.5B generative router matching queries to user-defined domain/action policies — [arXiv 2506.16655](https://arxiv.org/abs/2506.16655)
- RouterEval (2025): large benchmark of router evaluation records — [arXiv 2503.10657](https://arxiv.org/abs/2503.10657)

### Inferences
- **Labels for a HotpotQA model router** follow directly from these papers. Run each candidate (e.g., small LLM, large LLM) on HotpotQA train questions and score with EM/F1. Then either:
  - binary label "cheapest model with EM=1" (FrugalGPT/Adaptive-RAG style),
  - soft label from k samples (Hybrid-LLM r_prob), or
  - relaxed label "small F1 ≥ large F1 − t" (r_trans). This is useful because EM is very sparse on HotpotQA.
- For evaluation, report the full cost–accuracy curve by sweeping the threshold. Summarize it with APGR (for 2 models) or AIQ/area-under-curve (for more models), and include random-routing and oracle-routing reference lines (Routing Plateau shows the oracle gap is large).
- Cascades (post-generation, e.g., FrugalGPT/AutoMix) can use the answer and supporting evidence as features. Pre-generation routers (RouteLLM/Hybrid LLM) see only the question. On multi-hop QA, post-generation signals are likely more informative, but they cost one small-model call.

### Gaps
- No RouteLLM, Hybrid-LLM, AutoMix, or FrugalGPT results on HotpotQA were found. Among model routers, only Router-R1 reports HotpotQA numbers, and RAGRouter uses HotpotQA only for cross-domain testing.
- Details of Zooter, Tryage, Smoothie, Arch-Router, MixLLM, EmbedLLM, GraphRouter, RouterEval, and LLM-Blender are unverified: training-set sizes and losses unknown.
- Router-R1's exact cost-reward formula and α values were only summarized. Check the paper.

---

## Q2. Adaptive retrieval routing (Adaptive-RAG and successors): method, labels, classifier, results

### Takeaway
Adaptive-RAG (NAACL 2024) is the canonical query-complexity router for HotpotQA-style QA. A **T5-Large** classifier labels each query A/B/C (no retrieval / single-step / multi-step IRCoT). It is trained on about 400 queries per dataset, with silver labels taken from which strategy answered correctly (preferring the simplest) and dataset-origin labels as fallback. It cuts steps and time sharply, but its classifier accuracy is only about 54.5%. Follow-ups replace the classifier with a bandit (MBA-RAG), LLM hidden-state probers (Probing-RAG), or token-level uncertainty (DRAGIN, SeaKR). A 2025 benchmark study finds that simple uncertainty estimators match Adaptive-RAG's HotpotQA accuracy with about half the LM calls.

### Cited Findings

**Adaptive-RAG (Jeong, Baek, Cho, Hwang, Park; NAACL 2024; arXiv 2403.14403)**
- Classifier: **T5-Large** with three labels: A = no retrieval, B = single-step retrieval, C = multi-step retrieval (IRCoT-style). Trained with cross-entropy, AdamW, learning rate 3e-5, and early stopping on validation (best within 100 training iterations) — [arXiv 2403.14403 HTML](https://arxiv.org/html/2403.14403v2)
- Label construction:
  - (1) Run all three strategies on a query. If the simplest (no retrieval) gets the answer right, the label is A; otherwise B if single-step is right; otherwise C ("higher priority to a simpler model").
  - (2) For queries that no strategy answers correctly, use dataset inductive bias: label B if the query comes from a single-hop dataset (SQuAD, NQ, TriviaQA) and C if it comes from a multi-hop dataset (MuSiQue, HotpotQA, 2Wiki).
  — [arXiv 2403.14403 HTML](https://arxiv.org/html/2403.14403v2)
- Repo: separate classifiers per generator (FLAN-T5-XL, FLAN-T5-XXL, GPT-3.5) via `run_large_train_{xl,xxl,gpt}.sh` on `t5-large`. Silver labels come from `nor_qa` / `oner_qa` / `ircot_qa` predictions, and binary labels from dataset origin. They are merged by `concat_binary_silver_train.py`, and preprocessed data is shipped in `data.tar.gz` — [GitHub starsuzi/Adaptive-RAG](https://github.com/starsuzi/Adaptive-RAG)
- Training data: 400 queries sampled per dataset from 6 datasets, not overlapping the test queries. Retriever: BM25. Corpora: Wikipedia for single-hop and Trivedi et al.'s preprocessed corpora for multi-hop — [arXiv 2403.14403 HTML](https://arxiv.org/html/2403.14403v2)
- **HotpotQA (FLAN-T5-XL):**

  | Method | EM | F1 | Acc | Steps | Time/query |
  |---|---|---|---|---|---|
  | Multi-step | 44.60 | 56.54 | 47.00 | 5.53 | 9.38 |
  | Adaptive-RAG | 42.00 | 53.82 | 44.40 | 3.55 | 5.99 |
  | Oracle | 51.20 | 64.00 | 54.80 | 1.59 | 2.77 |

  Adaptive-RAG therefore cuts steps and time by about 36% at a cost of about 2.7 F1 against always-multi-step. The oracle router would beat always-multi-step by +7.5 F1 at far lower cost — [arXiv 2403.14403 HTML](https://arxiv.org/html/2403.14403v2). The same Adaptive-RAG HotpotQA numbers (42.00/53.82/44.40/3.55) are reproduced in [MBA-RAG](https://arxiv.org/html/2412.01572v2).
- Multi-hop average (FLAN-T5-XL) for Adaptive-RAG: F1 46.94, 2.17 steps, 3.60 s. With an oracle classifier (average over all datasets): F1 56.28, EM 45.00, 1.28 steps, 2.11 s — [arXiv 2403.14403 HTML](https://arxiv.org/html/2403.14403v2)
- **Classifier accuracy is only 54.52%** (FLAN-T5-XL labels) — [arXiv 2403.14403 HTML](https://arxiv.org/html/2403.14403v2)
  - Per-class (as extracted from Fig. 3; verify): No retrieval 30.52%, One-step 66.28%, Multi-step 65.45%.
  - Confusion: about 47% of A is predicted as B, about 31% of C as B, and about 23% of B as C.

**MBA-RAG (Tang et al., COLING 2025; arXiv 2412.01572): multi-armed bandit instead of a classifier**
- DistilBERT encoder outputs per-arm value estimates. The arms are the retrieval strategies (no / single / multi-step). Exploration is ε-greedy. Reward r_a = Acc(y, ŷ_a) − λ·C(a), which penalizes steps. The encoder is trained with squared-error loss between predicted and observed reward, learning rate 5e-5 — [arXiv 2412.01572 HTML](https://arxiv.org/html/2412.01572v2)
- HotpotQA (FLAN-T5-XL): MBA-RAG reaches EM 40.60 / F1 52.44 / Acc 42.60 with **2.25 steps**, against Adaptive-RAG's 42.00 / 53.82 / 44.40 with 3.55 steps. It reduces step cost by more than 20% across the multi-hop datasets — [arXiv 2412.01572 HTML](https://arxiv.org/html/2412.01572v2); [GitHub FUTUREEEEEE/MBA](https://github.com/FUTUREEEEEE/MBA)

**Probing-RAG (Baek et al., Findings of NAACL 2025; arXiv 2410.13339): hidden-state prober**
- Generator: Gemma-2B (18 layers). Probers are 1-hidden-layer feed-forward binary classifiers on the residual-stream "post" position of even layers from layer 6 onward. The decision sums the prober logits. Size is about 5 MB, which the authors say is about 2,000× smaller than Adaptive-RAG's classifier — [arXiv 2410.13339 HTML](https://arxiv.org/html/2410.13339v1)
- Labels: y = 1 (no retrieval needed) if the model's answer is correct, y = 0 if it is incorrect. Examples are collected both with and without retrieval. Training set: 26,060 train and 500 val examples from HotpotQA, NQ, and TriviaQA — [arXiv 2410.13339 HTML](https://arxiv.org/html/2410.13339v1)
- HotpotQA (Gemma-2B), accuracy / EM:
  - Probing-RAG 39.32 / 21.8, with 57.46% of queries skipping retrieval
  - Single-step 28.34 / 14.6
  - Adaptive-RAG 23.55 / 13.2
  - DRAGIN 22.55 / 19.8
  - FLARE 20.96 / 13.2

  Retrieval-call counts were extracted approximately (Probing-RAG about 400, Adaptive-RAG about 3,068, DRAGIN about 13,570, FLARE about 5,317); verify in the paper — [arXiv 2410.13339 HTML](https://arxiv.org/html/2410.13339v1)

**DRAGIN (Su et al., ACL 2024; arXiv 2403.10081): token-level dynamic retrieval**
- RIND trigger score per generated token: S = H_i · a_max(i) · s_i, where H_i is token entropy, a_max(i) is the maximum attention the token receives from later tokens, and s_i is a stopword mask. Retrieval fires when S > θ. QFS builds the query from the top-n tokens by self-attention. BM25 beat SGPT dense retrieval — [arXiv 2403.10081 HTML](https://arxiv.org/html/2403.10081)
- HotpotQA (LLaMA2-13B-chat), EM / F1:
  - No-RAG 0.223 / 0.310
  - Single-round RAG 0.263 / 0.371
  - FLARE 0.180 / 0.276
  - FL-RAG 0.177 / 0.268
  - **DRAGIN 0.314 / 0.424**

  Also evaluated on 2Wiki, StrategyQA, IIRC — [arXiv 2403.10081 HTML](https://arxiv.org/html/2403.10081)

**SeaKR (Yao et al., ACL 2025; arXiv 2406.19215)**
- Extracts self-aware uncertainty from the LLM's internal states. It retrieves when uncertainty is high, re-ranks snippets by how much they reduce uncertainty, and picks reasoning strategies by uncertainty. Reported **HotpotQA F1 39.7%, +5.5 over the best baseline** (search-snippet level; verify the table) — [arXiv 2406.19215](https://arxiv.org/abs/2406.19215); [ACL Anthology](https://aclanthology.org/2025.acl-long.1312/)

**"Adaptive Retrieval Without Self-Knowledge? Bringing Uncertainty Back Home" (Moskvoretskii et al., arXiv 2501.12835, Feb 2025): the key benchmark study**
- LLaMA-3.1-8B-Instruct on 500-question subsets of SQuAD, NQ, TriviaQA, MuSiQue, HotpotQA, and 2Wiki. Compares 8 end-to-end adaptive methods (IRCoT, Adaptive-RAG, FLARE, DRAGIN, Rowen×3, SeaKR) with 27 uncertainty estimators: logit-based (max/mean token entropy, probabilities), consistency-based (semantic entropy, lexical similarity, EigValLaplacian), internal-state (Mahalanobis), and hybrid — [arXiv 2501.12835 HTML](https://arxiv.org/html/2501.12835)
- **HotpotQA In-Accuracy / LM calls / retrieval calls:**

  | Method | In-Acc | LM calls | Retrieval calls |
  |---|---|---|---|
  | Mean-entropy UE | 0.414 | 2.0 | 0.99 |
  | Max-entropy UE | 0.414 | 2.0 | 0.99 |
  | Lexical-similarity UE | 0.410 | 2.0 | 0.95 |
  | Adaptive-RAG | 0.414 | 4.6 | 2.34 |
  | DRAGIN | 0.430 | 5.1 | 2.56 |

  — [arXiv 2501.12835 HTML](https://arxiv.org/html/2501.12835)
- Self-knowledge accuracy on HotpotQA: mean entropy 0.73, lexical similarity 0.73, SeaKR 0.71. The uncertainty methods need "fewer than one retriever call and two or less LM calls per question", and lose less than 4% QA performance out-of-domain — [arXiv 2501.12835 HTML](https://arxiv.org/html/2501.12835)

**Popularity-based adaptive retrieval (Mallen et al., ACL 2023; arXiv 2212.10511)**
- Retrieves only for low-popularity entities, using a Wikipedia page-view threshold per relation type. On PopQA (14k long-tail questions) this improved accuracy by 5.3% and halved API cost — [search summary / arXiv 2212.10511](https://arxiv.org/pdf/2212.10511); [GitHub AlexTMallen/adaptive-retrieval](https://github.com/AlexTMallen/adaptive-retrieval)

**Lightweight query-type routing (Bansal & Agarwal, arXiv 2604.03455, Apr 3 2026)**
- RAGRouter-Bench: 7,727 queries (MuSiQue, QuALITY, UltraDomain legal, GraphRAG-Bench medical) with labels factual/single-hop, reasoning/multi-hop, and summary.
- Five classical classifiers × three feature sets (TF-IDF with 3k uni+bigrams; MiniLM-L6 384-d embeddings; 23 hand-crafted structural features).
- Best: **TF-IDF + SVM, macro-F1 0.928, accuracy 93.2%**, 28.1% simulated token savings (perfect labels would save 35.2%). TF-IDF beats MiniLM by 3.1 macro-F1 and structural features by 14.0. No HotpotQA results — [arXiv 2604.03455 HTML](https://arxiv.org/html/2604.03455)

**Other adaptive-retrieval methods [unverified]**
- Self-RAG (Asai et al., ICLR 2024): reflection tokens [Retrieve], [IsREL], [IsSUP], [IsUSE], trained into the generator using critic-distilled labels. Main evaluations are PopQA, TriviaQA, PubHealth, ARC, ASQA, and bio generation, not HotpotQA — [arXiv 2310.11511](https://arxiv.org/abs/2310.11511)
- FLARE (Jiang et al., EMNLP 2023): generates a lookahead sentence and retrieves if any token probability falls below a threshold — [arXiv 2305.06983](https://arxiv.org/abs/2305.06983)
- SKR (Wang et al., Findings EMNLP 2023): self-knowledge elicitation (prompting, kNN over training questions, or a classifier) to decide whether to retrieve — [arXiv 2310.05002](https://arxiv.org/abs/2310.05002)
- CtrlA (2024): representation-engineering "honesty/confidence" directions control retrieval — [arXiv 2405.18727](https://arxiv.org/abs/2405.18727)
- IRCoT (Trivedi et al., ACL 2023): the multi-step strategy Adaptive-RAG routes to — [arXiv 2212.10509](https://arxiv.org/abs/2212.10509)
- Newer 2025–2026 items seen in search but not read:
  - SymRAG — [arXiv 2506.12981](https://arxiv.org/pdf/2506.12981)
  - Decide-Then-Retrieve — [arXiv 2601.03908](https://arxiv.org/pdf/2601.03908)
  - Skill-RAG (hidden-state probing + skill routing) — [arXiv 2604.15771](https://arxiv.org/abs/2604.15771)
  - RealRoute — [arXiv 2604.20860](https://arxiv.org/pdf/2604.20860)
  - Hi-Q — [arXiv 2608.30468](https://arxiv.org/pdf/2608.30468)
  - "Fast or Better?" (user-controlled cost/accuracy RAG) — [arXiv 2502.12145](https://arxiv.org/pdf/2502.12145)

### Inferences
- **The Adaptive-RAG classifier is weak (about 54.5% accuracy)**, yet it still gives most of the efficiency gain. The large oracle gap (HotpotQA F1 53.8 → 64.0) is the headroom a better router could capture, which makes it a natural student-project target.
- The labels are highly imbalanced and noisy. "No strategy correct" queries get a dataset-origin label, so many HotpotQA training labels are simply "C". A HotpotQA-only router has to rely on the silver labels and should consider soft/relaxed labels or F1-based thresholds instead of EM.
- Strong cheap baselines to beat:
  - (a) always-single-step vs always-multi-step;
  - (b) a mean-token-entropy threshold on a no-retrieval pass (matches Adaptive-RAG accuracy with about 2 LM calls);
  - (c) TF-IDF + linear SVM/LogReg on question text;
  - (d) a hidden-state prober if an open-weights generator is used.
- Numbers are not comparable across papers because the generator differs (FLAN-T5-XL, LLaMA2-13B, Gemma-2B, LLaMA-3.1-8B), as do the metric (EM, F1, In-Acc) and the subset (500 questions vs full dev).

### Gaps
- The exact per-class accuracies and confusion matrix for Adaptive-RAG were extracted from a figure by an automated summarizer. Verify them against Fig. 3 of the PDF.
- Adaptive-RAG's batch size and the exact number of labeled training examples per class were not found (the README omits hyperparameters).
- SeaKR's full HotpotQA table, Self-RAG HotpotQA numbers (likely none), and CtrlA results were not retrieved.
- The 2026 follow-ups (Skill-RAG, RealRoute, Hi-Q, SymRAG, Decide-Then-Retrieve) were identified but not read, so their methods and numbers are unknown here.

---

## Q3. Uncertainty- or confidence-based routing, calibration, and selective prediction

### Takeaway
Cheap uncertainty signals are competitive routers for deciding when to escalate on HotpotQA: token entropy on a no-retrieval pass, lexical/semantic consistency across samples, self-verification by sampling (AutoMix), or a small regression verifier (FrugalGPT). The Moskvoretskii et al. study found that mean-token-entropy matched Adaptive-RAG accuracy on HotpotQA with half the LM calls. For generative LMs, token-level quantile features beat raw sequence probability because sequence probability suffers from length bias (Gupta et al., ICLR 2024).

### Cited Findings
- Language Model Cascades: Token-level uncertainty and beyond (Gupta et al., Google; ICLR 2024).
  - Sequence-level uncertainty has **length bias**.
  - **Quantiles of token-level uncertainty** give better cost-quality deferral.
  - A post-hoc deferral rule trained on quantile features beats the other strategies for FLAN-T5 cascades.
  — [arXiv 2404.10136](https://arxiv.org/abs/2404.10136); [ICLR PDF](https://proceedings.iclr.cc/paper_files/paper/2024/file/11f5520daf9132775e8604e89f53925a-Paper-Conference.pdf)
- AutoMix's verifier samples a 4-shot entailment prompt k=8 times and uses the fraction judged correct as confidence. It routes by a threshold or a POMDP whose observation model is fit by KDE — [arXiv 2310.12963 HTML](https://arxiv.org/html/2310.12963v5)
- FrugalGPT's scorer is a DistilBERT regressor on (query, answer) that predicts correctness. It accepts an answer when the score exceeds a per-stage threshold (e.g., 0.96, then 0.37) — [FrugalGPT PDF](https://arxiv.org/pdf/2305.05176)
- DRAGIN's retrieval trigger is entropy × attention × non-stopword per token — [arXiv 2403.10081 HTML](https://arxiv.org/html/2403.10081)
- 27 uncertainty estimators on HotpotQA (LLaMA-3.1-8B):
  - Mean/max token entropy reach 0.414 In-Acc with 2.0 LM calls and about 0.99 retrieval calls. Lexical similarity reaches 0.410.
  - Consistency methods are strong on QA but weaker on self-knowledge. Internal-state methods are efficient but overconfident.
  - Self-knowledge accuracy is about 0.73 for the best estimators.
  — [arXiv 2501.12835 HTML](https://arxiv.org/html/2501.12835)
- Hidden-state probing (Probing-RAG) gives a trained confidence signal from intermediate layers, with correctness labels — [arXiv 2410.13339 HTML](https://arxiv.org/html/2410.13339v1)
- Related 2024–2026 items seen in search, not read:
  - LLM Cascade with Multi-Objective Optimal Consideration — [arXiv 2410.08014](https://arxiv.org/html/2410.08014v1)
  - Cascade-Aware Training of LMs — [arXiv 2406.00060](https://arxiv.org/pdf/2406.00060)
  - "Is Escalation Worth It? A Decision-Theoretic Characterization of LLM Cascades" (2026) — [arXiv 2605.06350](https://arxiv.org/pdf/2605.06350)
  - "Probabilities of Chat LLMs Are Miscalibrated but Still Predict Correctness on MC Q&A" — [arXiv 2402.13213](https://arxiv.org/pdf/2402.13213)
- Foundational methods [unverified]:
  - Semantic entropy (Kuhn et al., ICLR 2023): cluster sampled answers by bidirectional entailment, then take entropy over the clusters — [arXiv 2302.09664](https://arxiv.org/abs/2302.09664)
  - P(True) / self-evaluation (Kadavath et al., 2022) — [arXiv 2207.05221](https://arxiv.org/abs/2207.05221)
  - Verbalized confidence (Tian et al., EMNLP 2023) — [arXiv 2305.14975](https://arxiv.org/abs/2305.14975)
  - Temperature scaling (Guo et al., ICML 2017): a single scalar T fit on validation NLL to divide the logits — [arXiv 1706.04599](https://arxiv.org/abs/1706.04599)
  - Selective classification / risk-coverage (Geifman & El-Yaniv, NeurIPS 2017) — [arXiv 1705.08500](https://arxiv.org/abs/1705.08500)
  - Mixture-of-Thought cascades using answer consistency (Yue et al., ICLR 2024) — [arXiv 2310.03094](https://arxiv.org/abs/2310.03094)

### Inferences
- A practical HotpotQA escalation router works like this:
  - Run the cheap path (e.g., no retrieval or single-step with a small LLM).
  - Compute features: mean/max/quantile token entropy of the answer, agreement across k samples (lexical or semantic), answer length, and question features.
  - Fit a calibrated classifier (logistic regression plus temperature/Platt scaling) that predicts "cheap answer is correct (F1 ≥ τ)".
  - Escalate when the predicted probability is below a threshold chosen on validation to hit a target coverage or budget.
  - Report risk-coverage curves (the error rate of accepted answers against the fraction accepted) alongside cost-accuracy curves.
- Calibration matters mainly because the threshold must transfer from validation to test. For a binary router, temperature/Platt scaling on held-out data is cheap and standard.

### Gaps
- No paper was found that reports semantic entropy, P(True), or verbalized confidence specifically as HotpotQA routers with calibration (ECE) numbers. The Moskvoretskii study reports ROC-AUC and self-knowledge but not ECE for HotpotQA in the extracted summary.
- Gupta et al. (2024) did not use HotpotQA in the extracted summary. Their datasets were not verified.

---

## Q4. Mixture-of-experts routing inside a model (Switch, top-k gating, load balancing, MoE-LoRA) and type-specific experts

### Takeaway
Token-level MoE routing (Switch/GShard/top-k gating with an auxiliary load-balancing loss) and mixture-of-LoRA-experts variants are well established. Recent work focuses on dynamic or confidence-adaptive expert counts. No paper was found that builds explicit **question-type experts (bridge vs comparison) for HotpotQA**. The closest evidence is older HotpotQA systems that classify question type and route to type-specific pipelines.

### Cited Findings
- Switch Transformer (Fedus, Zoph, Shazeer; arXiv Jan 2021, JMLR 2022) uses top-1 expert routing and reports up to 7× pre-training speedup over T5-Base/Large and 4× over T5-XXL — [arXiv 2101.03961](https://arxiv.org/abs/2101.03961)
- The auxiliary load-balancing loss [unverified] is L_aux = α·N·Σ_i f_i·P_i with α = 10⁻². Here f_i is the fraction of tokens dispatched to expert i and P_i is the mean router probability for expert i. The paper also uses an expert capacity factor with token dropping — [arXiv 2101.03961](https://arxiv.org/abs/2101.03961)
- Mixture-of-LoRA-experts landscape, 2024–2026:
  - Mixture-of-LoRAs for multitask tuning — [arXiv 2403.03432](https://arxiv.org/pdf/2403.03432)
  - DynMoLE: hybrid routing — [arXiv 2504.00661](https://arxiv.org/pdf/2504.00661)
  - LD-MoLE: a shared MLP learns per-token/per-layer sparsity — [arXiv 2509.25684](https://arxiv.org/html/2509.25684)
  - LoRA-Mixer: routes LoRA experts in attention projections — [arXiv 2507.00029](https://arxiv.org/html/2507.00029v2)
  - HotMoE: hybrid routing, with expert-expert attention in lower layers and token-expert attention in higher layers — [AAAI](https://ojs.aaai.org/index.php/AAAI/article/view/40383)
  - CARE (Jul 2026): nucleus-style expert admission until cumulative router mass reaches a threshold, extended when experts disagree — [arXiv 2607.26052](https://arxiv.org/abs/2607.26052)
- Two paradigms for combining LoRA experts: static parameter merging (e.g., LoRAHub) vs dynamic output ensembling with a router — [search summary citing LoRA-MoE literature](https://arxiv.org/html/2507.00029v2)
- HotpotQA has **bridge** (entity-chaining) and **comparison** (extract-compare-choose) questions — [HotpotQA paper, EMNLP 2018](https://aclanthology.org/D18-1259.pdf)
- A Stanford CS224N HotpotQA report argues for pipelines specialized per question type: direct retrieval works better for comparison questions, and graph/GNN approaches may suit bridge questions (student report, low authority) — [CS224N report](https://web.stanford.edu/class/archive/cs/cs224n/cs224n.1194/reports/custom/15743318.pdf)
- [unverified]:
  - Sparsely-gated MoE with noisy top-k gating and an importance/load loss (Shazeer et al., 2017) — [arXiv 1701.06538](https://arxiv.org/abs/1701.06538)
  - GShard top-2 routing — [arXiv 2006.16668](https://arxiv.org/abs/2006.16668)
  - MoLoRA / MoV (Zadouri et al., 2023): soft-merged LoRA experts with a token-level router for instruction tuning — [arXiv 2309.05444](https://arxiv.org/abs/2309.05444)
  - LoraHub (Huang et al., 2023): gradient-free weighted composition of task LoRAs — [arXiv 2307.13269](https://arxiv.org/abs/2307.13269)
  - X-LoRA (Buehler & Buehler, 2024): hidden-state-driven dynamic mixing of LoRA adapters — [arXiv 2402.07148](https://arxiv.org/abs/2402.07148)

### Inferences
- A student team does not need token-level MoE. An "experts by question type" design can be built as a query-level router: a classifier over {bridge, comparison} using HotpotQA's `type` field as free labels (train split is about 80/20 bridge/comparison). That classifier can then choose between type-specific prompts, LoRA adapters, or retrieval strategies. The labels are clean and easy to learn, but whether type-specific experts actually help on HotpotQA is untested in the literature found.
- If LoRA experts are used, the load-balancing loss matters only for learned token-level gating. With a hard query-level type router, it is unnecessary.

### Gaps
- No peer-reviewed paper was found that trains bridge-vs-comparison experts or type-routed LoRAs on HotpotQA and reports gains.
- The Switch load-balancing formula and α are unverified (from recall).

---

## Q5. Which of these report HotpotQA results, and what are the numbers?

### Takeaway
HotpotQA results exist mainly for adaptive-retrieval routers (Adaptive-RAG, MBA-RAG, DRAGIN, SeaKR, Probing-RAG, the Moskvoretskii uncertainty study) and for the LLM-as-router Router-R1. The classic model routers (FrugalGPT, RouteLLM, Hybrid LLM, AutoMix, RouterBench) do not report on HotpotQA. None of the numbers below are directly comparable, because generators, metrics, and subsets differ.

### Cited Findings
| Method (paper date) | Generator | HotpotQA result | Cost signal | Source |
|---|---|---|---|---|
| Adaptive-RAG (NAACL 2024) | FLAN-T5-XL 3B | EM 42.00 / F1 53.82 / Acc 44.40 | 3.55 steps, 5.99 s (multi-step: 5.53 steps, 9.38 s; F1 56.54) | [arXiv 2403.14403](https://arxiv.org/html/2403.14403v2) |
| Adaptive-RAG oracle router | FLAN-T5-XL | EM 51.20 / F1 64.00 / Acc 54.80 | 1.59 steps, 2.77 s | [arXiv 2403.14403](https://arxiv.org/html/2403.14403v2) |
| MBA-RAG (COLING 2025) | FLAN-T5-XL | EM 40.60 / F1 52.44 / Acc 42.60 | 2.25 steps | [arXiv 2412.01572](https://arxiv.org/html/2412.01572v2) |
| DRAGIN (ACL 2024) | LLaMA2-13B-chat | EM 0.314 / F1 0.424 (FLARE 0.180/0.276; single-round 0.263/0.371) | retrieval triggered per token | [arXiv 2403.10081](https://arxiv.org/html/2403.10081) |
| SeaKR (ACL 2025) | (LLaMA-2 family; verify) | F1 39.7, +5.5 over best baseline | — | [arXiv 2406.19215](https://arxiv.org/abs/2406.19215) |
| Probing-RAG (NAACL Findings 2025) | Gemma-2B | Acc 39.32 / EM 21.8 (Adaptive-RAG 23.55/13.2; DRAGIN 22.55/19.8) | 57% of queries skip retrieval | [arXiv 2410.13339](https://arxiv.org/html/2410.13339v1) |
| Mean-entropy UE (Moskvoretskii 2025) | LLaMA-3.1-8B | In-Acc 0.414 (Adaptive-RAG 0.414; DRAGIN 0.430) | 2.0 LM / 0.99 retrieval calls (Adaptive-RAG 4.6 / 2.34; DRAGIN 5.1 / 2.56) | [arXiv 2501.12835](https://arxiv.org/html/2501.12835) |
| Router-R1 (NeurIPS 2025) | Qwen2.5-3B router over 6 LLMs | EM 0.352 (GraphRouter 0.244; Prompt-LLM 0.268; Search-R1 0.236) | more API calls on multi-hop than single-hop | [arXiv 2506.09033](https://arxiv.org/html/2506.09033) |
| RAGRouter (NeurIPS 2025) | 15 LLMs | HotpotQA used for cross-domain generalization only (numbers not extracted) | latency-aware threshold | [arXiv 2505.23052](https://arxiv.org/html/2505.23052) |

### Inferences
- Recommended design pattern for a student HotpotQA router, synthesized from the above:
  - **Actions**: {no-retrieval, single-step, multi-step} × optionally {small LLM, large LLM}.
  - **Labels**: run all actions on a train subset (400–2,000 questions is the scale Adaptive-RAG used), then label each question with the cheapest action whose F1 ≥ τ (or whose EM is correct). Fall back to the most expensive action when none succeed, or use soft labels (Hybrid-LLM r_prob/r_trans).
  - **Features**: question text via TF-IDF, DistilBERT/DeBERTa/T5 encoders, or MiniLM embeddings; HotpotQA `type` (bridge/comparison) as an auxiliary feature or label; optionally post-hoc features such as token entropy from a cheap pass.
  - **Model**: a small encoder classifier (cross-entropy) or a reward-regression/bandit (MBA-RAG style, reward = F1 − λ·steps).
  - **Evaluation**: EM/F1 against average steps, latency, or $ cost, with a threshold sweep; APGR/AIQ-style area metrics; oracle and random routers as bounds; router accuracy and confusion matrix.
- The most important baselines to beat on HotpotQA are always-multi-step (accuracy ceiling for fixed strategies), Adaptive-RAG (published router), and an entropy-threshold router (cheap and strong per Moskvoretskii).

### Gaps
- No single paper reports all methods on HotpotQA with the same generator and metric. A fair comparison needs re-running them.
- The SeaKR generator and exact HotpotQA table, Self-RAG HotpotQA numbers, and RAGRouter's HotpotQA cross-domain numbers were not extracted.
- Adaptive-RAG results with FLAN-T5-XXL and GPT-3.5 on HotpotQA specifically were not extracted, only the FLAN-T5-XL row.
