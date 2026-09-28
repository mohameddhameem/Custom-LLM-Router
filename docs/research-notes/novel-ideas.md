# Novel, Feasible Coursework Directions in Multi-hop QA (HotpotQA / 2WikiMultiHopQA / MuSiQue): Routing, Evidence Selection, Calibration, Training Objectives

Research date: 2026-09-27. Paper dates are arXiv first-submission dates unless noted.
Sources tagged **[bg]** are from recall; check their arXiv IDs before citing. Other sources were retrieved on 2026-09-27; several 2026 preprints were read at abstract level only.

---

## Q1. What already exists for adaptive routing and escalation in multi-hop QA, and what is the closest prior work to an "evidence-aware sufficiency gate"?

### Takeaway
Query-only complexity routing (Adaptive-RAG, 2024) and post-retrieval escalation routers built on shallow retrieval features (RASER, June 2026) already exist on exactly HotpotQA/2Wiki/MuSiQue. So "a router for multi-hop RAG" is **not** novel on its own. What remains open, and is testable on a single GPU: (i) a *learned cross-encoder sufficiency score* as the routing signal, checked against a **matched query-only control**; (ii) **cross-dataset transfer** of such routers; (iii) **distribution-free risk guarantees** on the escalation decision.

### Cited Findings
- **Adaptive-RAG (Jeong et al., NAACL 2024; arXiv Mar 2024)** trains a T5-Large classifier on automatically derived silver labels (Straightforward / Simple / Complex) to choose between no-retrieval, single-step and iterative RAG. Labels come from observed model behavior, not manual annotation. — [arXiv 2403.14403](https://arxiv.org/pdf/2403.14403); summary in [RAGRouter-Bench paper](https://arxiv.org/html/2604.03455v1)
- Criticism of Adaptive-RAG-style routers: "External classifiers in Adaptive-RAG often fail to fully leverage the internal decision-making capabilities of the language model, leading to unnecessary additional retrieval steps." This motivated **Probing-RAG** (Oct 2024), which reads LLM hidden states to make retrieve/no-retrieve decisions. — [Probing-RAG, arXiv 2410.13339](https://arxiv.org/html/2410.13339v1)
- **RASER (Li, Yan, Käfer; arXiv 2606.02488, June 1 2026)** is the closest prior work to a "post-retrieval escalation gate." It runs after one-shot RAG and uses a scikit-learn Gradient Boosting Machine (100 trees, depth 3) on six features: rule-based confidence, answer length, bridge cues (e.g. "X of the Y of Z" patterns), score_gap (top-1 minus top-5 retrieval score), score_top1 (cosine similarity), and question type. RASER-2 decides stop vs escalate; RASER-3 chooses among one-shot RAG, PRUNE bridge retrieval, and IRCoT. It makes no extra LLM calls. Evaluated on MuSiQue, 2Wiki and HotpotQA with six LLMs (GPT-OSS-120B down to Phi-4-mini). RASER-2 keeps F1 competitive at 39–57% of always-PRUNE's tokens; RASER-3 reaches 87–97% of IRCoT F1 at 47–62% of the token cost. **It uses no cross-encoder sufficiency score, no conformal guarantee, and no cross-dataset transfer** (5-fold CV within each dataset). Stated limitations: simplified baselines, only 200–500 questions per LLM/dataset (noisy), and "large LLMs sometimes answer from memory rather than retrieved passages." — [RASER HTML](https://arxiv.org/html/2606.02488)
- **Lightweight Query Routing on RAGRouter-Bench (Bansal & Agarwal; arXiv 2604.03455, Apr 2026)** compares TF-IDF, MiniLM embeddings and structural features. TF-IDF+SVM gives the best result (0.928 macro-F1, 28.1% simulated token savings), and TF-IDF beats MiniLM embeddings by 3.1 macro-F1. The router is **query-side only**, and the authors say their results "highlight the gap that corpus-aware routing must close." — [arXiv 2604.03455](https://arxiv.org/abs/2604.03455)
- **Important negative result to design around: "Beyond the Query: Do Retrieval Signals Improve Adaptive Multimodal RAG Routing?" (Li, Zhang, Ming; arXiv 2609.12437, Sept 11 2026).** In document, audio and video RAG, adding retrieval-state features to a matched query-only router did **not** consistently improve RUN/SKIP decisions on held-out data. The methodological rule it proposes: "retrieval-state features should not be credited with routing value unless they improve over a matched query-only control." This is multimodal, not text multi-hop QA. — [arXiv 2609.12437](https://arxiv.org/abs/2609.12437)
- **RegimeRouter (Bacellar; arXiv 2604.09019, Apr 2026)** is a binary classifier on 5 surface text features that chooses question-only vs question+relation-sentence retrieval. Trained on 2Wiki (n=881), it transfers zero-shot with +5.3 pp R@5 on MuSiQue (p=0.002) but only +1.1 pp (not significant) on HotpotQA. The author notes the gains may be "artifact-driven." — [arXiv 2604.09019](https://arxiv.org/abs/2604.09019)
- **R²Adapter (Guo et al.; arXiv 2609.02894, July 2026)** routes queries between vanilla RAG and graph-based RAG, and rewrites uncertain graph-routed queries. It cuts graph-RAG usage by up to 59% at comparable accuracy on three multi-hop benchmarks. — [arXiv 2609.02894](https://arxiv.org/abs/2609.02894)
- **MBA-RAG (Dec 2024)** uses a multi-armed bandit for adaptive retrieval strategy selection. — [arXiv 2412.01572](https://arxiv.org/pdf/2412.01572)
- **EfficientRAG (Zhuang et al., EMNLP 2024)** uses small Labeler and Filter models to produce next-hop queries and filter chunks without an LLM call per iteration. It reports 60–80% time-efficiency gains on HotpotQA, MuSiQue and 2Wiki. Its core insight is that "the types of relations in multi-hop questions are limited compared to entities," so small models can handle them. — [ACL Anthology 2024.emnlp-main.199](https://aclanthology.org/2024.emnlp-main.199/); [arXiv 2408.04259](https://arxiv.org/abs/2408.04259)
- **Sufficient Context (Joren et al., Google; ICLR 2025; arXiv Nov 2024)** used 500 HotpotQA, 500 MuSiQue-Ans and 452 FreshQA instances. About 56% of HotpotQA and about 46% of MuSiQue instances had *insufficient* retrieved context. The autoraters scored: Gemini 1.5 Pro 1-shot 93% accuracy, FLAMe-24B 87.8%, TRUE-NLI 82.6% (needs gold answer), "contains GT" 80.9%. Their selective generation feeds the sufficiency label plus self-rated confidence into logistic regression to *abstain*, which raises the fraction of correct answers among those given by 2–10%. It does **not** route to a larger model. Stated future work: "a fine-tuned entailment model," fine-grained (continuous) sufficiency scoring, and iterative retrieval. — [arXiv HTML 2411.06037](https://arxiv.org/html/2411.06037); [GitHub](https://github.com/hljoren/sufficientcontext); [Google blog](https://research.google/blog/deeper-insights-into-retrieval-augmented-generation-the-role-of-sufficient-context/)
- **Knowing When to Stop (Xie et al., NeurIPS 2025; arXiv 2502.01025)** finds that middle-layer attention heads encode "sufficiency signals" detectable by lightweight probes, giving +3.4% accuracy at 1.33× token reduction on 6 QA datasets with 1B–70B models. Probing (F1 91.1) beats a supervised fine-tuned classifier (79.5) and self-prompting (83.1) at 70B. 1B models are poor at self-prompting (F1 52.6), and "fine-tuning underperforms universally, likely due to misaligned sufficiency signals across models." — [arXiv 2502.01025](https://arxiv.org/abs/2502.01025); [NeurIPS PDF](https://proceedings.neurips.cc/paper_files/paper/2025/file/3afa7f9b7ac8da474b3d915570d58291-Paper-Conference.pdf)
- **Skill-RAG (Wei et al.; arXiv 2604.15771, Apr 2026, rev. Aug 2026)** uses a hidden-state prober to detect failure states, then a prompt-based skill router picks query rewriting, decomposition, evidence focusing, or exit. It finds the skills "occupy structured, separable regions of the failure state space," with strong OOD gains. — [arXiv 2604.15771](https://arxiv.org/html/2604.15771)
- Cross-encoder score-band escalation (auto-accept / auto-reject / escalate the middle band) is an established pattern in other domains, e.g. product linking with VLM-distilled cross-encoders (arXiv 2608.25037). There, "the escalation rate dominates aggregate spend." — [Retrieve, Match, Escalate](https://arxiv.org/pdf/2608.25037)
- **Cascade theory:** "Is Escalation Worth It?" (Bouchard; arXiv 2605.06350, May 2026) shows that cascade limitations come mainly from *structural cost*, i.e. paying for the cheap model before any escalation decision. Validated on MATH, MMLU, TriviaQA, SimpleQA and LiveCodeBench, with no multi-hop QA. — [arXiv 2605.06350](https://arxiv.org/abs/2605.06350)
- General LLM routing and cascade baselines: **RouteLLM** [bg] ([arXiv 2406.18665](https://arxiv.org/abs/2406.18665)), **FrugalGPT** [bg] ([arXiv 2305.05176](https://arxiv.org/abs/2305.05176)), **AutoMix** [bg] ([arXiv 2310.12963](https://arxiv.org/abs/2310.12963)); **Self-RAG** [bg] ([arXiv 2310.11511](https://arxiv.org/abs/2310.11511)) and **CRAG** [bg] (T5-based retrieval evaluator, [arXiv 2401.15884](https://arxiv.org/abs/2401.15884)).

### Inferences
- The "structural cost" point (Bouchard 2026) favors a *pre-generation* gate. A MiniLM cross-encoder (about 22–33M params) scoring (question, retrieved passages) sufficiency decides escalation *before* the small reader runs, which avoids the cheap-model cost that RASER pays (RASER needs a one-shot RAG draft first).
- Given the 2609.12437 negative result, any team claiming that "evidence signals improve routing" must report a matched query-only control (same classifier family, same labels, same data). That control could be the TF-IDF+SVM baseline from 2604.03455 or a MiniLM query-only classifier.
- Knowing-When-to-Stop's finding that fine-tuned sufficiency classifiers underperform probing was for classifiers predicting a specific LLM's sufficiency. A cross-encoder trained on *gold* HotpotQA supporting-fact labels predicts something model-agnostic ("are both gold paragraphs present?"). That is a different target, but the team should expect a gap between gold-sufficiency and "the small model will get it right." That gap is itself worth measuring.

### Gaps
- No paper was found that uses a small cross-encoder trained on HotpotQA supporting-fact labels as a *sufficiency gate for model escalation* (small reader → large LLM) with a matched query-only control on text multi-hop QA. Absence of evidence is not proof; recommend a final Semantic Scholar / Google Scholar check for "sufficiency" + "cascade" + "HotpotQA" before the claim.
- Only RASER's abstract and HTML summary were checked, not its full related-work list. It may cite additional close work.

---

## Q2. What exists for calibration and conformal risk control in routing and selective QA, and what is missing for multi-hop?

### Takeaway
Conformal RAG (TRAQ, C-RAG, Conformal-RAG), conformal cascades and routers (CP-Router, Conformal Cascade, RouteNLP, RACER, Conformal Arbitrage, LEC) and conformal abstention (COIN, Pause-and-Reflect) are all crowded in 2025–2026. The remaining gaps specific to multi-hop QA are: **open-ended answers** (Conformal Cascade explicitly leaves these to future work), **guarantees under dataset shift** (HotpotQA → MuSiQue/2Wiki), **group-conditional coverage by reasoning type** (bridge vs comparison, 2/3/4-hop), and the finding that **HotpotQA scores separate poorly**, so abstention helps little.

### Cited Findings
- **TRAQ (Li et al., NAACL 2024)** is billed as the first end-to-end statistical correctness guarantee for RAG. It builds prediction sets that contain a semantically correct answer with high probability, and uses Bayesian optimisation to shrink them by 16.2% on average. — [ACL Anthology 2024.naacl-long.210](https://aclanthology.org/2024.naacl-long.210/); [arXiv 2307.04642](https://arxiv.org/abs/2307.04642)
- **C-RAG** applies conformal risk analysis to RAG generation, including distribution shift. — mentioned in [TRAQ search results](https://www.semanticscholar.org/paper/TRAQ:-Trustworthy-Retrieval-Augmented-Question-via-Li-Park/28e078b52a761cf3fe5941d3428acb7bbf608f21); original [bg] [arXiv 2402.03181](https://arxiv.org/abs/2402.03181)
- **Conformal-RAG / conditional conformal factuality (SIGIR 2025)** gives group-conditional coverage across sub-domains and certifies sub-claim factuality with a retrieval-aware relevance score. — [ACM DL 10.1145/3726302.3730244](https://dl.acm.org/doi/10.1145/3726302.3730244); [Pith summary](https://pith.science/paper/2506.20978)
- **Conformal context engineering (Chakraborty et al.; arXiv 2511.17908, Nov 2025)** filters retrieved context with coverage control on the recall of supporting evidence, keeping 2–3× less context. Evaluated on NeuCLIR and RAGTIME, **not HotpotQA**. — [arXiv 2511.17908](https://arxiv.org/abs/2511.17908)
- **CP-Router (Su et al.; AAAI 2026; arXiv May 2025)** routes between an LLM and a reasoning model using CP prediction-set size from LLM output probabilities, plus a Full-and-Binary-Entropy threshold selector. It is training-free and targets MCQ-style outputs. — [arXiv 2505.19970](https://arxiv.org/abs/2505.19970)
- **Conformal Cascade (Dou, Lian, Li; arXiv 2607.25018, July 2026)** defers when the calibrated prediction set is not a singleton. The guarantee is ≥1−Kα (union bound), tightening to 1−α under a selection-preservation condition. It uses 18 **multiple-choice** benchmarks. Limitation: "Extension to open-ended generation requires an answer-clustering step that we leave for future work." — [arXiv 2607.25018](https://arxiv.org/abs/2607.25018)
- **RouteNLP (arXiv 2604.23577, 2026)** does conformal cascading following Blot et al. (2025), plus distillation on failure clusters. **RACER (arXiv 2603.06616)** is a post-hoc risk-control wrapper for any router. **Conformal Arbitrage (arXiv 2506.00911)** works at the API level without logits. **LEC (arXiv 2512.01556)** targets selection-conditioned risk control in selective prediction and routing. — [RouteNLP](https://arxiv.org/html/2604.23577v1); [RACER](https://arxiv.org/pdf/2603.06616); [Conformal Arbitrage](https://arxiv.org/pdf/2506.00911); [LEC](https://arxiv.org/pdf/2512.01556)
- **COIN (arXiv 2506.20178, 2025)** calibrates uncertainty thresholds for selective QA with provable control of the error rate among accepted answers. — [arXiv 2506.20178](https://arxiv.org/pdf/2506.20178)
- **Pause and Reflect (Gu et al.; arXiv 2605.14098, May 2026)** combines conformal risk control with weighted aggregation over CoT paths: 90.1% selective accuracy on GSM8K while abstaining on <5%. On HotpotQA, "the accuracy improvements from abstention are markedly smaller for all score families, … when separability is low, abstention cannot substantially improve selective accuracy." — [arXiv 2605.14098](https://arxiv.org/abs/2605.14098); snippet via [HTML](https://arxiv.org/html/2605.14098)
- **Conformal risk control** (Angelopoulos et al.) [bg] generalises CP to bounded monotone losses. — [arXiv 2208.02814](https://arxiv.org/abs/2208.02814)
- Selective QA under domain shift (Kamath, Jia, Liang, ACL 2020) [bg] shows that confidence calibrators degrade out-of-domain. — [arXiv 2006.09462](https://arxiv.org/abs/2006.09462)

### Inferences
- The HotpotQA low-separability finding (Pause and Reflect) is a concrete, citable motivation. LLM-internal confidence separates correct from incorrect poorly on HotpotQA, so an *external evidence signal* (sufficiency) may be what improves separability. This can be measured directly with AUROC or AURC of each score for predicting small-model correctness.
- Standard split conformal assumes exchangeability. Calibrating on HotpotQA and deploying on MuSiQue breaks it, and the size of the resulting violation for a multi-hop router appears unmeasured. That makes a clean, cheap experiment.

### Gaps
- No work was found that applies conformal risk control to a **two-tier (small reader → large LLM) escalation decision on open-ended multi-hop QA with EM/F1 loss**, with coverage broken down by bridge/comparison or hop count. RouteNLP's full evaluation datasets were not checked, so it may include a QA task.

---

## Q3. Training objectives and RL for small models: what has been done, and what compute does it need?

### Takeaway
Joint answer + supporting-fact multi-task learning is classic (2018–2020). Rationale distillation is established. RL-with-search (Search-R1 family) is crowded, including a *low-budget* variant (David-GRPO on 4× RTX 3090, ≤1.5B) and an RL-trained *router* over LLMs (Router-R1, NeurIPS 2025). A single consumer GPU is enough for supervised cross-encoder or small-reader training and for LoRA-GRPO on ≤1.5B models in the **closed distractor setting** (no live retriever). Full Search-R1 reproduction is out of reach.

### Cited Findings
- **HotpotQA (Yang et al., EMNLP 2018)** already defines supporting-fact supervision and a joint EM/F1 metric that requires both answer and support set to match. — [arXiv 1809.09600](https://arxiv.org/abs/1809.09600); [HotpotQA consistency snippet](https://arxiv.org/pdf/2205.09226)
- Classic joint answer+SP readers include SAE [bg] ([arXiv 1911.00484](https://arxiv.org/abs/1911.00484)) and HGN [bg] ([arXiv 1911.03631](https://arxiv.org/abs/1911.03631)). Beam Retrieval [bg] ([arXiv 2308.08973](https://arxiv.org/abs/2308.08973)) trains a multi-hop retriever end to end.
- **Distilling step-by-step (Hsieh et al., 2023)** trains small models on LLM labels plus rationales and beats larger LLMs with less data. — [arXiv 2305.02301](https://arxiv.org/pdf/2305.02301)
- **Search-R1** is RL (PPO/GRPO) with interleaved search. It reports +24% (Qwen2.5-7B) and +20% (Qwen2.5-3B) average relative gains, trained on NQ+HotpotQA and tested OOD on 2Wiki, MuSiQue and Bamboogle. GRPO converges faster than PPO. — [arXiv 2503.09516](https://arxiv.org/pdf/2503.09516); [GitHub](https://github.com/PeterGriffinJin/Search-R1)
- **R1-Searcher** is two-stage RL on Qwen-2.5-7B-Base, beating GPT-4o-mini baselines by up to 48.22% on HotpotQA. **ZeroSearch** replaces the real search engine with a simulated LLM search engine (a 7B simulator is comparable to Google Search, a 14B one surpasses it). — [R1-Searcher arXiv 2503.05592](https://arxiv.org/pdf/2503.05592); [ZeroSearch arXiv 2505.04588](https://arxiv.org/pdf/2505.04588)
- **Search-R1++ (Xu et al.; arXiv 2602.19526, Feb 2026)**, an ablation study: an EM reward beats F1 (F1 unstable); action-level penalties help; REINFORCE beats PPO; **GRPO had the lowest stability**. Qwen2.5-3B improves 0.289 → 0.331. — [arXiv 2602.19526](https://arxiv.org/abs/2602.19526)
- **David-GRPO ("Can David Beat Goliath?", Han et al.; arXiv 2601.21699, Jan 2026, v3 May 2026)** trains on **four RTX 3090 GPUs** with agents up to 1.5B. Expert-trajectory seeding plus evidence-guided exploration turns partial successes into coverage scores. Prior low-budget RL baselines "often skip retrieval or stop after shallow search." Trained on HotpotQA, evaluated on 6 benchmarks. — [arXiv 2601.21699](https://arxiv.org/abs/2601.21699); [GitHub](https://github.com/AsadalJung/David-GRPO)
- **Stratified GRPO (arXiv 2510.06214, Oct 2025)** notes that search-agent trajectories are structurally heterogeneous (differing number and placement of search calls). It normalises advantages within strata, gaining up to +11.3 points over GRPO. — [arXiv 2510.06214](https://arxiv.org/abs/2510.06214)
- **Tree-GRPO (ICLR 2026)** reports stable gains for Qwen2.5-1.5B and 3B over chain-based RL. — [GitHub](https://github.com/AMAP-ML/Tree-GRPO); [arXiv 2509.21240](https://arxiv.org/pdf/2509.21240)
- **CRAFT (Liu et al.; arXiv 2602.01348, Feb 2026, rev. July 2026)** trains 0.5B/1.5B/7B models with RL using format, answer and citation-validity rewards plus judge-based faithfulness rewards. Accuracy and faithfulness improve at ≥1.5B; "performance at 0.5B remains heavily template-dependent." — [arXiv 2602.01348](https://arxiv.org/abs/2602.01348)
- **Router-R1 (Zhang et al.; NeurIPS 2025; arXiv 2506.09033)** is an RL-trained LLM router that interleaves "think" and "route" actions over a pool of LLMs. It uses format, outcome and **cost** rewards, and its α coefficient trades performance against cost. Evaluated on 7 QA benchmarks including HotpotQA, where it makes more model calls on multi-hop questions. — [arXiv 2506.09033](https://arxiv.org/abs/2506.09033); [GitHub](https://github.com/ulab-uiuc/Router-R1)
- **R2-Reasoner / Route-and-Reason (WWW 2026; arXiv 2506.05901)** has a task decomposer plus a subtask allocator that sends each sub-task to one of 9 models (<1B to hundreds of billions), trained with SFT+RL. It reports 84.46% API-cost reduction on six reasoning benchmarks. — [arXiv 2506.05901](https://arxiv.org/abs/2506.05901); [ACM DL](https://dl.acm.org/doi/10.1145/3774904.3793038)
- Compute data points: TongSearch-QR RL used 4× A800-80G (1 for vLLM, 3 for training), about 16 h for 1.5B and about 48 h for 7B. A CS224R student project ran GRPO on Qwen2.5-3B with 2× A100-80GB. — [TongSearch-QR arXiv 2506.11603](https://arxiv.org/pdf/2506.11603); [CS224R report](https://cs224r.stanford.edu/projects/pdfs/CS224R_final_report%20(5)12.pdf)

### Inferences
- On one 24 GB consumer GPU, the feasible RL regime is roughly Qwen2.5-0.5B/1.5B with LoRA and small groups (e.g. 4–8 rollouts), in HotpotQA's *distractor* setting where the 10 paragraphs are given. That removes the retriever server entirely. Expect instability (Search-R1++ found GRPO least stable), so budget for REINFORCE/RLOO as a fallback.
- Supervised objectives (cross-encoder sufficiency, joint answer+SP+sufficiency heads on a small reader, distillation from LLM soft labels) are the safest core contribution. RL is a stretch goal.

### Gaps
- Exact GPU-hours for Search-R1 / R1-Searcher / ZeroSearch main runs are unknown.
- David-GRPO's wall-clock time on 4× 3090 was not stated in the pages fetched.

---

## Q4. Robustness, faithfulness and shortcut diagnostics: what is known?

### Takeaway
Shortcut and disconnected reasoning on HotpotQA is well documented (2019–2022), and counterfactual multi-hop benchmarks exist (CofCA 2025, CRiT-QA 2026). A fresh (Sept 2026) result says about half of per-hop failures are *extraction* failures that no retrieval intervention fixes. Almost nothing evaluates **routers** under these perturbations. A router can look good when the large LLM answers from parametric memory, as RASER itself observed.

### Cited Findings
- **Min et al. (ACL 2019)** show that compositional (HotpotQA) questions often do not require multi-hop reasoning. — [arXiv 1906.02900](https://arxiv.org/pdf/1906.02900)
- **Adversarial distractors for multi-hop (Jiang & Bansal, ACL 2019)** [bg]. — [arXiv 1906.07132](https://arxiv.org/abs/1906.07132)
- **DiRe (Trivedi et al., EMNLP 2020)** measures and reduces disconnected reasoning. — [arXiv 2005.00789 via awesomepapers](https://awesomepapers.io/llm-papers/papers/2005.00789)
- **MuSiQue (Trivedi et al., TACL 2022)** [bg] was built by composing connected single-hop questions to reduce shortcuts. It has 2–4 hop questions with gold decompositions and an answerable+unanswerable ("Full") variant. — [arXiv 2108.00573](https://arxiv.org/abs/2108.00573)
- **Counterfactual Multihop QA (2022)** takes a cause-effect approach to disconnected reasoning. — [arXiv 2210.07138](https://arxiv.org/abs/2210.07138)
- **CofCA** rewrites documents counterfactually. Counterfactual or knowledge-edited settings reopen a 15–25-point F1/EM gap, which indicates memorisation bias. — [Emergent Mind summary of counterfactual multi-hop benchmarks](https://www.emergentmind.com/topics/counterfactual-perturbation-and-verification) (secondary aggregator; verify against the CofCA paper)
- **CRiT-QA (Yun, Kwon, Kim; LREC 2026; arXiv 2607.10562)** combines counterfactual entity-replaced chains with "multi-anchor distractor chains" that diverge at different hops. LLMs degrade substantially and exploit "single-document cues or type-matching." — [arXiv 2607.10562](https://arxiv.org/abs/2607.10562)
- **Diagnosing the Fact-Grounding Gap (Mo, Mo, Zhu; arXiv 2609.17043, Sept 15 2026):** "Extraction failures account for nearly half of all per-hop deficiencies," these are invisible to retrieval metrics, and they "remain unresolved by every retrieval intervention." Retrieval and extraction failures are "fundamentally different bottlenecks requiring different solutions." — [arXiv 2609.17043](https://arxiv.org/abs/2609.17043)
- Survey-style synthesis: "over 50% of correctly answered questions may have one or more single-hop sub-questions answered incorrectly." Open challenges listed include >2-hop scalability, faithfulness, context-sensitive utility (vs topical relevance), and compute trade-offs. — [Emergent Mind topic page](https://www.emergentmind.com/topics/multi-hop-question-answering-qa-b6eff081-7dca-499a-807a-67dd4c3aa5b6) (aggregator; the underlying claim traces to [bg] Tang et al. 2021 "Do multi-hop QA systems answer sub-questions?")
- Decomposition evaluation: only 38.7% (GPT-4) and 33.4% (Llama3-70B) achieve "perfect reasoning" with all sub-questions correct. — [METU thesis, KG-augmented multi-hop QA](https://open.metu.edu.tr/bitstream/handle/11511/111317/knowledge-graph-augmented-multi-hop-question-answering-using-large-language-models.pdf)
- RASER limitation: "large LLMs sometimes answer from memory rather than retrieved passages." — [RASER HTML](https://arxiv.org/html/2606.02488)

### Inferences
- A router trained to predict "the large LLM gets it right" partly learns "the large LLM has memorised this Wikipedia fact." Under counterfactual contexts that reward disappears, so the router's cost/accuracy trade-off should shift. This is a cheap and novel evaluation angle.

### Gaps
- No paper was found that evaluates *routing or escalation policies* (rather than readers) under counterfactual or adversarial-distractor perturbations on HotpotQA/2Wiki/MuSiQue.

---

## Q5. Candidate novel, feasible coursework ideas (a: prior work, b: gap, c: single-GPU test plan, d: decisive ablation)

### Takeaway
The strongest ideas pair a cheap, model-agnostic **evidence sufficiency signal** (MiniLM cross-encoder) with **rigorous evaluation**: a matched query-only control, cross-dataset transfer, conformal guarantees under shift, and counterfactual robustness. These combinations are not covered by RASER, Adaptive-RAG, Sufficient Context or the conformal-cascade papers. Ideas 1–4 are the recommended core, 5–8 are good extensions, and 9–11 are riskier or lower-novelty.

### Cited Findings
- Every prior work used below is cited with a source in Q1–Q4. The idea cards repeat the key sources inline.

### Inferences

#### Idea 1: Pre-generation "sufficiency gate" escalation router with a matched query-only control (text multi-hop)
- **(a) Done:** query-only routers (Adaptive-RAG [arXiv 2403.14403](https://arxiv.org/pdf/2403.14403); TF-IDF/MiniLM query routers [arXiv 2604.03455](https://arxiv.org/abs/2604.03455)); post-draft escalation on shallow retrieval features with a GBM (RASER [arXiv 2606.02488](https://arxiv.org/html/2606.02488)); LLM-autorater sufficiency used for *abstention* (Sufficient Context [arXiv 2411.06037](https://arxiv.org/html/2411.06037)); white-box sufficiency probes ([arXiv 2502.01025](https://arxiv.org/abs/2502.01025)).
- **(b) Gap:** no text multi-hop study tests whether a small, black-box-compatible cross-encoder sufficiency score beats a matched query-only router. The only matched-control study is multimodal and negative ([arXiv 2609.12437](https://arxiv.org/abs/2609.12437)). RASER needs a cheap-model draft first, which is the "structural cost" in [arXiv 2605.06350](https://arxiv.org/abs/2605.06350).
- **(c) Test:** fine-tune `cross-encoder/ms-marco-MiniLM-L-6-v2` on HotpotQA distractor pairs. Two targets: (i) *gold sufficiency* = both gold SP paragraphs present in the top-k; (ii) *behavioural* = the small reader (e.g. Qwen2.5-1.5B/3B-Instruct or a fine-tuned DeBERTa reader) is correct. Route small → large (a larger open model or an API LLM). Report accuracy vs % escalated (cost curve) and AURC.
- **(d) Ablation:** same classifier head with inputs {query only} vs {query + retrieved passages} vs {+ RASER-style features}; gold-sufficiency labels vs behavioural labels; pre-generation gate vs post-draft gate at matched cost. **The contribution is shown only if query+evidence beats query-only on held-out data.**
- **Novelty:** moderate. It is a new combination with a strong, citable motivation. Frame it as "does evidence help routing in text multi-hop QA?", not "we invented a router."

#### Idea 2: Conformal risk-controlled escalation for open-ended multi-hop QA, with guarantee-violation analysis under dataset shift
- **(a) Done:** conformal RAG (TRAQ [ACL 2024](https://aclanthology.org/2024.naacl-long.210/); Conformal-RAG [SIGIR 2025](https://dl.acm.org/doi/10.1145/3726302.3730244)); conformal routers and cascades on MCQ (CP-Router [arXiv 2505.19970](https://arxiv.org/abs/2505.19970); Conformal Cascade [arXiv 2607.25018](https://arxiv.org/abs/2607.25018)); CRC abstention on HotpotQA with weak separability (Pause and Reflect [arXiv 2605.14098](https://arxiv.org/abs/2605.14098)).
- **(b) Gap:** Conformal Cascade explicitly leaves open-ended generation as future work. No one reports conformal escalation guarantees for a multi-hop small→large cascade, **group-conditional (Mondrian) by reasoning type / hop count**, or measures **violations when calibrated on HotpotQA and deployed on MuSiQue/2Wiki**.
- **(c) Test:** loss = 1 − F1 (bounded) or an EM error indicator. Apply CRC / Learn-then-Test to pick the gate threshold so that the error of non-escalated answers is ≤ α. Calibration set of 1–2k HotpotQA dev questions. Target α ∈ {0.1, 0.2, 0.3}. Evaluate realised risk on HotpotQA test, 2Wiki and MuSiQue. CPU-only once the scores exist.
- **(d) Ablation:** marginal vs Mondrian (bridge/comparison; 2/3/4-hop) calibration; nonconformity score = small-model confidence vs sufficiency score vs both (tests whether sufficiency fixes HotpotQA's low separability); i.i.d. vs shifted calibration, plus a weighted-conformal or small in-domain recalibration fix.
- **Novelty:** moderate–high as an empirical study. It is cheap and statistically clean.

#### Idea 3: Cross-dataset transfer of multi-hop routers (train HotpotQA → test 2Wiki / MuSiQue)
- **(a) Done:** RegimeRouter transfers 2Wiki → MuSiQue (+5.3 pp) but not → HotpotQA (+1.1 pp n.s.), and is flagged as "artifact-driven" ([arXiv 2604.09019](https://arxiv.org/abs/2604.09019)). RASER does within-dataset CV only ([RASER](https://arxiv.org/html/2606.02488)). A reasoning-step classifier transfers NQ → multi-hop with about −3 F1 ([arXiv 2604.26649](https://arxiv.org/html/2604.26649v1); abstract-level). Query-side TF-IDF routing is strongest in-domain ([arXiv 2604.03455](https://arxiv.org/abs/2604.03455)), which suggests reliance on surface cues.
- **(b) Gap:** no systematic transfer matrix for *escalation* routers comparing query-surface features against evidence (sufficiency) features.
- **(c) Test:** 3×3 train/test matrix over HotpotQA, 2Wiki and MuSiQue for each router variant from Idea 1. Report the cost-accuracy AUC and the escalation-rate drift.
- **(d) Ablation:** feature-group removal (lexical, embedding, sufficiency, retrieval-score) under shift. Hypothesis: query-only routers degrade more under shift than evidence-based ones.
- **Novelty:** moderate. It is a natural "evaluation contribution" that pairs with Ideas 1 and 2.

#### Idea 4: Hop-level (sub-question) routing driven by per-hop sufficiency / extraction confidence
- **(a) Done:** subtask allocation across model sizes via an RL-trained allocator on general reasoning benchmarks (R2-Reasoner [arXiv 2506.05901](https://arxiv.org/abs/2506.05901)); question-level routing (Adaptive-RAG, RASER); small-model per-hop query generation (EfficientRAG [arXiv 2408.04259](https://arxiv.org/abs/2408.04259)).
- **(b) Gap:** in multi-hop RAG, about half of per-hop failures are *extraction* failures that retrieval does not fix ([arXiv 2609.17043](https://arxiv.org/abs/2609.17043)). That suggests escalating *only the failing hop's extraction* to a larger model, which no found work does with an evidence-based per-hop gate.
- **(c) Test:** use MuSiQue gold decompositions (oracle) first, then predicted decompositions. The small reader answers each hop. A per-hop gate (MiniLM sufficiency on sub-question + passages, plus reader confidence) escalates only low-confidence hops. Compare against question-level escalation at equal total large-model tokens.
- **(d) Ablation:** question-level vs hop-level routing at matched cost; oracle vs predicted decomposition; a gate using sufficiency only vs confidence only.
- **Novelty:** moderate–high for multi-hop RAG specifically. The closest work is R2-Reasoner, which is not evidence-conditioned and not RAG-focused.

#### Idea 5: Two-headed gate that distinguishes "retrieve more" from "use a bigger reader" (typed failure routing)
- **(a) Done:** Skill-RAG routes among typed correction skills using hidden-state probes, so it needs white-box access ([arXiv 2604.15771](https://arxiv.org/html/2604.15771)). RASER-3 chooses between more retrieval (PRUNE/IRCoT) but never a bigger model ([RASER](https://arxiv.org/html/2606.02488)). The fact-grounding-gap paper separates retrieval failures from extraction failures ([arXiv 2609.17043](https://arxiv.org/abs/2609.17043)).
- **(b) Gap:** a black-box, cheap gate whose two heads predict (i) evidence insufficiency → re-retrieve, and (ii) sufficient-but-hard extraction → escalate the model. No such action space (retrieval action vs model-size action) was found for HotpotQA/MuSiQue.
- **(c) Test:** in HotpotQA's open (fullwiki) or a simulated setting, drop gold paragraphs to create "insufficient" cases. Action labels come from oracle runs of both actions.
- **(d) Ablation:** a single-head gate vs a two-head gate; per-action oracle upper bound; cost-matched comparison against always-escalate and always-re-retrieve.
- **Novelty:** moderate–high. It has more engineering than Idea 1.

#### Idea 6: Joint answer + supporting-fact + sufficiency training of a small reader, with counterfactual paragraph-removal contrast pairs
- **(a) Done:** joint answer+SP MTL (HotpotQA [arXiv 1809.09600](https://arxiv.org/abs/1809.09600); SAE/HGN [bg]); unanswerable contrast questions in MuSiQue-Full ([bg] [arXiv 2108.00573](https://arxiv.org/abs/2108.00573)). Sufficient Context lists "a fine-tuned entailment model" as future work ([arXiv 2411.06037](https://arxiv.org/html/2411.06037)). Knowing-When-to-Stop finds fine-tuned sufficiency classifiers underperform probes at 70B ([arXiv 2502.01025](https://arxiv.org/abs/2502.01025)).
- **(b) Gap:** no work found that trains a *small* reader (e.g. DeBERTa-v3 / ModernBERT / Qwen-0.5B) with an explicit sufficiency head built from minimally different pairs (the same question with one gold paragraph removed), and then checks whether that head is **better calibrated** and a **better router signal** than answer confidence.
- **(c) Test:** create an insufficient copy of each HotpotQA training example by deleting one gold paragraph. Train with loss = answer + SP + sufficiency (+ optional contrastive loss between the sufficient and insufficient pair). Evaluate EM/F1, joint EM, ECE of the sufficiency head, and router utility (plugging into Idea 1).
- **(d) Ablation:** heads {answer}, {answer+SP}, {answer+SP+suff}, and {+contrastive pair loss}. Measure calibration on MuSiQue-Full's unanswerable items as a transfer test.
- **Novelty:** moderate. The individual pieces exist; the counterfactual-pair sufficiency head used as a router signal is new as far as was found.

#### Idea 7: Distilling a continuous (graded) LLM sufficiency judgment into a tiny cross-encoder
- **(a) Done:** LLM autorater for binary sufficiency (Gemini 93%, FLAMe-24B 87.8%; [arXiv 2411.06037](https://arxiv.org/html/2411.06037)); rationale distillation ([arXiv 2305.02301](https://arxiv.org/pdf/2305.02301)); distillation of LLMs into cross-encoders for *reranking relevance* ([arXiv 2607.11933](https://arxiv.org/pdf/2607.11933), title-level only).
- **(b) Gap:** Sufficient Context names the lack of fine-grained (continuous) sufficiency scoring as a limitation. No work was found that distils a graded LLM sufficiency score (e.g. P(sufficient) from logprobs or multi-sample agreement) into a ~22M MiniLM gate for multi-hop routing.
- **(c) Test:** label 5–10k HotpotQA/MuSiQue (question, top-k) pairs with a local 7–8B instruct model or a cheap API model. Train MiniLM with a soft-label BCE/KL loss. Compare against gold-SP-derived hard labels.
- **(d) Ablation:** hard gold labels vs hard LLM labels vs soft LLM labels vs mixed; label budget curve (1k/5k/10k); downstream router AURC and cross-dataset transfer.
- **Novelty:** moderate–high and cheap. A good "training objective" contribution.

#### Idea 8: Counterfactual stress test of routers (does the router rely on the large model's memorisation?)
- **(a) Done:** counterfactual multi-hop benchmarks for *readers* (CofCA; CRiT-QA [arXiv 2607.10562](https://arxiv.org/abs/2607.10562)); disconnected-reasoning metrics (DiRe [arXiv 2005.00789](https://awesomepapers.io/llm-papers/papers/2005.00789)); RASER notes memory answering ([RASER](https://arxiv.org/html/2606.02488)).
- **(b) Gap:** no evaluation found of *routing policies* under counterfactual evidence. Routers trained on factual data may over-escalate "memorable" questions and under-escalate when the context contradicts memory.
- **(c) Test:** entity-swap the bridge entity and answer in the gold paragraphs of 500–1000 HotpotQA dev items (scripted, as in CofCA-style editing). Measure small-model accuracy, large-model accuracy, the gate's decisions, and a "memory leak rate" (large model outputs the original, pre-edit answer).
- **(d) Ablation:** query-only vs evidence-aware routers on factual vs counterfactual sets. Hypothesis: evidence-aware gates degrade less. Add adversarial distractor insertion (Jiang & Bansal-style [bg]) as a second perturbation.
- **Novelty:** high as an evaluation contribution, and very cheap.

#### Idea 9: Reasoning-type-specialised LoRA experts for a small reader (bridge / comparison; 2Wiki's 4 types)
- **(a) Done:** generic Mixture-of-LoRA-Experts (SMoRA [arXiv 2501.15103](https://arxiv.org/abs/2501.15103); DynMoLE [arXiv 2504.00661](https://arxiv.org/pdf/2504.00661)); type-specific *prompting* operators (BELLE [arXiv 2505.11811](https://arxiv.org/pdf/2505.11811)); query-type routers.
- **(b) Gap:** no work was found that trains per-reasoning-type LoRA experts for HotpotQA/2Wiki readers and compares them with a parameter-matched single LoRA. Searching "HotpotQA bridge comparison adapters experts" returned nothing specific.
- **(c) Test:** Qwen2.5-0.5B/1.5B with QLoRA on one GPU. Experts for bridge and comparison (HotpotQA labels), then compositional / inference / comparison / bridge-comparison (2Wiki labels). Oracle-type routing vs a learned type classifier vs soft MoE gating.
- **(d) Ablation:** parameter-matched single LoRA; oracle vs predicted type; transfer to MuSiQue, which has no type labels.
- **Novelty:** moderate, but the **risk is high**. Comparison questions are already easy, so gains may be small or zero. Report honestly either way.

#### Idea 10: Hop-count / reasoning-type prediction as an auxiliary task for routers
- **(a) Done:** Adaptive-RAG's 3-way complexity classes; BELLE's question typing; a step-count classifier in [arXiv 2604.26649](https://arxiv.org/html/2604.26649v1) (abstract-level).
- **(b) Gap:** it is untested whether *auxiliary* hop-count (MuSiQue 2/3/4) and type (HotpotQA/2Wiki) prediction improves escalation-router generalisation (Idea 3), compared with direct escalation labels only.
- **(c) Test:** multi-task MiniLM with heads {escalate, hop count, type} trained on the pooled datasets.
- **(d) Ablation:** with vs without auxiliary heads; in-domain vs OOD.
- **Novelty:** low–moderate. It works best as an ablation inside Idea 1 or 3.

#### Idea 11 (stretch): Low-budget RL with an explicit, costed "escalate" action in the closed distractor setting
- **(a) Done:** RL search agents (Search-R1, R1-Searcher, ZeroSearch); low-budget RL on 4× 3090 (David-GRPO [arXiv 2601.21699](https://arxiv.org/abs/2601.21699)); an RL router with a cost reward over LLM pools **already evaluated on HotpotQA** (Router-R1 [arXiv 2506.09033](https://arxiv.org/abs/2506.09033)); citation-faithfulness rewards (CRAFT [arXiv 2602.01348](https://arxiv.org/abs/2602.01348)).
- **(b) Gap (narrow):** Router-R1's policy is itself an LLM (3B+) that calls other LLMs. The open question is whether a ≤1.5B reader, trained with LoRA-GRPO/RLOO on a single consumer GPU, can learn *when to escalate* with a reward of EM − λ·cost + SP-F1 bonus. The evidence-selection reward (SP-F1 from gold HotpotQA labels) makes this distinct from Router-R1's outcome+cost reward.
- **(c) Test:** Qwen2.5-0.5B/1.5B, HotpotQA distractor (no retriever), and a "large model" simulated by cached answers from a big model (pre-computed offline, so no inference cost during RL).
- **(d) Ablation:** reward with vs without the SP-F1 term; λ sweep → learned escalation rate vs a Pareto frontier from the supervised gate (Idea 1). Compare against a static threshold router at equal cost.
- **Novelty:** low–moderate. Expect instability (GRPO least stable per [arXiv 2602.19526](https://arxiv.org/abs/2602.19526)). Keep this as a stretch goal.

### Gaps
- Novelty checks were done by web search, and for the newest 2026 preprints only at abstract or summary level. Before claiming novelty, the team should run a targeted Semantic Scholar / arXiv listing search for each idea's exact keyword combination. The field moves weekly: 2609.12437 and 2609.17043 appeared in September 2026.
- No 2025–2026 *survey* dedicated to multi-hop QA open problems was found. The open-problem synthesis relies on Emergent Mind aggregator pages and individual papers' limitation sections.

---

## Q6. Ideas that sound novel but are already done (avoid false novelty claims)

### Takeaway
Do not claim novelty for any of the items below. Cite them as baselines or related work instead.

### Cited Findings
- **"Route queries by predicted complexity (no / single / multi-step retrieval)"** — Adaptive-RAG, NAACL 2024. — [arXiv 2403.14403](https://arxiv.org/pdf/2403.14403)
- **"A lightweight classifier that escalates after one-shot RAG on HotpotQA/2Wiki/MuSiQue using retrieval scores"** — RASER, June 2026. — [arXiv 2606.02488](https://arxiv.org/html/2606.02488)
- **"Query-only lightweight routers (TF-IDF / MiniLM) for adaptive RAG"** — RAGRouter-Bench baseline study, Apr 2026. — [arXiv 2604.03455](https://arxiv.org/abs/2604.03455)
- **"Use context sufficiency to decide abstention"** — Sufficient Context, ICLR 2025. — [arXiv 2411.06037](https://arxiv.org/abs/2411.06037)
- **"Probe LLM internals for sufficiency or retrieval necessity"** — Knowing When to Stop (NeurIPS 2025), Probing-RAG (2024), Skill-RAG (2026). — [2502.01025](https://arxiv.org/abs/2502.01025); [2410.13339](https://arxiv.org/html/2410.13339v1); [2604.15771](https://arxiv.org/html/2604.15771)
- **"Conformal guarantees for RAG QA"** — TRAQ (NAACL 2024), C-RAG [bg], Conformal-RAG (SIGIR 2025), conformal context filtering (Nov 2025). — [TRAQ](https://aclanthology.org/2024.naacl-long.210/); [Conformal-RAG](https://dl.acm.org/doi/10.1145/3726302.3730244); [2511.17908](https://arxiv.org/abs/2511.17908)
- **"Conformal routing / cascades between small and large LLMs"** — CP-Router (AAAI 2026), Conformal Cascade (July 2026), RouteNLP, RACER, Conformal Arbitrage, LEC. — [2505.19970](https://arxiv.org/abs/2505.19970); [2607.25018](https://arxiv.org/abs/2607.25018); [2604.23577](https://arxiv.org/html/2604.23577v1); [2603.06616](https://arxiv.org/pdf/2603.06616); [2506.00911](https://arxiv.org/pdf/2506.00911); [2512.01556](https://arxiv.org/pdf/2512.01556)
- **"CRC-based abstention on HotpotQA"** — Pause and Reflect, May 2026. — [2605.14098](https://arxiv.org/abs/2605.14098)
- **"Decompose then route sub-tasks to different-size models"** — R2-Reasoner (WWW 2026). — [2506.05901](https://arxiv.org/abs/2506.05901)
- **"RL-trained router with a cost reward evaluated on HotpotQA"** — Router-R1 (NeurIPS 2025). — [2506.09033](https://arxiv.org/abs/2506.09033)
- **"GRPO search agents on HotpotQA"** — Search-R1, R1-Searcher, ZeroSearch, Tree-GRPO, Stratified GRPO, Search-R1++. — [2503.09516](https://arxiv.org/pdf/2503.09516); [2503.05592](https://arxiv.org/pdf/2503.05592); [2505.04588](https://arxiv.org/pdf/2505.04588); [2509.21240](https://arxiv.org/pdf/2509.21240); [2510.06214](https://arxiv.org/abs/2510.06214); [2602.19526](https://arxiv.org/abs/2602.19526)
- **"Low-budget RL for small multi-hop agents on consumer GPUs"** — David-GRPO (4× RTX 3090, ≤1.5B). — [2601.21699](https://arxiv.org/abs/2601.21699)
- **"RL rewards for faithful citations / supporting evidence"** — CRAFT (0.5B–7B). — [2602.01348](https://arxiv.org/abs/2602.01348)
- **"Small models doing iterative multi-hop retrieval without LLM calls"** — EfficientRAG (EMNLP 2024). — [2408.04259](https://arxiv.org/abs/2408.04259)
- **"Bandit-based adaptive retrieval"** — MBA-RAG. — [2412.01572](https://arxiv.org/pdf/2412.01572)
- **"Route between vanilla RAG and GraphRAG"** — R²Adapter (July 2026). — [2609.02894](https://arxiv.org/abs/2609.02894)
- **"Joint answer + supporting-fact multi-task training"** — HotpotQA baseline (2018), SAE/HGN [bg]. — [1809.09600](https://arxiv.org/abs/1809.09600)
- **"Rationale distillation into small models"** — Distilling step-by-step (2023). — [2305.02301](https://arxiv.org/pdf/2305.02301)
- **"Counterfactual multi-hop evaluation sets" / "disconnected reasoning"** — CofCA, CRiT-QA (LREC 2026), DiRe (2020), Counterfactual Multihop QA (2022). — [2607.10562](https://arxiv.org/abs/2607.10562); [2005.00789](https://awesomepapers.io/llm-papers/papers/2005.00789); [2210.07138](https://arxiv.org/abs/2210.07138)
- **"Retrieval vs extraction failure decomposition"** — Fact-Grounding Gap, Sept 2026. — [2609.17043](https://arxiv.org/abs/2609.17043)
- **"Query-type-regime router with zero-shot transfer 2Wiki→MuSiQue"** — RegimeRouter (Apr 2026). — [2604.09019](https://arxiv.org/abs/2604.09019)

### Inferences
- The honest novelty framing for a coursework report: "We test whether a cheap, model-agnostic evidence-sufficiency signal improves multi-hop escalation routing *beyond a matched query-only control*, whether its conformal risk guarantees hold under cross-dataset shift, and whether it is robust to counterfactual evidence." Each clause maps to a gap left by RASER, 2609.12437, Conformal Cascade, and CRiT-QA respectively.

### Gaps
- It is possible that Adaptive-RAG follow-ups or RASER's related work already include a cross-encoder-based escalation gate that was not surfaced by these searches. Only RASER's HTML summary was read, not its full bibliography.
