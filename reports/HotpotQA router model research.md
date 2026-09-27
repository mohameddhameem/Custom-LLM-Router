# Gate HotpotQA escalation on evidence, not questions

The best route for this team is a **cost-aware cascade over three or four HotpotQA "experts"**. The existing nano-jev MiniLM cross-encoder does double duty as the paragraph selector and as the router. The router is trained on per-expert correctness labels and judged by a cost–F1 curve against oracle and "zero-router" baselines. Building a new reader to chase the leaderboard is the wrong target. The distractor leaderboard has barely moved since 2021: Joint F1 went from 75.7 to **77.5 (Beam Retrieval, Aug 2023)**, and gold-paragraph selection is already about **97.5% solved**. The headroom is in routing. On HotpotQA, Adaptive-RAG's *oracle* router reaches **F1 64.0 at 1.59 retrieval steps**, while its learned classifier, which is only **54.5% accurate**, reaches 53.8 F1 at 3.55 steps. Routing on HotpotQA is not new in itself. Adaptive-RAG, MBA-RAG, Probing-RAG, Router-R1 and RASER (June 2026) all do it, and a plain token-entropy threshold already matches Adaptive-RAG's accuracy with half the LM calls. Four questions are still open and can be tested on one consumer GPU in a few weeks:

- Does a cross-encoder **evidence-sufficiency signal** beat a *matched* query-only router in text multi-hop QA? The only matched-control study found no gain, but it was multimodal.
- Does conformal risk control of the escalation decision hold up when calibration is split by reasoning type and when the router moves from HotpotQA to MuSiQue?
- Does a sufficiency head trained on counterfactual paragraph-removal pairs make a better routing signal?
- Do routers stay reliable under counterfactual evidence?

The distractor setting also offers a gap few people use. LLM-era papers report answer-only EM on subsets, so a router pipeline scored with the **official Joint EM/F1 on all 7,405 dev questions** would be directly comparable to the classic readers.

## Seven years of readers pushed distractor Joint F1 from 40 to 77.5

HotpotQA's distractor setting pairs each question with its 2 gold Wikipedia paragraphs and **8 distractors retrieved by bigram TF-IDF** on the question ([Yang et al. 2018](https://aclanthology.org/D18-1259.pdf)). The official scorer is strict about how answer and evidence combine. Joint precision is answer precision × supporting-fact precision, and Joint EM is answer EM × supporting-fact EM, so a correct answer with sloppy evidence scores zero Joint EM ([hotpot_evaluate_v1.py](https://github.com/hotpotqa/hotpot/blob/master/hotpot_evaluate_v1.py)). One 2021 entry shows how much this matters: it reached answer F1 80.36, but its supporting-fact EM was 1.10, which held Joint F1 to 52.50 ([leaderboard](https://hotpotqa.github.io/)). The scorer also gives **zero F1 whenever a yes/no answer mismatches**, with no partial credit ([hotpot_evaluate_v1.py](https://raw.githubusercontent.com/hotpotqa/hotpot/master/hotpot_evaluate_v1.py)). About 6% of sampled questions are yes/no ([Yang et al. 2018](https://aclanthology.org/D18-1259.pdf)).

| System (date) | Ans F1 | Sup F1 | Joint EM | Joint F1 |
|---|---|---|---|---|
| Baseline (Oct 2018) | 59.02 | 64.49 | 10.83 | 40.16 |
| DFGN (ACL 2019) | 69.69 | 81.62 | 33.62 | 59.82 |
| SAE-large (AAAI 2020) | 79.62 | 86.86 | 45.36 | 71.45 |
| C2F Reader / "Is Graph Necessary?" | 81.24 | 87.63 | 44.67 | 72.73 |
| Longformer-large | 81.25 | 88.34 | 45.91 | 73.16 |
| HGN-large (Dec 2019) | 82.19 | 88.47 | 47.11 | 74.21 |
| FE2H on ALBERT (Jan 2022) | 84.44 | 89.14 | 50.04 | 76.54 |
| **Beam Retrieval (Aug 2023, #1)** | **85.04** | **90.09** | **50.53** | **77.54** |

*Distractor test set, official leaderboard ([HotpotQA](https://hotpotqa.github.io/)).*

Two ingredients explain nearly all of the gain since 2019. The first is a larger pretrained encoder (BERT, then RoBERTa-large, then ALBERT-xxlarge, ELECTRA-large and DeBERTa-xxlarge). The second is a near-perfect paragraph selector. HGN's own dev ablation shows the encoder effect: the same architecture scores Joint F1 66.90 with BERT-base and **75.79 with ALBERT-xxlarge** ([HGN](https://arxiv.org/abs/1911.03631)). The graph modules that dominated 2019–2021 turned out to add nothing once the encoder is fine-tuned. Shao et al. found Joint F1 of 73.93 with a graph-fusion block and 73.78 without it. The graph helped only when the encoder was frozen, and graph attention over a fully connected entity graph reduces to self-attention ([Shao et al. 2020](https://arxiv.org/abs/2004.03096)). Beam Retrieval, the current leader, runs a beam search over *sequences* of passages, with one encoder scoring each hop conditioned on the passages already chosen. It reaches **97.52 gold-pair EM on dev**, against 96.32 for FE2H and 91.98 for SAE. The DeBERTa-large variant was trained on a single RTX 4090, but the leaderboard entry used DeBERTa-xxlarge on an A100 ([Beam Retrieval](https://arxiv.org/abs/2308.08973)). For the team, this means paragraph selection is effectively solved at the top end. The errors left are in answer spans and sentence-level evidence. Supporting-fact EM (~66) and Joint EM (~50) are the least saturated metrics.

The fullwiki (open-domain) track tells the same story with retrieval added in front. Systems moved from TF-IDF plus hyperlinks (GoldEn Retriever, Joint F1 39.13) to learned path retrievers (Graph Recurrent Retriever, 61.18). They then moved to multi-hop dense retrieval (MDR, 66.55) and iterative rerankers, ending with **AISO at 72.00 Joint F1 and Chain-of-Skills at the best Joint EM of 45.65** ([leaderboard](https://hotpotqa.github.io/)). MDR lifted dev recall of the full gold pair at 20 passages from 36.8 (TF-IDF) to 80.2 ([MDR](https://arxiv.org/abs/2009.12756)). Baleen then reached P-R@20 93.3 and argued that this "effectively reduces the open-retrieval task to a narrow distractor-setting one" ([Baleen](https://arxiv.org/abs/2101.00436)). Both leaderboards look passively maintained. The last distractor entry is from August 2023, the last fullwiki entry from June 2024, and it is unverified whether CodaLab still scores test submissions ([HotpotQA site](https://hotpotqa.github.io/)). A student team should plan on reporting dev-set numbers from the official script and saying so.

Every HotpotQA project needs one caveat up front: many questions are not truly multi-hop. A single-paragraph BERT that scores each paragraph independently reaches **67.08 F1 in the distractor setting** ([Min et al. 2019](https://arxiv.org/abs/1906.02900)). Sentence-factored models, which cannot chain facts by construction, still find the answer sentence 45–57% of the time ([Chen & Durrett 2019](https://arxiv.org/abs/1904.12106)). Adversarial distractor documents cut a 1-hop baseline from 42.32 to 26.67 EM ([Jiang & Bansal 2019](https://arxiv.org/abs/1906.07132)). These findings led to 2WikiMultiHopQA and to MuSiQue, which composes connected single-hop questions and on which a single-hop model loses 30 F1 ([MuSiQue](https://arxiv.org/abs/2108.00573)). As recently as 2024, the HippoRAG authors called HotpotQA "a much weaker test for multi-hop reasoning due to many spurious signals" ([HippoRAG](https://arxiv.org/abs/2405.14831)). Any claim that a router "improves multi-hop reasoning" therefore needs a MuSiQue or 2Wiki check behind it.

## LLM-era numbers scatter across incompatible protocols

LLM papers almost never use the official protocol. They report answer-only EM/F1, sometimes cover-EM or LLM-judge accuracy. They evaluate on 500- or 1,000-question dev subsets, and each brings its own corpus and retriever, so **the numbers below cannot be compared with the leaderboard or with each other** unless they share a protocol.

Prompted pipelines built on GPT-3-class models reached roughly 35–51 EM. ReAct with PaLM-540B scored 35.1 EM using CoT-SC as a fallback ([ReAct](https://arxiv.org/abs/2210.03629)). DSP scored 51.4/62.9 EM/F1 on a 1,000-question subset ([DSP](https://arxiv.org/abs/2212.14024)), IRCoT 49.3/60.7 on 500 questions ([IRCoT](https://arxiv.org/abs/2212.10509)), and Iter-RetGen 45.8/61.1 after four iterations ([Iter-RetGen](https://arxiv.org/abs/2305.15294)). EfficientRAG replaced per-hop LLM calls with a 304M DeBERTa labeler and filter and reached 50.59/57.93 with a Llama-3-8B reader ([EfficientRAG](https://arxiv.org/abs/2408.04259)). Graph-memory methods report much higher F1, but on a closed corpus of about 9k passages rather than fullwiki. HippoRAG 2 reaches **75.5 F1 with a 70B reader** ([HippoRAG 2](https://arxiv.org/abs/2502.14802)).

The 2025 RL-trained search agents settled on a shared protocol: E5 retrieval over the 2018 Wikipedia dump, top-3 passages, and EM on the full dev set. Under it, Search-R1 reaches **0.433 EM at 7B and 0.324 at 3B-instruct**, but it trains on HotpotQA itself, and the reported training setup used 8× H100 ([Search-R1](https://arxiv.org/abs/2503.09516)). ReSearch, trained only on MuSiQue, reaches 43.52 EM on HotpotQA out-of-domain ([ReSearch](https://arxiv.org/abs/2503.19470)). StepSearch's per-step information-gain rewards score 0.386 at 7B, which does *not* beat in-domain Search-R1 ([StepSearch](https://arxiv.org/abs/2505.15107)). None of these systems predict supporting facts.

The most useful LLM numbers for a distractor-setting project come from Bactrainus. Given **gold supporting facts only**, Llama 3.1 8B zero-shot scores **74.52 F1**. Given all candidate paragraphs, it drops to **57.20 F1**, a 17-point loss; the 70B model loses 21 points. Chain-of-thought *hurt* the 8B model in every shot setting, by 0.9–3.9 F1. LoRA fine-tuning on gold context lifted the 8B model to **86.46 F1** ([Bactrainus](https://arxiv.org/html/2501.06286)). Separately, "plausible distractors" built from alternate reasoning chains cut GPT-3.5 from 77.2 to 52.7 F1 ([Bhuiya et al. 2024](https://arxiv.org/pdf/2409.05197)). For this project, paragraph selection is the biggest cheap lever for a small LLM, and "how much context to show" is a routing decision in its own right.

### Routers on HotpotQA show large oracle gaps, weak classifiers and strong cheap baselines

| Method | Generator | HotpotQA result | Cost | Source |
|---|---|---|---|---|
| Always multi-step (IRCoT) | FLAN-T5-XL | EM 44.60 / F1 56.54 | 5.53 steps | [Adaptive-RAG](https://arxiv.org/abs/2403.14403) |
| Adaptive-RAG (T5-Large classifier) | FLAN-T5-XL | EM 42.00 / F1 53.82 | 3.55 steps | [Adaptive-RAG](https://arxiv.org/abs/2403.14403) |
| Adaptive-RAG, oracle router | FLAN-T5-XL | **EM 51.20 / F1 64.00** | **1.59 steps** | [Adaptive-RAG](https://arxiv.org/abs/2403.14403) |
| MBA-RAG (bandit) | FLAN-T5-XL | EM 40.60 / F1 52.44 | 2.25 steps | [MBA-RAG](https://arxiv.org/html/2412.01572v2) |
| DRAGIN (token-level trigger) | LLaMA2-13B | EM 0.314 / F1 0.424 | per-token retrieval | [DRAGIN](https://arxiv.org/html/2403.10081) |
| Probing-RAG (hidden-state prober) | Gemma-2B | Acc 39.32 (Adaptive-RAG 23.55) | 57% skip retrieval | [Probing-RAG](https://arxiv.org/html/2410.13339v1) |
| Mean-token-entropy threshold | LLaMA-3.1-8B | In-Acc 0.414 (Adaptive-RAG 0.414) | **2.0 vs 4.6 LM calls** | [Moskvoretskii et al.](https://arxiv.org/html/2501.12835) |
| Router-R1 (RL, 3B router over 6 LLMs) | Qwen2.5-3B | EM 0.352 (GraphRouter 0.244) | cost reward | [Router-R1](https://arxiv.org/html/2506.09033) |
| RASER-3 (GBM on 6 features) | 6 LLMs | 87–97% of IRCoT F1 | 47–62% of tokens | [RASER](https://arxiv.org/html/2606.02488) |

Adaptive-RAG is the direct prior art for query-complexity routing on HotpotQA. It labels each training query with the *simplest* strategy that answered correctly (no retrieval, single-step or multi-step). Queries that nothing answers fall back to a label based on which dataset they came from. It then trains a T5-Large classifier on only **400 queries per dataset** ([Adaptive-RAG](https://arxiv.org/abs/2403.14403); [code](https://github.com/starsuzi/Adaptive-RAG)). Its classifier accuracy is just **54.52%**, yet it still cuts steps by about 36% for a 2.7-point F1 loss. The oracle beats always-multi-step by 7.5 F1 at under a third of the cost. That oracle gap is the headroom a better router could capture. Cheap baselines are hard to beat, though. Across 27 uncertainty estimators on HotpotQA, mean token entropy matched Adaptive-RAG's accuracy with **2.0 rather than 4.6 LM calls**, and the best "self-knowledge" accuracy was only about 0.73 ([Moskvoretskii et al.](https://arxiv.org/html/2501.12835)).

The general-purpose model routers (FrugalGPT, RouteLLM, Hybrid LLM, AutoMix, RouterBench) supply the label and metric recipes, but **none of them report HotpotQA results** ([FrugalGPT](https://arxiv.org/pdf/2305.05176); [RouteLLM](https://arxiv.org/html/2406.18665v4); [Hybrid LLM](https://arxiv.org/html/2404.14618v1); [AutoMix](https://arxiv.org/html/2310.12963v5)). Two findings from that line of work should temper expectations. RouterBench's predictive kNN and MLP routers sometimes failed to beat the "zero router," which is just the best random mix of single models ([RouterBench](https://arxiv.org/pdf/2403.12031)). A May 2026 study of 21 methods across 5 benchmarks found that most learned routers "converge to a narrow performance range that remains far below the oracle router" and mostly learn average model-quality trends rather than query-specific signals ([Routing Plateau](https://arxiv.org/abs/2606.07587)).

## A blueprint that reuses nano-jev and rizzo-flow

### A four-tier expert pool where context size is itself a routing action

The distractor setting reverses the usual adaptive-RAG problem. The 10 paragraphs are always given, so retrieval costs nothing. The real costs are **reader size, context length and output length**. A sensible pool, from cheapest to most expensive, looks like this:

| Tier | Expert | What it costs | What it is good at |
|---|---|---|---|
| E1 | DeBERTa-v3-base extractive reader with a 3-way yes/no/span head | one encoder pass | span answers |
| E2 | Qwen2.5-1.5B/3B-Instruct over the **top-2 or top-4 paragraphs chosen by nano-jev** | a short LLM prompt | cheap reading once distractors are removed |
| E3 | Qwen2.5-7B or Llama-3.1-8B (4-bit) over all 10 paragraphs, or a rizzo-flow LoRA-tuned reader | the full prompt | harder questions |
| E4 (optional) | API model | real money | an escalation ceiling |

This pool is exactly where the 17–21 F1 drop from gold to all-candidate context appears ([Bactrainus](https://arxiv.org/html/2501.06286)). The public HotpotQA DeBERTa checkpoint reports 60.53/74.21 EM/F1, but **it drops yes/no questions** and was scored on a non-standard 4,687-question split ([HF model card](https://huggingface.co/MhoOmm/HotPotQA_DEBERT)). The team therefore has to add the yes/no head itself. No peer-reviewed zero-shot numbers were found for Qwen 0.5–3B on the distractor setting, so a 500–1,000-question pilot of every expert should be the first milestone. Since CoT hurt Llama 3.1 8B on gold context, measure it on the pilot before paying for it as a separate expert.

### Label with the cheapest correct expert, but keep the whole correctness vector

Run every expert on every router-training question and score it with the official EM/F1. Store `em`, `f1`, token cost, latency and answer-token log-probabilities for each (question, expert) pair. From that table, derive three targets, following RouteLLM, Hybrid LLM and Adaptive-RAG:

- **`y_cheapest`**: the lowest-cost expert with F1 ≥ τ (τ ≈ 0.8), used as a multi-class target;
- **`y_multi`**: a per-expert success vector, used for multi-label BCE;
- **`y_soft`**: the success rate over 3–5 sampled answers per LLM expert. Hybrid LLM's probabilistic labels (10 samples) and relaxed-margin labels produced better routers when the quality gap was large ([Hybrid LLM](https://arxiv.org/html/2404.14618)).

An F1 threshold beats strict EM here because LLMs paraphrase and EM on HotpotQA is sparse. Report sensitivity to τ, and hand-audit about 200 disagreements. No published estimate was found of how often small-LLM answers are semantically right but EM-wrong.

Three traps can invalidate the labels:

- **The `level` field.** Train-easy is "mostly single-hop", but **dev and test are 100% "hard"**, and "hard" was defined by a 2018 baseline's confidence ([Yang et al. 2018](https://aclanthology.org/D18-1259.pdf)). A router trained on `level` learns a feature that is constant at evaluation. Use `type` (bridge/comparison) instead; it is present in every split.
- **Gold evidence at test time.** Never feed `supporting_facts` or the gold answer's yes/no status into the router at test time.
- **Memorisation.** If E1 or E2 are fine-tuned on HotpotQA train, their correctness on those same questions is inflated. Build router labels from held-out train questions or with 2-fold cross-fitting, and keep the 7,405 dev questions untouched for the final table.

### Map router objectives onto the two reference codebases

nano-jev already solves the right shape of problem: grouped softmax cross-entropy over (question, option) pairs, followed by temperature scaling. Treating each expert as an "option" makes it an Adaptive-RAG-style multi-class router. Swapping the grouped softmax for one sigmoid per option turns it into the **multi-label BCE success predictor** that Hybrid LLM trains on a 300M DeBERTa ([Hybrid LLM](https://arxiv.org/html/2404.14618)). The BCE version is the recommended default. Any cost vector can be applied after training by choosing `argmax p̂_success(e) − λ·cost(e)`, and utility-based routing needs calibrated probabilities, which nano-jev's temperature scaling supplies.

rizzo-flow's soft cross-entropy over a restricted set of answer-token logits is the same pattern as RouteLLM's **causal-LLM router**, which predicts label tokens with Llama-3-8B ([RouteLLM](https://arxiv.org/html/2406.18665v4)). Each expert gets a label token, and the normalised `y_soft` vector is the target. This gives a second router family to compare against the 33M encoder at the same labels.

The router's inputs should grow in three matched steps:

1. **Question only**: text plus lexical cues such as auxiliary-verb openings for yes/no and "which … or …" for comparisons.
2. **Plus cheap evidence signals from nano-jev's paragraph scores**: the sum of the top-2 scores, the margin between the second and third score, and the softmax entropy over the 10 paragraphs.
3. **Plus post-hoc cascade features from the cheap expert**: for a generative expert, use quantiles of per-token uncertainty rather than sequence probability. Quantiles avoid length bias and beat aggregate probability for deferral ([Gupta et al., ICLR 2024](https://arxiv.org/abs/2404.10136)).

Choose λ, or the escalation threshold, on a separate calibration split of about 2k training questions. Then report two conventional operating points: Hybrid LLM's "≤1% quality drop" point and RouteLLM's CPT(50%/80%), the share of strong-model calls needed to recover that fraction of the quality gap ([RouteLLM](https://arxiv.org/html/2406.18665v4)).

### Evaluate on cost–quality curves, not classifier accuracy

Adaptive-RAG shows why classifier accuracy is the wrong headline: 54.5% accuracy still bought a useful trade-off. Plot answer F1 and EM against average cost (tokens as the primary unit, wall-clock time on the team's GPU as the secondary one). Draw each single expert as a point and add:

- the **zero router**, i.e. the convex hull of the single experts;
- a random router at matched call rates;
- the **oracle** (cheapest correct expert per question);
- an Adaptive-RAG-style question-only classifier;
- a mean-token-entropy threshold, the baseline that matched Adaptive-RAG.

Summarise each curve with RouterBench's AIQ, the area under the non-decreasing convex hull ([RouterBench](https://arxiv.org/pdf/2403.12031)). Break results down by bridge vs comparison and by yes/no vs span, and show reliability diagrams before and after temperature scaling.

One addition is cheap and makes the project visible against the literature. A nano-jev-style cross-encoder can be trained to score *sentences* as well as paragraphs, so the pipeline can emit supporting facts and be scored for **official Joint EM/F1 on the full dev set**. Almost no LLM-era system reports this, so the team's numbers would sit directly next to the classic readers in the first table ([hotpot_evaluate_v1.py](https://github.com/hotpotqa/hotpot/blob/master/hotpot_evaluate_v1.py)).

### Labelling fits in days, not weeks, on one GPU

On an RTX 4090 in vLLM offline mode, a vendor benchmark measured **Qwen2.5-3B at 10.3 requests/s and 7–8B distills at 4–6 requests/s**, using a decode-heavy 100-in/600-out workload ([DatabaseMart](https://www.databasemart.com/blog/vllm-gpu-benchmark-rtx4090)). HotpotQA direct answering is prefill-heavy, with about 1–1.5k input tokens (the team should measure this) and fewer than 20 output tokens. At a conservative 5 requests/s, a 3B expert covers the dev set in about 25 minutes and a 20k-question label set in about an hour; this is an inference to confirm with a pilot. A T4 or P100 will be several times slower, so on those GPUs stay at 3B or below, or use 4-bit 7B models. A realistic month runs as follows:

| Week | Work |
|---|---|
| 1 | Cache all expert outputs on dev and a 20k train subset, with soft samples for the LLM tiers |
| 2 | Build labels; train the question-only router and the Adaptive-RAG-style baseline |
| 3 | Add evidence features, the cascade variant, calibration and the λ sweep |
| 4 | Curves, AIQ, breakdowns and the novelty experiment below |

Routers themselves are cheap to train: RouteLLM's BERT router trained for about 2,000 steps, and Adaptive-RAG used only 400 queries per dataset ([RouteLLM](https://arxiv.org/html/2406.18665v4); [Adaptive-RAG](https://arxiv.org/abs/2403.14403)).

## Evidence-sufficiency gating is the defensible novelty wedge

Several ideas sound new but are already published. Complexity routing (Adaptive-RAG), bandit routing (MBA-RAG), hidden-state retrieval probes (Probing-RAG, Skill-RAG), sufficiency-based *abstention* (Sufficient Context, ICLR 2025), conformal RAG and conformal cascades (TRAQ, CP-Router, Conformal Cascade), RL routers with a cost reward evaluated on HotpotQA (Router-R1), and low-budget GRPO search agents (David-GRPO, on 4× RTX 3090) all exist. The closest threat to a "post-retrieval escalation gate" is **RASER (June 2026)**. It runs a gradient-boosting classifier on six shallow features (answer confidence, answer length, bridge cues, top-1 retrieval score, top-1-to-top-5 score gap and question type) after a one-shot RAG draft. It then decides whether to stop or to escalate to bridge retrieval or IRCoT, on HotpotQA, 2Wiki and MuSiQue ([RASER](https://arxiv.org/html/2606.02488)). RASER has **no learned cross-encoder sufficiency score, no conformal guarantee and no cross-dataset transfer**. It uses only 200–500 questions per LLM/dataset, and it notes that large LLMs sometimes answer from memory rather than from the retrieved passages.

The most important constraint on any claim comes from a September 2026 multimodal study. It found that retrieval-state features **did not consistently improve** routing over a matched query-only router, and it proposed a rule: such features "should not be credited with routing value unless they improve over a matched query-only control" ([Beyond the Query](https://arxiv.org/abs/2609.12437)). A related finding points the same way. On HotpotQA, abstention improves selective accuracy far less than on GSM8K, because model confidence separates correct from incorrect answers poorly ([Pause and Reflect](https://arxiv.org/abs/2605.14098)). If an *external* evidence signal fixes that separability problem, it is a real finding. If it does not, that is a real finding too.

| Idea | Closest prior work | What remains open | Honest novelty / risk |
|---|---|---|---|
| **1. Pre-generation sufficiency gate vs matched query-only router** | Adaptive-RAG; RASER; TF-IDF/MiniLM query routers ([2604.03455](https://arxiv.org/abs/2604.03455)); Sufficient Context ([2411.06037](https://arxiv.org/html/2411.06037)) | No text multi-hop study tests a cross-encoder sufficiency score against a matched query-only control. The only such study is multimodal and negative | Moderate. It is a new combination, not a new router. Frame it as "does evidence help routing?" |
| **2. Conformal escalation for open-ended multi-hop QA** | TRAQ ([NAACL 2024](https://aclanthology.org/2024.naacl-long.210/)); Conformal Cascade ([2607.25018](https://arxiv.org/abs/2607.25018)); CP-Router ([2505.19970](https://arxiv.org/abs/2505.19970)) | Conformal Cascade leaves open-ended generation to future work. Nobody reports bridge/comparison-conditional coverage or violations from HotpotQA to MuSiQue | Moderate–high as an empirical study. It runs on CPU once scores exist |
| **3. Cross-dataset router transfer (3×3 matrix)** | RegimeRouter: 2Wiki→MuSiQue +5.3 pp, →HotpotQA +1.1 pp n.s. ([2604.09019](https://arxiv.org/abs/2604.09019)) | No transfer matrix compares query-surface and evidence features for *escalation* routers | Moderate. Best paired with 1 and 2 |
| **4. Hop-level escalation** | R2-Reasoner sub-task allocation ([2506.05901](https://arxiv.org/abs/2506.05901)); EfficientRAG | About half of per-hop failures are extraction failures that retrieval never fixes ([2609.17043](https://arxiv.org/abs/2609.17043)), so escalate only the failing hop | Moderate–high. Needs MuSiQue decompositions |
| **5. Counterfactual-pair sufficiency head** | Joint answer+SP training (HotpotQA, SAE, HGN); MuSiQue-Full contrast items | Train the head on the same question with one gold paragraph deleted, then test whether it is better calibrated and a better router signal than answer confidence | Moderate. The pieces exist; the combination is unreported |
| **6. Distil graded LLM sufficiency into MiniLM** | Sufficient Context autoraters (Gemini 93%, FLAMe 87.8%) | Sufficient Context itself names continuous sufficiency scoring and a fine-tuned entailment model as future work | Moderate–high and cheap |
| **7. Counterfactual stress test of routers** | CRiT-QA ([2607.10562](https://arxiv.org/abs/2607.10562)); CofCA; RASER's memory caveat | No one evaluates *routing policies*, as opposed to readers, under entity-swapped evidence | High as evaluation. Very cheap |
| 8. Bridge/comparison LoRA experts | Mixture-of-LoRA work; type-specific prompting | No parameter-matched test on HotpotQA | Moderate, **high risk**: comparison questions are already easy |
| 9. LoRA-GRPO ≤1.5B with a costed "escalate" action | Router-R1; David-GRPO ([2601.21699](https://arxiv.org/abs/2601.21699)) | An SP-F1 evidence reward distinguishes it from Router-R1's outcome+cost reward | Low–moderate. GRPO was the least stable variant in Search-R1++ ([2602.19526](https://arxiv.org/abs/2602.19526)) |

The recommended core pairs **Idea 1 with Ideas 5 and 7**, with Idea 2 as a CPU-only add-on. The decisive ablation holds the classifier head, labels and data fixed and varies only the input: question alone, question plus nano-jev evidence features, and question plus evidence plus RASER-style features. A pre-generation gate is also compared with a post-draft gate at matched cost. The pre-generation design has a theoretical argument behind it. A May 2026 analysis attributes most cascade limitations to "structural cost", the price of running the cheap model before any decision ([Is Escalation Worth It?](https://arxiv.org/abs/2605.06350)). A 33M cross-encoder deciding *before* any LLM runs avoids that cost.

Two caveats specific to the distractor setting need to be designed in. First, a "gold sufficiency" label (are both gold paragraphs in the top-k?) nearly saturates when the selector is strong. Selectors of the strongest leaderboard systems reach about 96–97.5% pair EM ([Beam Retrieval](https://arxiv.org/abs/2308.08973)). The team should measure nano-jev's top-2 pair EM first. If it is high, gold sufficiency will be almost constant and carry little routing signal. In that case either use the *behavioural* target (will E2 be correct?) or create insufficient cases by deleting gold paragraphs, which is Idea 5. Sufficient Context found about 56% of HotpotQA instances insufficient, but that was under open retrieval, not distractor ([Sufficient Context](https://arxiv.org/html/2411.06037)). Second, gold sufficiency and small-model correctness are different targets. "Knowing When to Stop" found that fine-tuned sufficiency classifiers underperform attention-head probes (F1 79.5 vs 91.1 at 70B), probably because sufficiency signals differ across models ([Xie et al., NeurIPS 2025](https://arxiv.org/abs/2502.01025)). The gap between "the evidence is present" and "E2 answers correctly" is worth measuring and reporting in its own right. It is also consistent with the extraction-failure finding.

RL fits on one GPU only as a stretch goal. It needs a model of 1.5B or smaller with LoRA, the closed distractor setting (so no retriever server), and the large model's answers cached offline so escalation costs nothing during training. Search-R1++ found that EM rewards beat F1 rewards and that REINFORCE beat PPO, so keep RLOO or REINFORCE as the fallback ([Search-R1++](https://arxiv.org/abs/2602.19526)). Novelty for every idea was checked by web search, and only at abstract level for the newest 2026 preprints. Two directly relevant papers appeared in September 2026 alone, so run a targeted Semantic Scholar check on each exact keyword combination before writing "first."

## Conclusion

The field's own record says the router is not the contribution; the **controlled comparison** is. Learned routers across the literature sit far below their oracles. Adaptive-RAG, the canonical HotpotQA complexity router, is only 54.5% accurate on a three-way task, and a simple entropy threshold matches its accuracy at half the LM calls. The one matched-control test of retrieval-state features came back negative. A team that runs the same MiniLM head on question-only and question-plus-evidence inputs, under identical labels and cost curves, will produce a result worth reporting whichever way it falls. The distractor setting helps: with retrieval free, the causal effect of *evidence quality* on escalation is unusually clean to isolate.

The second takeaway is about where to spend compute. Selection may matter more than scale at the cheap end: Bactrainus shows distractors cost an 8B–70B model 17–21 F1 relative to gold-only context, so a small LLM reading nano-jev's top-2 paragraphs could recover much of that gap. That is untested and should be the first pilot measurement. If it holds, the expensive tiers mostly serve the questions whose evidence is ambiguous, which is exactly the signal a sufficiency gate measures. Scoring the whole pipeline with the official Joint EM/F1 on the full dev set, a standard the LLM era abandoned, lets a four-week student project sit credibly next to the leaderboard systems.
