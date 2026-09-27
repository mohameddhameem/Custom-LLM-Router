# Prior Methods on HotpotQA (Distractor, Fullwiki, and LLM-era), Critiques, and Evaluation

Research date: 2026-09-27. Numbers were taken from the official leaderboard HTML (fetched 2026-09-27) and from the primary PDFs (arXiv), read as text. Unless stated otherwise, "test" means the hidden official test set scored by the HotpotQA team. The dev set has 7,405 questions ([ReSearch](https://arxiv.org/abs/2503.19470) lists 7,405 for HotpotQA dev). LLM-era papers usually report on **dev subsets** (500 or 1,000 questions) with **different corpora and retrievers**, so their numbers are **not comparable** to the leaderboard or to each other unless the notes say otherwise.

---

## Q1. Distractor-setting readers: what has been tried and what did it score?

### Takeaway
On the distractor test set, results went from the 2018 baseline (Joint F1 40.16) to Beam Retrieval (Aug 2023, Joint EM/F1 50.53/77.54), which is still #1 on the official leaderboard. Almost every gain since 2019 came from (a) a stronger, larger pre-trained encoder (BERT → RoBERTa-large → ALBERT-xxlarge / ELECTRA-large / DeBERTa-xxlarge) and (b) a near-perfect paragraph selector: gold-pair selection EM is about 96–97.5% on dev. Graph modules added little once encoders were fine-tuned. Nothing new has reached the top of the distractor leaderboard since Aug 2023.

### Cited Findings
**Official distractor leaderboard, TEST set (Ans EM/F1 | Sup EM/F1 | Joint EM/F1)**, all from [HotpotQA leaderboard](https://hotpotqa.github.io/):
- #1 Beam Retrieval (BUPT & Tencent, Aug 7 2023): 72.69/85.04 | 66.25/90.09 | 50.53/77.54 — [leaderboard](https://hotpotqa.github.io/)
- #2 PipNet (Tencent Cloud Xiaowei, Jul 2022, no paper listed): 72.26/84.86 | 63.71/89.41 | 48.76/76.95 — [leaderboard](https://hotpotqa.github.io/)
- #3 Smoothing R3 (Fudan & Huawei, Jun 2022, "Rethinking Label Smoothing on Multi-hop QA"): 72.07/84.34 | 65.44/89.55 | 49.73/76.69 — [leaderboard](https://hotpotqa.github.io/)
- #4 FE2H on ALBERT (Nanjing U, Jan 2022): 71.89/84.44 | 64.98/89.14 | 50.04/76.54 — [leaderboard](https://hotpotqa.github.io/)
- #5 R3 (Fudan & Huawei, May 2022): 71.27/83.57 | 65.25/88.98 | 49.81/76.02 — [leaderboard](https://hotpotqa.github.io/)
- SAE+ (JD AI, May 2021): 70.74/83.61 | 63.70/88.95 | 48.15/75.72; S2G+EGA (SJTU, Jul 2021): 70.92/83.44 | 63.86/88.68 | 48.76/75.47; S2G+ (SJTU, Feb 2021): 70.72/83.53 | 64.30/88.72 | 48.60/75.45; AMGN+ (Jan 2021): 70.53/83.37 | 63.57/88.83 | 47.77/75.24 — [leaderboard](https://hotpotqa.github.io/)
- FE2H on ELECTRA (Feb 2022): 69.54/82.69 | 64.78/88.71 | 48.46/74.90; SpiderNet-large (Kingsoft, Sep 2020): 70.15/83.02 | 63.82/88.85 | 47.54/74.88; GIT (KAIST, Feb 2023): 70.07/82.86 | 62.59/88.53 | 47.22/74.84 — [leaderboard](https://hotpotqa.github.io/)
- HGN-large (Dec 2019): 69.22/82.19 | 62.76/88.47 | 47.11/74.21; HGN (initial, Sep 2019): 66.07/79.36 | 60.33/87.33 | 43.57/71.03 — [leaderboard](https://hotpotqa.github.io/)
- BFR-Graph 70.06/82.20 | 61.33/88.41 | 45.92/74.13; GSAN-large 68.57/81.62 | 62.36/88.73 | 46.06/73.89; FFReader-large (Kyoto U) 68.89/82.16 | 62.10/88.42 | 45.61/73.78 — [leaderboard](https://hotpotqa.github.io/)
- **Long-context transformers**: ETC-large 68.12/81.18 | 63.25/89.09 | 46.40/73.62; Longformer (large) 68.00/81.25 | 63.09/88.34 | 45.91/73.16; RealFormer 67.41/80.59 | 63.38/89.00 | 46.14/73.13 — [leaderboard](https://hotpotqa.github.io/). The Longformer paper reports its HotpotQA result as distractor test joint F1 and says Longformer "places second" among published models at the time — [Longformer](https://arxiv.org/abs/2004.05150). No BigBird HotpotQA-distractor leaderboard entry exists. ETC's entry is the ETC/BigBird-family result, cited in HGN as "ETC-large (Zaheer et al., 2020)" — [HGN](https://arxiv.org/abs/1911.03631).
- C2F Reader (HIT-iFLYTEK, Oct 2019; = Shao et al. 2020 "Is Graph Structure Necessary?"): 67.98/81.24 | 60.81/87.63 | 44.67/72.73 — [leaderboard](https://hotpotqa.github.io/)
- SAE-large (JD AI, AAAI 2020): 66.92/79.62 | 61.53/86.86 | 45.36/71.45; SAE (base): 60.36/73.58 | 56.93/84.63 | 38.81/64.96 — [leaderboard](https://hotpotqa.github.io/)
- TAP2 (IBM, ensemble) 66.64/79.82 | 57.21/86.69 | 41.21/70.65; EPS+BERT(wwm) 65.79/79.05 | 58.50/86.26 | 42.47/70.48 — [leaderboard](https://hotpotqa.github.io/)
- DFGN (SJTU & ByteDance, ACL 2019): 56.31/69.69 | 51.50/81.62 | 33.62/59.82 — [leaderboard](https://hotpotqa.github.io/)
- QFE (NTT, ACL 2019): 53.86/68.06 | 57.75/84.49 | 34.63/59.61 — [leaderboard](https://hotpotqa.github.io/)
- KGNN (Tsinghua) 50.81/65.75 | 38.74/76.79 | 22.40/52.82 — [leaderboard](https://hotpotqa.github.io/)
- **Baseline (Yang et al. 2018, CMU/Stanford/UdeM, Oct 2018)**: 45.60/59.02 | 20.32/64.49 | 10.83/40.16 — [leaderboard](https://hotpotqa.github.io/)
- Answer-only entries (no supporting facts): Unsupervised Decomposition (Perez et al., EMNLP 2020) Ans 66.33/79.34; ChainEx (Chen et al. 2019) 61.20/74.11; DecompRC (Min et al. ACL 2019) 55.20/69.63 — [leaderboard](https://hotpotqa.github.io/)
- A 2021 entry "RoBERTa-L Two-step Model" gets Ans 67.61/80.36 but Sup EM 1.10, so Joint F1 is only 52.50. This shows that Joint depends on supporting-fact prediction — [leaderboard](https://hotpotqa.github.io/)

**Method details from primary papers:**
- **Beam Retrieval** (Zhang et al., 2023): end-to-end beam search over passage *sequences*. A single encoder scores each hop conditioned on the passages already chosen, and beam size B keeps the top-B partial chains. It is trained jointly across hops. Encoders: DeBERTa large for MuSiQue/2Wiki and the "xxlarge version of DeBERTa" for HotpotQA. The large reader was trained on a single RTX4090 and the xxlarge on a single A100. Dev passage-retrieval EM/F1 on HotpotQA: 97.29/98.55 (beam 1) and 97.52/98.68 (beam 2), versus FE2H 96.32/98.02, Smoothing R3 96.85/98.32, S2G 95.77/97.82, SAE 91.98/95.76. Inference time: 124.6 ms/question (beam 1) and 196.4 ms (beam 2), versus FE2H 96.8 ms and R3 127.8 ms. Also SOTA on MuSiQue-Ans test (An 69.2 / Sp 91.4) and the 2Wiki test (Ans EM 88.47) — [Beam Retrieval](https://arxiv.org/abs/2308.08973)
- **FE2H** ("From Easy to Hard", Li et al.): a two-stage document selector (ELECTRA-large) followed by a reader trained first on single-hop SQuAD and then on HotpotQA. Reader PLM is ELECTRA-large or ALBERT-xxlarge-v2. ALBERT reader: 3× A100, ~25 h per stage; ELECTRA: 2× A100, ~6.5 h. Dev selector EM/F1: 96.32/98.02 (large) — [FE2H](https://arxiv.org/abs/2205.11729)
- **HGN** (Fang et al., EMNLP 2020): a hierarchical graph with question, paragraph, sentence and entity nodes, reasoned over with a GAT, plus a two-hop paragraph selector. RoBERTa-large on the test entry. Dev Joint F1 by encoder: BERT-base 66.90; BERT-wwm 72.77; RoBERTa-large 74.37; ALBERT-xxlarge-v2 75.79 (dev, distractor). The same table gives dev Joint F1 for DFGN (BERT-base) 59.89 and SAE (RoBERTa) 72.75 — [HGN](https://arxiv.org/abs/1911.03631)
- **"Is Graph Structure Necessary?"** (Shao et al., EMNLP 2020; the C2F Reader): RoBERTa-large retriever plus reader. In fine-tuning mode, the model with a graph-fusion block gets Joint EM/F1 45.91/73.93 and without it 45.98/73.78, i.e., no gain. The graph helps only when the PLM is frozen (feature-based: 36.45/63.75 with the graph vs 32.26/59.76 without). The authors show that graph attention on a fully connected entity graph degenerates to self-attention. (The paper's Table 2 does not state the split explicitly; it is likely dev.) — [Shao et al. 2020](https://arxiv.org/abs/2004.03096)
- **SAE** (Tu et al., AAAI 2020): "Select, Answer, Explain". A BERT-based document selector with a pairwise learning-to-rank loss, then joint answer and supporting-sentence prediction with a sentence-level GNN. Selector EM 91.98 on dev (as reported in the Beam Retrieval table) — [SAE](https://arxiv.org/abs/1911.00484); [Beam Retrieval](https://arxiv.org/abs/2308.08973)
- **DFGN** (Xiao/Qiu et al., ACL 2019): a dynamically fused entity graph with query-guided masks; BERT-base features — [HGN table](https://arxiv.org/abs/1911.03631); [leaderboard](https://hotpotqa.github.io/)
- **Quark** (Groeneveld et al., AI2, "A Simple Yet Strong Pipeline for HotpotQA", EMNLP 2020): independent sentence scoring, then a QA model on the selected sentences, then a supporting-fact model. The Longformer paper notes "Quark's test results are not available" for distractor. Quark does have a fullwiki test entry (see Q2) — [Longformer](https://arxiv.org/abs/2004.05150); [leaderboard](https://hotpotqa.github.io/)
- **Longformer** reported HotpotQA as distractor test joint F1 and noted that the supporting-fact auxiliary task mattered — [Longformer](https://arxiv.org/abs/2004.05150)

### Inferences
- The ceiling of the distractor leaderboard has barely moved since 2021: Joint F1 went 75.7 (SAE+) → 77.5 (Beam Retrieval). Sup EM (~66) and Joint EM (~50) are the least saturated metrics. A student project claiming improvement on distractor must beat about 50.5 Joint EM / 77.5 Joint F1 on test. That needs leaderboard submission, and it is unclear whether the leaderboard still scores new entries.
- Paragraph selection is effectively solved (≈97.5% gold-pair EM on dev). The remaining errors are in answer-span and sentence-level evidence prediction.
- "Graph reasoning" readers (DFGN, HGN, SAE, AMGN, BFR-Graph, etc.) are heavily explored, and Shao et al. show they are not necessary with fine-tuned transformers. A new graph module is unlikely to count as novel on its own.

### Gaps
- Exact compute for most leaderboard entries (PipNet, R3, S2G) was not reported on the leaderboard. I did not read those papers.
- BigBird's own HotpotQA distractor numbers: the BigBird paper reports HotpotQA results, but I did not verify them from the PDF. The only related leaderboard row is "ETC-large".
- I did not verify the "RD Model", "GIT" and "EGF Reader" papers.
- Many leaderboard entries are anonymous and unpublished. Treat them as existence proofs only.

---

## Q2. Fullwiki (open-domain) setting: retrieval + QA methods and numbers

### Takeaway
The fullwiki leaderboard (test) is topped by AISO (May 2021, Joint EM/F1 44.87/72.00) and Chain-of-Skills (Jan 2023, 45.65/71.65, the best Joint EM). The field moved from TF-IDF plus hyperlinks (Cognitive Graph, GoldEn Retriever, SR-MRS) to learned path retrievers (GRR, HopRetriever, TPRR), then to multi-hop dense retrieval (MDR) and iterative rerankers/condensers (IRRR, Baleen, AISO). By 2021, retrieval recall@20 of the gold pair was above 90% on dev (Baleen), and the bottleneck moved back to the reader.

### Cited Findings
**Official fullwiki leaderboard, TEST set (Ans EM/F1 | Sup EM/F1 | Joint EM/F1)** — [HotpotQA leaderboard](https://hotpotqa.github.io/):
- #1 AISO (ICT-CAS, Zhu, Pang et al., EMNLP 2021; May 2021): 67.46/80.52 | 61.17/86.02 | 44.87/72.00
- #2 Chain-of-Skills (CMU/MSR/UIUC, Ma et al., ACL 2023; Jan 2023): 67.38/80.14 | 61.25/85.31 | 45.65/71.65
- #3 TPRR (Huawei Poisson Lab, Feb 2021): 66.95/79.50 | 59.43/84.25 | 44.37/70.83
- #4 HopRetriever + Sp-search (Huawei Noah's Ark, Li et al. 2020; Jan 2021): 67.13/79.91 | 57.38/83.52 | 43.20/70.61; HopRetriever (Dec 2020): 67.13/79.91 | 57.23/82.59 | 43.10/69.84
- #5 EBS-Large (Samsung SDS, Dec 2020): 66.18/79.32 | 57.29/83.98 | 41.95/70.04
- IRRR+ (Stanford & Samsung, Qi et al. 2020; Nov 2020): 66.33/79.10 | 56.92/83.24 | 42.75/69.60; IRRR (Aug 2020): 65.71/78.19 | 55.93/82.05 | 42.14/68.59
- Recursive Dense Retriever = **MDR** (Facebook AI, Xiong et al., ICLR 2021; Aug 2020): 62.28/75.29 | 57.46/80.86 | 41.78/66.55
- Step-by-Step Retriever (HIT-iFLYTEK, May 2020): 62.95/75.43 | 54.61/80.00 | 40.36/66.22
- DDRQA (Georgia Tech & PKU, May 2020): 62.53/75.91 | 51.01/78.86 | 36.04/63.88
- HGN-albert + SemanticRetrievalMRS IR (Feb 2020): 59.74/71.41 | 51.03/77.37 | 37.92/62.26; HGN + SR-MRS IR (Oct 2019): 56.71/69.16 | 49.97/76.39 | 35.63/59.86
- **Graph Recurrent Retriever** (Salesforce & UW, Asai et al., ICLR 2020), "Robustly Fine-tuned" version, Nov 2019: 60.04/72.96 | 49.08/76.41 | 35.35/61.18; initial Sep 2019 version: 56.04/68.87 | 44.14/73.03 | 29.18/55.31
- Quark + SemanticRetrievalMRS IR (AI2 & IIT, Dec 2019): 55.50/67.51 | 45.64/72.95 | 32.89/56.23
- Transformer-XH (UMD & Microsoft, ICLR 2020, BERT-base): 51.60/64.07 | 40.91/71.42 | 26.14/51.29
- **SemanticRetrievalMRS** (UNC, Nie et al., EMNLP 2019; May 2019): 45.32/57.34 | 38.67/70.83 | 25.14/47.60
- DrKIT (CMU & Google, ICLR 2020): 42.13/51.72 | 37.05/59.84 | 24.69/42.88
- **GoldEn Retriever** (Stanford, Qi et al., EMNLP 2019; May 2019): 37.92/48.58 | 30.69/64.24 | 18.04/39.13
- **Cognitive Graph QA** (Tsinghua KEG & Alibaba, Ding et al., ACL 2019; Feb 2019): 37.12/48.87 | 22.82/57.69 | 12.42/34.92
- MUPPET (Technion, ACL 2019): 30.61/40.26 | 16.65/47.33 | 10.85/27.01; QFE fullwiki 28.66/38.06 | 14.20/44.35 | 8.69/23.10
- **Baseline** (Oct 2018): 23.95/32.89 | 3.86/37.71 | 1.85/16.15
- Recent low entries: "HGN Model-reproduce" (Peking U, May 2023) Joint F1 28.40, and "Mistral multi hop with very large sources" (Jun 25 2024), Ans 7.98/22.14 with no supporting facts. These show that naive LLM or reproduced systems score badly under the official protocol — [leaderboard](https://hotpotqa.github.io/)

**Retrieval metrics (DEV, fullwiki):**
- MDR direct retrieval (passage-chain recall of both gold passages): R@2/R@10/R@20 = 65.9/77.5/80.2, versus TF-IDF 10.3/29.1/36.8, TF-IDF+Linked 17.3/50.0/62.7, DrKIT 38.3/67.2/71.0. Ablations: without order 17.6 R@2; single-hop DPR 25.2/45.4/52.1 — [MDR](https://arxiv.org/abs/2009.12756)
- Reranked retrieval (input to the reader), SP-passage EM / answer recall: Semantic Retrieval 63.9/77.9; Graph Recurrent Retriever 75.7/87.5; MDR direct 65.9/75.4; MDR + reranking 81.2/88.2 — [MDR](https://arxiv.org/abs/2009.12756)
- MDR compute: 8× 32GB V100. The best reader is ELECTRA-large. MDR does not use hyperlinks, whereas GRR, Transformer-XH and HGN exploit Wikipedia hyperlinks, which the authors call an "unreasonably strong advantage" on HotpotQA — [MDR](https://arxiv.org/abs/2009.12756)
- MDR test (from its Table 5): GoldEn 37.9/48.6 | 30.7/64.2 | 18.9/39.1 (the leaderboard shows Joint EM 18.04); SR 46.5/58.8 | 39.9/71.5 | 26.6/49.2 (the leaderboard shows 45.32/57.34 … 25.14/47.60, a possible version difference); MDR 62.3/75.3 | 57.5/80.9 | 41.8/66.6 — [MDR](https://arxiv.org/abs/2009.12756). *Conflict flag: small mismatches between the numbers MDR copied and the current leaderboard (GoldEn Joint EM, SR-MRS). Prefer the leaderboard.*
- **Baleen** (Khattab, Potts, Zaharia, NeurIPS 2021): FLIPR retriever with condensed retrieval (keeps a short set of facts across hops) and latent hop ordering. HotpotQA dev: Passage-pair EM 86.7, P-R@20 93.3, Ans-R@20 96.3, versus MDR 81.2/82.9/89.4, IRRR P-EM 84.1, ColBERT-Hop 85.2/90.3/94.7. Reader: ELECTRA-large. The authors argue this "effectively reduces the open-retrieval task to a narrow distractor-setting one" and therefore focus on HoVer (up to 4 hops) — [Baleen](https://arxiv.org/abs/2101.00436)
- **Graph Recurrent Retriever** (Asai et al.): an RNN over BERT paragraph encodings traverses the Wikipedia hyperlink graph with beam search, and a reader re-ranks the reasoning paths. It is evaluated with Answer Recall and Paragraph EM — [GRR](https://arxiv.org/abs/1911.10470)
- **IRRR** (Qi et al., EMNLP 2021 "Answering Open-Domain Questions of Varying Reasoning Steps from Text"): an iterative retriever, reader and reranker (ELECTRA) issuing BM25 queries. It introduced the BeerQA benchmark, which merges SQuAD Open, HotpotQA and new 3+ hop questions — [IRRR](https://arxiv.org/abs/2010.12527); [HotpotQA site](https://hotpotqa.github.io/)

### Inferences
- Fullwiki test SOTA has been static since 2021–2023 at about 45 Joint EM / 72 Joint F1. No LLM-era system appears on the official fullwiki leaderboard with competitive supporting-fact predictions.
- Retrieval-only claims should be reported as dev P-EM / P-R@k with the official 2017 Wikipedia abstracts corpus so they compare with MDR (81.2 reranked P-EM) and Baleen (86.7 P-EM, 93.3 P-R@20).

### Gaps
- I did not read the AISO, TPRR, Chain-of-Skills or HopRetriever papers for their retrieval metrics. My arXiv fetch for HopRetriever returned the wrong paper. The fullwiki test numbers above come from the leaderboard.
- I did not verify Cognitive Graph or GoldEn Retriever compute.

---

## Q3. LLM-era approaches evaluated on HotpotQA

### Takeaway
LLM papers almost never use the official protocol. They report answer-only EM/F1 (sometimes "cover EM" or LLM-judge accuracy), on 500 or 1,000 dev questions or the full dev set, with their own corpus (2018 DPR Wikipedia, the 2017 HotpotQA abstracts, or a small closed corpus) and retriever. Prompted few-shot systems get about 35–51 EM (ReAct 35.1 with CoT-SC hybrid; DSP 51.4/62.9 on a 1k subset; IRCoT 49.3/60.7 on 500). RL-trained 3B–7B search agents (2025) report about 38–47 EM on full dev under the Search-R1/FlashRAG protocol (top-3 E5 passages). None report supporting facts.

### Cited Findings
**Prompting and pipeline methods (2022–2024):**
- **ReAct** (Yao et al., ICLR 2023), PaLM-540B, HotpotQA question-only with a Wikipedia API (search/lookup): ReAct EM 27.4; Act 25.7; CoT-SC→ReAct 34.2; ReAct→CoT-SC 35.1. Reported comparisons: Standard 27.1, CoT 28.9, CoT-SC 33.8 (from Wang et al.), and "supervised SoTA" 67.5. Evaluated on a dev subset — [ReAct](https://arxiv.org/abs/2210.03629)
- **Self-Ask** (Press et al., 2022): the paper does not evaluate HotpotQA at all (no mention in the text). It uses 2Wiki, MuSiQue, Bamboogle and Compositional Celebrities — [Self-Ask](https://arxiv.org/abs/2210.03350). Other papers re-implement it on HotpotQA (see below).
- **DSP** (Khattab et al., 2022), GPT-3.5 text-davinci-002 + ColBERTv2, open "fullwiki" setting with dev/test **subsampled to 1,000 questions**: Task-aware DSP 51.4 EM / 62.9 F1; Retrieve-then-Read 36.9/46.1; Vanilla LM 28.3/36.4; Self-ask w/ ColBERTv2 25.2/33.2 — [DSP](https://arxiv.org/abs/2212.14024)
- **IRCoT** (Trivedi et al., ACL 2023): interleaves a CoT step with BM25 retrieval. Uses 100 dev questions for tuning and 500 other dev questions for testing. GPT3 code-davinci-002: HotpotQA EM/F1 49.3/60.7 (bridge subset 45.8/58.5). It also compares with DecomP (2023) F1 53.5 and DSP 51.4/62.9 (DSP is 2 F1 points higher than IRCoT on HotpotQA). Flan-T5-XXL answer F1: NoR-Direct 25.3, OneR-Direct 49.7, IRCoT-Direct 59.1; GPT3 IRCoT-CoT 60.7 F1. Retrieval recall improves by 11–21 points over one-step retrieval — [IRCoT](https://arxiv.org/abs/2212.10509)
- **Iter-RetGen** (Shao et al., EMNLP Findings 2023), text-davinci-003, **first 500 dev questions**, 3-shot: HotpotQA EM/F1/Acc† (Acc† is judged by an LLM) — Direct (no retrieval) 21.9/36.8/44.8; CoT 30.0/44.1/50.0; Direct w/ retrieval 31.6/44.7/53.3; ReAct 24.9/44.7/61.1; Self-Ask 36.8/55.2/64.8; DSP 43.8/55.0/60.8; Iter-RetGen T=2 44.1/58.6/71.2; T=4 45.8/61.1/73.4 — [Iter-RetGen](https://arxiv.org/abs/2305.15294)
- **Adaptive-RAG** (Jeong et al., NAACL 2024): a T5 classifier routes each query to no-retrieval, single-step or multi-step (IRCoT-style) retrieval by predicted complexity. It uses 500 test samples per dataset. FLAN-T5-XL (3B), HotpotQA EM/F1/Acc/steps/time: No Retrieval 16.60/22.71/17.20; Single-step 34.40/46.15/36.40 (1.00 step); Adaptive Retrieval 23.60/32.22/25.00; Self-RAG* 6.80/17.53/29.60; **Adaptive-RAG 42.00/53.82/44.40 (3.55 steps, 5.99× time)**; Multi-step 44.60/56.54/47.00 (5.53 steps, 9.38× time); Adaptive-RAG with an oracle classifier 51.20/64.00/54.80. Also run with FLAN-T5-XXL and GPT-3.5-turbo-instruct — [Adaptive-RAG](https://arxiv.org/abs/2403.14403). **Highly relevant to a "router model" project: Adaptive-RAG is the direct prior art for query-complexity routing on HotpotQA.**
- **Self-RAG**: its HotpotQA numbers exist only as a baseline re-run in Adaptive-RAG (Self-RAG* with LLaMA2-7B checkpoint: EM 6.80, F1 17.53, Acc 29.60), which the Adaptive-RAG authors mark as not directly comparable — [Adaptive-RAG](https://arxiv.org/abs/2403.14403)
- **EfficientRAG** (Zhuang et al., EMNLP 2024): a small DeBERTa-v3-large (304M) "Labeler/Tagger" and "Filter" generate the next-hop query without calling the LLM each hop. Llama-3-8B-Instruct reader, Contriever-MSMARCO retriever. HotpotQA EM/F1/Acc: EfficientRAG 50.59/57.93/57.86; Iter-RetGen iter3 56.76/60.89/57.56; SelfAsk 33.58/39.10/42.36; Direct-R@10 38.24/44.55/44.56; CoT 27.99/34.05/30.53. Retrieval recall@K on HotpotQA 81.84 with only 6.41 chunks — [EfficientRAG](https://arxiv.org/abs/2408.04259)
- **HippoRAG** (Gutiérrez et al., NeurIPS 2024): a KG built by an LLM plus Personalized PageRank. It uses **1,000 dev questions** and a **closed corpus of 9,221 passages** (all candidate passages of those questions, following IRCoT), which is *not fullwiki*. The authors themselves note HotpotQA "has been found to be a much weaker test for multi-hop reasoning due to many spurious signals" — [HippoRAG](https://arxiv.org/abs/2405.14831)
- **HippoRAG 2** (Feb 2025), same style of subset and corpus, Llama-3.3-70B reader. HotpotQA QA F1: None 47.3; Contriever 62.3; BM25 63.4; NV-Embed-v2 75.3; RAPTOR 69.5; GraphRAG 68.6; LightRAG 2.4; HippoRAG 63.5; **HippoRAG 2 75.5**. Passage Recall@5: BM25 74.8; NV-Embed-v2 94.5; RAPTOR 86.9; HippoRAG 77.7; HippoRAG 2 96.3 — [HippoRAG 2](https://arxiv.org/abs/2502.14802)
- **Search-o1** (Li et al., Jan 2025): QwQ-32B large reasoning model with agentic search and a "Reason-in-Documents" module. HotpotQA EM/F1: Search-o1 45.2/57.3; RAgent-QwQ-32B 43.0/55.2; RAG-QwQ-32B 34.2/46.4; direct QwQ-32B 25.4/33.3; Llama3.3-70B direct 37.8/49.1 — [Search-o1](https://arxiv.org/abs/2501.05366) (the eval subset size was not stated in the text I parsed; see Gaps)

**RL-trained search agents (2025–2026):**
- **Search-R1** (Jin et al., COLM 2025): PPO/GRPO RL with a masked loss on retrieved tokens and an outcome EM reward. Training data is NQ + HotpotQA train (so HotpotQA is in-domain). The retriever is E5 over the 2018 DPR Wikipedia dump with top-3 passages. Evaluated on full HotpotQA dev with EM. Qwen2.5-7B: Search-R1-base 0.433, Search-R1-instruct 0.370; RAG 0.299; IRCoT 0.133; Search-o1 0.187; R1 (no search) 0.242; Rejection-sampling SFT 0.331; Direct 0.183; CoT 0.092. Qwen2.5-3B: Search-R1-instruct 0.324, base 0.284. GRPO variants: 7B-instruct 0.386. Compute: one node with 8× H100 — [Search-R1](https://arxiv.org/abs/2503.09516)
- **R1-Searcher** (Song et al., Mar 2025): two-stage outcome RL (retrieve reward, then answer reward) with Reinforce++, trained on selected HotpotQA and 2Wiki train items (in-domain). Metrics are Cover-EM (ACC_R) and GPT-4o-mini judge (ACC_L). HotpotQA: Qwen-2.5-7B-Base RL-Zero ACC_R 0.654 / ACC_L 0.750; Llama-3.1-8B-Instruct 0.648/0.746. Baselines with GPT-4o-mini: IRCoT 0.434/0.308, Self-Ask 0.392/0.462, Iter-RetGen 0.374/0.456, Standard RAG 0.342/0.450. Three-decimal values in steps of 0.002 suggest a 500-question subset (inference, not stated in the parsed text) — [R1-Searcher](https://arxiv.org/abs/2503.05592)
- **ReSearch** (Chen et al., Mar 2025): GRPO RL with search as part of the reasoning chain, trained **only on MuSiQue** (so HotpotQA is out-of-domain). Full HotpotQA dev (7,405). Training on 8×8 H800. HotpotQA EM / LLM-judge: ReSearch-Qwen-7B-Instruct 43.52/63.62; ReSearch-Qwen-32B-Instruct 46.73/67.70; baselines (7B) Naive RAG 31.90/49.59, Iter-RetGen 34.36/52.22, IRCoT 30.33/52.06 — [ReSearch](https://arxiv.org/abs/2503.19470)
- **StepSearch** (May 2025): step-wise PPO (StePPO) with information-gain and redundancy-penalty rewards per search step, trained on MuSiQue-derived data (19k). HotpotQA EM/F1 (wiki-18): Qwen2.5-7B StepSearch-base 0.380/0.493, instruct 0.386/0.502. Its reproduced baselines: Search-R1-base 0.432/0.547, ZeroSearch-instruct 0.325/0.432, ReSearch-instruct 0.362/0.471. 3B: StepSearch-instruct 0.345/0.452 vs Search-R1-instruct 0.304/0.401 — [StepSearch](https://arxiv.org/abs/2505.15107). Note: StepSearch does *not* beat Search-R1 on HotpotQA at 7B, likely because Search-R1 trains in-domain on HotpotQA.
- **ZeroSearch** (Alibaba, May 2025): RL in which a fine-tuned LLM simulates the search engine (useful or noisy documents) instead of calling a real search engine during training. HotpotQA numbers as reproduced in StepSearch: Qwen2.5-7B instruct EM/F1 0.325/0.432 — [ZeroSearch](https://arxiv.org/abs/2505.04588); [StepSearch](https://arxiv.org/abs/2505.15107)
- A 2025 survey catalogues the RL-based agentic search line (Search-R1, R1-Searcher, ReSearch, ZeroSearch, StepSearch, etc.), where HotpotQA is part of the standard 7-dataset QA suite (NQ, TriviaQA, PopQA, HotpotQA, 2Wiki, MuSiQue, Bamboogle) — [RL Agentic Search survey](https://arxiv.org/abs/2510.16724); [Awesome list](https://github.com/ventr1c/Awesome-RL-based-Agentic-Search-Papers)

### Inferences
- The "Search-R1 protocol" (E5 on the 2018 Wikipedia dump, top-3 passages, EM on full dev, Qwen2.5-3B/7B) has become the de-facto 2025–2026 comparison point for small-LLM agents. On it, HotpotQA dev EM is about 0.43 (7B) and about 0.32 (3B).
- Under a small closed corpus (IRCoT/HippoRAG style, 1k questions), F1 of 75+ is achievable with strong 7B embedders and a 70B reader. That is a very different task from fullwiki. Do not compare these with leaderboard fullwiki numbers (Ans F1 80.5 on test).
- For a **router** project: Adaptive-RAG (complexity classifier), Self-RAG (reflection tokens), EfficientRAG (small model decides next hop / termination) and "adaptive search depth" RL (e.g., AutoSearch, 2026) are the closest prior art. Novelty must be claimed relative to these.

### Gaps
- **FLARE** (Jiang et al., 2023): I believe it reports on 2WikiMultihopQA, not HotpotQA, but I did not verify this.
- **GenGround** (Shi et al., ACL 2024): not retrieved, no numbers verified.
- The Search-o1 open-domain QA subset size was not confirmed from the parsed text.
- **AutoSearch** (arXiv 2604.17337, 2026, adaptive search depth via RL): found in search, but the PDF text did not parse, so no numbers.
- Chain-of-thought-only numbers for GPT-4-class or 2025–2026 frontier models on HotpotQA under a standard protocol were not found in primary sources.

---

## Q4. Dataset critiques, robustness sets, and follow-up datasets

### Takeaway
Several 2019 studies showed that much of HotpotQA can be solved without multi-hop reasoning. A single-paragraph BERT reaches 67 F1 in distractor. Sentence-factored models locate the answer in 45–57% of cases. Adversarial distractor documents cause large drops. These findings motivated 2WikiMultiHopQA (2020, templated from Wikidata with evidence triples), MuSiQue (2021/22, composed single-hop questions built to resist "disconnected reasoning"), and Bamboogle (2022, questions not answerable by Google search). Recent LLM papers still say HotpotQA is a weaker multi-hop test.

### Cited Findings
- **Min et al. 2019, "Compositional Questions Do Not Necessitate Multi-hop Reasoning"** (ACL 2019): single-paragraph BERT (scores each paragraph independently) gets distractor F1 67.08 and open (fullwiki) F1 38.40. For comparison: QFE 68.06/38.06, DFGN+BERT 68.49, DecompRC 69.63/40.65, GRN 66.71/36.48. Humans shown only partial evidence can still answer over 80% of questions. Many questions are solvable in one hop because of entity-type cues or redundant facts — [Min et al. 2019](https://arxiv.org/abs/1906.02900)
- **Chen & Durrett 2019, "Understanding Dataset Design Choices for Multi-hop Reasoning"** (NAACL 2019): sentence-factored models (which by design cannot do multi-hop reasoning) find the answer sentence on HotpotQA dev in 45.4% of cases (factored) and 57.2% (factored BiDAF), versus 5.4% random — [Chen & Durrett 2019](https://arxiv.org/abs/1904.12106)
- **Jiang & Bansal 2019, "Avoiding Reasoning Shortcuts"** (ACL 2019): builds adversarial distractor documents (AddDoc) that mimic the shortcut answer. The 1-hop baseline EM falls from 42.32 (regular dev) to 26.67 (adversarial dev). Adversarial training plus a 2-hop model with a control unit and supporting-fact supervision recovers 46.87 on adversarial dev — [Jiang & Bansal 2019](https://arxiv.org/abs/1906.07132)
- **2WikiMultiHopQA** (Ho et al., COLING 2020): combines Wikipedia text with Wikidata triples. Templates and logical rules "guarantee the multi-hop steps". It adds *evidence* (reasoning path as triples) for explanation and evaluating reasoning — [2WikiMultiHopQA](https://arxiv.org/abs/2011.01060)
- **MuSiQue** (Trivedi et al., TACL 2022): built bottom-up by composing connected single-hop questions (2–4 hops) to block disconnected reasoning. It has a 3x larger human-machine gap than earlier datasets, a single-hop model drops 30 F1, and it adds unanswerable contrast questions (MuSiQue-Full). The authors attribute shortcuts in prior datasets to "overly specific sub-questions, train-test leakage, and insufficient distractors" — [MuSiQue](https://arxiv.org/abs/2108.00573)
- **Bamboogle** (Press et al., 2022; 125 questions): used in many LLM-era evaluations. ReSearch lists 125 test samples — [Self-Ask](https://arxiv.org/abs/2210.03350); [ReSearch](https://arxiv.org/abs/2503.19470)
- **BeerQA** (Qi et al. 2021): open-domain questions with varying hop counts, which incorporates HotpotQA — [HotpotQA site](https://hotpotqa.github.io/); [IRRR](https://arxiv.org/abs/2010.12527)
- The HippoRAG authors (2024) include HotpotQA "even though it has been found to be a much weaker test for multi-hop reasoning due to many spurious signals" — [HippoRAG](https://arxiv.org/abs/2405.14831)
- MDR notes that HotpotQA's construction guarantees that gold passage chains follow hyperlinks. This gives hyperlink-based retrievers an "unreasonably strong advantage" — [MDR](https://arxiv.org/abs/2009.12756)

### Inferences
- Any HotpotQA-only novelty claim for "multi-hop reasoning" is weak unless it is also evaluated on MuSiQue and/or 2Wiki, or on adversarial HotpotQA (Jiang & Bansal's AddDoc). Reporting the bridge and comparison splits separately is also informative.

### Gaps
- I did not verify whether the adversarial HotpotQA set (AddDoc) is still publicly downloadable. The paper's code release was not checked.
- I did not look up Mavi et al.'s "Multi-hop Question Answering" survey (2022/2024) for its taxonomy.

---

## Q5. Official evaluation script, metric definitions, and leaderboard status

### Takeaway
The official scorer is `hotpot_evaluate_v1.py`. Answer EM/F1 use SQuAD-style normalization plus special handling for yes/no. Supporting-fact EM/F1 are computed over the set of (title, sentence_id) pairs. **Joint precision = answer precision × SP precision, joint recall = answer recall × SP recall, Joint F1 = their harmonic mean, and Joint EM = Ans EM × SP EM**, averaged per question. The leaderboard page is still online and listed submissions up to June 2024. No 2025–2026 entries are visible.

### Cited Findings
- Usage: `python hotpot_evaluate_v1.py <path_to_prediction> <path_to_gold>`. The test set is scored through a submission guide on CodaLab — [HotpotQA site](https://hotpotqa.github.io/)
- Script logic: `f1_score` returns zero if either prediction or gold normalizes to 'yes', 'no' or 'noanswer' and they differ. `update_sp` treats predicted and gold SP as sets of tuples and computes tp/fp/fn → precision/recall/F1, with SP EM = 1 only if fp+fn = 0. Joint: `joint_prec = prec * sp_prec`, `joint_recall = recall * sp_recall`, `joint_f1 = 2*joint_prec*joint_recall/(joint_prec+joint_recall)`, `joint_em = em * sp_em`. All metrics are averaged over N gold examples, and a missing answer or SP counts as zero for that question — [hotpot_evaluate_v1.py](https://github.com/hotpotqa/hotpot/blob/master/hotpot_evaluate_v1.py)
- Data: training set (535MB), dev distractor, dev fullwiki, test fullwiki (46MB), plus the processed Wikipedia corpus used for fullwiki (CC BY-SA 4.0) — [HotpotQA site](https://hotpotqa.github.io/)
- Leaderboard activity: the most recent visible entries are distractor Beam Retrieval (Aug 7, 2023) and GIT (Feb 25, 2023), and fullwiki "HGN Model-reproduce" (May 15, 2023), an unnamed entry (Feb 12, 2024) and "Mistral multi hop with very large sources" (Jun 25, 2024). The page footer still says "Copyright © HotpotQA Team, 2018-2019" — [HotpotQA site](https://hotpotqa.github.io/)

### Inferences
- The leaderboard appears to be passively maintained: it accepted scores at least until mid-2024. Whether CodaLab submissions still work in 2026 is unverified (CodaLab Worksheets availability may have changed). A student team should plan on reporting **dev-set** numbers with the official script and say so clearly.
- LLM-era answer-only numbers ignore SP. A project that predicts supporting facts with an LLM or router pipeline and reports Joint EM/F1 on the full dev set would be directly comparable to the classic readers, and few LLM papers do this.

### Gaps
- I did not verify the exact test-set size or the CodaLab submission instructions (the guide page was not fetched).
- I did not check whether the HotpotQA team formally closed the leaderboard.
