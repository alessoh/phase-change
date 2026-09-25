# Technical Frontiers: Open Problems for AI in Logistics and Supply Chain Decision-Making (state as of 2025-2026)

Method note: arxiv.org, huggingface.co, ijcai.org and cra.org were blocked for full-text fetching in this session, so findings below come from search-result abstracts/snippets of the cited primary papers (arXiv abstract pages, Amazon Science, INFORMS, GitHub). Numbers were cross-checked where more than one snippet reported them, but the report writer should treat exact figures as "as stated in abstract." Scope deliberately excludes classical solver state-of-the-art and generic neural-solver gaps (covered in `research_notes/Logistics and scheduling algorithms/ml_llm_optimization.md`); focus here is on what must be improved or created next.

## Q1. What do surveys, position papers and roadmaps identify as the key open problems (generalization, uncertainty/disruptions, decision-focused learning, real-time re-optimization, multi-agent coordination, causal inference)?

### Takeaway
The recurring research agenda is: (1) make learned methods generalize to large, structurally irregular, real instances (current ML solvers still lose to classical solvers as scale/structure grows); (2) integrate prediction and optimization under uncertainty (decision-focused learning) in a way that scales to discrete, large, online problems; (3) extend static solvers to dynamic/stochastic re-optimization; (4) use hybrid "learn to shrink the problem, let the solver finish" designs at industrial scale; and (5) move from correlational to causal/counterfactual models for disruption response. Multi-agent coordination and causal methods are much less mature than the first three, with mostly small-scale or simulation evidence.

### Cited Findings

Generalization to large and real instances
- FrontierCO (ICLR 2026) evaluates 16 ML solvers (GNN, hybrid neural-symbolic, LLM agents) on 8 CO problems across 5 domains (routing, scheduling, facility location, graph problems), using instances from competitions and public repositories (DIMACS, TSPLib) with "easy" (now solvable) and "hard" (open or computationally intensive) sets. Finding: "a persistent performance gap widens under structurally challenging and large instance sizes (e.g., TSP up to 10M nodes; MIS up to 8M)", while some cases exist where ML beats classical solvers — [arXiv 2505.16952](https://arxiv.org/abs/2505.16952); [ML Anthology ICLR 2026](https://mlanthology.org/iclr/2026/feng2026iclr-frontierco/); data at [GitHub](https://github.com/sunnweiwei/FrontierCO)
- The FrontierCO authors frame the core limitation as: much reported ML progress "relies on small-scale, synthetic benchmarks that fail to capture real-world structure and scale", with methods trained/evaluated on synthetic generators, "leaving open how they perform on irregular, competition-grade, or industrial datasets" — [arXiv 2505.16952v1](https://arxiv.org/html/2505.16952v1)
- Position paper "Time to Rethink AI for Combinatorial Optimization" / "Unrealized Expectations" (TMLR 04/2026): even on in-distribution random graphs, leading AI-inspired GPU methods are consistently beaten by classical KaMIS for Maximum Independent Set, and some fail to beat a simple degree-based greedy heuristic; a "serialization" analysis shows non-backtracking methods (e.g., GFlowNet-based LTFT) effectively reason like greedy. Recommendations: more rigorous benchmarking against established solvers, principled integration of classical heuristics, adding explicit backtracking — [arXiv 2502.03669](https://arxiv.org/abs/2502.03669); [ADS abstract](https://ui.adsabs.harvard.edu/abs/2025arXiv250203669W/abstract)
- Foundation models for routing: RouteFinder (TMLR 2025 → ICML 2026, AI4CO) builds a single model over many VRP variants (24, later 48) via a unified environment, attribute embeddings, mixed-batch training, and adapter layers for unseen attributes — but generalization comes "at a slight expense in solution quality" vs. variant-specific models; authors list outperforming SOTA traditional OR solvers (e.g., via decomposition) as future work — [GitHub ai4co/routefinder](https://github.com/ai4co/routefinder); [arXiv 2406.15007](https://arxiv.org/abs/2406.15007)

Hybrid ML+solver at industrial scale
- PROPEL (Akhlaghi, Zandehshahvar, Van Hentenryck, Apr 2025) targets industrial supply chain planning MIPs with integer (non-binary) + continuous variables and flow-balance/capacity constraints; largest instances >1.1M rows, 2.1M columns, ~1M integer variables. Uses supervised learning to predict which variables are fixed to zero (not all values), plus deep RL to choose which fixed variables to relax when SL falls short — [arXiv 2504.07383](https://arxiv.org/abs/2504.07383)

Uncertainty: decision-focused learning (DFL) / predict-then-optimize
- The main DFL survey + benchmark (Mandi, Kotary, Berden, Mulamba, Bucarey, Guns, Fioretto; JAIR vol. 80, 2024) reviews gradient-based and gradient-free methods and empirically evaluates 11 methods on 7 problems; code/benchmark at PredOpt — [arXiv 2307.13565](https://arxiv.org/abs/2307.13565); [JAIR](https://www.jair.org/index.php/jair/article/view/15320/27076); [PredOpt benchmarks](https://github.com/PredOpt/predopt-benchmarks)
- Core open technical problems stated in the DFL literature: differentiating through the optimization problem; discrete decision variables give discontinuous/piecewise-constant mappings that break gradient learning; designing surrogates becomes harder as number and types of variables/constraints grow — [Emergent Mind summary of PtO literature](https://www.emergentmind.com/topics/predict-then-optimize-po-approach); [Guns CPAIOR'24 keynote](https://people.cs.kuleuven.be/~tias.guns/files/CPAIOR24-keynote-DFL.pdf)
- Active 2025-2026 responses: scalable DFL via online-trainable surrogates ([arXiv 2512.03861](https://arxiv.org/pdf/2512.03861)); online DFL (ICLR 2026, [arXiv 2505.13564](https://arxiv.org/pdf/2505.13564)); score-function gradient estimation to widen DFL applicability ([arXiv 2307.05213](https://arxiv.org/pdf/2307.05213)); decision-focused fine-tuning for limited data ([arXiv 2501.01874](https://arxiv.org/html/2501.01874v1)); moving beyond local loss functions ([arXiv 2305.16830](https://arxiv.org/pdf/2305.16830))
- Evidence DFL matters in supply chains: decision-aware learning for health-supply-chain medicine allocation reported up to an order-of-magnitude improvement vs. naive predict-then-optimize — [arXiv 2211.08507](https://arxiv.org/pdf/2211.08507)
- Industrial evidence that accuracy metrics mislead: a decision-aware benchmark for intermittent spare-parts demand (contract logistics) evaluated 38 methods (classical, intermittent, ML, DL, pretrained foundation models) on 20,330 real multi-item orders; forecast-accuracy rank and order-service rank were *negatively* correlated (−0.555); service tracked direction of cumulative forecast bias (incl. over-prediction in zero-demand periods), not point accuracy — [arXiv 2609.13840](https://arxiv.org/abs/2609.13840)

Real-time / dynamic re-optimization
- EURO Meets NeurIPS 2022 VRP Competition (ORTEC real data) included a dynamic VRPTW where orders arrive over the day in delivery waves; organizers noted "extending a static solver to deal with dynamic requests is a significant challenge", and that the OR community uses simplistic ML while the ML community's deep learning "fails to outperform OR baselines" — [PMLR v220 (Kool et al.)](https://proceedings.mlr.press/v220/kool23a.html); [competition site](https://euro-neurips-vrp-2022.challenges.ortec.com/)
- Winning dynamic approach (TUM, Baty et al.) used combinatorial-optimization-enriched ML (a structured learning layer choosing which requests to dispatch, with a CO layer) — published in Transportation Science — [INFORMS TS](https://pubsonline.informs.org/doi/abs/10.1287/trsc.2023.0107); [arXiv 2304.00789](https://arxiv.org/pdf/2304.00789); [code](https://github.com/tumBAIS/euro-meets-neurips-2022)

Roadmaps
- CCC/INFORMS/ACM SIGAI AI/OR workshop series (2021–2024, three workshops; AI/OR-3 report published Apr 2025, "Making a Case for Research Collaboration Between AI and OR Experts") aims at a joint strategic research vision; decision-making under uncertainty (RL, MDPs and generalizations) flagged as an area active in both communities — [CCC workshop reports](https://cra.org/ccc/resources/workshop-reports/); [AI/OR-3 report PDF](https://cra.org/ccc/wp-content/uploads/sites/2/2025/04/Making-a-Case-for-Research-Collaboration-Between-Artificial-Intelligence-and-Operations-Research-Experts-AI-OR-3-Report.pdf); [AI/OR Workshop 2 report](https://www.researchgate.net/publication/369924531_Artificial_IntelligenceOperations_Research_Workshop_2_Report_Out)

Multi-agent coordination
- LLM-based multi-agent consensus-seeking for supply chains (IJPR 2025) reported reduced bullwhip effect in simulation — [Taylor & Francis IJPR](https://www.tandfonline.com/doi/full/10.1080/00207543.2025.2604311)
- REALM-Bench (2025) targets real-world planning for LLMs and multi-agent systems and names limitations such as chain-of-thought error propagation and inconsistent reasoning — [arXiv 2502.18836](https://arxiv.org/html/2502.18836v1)
- Conflict-aware agentic last-mile delivery (ORBITER, 2026) — [arXiv 2608.18846](https://arxiv.org/pdf/2608.18846)

Causal inference for supply chains
- Causal ML for supply chain risk prediction and intervention planning (IJPR 2025) — [Taylor & Francis](https://www.tandfonline.com/doi/full/10.1080/00207543.2025.2458121); RL-based causal discovery for root-cause attribution of delivery risks — [arXiv 2408.05860](https://arxiv.org/pdf/2408.05860); causal AI to diagnose out-of-stock events — [ScienceDirect](https://www.sciencedirect.com/science/article/pii/S2666827025001483)
- Stated open problems: counterfactuals are unobserved so validation relies on synthetic-data refutation/sensitivity tests; lack of uncertainty quantification on outputs; low practitioner trust in black-box models — [comparative causal ML assessment, automotive, ScienceDirect](https://www.sciencedirect.com/science/article/pii/S2949863525000160); [Causal ML IJPR 2025](https://www.tandfonline.com/doi/full/10.1080/00207543.2025.2458121)
- Causal analysis is also emerging for time series foundation models themselves — [arXiv 2608.24303](https://arxiv.org/pdf/2608.24303)

### Inferences
- The most credible near-term frontier is hybrid: ML that reduces/warm-starts/decomposes problems for exact or metaheuristic solvers (PROPEL, CO-enriched ML for DVRPTW) rather than end-to-end neural solvers, given FrontierCO and the MIS position paper.
- DFL is methodologically mature on small benchmarks but the stated open problems (discrete decisions, scale, many constraint types, online settings) are exactly what logistics requires; the negative accuracy/service correlation result is strong empirical motivation for decision-aware evaluation.
- Multi-agent coordination and causal inference in supply chains are at an early stage: evidence is mostly simulations (beer-game-style bullwhip) or single-company case studies, not validated counterfactual decision-making at network scale.

### Gaps
- Could not retrieve full text of the AI/OR-3 CCC report (cra.org blocked) to extract its specific recommendations; only its title/date and series goals are cited.
- No NSF/DOE roadmap specific to logistics AI was found in this round.
- No rigorous evaluation found of causal-inference methods for disruption response at industrial scale.
- No published quantitative results for PROPEL (speedup/gap) could be extracted from snippets.

## Q2. Limits of LLMs and AI agents for logistics decisions today, and proposed fixes

### Takeaway
LLMs fail on (a) correct formulation (benchmarks themselves are noisy), (b) interacting with solvers to repair infeasible models, (c) operational rationality (e.g., newsvendor/inventory theory), (d) human-like biases (demand anchoring, bullwhip), and (e) long-horizon consistent policies. The best-evidenced fixes are solver-in-the-loop training with verifiable (solver-checked) rewards, IIS-guided repair, domain rationality checks, strategy/execution separation, and "OR orchestrates GenAI" architectures. Time-series foundation models (Chronos-2, TimesFM-2.5) are strong on general benchmarks but have documented weaknesses for intermittent/sparse demand and evaluation leakage.

### Cited Findings

Formulation errors and noisy benchmarks
- LLM4OR survey (Xiao et al., IJCAI 2025): error rates in optimization-modeling benchmarks are high — most exceed 15%, IndustryOR up to 54%; NL4Opt ≥26.4% error; MAMO EasyLP ≥8.13%, ComplexLP ≥23.7%. Error causes: logically flawed problem descriptions (e.g., unbounded), poorly defined parameters making models unsolvable, etc. Cleaned versions released and adopted by later work (e.g., 214 NL4Opt, 545 EasyLP, 111 ComplexLP instances) — [arXiv 2508.10047](https://arxiv.org/pdf/2508.10047); [IJCAI 2025](https://www.ijcai.org/proceedings/2025/1192.pdf); usage in [Opt-Verifier arXiv 2605.29556](https://arxiv.org/pdf/2605.29556)
- New evaluation proposals: graph-theoretic evaluation of formulations (ORGEval) rather than only matching optimal objective values — [arXiv 2510.27610](https://arxiv.org/pdf/2510.27610); dual-side verification (Opt-Verifier) — [arXiv 2605.29556](https://arxiv.org/pdf/2605.29556)

Solver interaction / repair (supply chain specific)
- OptiRepair (Ao, Simchi-Levi, Wang — MIT; Feb 2026): 22 API models from 7 families on 976 multi-echelon supply chain problems. API models restored only 27.6% of infeasible formulations (avg), best API model reached 42.2% "Rational Recovery Rate" (feasible and operationally rational) vs. 21.3% average; two trained 8B models reached 81.7%. Roughly 1 in 4 feasible repairs violate supply chain theory. Method: IIS-guided feasibility repair + five inventory-theory rationality checks; training with self-taught reasoning and solver-verified rewards — [arXiv 2602.19439](https://arxiv.org/abs/2602.19439); [SSRN](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=6288939)
- ORLoopBench (2026): OR-Debug-Bench + OR-Bias-Bench (2,000 newsvendor instances). Solver-verified RLVR let an 8B model beat frontier APIs on LP repair (95.3% vs 92.4% RR@5), using 2.8x fewer tokens per success than o4-mini and +14.6 pp on root-cause identification; curriculum training cut out-of-distribution behavioral bias from 20.0% to 10.4%. Crucially exposes "semantic drift": feasible regenerated MILPs can solve the wrong problem — [arXiv 2601.21008](https://arxiv.org/abs/2601.21008)

Behavioral biases and long-horizon planning
- AIM-Bench (Aug 2025): most LLMs anchor on mean demand (GPT-4o anchoring factor ~1 and 0.925); GPT-4.1 and DeepSeek still biased; only Gemini-2.5-flash-lite was immune on newsvendor; attempts to overcome bullwhip were "disappointing". Mitigations tested: cognitive reflection, information sharing — [arXiv 2508.11416](https://arxiv.org/html/2508.11416v1)
- RetailBench (2026): thousand-day-scale single-store supermarket simulation (pricing, replenishment, supplier selection, assortment, aging inventory, cash flow). "Current agents remain far from stable, high-performing long-horizon operational policies"; performance degrades with complexity; failure modes: incomplete evidence acquisition, surface-level decisions, no consistent long-horizon policy. Proposed fix: separate high-level strategy from low-level execution ("Evolving Strategy & Execution") — [arXiv 2603.16453](https://arxiv.org/abs/2603.16453)
- "LLMs for Supply Chain Management" (May 2025) notes restricted numerical reasoning while highlighting knowledge integration, retrieval, and multi-agent communication strengths — [arXiv 2505.18597](https://arxiv.org/pdf/2505.18597)
- Other 2026 agent evaluations: STOCKTAKE (gap between perception and action, with fair oracle) — [arXiv 2607.13618](https://arxiv.org/html/2607.13618v1); SupChain-Bench (ACL Findings 2026) — [ACL Anthology](https://aclanthology.org/2026.findings-acl.371.pdf); reliability of autonomous agents in SCM — [arXiv 2605.17036](https://arxiv.org/pdf/2605.17036)

Architectures proposed
- Microsoft OptiGuide: LLM as interpretive layer translating natural-language what-if queries into optimization code; claimed production use for cloud supply chain planners — per summary in [arXiv 2509.03811](https://arxiv.org/pdf/2509.03811)
- "Assured autonomy: how operations research powers and orchestrates generative AI systems" (Dec 2025) argues OR should orchestrate/verify GenAI rather than the reverse — [arXiv 2512.23978](https://arxiv.org/pdf/2512.23978)
- Uncertainty-guided multi-agent KG construction (Helicase) attaches calibrated confidence to agent-extracted supply-chain facts, motivated by agents producing "confidently wrong decisions" — [arXiv 2605.26835](https://arxiv.org/pdf/2605.26835)
- Practitioner skepticism: Lokad argues "chatting with your supply chain won't fix it" — [Lokad blog, Dec 2025](https://www.lokad.com/blog/2025/12/24/chatting-with-your-supply-chain-wont-fix-it/) (vendor opinion)

Foundation models for time series
- Chronos-2 (Amazon, Oct 2025): zero-shot univariate, multivariate and covariate-informed forecasting via group attention/in-context learning; on fev-bench 90.7% win rate and 47.3% skill score vs TiRex (80.8%) and TimesFM-2.5 (75.9%); covariates raise skill 40.9% → 47.0% — [Amazon Science blog](https://www.amazon.science/blog/introducing-chronos-2-from-univariate-to-universal-forecasting); [arXiv 2510.15821](https://arxiv.org/abs/2510.15821); fev-bench [arXiv 2509.26468](https://arxiv.org/pdf/2509.26468)
- Limitations reported: TSFMs struggle on high-frequency data where AutoTheta can win; Chronos-2 strongly smooths jagged/intermittent retail patterns (Rossmann case); limited static-variable handling; multivariate still underexplored — [arXiv 2605.17045](https://arxiv.org/pdf/2605.17045); [AI Horizon Forecast](https://aihorizonforecast.substack.com/p/time-series-foundation-models-a-deep-dive-into-strengths-and-limitations) (secondary); target–covariate relationship probes [arXiv 2605.12200](https://arxiv.org/pdf/2605.12200)
- In the intermittent spare-parts decision-aware benchmark, pretrained foundation models were among 38 methods but accuracy rank was negatively correlated with service — [arXiv 2609.13840](https://arxiv.org/abs/2609.13840)

### Inferences
- The strongest evidence-backed fix pattern in 2026 is "small model + solver as verifier + RL with verifiable rewards" (OptiRepair, ORLoopBench), which beat much larger frontier APIs on narrow OR repair tasks; this is specialist training, not general LLM capability.
- Feasibility is not correctness: semantic drift and "feasible but irrational" repairs mean verification must check both solver output and domain theory/intent.
- LLM agents are currently better suited as interfaces/orchestrators to solvers (OptiGuide pattern) than as autonomous long-horizon decision policies (RetailBench, AIM-Bench).
- TSFMs are a plausible default for dense demand but need decision-aware evaluation and intermittent-demand handling before they replace specialist methods for spare parts/long-tail SKUs (speculative extrapolation from cited results).

### Gaps
- No independent, peer-reviewed production audit of OptiGuide's impact was found.
- TimesFM-specific supply chain evaluations (Google) were not found beyond fev-bench comparisons.
- Formal verification (theorem-prover-style) of LLM-generated optimization models for logistics: no concrete deployed work found; current "verification" is solver/IIS/graph-based.

## Q3. Missing benchmarks, datasets and evaluation methods

### Takeaway
The field lacks realistic, dynamic, stochastic, decision-aware benchmarks with genuine industrial data, and many existing benchmarks are noisy (LLM-OR) or leaky (TSFM). Recent efforts (FrontierCO, fev-bench, ORLoopBench, RetailBench, decision-aware intermittent benchmark) begin to fill specific holes, but few public industrial datasets exist beyond Amazon's 2021 last-mile data and Cainiao's LaDe.

### Cited Findings
- Synthetic-generator dependence of ML4CO evaluation — [FrontierCO arXiv 2505.16952](https://arxiv.org/html/2505.16952v1); [MIS position paper arXiv 2502.03669](https://arxiv.org/abs/2502.03669)
- LLM-for-OR benchmarks error rates 8–54% (see Q2) — [arXiv 2508.10047](https://arxiv.org/pdf/2508.10047)
- TSFM evaluation leakage: two types — train-test overlap from multi-purpose dataset reuse, and temporal overlap of correlated series; analysis of 15 TSFMs finds widespread overlap; documented leakage cases yielded >50% better test scores; recommend benchmark designers guarantee genuinely novel test data — [arXiv 2510.13654](https://arxiv.org/html/2510.13654v3)
- Decision-aware forecasting benchmarks are needed because accuracy ≠ service — [arXiv 2609.13840](https://arxiv.org/abs/2609.13840)
- Solver-in-the-loop evaluation instead of one-shot code generation — [ORLoopBench arXiv 2601.21008](https://arxiv.org/abs/2601.21008)
- Long-horizon, data-grounded agent simulations — [RetailBench arXiv 2603.16453](https://arxiv.org/abs/2603.16453); behavioral-bias inventory benchmark — [AIM-Bench arXiv 2508.11416](https://arxiv.org/html/2508.11416v1)
- LLM algorithm-design benchmarks for large-scale optimization: FrontierOR (2026) — [arXiv 2605.25246](https://arxiv.org/pdf/2605.25246); CO-Bench — [arXiv 2504.04310](https://arxiv.org/pdf/2504.04310); ALE-Bench long-horizon algorithm engineering — [arXiv 2506.09050](https://arxiv.org/pdf/2506.09050)
- Industrial datasets that exist: Amazon 2021 Last Mile Routing Research Challenge (with MIT CTL) — 9,184 real routes from 2018 in 5 US metros (6,112 train / 3,072 eval), obfuscated locations; described as the first large public dataset from real operational routing — [AWS Open Data](https://registry.opendata.aws/amazon-last-mile-challenges/); [Transportation Science](https://pubsonline.informs.org/doi/10.1287/trsc.2022.1173). Cainiao LaDe (KDD 2024): 10.677M packages, 21k couriers, 6 months, multiple cities; supports route prediction, ETA, spatio-temporal forecasting — [arXiv 2306.10675](https://arxiv.org/html/2306.10675v2); [ACM DL](https://dl.acm.org/doi/10.1145/3637528.3671548)
- ORTEC real-data dynamic VRPTW from the 2022 competition remains a rare dynamic benchmark — [PMLR v220](https://proceedings.mlr.press/v220/kool23a.html); [quickstart repo](https://github.com/ortec/euro-neurips-vrp-2022-quickstart)
- DFL benchmark suite (7 problems) — [PredOpt](https://github.com/PredOpt/predopt-benchmarks)

### Inferences
- Missing: a public multi-echelon, stochastic, disruption-laden supply chain benchmark with real demand, lead-time and capacity data and decision-quality (cost/service) metrics; the cited benchmarks each cover one slice (routing, forecasting, LLM repair, retail agent sim).
- Missing: benchmarks where test data is provably post-training-cutoff (for TSFMs and LLMs) — the leakage paper argues benchmarks must enforce this.
- Missing: standard evaluation of re-optimization stability (plan nervousness), latency, and human override rates — not found in any benchmark reviewed (inference; no source found).

### Gaps
- No public benchmark found for multi-agent (multi-firm) supply chain coordination with real data.
- Did not find industrial-scale public datasets for warehouse/fulfillment scheduling or freight/linehaul.

## Q4. Which labs are working on these frontiers?

### Takeaway
Activity clusters around a few academic OR/ML groups (MIT, Georgia Tech, KU Leuven, UVA, TUM, KAIST/AI4CO) and industry labs (Amazon Science, Google Research, Microsoft Research, ORTEC, Cainiao/Alibaba).

### Cited Findings
- MIT (David Simchi-Levi, with Ruicheng Ao, Xinshang Wang): LLM agents for supply chain model repair (OptiRepair) — [SSRN](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=6288939)
- Georgia Tech (Pascal Van Hentenryck, with Akhlaghi, Zandehshahvar): PROPEL large-scale supply chain planning — [arXiv 2504.07383](https://arxiv.org/abs/2504.07383)
- KU Leuven (Tias Guns) and University of Virginia (Ferdinando Fioretto) et al.: decision-focused learning survey/benchmark — [arXiv 2307.13565](https://arxiv.org/abs/2307.13565)
- TUM (tumBAIS group): CO-enriched ML for dynamic VRPTW, EURO-NeurIPS 2022 winner — [GitHub](https://github.com/tumBAIS/euro-meets-neurips-2022); [Transportation Science](https://pubsonline.informs.org/doi/abs/10.1287/trsc.2023.0107)
- ORTEC (with competition organizers incl. Wouter Kool): real-data dynamic VRP competition — [PMLR v220](https://proceedings.mlr.press/v220/kool23a.html)
- AI4CO open-source community (RouteFinder; KAIST-affiliated per repo) — [GitHub ai4co/routefinder](https://github.com/ai4co/routefinder)
- Amazon Science: Chronos / Chronos-2 forecasting foundation models; Last Mile Routing Challenge dataset with MIT CTL — [Amazon Science blog](https://www.amazon.science/blog/introducing-chronos-2-from-univariate-to-universal-forecasting); [GitHub chronos-forecasting](https://github.com/amazon-science/chronos-forecasting); [Amazon Science publication](https://www.amazon.science/publications/2021-amazon-last-mile-routing-research-challenge-data-set)
- Google Research: TimesFM-2.5 (benchmarked in fev-bench) — [Amazon Science Chronos-2 comparison](https://www.amazon.science/blog/introducing-chronos-2-from-univariate-to-universal-forecasting)
- Microsoft Research: OptiGuide — as described in [arXiv 2509.03811](https://arxiv.org/pdf/2509.03811)
- Cainiao (Alibaba) with HKUST: LaDe dataset — [HKUST portal](https://researchportal.hkust.edu.hk/en/publications/lade-the-first-comprehensive-last-mile-express-dataset-from-indus/)
- FrontierCO / CO-Bench team (repo owner sunnweiwei) — [GitHub](https://github.com/sunnweiwei/FrontierCO)
- CCC / INFORMS / ACM SIGAI: AI/OR workshop series — [CCC events](https://cra.org/ccc/events/artificial-intelligence-operations-research-workshop-iii/)

### Inferences
- Pascal Van Hentenryck's group is likely linked to the NSF AI Institute for Advances in Optimization (AI4OPT) at Georgia Tech (from background knowledge; not verified with a source in this round).

### Gaps
- Institutional affiliations of FrontierCO, RetailBench, AIM-Bench and ORLoopBench authors were not verified (full text unavailable).
- No information gathered on Chinese industrial labs beyond Cainiao (e.g., JD, Huawei Noah's Ark) or on dedicated European logistics AI programs.
