# ML, RL and LLMs for Logistics and Scheduling Optimization (state as of 2025-2026)

Method note: arxiv.org, deepmind.google and careersatdoordash.com could not be fetched from this environment (proxy blocked them). Many findings below therefore come from search-result abstracts and snippets of primary sources (arXiv abstracts, ICLR/ICML proceedings, INFORMS pages, vendor blogs), not full-text reads. Figures from secondary blogs are marked as such.

## Q1. Neural combinatorial optimization (NCO) vs HGS/LKH/CP-SAT on large, realistic instances

### Takeaway
Neural constructive and diffusion solvers (POMO-family, DIFUSCO, multi-task "foundation" routers such as RouteFinder, MVMoE and GOAL) have improved steadily on synthetic uniform instances of 50 to 1,000 nodes. Independent 2024-2026 evaluations still find that classical solvers (HGS, LKH-3, KaMIS, CP-SAT) win on large, structured, real-world instances, and the gap widens as instances grow. The honest summary: learned solvers are fast and adaptable across variants, but they do not beat the state of the art on solution quality at scale.

### Cited Findings
- **Standard yardstick.** NCO papers report optimality gaps against LKH-3 for TSP and HGS for CVRP. Recent large-scale work evaluates on TSPLIB/CVRPLIB instances with more than 1K nodes (33 TSP and 14 CVRP instances), with POMO, BQ, LEHD, INVIT and SIGD as neural baselines — [Test-Time Projection Learning, arXiv 2506.02392](https://arxiv.org/html/2506.02392); [Learning to Reduce Search Space, arXiv 2503.03137](https://arxiv.org/pdf/2503.03137)
- **FrontierCO (ICLR 2026).** It covers 8 CO problems (routing, scheduling, facility location, graph problems) using real competition and repository instances (DIMACS, TSPLib). It evaluates 16 ML solvers (GNN-based, hybrid neural-symbolic and LLM agents) against state-of-the-art classical solvers. It finds "a persistent performance gap that widens under structurally challenging and large instance sizes (e.g., TSP up to 10M nodes; MIS up to 8M)", while also identifying cases where ML methods win. It concludes that ML methods "still lag behind state-of-the-art human-designed algorithms in terms of efficiency, generalization, and scalability". Neural solvers do well on structured problems, and LLM agents show novel strategy discovery on hard instances — [arXiv 2505.16952](https://arxiv.org/abs/2505.16952); [ICLR 2026 paper](https://proceedings.iclr.cc/paper_files/paper/2026/file/c131c8875c7b1133ffdad2b53cb10e91-Paper-Conference.pdf); [arXiv v3 html](https://arxiv.org/html/2505.16952v3)
- **Position paper "Unrealized Expectations" / "Time to Rethink AI for CO: Classical Algorithms Remain Tough to Match" (2025).** On Maximum Independent Set, leading GPU-based AI methods are consistently beaten by the classical KaMIS solver, even on in-distribution random graphs. Some AI methods often fail to beat a simple degree-based greedy heuristic — [arXiv 2502.03669](https://arxiv.org/pdf/2502.03669); [ADS abstract](https://ui.adsabs.harvard.edu/abs/2025arXiv250203669W/abstract)
- **Heatmap and diffusion solvers (DIFUSCO) critiqued (ICML 2024 Oral position paper).**
  - SoftDist is a parameter-free-style heatmap (a softmax over the distance matrix with one tuned temperature). When fed into the same MCTS search, it matches or approaches learned heatmaps.
  - DIFUSCO heatmap generation takes 3.61 min on TSP-500, 11.86 min on TSP-1000 and 28.51 min on TSP-10000. That is 1.7 to 3.6 times the MCTS search time. SoftDist takes under 0.1 s.
  - Implication: in "heatmap + MCTS" pipelines, most of the quality comes from the search, not the learned model.
  - Sources: [arXiv 2406.03503](https://arxiv.org/pdf/2406.03503); [GitHub rethink_mcts_for_tsp](https://github.com/xyfffff/rethink_mcts_for_tsp); a follow-up component analysis, [Beyond the Heatmap, arXiv 2411.09238](https://arxiv.org/pdf/2411.09238)
- **Rebuttal.** A published comment disputes parts of the position paper, so the debate is not settled — [arXiv 2406.09441](https://arxiv.org/pdf/2406.09441)
- **Distribution dependence.** A 2025 study, "On Distributional Dependent Performance of Classical and Neural Routing Solvers", examines how rankings shift with the instance distribution. Only the abstract listing was accessible — [arXiv 2508.02510](https://arxiv.org/pdf/2508.02510)
- **Routing Arena.** A benchmark suite built specifically to evaluate neural routing solvers fairly against classical ones — [arXiv 2310.04140](https://arxiv.org/pdf/2310.04140)
- **RouteFinder** (AI4CO / KAIST-affiliated code base).
  - It is a "foundation model" framework that treats VRP variants as attribute subsets of one generalized problem.
  - It uses a transformer encoder, global attribute embeddings, mixed-batch training, multi-variant reward normalization, and adapter layers for fine-tuning to unseen attributes.
  - It is evaluated on 48 VRP variants and outperforms *neural* baselines only. The claim is not that it beats HGS or PyVRP.
  - Sources: [arXiv 2406.15007](https://arxiv.org/abs/2406.15007); [code](https://github.com/ai4co/routefinder); [OpenReview](https://openreview.net/pdf?id=hCiaiZ6e4G)
- **MVMoE (ICML 2024).** A unified mixture-of-experts neural solver for 16 VRP variants, including zero-shot use — [GitHub](https://github.com/RoyalSkye/Routing-MVMoE); [arXiv 2405.01029](https://arxiv.org/pdf/2405.01029); [SMU record](https://smusg.elsevierpure.com/en/publications/mvmoe-multi-task-vehicle-routing-solver-with-mixture-of-experts/)
- **MVMoE's own reviews.** A reviewer noted that MoE gives only a "marginal improvement upon a multi-task learning baseline (gap improvement of 0.1-0.5%)" — [ICML24 reviews file in repo](https://github.com/RoyalSkye/Routing-MVMoE/blob/main/assets/Reviews_ICML24.md)
- **GOAL (Generalist CO Agent Learner).** A 2.1M-parameter backbone with lightweight per-task adapters of a few thousand parameters, across multiple CO problems — [arXiv 2406.15079](https://arxiv.org/pdf/2406.15079). Venue: a search snippet said ICLR 2024; I could not verify this, and it may be a later venue.
- **Multi-task VRP work continues in 2026.** Examples include cross-task knowledge sharing at ICLR 2026, Chain-of-Context learning, and vision-assisted foundation models — [ICLR 2026](https://proceedings.iclr.cc/paper_files/paper/2026/file/4572bc2f514e627914cbe60d0398a2d1-Paper-Conference.pdf); [arXiv 2603.01667](https://arxiv.org/pdf/2603.01667); [arXiv 2606.10431](https://arxiv.org/pdf/2606.10431)
- **Survey.** A comprehensive NCO-for-VRP survey with perspectives — [arXiv 2406.00415](https://arxiv.org/html/2406.00415v2)
- **Job-shop scheduling (JSSP), baseline results.** L2D was the first deep RL method for JSSP (a GIN network trained with PPO on the disjunctive graph, learning dispatching rules). It beats priority dispatching rules but underperforms OR-Tools CP-SAT — [JSSP benchmark envs, arXiv 2308.12794](https://arxiv.org/pdf/2308.12794); [GNN-for-JSSP survey, arXiv 2406.14096](https://arxiv.org/pdf/2406.14096)
- **JSSP, SEVAL claim (2025).** On the largest Taillard instances (100x20), SEVAL reports a 0.5% mean gap and claims to beat OR-Tools even with a 1-hour limit per instance. This is a self-reported result that I have not independently verified — [arXiv 2502.08684](https://arxiv.org/pdf/2502.08684)
- **CP solver benchmark.** A dedicated benchmark of constraint programming solvers for makespan JSSP exists (2026) — [Mathematics 14(12):2179](https://doi.org/10.3390/math14122179)

### Inferences
- The field's own critics (the position papers and FrontierCO) now argue that uniform-random benchmarks at 100 nodes overstate progress. The credible value of NCO today is:
  - speed at inference (milliseconds per instance);
  - one model covering many VRP variants;
  - use as a component that learns to prune or guide search inside classical metaheuristics.
- NCO is not a replacement for HGS or LKH on large real instances.
- Multi-task "foundation" routing models (RouteFinder, MVMoE, GOAL) mainly compete with each other. Their headline wins are against neural baselines, not against HGS or PyVRP.

### Gaps
- I could not access full-text tables with exact current gaps on CVRPLIB X-instances or 10K+ node instances (arxiv.org was blocked). Specific percentages for RouteFinder, MVMoE and GOAL versus HGS are not included for this reason.
- I found no independent third-party replication of the SEVAL JSSP claim.

## Q2. ML inside solvers, GPU solvers, and LLM-for-OR (modeling and heuristic discovery)

### Takeaway
The most tangible progress is not neural solvers. It is in three areas:
- **GPU first-order LP** (PDLP; cuOpt), which is now competitive in public LP benchmarks;
- **LLM-driven program search** (FunSearch, AlphaEvolve, EoH, ReEvo), which yields interpretable heuristics, with at least one confirmed large production deployment at Google;
- **LLM "copilots"** that formulate or explain optimization models while leaving the solving to classical solvers.

Learned branching and diving (DeepMind 2020) showed gains on specific homogeneous datasets but have not displaced commercial solver defaults. LLM-modeling benchmarks have serious data-quality problems.

### Cited Findings
**ML inside MIP solvers**
- **DeepMind/Google "Solving MIPs Using Neural Networks" (2020).**
  - Neural Diving produces partial assignments, and SCIP solves the resulting sub-MIPs.
  - Neural Branching imitates a GPU-scalable variant of Full Strong Branching.
  - At large time limits it gave 1.5x, 2x and 104x better primal-dual gaps on 3 of 5 large datasets, reached a 10% gap 5x faster on a 4th, and matched SCIP on the 5th. The datasets included Google production MIPs and MIPLIB.
  - Sources: [arXiv 2012.13349](https://arxiv.org/abs/2012.13349); [DeepMind publication page](https://deepmind.com/research/publications/2021/Solving-Mixed-Integer-Programs-Using-Neural-Networks)

**GPU solvers**
- **PDLP (Google, open source in OR-Tools).**
  - It is a first-order LP solver (PDHG plus preconditioning, presolve, adaptive steps and restarts, feasibility polishing) built on matrix-vector products. It needs no factorization, so it suits GPUs and distributed systems.
  - It was co-awarded the Beale-Orchard-Hays Prize (ISMP, July 2024).
  - GPU ports cuPDLP.jl and cuPDLP-C are competitive with commercial solvers on large LPs.
  - Sources: [Google Research blog](https://research.google/blog/scaling-up-linear-programming-with-pdlp/); [arXiv 2501.07018](https://arxiv.org/abs/2501.07018); [relationships among GPU first-order LP methods, arXiv 2509.23903](https://arxiv.org/html/2509.23903)
- **NVIDIA cuOpt (open-sourced 2025), LP results.**
  - Its GPU barrier method, run concurrently with PDLP, ranked 1st among open-source solvers and 2nd of 11 overall in public LP benchmarks retrieved on 20 Oct 2025 (vendor claim) — [NVIDIA tech blog](https://developer.nvidia.com/blog/solve-linear-programs-using-the-gpu-accelerated-barrier-method-in-nvidia-cuopt/)
  - NVIDIA claims "up to 50x" speedups on real power-grid instances — [NVIDIA blog](https://developer.nvidia.com/blog/accelerate-decision-optimization-using-open-source-nvidia-cuopt/)
- **cuOpt with HiGHS on MIP.** On MIPLIB with a 5-minute limit, HiGHS alone reached a 28% gap and HiGHS plus cuOpt on an H100 reached 21% — [U. Edinburgh blog](https://blogs.ed.ac.uk/mathematics/2025/03/18/highs-and-nvidia-cuopt-driving-open-source-innovation-in-optimization/)
- **cuOpt routing records (vendor claim).** NVIDIA says cuOpt holds 23 world records on the largest routing benchmarks from the last three years, as verified by SINTEF. Note that cuOpt routing is GPU-parallel metaheuristic search, not a learned model — [NVIDIA blog](https://developer.nvidia.com/blog/accelerate-decision-optimization-using-open-source-nvidia-cuopt/)
- **Early third-party user results** are reported by AMPL — [AMPL blog](https://ampl.com/blog/breaking-barriers-in-optimization-ampls-early-results-with-nvidia-cuopt/)

**LLM-driven heuristic discovery**
- **AlphaEvolve (Google DeepMind, May 2025).**
  - It evolved a Borg data-center scheduling heuristic that recovered about 0.7% of Google's fleet-wide compute resources.
  - The discovered function is a short, interpretable expression over CPU and memory residuals.
  - Sources: the primary DeepMind blog could not be fetched; the figure is reported by secondary sources [Medium summary](https://medium.com/@milesk_33/forget-the-benchmarks-heres-what-alphaevolve-truly-changed-3ecf95fdbd76) and [Security Boulevard](https://securityboulevard.com/2025/05/google-deepminds-alphaevolve-a-breakthrough-ai-coding-agent/). Primary: https://deepmind.google/discover/blog/alphaevolve-a-gemini-powered-coding-agent-for-designing-advanced-algorithms/
  - Unverified: a "$1.225B" valuation that appears in blogs is a blogger's extrapolation and should not be cited as fact.
- **EoH.** It evolves "thoughts" (natural-language heuristic ideas) together with code. It beats handcrafted heuristics and FunSearch on online bin packing using about 2,000 LLM queries versus about 1 million for FunSearch — [EoH/related, arXiv 2407.10873](https://arxiv.org/pdf/2407.10873); [EoH-S, arXiv 2508.03082](https://www.arxiv.org/pdf/2508.03082)
- **ReEvo (NeurIPS 2024).** It uses reflective evolution (short- and long-term reflection, crossover, elitist mutation). It reports state-of-the-art or competitive heuristics across 5 algorithm types and 6 CO problems, and it is more sample-efficient than earlier LLM hyper-heuristics — [arXiv 2402.01145](https://arxiv.org/abs/2402.01145)

**LLM-for-OR (modeling)**
- **OptiMUS (Stanford, Udell group).**
  - It is a modular multi-agent LLM system that formulates MILPs from natural language, writes and debugs gurobipy code, and evaluates the results.
  - OptiMUS-0.3 adds an error-correction module and compares against the fine-tuned ORLM and LLMOPT.
  - Sources: [arXiv 2310.06116](https://arxiv.org/pdf/2310.06116); [arXiv 2402.10172 (ICML 2024)](https://arxiv.org/pdf/2402.10172); [OptiMUS-0.3, arXiv 2407.19633](https://arxiv.org/html/2407.19633)
- **ORLM** fine-tunes open LLMs on semi-synthetic data. Its numerical performance is comparable to OptiMUS — [arXiv 2407.19633](https://arxiv.org/html/2407.19633)
- **Microsoft OptiMind (2025)** teaches LLMs expert optimization reasoning — [arXiv 2509.22979](https://arxiv.org/html/2509.22979v2)
- **Microsoft OptiGuide.**
  - It is an LLM front-end that answers what-if questions by calling the optimization solver, so proprietary data is not sent to the LLM.
  - It was deployed as the IFS AI assistant in Microsoft's Cloud Supply Chain, the first production LLM in that organization. INFORMS credits the broader optimization program with halving cycle times and saving "tens to hundreds of millions of dollars" a year. That saving is for the whole program, not the LLM alone.
  - Sources: [MSR project](https://www.microsoft.com/en-us/research/project/optiguide-genai-for-supply-chain-optimization/); [INFORMS success story](https://www.informs.org/Impact/O.R.-Analytics-Success-Stories/Microsoft-Cloud-Supply-Chain); [arXiv 2307.03875](https://arxiv.org/abs/2307.03875); [GitHub](https://github.com/microsoft/OptiGuide)
- **Gurobi.**
  - "Gurobi AI Modeling" (Feb 2025) provides custom GPTs that turn text problems into math and gurobipy code — [Gurobi news](https://www.gurobi.com/news/gurobi-ai-modeling-empowers-users-with-accessible-optimization-resources/); [docs](https://gurobi-ai-modeling.readthedocs.io/en/latest/)
  - "Gurobot" (24 June 2025) is an LLM support agent that can escalate to human experts, part of the "Gurobi Intelligence" framework — [Gurobi news](https://www.gurobi.com/news/gurobi-announces-new-ai-assistant-to-provide-optimization-users-with-instant-support-and-resources/)
- **Benchmark quality problems.**
  - Estimated error rates are at least 26.4% for NL4Opt, at least 8.13% for MAMO-EasyLP and at least 23.7% for MAMO-ComplexLP. IndustryOR reaches about 54%.
  - Xiao et al. (2025) found that 30-50% of the data in six mainstream benchmarks were problematic and released cleaned subsets.
  - SIRL released cleaned NL4Opt, IndustryOR and MAMO versions.
  - Sources: [Survey: Optimization Modeling Meets LLMs, IJCAI 2025](https://www.ijcai.org/proceedings/2025/1192.pdf); [arXiv 2508.10047](https://arxiv.org/pdf/2508.10047); [ORGEval, arXiv 2510.27610](https://arxiv.org/pdf/2510.27610)
- **Newer 2026 work.** This includes execution-verified RL for modeling and production-oriented agents such as ORPilot. There is also a FrontierOR benchmark for LLM algorithm design at scale — [arXiv 2604.00442](https://arxiv.org/pdf/2604.00442); [arXiv 2605.02728](https://arxiv.org/pdf/2605.02728); [arXiv 2605.25246](https://arxiv.org/pdf/2605.25246)

### Inferences
- **"Hype vs evidence" ranking**, from strongest production evidence to weakest:
  1. GPU LP (PDLP/cuOpt): benchmark-verified.
  2. LLM-evolved heuristics: AlphaEvolve at Google scale.
  3. LLM copilots over classical solvers: OptiGuide.
  4. Learned MIP components: dataset-specific gains.
  5. Pure LLM auto-formulation: accuracy claims are unreliable because the benchmarks are noisy.
- Heuristics found by LLM program search are attractive because they are short and auditable, unlike neural policies. This lowers the barrier to deployment.

### Gaps
- I did not find evidence that learned branching has been adopted into Gurobi, CPLEX or Xpress default settings; this could not be verified either way.
- I have no cuOpt MIP results from Mittelmann's benchmarks beyond the HiGHS figure.
- I could not fetch the Chain-of-Experts (ICLR 2024) accuracy numbers or current state-of-the-art accuracy on cleaned benchmarks.

## Q3. Leading labs and companies

### Takeaway
Leadership is split three ways:
- Big tech with production OR problems: Google (OR-Tools, PDLP, AlphaEvolve), Microsoft (OptiGuide, OptiMind), NVIDIA (cuOpt) and Amazon (supply-chain RL).
- Chinese platforms with Edelman/Wagner-level deployments: Alibaba/Cainiao, JD and Meituan.
- An academic NCO community centred on KAIST/AI4CO (RL4CO, RouteFinder), NTU/SMU/A*STAR (MVMoE), Stanford (OptiMUS) and SJTU (awesome-ml4co).

### Cited Findings
- **Google Research/DeepMind:** PDLP in OR-Tools, Neural Diving/Branching, AlphaEvolve — see the Q2 sources ([Google Research blog](https://research.google/blog/scaling-up-linear-programming-with-pdlp/); [arXiv 2012.13349](https://arxiv.org/abs/2012.13349))
- **Microsoft Research:** OptiGuide and OptiMind — [MSR](https://www.microsoft.com/en-us/research/project/optiguide-genai-for-supply-chain-optimization/); [arXiv 2509.22979](https://arxiv.org/html/2509.22979v2)
- **NVIDIA:** cuOpt, open-sourced 2025 — [NVIDIA blog](https://developer.nvidia.com/blog/accelerate-decision-optimization-using-open-source-nvidia-cuopt/)
- **Amazon (supply chain optimization):** DirectBackprop / Deep Inventory Management — [arXiv 2210.03137](https://arxiv.org/pdf/2210.03137); [Amazon Science](https://www.amazon.science/publications/deep-inventory-management-with-supply-and-capacity-risk-awareness)
- **Alibaba/Cainiao:**
  - 2021 Franz Edelman Award — [INFORMS](https://www.informs.org/Recognizing-Excellence/INFORMS-Prizes/Franz-Edelman-Award/Franz-Edelman-Laureates2/Franz-Edelman-Laureates-Class-of-2021)
  - GreedRL open neural solver — [Hugging Face](https://huggingface.co/Cainiao-AI/GreedRL)
- **JD.com:** 2024 INFORMS Wagner Prize for e-commerce optimization, according to a search summary; this was not verified on the primary INFORMS page — [INFORMS](https://www.informs.org/Impact/O.R.-Analytics-Success-Stories/Smart-Parcel-Consolidation-at-Cainiao)
- **KAIST/AI4CO:** RouteFinder code is hosted by the AI4CO organization (which also maintains RL4CO) — [GitHub ai4co/routefinder](https://github.com/ai4co/routefinder)
- **SMU/NTU:** MVMoE (Zhou et al.) — [SMU record](https://smusg.elsevierpure.com/en/publications/mvmoe-multi-task-vehicle-routing-solver-with-mixture-of-experts/)
- **Stanford (Udell):** OptiMUS — [arXiv 2402.10172](https://arxiv.org/pdf/2402.10172)
- **SJTU Thinklab:** maintains the community awesome-ml4co paper list — [GitHub](https://github.com/Thinklab-SJTU/awesome-ml4co)
- **Meituan:** RL plus hyper-heuristic dispatch research, published in Transportation Science 2025 — [INFORMS TS](https://pubsonline.informs.org/doi/abs/10.1287/trsc.2025.0129?af=R)

### Inferences
- In routing and scheduling practice, "leading" is better measured by deployed OR plus ML systems (Cainiao, Meituan, Amazon, Google) than by NCO leaderboard papers.
- NVIDIA is the leader in GPU solver infrastructure. Google is the leader in open-source first-order LP and in LLM algorithm discovery.

### Gaps
- I did not independently verify current (2025-26) outputs for Huawei Noah's Ark, InstaDeep, TU Delft, Georgia Tech, MIT (Cathy Wu; Bertsimas) or A*STAR. I found no sources in this pass, so they are not claimed here.

## Q4. Real deployments and reported results

### Takeaway
Credible production evidence exists mostly for three kinds of system:
- RL or learned policies for *dynamic dispatch and inventory* (Amazon, Meituan, DoorDash, DiDi);
- hybrid ALNS-plus-neural routing (Alibaba/Cainiao);
- LLM-discovered heuristics and LLM copilots (Google Borg, Microsoft Azure supply chain).

I found no public evidence of a pure neural VRP solver replacing classical routing solvers at scale.

### Cited Findings
- **Amazon.**
  - A DirectBackprop deep RL buying policy cut inventory by about 12% with no revenue loss in production tests against the production system — [arXiv 2210.03137](https://arxiv.org/pdf/2210.03137)
  - Deep RL policies improved profitability over base-stock policies in large real-world A/B tests — [arXiv 2306.11246](https://arxiv.org/html/2306.11246)
- **Alibaba (VRP).** Open-architecture ALNS plus a deep-learning model trained offline for near-instant online solutions. Implemented across subsidiaries, with more than $50M in annual savings — [INFORMS J. Applied Analytics](https://pubsonline.informs.org/doi/10.1287/inte.2021.1108)
- **Cainiao (parcel consolidation, since 2022).** Covers more than 50 countries and regions, saves tens of millions of dollars a year and cuts delivery time by at least 50% — [INFORMS success story](https://www.informs.org/Impact/O.R.-Analytics-Success-Stories/Smart-Parcel-Consolidation-at-Cainiao)
- **Meituan (more than 60M orders a day).** Uses n-step SARSA with value-function approximation plus a bandit hyper-heuristic over 7 low-level heuristics. It reports a 12% cost reduction via strategic order postponement, and notes that 10% more couriers helps more than the algorithm does. This is evaluated in a simulator built from real data, not reported as a live A/B result — [Optimization Online](https://optimization-online.org/2025/10/data-driven-optimization-for-meal-delivery-a-reinforcement-learning-approach-for-order-courier-assignment-and-routing-at-meituan/); [INFORMS TS](https://pubsonline.informs.org/doi/abs/10.1287/trsc.2025.0129?af=R)
- **DoorDash.** It publishes on RL for on-demand logistics and uses switchback experiments, randomized by region and time, to test dispatch changes. A 2026 multi-agent RL paper reports online experiments in which a learned objective-weight policy increased batching, reduced courier time costs and improved dinner-hour lateness. The DoorDash link to that paper is inferred from the search snippet and not confirmed — [DoorDash blog](https://careersatdoordash.com/blog/reinforcement-learning-for-on-demand-logistics/); [switchback blog](https://careersatdoordash.com/blog/switchback-tests-and-randomized-experimentation-under-network-effects-at-doordash/); [arXiv 2606.13604](https://arxiv.org/pdf/2606.13604)
- **DiDi (ride-hailing).** An RL dispatching algorithm was deployed with A/B tests in five cities and improved total driver income — [arXiv 2202.05118](https://arxiv.org/pdf/2202.05118)
- **Google.** AlphaEvolve's Borg heuristic recovered about 0.7% of fleet compute (see Q2).
- **Microsoft.** OptiGuide runs in production in the Azure cloud supply chain (see Q2).

### Inferences
- The consistent pattern: learned components score, value or parameterize decisions, and classical optimization or heuristics still perform the combinatorial search.
- The reported dollar figures are mostly for OR programs as a whole, not for the ML component in isolation.

### Gaps
- I found no primary quantitative Uber or JD figures for learned dispatch or routing in this pass.
- I could not access DoorDash's blog text (blocked), so its specific metrics are not quoted.
