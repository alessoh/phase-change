# Classical OR Algorithms and Solvers for Logistics and Scheduling (state as of 2025-2026)

Method note: This session's WebFetch was blocked by the egress proxy for most primary domains (arxiv.org, dimacs.rutgers.edu, proceedings.mlr.press, neurips.cc, hexaly.com, pyvrp.readthedocs.io, galgos.inf.puc-rio.br, minizinc.org, scheduleopt.github.io). Findings below come from WebSearch result snippets pointing to those primary pages; a few items marked "[background]" are well-established facts from prior knowledge with the canonical URL given but not re-verified in this session. The report writer should treat snippet-derived numbers as reliable-but-unverified-in-full-text.

## Q1. Best-performing methods on standard benchmarks (CVRP, VRPTW, PDP, dynamic VRP, JSSP/FJSP, crew scheduling)

### Takeaway
For vehicle routing, the heuristic state of the art is dominated by Hybrid Genetic Search (HGS, Vidal) and its descendants (HGS-CVRP, PyVRP, ORTEC's HGS-based solver that won DIMACS VRPTW and EURO-NeurIPS 2022), while on very large CVRP (1,000-10,000 customers) the adaptive iterated local search AILS-II now leads (93/100 initial BKS on the new XL set, 2025-2026). Exact VRP is led by branch-cut-and-price (VRPSolver), which reliably proves optimality up to ~100-200 customers. For job-shop scheduling, CP solvers (OR-Tools CP-SAT, IBM CP Optimizer, OptalCP, Hexaly) are the reference tools and are still closing/improving Taillard instances in 2025-2026; crew scheduling remains dominated by column generation (GENCOL lineage) increasingly augmented with ML.

### Cited Findings

**CVRP / VRPTW heuristics: HGS family and PyVRP**
- PyVRP is a Python package providing a high-performance implementation of hybrid genetic search, built on HGS-CVRP but redesigned as a customizable package while keeping state-of-the-art performance; it supports CVRP, VRPTW and prize-collecting variants, with documented strong performance on CVRP and VRPTW instances up to 1,000 customers — [PyVRP paper, arXiv 2403.13795](https://arxiv.org/pdf/2403.13795)
- Citation: Wouda, N.A., L. Lan, W. Kool (2024), "PyVRP: a high-performance VRP solver package", INFORMS Journal on Computing 36(4): 943-955 — [arXiv 2403.13795](https://arxiv.org/pdf/2403.13795); [ResearchGate](https://www.researchgate.net/publication/377789326_PyVRP_A_High-Performance_VRP_Solver_Package)
- PyVRP maintains a public benchmarks page (gaps vs BKS per variant) — [PyVRP benchmarks docs](https://pyvrp.readthedocs.io/en/latest/setup/benchmarks.html) (page could not be fetched; numbers not extracted)
- 2026 work continues to build on PyVRP, e.g. "PyVRP+: LLM-Driven Metacognitive Heuristic Evolution" — [arXiv 2604.07872](https://arxiv.org/pdf/2604.07872v1)
- [background] HGS-CVRP (Vidal 2022, Computers & Operations Research) is the open-source reference implementation of HGS with SWAP* neighborhood, widely considered the leading CVRP metaheuristic on the Uchoa X set (100-1,000 customers) — [HGS-CVRP, arXiv 2012.10384](https://arxiv.org/abs/2012.10384)

**Large-scale CVRP: new XL set and CVRPLib BKS Challenge (2025-2026)**
- A new XL set of 100 CVRP instances with 1,000 to 10,000 customers was built following the same design principles as the Uchoa X set — [The XL Instances and the CVRPLib BKS Challenge, arXiv 2601.11467](https://arxiv.org/html/2601.11467v2); [technical report](https://galgos.inf.puc-rio.br/cvrplib/uploads/files/report.pdf)
- Initial BKS on XL: AILS-II found 93/100, FILO2 6, FILO 1; mean gap to initial BKS: AILS-II 0.07%, FILO2 0.21%, FILO 0.25%, KGLS 1.00% — [arXiv 2601.11467 (via search snippet)](https://arxiv.org/html/2601.11467v2)
- HGS-CVRP, often considered the premier solver at standard scales, degraded on the largest instances — [arXiv 2601.11467](https://arxiv.org/html/2601.11467v2)
- A 30-day CVRPLib BKS Challenge on the XL set yielded 1,932 BKS improvements from participating teams; scoring was "lead-time" based (points for how long a solution stays BKS) — [arXiv 2601.11467](https://arxiv.org/html/2601.11467v2); [CVRPLib BKS Challenge overview](https://galgos.inf.puc-rio.br/cvrplib/index.php/en/bks_challenge/overview); [INFORMS Open Forum announcement](https://connect.informs.org/discussion/cvrplib-best-known-solution-bks-challenge)
- AILS-II (Máximo, Cordeau, Nascimento) is an adaptive iterated local search designed for large-scale CVRP — [arXiv 2205.12082](https://arxiv.org/pdf/2205.12082)
- 2026: AILS-AHD (AILS with LLM-driven automatic heuristic design of ruin operators) reports new BKS for 8 of 10 instances in a CVRPLib large-scale benchmark — [arXiv 2602.23092](https://arxiv.org/pdf/2602.23092)
- Asynchronous cooperative (parallel) optimization for CVRP is another 2025 line — [arXiv 2511.19445](https://www.arxiv.org/pdf/2511.19445)

**DIMACS 12th Implementation Challenge (VRP), 2021-2022 [historical but still the most recent broad VRP challenge]**
- Launched fall 2021, concluded April 2022 with an online workshop; two phases: qualification on competitors' machines, then a final phase run by organizers on identical DIMACS machines on unseen instances; 8 of 16 CVRP solvers and 7 of 12 VRPTW solvers reached the final — [DIMACS news](https://dimacs.rutgers.edu/news/news-details/the-12th-dimacs-implementation-challenge-vrp)
- CVRP track won (tightly) by team Alkaid-X from Huawei Cloud; VRPTW track won "convincingly" by team "Wouter & Co" from ORTEC — [DIMACS news](https://dimacs.rutgers.edu/news/news-details/the-12th-dimacs-implementation-challenge-vrp); results page: [DIMACS VRP results](http://dimacs.rutgers.edu/programs/challenge/vrp/results/)

**VRPTW / PDP: NVIDIA cuOpt claims (vendor)**
- NVIDIA claims 23 "world records" for cuOpt: 15 on Gehring & Homberger VRPTW and 8 on Li & Lim pickup-and-delivery, "verified by" SINTEF (the "Norway Institute of Science") — [NVIDIA blog](https://blogs.nvidia.com/blog/cuopt-route-optimization-metropolis-omniverse/); [cuOpt product page](https://www.nvidia.com/en-us/ai-data-science/products/cuopt/)
- Same vendor sources state a 2.98% average gap on Gehring & Homberger (earlier tests: 2.96% gap vs best-known number of vehicles) — [NVIDIA cuOpt page](https://www.nvidia.com/en-us/ai-data-science/products/cuopt/); [NVIDIA supply chain blog](https://blogs.nvidia.com/blog/cuopt-ai-software-supply-chain/)
- Note: the "records" and a ~3% average gap are not contradictory only if records are on specific instances/objectives; vendor claims, not peer-reviewed.

**Exact VRP: branch-cut-and-price**
- VRPSolver (Pessoa, Sadykov, Uchoa, Vanderbeck, "A generic exact solver for vehicle routing and related problems", Mathematical Programming, 2020) is a generic BCP code; it outperforms prior exact methods across many variants and solved several instances to optimality for the first time — [Springer](https://link.springer.com/article/10.1007/s10107-020-01523-z); [Uchoa SPOC slides](https://www.lamsade.dauphine.fr/poc/sites/default/files/Uchoa-SPOC20.pdf)
- Guidance from its Python wrapper: expect proven optima for most instances up to 100 customers and a significant share of 101-200 customer instances — [VRPSolverEasy GitHub](https://github.com/inria-UFF/VRPSolverEasy); [PyPI](https://pypi.org/project/VRPSolverEasy/)
- Gap: exact count of Uchoa X instances now solved to optimality not found.

**Job-shop and flexible job-shop scheduling**
- A 2025 study (arXiv 2504.16106, "Updating Lower and Upper Bounds for the JSSP Test Instances") used OR-Tools CP-SAT to close Taillard ta33 (LB raised 1790 -> 1791), prove optimality of ta45 and car5, and improve UBs on ta26, ta45 and others, plus FJSP bounds — [Pith summary of arXiv 2504.16106](https://pith.science/paper/2504.16106)
- ScheduleOpt maintains JSPLib/FJSPLib BKS tables; an OptalCP result of 809,549 on the huge Taillard-style instance tai_1000_1000_4 is dated 2026-07-10 — [JSPLib](https://scheduleopt.github.io/benchmarks/jsplib/); [GitHub issue #5](https://github.com/ScheduleOpt/benchmarks/issues/5); [FJSPLib](https://scheduleopt.github.io/benchmarks/fjsplib/)
- June 2026 comparative study benchmarked IBM CP Optimizer, OR-Tools CP-SAT, Hexaly and OptalCP on 332 JSSP instances from nine benchmark families including 80 Taillard instances — [Mathematics 14(12):2179](https://doi.org/10.3390/math14122179)
- New CP solver Tempo (Hebrard, CP 2025) targets disjunctive scheduling; it struggled on the 10 largest Taillard instances (100 jobs x 20 machines) — [LIPIcs CP 2025](https://drops.dagstuhl.de/storage/00lipics/lipics-vol340-cp2025/LIPIcs.CP.2025.13/LIPIcs.CP.2025.13.pdf)
- PyJobShop (2025) offers a Python modeling layer on CP solvers (OR-Tools CP-SAT, CP Optimizer) for scheduling — [arXiv 2502.13483](https://arxiv.org/pdf/2502.13483)
- Hexaly publishes a very-large-scale JSSP benchmark against CP Optimizer, OR-Tools, Gurobi, CPLEX — [Hexaly JSSP benchmark](https://www.hexaly.com/benchmarks/hexaly-vs-cp-optimizer-vs-or-tools-on-the-job-shop-scheduling-problem-jssp) (vendor; contents not fetched)
- OR-Tools CP-SAT "has been consistently winning gold medals" in the MiniZinc Challenge; 2025 results announced at CP 2025 (Glasgow, Aug 10-15 2025) — [MiniZinc 2025 results](https://www.minizinc.org/challenge/2025/results/); [OR-Tools Wikipedia](https://en.wikipedia.org/wiki/OR-Tools); [CP-SAT 2024 entry description](https://www.minizinc.org/challenge/2024/description_or-tools_cp-sat.txt)

**Crew and fleet scheduling**
- GENCOL (GERAD/Montréal lineage) is a commercial column-generation toolbox for real-world airline crew scheduling — [Transportation Science, Improved Integral Column Generation with ML for Aircrew Pairing](https://pubsonline.informs.org/doi/10.1287/trsc.2021.1084)
- Integral column generation with DNN-predicted flight-connection probabilities outperforms state-of-the-art integral CG and branch-and-price heuristics used in commercial planning software, in both cost and time — [Transportation Science 2021/22](https://pubsonline.informs.org/doi/10.1287/trsc.2021.1084)
- Commercial GENCOL-DCA with ML-built initial clusters handles monthly crew pairing problems up to 50,000 flights; NN connection prediction reached 99.7% accuracy (82.5% on harder instances) — [ScienceDirect, EURO J. Transp. Logist.](https://www.sciencedirect.com/science/article/pii/S2192437620300236); [Semantic Scholar](https://www.semanticscholar.org/paper/f8865cd38bb8a35f9c9cd965753518f609b2a8ff)
- 2025: ML-accelerated windowing for crew rostering — [arXiv 2503.00160](https://arxiv.org/pdf/2503.00160); GNN-based partial column generation for team formation and routing — [arXiv 2509.15275](https://arxiv.org/pdf/2509.15275)

**Network design**
- Benders decomposition variants remain the leading exact approach for (service) network design: Partial Benders (Crainic, Hewitt, Maggioni, Rei, Transportation Science 2021), Meta Partial Benders for logistics SND, branch-and-Benders-cut for stochastic SND with crowdsourced capacity (Transportation Science, 2024/25) — [Meta Partial Benders, arXiv 2009.14628](https://arxiv.org/pdf/2009.14628); [Transportation Science 10.1287/trsc.2024.0752](https://doi.org/10.1287/trsc.2024.0752)
- 2026: adaptive subproblem selection in Benders for survivable network design — [arXiv 2604.09031](https://arxiv.org/pdf/2604.09031)
- Reference text: Crainic, Gendreau, Gendron (eds.), "Network Design with Applications to Transportation and Logistics" (Springer, 2021) — [cited in arXiv 2303.14217](https://arxiv.org/pdf/2303.14217)

### Inferences
- HGS remains the default "best general-purpose" VRP metaheuristic up to ~1,000 customers, but the 2025-2026 XL results show scale-specific methods (AILS-II, FILO/FILO2 — granular neighborhoods, sparsification) win beyond that; the field is moving to 10k-customer benchmarks.
- LLM-driven heuristic design (AILS-AHD, PyVRP+) is emerging as a way to improve classical metaheuristics rather than replace them.
- For JSSP, benchmark progress is now driven by general CP solvers with LNS and massive compute rather than bespoke tabu search, though bespoke tabu search (e.g., Nowicki-Smutnicki lineage) remains historically important.

### Gaps
- Could not fetch exact current PyVRP gaps on X / Gehring-Homberger, or LKH-3 current performance numbers (LKH-3 is known as a strong generic VRP heuristic by Helsgaun; not re-verified this session).
- No reliable independent data found comparing OR-Tools Routing gaps on X instances apart from Hexaly's vendor claim below.
- Number of open Taillard instances as of 2026 not obtained (JSPLib page blocked).
- Dynamic/stochastic VRP beyond EURO-NeurIPS 2022 (see Q3) not covered in depth.

## Q2. Where commercial and open-source MIP/CP/routing solvers stand; GPU LP (cuOpt, PDLP)

### Takeaway
Gurobi 13.0 (Nov 2025) claims ~16% faster on hard MIPs and added a PDHG (PDLP-style) LP method with GPU support; Gurobi (Aug 2024) and MindOpt (Dec 2024) withdrew from Mittelmann's public benchmarks, so independent head-to-head MIP data now mostly covers CPLEX, Xpress, COPT, HiGHS, SCIP etc. GPU first-order LP (PDLP/PDHG) has gone mainstream: NVIDIA open-sourced cuOpt (Apache 2.0, June 2025) with GPU PDLP, barrier, MIP heuristics and routing, and commercial vendors (Gurobi, FICO, COPT) incorporated PDLP variants.

### Cited Findings
**Gurobi**
- Gurobi 13.0 released Nov 18, 2025: ~16% faster on difficult MIP models, >2x faster on non-trivial MINLPs, new PDHG implementation for large LPs with CPU and GPU support, and a nonlinear barrier for local NLP solutions — [Gurobi newsroom](https://www.gurobi.com/company/newsroom/gurobi-releases-version-13-0-with-improved-performance-and-new-solving-capabilities); [BusinessWire](https://www.businesswire.com/news/home/20251118616985/en/Gurobi-Releases-Version-13.0-with-Improved-Performance-and-New-Solving-Capabilities)
- Gurobi 12.0 (Nov 2024): significant speed-ups for MIP, MIQP, nonconvex MIQCP; new nonlinear (MINLP) capabilities — [Gurobi 12.0 news](https://www.gurobi.com/news/gurobi-12-0-brings-new-performance-improvements-innovative-nonlinear-capabilities-and-smarter-resource-management/)

**Mittelmann benchmarks**
- Gurobi withdrew from Hans Mittelmann's benchmarks in August 2024 (results removed); MindOpt withdrew Dec 24, 2024 — [Ceris blog MIP comparison 2025](https://ceris.fyi/blog/mip-solver-comparison-2025/); [Mittelmann benchmarks](https://plato.asu.edu/bench.html)
- Mittelmann's 2026 talk "Benchmarking Optimization Software: a (Hi)Story" focuses MILP results on CPLEX, Gurobi, Xpress as the "big three" — [plato.asu.edu talk (Hong Kong 2026)](https://plato.asu.edu/talks/hongkong26.pdf)
- FICO Xpress 9.6 released April 2025 — [Ceris blog](https://ceris.fyi/blog/mip-solver-comparison-2025/)
- HiGHS is the leading open-source LP/MIP solver on Mittelmann-style tests — [Ceris blog](https://ceris.fyi/blog/mip-solver-comparison-2025/)
- FICO cautions against misuse of MIPLIB-based benchmark claims — [FICO blog](https://www.fico.com/blogs/mip-benchmarking-don-t-abuse-standards)
- 2026 survey of math programming solvers (theory, development, recent advances) — [Frontiers of Engineering Management](https://journal.hep.com.cn/fem/EN/10.1007/s42524-026-5153-z)

**GPU LP: PDLP / cuOpt**
- PDLP originated with Applegate et al. (2021) "Practical Large-Scale Linear Programming using Primal-Dual Hybrid Gradient" (Google), prompting many solvers to add GPU PDLP — [NVIDIA cuOpt GitHub / summaries](https://github.com/NVIDIA/cuopt); [overview of GPU first-order LP methods, arXiv 2506.02174](https://arxiv.org/pdf/2506.02174)
- cuPDLP-C (C implementation of GPU PDLP, Lu & Yang et al.) — [arXiv 2312.14832](https://arxiv.org/pdf/2312.14832)
- NVIDIA open-sourced cuOpt on June 11, 2025 under Apache 2.0, covering LP, MIP and VRP; C++ core with C, Python and server APIs; also added to COIN-OR — [NVIDIA/cuopt GitHub](https://github.com/NVIDIA/cuopt); [blockchain.news summary](https://blockchain.news/news/nvidia-open-source-cuopt-decision-optimization); [GAMS blog, Sept 2025](https://www.gams.com/blog/2025/09/gpu-accelerated-optimization-with-gams-and-nvidia-cuopt/)
- NVIDIA benchmark claims: on MIPLIB-derived LPs where both converged, cuOpt PDLP faster than a state-of-the-art CPU LP solver on 60% of instances, >10x faster on 20%, max 5000x speedup on a large multi-commodity flow problem — [GAMS blog](https://www.gams.com/blog/2025/09/gpu-accelerated-optimization-with-gams-and-nvidia-cuopt/)
- cuOpt GPU barrier: >8x average speedup vs a leading open-source CPU solver and >2x vs a popular commercial CPU solver on a public large-LP test set (vendor) — [NVIDIA developer blog](https://developer.nvidia.com/blog/solve-linear-programs-using-the-gpu-accelerated-barrier-method-in-nvidia-cuopt/)
- FICO reports up to 100x speedups on individual instances vs traditional LP solvers and 25x vs in-house CPU PDLP — [FICO blog on cuOpt](https://www.fico.com/blogs/gpu-powered-optimization-nvidia-cuopt)
- Mittelmann ran GPU LP benchmarks (6 very large LPs on two GPUs) on June 22, 2025 — [Akamai cuOpt blog](https://www.akamai.com/blog/cloud/how-to-get-started-nvidia-cuopt)
- cuOpt MIP relies mainly on GPU-accelerated primal heuristics; cuOpt 26.04 (April 2026) added MIP heuristic tuning options — [NVIDIA/cuopt GitHub](https://github.com/NVIDIA/cuopt)
- PSR (energy) reports on GPU algorithms for large-scale optimization — [PSR report](https://www.psr-inc.com/en/analytics-report/post/gpu-based-algorithms-for-large-scale-optimization/)

**Hexaly (formerly LocalSolver)**
- Hexaly 14.0 claims average optimality gap below [snippet: "6%"—possibly misread; verify] after 10 minutes on all CVRPLIB X instances up to 1,000 clients, and says OR-Tools gives >5% gaps even on medium instances (vendor benchmark) — [Hexaly CVRP benchmark](https://www.hexaly.com/benchmarks/hexaly-vs-gurobi-capacitated-vehicle-routing-problem-cvrp); [Hexaly 14.0 announcement](https://www.hexaly.com/announcements/hexaly-14-0); [Hexaly 14.5](https://www.hexaly.com/announcements/hexaly-optimizer-14-5)
- Hexaly also publishes CVRPTW, large-scale CVRP and RCPSP vendor benchmarks — [CVRPTW](https://www.hexaly.com/benchmarks/hexaly-gurobi-or-tools-capacitated-vehicle-routing-problem-with-time-windows-cvrptw); [large-scale CVRP](https://www.hexaly.com/benchmarks/large-scale-instances-capacitated-vehicle-routing-problem-cvrp); [RCPSP](https://www.hexaly.com/benchmarks/hexaly-vs-or-tools-on-the-resource-constrained-project-scheduling-problem-rcpsp)
- 2026 experimental report on Hexaly thread scaling on the TDVRPTW — [arXiv 2608.10079](https://arxiv.org/pdf/2608.10079)

**Timefold (OptaPlanner successor)**
- Timefold founded early 2023 by Geoffrey De Smet and Maarten Vandenbroucke, forking OptaPlanner; claims faster/lighter than OptaPlanner — [Timefold fork announcement](https://timefold.ai/blog/timefold-announces-optaplanner-fork); [Wikipedia](https://en.wikipedia.org/wiki/Timefold)
- Offers a Field Service Routing API with 50+ prebuilt constraints; Python solver released; Nov 2025 Java 25 performance benchmark — [Timefold Java 25 blog](https://timefold.ai/blog/how-fast-is-java-25); [Python solver blog](https://timefold.ai/blog/new-open-source-solver-python); [GitHub](https://github.com/TimefoldAI/timefold-solver)

### Inferences
- Because Gurobi left Mittelmann's benchmarks, independent 2025-2026 comparisons across all top MIP solvers are incomplete; vendor claims (Gurobi +16%, cuOpt speedups) are largely self-reported.
- GPU PDLP is strongest on very large, sparse LPs (e.g., multi-commodity flow, LP relaxations in network design) and at moderate accuracy; barrier/simplex on CPU remain preferred for high-accuracy or small/medium LPs and for MIP node solves.
- In routing practice, specialized metaheuristics (HGS/PyVRP, AILS, LKH-3) beat general MIP solvers by large margins; Hexaly and OR-Tools are general-purpose tools with larger gaps.

### Gaps
- No verified 2025-2026 numbers for CPLEX, COPT (8.x), Xpress or HiGHS relative speed; no numbers for COPT's GPU PDLP (cuPDLP integration) found in this session.
- Hexaly's exact CVRP gap figure needs verification (snippet said "below 6%"; could be misparsed).

## Q3. What recent competitions revealed about winning methods

### Takeaway
Across DIMACS 2021-22, EURO Meets NeurIPS 2022, and the Amazon 2021 challenge, classical OR metaheuristics (HGS, LKH-3-based local search) won or formed the core of winning entries; ML contributed mainly via hybridization (e.g., learned dispatching policies wrapped around HGS in dynamic VRP, learned driver preferences as penalties).

### Cited Findings
- EURO Meets NeurIPS 2022 Vehicle Routing Competition (organized by Kool, Bliek, Numeroso, Zhang, Catshoek, Tierney, Vidal, Gromicho with ORTEC data) had a static VRPTW track and a dynamic track where orders arrive during the day; >50 teams over a 13-week submission period — [PMLR v220 Kool et al. 2023](https://proceedings.mlr.press/v220/kool23a.html); [ORTEC announcement](https://ortec.com/en/news/ortec-initiates-routing-competition); [quickstart repo (HGS baseline)](https://github.com/ortec/euro-neurips-vrp-2022-quickstart)
- The ORTEC-affiliated team "WouterCo" (Kool, Visser et al.) is reported as winning the static VRPTW track (per search summary of competition sources) — [PMLR v220](https://proceedings.mlr.press/v220/kool23a.html); [NeurIPS slides](https://neurips.cc/media/neurips-2022/Slides/50085.pdf). Note: this conflicts with organizer involvement; verify in the PMLR paper (the snippet may conflate DIMACS team "Wouter & Co" with the EURO-NeurIPS ranking).
- A top dynamic-track approach combined combinatorial optimization with ML ("Combinatorial Optimization enriched Machine Learning", Baty et al.), using HGS inside a learned policy — [arXiv 2304.00789](https://arxiv.org/pdf/2304.00789)
- DIMACS 2021-22: Huawei Cloud's Alkaid-X won CVRP; ORTEC's "Wouter & Co" won VRPTW convincingly — [DIMACS news](https://dimacs.rutgers.edu/news/news-details/the-12th-dimacs-implementation-challenge-vrp)
- Amazon Last Mile Routing Research Challenge 2021 (Amazon + MIT CTL): 45 final submissions; winners announced July 30, 2021; $100k top prize to William Cook (Waterloo), Stephan Held (Bonn) and Keld Helsgaun (Roskilde) — [MIT CTL](https://ctl.mit.edu/news/amazon-and-ctl-announce-winners-last-mile-routing-research-challenge); [routingchallenge.mit.edu winners](https://routingchallenge.mit.edu/last-mile-routing-research-challenge-2021-winners)
- The winning method was penalty-based local search extending LKH (LKH-3 approach) to handle constraints inferred from historical driver routes — [Amazon Science](https://www.amazon.science/academic-engagements/winning-last-mile-challenge-team-addresses-problem-of-combining-mathematical-routes-with-driver-knowledge); [arXiv 2205.04001](https://arxiv.org/pdf/2205.04001); dataset paper — [Transportation Science](https://pubsonline.informs.org/doi/10.1287/trsc.2022.1173)
- ROADEF/EURO 2022 (Renault 3D truck loading): Hexaly team won the first stage (19/30 BKS using Hexaly); final phase July 2023 with 51 registered teams; a two-phase matheuristic won the scientific prize — [ROADEF 2022](https://roadef.org/challenge/2022/en/); [Hexaly event page](https://www.hexaly.com/event/renault-truck-loading-problem); [EJOR 2024 matheuristic](https://www.sciencedirect.com/science/article/pii/S0377221724007951); [Erasmus news](https://www.eur.nl/en/news/bart-van-rossum-and-rick-willemsen-best-junior-team-2022-roadefeuro-challenge)
- ROADEF/EURO 2020 (RTE maintenance planning): MIP and heuristic approaches documented — [arXiv 2111.01047](https://arxiv.org/pdf/2111.01047); [J. Heuristics 2023](https://ideas.repec.org/a/spr/joheur/v29y2023i1d10.1007_s10732-022-09508-1.html)
- ROADEF/EURO 2026 challenge "Keep the Flow!" is under way — [ROADEF challenge page](https://roadef.org/challenge/)
- CVRPLib BKS Challenge (30 days, XL set, 1,932 improvements) — see Q1 — [arXiv 2601.11467](https://arxiv.org/html/2601.11467v2)

### Inferences
- Pattern: well-engineered metaheuristics (HGS, LKH-3, AILS) + engineering (speed, neighborhood pruning) win; pure end-to-end neural solvers have not won classical OR competitions.
- ORTEC's repeated success (DIMACS VRPTW, EURO-NeurIPS organization and HGS baselines, PyVRP authorship) makes it a key industrial actor.

### Gaps
- Exact final rankings of the EURO-NeurIPS 2022 static and dynamic tracks could not be verified (PMLR/NeurIPS pages blocked).
- No results found for a 2024 ROADEF challenge (the challenge appears biennial: 2020, 2022, then 2026 listed).

## Q4. Leading academic and industrial groups

### Takeaway
Leadership is concentrated in Montréal (Vidal at Polytechnique/CIRRELT/GERAD; GERAD's GENCOL crew scheduling lineage; Cordeau, Crainic, Gendron, Gendreau), Brazil/France (Uchoa, Pessoa at UFF with Sadykov at Inria Bordeaux: VRPSolver, CVRPLib), Bologna (Toth/Vigo; FILO by Accorsi & Vigo), Google OR (OR-Tools CP-SAT, PDLP), ORTEC/Amsterdam (Kool, Wouda: PyVRP), plus NVIDIA (cuOpt) and Hexaly in industry.

### Cited Findings
- Vidal co-organized EURO-NeurIPS 2022 and co-authored the XL/BKS challenge work with Uchoa and Queiroga — [PMLR v220](https://proceedings.mlr.press/v220/kool23a.html); [arXiv 2601.11467](https://arxiv.org/html/2601.11467v2); [Vidal LinkedIn on DIMACS](https://www.linkedin.com/posts/thibaut-vidal-7a877055_the-12th-dimacs-implementation-challenge-activity-7026223819711238145-dCIr)
- VRPSolver: Pessoa, Sadykov, Uchoa, Vanderbeck (Inria/UFF) — [Math Programming 2020](https://link.springer.com/article/10.1007/s10107-020-01523-z); [VRPSolverEasy (inria-UFF)](https://github.com/inria-UFF/VRPSolverEasy)
- CVRPLib hosted by Galgos, PUC-Rio — [CVRPLib BKS Challenge](https://vrp.galgos.inf.puc-rio.br/index.php/en/bks-challenge)
- GERAD/Montréal crew pairing with GENCOL and ML (Desaulniers, Yaakoubi, Soumis lineage) — [Transportation Science](https://pubsonline.informs.org/doi/10.1287/trsc.2021.1084)
- Crainic, Gendreau, Gendron (Montréal) lead network design — [arXiv 2009.14628](https://arxiv.org/pdf/2009.14628); [arXiv 2303.14217](https://arxiv.org/pdf/2303.14217)
- ORTEC/UvA: Wouda, Lan, Kool (PyVRP) — [arXiv 2403.13795](https://arxiv.org/pdf/2403.13795)
- Cook (Waterloo), Held (Bonn), Helsgaun (Roskilde, LKH) — [Amazon Science](https://www.amazon.science/academic-engagements/winning-last-mile-challenge-team-addresses-problem-of-combining-mathematical-routes-with-driver-knowledge)
- Google OR team: OR-Tools CP-SAT, PDLP (Applegate et al.) — [OR-Tools Wikipedia](https://en.wikipedia.org/wiki/OR-Tools); [arXiv 2506.02174](https://arxiv.org/pdf/2506.02174)
- Mittelmann (Arizona State) runs the reference public solver benchmarks — [plato.asu.edu](https://plato.asu.edu/bench.html)
- Huawei Cloud (Alkaid-X) is a notable industrial CVRP heuristic group — [DIMACS news](https://dimacs.rutgers.edu/news/news-details/the-12th-dimacs-implementation-challenge-vrp)

### Inferences
- Zuse Institute Berlin (SCIP), MIT ORC and Georgia Tech ISyE are major OR centers, but this session found no specific 2025-2026 routing/scheduling benchmark leadership claims tied to them; treat as general reputation.

### Gaps
- No sourced evidence gathered this session on ZIB (SCIP 9/10), MIT ORC, or Georgia Tech ISyE-specific contributions to routing/scheduling benchmarks.
- LKH-3 current status and FILO2 (Accorsi & Vigo) primary papers not fetched.
