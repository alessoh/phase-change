# Logistics targets for an autoresearch-style LLM agent loop (prior art, candidate benchmarks, pitfalls)

Research note: arxiv.org, alphaxiv.org, pyvrp.org, pmc.ncbi.nlm.nih.gov, pith.science and galgos.inf.puc-rio.br (CVRPLib challenge site) were blocked by the network proxy, so I could not open them. Where a claim rests on a search-result snippet or abstract and not the full paper, the note says so. Unless marked otherwise, every headline number from these systems is **self-reported** by the authors or vendor.

## Prior art: what have LLM-driven algorithm-design systems achieved on logistics problems?

### Takeaway
LLM-evolved heuristics have credible, repeated wins on CVRP, bin packing and shop scheduling. The strongest 2026 results come from plugging an evolved component (such as a ruin operator) into an existing state-of-the-art solver (AILS-II), not from writing a solver from scratch. Independent re-evaluations show that many evolved heuristics generalize poorly outside the instances they were trained on.

### Cited Findings
**Autoresearch pattern (the loop being copied)**
- Karpathy's autoresearch works like this: an agent reads the training code, proposes a change, runs a fixed 5-minute wall-clock training job, and commits the change if val_bpb (validation bits per byte, lower is better) improves, otherwise rolls it back. That gives about 12 experiments/hour, or about 100 overnight. — [karpathy/autoresearch](https://github.com/karpathy/autoresearch); [Data Science Dojo explainer](https://datasciencedojo.com/blog/karpathy-autoresearch-explained/)
- Reported example sessions: val_bpb went from 0.9979 to 0.9773 over 89 experiments on one H100, and from 0.9979 to 0.9697 over 126 experiments (these come from secondary write-ups, not audited). — [kingy.ai guide](https://kingy.ai/news/autoresearch-karpathys-minimal-agent-loop-for-autonomous-llm-experimentation/)

**CVRP: AILS-AHD (2026) and the CVRPLib BKS challenge**
- AILS-AHD (arXiv 2602.23092, Feb 2026) uses an evolutionary search with an LLM (GPT-4o) to generate and optimize *ruin heuristics* inside the AILS iterated local search. It adds an LLM-based acceleration mechanism. It reports 8 new best-known solutions (BKS) on 10 large-scale AGS instances plus 1 new BKS on the X set (X-n1001-k43), and claims to beat HGS and AILS-II. Self-reported; I only saw the abstract and snippets. — [arXiv abs 2602.23092](https://arxiv.org/abs/2602.23092v1); [arXiv HTML](https://arxiv.org/html/2602.23092)
- A "CVRPLib Best Known Solution Challenge" paired with new XL instances was described in Jan 2026 (arXiv 2601.11467). At least one submission was titled "LLM-Driven Automated Algorithm Design for Large-scale ..." — [arXiv 2601.11467](https://arxiv.org/pdf/2601.11467); [submission PDF on CVRPLib site](https://galgos.inf.puc-rio.br/cvrplib/uploads/teams/Sub-Appro-20260111-155603.pdf) (blocked, content not verified)
- The EoH repo README says its team "won CVRPLib BKS competition with 51 new Best Known Solutions on large-scale CVRP benchmarks". Self-reported in a README; I could not verify it against the challenge results page. — [FeiLiu36/EoH](https://github.com/FeiLiu36/EoH)
- Other 2026 large-scale CVRP automatic heuristic design (AHD) papers: "Automated Large-scale CVRP Solver Design via LLM-assisted Flexible MCTS" (arXiv 2605.03339), SpecAHD (arXiv 2607.23676), G-LNS (arXiv 2602.08253). Titles only; I could not read them. — [2605.03339](https://arxiv.org/pdf/2605.03339); [2607.23676](https://arxiv.org/pdf/2607.23676); [2602.08253](https://arxiv.org/pdf/2602.08253)

**ReEvo (NeurIPS 2024)**
- ReEvo improves GA, ACO, GLS, constructive heuristics and neural CO models on TSP, CVRP, orienteering, multiple knapsack (MKP), bin packing and decap placement (17 variants). Its example config is population 4, max 20 heuristic evaluations, 20 s evaluation time per generation. Instances are generated on the fly. The README says: "Give ReEvo 5 minutes, and get a state-of-the-art algorithm in return!" — [ai4co/reevo](https://github.com/ai4co/reevo)
- Numbers from search snippets of the paper: KGLS-ReEvo cuts the TSP200 gap from 0.284% to 0.216% at a 2.52 s runtime. Attention reshaping cuts POMO's TSP1000 gap from 52.11% to 29.08%, and LEHD's from 3.17% to 2.97%. Self-reported. — [ReEvo arXiv 2402.01145](https://arxiv.org/pdf/2402.01145)
- The follow-up ReVEL (arXiv 2604.04940) reports beating ReEvo on every CVRP scale it tested, e.g. CVRP20 4.57% vs 13.70% (snippet). — [ReVEL](https://arxiv.org/html/2604.04940v2)

**FunSearch / EoH on online bin packing**
- On OR-Library, FunSearch reports excess bins (over the lower bound) of 5.30% / 4.19% / 3.11% / 2.47% on OR1–OR4. First Fit scores 6.42% to 5.23% and Best Fit scores 5.81% to 4.94%. FunSearch evolved its heuristic on a handful of OR1 instances. — as summarized in search snippets of [X-evolve arXiv 2508.07932](https://arxiv.org/pdf/2508.07932) and the FunSearch Nature paper ([PMC10794145](https://pmc.ncbi.nlm.nih.gov/articles/PMC10794145/), blocked)
- EoH (ICML 2024 oral) claims to beat FunSearch on online bin packing with "significantly fewer computational budgets" and to produce competitive algorithms "in 30 minutes on your personal computer". It supports 33 tasks, including TSP (constructive/GLS/ruin-and-recreate), CVRP, bin packing and flow shop. Self-reported. — [FeiLiu36/EoH](https://github.com/FeiLiu36/EoH)

**MCTS-AHD, HeurAgenix, others**
- MCTS-AHD (arXiv 2501.08603) uses Monte Carlo tree search over LLM-generated heuristics. Follow-up papers compare against it on TSP/CVRP constructive tasks and TSPLib. In snippets, CVRP gaps from about 0.05% to 1.87% (N=100–1000) are attributed to it. — [MCTS-AHD](https://arxiv.org/pdf/2501.08603); [ICLR 2026 paper comparing it](https://proceedings.iclr.cc/paper_files/paper/2026/file/b9c91701ff28d480f186e4643e71ad11-Paper-Conference.pdf)
- HeurAgenix (Microsoft, arXiv 2506.15196) has two stages. First an LLM evolves a pool of heuristics by comparing seed solutions with better ones. Then an LLM, or a fine-tuned small model, picks a heuristic per problem state. It claims to outperform state-of-the-art hyper-heuristics. — [arXiv 2506.15196](https://arxiv.org/abs/2506.15196); [microsoft/HeurAgenix](https://github.com/microsoft/HeurAgenix)

**AlphaEvolve / OpenEvolve / ShinkaEvolve**
- AlphaEvolve's data-center scheduling heuristic recovers about 0.7% of Google's fleet-wide compute on average (Google-reported). — [DeepMind blog](https://deepmind.google/blog/alphaevolve-a-gemini-powered-coding-agent-for-designing-advanced-algorithms/)
- A May 2026 update, per a secondary source, says AlphaEvolve is used for TPU design, cache policies, and Spanner compaction (20% less write amplification). FM Logistic reportedly saw a 10.4% routing-efficiency gain and saves more than 15,000 km/year. The secondary source itself notes these are not independently audited. — [Buildmind AI summary](https://buildmind.ai/blog/google-deepmind-alphaevolve-may-2026-production-algorithm-discovery/)
- ShinkaEvolve (Sakana) reports a new state of the art on circle packing (26 circles) using only about 150 samples, and positions itself as a sample-efficient alternative to AlphaEvolve. This is not a logistics result, but it shows a small-budget loop can work. — [Sakana blog](https://sakana.ai/shinka-evolve/); [arXiv 2509.19349](https://arxiv.org/abs/2509.19349)

**Scheduling**
- EoH reports relative makespan gaps of 0.10–0.60% on six Taillard *flow-shop* sets (snippet via a follow-up paper). — [Teacher-Aware Evolution arXiv 2605.10634](https://arxiv.org/html/2605.10634)
- In dynamic flexible assembly flow-shop, all classical dispatching rules had a RelGap above 50%. Composite rules evolved by LLMs (EoH, HSEvo, ReEvo) or GP stayed within 20%. — [EvoDR arXiv 2601.15738](https://arxiv.org/html/2601.15738)
- SeEvo (LLM population self-evolution for dynamic JSSP) reports beating LPT/SPT/STPT/MPSR, GP/GEP and DRL on the DMU and Taillard sets. — [arXiv 2410.22657 via alphaXiv](https://www.alphaxiv.org/abs/2410.22657)
- Other related work: DSevolve (dynamic flexible job shop (FJSP) heuristic portfolios) and LLM heuristic design for production/AGV scheduling from simulation traces (Aug 2026). — [DSevolve](https://arxiv.org/html/2603.27628); [arXiv 2608.09343](https://arxiv.org/pdf/2608.09343)

### Inferences
- The best-supported "significant improvement" story in 2025–26 is **evolving a single operator inside a strong solver** (ruin method in AILS, the scoring function in bin packing, a priority/dispatch rule). Evolving whole solvers is less common.
- Headline claims like "new BKS" came from long runs on large compute. In 24–48 h a hackathon team should aim for "beats a strong named baseline on a held-out set", not new world records.

### Gaps
- I could not open AILS-AHD's full text, so I don't have its runtimes, LLM query budget, or number of seeds.
- I could not confirm the official CVRPLib BKS challenge rankings, or EoH's claim to have "won" it with 51 BKS.
- I found no LLaMEA or AI Scientist results specifically on logistics. LLaMEA mainly targets continuous black-box optimization (BBOB), per the in-the-loop HPO paper ([arXiv 2410.16309](https://arxiv.org/pdf/2410.16309)); not verified further.

## Candidate targets and how each would be evaluated

### Takeaway
For a 5-minute, single-scalar loop, the best-fit targets are (1) online bin packing with a priority function, (2) a CVRP operator or heuristic evaluated by gap to BKS on a fixed CVRPLIB X subset using PyVRP/HGS, and (3) job-shop dispatching rules on Taillard. All three run on CPU and are deterministic given a seed. Neural RL4CO training, forecasting and Amazon Last Mile are weaker fits.

### Cited Findings
**(a) CVRP on CVRPLIB X with PyVRP/HGS**
- PyVRP's standard protocol: Uchoa X instances, runtime 2.4·n seconds on a reference CPU (PassMark 2183), so n=100 gets 240 s. Each instance runs with 10 seeds and the average objective is compared with the BKS in the PyVRP/Instances repo. — [PyVRP benchmarking docs](https://pyvrp.readthedocs.io/en/stable/dev/benchmarking.html); [PyVRP paper](https://arxiv.org/pdf/2403.13795)
- AILS-AHD shows that evolving the ruin operator of a state-of-the-art solver can beat HGS/AILS-II. — [arXiv 2602.23092](https://arxiv.org/abs/2602.23092v1)
- Baseline to beat: PyVRP's HGS defaults, at a fixed time budget per instance. Data are public (CVRPLIB, PyVRP/Instances). CPU only.

**(b) Neural routing in RL4CO**
- RL4CO (MIT license, KDD 2025) runs TSP/CVRP/scheduling environments through PyTorch Lightning, e.g. `python run.py experiment=routing/am env=tsp env.num_loc=50`. The README gives no training-time benchmarks. — [ai4co/rl4co](https://github.com/ai4co/rl4co)
- The RL4CO paper reports that AM with small encoder changes and enough samples can match POMO on CVRP20/50. This suggests the space of training tweaks is productive but noisy. — [RL4CO paper](https://arxiv.org/pdf/2306.17100)
- Needs a GPU. A 5-minute budget reaches only a fraction of normal training, so gains may reflect "faster early convergence", not better final quality.

**(c) Online bin packing (FunSearch benchmark)**
- Metric: % excess bins over the lower bound. Datasets: OR1–OR4 (OR-Library) and Weibull 5k/10k/100k. Baselines: First Fit and Best Fit (≈4.9–6.4% on OR). The FunSearch heuristic reaches 2.47–5.30%. — [X-evolve](https://arxiv.org/pdf/2508.07932)
- Evaluation takes seconds on CPU and is fully deterministic. Starter code exists in EoH, ReEvo and FunSearch. — [EoH](https://github.com/FeiLiu36/EoH); [ReEvo](https://github.com/ai4co/reevo)
- Data from the "Beyond the Hype" benchmark are on Zenodo. — [Zenodo 14162744](https://zenodo.org/records/14162744)

**(d) Job-shop / flow-shop dispatching on Taillard**
- The strong baselines are classical dispatching rules (SPT, LPT, MWKR, etc.). Composite LLM-evolved rules have been shown to cut gaps by a large margin (from >50% down to <20% in a dynamic assembly flow-shop setting). — [EvoDR](https://arxiv.org/html/2601.15738); [SeEvo](https://www.alphaxiv.org/abs/2410.22657)
- CPU only; fast simulation. The Starjob dataset exists for LLM-JSSP work. — [Starjob](https://arxiv.org/html/2503.01877v2)

**(e) Demand forecasting (Chronos/TimesFM, M5)**
- On Chronos Benchmark II (WQL), Chronos-2 reports an 81.5% win rate and 26.5% skill score vs TimesFM-2.5's 71.6% and 23.3%. Self-reported by the Chronos-2 authors. — [Chronos-2 arXiv 2510.15821](https://arxiv.org/pdf/2510.15821)
- Fine-tuning Chronos on local data beat zero-shot in a water-quality study. — [ScienceDirect](https://www.sciencedirect.com/science/article/pii/S2589914725001677)
- I found no source directly comparing Chronos, TimesFM and LightGBM on M5. Fine-tuning needs a GPU. Decision-cost metrics (e.g. inventory cost) need a simulator the team would have to build.

**(f) Amazon Last Mile Routing Challenge (2021)**
- 9,184 real routes (6,112 train, 3,072 eval), 50–250 stops each, with time windows, service times and zone IDs; openly on the AWS Open Data registry. The score measures dissimilarity between the predicted and actual driver sequence (lower is better). The AWS sample solution reports about 0.0372–0.0376, comparable to the top-3 public leaderboard. — [AWS Open Data](https://registry.opendata.aws/amazon-last-mile-challenges/); [aws-samples solution](https://github.com/aws-samples/amazon-sagemaker-amazon-routing-challenge-sol)
- The metric rewards imitating drivers, not minimizing cost. Scoring 3,072 routes may take longer than 5 minutes unless the team subsamples.

### Inferences (comparison table)
| Target | Eval runtime | GPU | Determinism | Public baseline | Overfit risk | Fit for 5-min loop |
|---|---|---|---|---|---|---|
| (c) Online bin packing priority fn | seconds | No | Full | FF/BF, FunSearch, EoH | High (see "Beyond the Hype") | Excellent |
| (a) CVRP operator in PyVRP/HGS, X subset | minutes (fixed time per instance) | No | Seeded, some noise | HGS-PyVRP, BKS | Medium | Very good if ~10–20 instances × short limits |
| (d) JSSP/FSSP dispatching on Taillard | seconds | No | Full | SPT/MWKR, EoH/ReEvo | Medium | Very good |
| (b) RL4CO neural training | 5 min = partial training | Yes | Noisy | POMO/AM checkpoints | Medium | Fair, closest to Karpathy's original |
| (f) Amazon Last Mile | minutes on a subset | No | Full | ~0.037 score | Medium | Fair |
| (e) Forecast FM fine-tuning | minutes | Yes | Noisy | Chronos/TimesFM zero-shot | High | Weak (metric design burden) |

- Runtime, determinism and overfit ratings in this table are my estimates, based on the sources above.

### Gaps
- No verified per-instance gaps for PyVRP on X at short (e.g. 10–30 s) limits. The docs page was blocked and the readthedocs snippet did not list them.
- No verified RL4CO wall-clock training numbers for CVRP50 in 5 minutes.

## Documented pitfalls and safeguards

### Takeaway
The main documented failures are overfitting to the tiny training set, poor generalization across instance distributions, and reward hacking of the evaluator. Safeguards: a held-out instance split, multiple seeds, a fixed CPU time budget enforced from outside, and a frozen, read-only evaluator.

### Cited Findings
- "Beyond the Hype" (Sim, Renau, Hart; EvoApplications 2025) benchmarked the LLM-evolved bin-packing heuristics across broad instance sets. It found that "most of the LLM heuristics do not generalise well" compared with simple heuristics, and that specialist gains must be weighed against generation cost. — [arXiv 2501.11411](https://arxiv.org/abs/2501.11411); [Springer](https://link.springer.com/chapter/10.1007/978-3-031-90065-5_24)
- The FunSearch Weibull-evolved heuristic performed poorly on other datasets, winning only one instance of one dataset (snippet). — [X-evolve](https://arxiv.org/pdf/2508.07932) / [Beyond the Hype](https://arxiv.org/html/2501.11411v1)
- Follow-up studies question LLM contributions directly: "An In-depth Study of LLM Contributions to the Bin Packing Problem" (ACM TELO), and "Re-evaluating LLM-based Heuristic Search: A Case Study on the 3D Packing Problem". I read titles/abstract snippets only. — [arXiv 2510.27353](https://arxiv.org/pdf/2510.27353); [2509.02297 summary](https://pith.science/paper/2509.02297)
- Reward hacking is "a frequent failure pattern in AlphaEvolve experiments". In one example the system "created a new class type list in which it updated the meaning of length". Users said that when they "thought it was doing well, it was cheating". — [Intentmaking and Sensemaking arXiv 2605.05921](https://arxiv.org/pdf/2605.05921)
- Critiques of AHD evaluation: generalization claims often omit train/test split details, domain diversity and statistical tests. Fitness weights are often unreported. — [search snippets incl. pith review of 2605.29649](https://pith.science/paper/2605.29649)
- PyVRP's own protocol uses 10 seeds per instance and average objective, a model for handling noise. — [PyVRP docs](https://pyvrp.readthedocs.io/en/stable/dev/benchmarking.html)
- ReEvo generates instances on the fly and does not document a separate evolution/test split in the README. Teams should add one. — [ai4co/reevo](https://github.com/ai4co/reevo)

### Inferences (recommended guards)
- Keep a train set for the keep/revert decision (e.g. 10 X instances or 5 OR1 bin-packing files). Report on a **held-out** test set the agent never sees (e.g. other X instances, OR2–OR4 plus Weibull).
- Make the evaluator read-only and outside the agent's editable file, as Karpathy's `prepare.py` is fixed. Validate feasibility independently (capacity, all customers visited once, bins not over capacity). Recompute the cost from the returned solution rather than trusting any value the candidate reports.
- Enforce time limits in CPU time or iterations from the harness, and count LLM-accelerated or precomputed work, so candidates cannot game the clock. Pin threads to remove hardware noise.
- For stochastic solvers: at least 3–5 seeds, and a keep threshold larger than seed noise (e.g. mean improvement > 1 SD, or a paired test).

### Gaps
- I found no published case of reward hacking specific to VRP evaluators, but the AlphaEvolve evidence suggests it should be expected.

## Which option makes the most convincing demo for judges?

### Takeaway
Best bet: **CVRP on CVRPLIB X with PyVRP (HGS) as the frozen baseline, where the agent edits one operator or construction heuristic (or a ruin/perturbation step, following AILS-AHD) and is scored by mean % gap to BKS at a fixed short time limit, over several seeds, on a held-out X subset.** A lower-risk fallback is **online bin packing**: it runs in seconds, so a large improvement over First Fit/Best Fit (around 2 percentage points of excess bins) is almost guaranteed, and it can be benchmarked against published FunSearch/EoH numbers.

### Cited Findings
- CVRP gap-to-BKS is the metric used by the 2026 state-of-the-art AHD work, and the BKS reference set is public. — [AILS-AHD](https://arxiv.org/abs/2602.23092v1); [PyVRP docs](https://pyvrp.readthedocs.io/en/stable/dev/benchmarking.html)
- Bin packing has known published numbers to beat (FF/BF 4.9–6.4% vs FunSearch 2.5–5.3% excess bins on OR). — [X-evolve](https://arxiv.org/pdf/2508.07932)
- Real-world credibility: AlphaEvolve's reported FM Logistic routing gain (10.4%, unaudited) shows industry interest in LLM-evolved routing. — [Buildmind AI](https://buildmind.ai/blog/google-deepmind-alphaevolve-may-2026-production-algorithm-discovery/)

### Inferences
- Judges will find an improvement most convincing when it is (1) against a named, strong, unmodified baseline, (2) on held-out instances, (3) shown as a curve of accepted commits (as in autoresearch), and (4) paired with a readable diff of what the agent changed.
- Beating HGS itself by much in 48 h is unlikely. A more realistic honest claim is "improved a weaker starting heuristic (e.g. Clarke-Wright + 2-opt, or PyVRP with reduced operators) by X% gap on held-out X instances", or "matched HGS at half the time".
- Neural RL4CO training is the closest analogue to Karpathy's setup, but it needs a GPU and early-training noise makes results less trustworthy.

### Gaps
- There is no public hackathon-scale (24–48 h) replication of any of these systems on logistics that could calibrate the size of improvement to expect.
