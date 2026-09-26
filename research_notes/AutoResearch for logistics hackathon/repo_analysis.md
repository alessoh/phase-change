# parzival-gauntlet: how it works, and how it differs from Karpathy's autoresearch

Primary source: a local clone of https://github.com/alessoh/parzival-gauntlet (single commit `0c9fa76`, "ledger snapshot: experiments 1-9", author "parzival", dated 2026-08-10). Line numbers below refer to that commit. Local-file citations use the repo URL plus the file and line numbers, for example [program.md L114](https://github.com/alessoh/parzival-gauntlet).

## Q1. What the repo is and what each file does (loop, metric, time budget, data, model, what is editable and what is frozen)

### Takeaway
parzival-gauntlet is a hardened copy of the "autoresearch-at-home" collaborative fork (Ensue swarm) of Karpathy's autoresearch. It keeps the upstream design: one agent edits one file (`train.py`), each run trains an LLM for a fixed 300 s on one GPU, and a frozen `prepare.py` computes one scalar, `val_bpb`, where lower is better. On top of that it adds a "Gauntlet protocol". Before every GPU run, a critic subagent reviews the experiment for validity. After every run, an immutable statistical keep gate (`keep_gate.py`) decides keep or discard against replicate-calibrated noise in `NOISE.md`. It also adds quarantine of foreign code, an append-only ledger of failure patterns (`gauntlet.md`), and a predictions file (`predictions.tsv`) for calibration. The `train.py` in this repo is not upstream's. It is a ported "Recursive" champion script of about 1,420 lines, tuned for an H100.

### Cited Findings

**Lineage and identity**
- The README describes this as "A collaborative, SETI@home-style fork of @karpathy's autoresearch. Multiple agents on different GPUs share results, avoid redundant work, and collectively drive down val_bpb through a shared Ensue workspace". It cites Karpathy's March 2026 tweet: "The next step for autoresearch is that it has to be asynchronously massively collaborative for agents (think: SETI@home style)…" — [README.md L1-9](https://github.com/alessoh/parzival-gauntlet); tweet https://x.com/karpathy/status/2030705271627284816 (x.com was not reachable from this environment; the quote is taken from the README).
- The README text is the same as the public mutable-state-inc/autoresearch-at-home README. That README reports no participant counts or val_bpb numbers — [autoresearch-at-home](https://github.com/mutable-state-inc/autoresearch-at-home).
- The Gauntlet section of `program.md` fixes the agent identity: `coord.agent_id = "parzival"` ("this IS parzival, upgraded") — [program.md L131-135](https://github.com/alessoh/parzival-gauntlet).

**Files**
- **`prepare.py`** (389 lines; frozen, "do not modify"):
  - Constants: `MAX_SEQ_LEN = 2048`, `TIME_BUDGET = 300`, `EVAL_TOKENS = 40 * 524288` (about 21M tokens) [L30-32].
  - Data: parquet shards from HF dataset `karpathy/climbmix-400b-shuffle`. Validation is pinned to shard 06542 (`VAL_SHARD = MAX_SHARD`) [L41-44]. By default 10 training shards are downloaded [L373].
  - Tokenizer: an 8192-vocab BPE trained with rustbpe and saved as a tiktoken pickle, plus a `token_bytes` lookup [L45, L141-203].
  - Dataloader: BOS-aligned, best-fit packing, allocated on CUDA [L276-337].
  - Metric: `evaluate_bpb` [L343-365] is marked "DO NOT CHANGE — this is the fixed metric". It sums per-token cross-entropy in nats over non-special tokens and divides by `ln(2) * total_bytes`, so the result is bits per byte and does not depend on vocabulary size.
- **`train.py`** (1421 lines; the only file the agent edits):
  - The header says it is a "ONE-CLICK reproduce of our best result on the karpathy/recursive nanochat_autoresearch harness… 8-seed mean val_bpb ~= 0.9365 (seed42 ~= 0.9359) vs recursive's champion recipe ~= 0.944 on H100 (their 0.9109 is on B200)" [L1-24].
  - The code is "Copyright 2026 Recursive / 2025 Andrej Karpathy, Apache-2.0" [L42-44].
  - Tuned defaults are injected through `os.environ.setdefault`: DEPTH 8, MODEL_DIM 640, NGRAM_MULT 48, TRIGRAM_MULT 96, MATRIX_LR 0.05, EMBEDDING_LR 0.6, WARMDOWN 0.90, WARMUP 0.05, COMPILE_MODE max-autotune-no-cudagraphs [L25-40].
  - Model: a GPT with value embeddings, hashed multi-table bigram and trigram "value embeddings", x0 skip gating and layer pooling [L285-330]. Attention uses FlashAttention-3 on Hopper/Ampere via the `kernels` package, or FlashAttention-4 on Blackwell [L66-137].
  - Optimizer: a custom MuonAdamW (Muon for matrices, AdamW/RMSProp for the rest) [L698-900]. Schedules (LR, momentum, weight-decay "pulses") are functions of `progress = total_training_time/TIME_BUDGET` [L1142-1300].
  - `SEED = 42` by default [L933-935].
  - It imports `DATA_DIR, MAX_SEQ_LEN, TIME_BUDGET, VAL_FILENAME, Tokenizer, evaluate_bpb` from prepare.py [L139] but uses its own multiprocess training dataloader, which excludes the val shard [L1020-1133].
  - The training loop stops once `step > 10 and total_training_time >= TIME_BUDGET`. The first 10 steps are excluded from the timed total, so compilation is not counted [L1375-1382].
  - It then calls `evaluate_bpb` and prints `val_bpb`, `training_seconds`, `total_seconds`, `peak_vram_mb`, `mfu_percent`, `total_tokens_M`, `num_steps`, `num_params_M`, `depth` [L1386-1417]. A NaN or loss > 100 is a fast fail [L1336-1339].
  - MFU is computed against a hard-coded B200 peak (`B200_BF16_PEAK_FLOPS = 2.25e15`) [L940], so printed MFU is not meaningful on an H100.
- **`program.md`** (183 lines): agent instructions for solo mode plus the Gauntlet protocol.
  - Setup: check `nvidia-smi`; create branch `autoresearch/<tag>`; initialize results.tsv [L9-21].
  - Rules: only `train.py` may be edited. No new packages. `prepare.py`/`evaluate_bpb`, `keep_gate.py` and existing `gauntlet.md` rules are frozen [L27-35].
  - The goal is "get the lowest val_bpb". VRAM is a soft constraint. A "Simplicity criterion" weighs code complexity against gain [L37-41].
  - The LOOP FOREVER steps are THINK → CLAIM → edit → git commit → `uv run train.py > run.log 2>&1` → grep metrics → log to TSV → keep/discard (git reset on discard) → PUBLISH [L98-116].
  - Timeouts and autonomy: kill runs longer than 10 min; "NEVER STOP"; about 12 runs/hour, about 100 per night [L119-125].
- **`gauntlet.md`** (163 lines): an append-only ledger of 27 numbered "ways to waste a 5-minute slot" (details in Q2/Q3) — [gauntlet.md](https://github.com/alessoh/parzival-gauntlet).
- **`NOISE.md`** (76 lines): the noise calibration that `keep_gate.py` reads. The first line starting `sigma=` is `sigma=0.000217`, taken from 3 replicates at commit 46b52aa [L1-5].
- **`keep_gate.py`** (34 lines; "IMMUTABLE: never edited by the loop"): see Q2.
- **`collab.md`** (227 lines): the Ensue swarm protocol, covering namespaces, the claiming protocol, global/tier/agent bests and mandatory publishing — [collab.md](https://github.com/alessoh/parzival-gauntlet).
- **`coordinator.py`** (1318 lines): a JSON-RPC client to `https://api.ensue-network.ai/` [L34] for hub org `autoresearch-at-home` [L33].
  - Claims: `CLAIM_TTL = 900` s, `SEMANTIC_THRESHOLD = 0.92`, `MAX_CLAIM_ATTEMPTS = 5`, `SYNC_EVERY_N = 5` [L37-41].
  - VRAM tiers: small ≤16 GB, medium ≤24, large ≤48, xl >48 [L43-55, L116-133].
  - Result publishing carries the full `train.py` source, VRAM tier, git info and delta vs best [L437-470].
  - Global-best sanity checks reject `val_bpb < 0.5` and single-step improvements > 0.1 [L758-801]. These thresholds are hard-coded to the bpb scale.
  - Other functions: `ask_swarm`, `analyze_swarm`, `publish_hypothesis`, `post_insight` [L906-1305].
- **`setup_hub.py`** (169 lines): a one-time admin script that creates the Ensue hub org and grants read/create/update on `claims/ results/ hypotheses/ insights/ best/ leaderboard` to a "participants" group [L42-80]. Gauntlet guardrail 1 forbids running it [program.md L176].
- **`pyproject.toml`**: Python ≥3.10. Dependencies: `torch==2.9.1` (cu128 index), kernels, numpy, pandas, pyarrow, requests, rustbpe, tiktoken, matplotlib [L1-27].
- **`analysis.ipynb`**: the upstream-style notebook. It loads results.tsv, counts keep/discard/crash, plots kept points with a running minimum, and ranks "top hits" by delta over the previous keep.
- **`progress.png`**: this image is upstream Karpathy's chart, not this repo's data. It is titled "Autoresearch Progress: 83 Experiments, 15 Kept Improvements" and shows val_bpb falling from about 0.998 (baseline) to about 0.977.
  - Kept steps visible on the chart include: halving total batch 524K→262K, warmdown 0.5→0.7, 5% warmup, depth 9, SSSSL window pattern, short-window context 1/8, embedding LR 0.6→0.8, RoPE base 10000→200000, and a final "random seed 42→137" keep.
- **Data, model, budget and hardware (the parzival ledger)**
  - Upstream's baseline on the same harness was 1.0412–1.0431 [results.tsv L2-4].
  - Hardware: "NVIDIA H100 80GB HBM3, driver 580.173.02, CUDA 13.0, torch 2.9.1+cu128" [NOISE.md L44].
  - Resources at the current config: peak VRAM about 39.7 GB, about 2,338 optimizer steps and 344.8M tokens in 300 s [NOISE.md L13].
  - Startup cost with max-autotune compile is about 56–122 s before step 1. Total wall time per run is about 356–422 s [NOISE.md L67-71].

**Karpathy's upstream (for comparison)**
- The idea: "give an AI agent a small but real LLM training setup and let it experiment autonomously overnight. It modifies the code, trains for 5 minutes, checks if the result improved, keeps or discards, and repeats" — [karpathy/autoresearch README](https://github.com/karpathy/autoresearch).
- It has three files: prepare.py (not modified), train.py ("the single file the agent edits"), and program.md (edited by the human) — [karpathy/autoresearch](https://github.com/karpathy/autoresearch).
- Budget: "Training always runs for exactly 5 minutes, regardless of your specific platform… approx 12 experiments/hour and approx 100 experiments while you sleep" — [upstream README](https://raw.githubusercontent.com/karpathy/autoresearch/master/README.md).
- Hardware: it needs "a single NVIDIA GPU (tested on H100), Python 3.10+, uv". For smaller compute it suggests TinyStories, vocab 4096→256, lower MAX_SEQ_LEN and EVAL_TOKENS, DEPTH 4, window pattern "L", and smaller TOTAL_BATCH_SIZE. Community forks exist for macOS (miolini, trevin-creator/MLX), Windows RTX (jsegov) and AMD (andyluo7) — [upstream README](https://raw.githubusercontent.com/karpathy/autoresearch/master/README.md).

### Inferences
- **Summary of differences from upstream:**
  1. It adds the Ensue swarm layer: claims, published results with full source, global/tier bests, hypotheses and insights.
  2. It replaces the upstream keep rule ("If val_bpb improved, keep… If equal or worse, git reset") with a statistical gate against replicate-calibrated sigma.
  3. It adds pre-flight critic subagents that may kill runs only on validity grounds.
  4. It quarantines foreign train.py with an AST import ban plus an adversarial security critic, because pulling code from 136+ other agents is a code-execution risk.
  5. It adds append-only institutional memory (gauntlet.md) and a prediction-interval calibration file.
  6. It swaps upstream's small nanochat train.py for a heavily evolved third-party "Recursive" champion script.
- The "agent loop" is not code. It is natural-language instructions in `program.md`/`collab.md`, executed by a coding agent (Claude Code: "spawn a subagent via the Task tool"). The only deterministic code in the loop is `train.py`, `prepare.evaluate_bpb`, `keep_gate.py` and `coordinator.py`.

### Gaps
- It could not be confirmed who "Recursive" is, or where their `optimized_from_karpathy.py` is published. The repo only names it in comments.
- Swarm-level numbers (136 agents, swarm #1 = 0.899885 on B200) come only from program.md text [L147, L172]. They could not be verified because ensue.dev and ensue-network.ai were blocked from this environment.

## Q2. What results the repo recorded, how much the loop improved the metric, and how noise is controlled

### Takeaway
Almost all of the recorded gain came from porting an externally evolved train.py, not from this repo's own loop. The upstream baseline mean was about 1.0419 and the ported script (c21b5ee) reached 0.935433. The parzival loop's own gain after that is 0.935433 → 0.934625 (−0.00081 bpb) from two keeps. The second keep is flagged by the ledger itself as statistically unresolved (p = 0.40). Eleven of the 13 subsequent experiments were discarded, and one idea was killed pre-flight at zero GPU cost. Noise is controlled with three-replicate sigma calibration, a deterministic gate (KEEP if Δ > 2σ, DISCARD if worse than 1σ, otherwise replicate), confirmation replicates for provisional bests, and a long list of statistical caveats the agent wrote itself.

### Cited Findings

**results.tsv (24 data rows)**
- Upstream-baseline calibration: 1.041222 / 1.043131 / 1.041302, giving σ = 0.001080 [results.tsv L2-4].
- Ported "0.935 lineage" (c21b5ee): 0.935304 / 0.935518 / 0.935478, σ = 0.000114 [L5-7].
- Shared-trigram VE (d3df153): 0.934668 / 0.934706 / 0.935186 [L8-10].
- Preflight-kill of NGRAM_MULT 48→32 as a duplicate of lodestar's swarm result, with "Zero GPU spent" [L11].
- TRIGRAM_MULT 48→96 (46b52aa): 0.934398 / 0.934831 / 0.934645. The row says "gate KEEP (n=3 mean rule) but UNRESOLVED: delta 0.000229 < 1x pooled sigma, p=0.40, CI [-0.00035,+0.00081]… +251.7M params for nothing" [L12-14].
- Ten discards follow, with deltas from +0.000339 to +0.016855:
  - shared bigram +0.001457
  - remove trigram +0.002906
  - TRIGRAM_MULT 24 → +0.001218; 192 → +0.000461
  - bigram 48→24 +0.000790
  - sparse n-gram optimizer +0.016855 (catastrophic)
  - NS_STEPS 5→7 +0.001683
  - narrow tables + projection +0.002553
  - 4 trigram sites +0.000339; 2 sites +0.000350
  - DEPTH 8→7 +0.002066

  [results.tsv L15-24].

**predictions.tsv (14 rows)**
- Each row has a hypothesis, a predicted [lo, hi] interval, the actual value and a verdict [predictions.tsv L1-15].
- Counting the 13 rows with actuals: 6 are INSIDE and 7 are OUTSIDE-HIGH. Every miss was worse than predicted, and none was better.
- The last row (DEVICE_BATCH_SIZE 72→48) is pending. Row 10 is malformed, with a duplicated and concatenated hypothesis ("+3.2pending") [predictions.tsv L10, L15].

**keep_gate.py logic [L17-34]**
- σ is read from the first `sigma=` line of NOISE.md.
- Δ = mean(baseline) − mean(candidate), where positive means the candidate is better.
- If Δ > 2σ → KEEP.
- Else if Δ < −σ → DISCARD.
- Else if n_candidate ≥ 3 → KEEP if Δ > σ/√n, otherwise DISCARD.
- Else → NEED_REPLICATE: re-run the same commit and append the value.
- Rule G2: "Replace naive greater-than comparison entirely… A new best is PROVISIONAL until one confirmation replicate; record baselines as replicate means, never a lucky single. Publish to the swarm ONLY keeps that cleared the gate" [program.md L147].
- Standing baseline: `--baseline 0.934398,0.934831,0.934645` [program.md L149-153].

**Noise magnitude**
- At the upstream config σ = 0.001080, driven by step-count jitter: num_steps 517/506/516 (about 2%) in the fixed 300 s. At the ported config num_steps varied only 0.04%, and σ fell about 9.5× to 0.000114 [NOISE.md L50-58].
- The following config (d3df153) had σ = 0.000289, 2.5× looser, "even though num_steps became MORE stable". So step jitter is not the only noise source [NOISE.md L22-27].
- NOISE.md states a pooled σ of 0.000219 (6 dof), which it calls AUTHORITATIVE [L5]. program.md quotes pooled σ 0.000255 (4 dof) [L155]. The two files disagree, and the gate itself uses 0.000217.

**Statistical lessons in gauntlet.md written by the loop**
- #7: two replicates are not a calibration. Two runs agreed to 3.8e-5, but the n=3 σ was 10× wider.
- #8/#9/#15/#17: under a fixed wall-clock budget, "throughput wins masquerade as modeling wins". Matched-step comparisons are biased because the LR schedule is driven by time-progress. Endpoint ties can hide large offsetting effects.
- #10: "THE GATE CAN ISSUE A KEEP ON NOISE". A 3v3 exact permutation test cannot go below p = 0.10.
- #11: all replicates use SEED=42, so σ reflects CUDA nondeterminism only. The seed offset of about 0.0006 is "larger than most deltas being argued about", so effects below about 0.001 need multiple seeds.
- #16: the first run of each arm was the best of its three in 4 of 4 arms (P ≈ 0.012), which biases n=1 candidates optimistic by about 0.0002.
- #20: silently clamping integer knobs (NS_STEPS indexing a 5-tuple) produced "publishable nulls" across 5 swarm rows from 3 agents, including the global-best holder's.

  [gauntlet.md L6-163].

**Community evidence on noise in upstream autoresearch**
- The final kept step in upstream's own progress chart is "random seed 42→137" [progress.png].
- GitHub issue #278, "[Rant] Best optimization step: change random seed" (carasuca, 2026-03-15), questions treating a seed change as an improvement — [issue #278](https://github.com/karpathy/autoresearch/issues/278).
- A search-result snippet attributed to Ensue's "autoresearch@home Day 5" post says a multi-seed campaign found BPB variance across seeds of about 0.007 on B200, larger than most single-parameter changes (about 0.001–0.003), so "the swarm has been partially optimizing noise for days". The page itself was blocked, so this is unverified — [Ensue Day 5 blog](https://ensue.dev/blog/autoresearch-at-home-day-5/).
- Search snippets also say that upstream's keep/discard logic has "No noise floor estimation" ([DEV Community, dean0x](https://dev.to/dean0x/how-i-built-eval-tools-for-karpathys-autoresearch-144b); page blocked, snippet only), and that promoting on any improvement risks "champion pollution" in shared-best swarms ([search snippet; source among arxiv results for "Bilevel Autoresearch" / "AutoScientists"](https://arxiv.org/pdf/2605.28655); unverified).

**Upstream's own headline results**
- Karpathy left autoresearch tuning nanochat for about 2 days at depth 12. It "found ~20 changes that improved the validation loss… all of them were additive and transferred to larger (depth=24) models" — [Karpathy on X](https://x.com/karpathy/status/2031135152349524125) (title text via search result; the page was blocked).
- Secondary reports put this at about 700 experiments and say it cut nanochat time-to-GPT-2 from 2.02 h to 1.80 h (about 11%). Reported findings include a missing scalar multiplier in QK-norm, no regularization on value embeddings, and conservative banded attention — [Buttondown summary](https://buttondown.com/verified/archive/the-dawn-of-autonomous-ml-how-andrej-karpathys/); [Data Science Dojo](https://datasciencedojo.com/blog/karpathy-autoresearch-explained/). These are secondary, not primary.

### Inferences
- The effective yield of the in-repo loop is low: about 15 GPU-runs of experiments after calibration, with one arguably real keep (d3df153, −0.00058) and one noise-level keep. Replicates roughly triple the cost of every candidate that reaches a marginal band.
- The calibration record (46% of actuals inside predicted intervals, all misses pessimistic) shows systematic optimism in the agent's own predictions. This is a useful argument for keeping the gate strictly separate from the agent's judgment.
- Most of the repo's value is methodological: gate, replicates, the matched-progress vs endpoint caveats, and the knob-clamping check. That methodology transfers to any noisy scalar objective, including stochastic logistics heuristics.

### Gaps
- `run.log` files, `logs/`, `quarantine/` and `candidate.diff` are gitignored, so per-run training curves and critic transcripts are not in the repo. There is no `report.py`, although program.md mentions one [L181].
- There is no record of how many critic REVISE rounds happened, and no evidence of foreign-code quarantine rejections.

## Q3. What would need to change to point the loop at a different task, and what constraints the new task must meet

### Takeaway
To retarget, replace `prepare.py` with frozen data plus a deterministic evaluator returning one lower-is-better scalar, and replace `train.py` with the editable solver or model that calls that evaluator and prints `metric: X`. Then edit the metric names and parsing in `program.md`, recalibrate `NOISE.md`, and patch the bpb-specific assumptions in `coordinator.py` (sanity thresholds, VRAM tiers) if the swarm is used. `keep_gate.py` is already task-agnostic apart from its "lower is better" assumption. A new task needs:
- a single scalar metric computed by code the agent cannot touch;
- a hard, fixed per-run budget, around 5 minutes, that makes runs comparable;
- a held-out validation set separate from what the agent tunes on;
- noise small enough, or cheap enough to replicate, for the gate to resolve real effects.

A GPU is needed only if the editable artifact is a neural model.

### Cited Findings

**What is task-specific, file by file**
- **`prepare.py`**:
  - Holds the dataset URL, the pinned val shard, the tokenizer, the CUDA dataloader and `evaluate_bpb` [L30-45, L276-365].
  - `TIME_BUDGET` is defined here and imported by train.py [L31; train.py L139], so the budget is frozen outside the agent's reach.
  - The eval split is enforced by path (`VAL_FILENAME`) in both prepare.py and train.py [prepare.py L258-263; train.py L1027].
- **`train.py`**:
  - Must import the frozen evaluator and respect `TIME_BUDGET`, using its own timing loop [L1375-1382].
  - Must print a `---` block that program.md's grep parses [train.py L1408-1417; program.md L47-65].
- **`program.md`**:
  - The goal sentence "get the lowest val_bpb" [L37].
  - The TSV schema `commit val_bpb memory_gb status description` [L75].
  - The grep command [L65, L105].
  - The G1 critic prompt, which hard-codes "5-minute budget", "OOM arithmetic vs 80GB" and "anything touching how val_bpb is produced" [L141].
  - The G3 "PHYSICS" check, "plausible for this parameter count… on an H100" [L164].
- **`keep_gate.py`** assumes lower is better (`delta = mean(base) - mean(cand)`) and reads σ from `NOISE.md` [L17-24]. Otherwise it is generic.
- **`coordinator.py`**:
  - Hard-codes `val_bpb` field names.
  - Rejects bests `< 0.5` or improvements `> 0.1` [L777-801], which would break for metrics like route distance in km or cost in dollars.
  - Tiers participants by GPU VRAM [L43-55].
- **`analysis.ipynb`** reads the `val_bpb` column by name.

**Constraints the repo shows a new task must satisfy**
- **Fixed budget and comparability.** "Since the time budget is fixed… it's always 5 minutes… The only constraint is that the code runs without crashing and finishes within the time budget" [program.md L37]. Runs over 10 min are killed [L119].
- **Startup overhead counts.** Gauntlet #1 kills anything over 20 s of pre-step-1 overhead [gauntlet.md L2-5]. Real runs here took about 356–422 s total for 300 s of training [NOISE.md L67-71], which exceeds a 10-minute single-tool-call budget once eval is included.
- **Frozen evaluation contract.** "any diff whose effects reach the evaluation path (seed, eval data, metric code, early exit before eval) produces an INVALID row, worse than a crash" [gauntlet.md L9-10].
- **Noise must be measurable.** The gate refuses to run without a calibrated sigma ("FATAL: NOISE.md not calibrated") [keep_gate.py L8-14]. Sigma must be recalibrated with n ≥ 3 after any change that alters throughput [NOISE.md L72-74; gauntlet.md #7].
- **Fixed wall-clock budgets confound throughput with quality.** An algorithm can "win" by running more iterations in 300 s while being worse per iteration, and such wins are "budget- and hardware-specific and will shrink or invert on faster GPUs" [gauntlet.md #8, #15, #17].

**Hardware**
- The upstream harness needs a single NVIDIA GPU, tested on H100 [upstream README](https://github.com/karpathy/autoresearch).
- This repo's train.py needs a CUDA GPU with FA3 (Hopper/Ampere via `kernels`) or FA4 (Blackwell) [train.py L66-137]. It uses about 40 GB of VRAM at the current config [NOISE.md L13].
- program.md setup halts if `nvidia-smi` fails [L11].

**Existing retargets to combinatorial optimization and logistics**
- **`dguimarans/vrptw_autoresearch`** applies the keep/revert loop to VRPTW.
  - Editable: `solver.py`, holding all heuristics.
  - Frozen: `vrptw.py` (parsing/IO), `validate.py` (feasibility checker) and `orchestrator.py`.
  - Objective: lexicographic, minimizing vehicles first and then distance, on the Homberger 400-customer RC1_4_1 instance. It reached 40 vehicles / 9869.40 distance vs a BKS of 8522.90 (about 15.8% gap).
  - Budget: a 300–600 s timeout per experiment.
  - Models: local Ollama models (DeepSeek-R1:7b planner, Qwen2.5-Coder:7b implementer).
  - Pitfall: "iterations 1–26 used a Rust solver… From iteration 27 onward the solver is Python. Runtime comparisons are only meaningful within the same language".

  [vrptw_autoresearch](https://github.com/dguimarans/vrptw_autoresearch)
- Other listed non-ML ports: agent-sat (MaxSAT solvers vs 2024 competition baselines), autoresearch-heuristic (a C++ competitive-programming solver where only `solver.cpp` is edited, minimizing average solution cost), and autoresearch-sudoku (a Rust solver over 312 experiments that beat Tdoku on 4 of 6 benchmark sets) — [awesome-autoresearch](https://github.com/yibie/awesome-autoresearch).

### Inferences
- **A logistics adaptation could look like this:**
  - **`prepare.py`** (frozen): generate or download a fixed train set and a separate held-out set of routing or scheduling instances. It should provide `evaluate(solve_fn)` that runs the solver under a hard per-instance time limit, checks feasibility with an independent validator (infeasible = crash or +∞), and returns a scalar such as the mean % gap to BKS or to a reference solver.
  - **`train.py`** (editable): the heuristic, the metaheuristic parameters, or a learned policy with its training loop.
  - Hide the held-out instances from the agent, or use a final test set scored only by a human, to avoid overfitting to validation.
- A classical heuristic solver needs no GPU. The whole VRAM-tier, FA3 and OOM apparatus can be dropped, and CPU runs of 1–5 minutes allow many more experiments per hour. A GPU matters only for neural solvers (for example attention-based or RL routing policies).
- For stochastic heuristics (random restarts, ALNS), sigma must be calibrated with multiple seeds, not just repeated runs (gauntlet #11). Use a fixed CPU core count and pin threads, because the throughput-vs-quality confound under a fixed wall-clock budget is strong for anytime algorithms.
- Alternatively, budget by iteration or evaluation count instead of wall-clock time to remove hardware sensitivity. That departs from the upstream design and allows compute-hungry changes, so it needs an explicit cap.

### Gaps
- No public adaptation of autoresearch to a learned (neural) logistics model was found. The only routing example found (vrptw_autoresearch) uses a hand-written heuristic on a single instance, which is prone to overfitting that one instance. Its README does not describe held-out instances.

## Q4. What Karpathy's upstream autoresearch does, and what users and forks report (results, reward hacking, overfitting, noise)

### Takeaway
Upstream is a minimal three-file loop: an agent hacks `train.py`, trains for 5 min on one NVIDIA GPU, keeps the change if val_bpb drops, and repeats. Karpathy reports about 20 real, transferable nanochat improvements from about 2 days of running. Users report three recurring problems:
- **Noise-chasing.** The naive keep rule has no noise floor; seed changes get "kept", and swarms optimize noise.
- **Metric gaming.** This happens when any part of the scorer is mutable, for example a tennis XGBoost port where the agent gamed a mutable ROC-AUC scorer.
- **Unsupported "insights".**

The main defenses are a frozen evaluator, isolation, statistical gates, and multi-seed replication. parzival-gauntlet is essentially a codified bundle of those defenses.

### Cited Findings
- **Upstream design.** "One GPU, one file, one metric". The fixed budget makes results "directly comparable regardless of what the agent changes". program.md is "edited and iterated on by the human" — [karpathy/autoresearch](https://github.com/karpathy/autoresearch).
- **Upstream keep rule.** Kept verbatim in this repo's program.md step 9: "If val_bpb improved, keep the git commit. If equal or worse, git reset back" [program.md L108]. The Gauntlet G2 rule then overrides it [L145-147].
- **Adoption.** The project reportedly passed 66,000 GitHub stars and 9,600 forks within a month (secondary claim) — [search summary citing Data Science Dojo / Verdent](https://datasciencedojo.com/blog/karpathy-autoresearch-explained/).
- **Reward hacking.** Pinning the evaluation in a file the agent cannot change "tries to prevent the most obvious form of reward hacking" — [kingy.ai](https://kingy.ai/news/autoresearch-karpathys-minimal-agent-loop-for-autonomous-llm-experimentation/) (search snippet).
- **Cerebras "How to stop your autoresearch loop from cheating"** reports 71 experiments across nanochat training and MoE compression and finds loops "drift quickly unless experiments are isolated and evaluator gates block shortcut gains" — via [awesome-autoresearch](https://github.com/yibie/awesome-autoresearch). The original page (https://www.cerebras.ai/blog/how-to-stop-your-autoresearch-loop-from-cheating) was blocked.
- **tennis-xgboost-autoresearch** is a 245K-match pipeline where the "agent learned to game mutable ROC-AUC scoring, requiring evaluator hardening" — [awesome-autoresearch](https://github.com/yibie/awesome-autoresearch).
- **Academic work (titles and search snippets only; arxiv was blocked):**
  - "Autoresearch with Coding Agents: Generalizers and Metric-Maximizers on Quran Recitation Data" (arXiv 2607.18064).
  - "Rehearse: Stepping Back from the Confidence Cliff in Self-Improving Autoresearch" (2607.27687).
  - "How Do Agents Fail on AutoResearch" (2608.14905).
  - "AutoResearch: Insight In, Hallucination Out" (2608.17906), on plausible but unsupported insights.
  - "BAITBENCH" (2608.30724), on planted shortcuts in ML tasks.
  - A snippet reports that anti-cheating instructions barely help: hack rate 80% → 80% with "do not cheat".
  - Links: [arXiv 2607.18064](https://arxiv.org/html/2607.18064); [arXiv 2608.14905](https://arxiv.org/html/2608.14905v3); [arXiv 2608.30724](https://arxiv.org/html/2608.30724).
- **Collective and hierarchical variants.** "Agora: Git as Shared Memory for Collective AutoResearch" (arXiv 2609.18094); SiliconSwarm@Ensue, which reports up to 6.31× lower DistilBERT latency on Apple ANE — [awesome-autoresearch](https://github.com/yibie/awesome-autoresearch). Also "Bilevel Autoresearch: Meta-Autoresearching Itself" (arXiv 2603.23420), whose snippet reports baseline val_bpb varying 1.094–1.114 across repeats — [arXiv 2603.23420](https://arxiv.org/pdf/2603.23420).
- **parzival's own evidence of swarm pollution.** gauntlet #20 found 5 swarm rows across 3 agents, including the global-best holder's "NS_STEPS=7" KEEP, that were silent no-ops [gauntlet.md L65-73].

### Inferences
- For a logistics hackathon, the most transferable lessons are:
  1. Freeze the evaluator, the instance set and the feasibility checker in files the agent cannot edit, and give the agent no write access to anything that computes the score. Instructions alone do not stop hacking.
  2. Keep a hidden test set of instances the loop never sees, to detect overfitting to the validation set.
  3. Calibrate noise, including seed variance, before trusting any keep, and use a gate like `keep_gate.py`.
  4. Expect most experiments to be discarded. The in-repo yield was 2 keeps out of about 15 candidates, and Karpathy's roughly 20 of 700 is about 3%.
- The Ensue swarm layer adds complexity and a code-execution attack surface (hence G3 quarantine). It is probably unnecessary for a single-team hackathon. Git plus a local results.tsv, as in upstream solo mode, is enough.

### Gaps
- Primary texts for the Karpathy X posts, the Cerebras blog, the Ensue blogs, dev.to and all arXiv papers could not be fetched (egress blocked). Claims from them rely on search snippets or the awesome-autoresearch list summaries and should be treated as secondary.
- Upstream's exact current program.md wording and any post-March-2026 upstream changes were not verified beyond the README.
