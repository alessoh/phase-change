# Diffusion models for last-mile delivery sequencing (Amazon Last Mile data)

This folder is a research prototype that tests one idea on real data: a diffusion model trained only on routes that experienced drivers executed well should be able to generate a sensible delivery sequence for a new, unseen route, without anyone writing down the rules that drivers follow. The model is compared side by side with classical operations-research methods and with a non-learned control on the same routes, using the real travel times and the official scoring rule of the 2021 Amazon Last Mile Routing Research Challenge.

No synthetic or fictitious data is used anywhere. Every route, stop, zone, travel time and driver sequence comes from the public challenge dataset.

## Headline results

The short answer is that the idea works, but only at the right level of abstraction.

The stop-level diffusion model that the study set out to build (DIFUSCO-style Bernoulli diffusion over stop-to-stop edges, greedy decoding, 2-opt) learned mostly local structure. On the 300 evaluation-dataset routes its sequences were closer to the drivers' than OR-Tools, nearest neighbour and the non-learned SoftDist control (mean score 0.0763 against 0.0881 for OR-Tools, paired difference -0.0117 with 95% CI [-0.0158, -0.0078]), but clearly worse than a simple zone-change heuristic (0.0523).

Moving the same diffusion model one level up, to generating the order in which the route's zones are served and letting OR-Tools fill in the stops (`M_diffusion_zone`), gave the most driver-like sequences of every method on both test sets: mean score 0.0409 (median 0.0277) on the evaluation routes and 0.0389 on the held-out High routes. On the evaluation routes it beat the zone-change heuristic on 69% of routes (paired difference -0.0114, CI [-0.0150, -0.0078]), plain OR-Tools on 92% (-0.0472, CI [-0.0516, -0.0429]), and the non-learned control that decodes the same historical zone statistics without the model by -0.0126 (CI [-0.0160, -0.0092]; better on 58%, identical on 21%). Its tours are on average 14.2 minutes longer than OR-Tools' but 4.5 minutes shorter than the drivers' own. All 10 methods produced feasible sequences on every route. Full tables with paired tests are in `results/summary.md`; per-route numbers are in `results/results.json`; a Word version of the write-up is `results/summary.docx`.

**300 routes of the official evaluation dataset (primary test).** Amazon score: lower is better, driver = 0. Brackets are bootstrap 95% confidence intervals of the mean.

| Method | Amazon score, mean [95% CI] | Median | Travel time (h) | Driver edges | Zone tau | Runtime (s) |
|---|---|---|---|---|---|---|
| **M_diffusion_zone (zone diffusion + OR-Tools)** | 0.0409 [0.0365, 0.0455] | 0.0277 | 3.23 | 40.5% | 0.47 | 7.69 |
| M_zonehist (history zone order + OR-Tools) | 0.0535 [0.0492, 0.0579] | 0.0466 | 3.29 | 39.1% | 0.26 | 5.01 |
| M_zone (zone-change penalty, 5 s) | 0.0523 [0.0491, 0.0556] | 0.0470 | 3.09 | 34.8% | 0.04 | 5.01 |
| M_diffusion greedy + 2-opt | 0.0763 [0.0713, 0.0817] | 0.0692 | 3.16 | 32.4% | 0.04 | 1.28 |
| M_diffusion greedy | 0.0856 [0.0811, 0.0903] | 0.0776 | 3.67 | 28.6% | 0.02 | 1.25 |
| M_softdist greedy + 2-opt | 0.0874 [0.0822, 0.0930] | 0.0758 | 3.07 | 31.5% | 0.04 | 0.03 |
| M_softdist greedy | 0.0967 [0.0914, 0.1022] | 0.0830 | 3.30 | 29.7% | 0.05 | 0.00 |
| M_ortools (GLS, 5 s) | 0.0881 [0.0823, 0.0942] | 0.0727 | 3.00 | 30.9% | 0.03 | 5.00 |
| M_nn | 0.1058 [0.0992, 0.1130] | 0.0915 | 3.39 | 28.2% | 0.01 | 0.00 |
| M_driver (actual driver sequence) | 0.0000 [0.0000, 0.0000] | 0.0000 | 3.31 | 100.0% | 1.00 | 0.00 |

**150 held-out High-quality routes (secondary test).** Amazon score: lower is better, driver = 0. Brackets are bootstrap 95% confidence intervals of the mean.

| Method | Amazon score, mean [95% CI] | Median | Travel time (h) | Driver edges | Zone tau | Runtime (s) |
|---|---|---|---|---|---|---|
| **M_diffusion_zone (zone diffusion + OR-Tools)** | 0.0389 [0.0332, 0.0448] | 0.0312 | 3.16 | 41.9% | 0.53 | 10.48 |
| M_zonehist (history zone order + OR-Tools) | 0.0461 [0.0398, 0.0527] | 0.0388 | 3.19 | 41.3% | 0.45 | 5.01 |
| M_zone (zone-change penalty, 5 s) | 0.0644 [0.0581, 0.0715] | 0.0559 | 3.05 | 35.1% | -0.02 | 5.01 |
| M_diffusion greedy + 2-opt | 0.0813 [0.0731, 0.0902] | 0.0702 | 3.12 | 32.3% | 0.01 | 2.73 |
| M_diffusion greedy | 0.0911 [0.0838, 0.0990] | 0.0779 | 3.61 | 28.6% | 0.01 | 2.70 |
| M_softdist greedy + 2-opt | 0.0962 [0.0864, 0.1070] | 0.0763 | 3.05 | 31.9% | 0.03 | 0.03 |
| M_softdist greedy | 0.1049 [0.0949, 0.1161] | 0.0917 | 3.27 | 29.9% | 0.02 | 0.00 |
| M_ortools (GLS, 5 s) | 0.0893 [0.0811, 0.0983] | 0.0754 | 2.97 | 31.5% | -0.04 | 5.00 |
| M_nn | 0.1104 [0.1005, 0.1211] | 0.0939 | 3.37 | 27.1% | -0.04 | 0.00 |
| M_driver (actual driver sequence) | 0.0000 [0.0000, 0.0000] | 0.0000 | 3.27 | 100.0% | 1.00 | 0.00 |

Figures in `results/figures/`: `route_*.png` show one route solved by every method (routes picked deterministically at the 25th, 50th and 75th percentile of the zone-diffusion score, not cherry-picked), `blind_*.png` show six sequences for the same route with shuffled anonymous labels (answer key in `blind_key.json`), `heatmap_*.png` compares the learned stop-level edge probabilities with SoftDist, `zone_orders_*.png` compares the driver's, the generated and the history-only zone tours, `score_distributions.png` and `paired_diffusion_vs_ortools.png` show per-route distributions, and `training_curves.png` shows both training runs.

For context only: the winning team of the 2021 challenge (Cook, Held and Helsgaun; Transportation Science 2024, arXiv 2112.15192) is reported to have scored 0.0248 on the full evaluation set. I saw that number in search results but could not open the paper from this environment to verify it, and our 300-route subset is not a leaderboard submission.

## Data

The data is the Amazon Last Mile Routing Research Challenge dataset (2021), published on the AWS Open Data registry at `https://amazon-last-mile-challenges.s3.amazonaws.com/almrrc2021/` and readable without credentials. It is licensed under Creative Commons Attribution-NonCommercial 4.0 (see `data/raw/License.txt` after download); this prototype uses it for non-commercial research only and the raw files are not redistributed (`data/` is in `.gitignore`). Citation: Merchán, D., Arora, J., Pachon, J., Konduri, K., Winkenbach, M., Parks, S., Noszek, J. (2024). 2021 Amazon Last Mile Routing Research Challenge: Data Set. Transportation Science 58(1), 8 to 11.

The training dataset has 6,112 routes from 17 depots (Los Angeles, Seattle, Chicago, Boston, Austin). Amazon labelled each route's execution quality as High (2,718), Medium (3,292) or Low (102). Only the 2,718 High routes are used; they are the "past successes" the model learns from. They were split at random (seed 0) into 2,508 training routes, 60 validation routes (used for checkpoint monitoring and for choosing every hyper-parameter of every method) and 150 held-out routes that no model ever sees during training or tuning.

The primary test set is 300 routes drawn at random (seed 0) from the separate official evaluation dataset of 3,052 routes (`almrrc2021-data-evaluation`), whose driver sequences were released after the challenge. The evaluation dataset does not publish a route quality label, so these test routes are of mixed quality, unlike the training data. The 150 held-out High routes are reported as a second test set because they match the training distribution exactly. No route appears in more than one split (checked in `data.py`).

Travel times come from the dataset's `travel_times.json` files (1.8 GB and 0.8 GB). They are streamed with `ijson` and only the needed routes are kept. The matrices are asymmetric and in seconds. Package data (time windows, dimensions) is not used.

## Method

### Stop-level edge diffusion (the method specified for this study)

Each route is a graph whose node 0 is the depot and whose other nodes are the stops, listed in stop-id order so that node order carries no information about the answer. Edges are directed: every stop is linked to its 16 nearest stops by real travel time (and the reverse edges are added), and the depot is linked to every stop in both directions. On average 97.5% of the driver's tour edges lie inside this sparse graph (reported per split in `results/data_stats.json`).

Node features are the normalised stop coordinates, a depot flag, the rank of the stop's zone and super-zone among the route's zones, the zone's sub-letter and the route size. Edge features are the real travel times in both directions (scaled by the route's median), their logarithm, the k-nearest-neighbour rank, the planar distance, same-zone, same-super-zone and same-sub-zone flags, numeric differences between the zone-id components, the signed difference of zone ranks (a zone-order signal) and a depot flag.

The target is the directed adjacency matrix of the driver's executed closed tour (depot, stops in driver order, back to the depot). A DIFUSCO-style discrete diffusion process (Sun and Yang, NeurIPS 2023; the Bernoulli case of D3PM, Austin et al. 2021) corrupts this 0/1 edge vector by independent symmetric bit flips with a linear schedule of 1,000 steps. The denoiser is an anisotropic edge-gated graph network (Bresson and Laurent 2017; Joshi et al. 2019) with 12 layers and 64 hidden units, conditioned on the diffusion time step through an embedding added to every edge in every layer, and trained with binary cross-entropy to predict the clean edges from the noisy ones. Training used AdamW, a warm-up plus cosine learning-rate schedule, an exponential moving average of the weights, and random rotations and reflections of the coordinate features.

At inference the model starts from random edges and runs 10 DDIM-style skipping denoising steps with the exact Bernoulli posterior; four independent samples are averaged into an edge heat-map. The heat-map is decoded into a valid tour by greedy edge insertion adapted to directed edges (highest probability first, each stop gets one successor and one predecessor, no sub-tours; remaining fragments are joined by shortest travel time), and the tour is then improved by 2-opt on the asymmetric real travel-time matrix with the depot fixed at position 0. Results are reported with and without 2-opt.

### Zone-level edge diffusion (an extension added after diagnosing the stop-level model)

The stop-level model's most probable successor matched the driver's actual next stop 39.5% of the time on validation routes, against 38.9% for simply taking the nearest stop. In other words it learned mostly local structure. The same validation routes show where the driver's plan really lives: 85% of consecutive stops share a zone and about 96% of zones are served in one contiguous block. The global part of a driver's plan is the order of zones.

`zone_level.py` therefore applies the identical diffusion machinery (same noise process, same network code, same training script) to a much smaller graph whose nodes are the route's zones plus the depot, fully connected with directed edges. Zone features are centroids, zone-id ranks and sizes; edge features are mean and minimum real travel times between the zones, zone-id relations, and three history features computed only from the High-quality training routes of the same depot: how often drivers went from zone a to zone b, the reverse, and the log count. For training routes these counts are leave-one-out, so a route never sees its own transitions (unit-tested). The model (12 layers, 128 hidden units) generates the zone tour: 16 samples are averaged and the maximum-likelihood tour under the heat-map (the tour minimising the sum of minus log probabilities, found with OR-Tools on the roughly 20-node zone graph) is taken. The stop sequence is then completed by the same OR-Tools solver as the baselines, on the real travel-time matrix plus a penalty on every arc that leaves the generated zone order.

### Methods compared (every one outputs a complete sequence starting at the depot)

`M_driver` is the driver's executed sequence (the reference, score 0 by definition). `M_nn` is nearest neighbour from the depot on real travel times. `M_ortools` is Google OR-Tools routing (single-vehicle TSP from the depot) on the real asymmetric matrix with PATH_CHEAPEST_ARC and guided local search, 5 seconds per route. `M_zone` is the same solver on the real matrix plus a penalty on every arc that changes zone, which forces zones to be served contiguously but lets the solver pick their order (the kind of rule many challenge teams used). `M_softdist` is the non-learned SoftDist heat-map of Xia et al. (ICML 2024), a softmax of minus scaled travel time with a temperature, read on the same sparse graph and decoded by the same greedy insertion and 2-opt as the diffusion model; it is the control that tells whether the learned heat-map adds anything over distance. `M_zonehist` is the non-learned control for the zone-level model: the zone order is decoded greedily from the historical transition frequencies alone (the same statistics the zone model receives as input), then the same OR-Tools completion is used. `M_diffusion` (greedy, and greedy plus 2-opt) is the stop-level model, and `M_diffusion_zone` is the zone-level model.

Every tunable setting (SoftDist temperature, the zone penalties, the number of diffusion samples and denoising steps, the zone decoding rule) was chosen on the 60 validation routes by the mean Amazon score and then frozen; nothing was tuned on either test set. The grids and validation scores are in `results/tuning.json`.

### Metrics

Travel time is the closed tour from the depot through all stops and back, summed on the real travel-time matrix. The Amazon score is the official challenge metric: sequence deviation multiplied by the edit distance with real penalty (ERP) computed on z-score normalised travel times, divided by the number of ERP edits, all measured against the driver's sequence. Lower is better; the driver's own sequence scores exactly 0. It is re-implemented in `score.py` from the official scoring script (`https://github.com/MIT-CAVE/rc-cli/blob/main/scoring/score.py`), with the recursion rewritten bottom-up but with the same arithmetic order and tie-breaking. `tests/test_basic.py` checks that it returns bit-for-bit the same value as the official file on real routes. Note one property of the official metric that we reproduce faithfully: the sequence-deviation factor is invariant to reversing the whole sequence, so a perfectly reversed driver route scores 0.

We also report the share of the driver's directed stop-to-stop edges that a method reproduces, the Kendall rank correlation between the method's zone order and the driver's zone order, the runtime per route and feasibility (every stop exactly once, starting at the depot). Means come with bootstrap 95% confidence intervals (10,000 resamples of routes), and each diffusion method is compared with every other method route by route (mean paired difference with bootstrap confidence interval, share of routes where the diffusion method is better, and a Wilcoxon signed-rank p-value).

## Reproducing

```
pip install -r requirements.txt
python data.py download                 # about 3.2 GB into data/raw (plus the official scoring script)
python data.py preprocess --k 16        # data/processed/*.pkl, results/splits.json, results/data_stats.json
python zone_level.py build              # data/processed/zone_*.pkl
python tests/test_basic.py              # fast self-tests (a few seconds)
./run_training.sh                       # stop-level model, resumable 9-minute chunks, 7,000 steps
LEVEL=zone ./run_training.sh            # zone-level model, 2,500 steps
python evaluate.py tune                 # all settings chosen on the validation routes
python evaluate.py run --split test     # 300 evaluation-dataset routes
python evaluate.py run --split heldout  # 150 held-out High-quality routes
python evaluate.py report               # results/results.json and results/summary.md
python make_figures.py                  # results/figures
python make_report_docx.py              # results/summary.docx
```

The sandbox used here kills any command after 10 minutes, so `train.py` takes `--max-minutes` and resumes from its checkpoint (model, EMA weights, optimiser, RNG states and data order), and `run_training.sh` simply calls it until the planned number of steps is reached. OR-Tools solutions are cached in `results/cache` so evaluation can also be interrupted and resumed.

## Files

`data.py` downloads and preprocesses the data. `model.py` is the gated graph network denoiser. `diffusion.py` holds the Bernoulli noise schedule, training loss, posterior and sampler, and graph batching. `decode.py` has greedy edge insertion and asymmetric 2-opt. `baselines.py` has nearest neighbour, OR-Tools, the zone penalty and SoftDist. `zone_level.py` builds the zone graphs, the leave-one-out history features and the zone-order penalty. `score.py` computes travel time and the Amazon score. `train.py` trains either model. `evaluate.py` tunes, runs and reports. `make_figures.py` draws the figures and `make_report_docx.py` writes the Word summary. `tests/test_basic.py` is the self-test. `run_training.sh` wraps the chunked training. The trained weights (EMA) are kept in `results/models/` (the full resumable checkpoints live in the git-ignored `checkpoints/`); `evaluate.py --ckpt results/models/diffusion_ema.pt` loads the stop-level one directly, and the zone-level one can be copied to `checkpoints/zone_diffusion.pt`.

## Limitations and honest caveats

The stop-level diffusion model, which is the method this study was specified to test, does not beat the zone-aware heuristic; its learned heat-map is only slightly more informative than distance (top-1 successor accuracy 39.5% against 38.9% for the nearest stop on validation routes). The positive result comes from the zone-level extension, which was designed after that diagnosis, using the validation split only, and was then evaluated once on both test sets.

The zone-level method is a hybrid, not a pure generative solution: the diffusion model generates the zone order, and OR-Tools sequences the stops inside that order. Its advantage over the non-learned history control is statistically clear on both test sets but moderate, and on the held-out High routes the two methods return the identical sequence on 43% of routes (better on 41%, worse on 17%).

The evaluation dataset does not publish route-quality labels, so the 300 test routes mix good and ordinary executions; the held-out High routes are the like-for-like test. The test set is a random 300-route subset of the 3,052 evaluation routes, and the held-out set has 150 routes.

The Amazon score is reproduced exactly from the official script (bit-for-bit, tested), including its known property that the sequence-deviation factor ignores a full reversal of the route. A lower score means closer to what the driver did, which is not necessarily a better route.

Runtimes are wall-clock times on a 4-core machine that was shared with other jobs during the experiment. The held-out runtimes of the neural methods are inflated because the first 50 held-out routes ran while figures were being generated, and the zone model's reported training time (30.9 minutes) is inflated by contention; at the uncontended speed of about 0.38 s per step its 2,500 steps take about 16 minutes. OR-Tools methods always use the full 5 s limit, and their search quality may also have been slightly affected by contention.

Model selection used small grids on 60 validation routes (for example 8 settings for the zone-level method and 4 for the history control), so validation scores are slightly optimistic; test and held-out results are unaffected because nothing was tuned on them. Only one training seed per model was run for lack of compute, so run-to-run variance of training is not measured. Package data (time windows, sizes) was not used, and the models are small (0.3 M and 1.2 M parameters) and trained for about 64 and 16 CPU-minutes.
