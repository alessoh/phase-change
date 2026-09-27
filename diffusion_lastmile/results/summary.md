# Results summary

Amazon Last Mile route score: lower is better, and the driver's own sequence scores 0. It measures how closely a
generated stop sequence matches what the driver actually did, not travel time. Travel times use Amazon's released
travel-time matrices. Confidence intervals are paired bootstrap intervals over routes (2,000 resamples). Runtimes are
wall-clock on a shared 4-core CPU and are indicative only.

## 300 routes from the official evaluation dataset (primary test, route quality unknown)

| Method | Valid | Amazon score mean [95% CI] | Median | Travel time mean (h) | Median runtime (s) |
|---|---|---|---|---|---|
| Driver's actual sequence (reference) | 300/300 | 0.0000 [0.0000, 0.0000] | 0.0000 | 3.31 | 0.00 |
| Nearest neighbour on travel times | 300/300 | 0.1058 [0.0989, 0.1130] | 0.0915 | 3.39 | 0.00 |
| OR-Tools TSP, 5 s, guided local search | 300/300 | 0.0880 [0.0821, 0.0941] | 0.0719 | 3.00 | 5.00 |
| SoftDist non-learned heatmap + 2-opt | 300/300 | 0.0874 [0.0818, 0.0931] | 0.0758 | 3.07 | 0.03 |
| Stop-level diffusion + 2-opt | 300/300 | 0.0778 [0.0728, 0.0832] | 0.0675 | 3.16 | 1.07 |
| Stop-level supervised GNN (no diffusion) + 2-opt | 300/300 | 0.0741 [0.0693, 0.0793] | 0.0674 | 3.14 | 0.06 |
| Zone-change penalty + OR-Tools | 300/300 | 0.0537 [0.0499, 0.0575] | 0.0484 | 3.12 | 5.01 |
| Historical zone order (no model) + OR-Tools | 300/300 | 0.0537 [0.0492, 0.0584] | 0.0470 | 3.29 | 5.01 |
| Zone-order diffusion + OR-Tools | 300/300 | 0.0429 [0.0382, 0.0477] | 0.0299 | 3.24 | 6.31 |
| Zone-order supervised GNN (no diffusion) + OR-Tools | 300/300 | 0.0350 [0.0308, 0.0393] | 0.0210 | 3.19 | 5.33 |

| Paired comparison (score A minus B, negative favours A) | Mean | 95% CI |
|---|---|---|
| Stop-level diffusion + 2-opt vs SoftDist non-learned heatmap + 2-opt | -0.0096 | [-0.0135, -0.0054] |
| Stop-level supervised GNN (no diffusion) + 2-opt vs Stop-level diffusion + 2-opt | -0.0037 | [-0.0071, -0.0001] |
| Zone-order diffusion + OR-Tools vs Historical zone order (no model) + OR-Tools | -0.0107 | [-0.0143, -0.0074] |
| Zone-order supervised GNN (no diffusion) + OR-Tools vs Zone-order diffusion + OR-Tools | -0.0080 | [-0.0106, -0.0053] |
| Zone-order diffusion + OR-Tools vs OR-Tools TSP, 5 s, guided local search | -0.0450 | [-0.0498, -0.0405] |

Mean score across training seeds 0, 1 and 2: Stop-level diffusion + 2-opt: 0.0778, 0.0809, 0.0811; Stop-level supervised GNN (no diffusion) + 2-opt: 0.0741, 0.0709, 0.0710; Zone-order diffusion + OR-Tools: 0.0429, 0.0432, 0.0399; Zone-order supervised GNN (no diffusion) + OR-Tools: 0.0350, 0.0352, 0.0345.

## 150 held-out High-quality routes (secondary test, same distribution as training)

| Method | Valid | Amazon score mean [95% CI] | Median | Travel time mean (h) | Median runtime (s) |
|---|---|---|---|---|---|
| Driver's actual sequence (reference) | 150/150 | 0.0000 [0.0000, 0.0000] | 0.0000 | 3.27 | 0.00 |
| Nearest neighbour on travel times | 150/150 | 0.1104 [0.1005, 0.1216] | 0.0939 | 3.37 | 0.00 |
| OR-Tools TSP, 5 s, guided local search | 150/150 | 0.0898 [0.0813, 0.0994] | 0.0754 | 2.97 | 5.00 |
| SoftDist non-learned heatmap + 2-opt | 150/150 | 0.0962 [0.0864, 0.1075] | 0.0763 | 3.05 | 0.03 |
| Stop-level diffusion + 2-opt | 150/150 | 0.0806 [0.0735, 0.0884] | 0.0714 | 3.13 | 1.17 |
| Stop-level supervised GNN (no diffusion) + 2-opt | 150/150 | 0.0787 [0.0704, 0.0893] | 0.0630 | 3.09 | 0.06 |
| Zone-change penalty + OR-Tools | 150/150 | 0.0601 [0.0538, 0.0672] | 0.0503 | 3.06 | 5.01 |
| Historical zone order (no model) + OR-Tools | 150/150 | 0.0467 [0.0403, 0.0538] | 0.0397 | 3.20 | 5.01 |
| Zone-order diffusion + OR-Tools | 150/150 | 0.0430 [0.0364, 0.0503] | 0.0311 | 3.18 | 6.48 |
| Zone-order supervised GNN (no diffusion) + OR-Tools | 150/150 | 0.0328 [0.0273, 0.0385] | 0.0201 | 3.13 | 5.33 |

| Paired comparison (score A minus B, negative favours A) | Mean | 95% CI |
|---|---|---|
| Stop-level diffusion + 2-opt vs SoftDist non-learned heatmap + 2-opt | -0.0156 | [-0.0228, -0.0090] |
| Stop-level supervised GNN (no diffusion) + 2-opt vs Stop-level diffusion + 2-opt | -0.0018 | [-0.0081, +0.0054] |
| Zone-order diffusion + OR-Tools vs Historical zone order (no model) + OR-Tools | -0.0037 | [-0.0088, +0.0013] |
| Zone-order supervised GNN (no diffusion) + OR-Tools vs Zone-order diffusion + OR-Tools | -0.0102 | [-0.0144, -0.0065] |
| Zone-order diffusion + OR-Tools vs OR-Tools TSP, 5 s, guided local search | -0.0468 | [-0.0535, -0.0403] |

Mean score across training seeds 0, 1 and 2: Stop-level diffusion + 2-opt: 0.0806, 0.0868, 0.0807; Stop-level supervised GNN (no diffusion) + 2-opt: 0.0787, 0.0782, 0.0797; Zone-order diffusion + OR-Tools: 0.0430, 0.0407, 0.0372; Zone-order supervised GNN (no diffusion) + OR-Tools: 0.0328, 0.0337, 0.0330.

## What the numbers show

Every method produced a feasible sequence on every route. Learning from past High-quality routes clearly helps:
all learned methods match drivers better than nearest neighbour, OR-Tools and the non-learned SoftDist control.
The best results come from predicting the order of delivery zones and letting OR-Tools sequence stops inside it.
However, the diffusion process itself is not what delivers the gain. A supervised graph network of the same
architecture, trained without diffusion, matched or beat the diffusion model at both the stop level and the zone
level, and it runs faster. On the held-out High routes the zone-order diffusion model's advantage over simply
reusing historical zone order is not statistically significant. The learned methods trade some travel time for
driver-likeness: OR-Tools finds the shortest tours, drivers' own tours are longer, and the zone-order hybrids sit
in between.

These results come from small CPU-trained models (about 0.3 to 1.2 million parameters) on a random subset of the
evaluation data and are not a leaderboard submission.
