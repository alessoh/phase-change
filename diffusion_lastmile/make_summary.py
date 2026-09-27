"""Write results/summary.md from the saved per-route evaluation files.

Reads results/per_route_test.json (300 official evaluation routes) and
results/per_route_heldout.json (150 held-out High-quality training-distribution
routes), reports means, medians and paired bootstrap 95% confidence intervals.
Methods with several seeds report the seed-0 run in the main table and the
seed spread separately.
"""
import json
import random
import statistics as st
from pathlib import Path

RESULTS = Path(__file__).parent / "results"

LABELS = {
    "driver": "Driver's actual sequence (reference)",
    "nn": "Nearest neighbour on travel times",
    "ortools": "OR-Tools TSP, 5 s, guided local search",
    "softdist": "SoftDist non-learned heatmap + 2-opt",
    "diffusion": "Stop-level diffusion + 2-opt",
    "sup": "Stop-level supervised GNN (no diffusion) + 2-opt",
    "zone": "Zone-change penalty + OR-Tools",
    "zonehist": "Historical zone order (no model) + OR-Tools",
    "hier": "Zone-order diffusion + OR-Tools",
    "hier_sup": "Zone-order supervised GNN (no diffusion) + OR-Tools",
}
PAIRS = [
    ("diffusion", "softdist"),
    ("sup", "diffusion"),
    ("hier", "zonehist"),
    ("hier_sup", "hier"),
    ("hier", "ortools"),
]


def boot_ci(diffs, reps=2000, seed=0):
    rng = random.Random(seed)
    means = sorted(st.mean(rng.choices(diffs, k=len(diffs))) for _ in range(reps))
    return st.mean(diffs), means[int(0.025 * reps)], means[int(0.975 * reps) - 1]


def table(split):
    data = json.loads((RESULTS / f"per_route_{split}.json").read_text())
    routes = data["per_route"]
    lines = [
        "| Method | Valid | Amazon score mean [95% CI] | Median | Travel time mean (h) | Median runtime (s) |",
        "|---|---|---|---|---|---|",
    ]
    for key, label in LABELS.items():
        if key not in routes[0]:
            continue
        rows = [r[key] for r in routes]
        scores = [x["amazon_score"] for x in rows]
        mean, lo, hi = boot_ci(scores) if key != "driver" else (0.0, 0.0, 0.0)
        lines.append(
            f"| {label} | {sum(x['valid'] for x in rows)}/{len(rows)} | {mean:.4f} [{lo:.4f}, {hi:.4f}] | "
            f"{st.median(scores):.4f} | {st.mean(x['travel_time_s'] for x in rows) / 3600:.2f} | "
            f"{st.median(x['runtime_s'] for x in rows):.2f} |"
        )
    lines += ["", "| Paired comparison (score A minus B, negative favours A) | Mean | 95% CI |", "|---|---|---|"]
    for a, b in PAIRS:
        mean, lo, hi = boot_ci([r[a]["amazon_score"] - r[b]["amazon_score"] for r in routes])
        lines.append(f"| {LABELS[a]} vs {LABELS[b]} | {mean:+.4f} | [{lo:+.4f}, {hi:+.4f}] |")
    seeds = []
    for key in ("diffusion", "sup", "hier", "hier_sup"):
        vals = [st.mean(r[k]["amazon_score"] for r in routes) for k in (key, f"{key}@1", f"{key}@2") if k in routes[0]]
        if len(vals) > 1:
            seeds.append(f"{LABELS[key]}: " + ", ".join(f"{v:.4f}" for v in vals))
    if seeds:
        lines += ["", "Mean score across training seeds 0, 1 and 2: " + "; ".join(seeds) + "."]
    return len(routes), "\n".join(lines)


def main():
    n_test, t_test = table("test")
    n_held, t_held = table("heldout")
    text = f"""# Results summary

Amazon Last Mile route score: lower is better, and the driver's own sequence scores 0. It measures how closely a
generated stop sequence matches what the driver actually did, not travel time. Travel times use Amazon's released
travel-time matrices. Confidence intervals are paired bootstrap intervals over routes (2,000 resamples). Runtimes are
wall-clock on a shared 4-core CPU and are indicative only.

## {n_test} routes from the official evaluation dataset (primary test, route quality unknown)

{t_test}

## {n_held} held-out High-quality routes (secondary test, same distribution as training)

{t_held}

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
"""
    (RESULTS / "summary.md").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
