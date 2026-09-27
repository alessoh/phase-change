"""Figures for the results: route maps, a blind side-by-side panel, heat-maps, score distributions,
between-seed spread and training curves.

    python make_figures.py            (after evaluate.py run/report)

Outputs in results/figures/:
    route_<k>_<route>.png        the same real route solved by every method (sequence order shaded
                                 light -> dark along the tour)
    blind_<k>.png                six of those sequences with anonymous labels A-F in random order;
                                 the key is in blind_key.json (look at the figure first)
    heatmap_<route>.png          stop-level edge probabilities: diffusion vs. the one-shot supervised
                                 ablation vs. the SoftDist control, driver's tour drawn on top
    zone_orders_<route>.png      driver zone order vs. zone diffusion vs. one-shot zone model vs. history
    seed_spread.png              mean Amazon score of every learned model for each training seed,
                                 next to the non-learned baselines (fresh, test and held-out)
    ablation_diffusion_vs_supervised.png  per-route score of each diffusion model against its
                                 one-shot supervised twin (seed-averaged), stop and zone level
    training_curves.png          training / validation loss of all 12 main runs (+ spec-budget run)
    score_distributions.png      per-route Amazon score and travel time by method
    paired_diffusion_vs_ortools.png  per-route Amazon score, diffusion methods vs. OR-Tools
Per-route figures use the FINAL (fresh) test split when it has been evaluated, else the reused
test split. All learned-model panels except seed_spread.png and the ablation figure use the
seed-0 checkpoints.
"""
from __future__ import annotations

import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from matplotlib.collections import LineCollection  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

import data  # noqa: E402
from baselines import softdist_heatmap  # noqa: E402
from evaluate import HIER_SAMPLES, HIER_STEPS, RES, SPLITS, heatmap, load_learned  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(RES, "figures")

# Reference palette (dataviz skill): sequential blue ramp, categorical slots 1 and 2, text tokens.
BLUE_RAMP = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
SEQ = LinearSegmentedColormap.from_list("seq_blue", BLUE_RAMP)
SERIES_1 = "#2a78d6"  # diffusion
SERIES_2 = "#eb6834"  # one-shot supervised ablation
ORANGE = SERIES_2
TEXT = "#0b0b0b"
TEXT2 = "#52514e"
GRID = "#e4e3df"
SURFACE = "#fcfcfb"
STOP_GRAY = "#8f8e89"

MAP_METHODS = ["driver", "diffusion_greedy", "diffusion", "sup", "softdist",
               "ortools", "zone", "zonehist", "hier_sup", "hier"]
BLIND_METHODS = ["driver", "diffusion", "hier", "ortools", "zone", "zonehist"]
SHORT = {"driver": "Driver (actual)", "diffusion": "Diffusion + 2-opt", "diffusion_greedy": "Diffusion, no 2-opt",
         "ortools": "OR-Tools GLS 5 s", "zone": "Zone-aware OR-Tools", "softdist": "SoftDist + 2-opt",
         "nn": "Nearest neighbour", "softdist_greedy": "SoftDist, no 2-opt",
         "sup": "One-shot GNN + 2-opt", "sup_greedy": "One-shot GNN, no 2-opt",
         "hier": "Zone diffusion + OR-Tools", "hier_sup": "One-shot zone GNN + OR-Tools",
         "zonehist": "History zone order + OR-Tools"}

plt.rcParams.update({"font.size": 9, "axes.edgecolor": GRID, "axes.labelcolor": TEXT2, "xtick.color": TEXT2,
                     "ytick.color": TEXT2, "axes.titlecolor": TEXT, "figure.facecolor": SURFACE,
                     "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE})


def xy(ex):
    lat0 = float(np.mean(ex["lat"][1:]))
    x = (ex["lng"] - ex["lng"][1:].mean()) * np.cos(np.radians(lat0)) * 111.32
    y = (ex["lat"] - lat0) * 110.57
    return x, y


def clean(ax):
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_color(GRID)


def draw_route(ax, ex, seq, title):
    x, y = xy(ex)
    seq = np.asarray(seq)
    stops = seq[1:]  # leave out the long station legs so the neighbourhood stays readable
    pts = np.stack([x[stops], y[stops]], axis=1)
    segs = np.stack([pts[:-1], pts[1:]], axis=1)
    lc = LineCollection(segs, cmap=SEQ, norm=plt.Normalize(0, len(segs)), linewidths=1.6, capstyle="round")
    lc.set_array(np.arange(len(segs)))
    ax.scatter(x[1:], y[1:], s=9, color=STOP_GRAY, zorder=1, linewidths=0)
    ax.add_collection(lc)
    ax.scatter([x[stops[0]]], [y[stops[0]]], s=36, color=BLUE_RAMP[1], edgecolor=TEXT, linewidths=0.8, zorder=3)
    ax.scatter([x[stops[-1]]], [y[stops[-1]]], s=36, color=BLUE_RAMP[-1], edgecolor=TEXT, linewidths=0.8, zorder=3)
    ax.set_title(title, fontsize=8.5, loc="left")
    clean(ax)


def load_results(split):
    p = os.path.join(RES, f"per_route_{split}.json")
    s = os.path.join(RES, f"sequences_{split}.json")
    if not (os.path.exists(p) and os.path.exists(s)):
        return None, None
    return json.load(open(p))["per_route"], json.load(open(s))


def pick_routes(per_route, k=3):
    """Deterministic, non-cherry-picked choice: the routes at the 25th, 50th and 75th percentile of
    the zone-diffusion method's Amazon score."""
    sc = np.array([r["hier"]["amazon_score"] for r in per_route])
    order = np.argsort(sc)
    return [per_route[order[int(q * (len(order) - 1))]] for q in np.linspace(0.25, 0.75, k)]


def route_maps(split="fresh"):
    per_route, seqs = load_results(split)
    if per_route is None:
        print("no results yet")
        return
    exs = {ex["route_id"]: ex for ex in data.load_split(split)}
    rng = np.random.default_rng(2024)
    key = {}
    methods = [m for m in MAP_METHODS if m in per_route[0]]
    for k, r in enumerate(pick_routes(per_route)):
        ex = exs[r["route_id"]]
        rid = r["route_id"]
        pct = ["25th", "50th", "75th"][k]
        fig, axes = plt.subplots(2, 5, figsize=(17, 7.8))
        for ax in axes.flat:
            ax.axis("off")
        for ax, m in zip(axes.flat, methods):
            ax.axis("on")
            mm = r[m]
            draw_route(ax, ex, seqs[rid][m], f"{SHORT[m]}\nscore {mm['amazon_score']:.4f} | {mm['travel_time_s'] / 3600:.2f} h | "
                                              f"{mm['driver_edge_overlap'] * 100:.0f}% driver edges")
        fig.suptitle(f"{split} split, route {rid[8:16]} ({ex['station_code']}, {len(ex['stop_ids']) - 1} stops), {pct} percentile of the "
                     f"zone-diffusion score. Line shade runs light (first stop) to dark (last stop); station legs omitted. "
                     f"Learned models: seed 0.", fontsize=9, color=TEXT, x=0.01, ha="left")
        fig.tight_layout(rect=(0, 0, 1, 0.96))
        fig.savefig(os.path.join(FIG, f"route_{k + 1}_{rid[8:16]}.png"), dpi=140)
        plt.close(fig)

        # blind version: same maps, random anonymous labels, no scores
        perm = rng.permutation(len(BLIND_METHODS))
        letters = "ABCDEF"
        fig, axes = plt.subplots(2, 3, figsize=(11, 7.4))
        key[f"blind_{k + 1}"] = {"route_id": rid}
        for pos, (ax, mi) in enumerate(zip(axes.flat, perm)):
            m = BLIND_METHODS[mi]
            draw_route(ax, ex, seqs[rid][m], f"Sequence {letters[pos]}")
            key[f"blind_{k + 1}"][letters[pos]] = SHORT[m]
        fig.suptitle(f"Blind comparison {k + 1}: six sequences for the same real route. Which one did the driver execute?",
                     fontsize=9, color=TEXT, x=0.01, ha="left")
        fig.tight_layout(rect=(0, 0, 1, 0.96))
        fig.savefig(os.path.join(FIG, f"blind_{k + 1}.png"), dpi=140)
        plt.close(fig)
    with open(os.path.join(FIG, "blind_key.json"), "w") as f:
        json.dump(key, f, indent=1)


def heatmap_figure(split="fresh"):
    per_route, _ = load_results(split)
    L = load_learned(0)
    if per_route is None or "stop_diffusion" not in L:
        return
    tuning = json.load(open(os.path.join(RES, "tuning.json")))
    r = pick_routes(per_route)[1]
    idx = per_route.index(r)
    ex = [e for e in data.load_split(split) if e["route_id"] == r["route_id"]][0]
    T = ex["T"].astype(np.float64)
    panels = [(heatmap(*L["stop_diffusion"][:2], ex, tuning["diff_steps"], tuning["diff_samples"], idx),
               f"Diffusion model ({tuning['diff_steps']} steps x {tuning['diff_samples']} samples)")]
    if "stop_supervised" in L:
        panels.append((heatmap(*L["stop_supervised"][:2], ex, 1, 1, idx), "One-shot supervised GNN (ablation)"))
    x, y = xy(ex)
    ei = ex["edge_index"]
    keep = (ei[0] != 0) & (ei[1] != 0)
    h_s = softdist_heatmap(T, ei, tuning["softdist_tau"])
    panels.append((h_s / h_s[keep].max(), f"SoftDist control (tau = {tuning['softdist_tau']:g}), rescaled"))
    fig, axes = plt.subplots(1, len(panels), figsize=(5.4 * len(panels), 5.2))
    for ax, (h, name) in zip(np.atleast_1d(axes), panels):
        ids = np.where(keep)[0]
        ids = ids[np.argsort(h[ids])]
        segs = np.stack([np.stack([x[ei[0, ids]], y[ei[0, ids]]], 1), np.stack([x[ei[1, ids]], y[ei[1, ids]]], 1)], 1)
        lc = LineCollection(segs, cmap=SEQ, norm=plt.Normalize(0, 1), linewidths=0.4 + 2.2 * h[ids])
        lc.set_array(h[ids])
        ax.add_collection(lc)
        a = ex["actual_seq"][1:]
        ax.plot(x[a], y[a], color=ORANGE, lw=0.8, ls=(0, (2, 2)), label="driver's executed tour")
        ax.scatter(x[1:], y[1:], s=6, color=STOP_GRAY, zorder=3, linewidths=0)
        ax.set_title(name + "\np(driver uses edge)", fontsize=9, loc="left")
        clean(ax)
        ax.legend(loc="lower left", fontsize=7.5, frameon=False)
        fig.colorbar(lc, ax=ax, shrink=0.7, label="edge score")
    fig.suptitle(f"Stop-level edge heat-maps for route {r['route_id'][8:16]} ({split} split, seed-0 models); station edges "
                 "omitted", fontsize=9, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(os.path.join(FIG, f"heatmap_{r['route_id'][8:16]}.png"), dpi=140)
    plt.close(fig)


def zone_order_figure(split="fresh"):
    """Zone tours for one route: driver, zone diffusion, one-shot zone model and history, each decoded
    with the decoder chosen for that method on validation."""
    per_route, _ = load_results(split)
    L = load_learned(0)
    if per_route is None or "zone_diffusion" not in L:
        return
    from evaluate import _seq_from_zone_order, decode_zone_order, zone_order_tau
    from zone_level import driver_zone_order, history_heatmap

    tuning = json.load(open(os.path.join(RES, "tuning.json")))
    r = pick_routes(per_route)[1]
    ex = [e for e in data.load_split(split) if e["route_id"] == r["route_id"]][0]
    zex = [z for z in data.load_split("zone_" + split) if z["route_id"] == r["route_id"]][0]
    idx = per_route.index(r)
    orders = [("Driver's zone order", driver_zone_order(ex)),
              ("Zone diffusion (generated)", decode_zone_order(zex, heatmap(*L["zone_diffusion"][:2], zex, HIER_STEPS,
                                                                           HIER_SAMPLES, idx), tuning["hier_decode"]))]
    if "zone_supervised" in L:
        orders.append(("One-shot supervised zone GNN", decode_zone_order(zex, heatmap(*L["zone_supervised"][:2], zex, 1, 1, idx),
                                                                          tuning["hier_sup_decode"])))
    orders.append(("Historical frequencies only", decode_zone_order(zex, history_heatmap(zex), tuning["zonehist_decode"])))
    x, y = xy(ex)
    cent = {z: (x[[i for i in range(1, len(x)) if ex["zones"][i] == z]].mean(),
                y[[i for i in range(1, len(y)) if ex["zones"][i] == z]].mean()) for z in set(ex["zones"][1:])}
    fig, axes = plt.subplots(1, len(orders), figsize=(4.3 * len(orders), 4.8))
    for ax, (title, order) in zip(axes, orders):
        tau = zone_order_tau(ex, _seq_from_zone_order(ex, order))
        pts = np.array([cent[z] for z in order])
        segs = np.stack([pts[:-1], pts[1:]], axis=1)
        lc = LineCollection(segs, cmap=SEQ, norm=plt.Normalize(0, len(segs)), linewidths=2.0, capstyle="round")
        lc.set_array(np.arange(len(segs)))
        ax.scatter(x[1:], y[1:], s=5, color=STOP_GRAY, linewidths=0, zorder=1)
        ax.add_collection(lc)
        ax.scatter(pts[:, 0], pts[:, 1], s=22, color=BLUE_RAMP[2], edgecolor=TEXT, linewidths=0.5, zorder=3)
        ax.scatter([pts[0, 0]], [pts[0, 1]], s=60, marker="s", color=BLUE_RAMP[1], edgecolor=TEXT, linewidths=0.8, zorder=4)
        ax.set_title(f"{title}\nKendall tau vs. driver = {tau:.2f}", fontsize=8.5, loc="left")
        clean(ax)
    fig.suptitle(f"Zone tours for route {r['route_id'][8:16]} ({len(cent)} zones, seed-0 models). Dots are zone centroids, the "
                 "square is the first zone; shade runs light to dark along the tour.", fontsize=9, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(os.path.join(FIG, f"zone_orders_{r['route_id'][8:16]}.png"), dpi=140)
    plt.close(fig)


def distributions(split="fresh"):
    per_route, _ = load_results(split)
    if per_route is None:
        return
    methods = [m for m in ["nn", "ortools", "softdist_greedy", "softdist", "sup_greedy", "sup", "diffusion_greedy",
                           "diffusion", "zone", "zonehist", "hier_sup", "hier"] if m in per_route[0]]
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.0))
    for ax, key, lab, scale in [(axes[0], "amazon_score", "Amazon challenge score vs. driver (lower is better)", 1.0),
                                (axes[1], "travel_time_s", "Tour travel time, hours (lower is better)", 1 / 3600)]:
        vals = [np.array([r[m][key] for r in per_route]) * scale for m in methods]
        bp = ax.boxplot(vals, orientation="horizontal", widths=0.5, patch_artist=True, showfliers=False,
                        medianprops={"color": BLUE_RAMP[-1], "lw": 2}, whiskerprops={"color": TEXT2}, capprops={"color": TEXT2})
        for b in bp["boxes"]:
            b.set_facecolor(BLUE_RAMP[1])
            b.set_edgecolor(BLUE_RAMP[4])
        for i, v in enumerate(vals):
            ax.scatter([v.mean()], [i + 1], marker="D", s=22, color=ORANGE, zorder=3, label="mean" if i == 0 else None)
        if key == "travel_time_s":
            dv = np.array([r["driver"][key] for r in per_route]) * scale
            ax.axvline(np.median(dv), color=TEXT2, lw=1, ls="--", label="driver median")
        ax.set_yticks(range(1, len(methods) + 1))
        ax.set_yticklabels([SHORT[m] for m in methods] if ax is axes[0] else [""] * len(methods))
        ax.set_xlabel(lab)
        ax.grid(axis="x", color=GRID, lw=0.6)
        ax.set_axisbelow(True)
    handles, labels = [], []
    for ax in axes:
        for h, lb in zip(*ax.get_legend_handles_labels()):
            if lb not in labels:
                handles.append(h)
                labels.append(lb)
    fig.legend(handles, labels, loc="upper right", ncol=2, fontsize=8, frameon=False)
    fig.suptitle(f"Per-route distributions on the {split} split ({len(per_route)} routes, seed-0 models); boxes = quartiles, "
                 "whiskers = 1.5 IQR", fontsize=9, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(os.path.join(FIG, "score_distributions.png"), dpi=140)
    plt.close(fig)

    x = np.array([r["ortools"]["amazon_score"] for r in per_route])
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    lim = [0, max(x.max(), max(r[m]["amazon_score"] for r in per_route for m in ["diffusion", "hier"])) * 1.05]
    for ax, m in zip(axes, ["diffusion", "hier"]):
        y = np.array([r[m]["amazon_score"] for r in per_route])
        ax.plot(lim, lim, color=TEXT2, lw=1, ls="--")
        ax.scatter(x, y, s=14, color=BLUE_RAMP[4], alpha=0.75, edgecolor=SURFACE, linewidths=0.5)
        ax.set_xlim(lim)
        ax.set_ylim(lim)
        ax.set_aspect("equal")
        ax.set_xlabel("OR-Tools (GLS, 5 s) Amazon score")
        ax.set_ylabel(f"{SHORT[m]} Amazon score")
        ax.set_title(f"{SHORT[m]}: below the diagonal on {np.mean(y < x) * 100:.0f}% of routes\n(closer to the driver than OR-Tools)",
                     fontsize=8.5, loc="left")
        ax.grid(color=GRID, lw=0.6)
        ax.set_axisbelow(True)
    fig.suptitle(f"Each dot is one {split} route (seed-0 models)", fontsize=9, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(os.path.join(FIG, "paired_diffusion_vs_ortools.png"), dpi=140)
    plt.close(fig)


def seed_spread():
    """Dot plot: mean Amazon score per training seed for each learned model (blue circles =
    diffusion, orange squares = one-shot supervised ablation) with the non-learned baselines as
    gray reference lines, for both test sets."""
    R = os.path.join(RES, "results.json")
    if not os.path.exists(R):
        return
    res = json.load(open(R))["splits"]
    splits = [s for s in SPLITS if s in res and res[s].get("seed_summary")]
    if not splits:
        return
    rows = [("Stop level, greedy + 2-opt", "diffusion", "sup", "softdist", "SoftDist + 2-opt"),
            ("Stop level, greedy only", "diffusion_greedy", "sup_greedy", "softdist_greedy", "SoftDist, no 2-opt"),
            ("Zone level + OR-Tools", "hier", "hier_sup", "zonehist", "History zone order")]
    fig, axes = plt.subplots(1, len(splits), figsize=(5.4 * len(splits), 4.0), sharey=True)
    for ax, split in zip(np.atleast_1d(axes), splits):
        ss = res[split]["seed_summary"]
        summ = res[split]["summary"]["methods"]
        for yi, (label, d_key, s_key, base, base_lab) in enumerate(rows):
            yb = len(rows) - 1 - yi
            for key, col, mk, dy, lab in [(d_key, SERIES_1, "o", 0.12, "diffusion (one dot per seed)"),
                                          (s_key, SERIES_2, "s", -0.12, "one-shot supervised ablation (one dot per seed)")]:
                if key not in ss["methods"]:
                    continue
                v = [d["mean"] for d in ss["methods"][key]["per_seed"].values()]
                ax.scatter(v, [yb + dy] * len(v), s=40, marker=mk, color=col, edgecolor=SURFACE, linewidths=1.5, zorder=3,
                           label=lab if yi == 0 else None)
            if base in summ:
                bv = summ[base]["amazon_score"]["mean"]
                ax.plot([bv, bv], [yb - 0.3, yb + 0.3], color=TEXT2, lw=2, zorder=2,
                        label="non-learned control of the same row" if yi == 0 else None)
                ax.annotate(base_lab, (bv, yb + 0.32), fontsize=7, color=TEXT2, ha="center", va="bottom")
        if "zone" in summ:
            zv = summ["zone"]["amazon_score"]["mean"]
            ax.axvline(zv, color=GRID, lw=1.5, ls="--", zorder=1, label="M_zone heuristic")
        ax.set_yticks(range(len(rows)))
        ax.set_yticklabels([r[0] for r in rows[::-1]])
        ax.set_xlabel("mean Amazon score (lower = closer to the driver)")
        tag = "final" if split == "fresh" else "reused"
        ax.set_title(f"{split}, {tag} ({res[split]['summary']['n_routes']} routes)", fontsize=9, loc="left")
        ax.grid(axis="x", color=GRID, lw=0.6)
        ax.set_axisbelow(True)
        ax.set_ylim(-0.6, len(rows) - 0.2)
    h, lb = np.atleast_1d(axes)[0].get_legend_handles_labels()
    fig.legend(h, lb, loc="lower center", ncol=2, fontsize=7.5, frameon=False)
    fig.suptitle("Between-seed spread: each dot is one separately trained model (seeds 0, 1, 2)", fontsize=9, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0.13, 1, 0.94))
    fig.savefig(os.path.join(FIG, "seed_spread.png"), dpi=140)
    plt.close(fig)


def _read_log(p):
    recs = {}
    for line in open(p):
        if line.strip():
            r = json.loads(line)
            recs[(r["step"], "final" in r)] = r  # a resumed chunk may re-log a step; keep the last
    return [recs[k] for k in sorted(recs)]


def training_curve():
    fig, axes = plt.subplots(2, 2, figsize=(11, 6.6))
    for i, level in enumerate(["stop", "zone"]):
        for j, (obj, col) in enumerate([("diffusion", SERIES_1), ("supervised", SERIES_2)]):
            ax = axes[i, j]
            runs = [(f"seed {seed}", f"{level}_{obj}_seed{seed}_log.jsonl", ls, col, TEXT)
                    for seed, ls in zip([0, 1, 2], ["-", "--", ":"])]
            if level == "stop" and obj == "diffusion":
                runs.append(("spec-budget, 3,000 steps", "stop_diffusion_budget_seed0_log.jsonl", "-", BLUE_RAMP[2], TEXT2))
            for lab, fn, ls, c_tr, c_va in runs:
                p = os.path.join(HERE, "checkpoints", fn)
                if not os.path.exists(p):
                    continue
                recs = _read_log(p)
                tr = [(r["step"], r["loss"]) for r in recs if "loss" in r]
                va = [(r["step"], r["val_loss_ema"]) for r in recs if "val_loss_ema" in r and "final" not in r]
                ax.plot(*zip(*tr), color=c_tr, lw=1.0, ls=ls, alpha=0.55, label=f"{lab} training (50-step mean)")
                if va:
                    ax.plot(*zip(*va), color=c_va, lw=1.4, ls=ls, marker="o", ms=3, label=f"{lab} validation (EMA)")
            ax.set_xlabel("optimisation step")
            ax.set_ylabel("binary cross-entropy")
            ax.set_ylim(0, 0.2)
            ax.grid(color=GRID, lw=0.6)
            ax.legend(frameon=False, fontsize=6.5, ncol=2)
            name = "edge diffusion (loss averaged over noise levels)" if obj == "diffusion" else "one-shot supervised (ablation)"
            ax.set_title(f"{level.capitalize()} level, {name}", fontsize=9, loc="left")
    fig.suptitle("Training curves of the 12 main runs and the spec-budget run (High-quality routes only). Losses of the two "
                 "objectives are not directly "
                 "comparable: the diffusion loss includes noisy inputs.", fontsize=8.5, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(os.path.join(FIG, "training_curves.png"), dpi=140)
    plt.close(fig)


def ablation_scatter(split="fresh"):
    """Per route: seed-averaged score of each diffusion model (y) against its one-shot supervised twin
    (x), same network, data and decoder. Points above the diagonal: diffusion further from the driver."""
    per_route, _ = load_results(split)
    if per_route is None:
        return
    pairs = [("diffusion", "sup", "Stop level, greedy + 2-opt"), ("diffusion_greedy", "sup_greedy", "Stop level, greedy only"),
             ("hier", "hier_sup", "Zone level + OR-Tools")]
    pairs = [p for p in pairs if p[0] in per_route[0] and p[1] in per_route[0]]
    if not pairs:
        return

    def avg(m):
        keys = [k for k in [m, m + "@1", m + "@2"] if k in per_route[0]]
        return np.mean([[r[k]["amazon_score"] for r in per_route] for k in keys], axis=0), len(keys)

    fig, axes = plt.subplots(1, len(pairs), figsize=(4.6 * len(pairs), 4.7))
    for ax, (d, s_, title) in zip(np.atleast_1d(axes), pairs):
        y, nd = avg(d)
        x, ns = avg(s_)
        lim = [0, max(x.max(), y.max()) * 1.05]
        ax.plot(lim, lim, color=TEXT2, lw=1, ls="--")
        ax.scatter(x, y, s=12, color=SERIES_1, alpha=0.7, edgecolor=SURFACE, linewidths=0.4)
        ax.set_xlim(lim)
        ax.set_ylim(lim)
        ax.set_aspect("equal")
        ax.set_xlabel(f"one-shot supervised, mean of {ns} seeds")
        ax.set_ylabel(f"diffusion, mean of {nd} seeds")
        ax.set_title(f"{title}\nmean diff {np.mean(y - x):+.4f}; diffusion better on {np.mean(y < x) * 100:.0f}% of routes",
                     fontsize=8.5, loc="left")
        ax.grid(color=GRID, lw=0.6)
        ax.set_axisbelow(True)
    fig.suptitle(f"Diffusion vs. its one-shot supervised twin, per route ({split} split, {len(per_route)} routes; Amazon score, "
                 "lower is better; points above the diagonal favour the supervised model)", fontsize=9, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(os.path.join(FIG, "ablation_diffusion_vs_supervised.png"), dpi=140)
    plt.close(fig)


def main():
    os.makedirs(FIG, exist_ok=True)
    torch.set_num_threads(4)
    split = "fresh" if os.path.exists(os.path.join(RES, "per_route_fresh.json")) else "test"
    training_curve()
    seed_spread()
    ablation_scatter(split)
    route_maps(split)
    heatmap_figure(split)
    zone_order_figure(split)
    distributions(split)
    print(f"figures written to {FIG} (per-route figures from the {split} split)")


if __name__ == "__main__":
    main()
