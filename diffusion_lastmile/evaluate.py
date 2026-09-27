"""Evaluate all methods side by side on held-out routes.

Stages
    python evaluate.py round1                # OPTIONAL: rebuild the complete round-1 validation tuning
                                             # record from results/cache_round1 (results/tuning_round1.json);
                                             # skipped when that archive is absent (clean checkout)
    python evaluate.py tune                  # choose every setting of every method on the 60
                                             # validation routes (never on a test split); equal 21-point
                                             # grids for the three zone-order methods and the zone heuristic
    python evaluate.py run --split fresh     # FINAL test: 300 evaluation-dataset routes that were never
                                             # evaluated before the settings were frozen (data.py preprocess-fresh)
    python evaluate.py run --split test      # REUSED: 300 evaluation-dataset routes, also scored in round 1
    python evaluate.py run --split heldout   # REUSED: 150 High-quality training-dataset routes, also round 1
    python evaluate.py budget --split fresh  # sensitivity: OR-Tools baselines with a longer time limit
    python evaluate.py report                # results/results.json + results/summary.md
    python evaluate.py export                # EMA weights of all finished models -> results/models

Models are read from checkpoints/ (full resumable checkpoints written by train.py) or, when a
checkpoint is absent there, from the exported EMA weights in results/models/*_ema.pt, so the
evaluation can be reproduced without retraining.

Learned methods (each trained with seeds 0, 1 and 2, see train.py):
    diffusion / diffusion_greedy   stop-level Bernoulli edge diffusion, greedy decode (+ 2-opt)
    sup / sup_greedy               the SAME network trained as a one-shot supervised edge classifier
                                   (ablation control for the diffusion process), same decoder
    diffusion_budget(_greedy)      stop-level diffusion retrained within the specification's training
                                   budget (seed 0 only, 3,000 steps, train.py --tag budget); it uses
                                   the sampling settings tuned for the full model (no own tuning)
    hier                           zone-level edge diffusion -> zone order -> OR-Tools
    hier_sup                       the SAME zone network trained one-shot supervised -> same pipeline
Seed 0 results are stored under the plain method key, seeds 1 and 2 under "<key>@1", "<key>@2".

Every method outputs a full stop sequence starting at the station; each sequence is checked for
feasibility and scored with the real travel-time matrix (closed tour, seconds) and with the
official Amazon challenge score against the driver's executed sequence (lower is better).
OR-Tools jobs are cached in results/cache so interrupted runs resume where they stopped, and each
cached solve records its wall-clock and CPU seconds.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import time
from multiprocessing import Pool

import numpy as np
import torch

import data
from baselines import nearest_neighbour, softdist_heatmap, solve_ortools_job, zone_penalty_matrix
from decode import greedy_decode, two_opt
from diffusion import collate, make_process
from score import amazon_score, is_valid, route_travel_time
from train import build_model, default_ckpt
from zone_level import driver_zone_order, history_heatmap, zone_order_penalty_matrix

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.environ.get("DIFFLM_RESULTS", os.path.join(HERE, "results"))  # override only for smoke tests
CACHE = os.path.join(RES, "cache")
CACHE_R1 = os.path.join(HERE, "results", "cache_round1")
MODELS = os.path.join(HERE, "results", "models")
SPLITS = ["fresh", "test", "heldout"]

METHOD_LABELS = {
    "driver": "M_driver (actual driver sequence)",
    "nn": "M_nn (nearest neighbour)",
    "ortools": "M_ortools (OR-Tools GLS, 5 s)",
    "zone": "M_zone (OR-Tools GLS + zone-change penalty, 5 s)",
    "zonehist": "M_zonehist (historical zone order + OR-Tools, 5 s)",
    "softdist_greedy": "M_softdist greedy (no 2-opt)",
    "softdist": "M_softdist greedy + 2-opt",
    "sup_greedy": "M_supervised greedy (one-shot GNN ablation, no 2-opt)",
    "sup": "M_supervised greedy + 2-opt (one-shot GNN ablation)",
    "diffusion_greedy": "M_diffusion greedy (no 2-opt)",
    "diffusion": "M_diffusion greedy + 2-opt",
    "diffusion_budget_greedy": "M_diffusion, spec-budget model, greedy (no 2-opt)",
    "diffusion_budget": "M_diffusion, spec-budget model, greedy + 2-opt",
    "hier_sup": "M_supervised_zone (one-shot zone GNN + OR-Tools, 5 s)",
    "hier": "M_diffusion_zone (zone diffusion + OR-Tools, 5 s)",
}
METHOD_ORDER = list(METHOD_LABELS)
LEARNED = ["sup_greedy", "sup", "diffusion_greedy", "diffusion", "hier_sup", "hier"]
ORTOOLS_METHODS = ["ortools", "zone", "zonehist", "hier_sup", "hier"]
REFERENCES = ["diffusion", "hier", "hier_sup"]
# (a, b): paired gaps a - b that are reported for every seed (learned methods paired seed by seed)
GAPS = [("hier", "zonehist"), ("hier", "hier_sup"), ("hier_sup", "zonehist"), ("hier", "zone"),
        ("diffusion", "sup"), ("diffusion_greedy", "sup_greedy"), ("diffusion", "softdist"), ("sup", "softdist"),
        ("diffusion", "zone"), ("hier_sup", "zone"), ("diffusion_budget", "diffusion")]
ZONE_DECODERS = ["greedy", "ml", "greedy2opt"]
# Equal grids of 21 settings. Round 2 first used lam in {1, 4, 16, 64} (12 settings) and 1/16 ... 128 for
# M_zone; two zone-order optima landed on lam = 64, so every grid was widened by 3 settings (lam 256, M_zone
# up to 1024). The zone-diffusion optimum then landed on lam = 256, the new upper edge, so round 3 widened
# every grid by another 6 settings (lam 1024 and 4096; M_zone up to 65536). All widening used the
# validation split only.
ZONE_ORDER_LAMS = [1.0, 4.0, 16.0, 64.0, 256.0, 1024.0, 4096.0]  # 3 decoders x 7 = 21 settings per zone-order method
ZONE_LAMS = [2.0 ** k for k in range(-4, 17)]                    # 1/16 ... 65536: 21 settings for M_zone
HIER_STEPS, HIER_SAMPLES = 10, 16                              # fixed (chosen on val in round 1, see tuning_round1.json)


def mkey(m: str, seed: int) -> str:
    return m if seed == 0 else f"{m}@{seed}"


def base_method(key: str) -> str:
    return key.split("@")[0]


# ----------------------------------------------------------------------------------------
# models and helpers
# ----------------------------------------------------------------------------------------
def model_path(level: str, obj: str, seed: int, tag: str = "") -> str | None:
    """Full checkpoint in checkpoints/ if present, else the exported EMA weights in results/models."""
    p = default_ckpt(level, obj, seed, tag)
    if os.path.exists(p):
        return p
    q = os.path.join(MODELS, os.path.basename(p).replace(".pt", "_ema.pt"))
    return q if os.path.exists(q) and "DIFFLM_CKPT_DIR" not in os.environ else None


def load_model(ckpt_path: str):
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model = build_model(ck["cfg"])
    model.load_state_dict(ck["ema"])
    model.eval()
    return model, make_process(ck["cfg"]), ck


def load_learned(seed: int, budget: bool = False) -> dict:
    """{'stop_diffusion': (model, proc, ck, path), ...} for the checkpoints of one seed that exist."""
    out = {}
    runs = [(lvl, obj, "") for lvl in ["stop", "zone"] for obj in ["diffusion", "supervised"]]
    if seed == 0 and budget:
        runs.append(("stop", "diffusion", "budget"))
    for level, obj, tag in runs:
        p = model_path(level, obj, seed, tag)
        if p is not None:
            m, proc, ck = load_model(p)
            if ck["step"] < ck["cfg"]["total_steps"]:
                raise SystemExit(f"{p} is not fully trained ({ck['step']}/{ck['cfg']['total_steps']} steps)")
            out[f"{level}_{obj}" + (f"_{tag}" if tag else "")] = (m, proc, ck, p)
    return out


def driver_edge_overlap(seq, actual) -> float:
    """Fraction of the driver's directed tour edges that the proposed tour also uses."""
    a = set(zip(actual.tolist(), np.roll(actual, -1).tolist()))
    s = set(zip(list(seq), np.roll(seq, -1).tolist()))
    return len(a & s) / len(a)


def zone_order_tau(ex, seq) -> float:
    """Kendall tau between the zone order implied by a sequence and the driver's zone order."""
    from scipy.stats import kendalltau

    d = driver_zone_order(ex)
    fake = dict(ex)
    fake["actual_seq"] = np.asarray(seq)
    p = driver_zone_order(fake)
    rank = {z: k for k, z in enumerate(p)}
    if len(d) < 2:
        return 1.0
    return float(kendalltau(np.arange(len(d)), [rank[z] for z in d]).statistic)


def metrics(ex, seq, runtime, cpu=None) -> dict:
    seq = np.asarray(seq, dtype=np.int64)
    n = len(ex["stop_ids"])
    ok = is_valid(seq, n)
    out = {"valid": bool(ok), "runtime_s": float(runtime)}
    if cpu is not None:
        out["ortools_cpu_s"] = float(cpu)
    if ok:
        out["travel_time_s"] = route_travel_time(seq, ex["T"])
        out["amazon_score"] = float(amazon_score(ex["actual_seq"], seq, ex["T"]))
        out["driver_edge_overlap"] = driver_edge_overlap(seq, ex["actual_seq"])
        out["zone_order_tau"] = zone_order_tau(ex, seq)
    return out


def heatmap(model, proc, ex, steps, samples, seed):
    """Edge heat-map of one route. For the diffusion process the `samples` independent reverse
    chains run as one batched graph (samples copies of the route) and their final x_0
    probabilities are averaged; the one-shot supervised model needs a single forward pass."""
    g = torch.Generator().manual_seed(seed)
    if proc.objective == "supervised":
        return proc.sample_heatmap(model, collate([ex])).numpy()
    b = collate([ex] * samples)
    h = proc.sample_heatmap(model, b, steps=steps, n_samples=1, generator=g).numpy()
    return h.reshape(samples, -1).mean(axis=0)


def decode_zone_order(zex, heat, mode: str = "greedy") -> list:
    """Zone tour (list of zone names, station excluded) decoded from a zone-edge heat-map.

    greedy      the same greedy edge insertion as at stop level (ties broken by zone travel time)
    ml          maximum-likelihood tour: the tour minimising sum(-log p) over its zone edges, solved
                with OR-Tools (zone graphs have about 20 nodes, 0.3 s limit)
    greedy2opt  greedy, then 2-opt on the mean zone-to-zone travel times (this is the decoder behind
                the round-1 "*_2opt1_*" validation runs; see tuning_round1.json)
    """
    m = len(zex["zone_names"])
    zT = zex["zone_T"].astype(np.float64)
    if mode == "ml":
        from baselines import ortools_tsp

        P = np.full((m, m), 1e-4)
        ei = zex["edge_index"]
        P[ei[0], ei[1]] = np.maximum(heat, 1e-4)
        C = -np.log(P) * 1000.0
        np.fill_diagonal(C, 0.0)
        zs = ortools_tsp(C, time_limit_s=0.3, scale=1.0)
    elif mode == "greedy2opt":
        zs = two_opt(greedy_decode(m, zex["edge_index"], heat, zT), zT)
    elif mode == "greedy":
        zs = greedy_decode(m, zex["edge_index"], heat, zT)
    else:
        raise ValueError(mode)
    return [zex["zone_names"][k] for k in zs[1:]]


def run_fast_methods(ex, learned_by_seed, cfg, route_seed=0) -> tuple[dict, dict]:
    """All methods that do not need OR-Tools. Returns (metrics_by_method, sequences_by_method)."""
    T = ex["T"].astype(np.float64)
    n = len(ex["stop_ids"])
    seqs, res = {}, {}
    seqs["driver"] = ex["actual_seq"]
    res["driver"] = metrics(ex, ex["actual_seq"], 0.0)

    t0 = time.perf_counter()
    s = nearest_neighbour(T)
    seqs["nn"] = s
    res["nn"] = metrics(ex, s, time.perf_counter() - t0)

    def heat_methods(name, h_fn):
        t0 = time.perf_counter()
        h = h_fn()
        s = greedy_decode(n, ex["edge_index"], h, T)
        t1 = time.perf_counter()
        seqs[name + "_greedy" if "@" not in name else name.replace("@", "_greedy@")] = s
        res[name + "_greedy" if "@" not in name else name.replace("@", "_greedy@")] = metrics(ex, s, t1 - t0)
        s2 = two_opt(s, T)
        seqs[name] = s2
        res[name] = metrics(ex, s2, time.perf_counter() - t0)

    heat_methods("softdist", lambda: softdist_heatmap(T, ex["edge_index"], cfg["softdist_tau"]))
    for seed, L in learned_by_seed.items():
        if "stop_diffusion" in L:
            m, proc = L["stop_diffusion"][:2]
            heat_methods(mkey("diffusion", seed),
                         lambda: heatmap(m, proc, ex, cfg["diff_steps"], cfg["diff_samples"], route_seed))
        if "stop_supervised" in L:
            m, proc = L["stop_supervised"][:2]
            heat_methods(mkey("sup", seed), lambda: heatmap(m, proc, ex, 1, 1, route_seed))
        if "stop_diffusion_budget" in L:
            m, proc = L["stop_diffusion_budget"][:2]
            heat_methods("diffusion_budget",
                         lambda: heatmap(m, proc, ex, cfg["diff_steps"], cfg["diff_samples"], route_seed))
    return res, seqs


def ortools_costs(ex, zex, learned_by_seed, cfg, route_seed=0) -> dict:
    """Cost matrices (and preparation time: heat-map + zone decoding) for the OR-Tools methods."""
    T = ex["T"].astype(np.float64)
    out = {"ortools": (T, 0.0)}
    t0 = time.perf_counter()
    out["zone"] = (zone_penalty_matrix(T, ex["zones"], cfg["zone_lam"]), time.perf_counter() - t0)
    t0 = time.perf_counter()
    order = decode_zone_order(zex, history_heatmap(zex), cfg["zonehist_decode"])
    out["zonehist"] = (zone_order_penalty_matrix(T, ex["zones"], order, cfg["zonehist_lam"]), time.perf_counter() - t0)
    for seed, L in learned_by_seed.items():
        for name, kind in [("hier", "zone_diffusion"), ("hier_sup", "zone_supervised")]:
            if kind not in L or f"{name}_lam" not in cfg:
                continue
            m, proc = L[kind][:2]
            t0 = time.perf_counter()
            h = heatmap(m, proc, zex, HIER_STEPS, HIER_SAMPLES, route_seed)
            order = decode_zone_order(zex, h, cfg[f"{name}_decode"])
            out[mkey(name, seed)] = (zone_order_penalty_matrix(T, ex["zones"], order, cfg[f"{name}_lam"]),
                                     time.perf_counter() - t0)
    return out


def run_ortools(tag, exs, costs, limit, workers) -> dict:
    """costs: {route_id: cost matrix}. Returns {route_id: {"seq", "runtime_s", "cpu_s"}}, cached on disk."""
    os.makedirs(CACHE, exist_ok=True)
    path = os.path.join(CACHE, tag + ".json")
    cache = json.load(open(path)) if os.path.exists(path) else {}
    todo = [ex["route_id"] for ex in exs if ex["route_id"] not in cache]
    if todo:
        print(f"[ortools] {tag}: {len(todo)} routes to solve with {workers} workers", flush=True)
        jobs = [(rid, tag, costs[rid], limit) for rid in todo]
        t0 = time.time()
        with Pool(workers) as pool:
            for k, r in enumerate(pool.imap_unordered(solve_ortools_job, jobs)):
                cache[r["route_id"]] = {"seq": r["seq"], "runtime_s": r["runtime_s"], "cpu_s": r["cpu_s"]}
                if (k + 1) % 20 == 0 or k + 1 == len(jobs):
                    with open(path, "w") as f:
                        json.dump(cache, f)
                    print(f"  {k + 1}/{len(jobs)} ({time.time() - t0:.0f}s)", flush=True)
    return cache


def ortools_tag(method, split, cfg, limit=None):
    lim = f"lim{(limit or cfg['ortools_limit']):g}"
    m, seed = base_method(method), (int(method.split("@")[1]) if "@" in method else 0)
    if m == "ortools":
        return f"ortools_{split}_{lim}"
    if m == "zone":
        return f"zone_{split}_{lim}_lam{cfg['zone_lam']:g}"
    if m == "zonehist":
        return f"zonehist_{split}_{lim}_lam{cfg['zonehist_lam']:g}_{cfg['zonehist_decode']}"
    return (f"{m}_{split}_{lim}_lam{cfg[m + '_lam']:g}_{cfg[m + '_decode']}_s{HIER_STEPS}x{HIER_SAMPLES}"
            f"_seed{seed}")


# ----------------------------------------------------------------------------------------
# statistics
# ----------------------------------------------------------------------------------------
def bootstrap_ci(x, n_boot=10000, seed=0, stat=np.mean):
    x = np.asarray(x, dtype=np.float64)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), size=(n_boot, len(x)))
    bs = stat(x[idx], axis=1)
    return [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]


def describe(v) -> dict:
    v = np.asarray(v, dtype=np.float64)
    return {"mean": float(v.mean()), "median": float(np.median(v)), "std": float(v.std()),
            "q25": float(np.percentile(v, 25)), "q75": float(np.percentile(v, 75)), "ci95_mean": bootstrap_ci(v)}


def paired(per_route, a, b, key="amazon_score") -> dict:
    from scipy.stats import wilcoxon

    both = [r for r in per_route if a in r and b in r and r[a]["valid"] and r[b]["valid"]]
    x = np.array([r[a][key] for r in both])
    y = np.array([r[b][key] for r in both])
    diff = x - y
    try:
        p = float(wilcoxon(x, y).pvalue) if np.any(diff != 0) else 1.0
    except ValueError:
        p = 1.0
    return {"n": len(both), "mean_diff_ref_minus_other": float(diff.mean()), "ci95": bootstrap_ci(diff),
            "median_diff": float(np.median(diff)), "ref_better_rate": float(np.mean(diff < 0)),
            "tie_rate": float(np.mean(diff == 0)), "wilcoxon_p": p,
            "relative_mean_diff": float(diff.mean() / y.mean()) if y.mean() != 0 else None}


def summarize(per_route: list, methods: list) -> dict:
    out = {"n_routes": len(per_route), "methods": {}, "paired": {}}
    for m in methods:
        rows = [r[m] for r in per_route if m in r]
        valid = [r for r in rows if r["valid"]]
        d = {"n": len(rows), "feasible_rate": len(valid) / max(1, len(rows))}
        for key in ["travel_time_s", "amazon_score", "runtime_s", "driver_edge_overlap", "zone_order_tau", "ortools_cpu_s"]:
            v = [r[key] for r in valid if key in r]
            if v:
                d[key] = describe(v)
        if "ortools_cpu_s" in d:
            ratio = [r["ortools_cpu_s"] / r["runtime_s"] for r in valid if "ortools_cpu_s" in r]
            d["ortools_cpu_over_wall"] = {"median": float(np.median(ratio)), "min": float(np.min(ratio))}
        out["methods"][m] = d
    for ref in REFERENCES:
        if ref not in methods:
            continue
        out["paired"][ref] = {m: {k: paired(per_route, ref, m, k) for k in ["amazon_score", "travel_time_s"]}
                              for m in methods if m != ref}
    return out


def seed_summary(per_route: list, seeds: list) -> dict:
    """Between-seed spread of every learned method and of every gap in GAPS."""
    out = {"seeds": seeds, "methods": {}, "gaps": {}}

    def keys_for(m):
        return {s: (mkey(m, s) if m in LEARNED else m) for s in seeds}

    for m in LEARNED:
        ks = keys_for(m)
        if not all(k in per_route[0] for k in ks.values()):
            continue
        per_seed = {}
        for s, k in ks.items():
            v = np.array([r[k]["amazon_score"] for r in per_route])
            tt = np.array([r[k]["travel_time_s"] for r in per_route])
            per_seed[str(s)] = {"mean": float(v.mean()), "median": float(np.median(v)), "ci95_mean": bootstrap_ci(v),
                                "travel_time_mean_s": float(tt.mean())}
        means = np.array([d["mean"] for d in per_seed.values()])
        out["methods"][m] = {"per_seed": per_seed,
                             "across_seeds": {"mean": float(means.mean()), "sd": float(means.std(ddof=1)),
                                              "min": float(means.min()), "max": float(means.max())}}
    for a, b in GAPS:
        ka, kb = keys_for(a), keys_for(b)
        if not all(k in per_route[0] for k in list(ka.values()) + list(kb.values())):
            continue
        per_seed = {str(s): paired(per_route, ka[s], kb[s]) for s in seeds}
        diffs = np.array([d["mean_diff_ref_minus_other"] for d in per_seed.values()])
        # seed-averaged comparison: per route, average each method's score over the seeds first
        xa = np.mean([[r[ka[s]]["amazon_score"] for r in per_route] for s in seeds], axis=0)
        xb = np.mean([[r[kb[s]]["amazon_score"] for r in per_route] for s in seeds], axis=0)
        d = xa - xb
        out["gaps"][f"{a} - {b}"] = {
            "per_seed": per_seed,
            "across_seeds": {"mean": float(diffs.mean()), "sd": float(diffs.std(ddof=1)), "min": float(diffs.min()),
                             "max": float(diffs.max()),
                             "all_seed_cis_below_zero": bool(all(v["ci95"][1] < 0 for v in per_seed.values()))},
            "seed_averaged": {"mean_diff": float(d.mean()), "ci95": bootstrap_ci(d), "better_rate": float(np.mean(d < 0))},
        }
    return out


# ----------------------------------------------------------------------------------------
# stage: round-1 tuning record
# ----------------------------------------------------------------------------------------
def round1(args):
    """Rebuild the COMPLETE round-1 validation tuning record from the archived OR-Tools caches.

    Round 1 of this study published a tuning.json from which the 'greedy + 2-opt' zone-decoder
    runs had been removed; their cached solutions remained. This stage re-scores every cached
    validation run, so the record lists every configuration that was actually tried."""
    if not os.path.isdir(CACHE_R1):
        print(f"[round1] {os.path.relpath(CACHE_R1, HERE)} is absent (it is a git-ignored archive of an earlier "
              "round); this optional stage is skipped")
        return
    val = data.load_split("val")
    pub_path = os.path.join(CACHE_R1, "tuning_round1_as_published.json")
    pub = json.load(open(pub_path)) if os.path.exists(pub_path) else {}
    rec = {"note": "Every validation configuration tried in round 1, re-scored from the archived OR-Tools caches "
                   "(results/cache_round1). '2opt1' in a cache name is the greedy2opt zone decoder, whose results were "
                   "omitted from the round-1 tuning.json. softdist and stop-level diffusion sampling settings are copied "
                   "from the round-1 tuning.json (they need no OR-Tools cache). The round-1 models were retrained in "
                   "round 2, so these numbers belong to the round-1 checkpoints.",
           "softdist": pub.get("softdist"), "diffusion_sampling": pub.get("diffusion"), "configs": {}, "counts": {}}
    pat = re.compile(r"^(zone|zonehist|hier)_val_lim(?P<lim>[\d.]+)_lam(?P<lam>[\d.]+)(?:_(?P<dec>greedy|ml|2opt1))?"
                     r"(?:_s(?P<steps>\d+)x(?P<samples>\d+))?\.json$")
    for p in sorted(glob.glob(os.path.join(CACHE_R1, "*_val_*.json"))):
        name = os.path.basename(p)
        mt = pat.match(name)
        if not mt:
            continue
        method = name.split("_val_")[0]
        cache = json.load(open(p))
        if not all(ex["route_id"] in cache for ex in val):
            continue
        sc = _mean_scores(val, [cache[ex["route_id"]]["seq"] for ex in val])
        dec = mt.group("dec")
        rec["configs"].setdefault(method, {})[name[:-5]] = {
            "lam": float(mt.group("lam")), "decode": {"2opt1": "greedy2opt"}.get(dec, dec),
            "samples": int(mt.group("samples")) if mt.group("samples") else None, **sc}
    for m, c in rec["configs"].items():
        best = min(c, key=lambda k: c[k]["amazon_score"])
        rec["counts"][m] = len(c)
        rec[f"{m}_best"] = best
    rec["counts"]["softdist"] = len(pub.get("softdist") or {})
    rec["counts"]["diffusion_sampling"] = len(pub.get("diffusion") or {})
    with open(os.path.join(RES, "tuning_round1.json"), "w") as f:
        json.dump(rec, f, indent=1)
    print(json.dumps(rec["counts"]), {m: rec[f"{m}_best"] for m in rec["configs"]})


# ----------------------------------------------------------------------------------------
# stage: tuning on the validation split
# ----------------------------------------------------------------------------------------
def _mean_scores(exs, seqs):
    sc = [amazon_score(ex["actual_seq"], np.asarray(s), ex["T"]) for ex, s in zip(exs, seqs)]
    tt = [route_travel_time(np.asarray(s), ex["T"]) for ex, s in zip(exs, seqs)]
    return {"amazon_score": float(np.mean(sc)), "travel_time_s": float(np.mean(tt))}


def _seq_from_zone_order(ex, order):
    """Stops grouped by the given zone order (within a zone: stop id order). Only used to
    measure how well a zone order matches the driver's zone order."""
    rank = {z: k for k, z in enumerate(order)}
    stops = sorted(range(1, len(ex["stop_ids"])), key=lambda i: (rank[ex["zones"][i]], i))
    return np.array([0] + stops)


def _boundary(grid_vals, best):
    g = sorted(grid_vals)
    return "lower" if best == g[0] else ("upper" if best == g[-1] else "interior")


def tune(args):
    val = data.load_split("val")
    if args.limit:  # smoke tests only
        val = val[: args.limit]
    torch.set_num_threads(args.threads)
    path = os.path.join(RES, "tuning.json")
    out = json.load(open(path)) if os.path.exists(path) else {}
    out.update({"split": "val", "n_routes": len(val), "ortools_limit": args.ortools_limit,
                "note": "All settings chosen by the mean Amazon score on the validation routes; learned-model settings "
                        "were tuned with the seed-0 checkpoints and applied unchanged to seeds 1 and 2."})
    out.setdefault("grid_sizes", {})
    parts = args.parts.split(",")
    lim = args.ortools_limit

    if "softdist" in parts:  # non-learned control: pick tau by mean Amazon score on val
        sd = {}
        for tau in [0.02, 0.05, 0.1, 0.2, 0.5, 1.0]:
            seqs = []
            for ex in val:
                T = ex["T"].astype(np.float64)
                seqs.append(two_opt(greedy_decode(len(T), ex["edge_index"], softdist_heatmap(T, ex["edge_index"], tau), T), T))
            sd[str(tau)] = _mean_scores(val, seqs)
            print(f"[tune] softdist tau={tau}: {sd[str(tau)]}", flush=True)
        out["softdist"] = sd
        out["softdist_tau"] = float(min(sd, key=lambda k: sd[k]["amazon_score"]))
        out["grid_sizes"]["softdist"] = len(sd)

    if "zone" in parts:
        zl = {}
        for lam in ZONE_LAMS:
            costs = {ex["route_id"]: zone_penalty_matrix(ex["T"].astype(np.float64), ex["zones"], lam) for ex in val}
            cache = run_ortools(f"zone_val_lim{lim:g}_lam{lam:g}", val, costs, lim, args.workers)
            zl[f"{lam:g}"] = {"lam": lam, **_mean_scores(val, [cache[ex["route_id"]]["seq"] for ex in val])}
            print(f"[tune] zone lam={lam:g}: {zl[f'{lam:g}']}", flush=True)
        out["zone"] = zl
        best = min(zl.values(), key=lambda v: v["amazon_score"])
        out["zone_lam"] = best["lam"]
        out["grid_sizes"]["zone"] = len(zl)
        out.setdefault("boundary", {})["zone"] = _boundary(ZONE_LAMS, best["lam"])

    L0 = load_learned(0) if any(p in parts for p in ["diffusion", "hier", "hier_sup"]) else {}
    if "diffusion" in parts and "stop_diffusion" in L0:
        model, proc = L0["stop_diffusion"][:2]
        dg = {}
        for steps, samples in [(5, 1), (10, 1), (20, 1), (10, 4)]:
            seqs, raw, rt = [], [], []
            for ex in val:
                T = ex["T"].astype(np.float64)
                t0 = time.perf_counter()
                h = heatmap(model, proc, ex, steps, samples, 0)
                g = greedy_decode(len(T), ex["edge_index"], h, T)
                seqs.append(two_opt(g, T))
                rt.append(time.perf_counter() - t0)
                raw.append(g)
            key = f"{steps}x{samples}"
            dg[key] = {"steps": steps, "samples": samples, **_mean_scores(val, seqs),
                       "amazon_score_no2opt": _mean_scores(val, raw)["amazon_score"], "runtime_s": float(np.mean(rt))}
            print(f"[tune] diffusion {key}: {dg[key]}", flush=True)
        out["diffusion"] = dg
        best = min(dg.values(), key=lambda v: v["amazon_score"])
        out["diff_steps"], out["diff_samples"] = best["steps"], best["samples"]
        out["grid_sizes"]["diffusion_sampling"] = len(dg)
        out["grid_sizes"]["sup"] = 0

    zparts = [p for p in ["zonehist", "hier", "hier_sup"] if p in parts]
    if zparts:
        zval = {z["route_id"]: z for z in data.load_split("zone_val")}
        for method in zparts:
            if method == "zonehist":
                hm = {rid: history_heatmap(z) for rid, z in zval.items()}
            else:
                kind = "zone_diffusion" if method == "hier" else "zone_supervised"
                if kind not in L0:
                    print(f"[tune] {method}: no seed-0 checkpoint, skipped")
                    continue
                m, proc = L0[kind][:2]
                hm = {ex["route_id"]: heatmap(m, proc, zval[ex["route_id"]], HIER_STEPS, HIER_SAMPLES, i)
                      for i, ex in enumerate(val)}
            res = {}
            for dec in ZONE_DECODERS:
                orders = {rid: decode_zone_order(zval[rid], hm[rid], dec) for rid in hm}
                taus = [zone_order_tau(ex, _seq_from_zone_order(ex, orders[ex["route_id"]])) for ex in val]
                for lam in ZONE_ORDER_LAMS:
                    costs = {ex["route_id"]: zone_order_penalty_matrix(ex["T"].astype(np.float64), ex["zones"],
                                                                       orders[ex["route_id"]], lam) for ex in val}
                    suffix = "" if method == "zonehist" else f"_s{HIER_STEPS}x{HIER_SAMPLES}_seed0"
                    cache = run_ortools(f"{method}_val_lim{lim:g}_lam{lam:g}_{dec}{suffix}", val, costs, lim, args.workers)
                    key = f"lam{lam:g}_{dec}"
                    res[key] = {"lam": lam, "decode": dec, "zone_order_tau": float(np.mean(taus)),
                                **_mean_scores(val, [cache[ex["route_id"]]["seq"] for ex in val])}
                    print(f"[tune] {method} {key}: {res[key]}", flush=True)
            out[method] = res  # the complete record: every configuration tried is kept
            best = min(res.values(), key=lambda v: v["amazon_score"])
            out[f"{method}_lam"], out[f"{method}_decode"] = best["lam"], best["decode"]
            out["grid_sizes"][method] = len(res)
            out.setdefault("boundary", {})[method] = _boundary(ZONE_ORDER_LAMS, best["lam"])
        out["hier_steps"], out["hier_samples"] = HIER_STEPS, HIER_SAMPLES

    with open(path, "w") as f:
        json.dump(out, f, indent=1)
    print(json.dumps({k: v for k, v in out.items() if not isinstance(v, dict) or k in ("grid_sizes", "boundary")}))


# ----------------------------------------------------------------------------------------
# stage: run on a test split
# ----------------------------------------------------------------------------------------
CFG_KEYS = ["softdist_tau", "zone_lam", "diff_steps", "diff_samples", "zonehist_lam", "zonehist_decode", "hier_lam",
            "hier_decode", "hier_sup_lam", "hier_sup_decode", "hier_steps", "hier_samples"]


def run(args):
    tuning = json.load(open(os.path.join(RES, "tuning.json")))
    cfg = {k: tuning[k] for k in CFG_KEYS if k in tuning}
    cfg["ortools_limit"] = args.ortools_limit
    seeds = [int(s) for s in args.seeds.split(",")]
    exs = data.load_split(args.split)
    if args.limit:
        exs = exs[: args.limit]
    zsplit = {z["route_id"]: z for z in data.load_split("zone_" + args.split)}
    torch.set_num_threads(args.threads)
    learned = {s: load_learned(s, budget=(args.split == "fresh")) for s in seeds}

    per_route, seq_store = [], {}
    costs, prep = {}, {}
    old = None
    if args.reuse_fast:
        # Reuse the non-OR-Tools methods of an earlier run of this split when everything they depend on
        # (sampling settings, SoftDist tau, seeds, checkpoints) is unchanged; OR-Tools methods are always
        # re-read from their caches (and solved if their settings changed).
        pp, sp_ = (os.path.join(RES, f"{k}_{args.split}.json") for k in ["per_route", "sequences"])
        if os.path.exists(pp) and os.path.exists(sp_):
            ob = json.load(open(pp))
            same = (all(ob["info"]["config"].get(k) == cfg.get(k) for k in ["softdist_tau", "diff_steps", "diff_samples"])
                    and ob["info"]["seeds"] == seeds and [r["route_id"] for r in ob["per_route"]] == [e["route_id"] for e in exs]
                    and all({k: v["step"] for k, v in ob["info"]["checkpoints"].get(str(sd), {}).items()}
                            == {k: v[2]["step"] for k, v in learned[sd].items()} for sd in seeds))
            if same:
                old = ({r["route_id"]: r for r in ob["per_route"]}, json.load(open(sp_)))
                print(f"[run] reusing the non-OR-Tools methods of the previous {args.split} run", flush=True)
            else:
                print("[run] --reuse-fast: previous run differs, recomputing everything", flush=True)
    fast_keys = None
    t0 = time.time()
    for i, ex in enumerate(exs):
        if old is not None:
            orow, oseq = old[0][ex["route_id"]], old[1][ex["route_id"]]
            if fast_keys is None:
                fast_keys = [k for k in orow if isinstance(orow[k], dict) and base_method(k) not in ORTOOLS_METHODS]
            res = {k: orow[k] for k in fast_keys}
            seqs = {k: oseq[k] for k in fast_keys}
        else:
            res, seqs = run_fast_methods(ex, learned, cfg, route_seed=i)
        per_route.append({"route_id": ex["route_id"], "n_stops": len(ex["stop_ids"]) - 1,
                          "station_code": ex["station_code"], **res})
        seq_store[ex["route_id"]] = {m: np.asarray(s).tolist() for m, s in seqs.items()}
        for m, (c, pt) in ortools_costs(ex, zsplit[ex["route_id"]], learned, cfg, route_seed=i).items():
            costs.setdefault(m, {})[ex["route_id"]] = c
            prep.setdefault(m, {})[ex["route_id"]] = pt
        if (i + 1) % 25 == 0:
            print(f"[run] {args.split}: fast methods {i + 1}/{len(exs)} ({time.time() - t0:.0f}s)", flush=True)

    for method in sorted(costs, key=lambda m: (ORTOOLS_METHODS.index(base_method(m)), m)):
        cache = run_ortools(ortools_tag(method, args.split, cfg), exs, costs[method], args.ortools_limit, args.workers)
        for r, ex in zip(per_route, exs):
            c = cache[ex["route_id"]]
            r[method] = metrics(ex, c["seq"], c["runtime_s"] + prep[method][ex["route_id"]], c.get("cpu_s"))
            seq_store[ex["route_id"]][method] = c["seq"]

    def ckinfo(c, p):
        return {"path": os.path.relpath(p, HERE), "step": c["step"], "train_minutes": c.get("train_seconds", 0) / 60,
                "train_cpu_minutes": (c["train_cpu_seconds"] / 60 if c.get("train_cpu_seconds") is not None else None),
                "threads": c.get("threads"), "model_cfg": c["cfg"]}

    info = {"split": args.split, "config": cfg, "seeds": seeds,
            "checkpoints": {str(s): {k: ckinfo(v[2], v[3]) for k, v in L.items()} for s, L in learned.items()}}
    with open(os.path.join(RES, f"per_route_{args.split}.json"), "w") as f:
        json.dump({"info": info, "per_route": per_route}, f)
    with open(os.path.join(RES, f"sequences_{args.split}.json"), "w") as f:
        json.dump(seq_store, f)
    print(f"[run] wrote per-route results for {len(per_route)} routes")


def budget(args):
    """Sensitivity check for the compute budget. The zone-level learned methods spend a few seconds
    on heat-map sampling and zone decoding BEFORE their 5 s OR-Tools search, so here the OR-Tools
    baselines get a longer limit (default 10 s, more than the total per-route time of the zone
    diffusion method) on the same routes, and their scores are compared with the 5 s runs."""
    tuning = json.load(open(os.path.join(RES, "tuning.json")))
    cfg = {k: tuning[k] for k in CFG_KEYS if k in tuning}
    cfg["ortools_limit"] = args.ortools_limit
    exs = data.load_split(args.split)
    if args.limit:  # smoke tests only
        exs = exs[: args.limit]
    zsplit = {z["route_id"]: z for z in data.load_split("zone_" + args.split)}
    out = {"split": args.split, "limit_s": args.budget_limit, "methods": {}}
    for method in args.budget_methods.split(","):
        costs = {}
        for ex in exs:
            c = ortools_costs(ex, zsplit[ex["route_id"]], {}, cfg)
            costs[ex["route_id"]] = c[method][0]
        cache = run_ortools(ortools_tag(method, args.split, cfg, args.budget_limit), exs, costs, args.budget_limit,
                            args.workers)
        out["methods"][method] = {ex["route_id"]: metrics(ex, cache[ex["route_id"]]["seq"], cache[ex["route_id"]]["runtime_s"],
                                                          cache[ex["route_id"]].get("cpu_s")) for ex in exs}
        print(f"[budget] {method} {args.budget_limit:g} s: mean score "
              f"{np.mean([v['amazon_score'] for v in out['methods'][method].values()]):.4f}", flush=True)
    with open(os.path.join(RES, f"budget_{args.split}_lim{args.budget_limit:g}.json"), "w") as f:
        json.dump(out, f)


# ----------------------------------------------------------------------------------------
# stage: report
# ----------------------------------------------------------------------------------------
def fmt_ci(d, scale=1.0, nd=1):
    return f"{d['mean'] * scale:.{nd}f} [{d['ci95_mean'][0] * scale:.{nd}f}, {d['ci95_mean'][1] * scale:.{nd}f}]"


def fmt_gap(d):
    return f"{d['mean_diff_ref_minus_other']:+.4f} [{d['ci95'][0]:+.4f}, {d['ci95'][1]:+.4f}]"


def budget_summary(split, per_route):
    p = os.path.join(RES, f"budget_{split}_lim{10:g}.json")
    if not os.path.exists(p):
        return None
    b = json.load(open(p))
    out = {"limit_s": b["limit_s"], "methods": {}}
    for m, rows in b["methods"].items():
        if not all(r["route_id"] in rows for r in per_route):
            continue
        long = np.array([rows[r["route_id"]]["amazon_score"] for r in per_route])
        short = np.array([r[m]["amazon_score"] for r in per_route])
        d = {"mean_long": float(long.mean()), "ci95_long": bootstrap_ci(long), "mean_5s": float(short.mean()),
             "long_minus_5s": {"mean": float((long - short).mean()), "ci95": bootstrap_ci(long - short)},
             "runtime_median_s": float(np.median([v["runtime_s"] for v in rows.values()]))}
        for ref in ["hier", "hier_sup"]:
            if ref in per_route[0]:
                x = np.array([r[ref]["amazon_score"] for r in per_route])
                d[f"{ref}5s_minus_long"] = {"mean": float((x - long).mean()), "ci95": bootstrap_ci(x - long),
                                            "better_rate": float(np.mean(x < long))}
        out["methods"][m] = d
    return out


# ----------------------------------------------------------------------------------------
# training compute accounting
# ----------------------------------------------------------------------------------------
# How the runs were executed (the checkpoints of rounds 1 and 2 predate CPU-time recording).
EXEC_MAIN = ("single-threaded; run_training.sh ran 6 such runs at a time on the 4-core machine, so each run received "
             "about 4/6 of a core")
PARALLEL_SHARE = 4.0 / 6.0
BENCH_PATH = os.path.join(RES, "train_cpu_benchmark.json")


def benchmark(args):
    """Measure single-threaded training CPU seconds per optimisation step for every level/objective
    on an otherwise idle machine (a short throw-away run of train.py in a temporary directory).
    Used only to cross-check the compute accounting of runs that did not record CPU time."""
    import subprocess
    import sys
    import tempfile

    out = {"note": "process CPU seconds per step of train.py --threads 1, measured on an idle machine over a "
                   f"{args.bench_minutes:g}-minute throw-away run (includes batching and augmentation, excludes loading)",
           "per_step_cpu_s": {}}
    for level in ["stop", "zone"]:
        for obj in ["diffusion", "supervised"]:
            with tempfile.TemporaryDirectory() as d:
                ck = os.path.join(d, "bench.pt")
                subprocess.run([sys.executable, os.path.join(HERE, "train.py"), "--level", level, "--objective", obj,
                                "--seed", "0", "--threads", "1", "--ckpt", ck, "--max-minutes", str(args.bench_minutes),
                                "--save-every-min", "100"], check=True, stdout=subprocess.DEVNULL)
                c = torch.load(ck, map_location="cpu", weights_only=False)
                v = c["train_cpu_seconds"] / c["step"]
                out["per_step_cpu_s"][f"{level}_{obj}"] = {"cpu_s_per_step": v, "steps_measured": c["step"],
                                                           "wall_s_per_step": c["train_seconds"] / c["step"]}
                print(f"[benchmark] {level}/{obj}: {v:.3f} CPU s per step over {c['step']} steps", flush=True)
    with open(BENCH_PATH, "w") as f:
        json.dump(out, f, indent=1)


def training_compute() -> dict:
    bench = json.load(open(BENCH_PATH))["per_step_cpu_s"] if os.path.exists(BENCH_PATH) else {}
    runs = []

    def add(label, p, group, fallback_share, threads_note):
        ck = torch.load(p, map_location="cpu", weights_only=False)
        cfg = ck["cfg"]
        wall = ck.get("train_seconds", 0.0) / 60
        cpu = ck.get("train_cpu_seconds")
        b = bench.get(f"{cfg.get('level', 'stop')}_{cfg.get('objective', 'diffusion')}")
        row = {"run": label, "group": group, "steps": ck["step"], "wall_minutes": wall, "threads": ck.get("threads"),
               "execution": threads_note,
               "cpu_minutes_measured": None if cpu is None else cpu / 60,
               "cpu_minutes_from_wall": wall * fallback_share,
               "cpu_minutes_from_benchmark": None if b is None else ck["step"] * b["cpu_s_per_step"] / 60}
        row["cpu_minutes"] = row["cpu_minutes_measured"] if cpu is not None else row["cpu_minutes_from_wall"]
        row["cpu_minutes_basis"] = "measured" if cpu is not None else "estimated (wall x share of a core)"
        runs.append(row)

    for level in ["zone", "stop"]:
        for obj in ["diffusion", "supervised"]:
            for seed in [0, 1, 2]:
                p = model_path(level, obj, seed)
                if p:
                    add(f"{level}/{obj} seed {seed}", p, "main (round 2)", PARALLEL_SHARE, EXEC_MAIN)
    p = model_path("stop", "diffusion", 0, "budget")
    if p:
        add("stop/diffusion seed 0, spec-budget replicate (3,000 steps)", p, "spec-budget replicate (round 3)", 1.0,
            "single-threaded on its own core (process CPU time recorded)")
    for name in ["diffusion", "zone_diffusion"]:
        p = os.path.join(HERE, "checkpoints", "round1", f"{name}.pt")
        if os.path.exists(p):
            add(f"round 1 {'zone' if 'zone' in name else 'stop'}/diffusion (discarded)", p, "round 1 (discarded)", 4.0,
                "4 torch threads, shared machine; CPU given as the upper bound 4 x wall")
    tot = {}
    for r in runs:
        tot.setdefault(r["group"], 0.0)
        tot[r["group"]] += r["cpu_minutes"]
    budget_pipeline = [r for r in runs if r["run"] in ("zone/diffusion seed 0",) or r["group"].startswith("spec-budget")]
    return {"runs": runs, "totals_cpu_minutes": tot, "total_cpu_minutes": float(sum(tot.values())),
            "spec_budget_cpu_minutes": [45, 90],
            "spec_budget_pipeline": {"runs": [r["run"] for r in budget_pipeline],
                                     "cpu_minutes": float(sum(r["cpu_minutes"] for r in budget_pipeline))},
            "benchmark": bench}


# ----------------------------------------------------------------------------------------
# stage: report
# ----------------------------------------------------------------------------------------
SPLIT_TITLES = {
    "fresh": "FINAL test set: fresh evaluation-dataset routes, first evaluated after all settings were frozen",
    "test": "REUSED test set: evaluation-dataset routes already scored in round 1",
    "heldout": "REUSED held-out set: High-quality training-dataset routes already scored in round 1",
}
HEADLINE_GAPS = [
    ("diffusion - sup", "Stop level, greedy + 2-opt: diffusion minus one-shot supervised (same network)"),
    ("diffusion_greedy - sup_greedy", "Stop level, greedy only: diffusion minus one-shot supervised"),
    ("hier - hier_sup", "Zone level + OR-Tools: diffusion minus one-shot supervised (same network)"),
    ("hier - zonehist", "Zone level + OR-Tools: diffusion minus history-only control"),
    ("hier_sup - zonehist", "Zone level + OR-Tools: one-shot supervised minus history-only control"),
    ("diffusion - softdist", "Stop level + 2-opt: diffusion minus SoftDist (non-learned)"),
    ("hier - zone", "Zone diffusion + OR-Tools minus M_zone heuristic"),
    ("hier_sup - zone", "One-shot zone GNN + OR-Tools minus M_zone heuristic"),
]


def _verdict(ci):
    return "A better" if ci[1] < 0 else ("B better" if ci[0] > 0 else "no significant difference")


def report(args):
    results = {"description": "Diffusion vs. baselines on Amazon Last Mile routes (see README.md)", "splits": {},
               "split_status": {"fresh": "final, untouched until all settings were frozen",
                                "test": "reused (already evaluated in round 1)",
                                "heldout": "reused (already evaluated in round 1)"}}
    tuning = json.load(open(os.path.join(RES, "tuning.json")))
    results["tuning_on_val"] = tuning
    r1p = os.path.join(RES, "tuning_round1.json")
    if os.path.exists(r1p):
        results["tuning_round1_on_val"] = json.load(open(r1p))
    dgp = os.path.join(RES, "diagnostics.json")
    if os.path.exists(dgp):
        results["diagnostics"] = json.load(open(dgp))
    tc = training_compute()
    results["training_compute"] = tc
    stats = json.load(open(os.path.join(RES, "data_stats.json")))

    blobs = {}
    for split in SPLITS:
        p = os.path.join(RES, f"per_route_{split}.json")
        if not os.path.exists(p):
            continue
        blob = json.load(open(p))
        per_route = blob["per_route"]
        methods = [m for m in METHOD_ORDER if m in per_route[0]]
        summ = summarize(per_route, methods)
        seeds = blob["info"].get("seeds", [0])
        ss = seed_summary(per_route, seeds) if len(seeds) > 1 else None
        bs = budget_summary(split, per_route)
        results["splits"][split] = {"status": results["split_status"][split], "info": blob["info"], "summary": summ,
                                    "seed_summary": ss, "budget_sensitivity": bs, "per_route": per_route}
        blobs[split] = (summ, ss, bs, blob, methods, seeds)

    md = ["# Results: diffusion-generated delivery sequences vs. baselines", "",
          "All numbers are produced by `python evaluate.py report` from the per-route files. Lower Amazon score means closer "
          "to the driver's executed sequence; the driver scores 0. Every setting of every method was chosen on the 60 "
          "validation routes; no setting was selected by a test score.", ""]
    md += ["## Status of the test sets (read this first)", ""]
    if "fresh" in blobs:
        md.append(f"* **fresh ({blobs['fresh'][0]['n_routes']} routes): the final test set.** Drawn in round 3 at random "
                  f"(seed 1) from the {stats.get('n_eval_not_in_test', 'remaining'):,} routes of the official evaluation "
                  "dataset that are not in the old test split, and evaluated once, after every setting of every method had "
                  "been frozen on the validation routes. No decision in this study was informed by it. The headline numbers "
                  "come from this set.")
    md.append("* **test (300 routes) and heldout (150 High routes): reused, no longer untouched.** Both were already "
              "evaluated in round 1 of this study (archived in results/cache_round1/published_outputs). After seeing those "
              "results the models were retrained, the one-shot supervised ablations and training seeds 1 and 2 were added, "
              "every tuning grid was widened (twice), and the tuned M_zone penalty moved from 0.5 to 4. No setting was chosen "
              "by a test score, but the study design was revised after test results had been seen, so these sets are "
              "reported as development results only. There are no unused High-quality routes left for a fresh held-out set.")
    md.append("")

    # headline ablation table
    md += ["## Headline: does the diffusion process add anything over a supervised forward pass?", "",
           "Seed-averaged paired differences in mean Amazon score (per route, each learned method's score is first averaged "
           "over training seeds 0, 1 and 2; bootstrap 95% CI over routes). A negative value favours the first-named method "
           "(A). The one-shot supervised model is the identical network, features, data, batch size, optimiser, schedule, "
           "number of steps and decoder as the diffusion model; only the diffusion process is removed.", "",
           "| A - B | " + " | ".join(f"{s} ({blobs[s][0]['n_routes']} routes)" for s in blobs) + " |",
           "|---|" + "---|" * len(blobs)]
    headline = {}
    for g, title in HEADLINE_GAPS:
        cells = []
        for s in blobs:
            ss = blobs[s][1]
            if ss and g in ss["gaps"]:
                sa = ss["gaps"][g]["seed_averaged"]
                cells.append(f"{sa['mean_diff']:+.4f} [{sa['ci95'][0]:+.4f}, {sa['ci95'][1]:+.4f}] ({_verdict(sa['ci95'])})")
                headline.setdefault(g, {})[s] = sa
            else:
                cells.append("n/a")
        md.append(f"| {title} | " + " | ".join(cells) + " |")
    results["headline_gaps"] = headline
    prim = "fresh" if "fresh" in blobs else next(iter(blobs), None)
    if prim and blobs[prim][1]:
        ss = blobs[prim][1]
        g1, g2, g3 = (ss["gaps"].get(k, {}).get("seed_averaged") for k in
                      ["diffusion - sup", "diffusion_greedy - sup_greedy", "hier - hier_sup"])
        if g1 and g2 and g3:
            md += ["", f"Reading on the {prim} split: at stop level the diffusion model minus its one-shot supervised twin is "
                   f"{g1['mean_diff']:+.4f} [{g1['ci95'][0]:+.4f}, {g1['ci95'][1]:+.4f}] with 2-opt and {g2['mean_diff']:+.4f} "
                   f"[{g2['ci95'][0]:+.4f}, {g2['ci95'][1]:+.4f}] without; at zone level it is {g3['mean_diff']:+.4f} "
                   f"[{g3['ci95'][0]:+.4f}, {g3['ci95'][1]:+.4f}]. Positive values mean the diffusion model is further from "
                   "the driver. " + ("In every case the diffusion process is significantly worse than a single supervised "
                                     "forward pass of the same network, and it is also slower, so the diffusion process adds "
                                     "no measurable benefit here." if min(g1["ci95"][0], g2["ci95"][0], g3["ci95"][0]) > 0 else
                                     "See the table for which differences are significant.")]
    md.append("")

    # training compute
    md += ["## Training compute", "",
           f"The specification asked for about {tc['spec_budget_cpu_minutes'][0]} to {tc['spec_budget_cpu_minutes'][1]} "
           f"CPU-minutes of training in total. This study used about {tc['total_cpu_minutes']:,.0f} CPU-minutes, roughly "
           f"{tc['total_cpu_minutes'] / tc['spec_budget_cpu_minutes'][1]:.0f} times the upper end, so the budget was exceeded. "
           "The excess comes from the controls, not from the model under test: three seeds per model (to measure run-to-run "
           "variance) and the one-shot supervised twin of every diffusion model (the ablation that tests whether the "
           "diffusion process matters) multiply one stop-level and one zone-level run by 6, plus two discarded round-1 runs. "
           "Rounds 1 and 2 did not record CPU time, so for those runs it is estimated from wall-clock time and the share of a "
           "core each run received; an independent estimate (steps x single-thread CPU seconds per step measured on an idle "
           "machine, `python evaluate.py benchmark`) is shown alongside.", "",
           "| Run | Group | Steps | Wall min | CPU min (basis) | CPU min from benchmark | Execution |", "|---|---|---|---|---|---|---|"]
    for r in tc["runs"]:
        bm = "" if r["cpu_minutes_from_benchmark"] is None else f"{r['cpu_minutes_from_benchmark']:.0f}"
        md.append(f"| {r['run']} | {r['group']} | {r['steps']:,} | {r['wall_minutes']:.1f} | {r['cpu_minutes']:.0f} "
                  f"({r['cpu_minutes_basis']}) | {bm} | {r['execution']} |")
    md += ["", "Totals by group: " + "; ".join(f"{k}: {v:,.0f} CPU-minutes" for k, v in tc["totals_cpu_minutes"].items())
           + ".", "",
           f"Spec-budget pipeline: {', '.join(tc['spec_budget_pipeline']['runs'])} use "
           f"{tc['spec_budget_pipeline']['cpu_minutes']:.0f} CPU-minutes together, inside the specified budget. The "
           "spec-budget stop-level replicate is reported as `M_diffusion, spec-budget model` on the final test set (it uses "
           "the sampling settings tuned for the full model and was not tuned itself).", ""]

    for split, (summ, ss, bs, blob, methods, seeds) in blobs.items():
        md += [f"## {SPLIT_TITLES[split]} ({summ['n_routes']} routes)", ""]
        md += ["Amazon score and travel time: mean with bootstrap 95% confidence interval (10,000 resamples of routes) and "
               "median. Learned methods: seed-0 models in this table (all seeds below). Travel time is the closed tour from "
               "the station back to the station on the real travel-time matrix. Zone tau is the Kendall rank correlation "
               "between the zone order of the sequence and the driver's zone order. Runtime is wall-clock seconds per route, "
               "median and interquartile range.", ""]
        md += ["| Method | Feasible | Amazon score, mean [95% CI] | Amazon score, median | Travel time (h), mean [95% CI] "
               "| Travel time (h), median | Driver edges reproduced | Zone tau | Runtime (s), median [IQR] |",
               "|---|---|---|---|---|---|---|---|---|"]
        for m in methods:
            d = summ["methods"][m]
            rt = d["runtime_s"]
            md.append(f"| {METHOD_LABELS[m]} | {d['feasible_rate'] * 100:.0f}% | {fmt_ci(d['amazon_score'], 1, 4)} | "
                      f"{d['amazon_score']['median']:.4f} | {fmt_ci(d['travel_time_s'], 1 / 3600, 3)} | "
                      f"{d['travel_time_s']['median'] / 3600:.3f} | "
                      f"{d['driver_edge_overlap']['mean'] * 100:.1f}% | {d['zone_order_tau']['mean']:.3f} | "
                      f"{rt['median']:.2f} [{rt['q25']:.2f}, {rt['q75']:.2f}] |")
        for ref, title in [("diffusion", "M_diffusion (stop-level, greedy + 2-opt, seed 0)"),
                           ("hier", "M_diffusion_zone (zone-level diffusion + OR-Tools, seed 0)"),
                           ("hier_sup", "M_supervised_zone (one-shot zone GNN + OR-Tools, seed 0)")]:
            if ref not in summ["paired"]:
                continue
            md += ["", f"Paired comparisons per route: {title} minus the other method. Negative means {ref} is better "
                   "(lower score or shorter travel time).", "",
                   "| Other method | Amazon score diff, mean [95% CI] | Reference better / identical on | Wilcoxon p | "
                   "Travel time diff (min), mean [95% CI] | Reference shorter on | Wilcoxon p |", "|---|---|---|---|---|---|---|"]
            for m, dd in summ["paired"][ref].items():
                a, t = dd["amazon_score"], dd["travel_time_s"]
                md.append(f"| {METHOD_LABELS[m]} | {fmt_gap(a)} | "
                          f"{a['ref_better_rate'] * 100:.0f}% / {a['tie_rate'] * 100:.0f}% | {a['wilcoxon_p']:.2g} | "
                          f"{t['mean_diff_ref_minus_other'] / 60:+.1f} [{t['ci95'][0] / 60:+.1f}, {t['ci95'][1] / 60:+.1f}] | "
                          f"{t['ref_better_rate'] * 100:.0f}% | {t['wilcoxon_p']:.2g} |")
        if ss:
            md += ["", f"Between-seed spread of the learned methods (seeds {', '.join(map(str, seeds))}; each seed is a "
                   "separately trained model with its own initialisation and data order). Mean Amazon score per seed, and the "
                   "mean and sample standard deviation of those seed means.", "",
                   "| Method | " + " | ".join(f"seed {s}" for s in seeds) + " | Across seeds, mean (sd) | Range |",
                   "|---|" + "---|" * len(seeds) + "---|---|"]
            for m, d in ss["methods"].items():
                md.append(f"| {METHOD_LABELS[m]} | " + " | ".join(f"{d['per_seed'][str(s)]['mean']:.4f}" for s in seeds)
                          + f" | {d['across_seeds']['mean']:.4f} ({d['across_seeds']['sd']:.4f}) | "
                            f"{d['across_seeds']['min']:.4f} to {d['across_seeds']['max']:.4f} |")
            md += ["", "Paired gaps per seed (method A minus method B, mean Amazon score difference with bootstrap 95% CI; "
                   "negative favours A). Learned methods are paired seed with seed. The last column first averages each "
                   "method's per-route score over the three seeds and then compares.", "",
                   "| A - B | " + " | ".join(f"seed {s}" for s in seeds) + " | Across seeds, mean (sd) | Seed-averaged [95% CI] |",
                   "|---|" + "---|" * len(seeds) + "---|---|"]
            for g, d in ss["gaps"].items():
                sa = d["seed_averaged"]
                md.append(f"| {g} | " + " | ".join(fmt_gap(d["per_seed"][str(s)]) for s in seeds)
                          + f" | {d['across_seeds']['mean']:+.4f} ({d['across_seeds']['sd']:.4f}) | "
                            f"{sa['mean_diff']:+.4f} [{sa['ci95'][0]:+.4f}, {sa['ci95'][1]:+.4f}] |")
        orm = [m for m in methods if "ortools_cpu_over_wall" in summ["methods"][m]]
        if orm:
            md += ["", "OR-Tools compute actually received (each solve runs in its own single-threaded worker process, 4 "
                   "workers on the 4-core machine): median CPU seconds per solve and the ratio of CPU time to wall-clock time "
                   "(1.0 means the solver had a full core for its whole limit; lower values reveal contention).", "",
                   "| Method | CPU s per solve, median | CPU / wall, median | CPU / wall, minimum |", "|---|---|---|---|"]
            for m in orm:
                d = summ["methods"][m]
                md.append(f"| {METHOD_LABELS[m]} | {d['ortools_cpu_s']['median']:.2f} | "
                          f"{d['ortools_cpu_over_wall']['median']:.3f} | {d['ortools_cpu_over_wall']['min']:.3f} |")
        if bs and bs["methods"]:
            hr = summ["methods"]["hier"]["runtime_s"]
            md += ["", f"Compute-budget sensitivity: the OR-Tools baselines re-run with a {bs['limit_s']:g} s limit on the same "
                   f"routes. For comparison, M_diffusion_zone spends a median of {hr['median']:.2f} s per route (upper quartile "
                   f"{hr['q75']:.2f} s) on heat-map sampling, zone decoding and its 5 s search together.", "",
                   f"| Method | Mean score, 5 s | Mean score, {bs['limit_s']:g} s [95% CI] | {bs['limit_s']:g} s minus 5 s [95% CI] | "
                   f"M_diffusion_zone (5 s) minus {bs['limit_s']:g} s run [95% CI] | M_supervised_zone (5 s) minus "
                   f"{bs['limit_s']:g} s run [95% CI] |", "|---|---|---|---|---|---|"]
            for m, d in bs["methods"].items():
                cells = []
                for ref in ["hier", "hier_sup"]:
                    x = d.get(f"{ref}5s_minus_long")
                    cells.append("n/a" if x is None else f"{x['mean']:+.4f} [{x['ci95'][0]:+.4f}, {x['ci95'][1]:+.4f}] "
                                                         f"(better on {x['better_rate'] * 100:.0f}%)")
                md.append(f"| {METHOD_LABELS[m]} | {d['mean_5s']:.4f} | {d['mean_long']:.4f} [{d['ci95_long'][0]:.4f}, "
                          f"{d['ci95_long'][1]:.4f}] | {d['long_minus_5s']['mean']:+.4f} [{d['long_minus_5s']['ci95'][0]:+.4f}, "
                          f"{d['long_minus_5s']['ci95'][1]:+.4f}] | " + " | ".join(cells) + " |")
        c = blob["info"]["config"]
        md += ["", f"Settings (all chosen on the 60 validation routes, see the tuning section): diffusion sampling "
               f"{c['diff_steps']} denoising steps x {c['diff_samples']} samples with heat-maps averaged; SoftDist tau = "
               f"{c['softdist_tau']}; zone-change penalty {c['zone_lam']:g} x median travel time; zone-order penalty and "
               f"zone decoder: zone diffusion {c.get('hier_lam', 0):g} / {c.get('hier_decode')} ({HIER_STEPS} steps x "
               f"{HIER_SAMPLES} samples), one-shot zone GNN {c.get('hier_sup_lam', 0):g} / {c.get('hier_sup_decode')}, history "
               f"{c.get('zonehist_lam', 0):g} / {c.get('zonehist_decode')}; OR-Tools limit {c['ortools_limit']:g} s per route.",
               ""]

    # tuning record
    t = tuning
    md += ["## Tuning record (validation split, 60 routes)", "",
           "Every configuration evaluated is listed in `results/tuning.json`"
           + ("; the full round-1 record is in `results/tuning_round1.json`." if os.path.exists(r1p) else ".")
           + " 'Position in grid' says whether the chosen penalty lies on the edge of its grid.", "",
           "| Method | Settings tried | Chosen | Position in grid | Validation score of chosen |", "|---|---|---|---|---|"]
    zl = f"zone-change penalty lam in {{1/16, 1/8, ..., {ZONE_LAMS[-1]:g}}} ({len(ZONE_LAMS)})"
    zo = (f"{len(ZONE_DECODERS)} zone decoders x lam in {{{', '.join(f'{v:g}' for v in ZONE_ORDER_LAMS)}}} "
          f"({len(ZONE_DECODERS) * len(ZONE_ORDER_LAMS)})")
    for m, desc, chosen in [
        ("zone", zl, f"lam {t.get('zone_lam', 0):g}"),
        ("zonehist", zo, f"{t.get('zonehist_decode')}, lam {t.get('zonehist_lam', 0):g}"),
        ("hier_sup", zo, f"{t.get('hier_sup_decode')}, lam {t.get('hier_sup_lam', 0):g}"),
        ("hier", zo, f"{t.get('hier_decode')}, lam {t.get('hier_lam', 0):g}"),
    ]:
        if m not in t:
            continue
        best = min(t[m].values(), key=lambda v: v["amazon_score"])
        md.append(f"| {METHOD_LABELS[m]} | {desc} | {chosen} | {t.get('boundary', {}).get(m, '')} | {best['amazon_score']:.4f} |")
    if "diffusion" in t:
        best = min(t["diffusion"].values(), key=lambda v: v["amazon_score"])
        md.append(f"| {METHOD_LABELS['diffusion']} | denoising steps x samples in {{5x1, 10x1, 20x1, 10x4}} (4) | "
                  f"{t['diff_steps']}x{t['diff_samples']} | | {best['amazon_score']:.4f} |")
    if "softdist" in t:
        best = min(t["softdist"].values(), key=lambda v: v["amazon_score"])
        md.append(f"| {METHOD_LABELS['softdist']} | tau in {{0.02, 0.05, 0.1, 0.2, 0.5, 1}} (6) | tau {t['softdist_tau']:g} | | "
                  f"{best['amazon_score']:.4f} |")
    md.append(f"| {METHOD_LABELS['sup']} | none (deterministic one-shot heat-map, same decoder) | | | |")
    for m in ["hier", "hier_sup", "zonehist"]:
        if m in t and f"{m}_decode" in t:
            dec = t[f"{m}_decode"]
            row = [f"{lam:g}: {t[m][f'lam{lam:g}_{dec}']['amazon_score']:.4f}" for lam in ZONE_ORDER_LAMS
                   if f"lam{lam:g}_{dec}" in t[m]]
            md.append("")
            md.append(f"Validation score of {METHOD_LABELS[m]} by order penalty lam (chosen decoder {dec}): " + ", ".join(row) + ".")
    if t.get("boundary", {}).get("hier") in ("upper", "lower"):
        md += ["", f"Note: the chosen zone-diffusion penalty (lam {t.get('hier_lam', 0):g}) lies on the {t['boundary']['hier']} "
               "edge of the grid. The penalty-saturation diagnostic below shows whether larger values can still change the "
               "solutions."]
    if os.path.exists(r1p):
        r1 = json.load(open(r1p))
        md += ["", "Round 1 (earlier checkpoints, retrained since) tried "
               + ", ".join(f"{v} configurations for {k}" for k, v in r1["counts"].items()) + " on the same validation routes."]
    if "diagnostics" in results:
        dg = results["diagnostics"]
        md += ["", "## Diagnostics (python diagnostics.py; results/diagnostics.json; validation and training routes only)", ""]
        for line in dg.get("summary_lines", []):
            md.append(line)
    with open(os.path.join(RES, "results.json"), "w") as f:
        json.dump(results, f, indent=1)
    with open(os.path.join(RES, "summary.md"), "w") as f:
        f.write("\n".join(md) + "\n")
    print("\n".join(md))


def export(args):
    """Copy the EMA weights and configs of every finished checkpoint to results/models (small files
    that load_model can read directly; the full resumable checkpoints stay in checkpoints/)."""
    out_dir = args.export_dir or os.path.join(RES, "models")
    os.makedirs(out_dir, exist_ok=True)
    for seed in [0, 1, 2]:
        for key, (_, _, ck, p) in load_learned(seed, budget=True).items():
            name = os.path.basename(p)
            dst = os.path.join(out_dir, name if name.endswith("_ema.pt") else name.replace(".pt", "_ema.pt"))
            if os.path.abspath(dst) == os.path.abspath(p):
                print("already exported", os.path.relpath(dst, HERE))
                continue
            torch.save({"cfg": ck["cfg"], "ema": ck["ema"], "step": ck["step"], "train_seconds": ck.get("train_seconds"),
                        "train_cpu_seconds": ck.get("train_cpu_seconds"), "threads": ck.get("threads")}, dst)
            print("wrote", os.path.relpath(dst, HERE))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["round1", "tune", "run", "budget", "report", "export", "benchmark"])
    ap.add_argument("--split", default="fresh", choices=["fresh", "test", "heldout", "val"])
    ap.add_argument("--seeds", default="0,1,2", help="run: model seeds to evaluate")
    ap.add_argument("--ortools-limit", type=float, default=5.0)
    ap.add_argument("--budget-limit", type=float, default=10.0)
    ap.add_argument("--budget-methods", default="zone,zonehist,ortools")
    ap.add_argument("--bench-minutes", type=float, default=1.5, help="benchmark: minutes per throw-away run")
    ap.add_argument("--export-dir", default=None, help="export: output directory (default results/models)")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--reuse-fast", action="store_true",
                    help="run: reuse the non-OR-Tools methods of an earlier run of the same split if their settings match")
    ap.add_argument("--parts", default="softdist,zone,diffusion,zonehist,hier,hier_sup", help="tune: which settings to (re)tune")
    a = ap.parse_args(argv)
    os.makedirs(RES, exist_ok=True)
    {"round1": round1, "tune": tune, "run": run, "budget": budget, "report": report, "export": export,
     "benchmark": benchmark}[a.stage](a)


if __name__ == "__main__":
    main()
