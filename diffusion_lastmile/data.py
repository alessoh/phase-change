"""Download and preprocess the Amazon Last Mile Routing Research Challenge (2021) data.

Data source (public AWS Open Data bucket, no credentials needed):
    https://amazon-last-mile-challenges.s3.amazonaws.com/almrrc2021/
License: Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0),
see data/raw/License.txt after download. This code only uses the data for non-commercial research.

What this script produces (under data/processed/):
    train.pkl    High-quality training routes used to fit the diffusion model
    val.pkl      small High-quality split used to monitor the training loss (the final
                 checkpoint is always the one evaluated; val is NOT used to select checkpoints)
                 and to tune every setting of every method (evaluate.py tune)
    heldout.pkl  High-quality training-set routes that are never trained on (secondary test;
                 REUSED: it was already scored in round 1 of this study, see README)
    test.pkl     routes sampled from the SEPARATE official evaluation dataset (REUSED: also
                 scored in round 1, before models were retrained and grids re-tuned)
    fresh.pkl    (preprocess-fresh) a further random sample of evaluation-dataset routes that is
                 disjoint from test.pkl and was first evaluated only after every setting of every
                 method had been frozen: the FINAL test set
and results/splits.json with the route ids of every split.

Each processed route is a dict of numpy arrays (see build_example) holding node features,
a sparse directed k-nearest-neighbour graph with edge features, the Bernoulli edge target
(adjacency of the driver's executed closed tour) and the full real travel-time matrix.

Usage:
    python data.py download
    python data.py preprocess [--k 16] [--n-test 300] [--n-heldout 150] [--n-val 60]
    python data.py preprocess-fresh [--n-fresh 300] [--seed 1]
"""
from __future__ import annotations

import argparse
import json
import math
import os
import pickle
import re
import subprocess
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "data")
RAW_DIR = os.path.join(DATA_DIR, "raw")
PROC_DIR = os.path.join(DATA_DIR, "processed")
REF_DIR = os.path.join(DATA_DIR, "reference")
RESULTS_DIR = os.path.join(HERE, "results")

BUCKET = "https://amazon-last-mile-challenges.s3.amazonaws.com/almrrc2021"
FILES = {
    "License.txt": "License.txt",
    "Readme.txt": "Readme.txt",
    "route_data.json": "almrrc2021-data-training/model_build_inputs/route_data.json",
    "actual_sequences.json": "almrrc2021-data-training/model_build_inputs/actual_sequences.json",
    "invalid_sequence_scores.json": "almrrc2021-data-training/model_build_inputs/invalid_sequence_scores.json",
    "travel_times.json": "almrrc2021-data-training/model_build_inputs/travel_times.json",
    "eval_route_data.json": "almrrc2021-data-evaluation/model_apply_inputs/eval_route_data.json",
    "eval_travel_times.json": "almrrc2021-data-evaluation/model_apply_inputs/eval_travel_times.json",
    "eval_actual_sequences.json": "almrrc2021-data-evaluation/model_score_inputs/eval_actual_sequences.json",
    "eval_invalid_sequence_scores.json": "almrrc2021-data-evaluation/model_score_inputs/eval_invalid_sequence_scores.json",
}
# Official challenge scoring script (used only to cross-check score.py in the tests).
OFFICIAL_SCORE_URL = "https://raw.githubusercontent.com/MIT-CAVE/rc-cli/main/scoring/score.py"

NODE_FEAT_NAMES = ["x", "y", "is_station", "zone_rank", "super_rank", "sub_letter", "n_stops"]
EDGE_FEAT_NAMES = [
    "tt_ij", "tt_ji", "log_tt_ij", "knn_rank", "euclid",
    "same_zone", "same_super", "same_sub", "d_n1", "d_n2", "d_letter",
    "zone_rank_diff", "station_edge",
]
ZONE_RE = re.compile(r"^([A-Z])-(\d+)\.(\d+)([A-Z])$")


# ----------------------------------------------------------------------------------------
# Download
# ----------------------------------------------------------------------------------------
def download(force: bool = False) -> None:
    os.makedirs(RAW_DIR, exist_ok=True)
    os.makedirs(REF_DIR, exist_ok=True)
    for local, remote in FILES.items():
        dst = os.path.join(RAW_DIR, local)
        if os.path.exists(dst) and os.path.getsize(dst) > 0 and not force:
            print(f"[download] have {local} ({os.path.getsize(dst) / 1e6:.1f} MB)")
            continue
        url = f"{BUCKET}/{remote}"
        print(f"[download] {url}")
        subprocess.run(["curl", "-sS", "--fail", "--retry", "5", "-C", "-", url, "-o", dst], check=True)
    dst = os.path.join(REF_DIR, "official_score.py")
    if not os.path.exists(dst) or force:
        print(f"[download] {OFFICIAL_SCORE_URL}")
        r = subprocess.run(["curl", "-sS", "--fail", OFFICIAL_SCORE_URL, "-o", dst])
        if r.returncode != 0:
            print("[download] could not fetch the official scoring script; tests will skip the cross-check")


# ----------------------------------------------------------------------------------------
# Zone helpers
# ----------------------------------------------------------------------------------------
def _valid_zone(z) -> bool:
    return isinstance(z, str) and len(z) > 0


def parse_zone(z: str):
    """'P-12.3C' -> ('P', 12, 3, 'C'). Unparseable strings keep only exact-match information."""
    m = ZONE_RE.match(z)
    if m is None:
        return (z, -1, -1, "?")
    return (m.group(1), int(m.group(2)), int(m.group(3)), m.group(4))


def zone_sort_key(z: str):
    p = parse_zone(z)
    return (p[0], p[1], p[2], p[3])


# ----------------------------------------------------------------------------------------
# Example construction
# ----------------------------------------------------------------------------------------
def build_example(route_id: str, route: dict, actual: dict, tt: dict, k: int = 16) -> dict:
    """Turn one raw route into model-ready arrays.

    Node 0 is always the station; the remaining stops are sorted by stop id so that the node
    order carries no information about the driver's sequence.
    """
    stops = route["stops"]
    station = [s for s, v in stops.items() if v["type"] == "Station"]
    assert len(station) == 1, route_id
    station = station[0]
    others = sorted(s for s in stops if s != station)
    ids = [station] + others
    n = len(ids)
    idx = {s: i for i, s in enumerate(ids)}

    # Real travel-time matrix in seconds (full, including the zero diagonal as in the raw file).
    T = np.zeros((n, n), dtype=np.float64)
    for a in ids:
        row = tt[a]
        ia = idx[a]
        for b in ids:
            T[ia, idx[b]] = float(row[b])
    T32 = T.astype(np.float32)

    lat = np.array([stops[s]["lat"] for s in ids], dtype=np.float64)
    lng = np.array([stops[s]["lng"] for s in ids], dtype=np.float64)

    # Zone ids; the station has none, a few stops have NaN: impute from the nearest stop (by
    # travel time) that has a valid zone.
    zones = [stops[s]["zone_id"] if _valid_zone(stops[s]["zone_id"]) else None for s in ids]
    zones[0] = None
    valid = [i for i in range(1, n) if zones[i] is not None]
    for i in range(1, n):
        if zones[i] is None:
            if valid:
                j = min(valid, key=lambda v: T[i, v] + T[v, i])
                zones[i] = zones[j]
            else:
                zones[i] = "UNK"
    parsed = [None] + [parse_zone(z) for z in zones[1:]]

    # --- node features ------------------------------------------------------------------
    lat0 = float(np.mean(lat[1:]))
    xkm = (lng - float(np.mean(lng[1:]))) * math.cos(math.radians(lat0)) * 111.32
    ykm = (lat - lat0) * 110.57
    xmin, xmax = xkm[1:].min(), xkm[1:].max()
    ymin, ymax = ykm[1:].min(), ykm[1:].max()
    scale = max(xmax - xmin, ymax - ymin, 1e-3)
    cx, cy = (xmin + xmax) / 2, (ymin + ymax) / 2
    x = np.clip((xkm - cx) / scale + 0.5, -2.0, 3.0)
    y = np.clip((ykm - cy) / scale + 0.5, -2.0, 3.0)

    uz = sorted(set(zones[1:]), key=zone_sort_key)
    zrank = {z: r / max(1, len(uz) - 1) for r, z in enumerate(uz)}
    supers = sorted(set((p[0], p[1]) for p in parsed[1:]))
    srank = {s: r / max(1, len(supers) - 1) for r, s in enumerate(supers)}
    node = np.zeros((n, len(NODE_FEAT_NAMES)), dtype=np.float32)
    node[:, 0] = x
    node[:, 1] = y
    node[0, 2] = 1.0
    for i in range(1, n):
        p = parsed[i]
        node[i, 3] = zrank[zones[i]]
        node[i, 4] = srank[(p[0], p[1])]
        node[i, 5] = (ord(p[3]) - ord("A")) / 25.0 if p[3] != "?" else 0.0
    node[:, 6] = n / 200.0

    # --- sparse directed graph ---------------------------------------------------------
    kk = min(k, n - 2)
    Ts = T[1:, 1:].copy()
    np.fill_diagonal(Ts, np.inf)
    nbr = np.argsort(Ts, axis=1)[:, :kk] + 1  # node indices of the kk nearest stops
    rank = {}
    for i in range(1, n):
        for r, j in enumerate(nbr[i - 1]):
            rank[(i, int(j))] = r
    edges = set()
    for (i, j) in rank:
        edges.add((i, j))
        edges.add((j, i))
    for i in range(1, n):  # station is connected to every stop in both directions
        edges.add((0, i))
        edges.add((i, 0))
    edges = sorted(edges)
    E = len(edges)
    src = np.array([e[0] for e in edges], dtype=np.int64)
    dst = np.array([e[1] for e in edges], dtype=np.int64)

    off = T[1:, 1:][~np.eye(n - 1, dtype=bool)]
    tscale = float(np.median(off)) if off.size else 1.0
    tscale = max(tscale, 1.0)
    ef = np.zeros((E, len(EDGE_FEAT_NAMES)), dtype=np.float32)
    ef[:, 0] = np.minimum(T[src, dst] / tscale, 10.0)
    ef[:, 1] = np.minimum(T[dst, src] / tscale, 10.0)
    ef[:, 2] = np.log1p(T[src, dst] / tscale)
    ef[:, 3] = np.array([rank.get((a, b), kk) / kk for a, b in edges], dtype=np.float32)
    ef[:, 4] = np.minimum(np.hypot(x[src] - x[dst], y[src] - y[dst]), 3.0)
    for e, (a, b) in enumerate(edges):
        if a == 0 or b == 0:
            ef[e, 12] = 1.0
            continue
        pa, pb = parsed[a], parsed[b]
        same_letter = pa[0] == pb[0]
        same_super = same_letter and pa[1] == pb[1]
        same_sub = same_super and pa[2] == pb[2]
        ef[e, 5] = float(zones[a] == zones[b])
        ef[e, 6] = float(same_super)
        ef[e, 7] = float(same_sub)
        ef[e, 8] = min(abs(pa[1] - pb[1]) / 5.0, 1.0) if same_letter else 1.0
        ef[e, 9] = min(abs(pa[2] - pb[2]) / 5.0, 1.0) if same_super else 1.0
        ef[e, 10] = min(abs(ord(pa[3]) - ord(pb[3])) / 5.0, 1.0) if same_sub else 1.0
        ef[e, 11] = zrank[zones[b]] - zrank[zones[a]]

    # --- target: directed adjacency of the executed closed tour ------------------------
    order = sorted(actual.keys(), key=lambda s: actual[s])
    assert order[0] == station and len(order) == n, route_id
    seq = np.array([idx[s] for s in order], dtype=np.int64)
    tour_edges = set(zip(seq.tolist(), np.roll(seq, -1).tolist()))
    y_e = np.array([1 if e in tour_edges else 0 for e in edges], dtype=np.uint8)
    coverage = float(y_e.sum()) / n

    return {
        "route_id": route_id,
        "station_code": route.get("station_code"),
        "date": route.get("date_YYYY_MM_DD"),
        "route_score": route.get("route_score"),
        "stop_ids": ids,
        "zones": ["STATION"] + zones[1:],
        "lat": lat.astype(np.float32),
        "lng": lng.astype(np.float32),
        "node_feat": node,
        "edge_index": np.stack([src, dst]),
        "edge_feat": ef.astype(np.float16),
        "edge_y": y_e,
        "T": T32,
        "actual_seq": seq,
        "tour_edge_coverage": coverage,
        "k": kk,
    }


def _extract_travel_times(path: str, wanted: set) -> dict:
    """Stream the (very large) travel-time JSON and keep only the wanted routes."""
    import ijson

    out = {}
    t0 = time.time()
    with open(path, "rb") as f:
        for i, (rid, mat) in enumerate(ijson.kvitems(f, "", use_float=True)):
            if rid in wanted:
                out[rid] = mat
            if (i + 1) % 1000 == 0:
                print(f"  streamed {i + 1} routes ({time.time() - t0:.0f}s), kept {len(out)}")
            if len(out) == len(wanted):
                break
    missing = wanted - set(out)
    assert not missing, f"missing travel times for {len(missing)} routes"
    return out


def preprocess(k: int, n_test: int, n_heldout: int, n_val: int, seed: int) -> None:
    os.makedirs(PROC_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    rng = np.random.default_rng(seed)

    # ---- training dataset: keep only High route_score routes (the "past successes") ------
    print("[preprocess] loading training route_data / actual_sequences")
    rd = json.load(open(os.path.join(RAW_DIR, "route_data.json")))
    aseq = json.load(open(os.path.join(RAW_DIR, "actual_sequences.json")))
    high = sorted(r for r, v in rd.items() if v["route_score"] == "High")
    print(f"  {len(rd)} training routes, {len(high)} with route_score == High")
    perm = rng.permutation(len(high))
    high = [high[i] for i in perm]
    val_ids = high[:n_val]
    held_ids = high[n_val:n_val + n_heldout]
    train_ids = high[n_val + n_heldout:]

    # ---- evaluation dataset (separate official dataset, no route_score published) --------
    er = json.load(open(os.path.join(RAW_DIR, "eval_route_data.json")))
    ea = json.load(open(os.path.join(RAW_DIR, "eval_actual_sequences.json")))
    eval_all = sorted(er)
    test_ids = [eval_all[i] for i in rng.permutation(len(eval_all))[:n_test]]
    assert not (set(test_ids) & set(rd)), "evaluation routes overlap training routes"

    splits = {"seed": seed, "k": k, "train": train_ids, "val": val_ids, "heldout": held_ids, "test": test_ids}
    with open(os.path.join(RESULTS_DIR, "splits.json"), "w") as f:
        json.dump(splits, f, indent=1)

    print(f"[preprocess] streaming training travel times for {len(high)} routes")
    tt_train = _extract_travel_times(os.path.join(RAW_DIR, "travel_times.json"), set(high))
    print(f"[preprocess] streaming evaluation travel times for {len(test_ids)} routes")
    tt_eval = _extract_travel_times(os.path.join(RAW_DIR, "eval_travel_times.json"), set(test_ids))

    def build(ids, routes, acts, tts, name):
        out = []
        t0 = time.time()
        for i, rid in enumerate(ids):
            out.append(build_example(rid, routes[rid], acts[rid]["actual"], tts[rid], k=k))
            if (i + 1) % 250 == 0:
                print(f"  {name}: {i + 1}/{len(ids)} ({time.time() - t0:.0f}s)")
        cov = np.mean([e["tour_edge_coverage"] for e in out])
        print(f"  {name}: {len(out)} routes, mean fraction of tour edges inside the sparse graph = {cov:.4f}")
        with open(os.path.join(PROC_DIR, f"{name}.pkl"), "wb") as f:
            pickle.dump(out, f, protocol=pickle.HIGHEST_PROTOCOL)
        return cov

    covs = {}
    covs["val"] = build(val_ids, rd, aseq, tt_train, "val")
    covs["heldout"] = build(held_ids, rd, aseq, tt_train, "heldout")
    covs["test"] = build(test_ids, er, ea, tt_eval, "test")
    covs["train"] = build(train_ids, rd, aseq, tt_train, "train")
    with open(os.path.join(RESULTS_DIR, "data_stats.json"), "w") as f:
        json.dump({"n_train_total": len(rd), "n_high": len(high),
                   "n_eval_total": len(er), "split_sizes": {s: len(splits[s]) for s in ["train", "val", "heldout", "test"]},
                   "tour_edge_coverage": covs, "k": k}, f, indent=1)
    print("[preprocess] done")


def preprocess_fresh(n_fresh: int, seed: int) -> None:
    """Draw the final test set: n_fresh evaluation-dataset routes, uniformly at random (own seed),
    from the evaluation routes that are NOT in the (reused) test split. Adds "fresh" to
    results/splits.json and writes data/processed/fresh.pkl with the same k as the other splits."""
    sp_path = os.path.join(RESULTS_DIR, "splits.json")
    splits = json.load(open(sp_path))
    k = splits["k"]
    rd = json.load(open(os.path.join(RAW_DIR, "route_data.json")))
    er = json.load(open(os.path.join(RAW_DIR, "eval_route_data.json")))
    ea = json.load(open(os.path.join(RAW_DIR, "eval_actual_sequences.json")))
    used = set(splits["test"])
    pool = sorted(r for r in er if r not in used)
    rng = np.random.default_rng(seed)
    fresh = [pool[i] for i in rng.permutation(len(pool))[:n_fresh]]
    assert not (set(fresh) & used), "fresh routes overlap the reused test split"
    assert not (set(fresh) & set(rd)), "fresh routes overlap training-dataset routes"
    print(f"[fresh] {len(pool)} evaluation routes not in the reused test split; drew {len(fresh)} (seed {seed})")
    tt = _extract_travel_times(os.path.join(RAW_DIR, "eval_travel_times.json"), set(fresh))
    out = [build_example(rid, er[rid], ea[rid]["actual"], tt[rid], k=k) for rid in fresh]
    cov = float(np.mean([e["tour_edge_coverage"] for e in out]))
    print(f"  fresh: {len(out)} routes, mean fraction of tour edges inside the sparse graph = {cov:.4f}")
    with open(os.path.join(PROC_DIR, "fresh.pkl"), "wb") as f:
        pickle.dump(out, f, protocol=pickle.HIGHEST_PROTOCOL)
    splits["fresh"] = fresh
    splits["fresh_seed"] = seed
    with open(sp_path, "w") as f:
        json.dump(splits, f, indent=1)
    st_path = os.path.join(RESULTS_DIR, "data_stats.json")
    st = json.load(open(st_path))
    st["split_sizes"]["fresh"] = len(fresh)
    st["tour_edge_coverage"]["fresh"] = cov
    st["n_eval_not_in_test"] = len(pool)
    with open(st_path, "w") as f:
        json.dump(st, f, indent=1)


def load_split(name: str) -> list:
    with open(os.path.join(PROC_DIR, f"{name}.pkl"), "rb") as f:
        return pickle.load(f)


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("download")
    d.add_argument("--force", action="store_true")
    p = sub.add_parser("preprocess")
    p.add_argument("--k", type=int, default=16)
    p.add_argument("--n-test", type=int, default=300)
    p.add_argument("--n-heldout", type=int, default=150)
    p.add_argument("--n-val", type=int, default=60)
    p.add_argument("--seed", type=int, default=0)
    q = sub.add_parser("preprocess-fresh")
    q.add_argument("--n-fresh", type=int, default=300)
    q.add_argument("--seed", type=int, default=1)
    a = ap.parse_args(argv)
    if a.cmd == "download":
        download(a.force)
    elif a.cmd == "preprocess":
        preprocess(a.k, a.n_test, a.n_heldout, a.n_val, a.seed)
    else:
        preprocess_fresh(a.n_fresh, a.seed)


if __name__ == "__main__":
    main()
