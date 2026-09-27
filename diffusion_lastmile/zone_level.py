"""Hierarchical (zone-level) variant of the edge-diffusion model.

Motivation (computed on the validation split by diagnostics.py, which writes the numbers to
results/diagnostics.json): the stop-level heat-map's most likely successor is only marginally
more often the driver's actual next stop than the nearest stop is, while most of a driver's
consecutive stops share a zone and nearly all zones are served in one contiguous block. The
global part of a driver's plan is therefore the ORDER OF ZONES. This module
applies exactly the same Bernoulli edge diffusion + gated GNN to a much smaller graph whose nodes
are the route's zones (plus the station), so the model generates the zone tour, and the stop
sequence is then completed by the same OR-Tools solver used by the baselines with a penalty on
arcs that break the generated zone order.

Zone graph of one route (complete directed graph over station + zones):
  node features  centroid x, y (same normalisation as the stops), is_station, zone_rank,
                 super_rank, sub_letter, share of the route's stops in the zone, n_zones / 50
  edge features  mean and minimum real travel time between the two zones' stops (/ median stop
                 travel time), centroid distance, zone-id relations (same super zone, same sub
                 zone, |d n1|, |d n2|, |d letter|, signed zone-rank difference), station flag,
                 and three HISTORY features computed only from the High-quality training routes
                 of the same station: P(a -> b), P(b -> a) (transition counts divided by how
                 often the source zone occurred) and log(1 + count(a -> b)). For training routes
                 the counts are leave-one-out (the route's own transitions are removed), so the
                 feature never contains the answer.
  target         directed adjacency of the driver's zone tour station -> z1 -> ... -> zm -> station,
                 where zones are ordered by the median position of their stops in the driver's
                 sequence.

The same zone graphs are used by the one-shot supervised ablation (train.py --objective
supervised --level zone), so the two zone models see identical inputs.

    python zone_level.py build [split ...]   # writes data/processed/zone_{train,val,heldout,test,fresh}.pkl
                                             # (default: all splits whose stop-level file exists)
"""
from __future__ import annotations

import os
import pickle
from collections import defaultdict

import numpy as np

import data

ZONE_NODE_FEATS = ["x", "y", "is_station", "zone_rank", "super_rank", "sub_letter", "stop_share", "n_zones"]
ZONE_EDGE_FEATS = ["tt_mean", "tt_min", "euclid", "same_super", "same_sub", "d_n1", "d_n2", "d_letter",
                   "zone_rank_diff", "station_edge", "hist_p_ab", "hist_p_ba", "hist_log_cnt"]
STATION = "STATION"


def driver_zone_order(ex) -> list:
    """Zones ordered by the median position of their stops in the driver's sequence."""
    pos = defaultdict(list)
    for p, i in enumerate(ex["actual_seq"][1:]):
        pos[ex["zones"][i]].append(p)
    return sorted(pos, key=lambda z: (np.median(pos[z]), z))


def zone_transitions(order: list) -> list:
    seq = [STATION] + list(order) + [STATION]
    return list(zip(seq[:-1], seq[1:]))


class History:
    """Zone-transition statistics of the High-quality TRAINING routes, per station."""

    def __init__(self, train_examples: list):
        self.cnt = defaultdict(int)
        self.occ = defaultdict(int)
        for ex in train_examples:
            self.add(ex, +1)

    def add(self, ex, sign):
        st = ex["station_code"]
        order = driver_zone_order(ex)
        for a, b in zone_transitions(order):
            self.cnt[(st, a, b)] += sign
        for z in [STATION] + order:
            self.occ[(st, z)] += sign

    def feats(self, st, a, b):
        c_ab = self.cnt.get((st, a, b), 0)
        c_ba = self.cnt.get((st, b, a), 0)
        o_a = self.occ.get((st, a), 0)
        o_b = self.occ.get((st, b), 0)
        return (c_ab / o_a if o_a > 0 else 0.0, c_ba / o_b if o_b > 0 else 0.0, np.log1p(max(c_ab, 0)) / 5.0)


def build_zone_example(ex: dict, hist: History, leave_one_out: bool = False) -> dict:
    if leave_one_out:
        hist.add(ex, -1)
    try:
        zones = ex["zones"]
        n = len(zones)
        T = ex["T"].astype(np.float64)
        uz = sorted(set(zones[1:]), key=data.zone_sort_key)
        znodes = [STATION] + uz
        m = len(znodes)
        zidx = {z: k for k, z in enumerate(znodes)}
        members = [[0]] + [[i for i in range(1, n) if zones[i] == z] for z in uz]
        nf_stop = ex["node_feat"]
        parsed = [None] + [data.parse_zone(z) for z in uz]

        node = np.zeros((m, len(ZONE_NODE_FEATS)), dtype=np.float32)
        for k, mem in enumerate(members):
            node[k, 0] = nf_stop[mem, 0].mean()
            node[k, 1] = nf_stop[mem, 1].mean()
            node[k, 2] = float(k == 0)
            if k > 0:
                node[k, 3:6] = nf_stop[mem[0], 3:6]
                node[k, 6] = len(mem) / (n - 1)
        node[:, 7] = m / 50.0

        off = T[1:, 1:][~np.eye(n - 1, dtype=bool)]
        tscale = max(float(np.median(off)), 1.0)
        src, dst = np.nonzero(~np.eye(m, dtype=bool))
        E = len(src)
        ef = np.zeros((E, len(ZONE_EDGE_FEATS)), dtype=np.float32)
        st = ex["station_code"]
        for e, (a, b) in enumerate(zip(src, dst)):
            sub = T[np.ix_(members[a], members[b])]
            ef[e, 0] = min(sub.mean() / tscale, 10.0)
            ef[e, 1] = min(sub.min() / tscale, 10.0)
            ef[e, 2] = min(np.hypot(node[a, 0] - node[b, 0], node[a, 1] - node[b, 1]), 3.0)
            if a == 0 or b == 0:
                ef[e, 9] = 1.0
            else:
                pa, pb = parsed[a], parsed[b]
                same_letter = pa[0] == pb[0]
                same_super = same_letter and pa[1] == pb[1]
                same_sub = same_super and pa[2] == pb[2]
                ef[e, 3] = float(same_super)
                ef[e, 4] = float(same_sub)
                ef[e, 5] = min(abs(pa[1] - pb[1]) / 5.0, 1.0) if same_letter else 1.0
                ef[e, 6] = min(abs(pa[2] - pb[2]) / 5.0, 1.0) if same_super else 1.0
                ef[e, 7] = min(abs(ord(pa[3]) - ord(pb[3])) / 5.0, 1.0) if same_sub else 1.0
                ef[e, 8] = node[b, 3] - node[a, 3]
            ef[e, 10:13] = hist.feats(st, znodes[a], znodes[b])

        order = driver_zone_order(ex)
        tz = set((zidx[a], zidx[b]) for a, b in zone_transitions(order))
        y = np.array([1 if (a, b) in tz else 0 for a, b in zip(src, dst)], dtype=np.uint8)
        zone_T = np.zeros((m, m))
        for a in range(m):
            for b in range(m):
                if a != b:
                    zone_T[a, b] = T[np.ix_(members[a], members[b])].mean()
        return {
            "route_id": ex["route_id"], "station_code": st, "zone_names": znodes,
            "node_feat": node, "edge_index": np.stack([src, dst]).astype(np.int64), "edge_feat": ef.astype(np.float16),
            "edge_y": y, "zone_T": zone_T.astype(np.float32),
            "driver_zone_seq": np.array([0] + [zidx[z] for z in order], dtype=np.int64),
        }
    finally:
        if leave_one_out:
            hist.add(ex, +1)


def zone_order_penalty_matrix(T: np.ndarray, zones: list, zone_seq_names: list, lam: float) -> np.ndarray:
    """Real travel times plus lam * median(T) on every stop arc that does not follow the given zone
    order (allowed: stay in the zone, or move to the next zone; the station may only be left to
    the first zone and entered from the last zone)."""
    from baselines import route_time_scale

    rank = {z: k for k, z in enumerate(zone_seq_names)}
    last = len(zone_seq_names) - 1
    r = np.array([-1] + [rank[z] for z in zones[1:]])
    ri = r[:, None]
    rj = r[None, :]
    ok = (ri == rj) | (rj == ri + 1)
    ok[0, :] = r == 0
    ok[:, 0] = r == last
    ok[0, 0] = True
    return np.asarray(T, dtype=np.float64) + lam * route_time_scale(T) * (~ok)


def history_heatmap(zex: dict) -> np.ndarray:
    """Non-learned control: the historical transition probability P(a -> b) itself. Pairs never
    seen in history score 0 and are ordered by the decoder's travel-time tie-break."""
    return zex["edge_feat"][:, 10].astype(np.float64)


def build_all(names=("train", "val", "heldout", "test", "fresh")):
    tr = data.load_split("train")
    hist = History(tr)
    for name in names:
        if name != "train" and not os.path.exists(os.path.join(data.PROC_DIR, f"{name}.pkl")):
            print(f"[zone] {name}.pkl not found, skipped")
            continue
        exs = tr if name == "train" else data.load_split(name)
        out = [build_zone_example(ex, hist, leave_one_out=(name == "train")) for ex in exs]
        seen = np.mean([np.mean(z["edge_feat"][z["edge_y"] == 1, 10].astype(float) > 0) for z in out])
        print(f"[zone] {name}: {len(out)} routes, mean zones {np.mean([len(z['zone_names']) - 1 for z in out]):.1f}, "
              f"driver zone transitions with non-zero history {seen * 100:.1f}%")
        with open(os.path.join(data.PROC_DIR, f"zone_{name}.pkl"), "wb") as f:
            pickle.dump(out, f, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "build":
        build_all(tuple(sys.argv[2:]) or ("train", "val", "heldout", "test", "fresh"))
