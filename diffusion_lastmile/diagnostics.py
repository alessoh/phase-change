"""Diagnostics that motivate the zone-level model, computed from the data and the checkpoints.

    python diagnostics.py            -> results/diagnostics.json (+ printed summary)

1. Top-1 successor accuracy of stop-level heat-maps (validation routes). For every stop i of a
   route the predicted next stop is the out-neighbour j with the highest heat-map value on the
   sparse k-NN graph (the depot counts as a possible successor, so the last stop is included).
   It is compared with the driver's actual next stop. Also reported: top-3 recall (driver's
   next stop among the three highest-scoring out-edges). Heat-maps: stop-level diffusion (the
   tuned sampling setting), the one-shot supervised ablation, the SoftDist control, and the
   plain "nearest stop" rule (argmin of the real travel time over all other nodes). Seeds 0, 1
   and 2 are reported separately.
2. Zone structure of driver sequences (validation and test routes):
   - share of consecutive stop-to-stop moves (station legs excluded) whose two stops share a zone,
   - share of zones whose stops form one contiguous block in the driver's sequence,
   - share of routes in which every zone is contiguous.
   Zones are the dataset's zone_id values; the few stops without a zone_id inherit the zone of
   the nearest stop by travel time (data.py), which is included in these statistics.
3. Top-1 next-zone accuracy of zone-level heat-maps (validation routes): the zone diffusion
   model, the one-shot supervised zone model, the historical transition frequencies and the
   nearest zone by mean travel time, against the driver's zone order.
"""
from __future__ import annotations

import json
import os

import numpy as np
import torch

import data
from baselines import softdist_heatmap
from evaluate import HIER_SAMPLES, HIER_STEPS, RES, heatmap, load_learned
from zone_level import history_heatmap



def driver_successor(seq: np.ndarray, n: int) -> np.ndarray:
    succ = np.empty(n, dtype=np.int64)
    succ[seq] = np.roll(seq, -1)
    return succ


def successor_accuracy(ex, h, nodes) -> tuple[int, int, int]:
    """(top-1 hits, top-3 hits, count) of heat-map h over the given source nodes."""
    ei = ex["edge_index"]
    succ = driver_successor(ex["actual_seq"], len(ex["stop_ids"]))
    T = ex["T"]
    t1 = t3 = 0
    for i in nodes:
        idx = np.where(ei[0] == i)[0]
        # ties (e.g. saturated probabilities) are broken by shorter travel time, as in the decoder
        order = idx[np.lexsort((T[i, ei[1, idx]], -h[idx]))]
        cand = ei[1, order]
        t1 += int(cand[0] == succ[i])
        t3 += int(succ[i] in cand[:3])
    return t1, t3, len(nodes)


def nearest_accuracy(ex, nodes) -> tuple[int, int, int]:
    T = ex["T"].astype(np.float64).copy()
    np.fill_diagonal(T, np.inf)
    succ = driver_successor(ex["actual_seq"], len(ex["stop_ids"]))
    t1 = t3 = 0
    for i in nodes:
        o = np.argsort(T[i], kind="stable")
        t1 += int(o[0] == succ[i])
        t3 += int(succ[i] in o[:3])
    return t1, t3, len(nodes)


def zone_structure(exs) -> dict:
    same, moves, contig, zones_total, routes_all = 0, 0, 0, 0, 0
    for ex in exs:
        z = ex["zones"]
        a = ex["actual_seq"][1:]
        for u, v in zip(a[:-1], a[1:]):
            same += int(z[u] == z[v])
            moves += 1
        pos = {}
        for p, i in enumerate(a):
            pos.setdefault(z[i], []).append(p)
        c = [max(p) - min(p) + 1 == len(p) for p in pos.values()]
        contig += sum(c)
        zones_total += len(c)
        routes_all += int(all(c))
    return {"n_routes": len(exs), "same_zone_move_share": same / moves, "n_moves": moves,
            "contiguous_zone_share": contig / zones_total, "n_zones": zones_total,
            "routes_all_zones_contiguous_share": routes_all / len(exs)}


def zone_successor_accuracy(zex, h) -> tuple[int, int]:
    ei = zex["edge_index"]
    seq = zex["driver_zone_seq"]
    succ = driver_successor(seq, len(zex["zone_names"]))
    zT = zex["zone_T"]
    hits = 0
    for i in range(len(zex["zone_names"])):
        idx = np.where(ei[0] == i)[0]
        order = idx[np.lexsort((zT[i, ei[1, idx]], -h[idx]))]
        hits += int(ei[1, order[0]] == succ[i])
    return hits, len(zex["zone_names"])


def penalty_saturation(val, zval, tuning) -> dict:
    """For each zone-order method, the mean number of arcs per route that leave the method's zone
    order in the cached validation solutions, for every order-penalty weight in the tuning grid
    (decoder fixed to the chosen one). Equal counts at the largest weights mean the penalty has
    saturated, so differences between those settings are OR-Tools search noise."""
    from evaluate import CACHE, ZONE_ORDER_LAMS, decode_zone_order
    from zone_level import zone_order_penalty_matrix

    out = {}
    L = load_learned(0)
    for method in ["zonehist", "hier_sup", "hier"]:
        dec = tuning.get(f"{method}_decode")
        if dec is None:
            continue
        if method == "zonehist":
            hm = {ex["route_id"]: history_heatmap(zval[ex["route_id"]]) for ex in val}
            suffix = ""
        else:
            kind = "zone_diffusion" if method == "hier" else "zone_supervised"
            if kind not in L:
                continue
            m, p = L[kind][:2]
            hm = {ex["route_id"]: heatmap(m, p, zval[ex["route_id"]], HIER_STEPS, HIER_SAMPLES, i) for i, ex in enumerate(val)}
            suffix = f"_s{HIER_STEPS}x{HIER_SAMPLES}_seed0"
        orders = {rid: decode_zone_order(zval[rid], h, dec) for rid, h in hm.items()}
        res = {}
        for lam in ZONE_ORDER_LAMS:
            p = os.path.join(CACHE, f"{method}_val_lim{tuning['ortools_limit']:g}_lam{lam:g}_{dec}{suffix}.json")
            if not os.path.exists(p):
                continue
            cache = json.load(open(p))
            v = []
            for ex in val:
                T = ex["T"].astype(np.float64)
                viol = (zone_order_penalty_matrix(T, ex["zones"], orders[ex["route_id"]], 1.0) - T) > 0
                sq = np.asarray(cache[ex["route_id"]]["seq"])
                v.append(int(viol[sq, np.roll(sq, -1)].sum()))
            res[f"{lam:g}"] = {"mean_violating_arcs_per_route": float(np.mean(v)),
                               "routes_without_violation": float(np.mean(np.array(v) == 0)),
                               "val_amazon_score": tuning[method][f"lam{lam:g}_{dec}"]["amazon_score"]}
        out[method] = {"decode": dec, "by_lam": res}
        print(f"[diag] penalty saturation {method} ({dec}): " + ", ".join(
            f"lam {k}: {d['mean_violating_arcs_per_route']:.2f} arcs, score {d['val_amazon_score']:.4f}" for k, d in res.items()))
    return out


def main():
    torch.set_num_threads(4)
    tuning = json.load(open(os.path.join(RES, "tuning.json")))
    steps, samples, tau = tuning.get("diff_steps", 10), tuning.get("diff_samples", 4), tuning.get("softdist_tau", 0.5)
    val = data.load_split("val")
    zval = {z["route_id"]: z for z in data.load_split("zone_val")}
    out = {"split": "val", "n_routes": len(val), "stop_successor": {}, "zone_successor": {},
           "settings": {"diff_steps": steps, "diff_samples": samples, "softdist_tau": tau,
                        "hier_steps": HIER_STEPS, "hier_samples": HIER_SAMPLES}}

    def acc(name, fn):
        t1 = t3 = cnt = 0
        for k, ex in enumerate(val):
            a, b, c = fn(k, ex)
            t1, t3, cnt = t1 + a, t3 + b, cnt + c
        out["stop_successor"][name] = {"top1": t1 / cnt, "top3": t3 / cnt, "n_stops": cnt}
        print(f"[diag] stop successor {name}: top-1 {t1 / cnt * 100:.1f}%, top-3 {t3 / cnt * 100:.1f}%", flush=True)

    def stops(ex):
        return range(1, len(ex["stop_ids"]))

    acc("nearest_stop", lambda k, ex: nearest_accuracy(ex, stops(ex)))
    acc("softdist", lambda k, ex: successor_accuracy(ex, softdist_heatmap(ex["T"].astype(np.float64), ex["edge_index"], tau),
                                                     stops(ex)))
    zacc = {}

    def zone_acc(name, fn):
        hits = cnt = 0
        for k, ex in enumerate(val):
            a, c = zone_successor_accuracy(zval[ex["route_id"]], fn(k, zval[ex["route_id"]]))
            hits, cnt = hits + a, cnt + c
        zacc[name] = {"top1": hits / cnt, "n_zones_incl_station": cnt}
        print(f"[diag] zone successor {name}: top-1 {hits / cnt * 100:.1f}%", flush=True)

    zone_acc("nearest_zone", lambda k, z: -z["zone_T"][z["edge_index"][0], z["edge_index"][1]].astype(np.float64))
    zone_acc("history", lambda k, z: history_heatmap(z))
    for seed in [0, 1, 2]:
        L = load_learned(seed)
        if "stop_diffusion" in L:
            m, p = L["stop_diffusion"][:2]
            acc(f"diffusion_seed{seed}", lambda k, ex: successor_accuracy(ex, heatmap(m, p, ex, steps, samples, k), stops(ex)))
        if "stop_supervised" in L:
            m, p = L["stop_supervised"][:2]
            acc(f"supervised_seed{seed}", lambda k, ex: successor_accuracy(ex, heatmap(m, p, ex, 1, 1, k), stops(ex)))
        if "zone_diffusion" in L:
            m, p = L["zone_diffusion"][:2]
            zone_acc(f"zone_diffusion_seed{seed}", lambda k, z: heatmap(m, p, z, HIER_STEPS, HIER_SAMPLES, k))
        if "zone_supervised" in L:
            m, p = L["zone_supervised"][:2]
            zone_acc(f"zone_supervised_seed{seed}", lambda k, z: heatmap(m, p, z, 1, 1, k))
    out["zone_successor"] = zacc
    out["penalty_saturation"] = penalty_saturation(val, zval, tuning)
    out["zone_structure"] = {"val": zone_structure(val), "test": zone_structure(data.load_split("test"))}
    for s, d in out["zone_structure"].items():
        print(f"[diag] {s}: same-zone moves {d['same_zone_move_share'] * 100:.1f}%, contiguous zones "
              f"{d['contiguous_zone_share'] * 100:.1f}%, routes with all zones contiguous "
              f"{d['routes_all_zones_contiguous_share'] * 100:.1f}%")

    ss = out["stop_successor"]

    def pct_range(prefix, table):
        v = [d["top1"] * 100 for k, d in table.items() if k.startswith(prefix)]
        return f"{min(v):.1f}% to {max(v):.1f}%" if len(v) > 1 else (f"{v[0]:.1f}%" if v else "n/a")

    zv, zt = out["zone_structure"]["val"], out["zone_structure"]["test"]
    out["summary_lines"] = [
        f"Top-1 successor accuracy on the {len(val)} validation routes ({ss['nearest_stop']['n_stops']:,} stops): stop-level "
        f"diffusion {pct_range('diffusion_seed', ss)} across seeds, one-shot supervised ablation "
        f"{pct_range('supervised_seed', ss)}, SoftDist {ss['softdist']['top1'] * 100:.1f}%, nearest stop "
        f"{ss['nearest_stop']['top1'] * 100:.1f}%.",
        "",
        f"Top-1 next-zone accuracy on the same routes ({zacc['history']['n_zones_incl_station']:,} zone nodes including the "
        f"station): zone diffusion {pct_range('zone_diffusion_seed', zacc)}, one-shot supervised zone model "
        f"{pct_range('zone_supervised_seed', zacc)}, historical frequencies {zacc['history']['top1'] * 100:.1f}%, nearest zone "
        f"{zacc['nearest_zone']['top1'] * 100:.1f}%.",
        "",
        f"Zone structure of driver sequences: {zv['same_zone_move_share'] * 100:.1f}% of consecutive stop-to-stop moves stay in "
        f"the same zone on validation routes ({zt['same_zone_move_share'] * 100:.1f}% on test routes), and "
        f"{zv['contiguous_zone_share'] * 100:.1f}% of zones are served in one contiguous block "
        f"({zt['contiguous_zone_share'] * 100:.1f}% on test).",
    ]
    for method, d in out["penalty_saturation"].items():
        br = d["by_lam"]
        out["summary_lines"] += ["", f"Order-penalty saturation for {method} ({d['decode']} decoder, validation): mean arcs per "
                                 "route that leave the zone order, by penalty weight: " + ", ".join(
                                     f"{k}: {v['mean_violating_arcs_per_route']:.2f} (score {v['val_amazon_score']:.4f})"
                                     for k, v in br.items()) + "."]
    with open(os.path.join(RES, "diagnostics.json"), "w") as f:
        json.dump(out, f, indent=1)
    print("\n".join(out["summary_lines"]))


if __name__ == "__main__":
    main()
