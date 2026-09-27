"""Compare Amazon drivers' executed sequences with travel-time-optimal LKH tours.

Uses the public 2021 Amazon Last Mile Routing Research Challenge training data
(s3://amazon-last-mile-challenges/almrrc2021/almrrc2021-data-training/model_build_inputs).
For a seeded random sample of routes it computes, for the driver's actual sequence
and for an LKH tour that minimises the given average travel times:
  closed-tour travel time (seconds, excluding service time),
  share of time-windowed stops reached after the window closes, using the
  route's departure time, the travel-time matrix and planned service times,
  number of zone re-entries (a proxy for backtracking),
  and, for the LKH tour, the official challenge score against the driver sequence.

Preparation: download route_data.json, actual_sequences.json, travel_times.json and
package_data.json from the bucket, then create pk.json with
  sed 's/: NaN/: null/g' package_data.json > pk.json
(package_data.json contains bare NaN tokens that streaming JSON parsers reject).
Requires: pip install elkai ijson numpy.  Run: python 03_driver_vs_lkh_analysis.py 1000
The official scorer score.py must sit in the same directory.
The LKH tour minimises travel time only and is not given time windows.
"""
import json
import random
import sys
import time
from collections import defaultdict
from datetime import datetime

import elkai
import ijson
import numpy as np

sys.setrecursionlimit(100000)
import score as sc  # official MIT-CAVE scorer: https://github.com/MIT-CAVE/rc-cli/blob/main/scoring/score.py

N_SAMPLE = int(sys.argv[1]) if len(sys.argv) > 1 else 50
SEED = 7

routes = json.load(open("route_data.json"))
actual = json.load(open("actual_sequences.json"))
ids = sorted(routes)
random.Random(SEED).shuffle(ids)
sample = set(ids[:N_SAMPLE])

# Stream travel times for sampled routes only.
tt = {}
with open("travel_times.json", "rb") as f:
    for rid, mat in ijson.kvitems(f, "", use_float=True):
        if rid in sample:
            tt[rid] = mat
        if len(tt) == len(sample):
            break

# Stream package data for sampled routes only.
pk = {}
with open("pk.json", "rb") as f:
    for rid, stops in ijson.kvitems(f, "", use_float=True):
        if rid in sample:
            pk[rid] = stops
        if len(pk) == len(sample):
            break


def parse_ts(s):
    if s is None or (isinstance(s, float) and np.isnan(s)) or s == "NaN":
        return None
    try:
        return datetime.strptime(s, "%Y-%m-%d %H:%M:%S")
    except (TypeError, ValueError):
        return None


def evaluate_seq(seq, rid):
    """seq: list of stop ids starting at the station. Returns metrics."""
    r = routes[rid]
    mat = tt[rid]
    dep = datetime.strptime(r["date_YYYY_MM_DD"] + " " + r["departure_time_utc"], "%Y-%m-%d %H:%M:%S")
    t = 0.0
    travel = 0.0
    tw_stops = 0
    late = 0
    for a, b in zip(seq, seq[1:]):
        leg = mat[a][b]
        travel += leg
        t += leg
        pkgs = pk[rid].get(b, {})
        ends = [parse_ts(p["time_window"]["end_time_utc"]) for p in pkgs.values()]
        ends = [e for e in ends if e is not None]
        if ends:
            tw_stops += 1
            arrival = dep.timestamp() + t
            if arrival > min(ends).timestamp():
                late += 1
        t += sum(float(p.get("planned_service_time_seconds") or 0.0) for p in pkgs.values())
    travel += mat[seq[-1]][seq[0]]
    zones = [r["stops"][s].get("zone_id") for s in seq[1:]]
    zones = [z for z in zones if isinstance(z, str)]
    seen, reentries, prev = set(), 0, None
    for z in zones:
        if z != prev:
            if z in seen:
                reentries += 1
            seen.add(z)
        prev = z
    return travel, tw_stops, late, reentries


rows = []
t0 = time.time()
for rid in sorted(sample):
    r = routes[rid]
    stops = list(r["stops"])
    station = [s for s in stops if r["stops"][s]["type"] == "Station"][0]
    act = sorted(actual[rid]["actual"], key=lambda s: actual[rid]["actual"][s])
    assert act[0] == station
    order = [station] + [s for s in stops if s != station]
    M = [[int(round(tt[rid][a][b] * 10)) for b in order] for a in order]
    tour = elkai.DistanceMatrix(M).solve_tsp(runs=5)
    if tour[-1] == tour[0]:
        tour = tour[:-1]
    i0 = tour.index(0)
    tour = tour[i0:] + tour[:i0]
    opt = [order[i] for i in tour]
    d = evaluate_seq(act, rid)
    o = evaluate_seq(opt, rid)
    s_opt = sc.score(act + [act[0]], opt + [opt[0]], tt[rid])
    rows.append(dict(rid=rid, score=r["route_score"], station=r["station_code"], n=len(stops),
                     drv_tt=d[0], opt_tt=o[0], tw=d[1], drv_late=d[2], opt_late=o[2],
                     drv_reent=d[3], opt_reent=o[3], opt_challenge_score=s_opt))
print(f"processed {len(rows)} routes in {time.time() - t0:.0f}s", file=sys.stderr)
json.dump(rows, open(f"results_{N_SAMPLE}.json", "w"))

# Summaries
def summ(rs, label):
    ratio = np.array([x["drv_tt"] / x["opt_tt"] for x in rs])
    tw = sum(x["tw"] for x in rs)
    dl = sum(x["drv_late"] for x in rs)
    ol = sum(x["opt_late"] for x in rs)
    routes_tw = [x for x in rs if x["tw"] > 0]
    print(f"{label}: routes={len(rs)} mean_stops={np.mean([x['n'] for x in rs]):.1f} "
          f"drv/opt travel ratio mean={ratio.mean():.3f} median={np.median(ratio):.3f} "
          f"TW stops={tw} late drv={dl} ({dl / max(tw, 1):.3%}) late opt={ol} ({ol / max(tw, 1):.3%}) "
          f"routes with TW={len(routes_tw)} "
          f"zone re-entries drv mean={np.mean([x['drv_reent'] for x in rs]):.2f} opt mean={np.mean([x['opt_reent'] for x in rs]):.2f} "
          f"opt challenge score mean={np.mean([x['opt_challenge_score'] for x in rs]):.4f}")

summ(rows, "ALL")
by = defaultdict(list)
for x in rows:
    by[x["score"]].append(x)
for k in ("High", "Medium", "Low"):
    if by[k]:
        summ(by[k], k)
