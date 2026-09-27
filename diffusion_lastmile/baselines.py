"""Non-diffusion methods compared against the diffusion model.

M_nn       nearest neighbour on the real travel times, starting from the station.
M_ortools  Google OR-Tools routing solver (single vehicle TSP, depot = station) on the real,
           asymmetric travel-time matrix: PATH_CHEAPEST_ARC first solution + GUIDED_LOCAL_SEARCH
           metaheuristic with a fixed wall-clock limit per route.
M_zone     zone-aware heuristic: the same OR-Tools solver on a modified matrix that adds a
           penalty lam * median(T) to every arc between two different zones, which makes the
           solver finish a zone before moving to the next one (the kind of zone-based rule used
           by many 2021 challenge teams). Tour cost is still reported on the real matrix.
M_softdist the non-learned SoftDist heat-map of Xia, Yang, Liu, Liu, Song & Bian, "Position:
           Rethinking Post-Hoc Search-Based Neural Approaches for Solving Large-Scale Traveling
           Salesman Problems", ICML 2024:  H_ij = exp(-d_ij / tau) / sum_{k != i} exp(-d_ik / tau),
           here with d = real travel time divided by the route's median travel time, read off on
           the same sparse graph as the diffusion model and decoded by the same greedy + 2-opt.
M_zonehist non-learned control for the zone-level models (zone_level.py): the zone order is
           decoded from the historical zone-transition frequencies of the High-quality training
           routes (with the same zone decoders and the same tuning grid as the learned zone
           models), then stops are sequenced by the same OR-Tools solver.
"""
from __future__ import annotations

import time

import numpy as np


def nearest_neighbour(T: np.ndarray) -> np.ndarray:
    n = T.shape[0]
    visited = np.zeros(n, dtype=bool)
    visited[0] = True
    seq = [0]
    for _ in range(n - 1):
        row = np.where(visited, np.inf, T[seq[-1]])
        j = int(np.argmin(row))
        seq.append(j)
        visited[j] = True
    return np.array(seq, dtype=np.int64)


def route_time_scale(T: np.ndarray) -> float:
    n = T.shape[0]
    off = T[1:, 1:][~np.eye(n - 1, dtype=bool)]
    return max(float(np.median(off)), 1.0) if off.size else 1.0


def softdist_heatmap(T: np.ndarray, edge_index: np.ndarray, tau: float) -> np.ndarray:
    d = np.asarray(T, dtype=np.float64) / route_time_scale(T)
    logits = -d / tau
    np.fill_diagonal(logits, -np.inf)
    logits = logits - logits.max(axis=1, keepdims=True)
    H = np.exp(logits)
    H /= H.sum(axis=1, keepdims=True)
    return H[edge_index[0], edge_index[1]]


def ortools_tsp(cost: np.ndarray, time_limit_s: float = 5.0, scale: float = 10.0) -> np.ndarray:
    """Solve a single-vehicle TSP from node 0 with OR-Tools; returns the stop order."""
    from ortools.constraint_solver import pywrapcp, routing_enums_pb2

    n = cost.shape[0]
    C = np.rint(np.asarray(cost, dtype=np.float64) * scale).astype(np.int64)
    manager = pywrapcp.RoutingIndexManager(n, 1, 0)
    routing = pywrapcp.RoutingModel(manager)
    transit = routing.RegisterTransitMatrix(C.tolist())
    routing.SetArcCostEvaluatorOfAllVehicles(transit)
    params = pywrapcp.DefaultRoutingSearchParameters()
    params.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    params.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    ms = int(round(time_limit_s * 1000))
    params.time_limit.seconds = ms // 1000
    params.time_limit.nanos = (ms % 1000) * 1_000_000
    sol = routing.SolveWithParameters(params)
    if sol is None:
        raise RuntimeError("OR-Tools found no solution")
    seq = []
    idx = routing.Start(0)
    while not routing.IsEnd(idx):
        seq.append(manager.IndexToNode(idx))
        idx = sol.Value(routing.NextVar(idx))
    return np.array(seq, dtype=np.int64)


def zone_penalty_matrix(T: np.ndarray, zones: list, lam: float) -> np.ndarray:
    z = np.array(zones, dtype=object)
    diff = (z[:, None] != z[None, :]).astype(np.float64)
    diff[0, :] = 0.0  # no penalty on arcs leaving / entering the station
    diff[:, 0] = 0.0
    return np.asarray(T, dtype=np.float64) + lam * route_time_scale(T) * diff


def solve_ortools_job(job: tuple) -> dict:
    """Worker entry point for multiprocessing: job = (route_id, method, cost_matrix, time_limit_s).
    The cost matrix is the real travel-time matrix for M_ortools and a penalised copy of it for the
    zone-based methods; tours are always evaluated on the real matrix afterwards."""
    rid, method, cost, limit = job
    t0 = time.perf_counter()
    c0 = time.process_time()
    seq = ortools_tsp(cost, time_limit_s=limit)
    # cpu_s / runtime_s close to 1 shows that the solver had a whole core for its time limit
    return {"route_id": rid, "method": method, "seq": seq.tolist(), "runtime_s": time.perf_counter() - t0,
            "cpu_s": time.process_time() - c0}
