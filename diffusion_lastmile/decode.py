"""Turn an edge heat-map into a valid stop sequence, then optionally improve it with 2-opt.

greedy_decode: DIFUSCO-style greedy edge insertion adapted to directed edges. Candidate edges
(i -> j) of the sparse graph are visited in decreasing heat-map score (ties broken by shorter
real travel time) and accepted when i has no successor yet, j has no predecessor yet and the
edge does not close a sub-tour. Whatever fragments remain (because some tour edges are outside
the sparse graph, or scores are low) are joined by the same greedy rule on the complete graph
ordered by real travel time. The resulting Hamiltonian cycle is rotated to start at the
station (node 0).

two_opt: best-improvement 2-opt on the closed tour with the ASYMMETRIC real travel-time matrix.
Reversing the segment s[i..j] changes the cost by
    T[a, s_j] + R(i, j) + T[s_i, b] - T[a, s_i] - F(i, j) - T[s_j, b]
with a = s[i-1], b = s[j+1] and F / R the forward / reversed costs of the segment, obtained in
O(1) from prefix sums, so one full neighbourhood scan is a vectorised O(n^2) numpy operation.
The station stays at position 0.
"""
from __future__ import annotations

import numpy as np


class _DSU:
    def __init__(self, n):
        self.p = list(range(n))

    def find(self, a):
        p = self.p
        while p[a] != a:
            p[a] = p[p[a]]
            a = p[a]
        return a

    def union(self, a, b):
        self.p[self.find(a)] = self.find(b)


def greedy_decode(n: int, edge_index: np.ndarray, scores: np.ndarray, T: np.ndarray) -> np.ndarray:
    src, dst = edge_index[0], edge_index[1]
    order = np.lexsort((T[src, dst], -scores))  # primary: score desc, secondary: travel time asc
    nxt = np.full(n, -1, dtype=np.int64)
    has_pred = np.zeros(n, dtype=bool)
    dsu = _DSU(n)
    added = 0

    def try_add(i, j):
        nonlocal added
        if i == j or nxt[i] != -1 or has_pred[j]:
            return False
        if dsu.find(i) == dsu.find(j):
            return False
        nxt[i] = j
        has_pred[j] = True
        dsu.union(i, j)
        added += 1
        return True

    for e in order:
        if added == n - 1:
            break
        try_add(int(src[e]), int(dst[e]))

    if added < n - 1:
        # join the fragments: tails (no successor) -> heads (no predecessor), shortest first
        tails = np.where(nxt == -1)[0]
        heads = np.where(~has_pred)[0]
        cost = T[np.ix_(tails, heads)]
        flat = np.argsort(cost, axis=None, kind="stable")
        ti, hi = np.unravel_index(flat, cost.shape)
        for a, b in zip(ti, hi):
            if added == n - 1:
                break
            try_add(int(tails[a]), int(heads[b]))
    assert added == n - 1
    start = int(np.where(~has_pred)[0][0])
    seq = [start]
    while nxt[seq[-1]] != -1:
        seq.append(int(nxt[seq[-1]]))
    seq = np.array(seq, dtype=np.int64)
    k = int(np.where(seq == 0)[0][0])
    return np.roll(seq, -k)


def two_opt(seq: np.ndarray, T: np.ndarray, max_iter: int = 10000, tol: float = 1e-7) -> np.ndarray:
    s = np.array(seq, dtype=np.int64)
    n = len(s)
    if n < 4:
        return s
    T = np.asarray(T, dtype=np.float64)
    ii = np.arange(1, n)[:, None]
    jj = np.arange(1, n)[None, :]
    valid = jj > ii
    for _ in range(max_iter):
        f = T[s[:-1], s[1:]]
        r = T[s[1:], s[:-1]]
        Pf = np.concatenate([[0.0], np.cumsum(f)])
        Pr = np.concatenate([[0.0], np.cumsum(r)])
        a = s[ii - 1]
        b = s[(jj + 1) % n]
        si = s[ii]
        sj = s[jj]
        delta = T[a, sj] + (Pr[jj] - Pr[ii]) + T[si, b] - T[a, si] - (Pf[jj] - Pf[ii]) - T[sj, b]
        delta = np.where(valid, delta, np.inf)
        k = int(np.argmin(delta))
        if delta.flat[k] >= -tol:
            break
        i, j = divmod(k, n - 1)
        i += 1
        j += 1
        s[i:j + 1] = s[i:j + 1][::-1].copy()
    return s


def decode(n, edge_index, scores, T, use_two_opt=True):
    seq = greedy_decode(n, edge_index, scores, T)
    if use_two_opt:
        seq = two_opt(seq, T)
    return seq
