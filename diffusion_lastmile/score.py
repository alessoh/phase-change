"""Route metrics: real travel time and the Amazon Last Mile Routing Research Challenge score.

The challenge score is re-implemented from the official scoring script published by the
challenge organisers (MIT Center for Transportation & Logistics) in the rc-cli repository:
    https://github.com/MIT-CAVE/rc-cli/blob/main/scoring/score.py
and described in Merchan, Arora, Pachon, et al. (2024), "2021 Amazon Last Mile Routing Research
Challenge: Data Set", Transportation Science 58(1), 8-11.

    score(actual, sub) = SD(actual, sub) * ERP_norm(actual, sub) / ERP_edits(actual, sub)

where both sequences are closed lists [station, s1, ..., s_{n-1}, station], SD is the sequence
deviation of the submitted stop order measured in actual-sequence positions, ERP is the edit
distance with real penalty computed with the z-score-normalised (and min-shifted) travel-time
matrix and gap penalty g = 1000, and ERP_edits is the number of edit operations used by that
ERP alignment. The official code is a memoised recursion over list suffixes; here the same
recursion is evaluated bottom-up with identical arithmetic order and identical tie-breaking,
so the results are bit-for-bit equal (checked against the official file in tests/test_basic.py).
Lower is better; the driver's own sequence scores exactly 0.
"""
from __future__ import annotations

import numpy as np


def route_travel_time(seq, T) -> float:
    """Travel time (seconds) of the closed tour station -> seq[1] -> ... -> seq[-1] -> station."""
    seq = np.asarray(seq)
    return float(T[seq, np.roll(seq, -1)].sum())


def is_valid(seq, n: int) -> bool:
    """Every node 0..n-1 exactly once and the tour starts at the station (node 0)."""
    seq = np.asarray(seq)
    return len(seq) == n and seq[0] == 0 and len(set(seq.tolist())) == n and seq.min() == 0 and seq.max() == n - 1


def exact_T(T) -> np.ndarray:
    """The raw travel times are given with one decimal (e.g. 399.3). They are stored as
    float32 after preprocessing; rounding the float64 cast to one decimal recovers exactly the
    same doubles that json parsing of the raw file produces (verified in tests/test_basic.py),
    which keeps the score bit-for-bit identical to the official implementation."""
    return np.round(np.asarray(T, dtype=np.float64), 1)


def normalize_matrix(T) -> np.ndarray:
    """Official normalisation: z-score over every entry of the route's matrix (including the
    zero diagonal, which is present in the raw JSON), then shift so that the minimum is 0."""
    T = np.asarray(T, dtype=np.float64)
    avg = np.mean(T)
    std = np.std(T)
    Z = (T - avg) / std
    return Z - Z.min()


def seq_dev(actual: list, sub: list) -> float:
    """Sequence deviation; inputs are closed lists (station at both ends)."""
    actual = list(actual[1:-1])
    sub = list(sub[1:-1])
    pos = {s: i for i, s in enumerate(actual)}
    comp = [pos[s] for s in sub]
    comp_sum = 0
    for ind in range(1, len(comp)):
        comp_sum += abs(comp[ind] - comp[ind - 1]) - 1
    n = len(actual)
    return (2 / (n * (n - 1))) * comp_sum


def erp_per_edit(actual: list, sub: list, M: np.ndarray, g: float = 1000.0):
    """Returns (ERP, number_of_edits). Bottom-up version of the official memoised recursion.

    D[i][j] is the ERP between actual[i:] and sub[j:], C[i][j] the edit count of the chosen
    alignment. Tie-breaking follows the official code: substitution first, then deleting from
    actual, then deleting from sub.
    """
    la, lb = len(actual), len(sub)
    Ml = M.tolist()
    INF = float("inf")
    D = [[INF] * (lb + 1) for _ in range(la + 1)]
    C = [[0] * (lb + 1) for _ in range(la + 1)]
    # official base cases: if len(sub)==0 -> gap_sum(actual); elif len(actual)==0 -> gap_sum(sub)
    for i in range(la + 1):
        d = 0
        for _ in range(la - i):
            d += g
        D[i][lb] = d
        C[i][lb] = la - i
    for j in range(lb):
        d = 0
        for _ in range(lb - j):
            d += g
        D[la][j] = d
        C[la][j] = lb - j
    for i in range(la - 1, -1, -1):
        a = actual[i]
        Drow, Dnext = D[i], D[i + 1]
        Crow, Cnext = C[i], C[i + 1]
        Ma = Ml[a]
        for j in range(lb - 1, -1, -1):
            b = sub[j]
            o1 = Dnext[j + 1] + Ma[b]
            o2 = Dnext[j] + g
            o3 = Drow[j + 1] + g
            d = min(o1, o2, o3)
            if d == o1:
                c = Cnext[j + 1] if a == b else Cnext[j + 1] + 1
            elif d == o2:
                c = Cnext[j] + 1
            else:
                c = Crow[j + 1] + 1
            Drow[j] = d
            Crow[j] = c
    return D[0][0], C[0][0]


def amazon_score(actual_seq, sub_seq, T, g: float = 1000.0) -> float:
    """Challenge score of a proposed sequence against the driver's actual sequence.

    actual_seq / sub_seq are open sequences of node indices starting with the station (node 0);
    the station is appended at the end exactly like route2list in the official code.
    """
    actual = [int(v) for v in actual_seq] + [int(actual_seq[0])]
    sub = [int(v) for v in sub_seq] + [int(sub_seq[0])]
    if len(actual) != len(sub) or set(actual) != set(sub) or actual[0] != sub[0]:
        raise ValueError("invalid sequence (the official scorer would assign the invalid-route score)")
    M = normalize_matrix(exact_T(T))
    sd = seq_dev(actual, sub)
    total, count = erp_per_edit(actual, sub, M, g)
    erp = 0 if count == 0 else total / count
    return sd * erp


if __name__ == "__main__":
    # Tiny smoke test on a real processed route if available.
    import data

    ex = data.load_split("val")[0]
    s = ex["actual_seq"]
    print("driver vs itself:", amazon_score(s, s, ex["T"]))
    rev = np.concatenate([[0], s[1:][::-1]])
    print("reversed driver:", amazon_score(s, rev, ex["T"]))
    print("travel time (s):", route_travel_time(s, ex["T"]))
