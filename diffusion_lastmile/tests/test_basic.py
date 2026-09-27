"""Fast self-tests. Run with:  python tests/test_basic.py   (or python -m pytest tests -q)

They need the preprocessed data (python data.py download && python data.py preprocess). The
cross-check against the official challenge scoring script runs if data/reference/official_score.py
was downloaded by data.py.
"""
from __future__ import annotations

import importlib.util
import json
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

import data  # noqa: E402
from baselines import nearest_neighbour, ortools_tsp, softdist_heatmap  # noqa: E402
from decode import greedy_decode, two_opt  # noqa: E402
from diffusion import BernoulliDiffusion, collate  # noqa: E402
from model import EdgeDenoiser  # noqa: E402
from score import amazon_score, exact_T, is_valid, route_travel_time  # noqa: E402

torch.set_num_threads(1)
_VAL = None


def val():
    global _VAL
    if _VAL is None:
        _VAL = data.load_split("val")
    return _VAL


def _official():
    p = os.path.join(data.REF_DIR, "official_score.py")
    if not os.path.exists(p):
        return None
    spec = importlib.util.spec_from_file_location("official_score", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_example_structure():
    ex = val()[0]
    n = len(ex["stop_ids"])
    assert ex["node_feat"].shape == (n, len(data.NODE_FEAT_NAMES))
    assert ex["edge_feat"].shape[1] == len(data.EDGE_FEAT_NAMES)
    assert ex["actual_seq"][0] == 0 and is_valid(ex["actual_seq"], n)
    # every positive edge target is an edge of the driver's closed tour
    a = ex["actual_seq"]
    tour = set(zip(a.tolist(), np.roll(a, -1).tolist()))
    pos = {tuple(e) for e in ex["edge_index"][:, ex["edge_y"] == 1].T.tolist()}
    assert pos <= tour and len(pos) >= 0.85 * n
    assert ex["route_score"] == "High"


def test_exact_travel_times_match_raw_json():
    """float32 storage + rounding to one decimal reproduces the raw JSON doubles exactly."""
    import ijson

    raw = os.path.join(data.RAW_DIR, "eval_travel_times.json")
    with open(raw, "rb") as f:
        rid, mat = next(ijson.kvitems(f, "", use_float=True))
    er = json.load(open(os.path.join(data.RAW_DIR, "eval_route_data.json")))
    ea = json.load(open(os.path.join(data.RAW_DIR, "eval_actual_sequences.json")))
    ex = data.build_example(rid, er[rid], ea[rid]["actual"], mat, k=16)
    T = exact_T(ex["T"])
    ids = ex["stop_ids"]
    for i, a in enumerate(ids):
        for j, b in enumerate(ids):
            assert T[i, j] == mat[a][b]


def test_score_matches_official():
    off = _official()
    if off is None:
        print("  (official scoring script not available, skipped)")
        return
    ex = val()[1]
    T = exact_T(ex["T"])
    ids = ex["stop_ids"]
    cost = {a: {b: float(T[i, j]) for j, b in enumerate(ids)} for i, a in enumerate(ids)}
    act = ex["actual_seq"]
    rng = np.random.default_rng(0)
    cands = [nearest_neighbour(T), two_opt(nearest_neighbour(T), T),
             np.concatenate([[0], rng.permutation(act[1:])])]
    sys.setrecursionlimit(100000)
    for s in cands:
        a_list = [ids[i] for i in act] + [ids[0]]
        s_list = [ids[i] for i in s] + [ids[0]]
        ref = off.score(a_list, s_list, {k: dict(v) for k, v in cost.items()})
        mine = amazon_score(act, s, ex["T"])
        assert mine == ref, (mine, ref)
    assert amazon_score(act, act, ex["T"]) == 0.0


def test_decoder_and_two_opt():
    ex = val()[2]
    T = ex["T"].astype(np.float64)
    n = len(T)
    rng = np.random.default_rng(0)
    for trial in range(3):
        h = rng.random(ex["edge_index"].shape[1])
        s = greedy_decode(n, ex["edge_index"], h, T)
        assert is_valid(s, n)
        s2 = two_opt(s, T)
        assert is_valid(s2, n)
        assert route_travel_time(s2, T) <= route_travel_time(s, T) + 1e-6
    # a heat-map equal to the driver's edges must decode to (almost exactly) the driver's tour
    s = greedy_decode(n, ex["edge_index"], ex["edge_y"].astype(float), T)
    a = ex["actual_seq"]
    tour = set(zip(a.tolist(), np.roll(a, -1).tolist()))
    got = set(zip(s.tolist(), np.roll(s, -1).tolist()))
    assert len(tour & got) >= ex["edge_y"].sum() - 1
    # the 2-opt delta formula agrees with recomputing the tour cost
    s = nearest_neighbour(T)
    before = route_travel_time(s, T)
    i, j = 5, 40
    s_new = s.copy()
    s_new[i:j + 1] = s_new[i:j + 1][::-1]
    a_, b_ = s[i - 1], s[j + 1]
    F = sum(T[s[t], s[t + 1]] for t in range(i, j))
    R = sum(T[s[t + 1], s[t]] for t in range(i, j))
    delta = T[a_, s[j]] + R + T[s[i], b_] - T[a_, s[i]] - F - T[s[j], b_]
    assert abs(route_travel_time(s_new, T) - before - delta) < 1e-6


def test_softdist_and_ortools():
    ex = val()[3]
    T = ex["T"].astype(np.float64)
    h = softdist_heatmap(T, ex["edge_index"], 0.1)
    assert h.shape == (ex["edge_index"].shape[1],) and np.all(h >= 0) and np.all(h <= 1)
    s = ortools_tsp(T, time_limit_s=1.0)
    assert is_valid(s, len(T))
    assert route_travel_time(s, T) <= route_travel_time(nearest_neighbour(T), T) + 1e-6


def test_diffusion_process_and_model():
    diff = BernoulliDiffusion(1000)
    # marginal of q(x_t | x_0) matches the closed-form flip probability
    x0 = torch.zeros(200000, dtype=torch.long)
    for t in [1, 100, 1000]:
        xt = diff.q_sample(x0, torch.full((x0.numel(),), t, dtype=torch.long), torch.Generator().manual_seed(t))
        p = diff.flip_prob(torch.tensor([t]))[0].item()
        assert abs(xt.float().mean().item() - p) < 5e-3
    # posterior: with a perfectly confident x_0 prediction the posterior equals q(x_s | x_t, x_0)
    xt = torch.tensor([0, 1, 0, 1])
    p0 = torch.tensor([1.0, 1.0, 0.0, 0.0])
    ps = diff.posterior_one(xt, p0, 500, 100)
    assert torch.all(ps[:2] > 0.5) and torch.all(ps[2:] < 0.5)
    # model forward / backward and sampling on real graphs
    b = collate(val()[:2])
    m = EdgeDenoiser(len(data.NODE_FEAT_NAMES), len(data.EDGE_FEAT_NAMES), hidden=16, layers=2)
    loss = diff.training_loss(m, b, torch.Generator().manual_seed(0))
    loss.backward()
    assert torch.isfinite(loss)
    h = diff.sample_heatmap(m.eval(), collate(val()[:1]), steps=3, n_samples=1, generator=torch.Generator().manual_seed(0))
    assert h.shape[0] == val()[0]["edge_index"].shape[1] and torch.all((h >= 0) & (h <= 1))
    assert diff.timesteps(10)[0] == 1000 and diff.timesteps(10)[-1] == 1


def test_zone_level():
    from evaluate import _seq_from_zone_order, decode_zone_order
    from zone_level import History, build_zone_example, history_heatmap, zone_order_penalty_matrix

    exs = val()[:10]
    hist = History(exs)
    ex = exs[0]
    z = build_zone_example(ex, hist, leave_one_out=True)
    m = len(z["zone_names"])
    assert z["edge_index"].shape[1] == m * (m - 1) and z["edge_y"].sum() == m
    # leave-one-out: with only this route in the history, its own transitions must be invisible
    z_alone = build_zone_example(ex, History([ex]), leave_one_out=True)
    assert np.all(z_alone["edge_feat"][:, 10:13].astype(float) == 0)
    assert hist.cnt == History(exs).cnt  # history restored after leave-one-out
    order = decode_zone_order(z, history_heatmap(z), "greedy")
    assert sorted(decode_zone_order(z, history_heatmap(z), "ml")) == sorted(order)
    assert sorted(decode_zone_order(z, history_heatmap(z), "greedy2opt")) == sorted(order)
    assert sorted(order) == sorted(set(ex["zones"][1:]))
    # a sequence that follows the zone order exactly has no penalised arc
    seq = _seq_from_zone_order(ex, order)
    T = ex["T"].astype(np.float64)
    C = zone_order_penalty_matrix(T, ex["zones"], order, 4.0)
    assert np.isclose(route_travel_time(seq, C), route_travel_time(seq, T))


def test_supervised_ablation_and_batched_sampling():
    """The one-shot supervised control uses the identical network with constant x_t / t inputs;
    the batched multi-sample diffusion heat-map has the right shape and range."""
    from diffusion import OneShotSupervised, make_process
    from evaluate import heatmap

    torch.manual_seed(0)
    m = EdgeDenoiser(len(data.NODE_FEAT_NAMES), len(data.EDGE_FEAT_NAMES), hidden=16, layers=2)
    sup = make_process({"objective": "supervised", "diffusion_steps": 1000})
    assert isinstance(sup, OneShotSupervised) and make_process({"diffusion_steps": 1000}).objective == "diffusion"
    b = collate(val()[:2])
    loss = sup.training_loss(m, b)
    loss.backward()
    assert torch.isfinite(loss)
    m.eval()
    h1 = heatmap(m, sup, val()[0], 10, 4, seed=0)
    h2 = heatmap(m, sup, val()[0], 10, 4, seed=1)
    assert np.array_equal(h1, h2)  # deterministic: no noise input
    ex = val()[0]
    hd = heatmap(m, make_process({"diffusion_steps": 1000}), ex, 3, 3, seed=0)
    assert hd.shape == (ex["edge_index"].shape[1],) and np.all((hd >= 0) & (hd <= 1))


def test_train_resume_keeps_schedule():
    """Resuming without --total-steps keeps the checkpoint's schedule; an explicit value
    overrides it; the same seed gives the same initial weights."""
    import tempfile

    import train

    with tempfile.TemporaryDirectory() as d:
        base = ["--level", "zone", "--objective", "supervised", "--limit-train", "8", "--hidden", "8", "--layers", "1",
                "--batch", "4", "--warmup", "1", "--threads", "1"]
        p1, p2 = os.path.join(d, "a.pt"), os.path.join(d, "b.pt")
        train.main(base + ["--ckpt", p1, "--total-steps", "5", "--max-minutes", "0"])
        train.main(base + ["--ckpt", p2, "--total-steps", "5", "--max-minutes", "0"])
        c1, c2 = torch.load(p1, weights_only=False), torch.load(p2, weights_only=False)
        assert all(torch.equal(c1["model"][k], c2["model"][k]) for k in c1["model"])
        assert c1["cfg"]["total_steps"] == 5 and c1["cfg"]["objective"] == "supervised"
        train.main(base + ["--ckpt", p1, "--max-minutes", "0"])  # resume, flag omitted
        assert torch.load(p1, weights_only=False)["cfg"]["total_steps"] == 5
        train.main(base + ["--ckpt", p1, "--max-minutes", "1"])  # resume and finish
        c = torch.load(p1, weights_only=False)
        assert c["step"] == 5 and c["cfg"]["total_steps"] == 5
        assert abs(train.lr_at(c["cfg"]["total_steps"], c["cfg"]) - c["cfg"]["lr_min"]) < 1e-12  # schedule ends at lr_min
        train.main(base + ["--ckpt", p1, "--total-steps", "7", "--max-minutes", "0"])  # explicit override
        assert torch.load(p1, weights_only=False)["cfg"]["total_steps"] == 7


def test_splits_disjoint():
    """No route is in two splits; the final (fresh) split, when drawn, is disjoint from the reused test
    split and from every training-dataset route."""
    sp = json.load(open(os.path.join(ROOT, "results", "splits.json")))
    names = [k for k in ["train", "val", "heldout", "test", "fresh"] if k in sp]
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            assert not (set(sp[a]) & set(sp[b])), (a, b)
    assert len(sp["test"]) >= 200
    if "fresh" in sp:
        assert len(sp["fresh"]) >= 200


def test_cpu_time_recorded_and_round1_optional():
    """New training runs record process CPU time; the optional round1 stage skips cleanly when the
    round-1 archive is absent (clean checkout)."""
    import argparse
    import tempfile

    import evaluate
    import train

    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "c.pt")
        train.main(["--level", "zone", "--objective", "diffusion", "--limit-train", "8", "--hidden", "8", "--layers", "1",
                    "--batch", "4", "--warmup", "1", "--threads", "1", "--ckpt", p, "--total-steps", "3",
                    "--max-minutes", "1"])
        c = torch.load(p, weights_only=False)
        assert c["step"] == 3 and c["train_cpu_seconds"] is not None and c["train_cpu_seconds"] > 0
        old = evaluate.CACHE_R1
        try:
            evaluate.CACHE_R1 = os.path.join(d, "no_such_archive")
            evaluate.round1(argparse.Namespace())  # must return without error
        finally:
            evaluate.CACHE_R1 = old


def main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    t0 = time.time()
    for t in tests:
        s = time.time()
        t()
        print(f"ok  {t.__name__} ({time.time() - s:.1f}s)")
    print(f"all {len(tests)} tests passed in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
