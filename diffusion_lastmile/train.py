"""Train the gated-GNN edge model on High-quality driver routes only.

Two objectives share the identical network, features, data, batch size, optimiser, learning-rate
schedule and number of steps:

    diffusion   DIFUSCO-style Bernoulli edge diffusion (the method under study)
    supervised  the ablation control: the same network trained as a one-shot supervised edge
                classifier (no noisy edge input, fixed timestep; see diffusion.OneShotSupervised)

and two levels:

    stop        edges between stops of the sparse k-nearest-neighbour graph
    zone        edges between the zones of a route (zone_level.py)

Training is resumable and time-boxed so that it can run in chunks shorter than the shell's
10-minute limit. The runs reported in results/ used, for each seed s in 0, 1, 2 (run_training.sh,
6 single-threaded runs at a time on 4 cores):

    timeout 590 python train.py --level stop --objective diffusion  --seed s --threads 1 --max-minutes 8.5
    timeout 590 python train.py --level stop --objective supervised --seed s --threads 1 --max-minutes 8.5
    timeout 590 python train.py --level zone --objective diffusion  --seed s --threads 1 --max-minutes 8.5
    timeout 590 python train.py --level zone --objective supervised --seed s --threads 1 --max-minutes 8.5
    (each repeated until the log says "training finished")

plus the spec-budget replicate (one single-threaded run; the other 3 cores ran validation OR-Tools jobs):

    timeout 590 python train.py --level stop --objective diffusion --seed 0 --tag budget --total-steps 3000 \
        --threads 1 --max-minutes 8.5

Since round 3 the checkpoint and log also record process CPU time (train_cpu_seconds); older
checkpoints only have wall-clock time (train_seconds).

with the per-level defaults below (stop: 7,000 steps, 64 hidden units, batch 8; zone: 2,500
steps, 128 hidden units, batch 16; 12 layers in both). The full configuration is fixed when a
run starts and is stored in the checkpoint. On resume the checkpoint's configuration, including
total_steps and therefore the cosine schedule, is kept. --total-steps is only applied on resume
when it is given explicitly and differs, and then a warning is printed because it changes the
learning-rate schedule of the remaining steps.

The checkpoint stores model, EMA model, optimiser, step, RNG states and the data-order
permutation, so a chunked run follows the same trajectory as an uninterrupted one (up to
floating-point non-determinism of multi-threaded CPU kernels). The final checkpoint (EMA weights)
is the one evaluated; the validation loss is logged for monitoring only and is not used to pick
a checkpoint.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time

import numpy as np
import torch

import data
from diffusion import collate, make_process
from model import EdgeDenoiser, count_parameters

HERE = os.path.dirname(os.path.abspath(__file__))

LEVEL_DEFAULTS = {
    "stop": {"total_steps": 7000, "hidden": 64, "layers": 12, "batch": 8, "warmup": 300},
    "zone": {"total_steps": 2500, "hidden": 128, "layers": 12, "batch": 16, "warmup": 200},
}


def default_ckpt(level: str, objective: str, seed: int, tag: str = "") -> str:
    """checkpoints/{level}_{objective}[_{tag}]_seed{seed}.pt (the directory can be overridden with
    the DIFFLM_CKPT_DIR environment variable, which the smoke tests use). tag="budget" is the
    spec-budget replicate (see README, "Training compute")."""
    d = os.environ.get("DIFFLM_CKPT_DIR", os.path.join(HERE, "checkpoints"))
    mid = f"_{tag}" if tag else ""
    return os.path.join(d, f"{level}_{objective}{mid}_seed{seed}.pt")


def build_model(cfg: dict) -> EdgeDenoiser:
    node_in = cfg.get("node_in", len(data.NODE_FEAT_NAMES))
    edge_in = cfg.get("edge_in", len(data.EDGE_FEAT_NAMES))
    return EdgeDenoiser(node_in, edge_in, hidden=cfg["hidden"], layers=cfg["layers"])


def lr_at(step: int, cfg: dict) -> float:
    w = cfg["warmup"]
    if step < w:
        return cfg["lr"] * (step + 1) / w
    p = min(1.0, (step - w) / max(1, cfg["total_steps"] - w))
    return cfg["lr_min"] + 0.5 * (cfg["lr"] - cfg["lr_min"]) * (1 + math.cos(math.pi * p))


@torch.no_grad()
def val_loss(model, proc, val, cfg, n_t=(50, 200, 400, 600, 800)) -> float:
    """Monitoring only. Diffusion: denoising BCE on the validation routes at fixed timesteps with
    fixed noise. Supervised: the plain BCE of the one-shot classifier."""
    model.eval()
    g = torch.Generator().manual_seed(1234)
    tot, cnt = 0.0, 0
    for i in range(0, len(val), cfg["batch"]):
        b = collate(val[i:i + cfg["batch"]])
        if proc.objective == "supervised":
            logits = proc.logits(model, b)
            tot += torch.nn.functional.binary_cross_entropy_with_logits(logits, b["edge_y"].float(), reduction="sum").item()
            cnt += b["edge_y"].numel()
            continue
        for t in n_t:
            tt = torch.full((b["num_graphs"],), t, dtype=torch.long)
            xt = proc.q_sample(b["edge_y"], tt[b["edge_graph"]], g)
            logits = model(b["node_feat"], b["edge_feat"], b["edge_index"], b["edge_graph"], xt, tt)
            tot += torch.nn.functional.binary_cross_entropy_with_logits(logits, b["edge_y"].float(), reduction="sum").item()
            cnt += b["edge_y"].numel()
    model.train()
    return tot / cnt


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--level", choices=["stop", "zone"], default="stop",
                    help="stop: edges between stops; zone: edges between zones")
    ap.add_argument("--objective", choices=["diffusion", "supervised"], default="diffusion",
                    help="diffusion: Bernoulli edge diffusion; supervised: one-shot classifier ablation")
    ap.add_argument("--ckpt", default=None, help="default checkpoints/{level}_{objective}[_{tag}]_seed{seed}.pt")
    ap.add_argument("--tag", default="", help="optional run tag in the default checkpoint name (e.g. budget)")
    ap.add_argument("--hidden", type=int, default=None, help="default: per level (stop 64, zone 128)")
    ap.add_argument("--layers", type=int, default=None, help="default: 12")
    ap.add_argument("--batch", type=int, default=None, help="default: per level (stop 8, zone 16)")
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--lr-min", type=float, default=2e-5)
    ap.add_argument("--warmup", type=int, default=None, help="default: per level (stop 300, zone 200)")
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--ema", type=float, default=0.999)
    ap.add_argument("--total-steps", type=int, default=None,
                    help="new run: default per level (stop 7000, zone 2500). Resume: kept from the checkpoint "
                         "unless given explicitly")
    ap.add_argument("--diffusion-steps", type=int, default=1000)
    ap.add_argument("--max-minutes", type=float, default=9.0)
    ap.add_argument("--save-every-min", type=float, default=3.0)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit-train", type=int, default=0, help="debug: use only the first N training routes")
    a = ap.parse_args(argv)
    torch.set_num_threads(a.threads)
    t_start = time.time()
    if a.ckpt is None:
        a.ckpt = default_ckpt(a.level, a.objective, a.seed, a.tag)
    prefix = "" if a.level == "stop" else "zone_"

    os.makedirs(os.path.dirname(os.path.abspath(a.ckpt)), exist_ok=True)
    log_path = os.path.splitext(a.ckpt)[0] + "_log.jsonl"
    train = data.load_split(prefix + "train")
    if a.limit_train:
        train = train[: a.limit_train]
    val = data.load_split(prefix + "val")

    if os.path.exists(a.ckpt):
        ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
        cfg = ck["cfg"]
        cfg.setdefault("objective", "diffusion")
        if cfg["objective"] != a.objective or cfg.get("level", "stop") != a.level:
            raise SystemExit(f"[train] {a.ckpt} holds a {cfg.get('level', 'stop')}/{cfg['objective']} run, "
                             f"not {a.level}/{a.objective}")
        if a.total_steps is not None and a.total_steps != cfg["total_steps"]:
            print(f"[train] WARNING: overriding total_steps {cfg['total_steps']} -> {a.total_steps}; this changes the "
                  f"learning-rate schedule of the remaining steps")
            cfg["total_steps"] = a.total_steps
        print(f"[train] resuming {cfg['level']}/{cfg['objective']} seed {cfg['seed']} from step {ck['step']} / "
              f"{cfg['total_steps']}")
    else:
        ck = None
        d = LEVEL_DEFAULTS[a.level]
        pick = lambda v, k: d[k] if v is None else v  # noqa: E731
        cfg = {"hidden": pick(a.hidden, "hidden"), "layers": pick(a.layers, "layers"), "batch": pick(a.batch, "batch"),
               "lr": a.lr, "lr_min": a.lr_min, "warmup": pick(a.warmup, "warmup"), "weight_decay": a.weight_decay,
               "ema": a.ema, "total_steps": pick(a.total_steps, "total_steps"), "diffusion_steps": a.diffusion_steps,
               "seed": a.seed, "n_train": len(train), "level": a.level, "objective": a.objective,
               "node_in": train[0]["node_feat"].shape[1], "edge_in": train[0]["edge_feat"].shape[1]}

    # Seed BEFORE the networks are built so that the initial weights depend only on the seed.
    torch.manual_seed(cfg["seed"])
    model = build_model(cfg)
    ema = build_model(cfg)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    proc = make_process(cfg)
    if ck is None:
        ema.load_state_dict(model.state_dict())
        step, epoch, pos = 0, 0, 0
        rng = np.random.default_rng(cfg["seed"])
        perm = rng.permutation(len(train))
        gen = torch.Generator().manual_seed(cfg["seed"])
        train_seconds = 0.0
        train_cpu_seconds = 0.0
        print(f"[train] new {cfg['level']}/{cfg['objective']} model, seed {cfg['seed']}, "
              f"{count_parameters(model):,} parameters, {len(train)} High-quality routes, {cfg['total_steps']} steps")
    else:
        model.load_state_dict(ck["model"])
        ema.load_state_dict(ck["ema"])
        opt.load_state_dict(ck["opt"])
        step, epoch, pos = ck["step"], ck["epoch"], ck["pos"]
        rng = np.random.default_rng()
        rng.bit_generator.state = ck["np_rng"]
        perm = ck["perm"]
        torch.set_rng_state(ck["torch_rng"])
        gen = torch.Generator()
        gen.set_state(ck["gen_state"])
        train_seconds = ck.get("train_seconds", 0.0)
        # CPU time was only recorded from round 3 on; None marks checkpoints that lack it
        train_cpu_seconds = ck.get("train_cpu_seconds") if ck.get("train_cpu_seconds") is not None else (
            0.0 if step == 0 else None)
    ema.eval()

    def save(tag=""):
        state = {"cfg": cfg, "model": model.state_dict(), "ema": ema.state_dict(), "opt": opt.state_dict(),
                 "step": step, "epoch": epoch, "pos": pos, "np_rng": rng.bit_generator.state, "perm": perm,
                 "torch_rng": torch.get_rng_state(), "gen_state": gen.get_state(),
                 "train_seconds": train_seconds, "train_cpu_seconds": train_cpu_seconds, "threads": a.threads}
        tmp = a.ckpt + ".tmp"
        torch.save(state, tmp)
        os.replace(tmp, a.ckpt)
        print(f"[train] saved checkpoint at step {step} {tag}")

    model.train()
    last_save = time.time()
    run_losses = []
    t_chunk = time.time()
    while step < cfg["total_steps"]:
        if (time.time() - t_start) / 60.0 > a.max_minutes:
            break
        t0 = time.time()
        c0 = time.process_time()
        if pos + cfg["batch"] > len(perm):
            epoch += 1
            pos = 0
            perm = rng.permutation(len(train))
        idx = perm[pos: pos + cfg["batch"]]
        pos += cfg["batch"]
        batch = collate([train[i] for i in idx], augment=True, rng=rng)
        for gparam in opt.param_groups:
            gparam["lr"] = lr_at(step, cfg)
        loss = proc.training_loss(model, batch, gen)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        with torch.no_grad():
            d = cfg["ema"] if step > cfg["warmup"] else 0.0
            for pe, pm in zip(ema.parameters(), model.parameters()):
                pe.mul_(d).add_(pm.detach(), alpha=1 - d)
            for be, bm in zip(ema.buffers(), model.buffers()):
                be.copy_(bm)
        step += 1
        train_seconds += time.time() - t0
        if train_cpu_seconds is not None:
            train_cpu_seconds += time.process_time() - c0
        run_losses.append(loss.item())
        if step % 50 == 0:
            rec = {"step": step, "epoch": epoch, "loss": float(np.mean(run_losses)), "lr": lr_at(step, cfg),
                   "train_minutes": train_seconds / 60.0, "sec_per_step": (time.time() - t_chunk) / 50,
                   "train_cpu_minutes": None if train_cpu_seconds is None else train_cpu_seconds / 60.0}
            t_chunk = time.time()
            run_losses = []
            if step % 500 == 0 or step == cfg["total_steps"]:
                rec["val_loss_ema"] = val_loss(ema, proc, val, cfg)
            print(json.dumps(rec), flush=True)
            with open(log_path, "a") as f:
                f.write(json.dumps(rec) + "\n")
        if time.time() - last_save > a.save_every_min * 60:
            save()
            last_save = time.time()
    save("(end of chunk)")
    if step >= cfg["total_steps"]:
        vl = val_loss(ema, proc, val, cfg)
        with open(log_path, "a") as f:
            f.write(json.dumps({"step": step, "final": True, "val_loss_ema": vl, "train_minutes": train_seconds / 60.0,
                                "train_cpu_minutes": None if train_cpu_seconds is None else train_cpu_seconds / 60.0})
                    + "\n")
        cpu_txt = "" if train_cpu_seconds is None else f", {train_cpu_seconds / 60.0:.1f} CPU-minutes"
        print(f"[train] training finished: {step} steps, {train_seconds / 60.0:.1f} minutes{cpu_txt}, val loss (EMA) {vl:.4f}")
    else:
        print(f"[train] chunk done at step {step}/{cfg['total_steps']}; run again to continue")


if __name__ == "__main__":
    main()
