"""Bernoulli (binary categorical) discrete diffusion over graph edges, DIFUSCO style.

Forward process (D3PM with a symmetric 2x2 transition, Austin et al. 2021; DIFUSCO, Sun & Yang
2023): every edge variable x in {0,1} is flipped with probability beta_t at step t, so

    q(x_t | x_0): flip probability  p_t = (1 - prod_{s<=t}(1 - 2 beta_s)) / 2.

We keep c_t = prod_{s<=t}(1 - 2 beta_s) with c_0 = 1. For s < t the flip probability of the
s -> t transition is (1 - c_t / c_s) / 2.

Training: sample t ~ U{1..T} per graph, corrupt the driver-tour adjacency x_0 into x_t, and
train the denoiser to predict x_0 with binary cross-entropy (the x_0-parameterisation used by
DIFUSCO for categorical diffusion).

Sampling (DDIM-style skipping, as in DIFUSCO's fast inference): start from x_T ~ Bernoulli(0.5)
and walk a short decreasing list of timesteps t_K > ... > t_1 > 0. At each step predict
p_theta(x_0 = 1 | x_t) and sample x_s from the exact posterior
    p(x_s | x_t) = sum_{x_0} q(x_s | x_t, x_0) p_theta(x_0 | x_t).
The final step returns the x_0 probabilities, which form the edge heat-map that the decoder
turns into a tour.
"""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F


class BernoulliDiffusion:
    objective = "diffusion"

    def __init__(self, T: int = 1000, beta_start: float = 1e-4, beta_end: float = 0.02):
        self.T = T
        betas = np.linspace(beta_start, beta_end, T, dtype=np.float64)
        c = np.cumprod(1.0 - 2.0 * betas)
        self.c = torch.tensor(np.concatenate([[1.0], c]), dtype=torch.float64)  # c[0] = 1

    # --- forward process -------------------------------------------------------------------
    def flip_prob(self, t: torch.Tensor, s: torch.Tensor | None = None) -> torch.Tensor:
        """Flip probability of the s -> t transition (s = 0 by default)."""
        ct = self.c[t]
        cs = torch.ones_like(ct) if s is None else self.c[s]
        return ((1.0 - ct / cs) / 2.0).float()

    def q_sample(self, x0: torch.Tensor, t_edge: torch.Tensor, generator=None) -> torch.Tensor:
        p = self.flip_prob(t_edge)
        flip = torch.rand(x0.shape, generator=generator) < p
        return (x0.bool() ^ flip).long()

    def training_loss(self, model, batch, generator=None) -> torch.Tensor:
        G = batch["num_graphs"]
        t = torch.randint(1, self.T + 1, (G,), generator=generator)
        x0 = batch["edge_y"]
        xt = self.q_sample(x0, t[batch["edge_graph"]], generator)
        logits = model(batch["node_feat"], batch["edge_feat"], batch["edge_index"], batch["edge_graph"], xt, t)
        return F.binary_cross_entropy_with_logits(logits, x0.float())

    # --- reverse process -------------------------------------------------------------------
    def posterior_one(self, xt: torch.Tensor, p0: torch.Tensor, t: int, s: int) -> torch.Tensor:
        """P(x_s = 1 | x_t, predicted p(x_0 = 1)) for scalar timesteps t > s >= 1."""
        c = self.c
        a = float((1.0 - c[t] / c[s]) / 2.0)  # flip prob s -> t
        xt_f = xt.float()
        # q(x_t | x_s = 1) and q(x_t | x_s = 0)
        lik1 = xt_f * (1 - a) + (1 - xt_f) * a
        lik0 = xt_f * a + (1 - xt_f) * (1 - a)
        out = torch.zeros_like(p0)
        for x0v, w in ((1, p0), (0, 1.0 - p0)):
            b = float((1.0 - c[s]) / 2.0)  # flip prob 0 -> s
            prior1 = (1 - b) if x0v == 1 else b  # q(x_s = 1 | x_0)
            num = lik1 * prior1
            den = num + lik0 * (1 - prior1)
            out = out + w * num / den.clamp_min(1e-12)
        return out.clamp(0.0, 1.0)

    def timesteps(self, steps: int, schedule: str = "cosine") -> list:
        """Decreasing list [t_K, ..., t_1] of integer timesteps used for skipping inference."""
        if schedule == "linear":
            ts = np.linspace(self.T, 1, steps)
        else:  # DIFUSCO's cosine inference schedule: denser near t = 0
            i = np.arange(steps) / max(1, steps - 1)
            ts = 1 + (self.T - 1) * (1 - np.cos(i * math.pi / 2))[::-1]
        ts = [int(round(v)) for v in ts]
        out = []
        for v in ts:
            if not out or v < out[-1]:
                out.append(v)
        return out

    @torch.no_grad()
    def sample_heatmap(self, model, batch, steps: int = 10, n_samples: int = 1, schedule: str = "cosine",
                       generator=None) -> torch.Tensor:
        """Returns the mean over n_samples of the final p(x_0 = 1) per edge ([E])."""
        E = batch["edge_y"].shape[0]
        G = batch["num_graphs"]
        ts = self.timesteps(steps, schedule)
        acc = torch.zeros(E)
        for _ in range(n_samples):
            xt = (torch.rand(E, generator=generator) < 0.5).long()
            p0 = None
            for k, t in enumerate(ts):
                tt = torch.full((G,), t, dtype=torch.long)
                logits = model(batch["node_feat"], batch["edge_feat"], batch["edge_index"], batch["edge_graph"], xt, tt)
                p0 = torch.sigmoid(logits)
                if k + 1 < len(ts):
                    s = ts[k + 1]
                    ps = self.posterior_one(xt, p0, t, s)
                    xt = (torch.rand(E, generator=generator) < ps).long()
            acc += p0
        return acc / n_samples


class OneShotSupervised:
    """Ablation control for the diffusion process (same network, data, features and compute).

    The SAME EdgeDenoiser is trained as a plain one-shot supervised edge classifier: there is no
    noisy edge input and no timestep, i.e. x_t is fixed to all zeros and t is fixed to T for every
    graph, both in training and at inference. The loss is the same binary cross-entropy against the
    driver-tour adjacency. At inference one forward pass gives p(edge in driver tour) directly, and
    the result is decoded exactly like the diffusion heat-map. Comparing the two isolates what the
    denoising process adds over ordinary supervised learning with the same architecture.

    The interface mirrors BernoulliDiffusion (training_loss, sample_heatmap) so train.py and
    evaluate.py treat both identically; steps / n_samples / generator are accepted and ignored.
    """

    objective = "supervised"

    def __init__(self, T: int = 1000):
        self.T = T

    def _inputs(self, batch):
        E = batch["edge_y"].shape[0]
        return torch.zeros(E, dtype=torch.long), torch.full((batch["num_graphs"],), self.T, dtype=torch.long)

    def logits(self, model, batch):
        xt, t = self._inputs(batch)
        return model(batch["node_feat"], batch["edge_feat"], batch["edge_index"], batch["edge_graph"], xt, t)

    def training_loss(self, model, batch, generator=None) -> torch.Tensor:
        return F.binary_cross_entropy_with_logits(self.logits(model, batch), batch["edge_y"].float())

    @torch.no_grad()
    def sample_heatmap(self, model, batch, steps: int = 1, n_samples: int = 1, schedule: str = "cosine",
                       generator=None) -> torch.Tensor:
        return torch.sigmoid(self.logits(model, batch))


def make_process(cfg: dict):
    """The training / sampling process named in a checkpoint config (diffusion by default)."""
    if cfg.get("objective", "diffusion") == "supervised":
        return OneShotSupervised(cfg["diffusion_steps"])
    return BernoulliDiffusion(cfg["diffusion_steps"])


# ----------------------------------------------------------------------------------------
# Batching
# ----------------------------------------------------------------------------------------
def collate(examples: list, augment: bool = False, rng: np.random.Generator | None = None) -> dict:
    """Concatenate several route graphs into one disjoint batched graph.

    augment=True applies a random rotation / reflection of the coordinate features about the
    centre of the unit square (travel times and zone features are invariant to this).
    """
    node_feats, edge_feats, edge_idx, edge_graph, ys, offsets = [], [], [], [], [], []
    off = 0
    for g, ex in enumerate(examples):
        nf = ex["node_feat"].astype(np.float32).copy()
        if augment:
            ang = rng.uniform(0, 2 * np.pi)
            R = np.array([[np.cos(ang), -np.sin(ang)], [np.sin(ang), np.cos(ang)]], dtype=np.float32)
            if rng.random() < 0.5:
                R = R @ np.array([[1, 0], [0, -1]], dtype=np.float32)
            nf[:, :2] = (nf[:, :2] - 0.5) @ R.T + 0.5
        n = nf.shape[0]
        node_feats.append(nf)
        edge_feats.append(ex["edge_feat"].astype(np.float32))
        edge_idx.append(ex["edge_index"] + off)
        edge_graph.append(np.full(ex["edge_index"].shape[1], g, dtype=np.int64))
        ys.append(ex["edge_y"].astype(np.int64))
        offsets.append(off)
        off += n
    return {
        "num_graphs": len(examples),
        "node_feat": torch.from_numpy(np.concatenate(node_feats)),
        "edge_feat": torch.from_numpy(np.concatenate(edge_feats)),
        "edge_index": torch.from_numpy(np.concatenate(edge_idx, axis=1)),
        "edge_graph": torch.from_numpy(np.concatenate(edge_graph)),
        "edge_y": torch.from_numpy(np.concatenate(ys)),
        "node_offsets": offsets,
        "edge_counts": [ex["edge_index"].shape[1] for ex in examples],
    }


def split_edges(values: torch.Tensor, batch: dict) -> list:
    """Split a batched per-edge tensor back into per-graph numpy arrays."""
    return [v.numpy() for v in torch.split(values, batch["edge_counts"])]
