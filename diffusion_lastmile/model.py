"""Anisotropic edge-gated GNN denoiser for DIFUSCO-style Bernoulli edge diffusion.

The network follows the sparse GNN encoder of DIFUSCO (Sun & Yang, NeurIPS 2023), which in turn
uses the anisotropic, edge-gated graph convolution of Bresson & Laurent (2017) / Joshi et al.
(2019). For every layer l, with node states h and (directed) edge states e:

    e_hat_ij = A h_i + B h_j + C e_ij
    h_i'     = U h_i + sum_j sigmoid(e_hat_ij) * V h_j / (sum_j sigmoid(e_hat_ij) + eps)
    h_i      = h_i + SiLU(LN(h_i'))
    e_ij     = e_ij + SiLU(LN(e_hat_ij)) + W_t temb(t)

The input to each edge is its real-data features (travel times, kNN rank, zone relations,
zone-order difference, station flag) plus the embedding of the current noisy edge state x_t.
Node inputs are the normalised stop coordinates (raw and sinusoidal), the station flag and
zone-rank features. The output is one logit per directed edge: the model's estimate of
p(x_0 = 1 | x_t, t, route), i.e. whether the driver drove along that edge.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def sinusoidal(x: torch.Tensor, dim: int, max_period: float = 10000.0) -> torch.Tensor:
    """Standard transformer/diffusion sinusoidal embedding of a 1-D tensor."""
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(half, dtype=torch.float32, device=x.device) / half)
    args = x.float()[:, None] * freqs[None, :]
    return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)


def coord_embedding(xy: torch.Tensor, n_freq: int = 8) -> torch.Tensor:
    """Fourier features of the normalised coordinates (similar to DIFUSCO's PositionEmbeddingSine)."""
    freqs = (2.0 ** torch.arange(n_freq, dtype=torch.float32, device=xy.device)) * math.pi
    args = xy[:, :, None] * freqs[None, None, :]  # [N, 2, F]
    return torch.cat([torch.sin(args), torch.cos(args)], dim=-1).flatten(1)


class GatedGCNLayer(nn.Module):
    def __init__(self, hidden: int):
        super().__init__()
        self.U = nn.Linear(hidden, hidden)
        self.V = nn.Linear(hidden, hidden)
        self.A = nn.Linear(hidden, hidden)
        self.B = nn.Linear(hidden, hidden)
        self.C = nn.Linear(hidden, hidden)
        self.norm_h = nn.LayerNorm(hidden)
        self.norm_e = nn.LayerNorm(hidden)
        self.t_proj = nn.Sequential(nn.SiLU(), nn.Linear(hidden, hidden))

    def forward(self, h, e, src, dst, temb_edges):
        Uh, Vh, Ah, Bh = self.U(h), self.V(h), self.A(h), self.B(h)
        e_hat = Ah[src] + Bh[dst] + self.C(e)
        gates = torch.sigmoid(e_hat)
        num = torch.zeros_like(h).index_add_(0, src, gates * Vh[dst])
        den = torch.zeros_like(h).index_add_(0, src, gates)
        h_new = Uh + num / (den + 1e-6)
        h = h + F.silu(self.norm_h(h_new))
        e = e + F.silu(self.norm_e(e_hat)) + self.t_proj(temb_edges)
        return h, e


class EdgeDenoiser(nn.Module):
    def __init__(self, node_in: int, edge_in: int, hidden: int = 64, layers: int = 12, n_freq: int = 8):
        super().__init__()
        self.hidden = hidden
        self.n_freq = n_freq
        self.node_in = nn.Linear(node_in + 4 * n_freq, hidden)
        self.edge_in = nn.Linear(edge_in, hidden)
        self.xt_emb = nn.Embedding(2, hidden)
        self.time_mlp = nn.Sequential(nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, hidden))
        self.layers = nn.ModuleList([GatedGCNLayer(hidden) for _ in range(layers)])
        self.out = nn.Sequential(nn.LayerNorm(hidden), nn.SiLU(), nn.Linear(hidden, 1))

    def forward(self, node_feat, edge_feat, edge_index, edge_graph, x_t, t):
        """
        node_feat  [N, Fn]  float
        edge_feat  [E, Fe]  float
        edge_index [2, E]   long (global node indices of the batched graph)
        edge_graph [E]      long (graph id of every edge)
        x_t        [E]      long in {0,1} (noisy edge state)
        t          [G]      long diffusion timestep of every graph (1..T)
        returns    [E]      logits of p(x_0 = 1)
        """
        src, dst = edge_index[0], edge_index[1]
        h = self.node_in(torch.cat([node_feat, coord_embedding(node_feat[:, :2], self.n_freq)], dim=-1))
        e = self.edge_in(edge_feat) + self.xt_emb(x_t)
        temb = self.time_mlp(sinusoidal(t, self.hidden))[edge_graph]
        for layer in self.layers:
            h, e = layer(h, e, src, dst, temb)
        return self.out(e).squeeze(-1)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
