"""Informer (Zhou et al., 2021) implementado desde cero para el pronóstico directo de 48 h.

Componentes: embedding (Conv1d + codificación posicional), atención ProbSparse, distilling entre capas
del encoder y decoder generativo (todas las horas futuras en una sola pasada).
No hay embedding temporal porque el dataset no trae fechas.
"""

import math

import torch
import torch.nn.functional as F
from torch import nn


def full_attention(q, k, v, causal=False):
    """softmax(QKᵀ/√d)V con máscara causal opcional. q, k, v: [B, H, L, D]."""
    scores = q @ k.transpose(-2, -1) / math.sqrt(q.shape[-1])
    if causal:
        future = torch.ones(scores.shape[-2:], dtype=torch.bool, device=q.device).triu(1)
        scores = scores.masked_fill(future, float("-inf"))
    return torch.softmax(scores, dim=-1) @ v


def prob_attention(q, k, v, factor=5, causal=False):
    """Atención ProbSparse: solo las u = c·ln(L) queries más "activas" calculan atención completa.

    La actividad de una query es M(q, K) = max(qKᵀ/√d) - mean(qKᵀ/√d), estimada con c·ln(L_K) claves
    muestreadas (un mismo muestreo para todo el batch). Las queries "perezosas" reciben la media de V
    (o la media acumulada si la atención es causal).
    """
    B, H, L_q, D = q.shape
    L_k = k.shape[2]
    u = min(L_q, factor * math.ceil(math.log(L_q)))
    n_sample = min(L_k, factor * math.ceil(math.log(L_k)))
    scale = 1 / math.sqrt(D)

    sampled = torch.randint(L_k, (n_sample,), device=q.device)
    s = (q @ k[:, :, sampled].transpose(-2, -1)) * scale  # [B, H, L_q, n_sample]
    top = (s.max(-1).values - s.mean(-1)).topk(u, dim=-1).indices  # [B, H, u]

    if causal:  # self-attention causal: L_q == L_k
        steps = torch.arange(1, L_k + 1, device=v.device, dtype=v.dtype).view(1, 1, -1, 1)
        out = v.cumsum(dim=2) / steps
    else:
        out = v.mean(dim=2, keepdim=True).expand(B, H, L_q, D)

    index = top.unsqueeze(-1).expand(-1, -1, -1, D)
    scores = (torch.gather(q, 2, index) @ k.transpose(-2, -1)) * scale  # [B, H, u, L_k]
    if causal:
        future = torch.arange(L_k, device=q.device).view(1, 1, 1, -1) > top.unsqueeze(-1)
        scores = scores.masked_fill(future, float("-inf"))
    return out.scatter(2, index, torch.softmax(scores, dim=-1) @ v)


class AttentionLayer(nn.Module):
    """Atención multi-cabeza: proyecciones Q, K, V, atención (ProbSparse o completa) y proyección de salida."""

    def __init__(self, d_model, n_heads, sparse, causal=False, factor=5):
        super().__init__()
        self.n_heads, self.sparse, self.causal, self.factor = n_heads, sparse, causal, factor
        self.q = nn.Linear(d_model, d_model)
        self.k = nn.Linear(d_model, d_model)
        self.v = nn.Linear(d_model, d_model)
        self.out = nn.Linear(d_model, d_model)

    def forward(self, x_q, x_kv):
        B, L_q, _ = x_q.shape
        q = self.q(x_q).view(B, L_q, self.n_heads, -1).transpose(1, 2)
        k = self.k(x_kv).view(B, x_kv.shape[1], self.n_heads, -1).transpose(1, 2)
        v = self.v(x_kv).view(B, x_kv.shape[1], self.n_heads, -1).transpose(1, 2)
        if self.sparse:
            o = prob_attention(q, k, v, self.factor, self.causal)
        else:
            o = full_attention(q, k, v, self.causal)
        return self.out(o.transpose(1, 2).reshape(B, L_q, -1))


def feed_forward(d_model, d_ff, dropout):
    return nn.Sequential(nn.Linear(d_model, d_ff), nn.GELU(), nn.Dropout(dropout), nn.Linear(d_ff, d_model))


class EncoderLayer(nn.Module):
    """Self-attention ProbSparse + feed-forward, con conexiones residuales y LayerNorm."""

    def __init__(self, d_model, n_heads, d_ff, dropout, factor):
        super().__init__()
        self.attn = AttentionLayer(d_model, n_heads, sparse=True, factor=factor)
        self.ff = feed_forward(d_model, d_ff, dropout)
        self.norm1, self.norm2 = nn.LayerNorm(d_model), nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        x = self.norm1(x + self.drop(self.attn(x, x)))
        return self.norm2(x + self.drop(self.ff(x)))


class DistilConv(nn.Module):
    """Distilling: Conv1d -> BatchNorm -> ELU -> MaxPool(stride 2). Reduce la longitud a la mitad."""

    def __init__(self, d_model):
        super().__init__()
        self.conv = nn.Conv1d(d_model, d_model, kernel_size=3, padding=1, padding_mode="replicate")
        self.norm = nn.BatchNorm1d(d_model)
        self.pool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):  # [B, L, d] -> [B, L/2, d]
        x = self.pool(F.elu(self.norm(self.conv(x.transpose(1, 2)))))
        return x.transpose(1, 2)


class DecoderLayer(nn.Module):
    """Self-attention ProbSparse causal + cross-attention completa sobre el encoder + feed-forward."""

    def __init__(self, d_model, n_heads, d_ff, dropout, factor):
        super().__init__()
        self.self_attn = AttentionLayer(d_model, n_heads, sparse=True, causal=True, factor=factor)
        self.cross_attn = AttentionLayer(d_model, n_heads, sparse=False)
        self.ff = feed_forward(d_model, d_ff, dropout)
        self.norm1, self.norm2, self.norm3 = nn.LayerNorm(d_model), nn.LayerNorm(d_model), nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, memory):
        x = self.norm1(x + self.drop(self.self_attn(x, x)))
        x = self.norm2(x + self.drop(self.cross_attn(x, memory)))
        return self.norm3(x + self.drop(self.ff(x)))


class Embedding(nn.Module):
    """Proyección de las variables con Conv1d (k=3) + codificación posicional sinusoidal."""

    def __init__(self, c_in, d_model, dropout, max_len=512):
        super().__init__()
        # padding "replicate" en vez del circular original: no pega la hora 336 con la hora 1
        self.token = nn.Conv1d(c_in, d_model, kernel_size=3, padding=1, padding_mode="replicate")
        position = torch.arange(max_len).unsqueeze(1)
        freq = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2], pe[:, 1::2] = torch.sin(position * freq), torch.cos(position * freq)
        self.register_buffer("pe", pe)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):  # [B, L, c_in] -> [B, L, d_model]
        x = self.token(x.transpose(1, 2)).transpose(1, 2)
        return self.drop(x + self.pe[: x.shape[1]])


class Informer(nn.Module):
    """Encoder (2 capas + 1 distilling: 336 -> 168) y decoder generativo de 1 capa -> [B, 48]."""

    def __init__(self, c_in, horizon=48, label_len=48, d_model=256, n_heads=8, e_layers=2, d_layers=1,
                 d_ff=1024, dropout=0.1, factor=5):
        super().__init__()
        self.horizon, self.label_len = horizon, label_len
        self.enc_embedding = Embedding(c_in, d_model, dropout)
        self.dec_embedding = Embedding(c_in, d_model, dropout)
        self.encoder = nn.ModuleList([EncoderLayer(d_model, n_heads, d_ff, dropout, factor) for _ in range(e_layers)])
        self.distil = nn.ModuleList([DistilConv(d_model) for _ in range(e_layers - 1)])
        self.enc_norm = nn.LayerNorm(d_model)
        self.decoder = nn.ModuleList([DecoderLayer(d_model, n_heads, d_ff, dropout, factor) for _ in range(d_layers)])
        self.dec_norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, 1)

    def encode(self, x):
        h = self.enc_embedding(x)
        for i, layer in enumerate(self.encoder):
            h = layer(h)
            if i < len(self.distil):
                h = self.distil[i](h)
        return self.enc_norm(h)

    def forward(self, x):  # x: [B, 336, c_in]
        memory = self.encode(x)
        # decoder: "start token" = últimas label_len horas conocidas + ceros en las horas a pronosticar
        dec_in = torch.cat([x[:, -self.label_len :], torch.zeros_like(x[:, : self.horizon])], dim=1)
        d = self.dec_embedding(dec_in)
        for layer in self.decoder:
            d = layer(d, memory)
        return self.head(self.dec_norm(d))[:, -self.horizon :, 0]
