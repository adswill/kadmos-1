"""Llama-style decoder (RoPE + RMSNorm + SwiGLU, tied embeddings), written from scratch."""
import torch, torch.nn as nn, torch.nn.functional as F
from dataclasses import dataclass

EOT, USER, BOT, END = 0, 1, 2, 3  # special token ids (fixed by build_data.py)


@dataclass
class Config:
    vocab: int = 16384
    ctx: int = 512
    layers: int = 12
    heads: int = 8
    dim: int = 512
    dropout: float = 0.1


class RMSNorm(nn.Module):
    def __init__(s, d, eps=1e-6):
        super().__init__()
        s.w, s.eps = nn.Parameter(torch.ones(d)), eps

    def forward(s, x):
        return (x.float() * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + s.eps)).type_as(x) * s.w


def rope_table(ctx, hd, base=10000.0):
    inv = 1.0 / (base ** (torch.arange(0, hd, 2).float() / hd))
    f = torch.outer(torch.arange(ctx).float(), inv)
    return f.cos(), f.sin()  # (ctx, hd/2)


def apply_rope(x, cos, sin):  # x: (B, H, T, hd)
    T = x.shape[2]
    x1, x2 = x[..., ::2], x[..., 1::2]
    c, s_ = cos[:T], sin[:T]
    return torch.stack((x1 * c - x2 * s_, x1 * s_ + x2 * c), -1).flatten(-2)


class Block(nn.Module):
    def __init__(s, c):
        super().__init__()
        s.h = c.heads
        s.n1, s.n2 = RMSNorm(c.dim), RMSNorm(c.dim)
        s.qkv, s.proj = nn.Linear(c.dim, 3 * c.dim, bias=False), nn.Linear(c.dim, c.dim, bias=False)
        hid = int(8 * c.dim / 3 / 64) * 64  # SwiGLU hidden size, ~same params as a 4x MLP
        s.gate, s.up, s.down = (nn.Linear(c.dim, hid, bias=False), nn.Linear(c.dim, hid, bias=False),
                                nn.Linear(hid, c.dim, bias=False))
        s.drop = nn.Dropout(c.dropout)

    def forward(s, x, cos, sin):
        B, T, C = x.shape
        q, k, v = s.qkv(s.n1(x)).view(B, T, 3, s.h, C // s.h).permute(2, 0, 3, 1, 4)
        a = F.scaled_dot_product_attention(apply_rope(q, cos, sin), apply_rope(k, cos, sin), v, is_causal=True)
        x = x + s.drop(s.proj(a.transpose(1, 2).reshape(B, T, C)))
        h = s.n2(x)
        return x + s.drop(s.down(F.silu(s.gate(h)) * s.up(h)))


class GPT(nn.Module):
    def __init__(s, c=Config()):
        super().__init__()
        s.c = c
        s.tok = nn.Embedding(c.vocab, c.dim)
        s.blocks = nn.ModuleList(Block(c) for _ in range(c.layers))
        s.norm, s.head = RMSNorm(c.dim), nn.Linear(c.dim, c.vocab, bias=False)
        s.head.weight = s.tok.weight
        cos, sin = rope_table(c.ctx, c.dim // c.heads)
        s.register_buffer("cos", cos, persistent=False)
        s.register_buffer("sin", sin, persistent=False)
        s.apply(s._init)
        for b in s.blocks:  # scale residual-output projections (GPT-2 style)
            nn.init.normal_(b.proj.weight, std=0.02 / (2 * c.layers) ** 0.5)
            nn.init.normal_(b.down.weight, std=0.02 / (2 * c.layers) ** 0.5)

    def _init(s, m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, std=0.02)

    def forward(s, idx, targets=None):
        x = s.tok(idx)
        for b in s.blocks:
            x = b(x, s.cos, s.sin)
        logits = s.head(s.norm(x))
        loss = F.cross_entropy(logits.float().view(-1, logits.size(-1)), targets.view(-1)) if targets is not None else None
        return logits, loss

    @torch.no_grad()
    def generate(s, idx, max_new=200, temp=0.7, top_k=40, top_p=0.95, stop=(EOT, END), rep_penalty=1.1):
        was = s.training
        s.eval()
        start = idx.shape[1]
        for _ in range(max_new):
            logits = s(idx[:, -s.c.ctx:])[0][:, -1].float()
            if rep_penalty != 1.0:  # discourage loops: penalize tokens generated so far
                seen = idx[0, start:].unique()
                logits[0, seen] = torch.where(logits[0, seen] > 0, logits[0, seen] / rep_penalty,
                                              logits[0, seen] * rep_penalty)
            logits = logits / max(temp, 1e-5)
            v, _ = torch.topk(logits, top_k)
            logits[logits < v[:, [-1]]] = -float("inf")
            probs = F.softmax(logits, -1)
            sp, si = probs.sort(descending=True)
            drop = sp.cumsum(-1) - sp > top_p
            sp[drop] = 0
            nxt = si.gather(-1, torch.multinomial(sp / sp.sum(-1, keepdim=True), 1))
            idx = torch.cat([idx, nxt], 1)
            if nxt.item() in stop:
                break
        s.train(was)
        return idx
