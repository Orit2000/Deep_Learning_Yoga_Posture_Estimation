import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np  # only if you process joint indices for kp_mask

class CrossAttnBlock(nn.Module):
    def __init__(self, d, nhead=4, ff_mult=4, p=0.3):
        super().__init__()
        self.qkv_attn = nn.MultiheadAttention(d, nhead, dropout=p, batch_first=True)
        self.ln1 = nn.LayerNorm(d)
        self.ffn = nn.Sequential(nn.Linear(d, ff_mult*d), nn.GELU(), nn.Dropout(p),
                                 nn.Linear(ff_mult*d, d))
        self.ln2 = nn.LayerNorm(d)
        self.drop = nn.Dropout(p)

    # q: [B, Tq, d], kv: [B, Tk, d]
    def forward(self, q, kv, kv_mask=None):
        qn = self.ln1(q); kvn = self.ln1(kv)
        z, _ = self.qkv_attn(qn, kvn, kvn, key_padding_mask=kv_mask)  # cross-attn
        q = q + self.drop(z)
        q = q + self.drop(self.ffn(self.ln2(q)))
        return q
