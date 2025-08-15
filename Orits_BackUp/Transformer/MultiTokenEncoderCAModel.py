import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np  # only if you process joint indices for kp_mask
from CrossAttnBlock import CrossAttnBlock

class MultiTokenEncoderCAModel(nn.Module):
    def __init__(self, kp_dim=51, cnn_dim=512, num_classes=47,
                 d_model=256, nhead=4, n_layers=1, dim_ff=512, dropout=0.25):
        super().__init__()
        self.kp_splits   = (3,)*17  # sum to 34
        self.cnn_splits  = (512,)     # one CNN token
        self.n_tokens    = 1 + len(self.kp_splits) + len(self.cnn_splits)  # +1 for CLS

        # projections
        self.kp_proj  = nn.ModuleList([nn.Linear(d, d_model) for d in self.kp_splits])
        self.cnn_proj = nn.ModuleList([nn.Linear(d, d_model) for d in self.cnn_splits])

        # embeddings
        self.cls = nn.Parameter(torch.randn(1, 1, d_model))
        self.pos = nn.Parameter(torch.randn(1, self.n_tokens, d_model))  # learned positional enc
        self.type_kp  = nn.Parameter(torch.randn(1, 1, d_model))  # add to KP tokens
        self.type_cnn = nn.Parameter(torch.randn(1, 1, d_model))  # add to CNN token
        self.type_cls = nn.Parameter(torch.randn(1, 1, d_model))  # add to CLS

        # cross-attn: CNN token queries the KP tokens
        self.cx_cnn_from_kp = CrossAttnBlock(d_model, nhead=nhead, ff_mult=2, p=dropout)

        # optional self-attn encoder on the joint sequence
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_ff,
            dropout=dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(enc_layer, n_layers)

        self.head = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, num_classes))

    def forward(self, kp, cnn, kp_mask=None):
        B = kp.size(0)

        # ---- KP tokens
        kp_tokens, idx = [], 0
        for split, proj in zip(self.kp_splits, self.kp_proj):
            tok = proj(kp[:, idx:idx+split]).unsqueeze(1)   # [B,1,d]
            kp_tokens.append(tok)
            idx += split
        kp_seq = torch.cat(kp_tokens, dim=1)                # [B, J, d]
        kp_seq = kp_seq + self.type_kp                      # type embedding

                # ---- CNN token
        cnn_tokens, idx = [], 0
        for split, proj in zip(self.cnn_splits, self.cnn_proj):
            tok = proj(cnn[:, idx:idx+split]).unsqueeze(1)  # [B,1,d]
            cnn_tokens.append(tok); idx += split
        cnn_seq = torch.cat(cnn_tokens, dim=1)              # [B,8,d]
        cnn_seq = cnn_seq + self.type_cnn

        # pool to a single query token for cross-attn
        cnn_tok = cnn_seq.mean(dim=1, keepdim=True)         # [B,1,d]
        cnn_fused = self.cx_cnn_from_kp(cnn_tok, kp_seq, kv_mask=kp_mask)

        # ---- Build final sequence: [CLS | KP*J | CNNfused]
        cls_tok = self.cls.expand(B, 1, -1) + self.type_cls
        x = torch.cat([cls_tok, kp_seq, cnn_fused], dim=1)  # [B, 1+J+1, d]

        # positional enc
        x = x + self.pos[:, :x.size(1), :]

        # (optional) token dropout - FIXED indexing
        # if self.training:
        #     with torch.no_grad():
        #         keep = (torch.rand(B, x.size(1), device=x.device) >= 0.1)  # keep 90%
        #     x = x * keep.unsqueeze(-1).float()

        # ---- Encoder and classification from CLS
        x = self.encoder(x)                # [B, T, d]
        cls_out = x[:, 0]                 # [CLS]
        return self.head(cls_out)         # logits [B, C]
