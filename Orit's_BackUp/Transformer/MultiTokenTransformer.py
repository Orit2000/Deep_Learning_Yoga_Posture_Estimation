import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiTokenTransformer(nn.Module):
    def __init__(self,
                 kp_dim=34,
                 cnn_dim=128,
                 num_classes=47,
                 d_model=512,
                 nhead=4,
                 n_layers=1,
                 dim_ff=2*256,          # slimmer MLP
                 dropout=0.5):
        super().__init__()

        # --- token splits --------------------------------------------------
        self.kp_splits   = (10, 8, 8, 8)  #(9, 9, 8, 8)                 # 4 tokens
        self.cnn_splits  = (512,)#(64,)*8                      # 8 tokens
        self.n_tokens    = 1 + len(self.kp_splits) + len(self.cnn_splits)

        # linear projections: (token_dim → d_model)
        self.kp_proj  = nn.ModuleList(
            [nn.Linear(d, d_model) for d in self.kp_splits])
        self.cnn_proj = nn.ModuleList(
            [nn.Linear(d, d_model) for d in self.cnn_splits])

        self.cls = nn.Parameter(torch.randn(1, 1, d_model))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model      = d_model,
            nhead        = nhead,
            dim_feedforward = dim_ff,
            dropout      = dropout,
            batch_first  = True)
        self.encoder = nn.TransformerEncoder(encoder_layer, n_layers)

        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, num_classes))

    # ----------------------------------------------------------------------    
    def forward(self, kp, cnn):
        B = kp.size(0)

        # --- slice KP ----------------------------------------------------
        kp_tokens = []
        idx = 0
        for split, proj in zip(self.kp_splits, self.kp_proj):
            kp_tok = proj(kp[:, idx:idx+split]).unsqueeze(1)   # (B,1,d_model)
            kp_tokens.append(kp_tok)
            idx += split

        # --- slice CNN ---------------------------------------------------
        cnn_tokens = []
        idx = 0
        for split, proj in zip(self.cnn_splits, self.cnn_proj):
            cnn_tok = proj(cnn[:, idx:idx+split]).unsqueeze(1) # (B,1,d_model)
            cnn_tokens.append(cnn_tok)
            idx += split

        # ----- stack: [CLS | KP*4 | CNN*8] ---------------------------------
        cls_tok = self.cls.expand(B, -1, -1)
        x = torch.cat([cls_tok, *kp_tokens, *cnn_tokens], dim=1)  # (B, 13, d_model)

        # ----- token dropout (optional) ------------------------------------
        if self.training:
            mask = torch.rand_like(x[:,:,0]) < 0.1      # 10 % tokens
            x[mask] = 0.0

        # ----- Transformer -------------------------------------------------
        x = self.encoder(x)          # (B, 13, d_model)
        cls_out = x[:, 0]            # take CLS

        return self.head(cls_out)
