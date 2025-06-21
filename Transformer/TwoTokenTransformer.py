import torch.nn as nn
import torch

class TwoTokenTransformer(nn.Module):
    def __init__(self,
                 kp_dim=34,
                 cnn_dim=512,
                 d_model=512,
                 nhead=4,
                 depth=4,
                 num_classes= 47 ):
        super().__init__()

        # project each modality into the common d_model space
        self.kp_proj  = nn.Linear(kp_dim,  d_model)
        self.cnn_proj = nn.Linear(cnn_dim, d_model)

        # learnable positional / segment embeddings for the 2 tokens
        self.pos_embed = nn.Parameter(torch.zeros(2, d_model))

        encoder_layer = nn.TransformerEncoderLayer(
                            d_model=d_model,
                            nhead=nhead,
                            dim_feedforward=4*d_model,
                            dropout=0.1,
                            batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)

        # classification head on token-0 (can also pool)
        self.head = nn.Sequential(
                        nn.LayerNorm(d_model),
                        nn.Linear(d_model, num_classes))

    def forward(self, kp_vec, cnn_vec):
        # kp_vec, cnn_vec  shape: [B, dim]
        tok0 = self.cnn_proj(cnn_vec)
        tok1 = self.kp_proj(kp_vec)
        x = torch.stack([tok0, tok1], dim=1)        # [B, 2, d_model]
        x = x + self.pos_embed                      # add (learned) 2×d_model
        h = self.encoder(x)                         # transformer magic
        cls = h[:, 0]                               # take first token
        return self.head(cls)
