from .Contrastive_Learning.contrustive_learning_classes import ContrastiveModel,ContrastiveMLP
import torch
import torch.nn.functional as F
import torch, torch.nn.functional as F
from torch.utils.data import DataLoader
from MultiTokenTransformer import MultiTokenTransformer
import pandas as pd
import sys
import os
from transformer_train_loop import train_loop
from torch.utils.data import TensorDataset, DataLoader

def nt_xent_loss(z1, z2, temperature=0.5):
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)

    batch_size = z1.size(0)
    representations = torch.cat([z1, z2], dim=0)  # [2B, D]

    sim = F.cosine_similarity(representations.unsqueeze(1), representations.unsqueeze(0), dim=2)
    labels = torch.arange(batch_size).to(z1.device)
    labels = torch.cat([labels, labels], dim=0)

    mask = torch.eye(2 * batch_size, dtype=torch.bool).to(z1.device)
    sim = sim.masked_fill(mask, -1e9)
    sim = sim / temperature

    positives = torch.cat([torch.diag(sim, batch_size), torch.diag(sim, -batch_size)], dim=0)
    labels = torch.cat([torch.arange(batch_size) + batch_size, torch.arange(batch_size)], dim=0).to(z1.device)

    loss = F.cross_entropy(sim, labels)
    return loss

train_raw = pd.read_csv("train_set.csv")
test_raw = pd.read_csv("test_set.csv")
val_raw = pd.read_csv("val_set.csv")
KP_COLS   = [c for c in train_raw.columns if c.startswith("kp_")]
CNN_COLS  = [c for c in train_raw.columns if c.startswith("cnn_")]
kp_mu  = torch.tensor(train_raw[KP_COLS ].mean().values, dtype=torch.float32)
kp_std = torch.tensor(train_raw[KP_COLS ].std ().values + 1e-8, dtype=torch.float32)
cnn_mu = torch.tensor(train_raw[CNN_COLS].mean().values, dtype=torch.float32)
cnn_std= torch.tensor(train_raw[CNN_COLS].std ().values + 1e-8, dtype=torch.float32)

train_dl = DataLoader(make_tensor_ds("train_set.csv", kp_mu, kp_std, cnn_mu, cnn_std), batch_size=64, shuffle=True)
val_dl   = DataLoader(make_tensor_ds("val_set.csv", kp_mu, kp_std, cnn_mu, cnn_std),   batch_size=64)
test_dl  = DataLoader(make_tensor_ds("test_set.csv", kp_mu, kp_std, cnn_mu, cnn_std),   batch_size=64)

model = ContrastiveModel(resnet_dim=512, keypoint_dim=68, embed_dim=128).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)

for epoch in range(num_epochs):
    for images, kp_xy in dataloader:
        images, kp_xy = images.to(device), kp_xy.to(device)

        # 1) get frozen‐backbone embeddings
        with torch.no_grad():
            feats_img = ContrastiveMLP(images)    # [B, 512, 1, 1] → flatten → [B,512]
            feats_kp  = ContrastiveMLP(kp_xy)  # [B, keypoint_dim]

        # 2) project into shared space
        z_img, z_kp = model(feats_img, feats_kp)

        # 3) loss + update
        loss = nt_xent_loss(z_img, z_kp, temperature=0.5)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
