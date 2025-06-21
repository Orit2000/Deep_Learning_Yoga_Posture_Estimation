import torch, torch.nn.functional as F
from torch.utils.data import DataLoader
from TwoTokenTransformer import TwoTokenTransformer
import pandas as pd
import sys
import os
from transformer_train_loop import train_loop
from torch.utils.data import TensorDataset, DataLoader


#FULL_DF   = pd.read_parquet("combo_features.parquet")      # read once
FULL_DF   = pd.read_csv("combo_features.csv")  
KP_COLS   = [c for c in FULL_DF.columns if c.startswith("kp_")]
CNN_COLS  = [c for c in FULL_DF.columns if c.startswith("cnn_")]
print(FULL_DF[KP_COLS].dtypes)      # are they “object” (string) instead of float?
print(FULL_DF[CNN_COLS].dtypes)
#print(FULL_DF.loc[0, KP_COLS[:]])


# print("KP_COLS len :", len(KP_COLS))
# print("CNN_COLS len:", len(CNN_COLS))

# sets = YogaPairDataset("combo_features.csv", KP_COLS, CNN_COLS)
# train_set = sets.train_df
# val_set = sets.val_df
# train_set.to_csv("train_set.csv", index=False)
# val_set.to_csv("val_set.csv", index=False)
# print(train_set)
# val_set   = YogaPairDataset("combo_features.csv", KP_COLS, CNN_COLS, train=False,
#                             kp_mu=train_set.kp_mu, kp_std=train_set.kp_std,
#                             cnn_mu=train_set.cnn_mu, cnn_std=train_set.cnn_std)

def make_tensor_ds(csv_path):
    df   = pd.read_csv(csv_path)

    kp   = torch.tensor(df[KP_COLS ].values, dtype=torch.float32)
    cnn  = torch.tensor(df[CNN_COLS].values, dtype=torch.float32)
    y    = torch.tensor(df["label_idx"].values, dtype=torch.long)

    # z-score with the *training* statistics
    kp   = (kp  - kp_mu ) / kp_std
    cnn  = (cnn - cnn_mu) / cnn_std
    return TensorDataset(kp, cnn, y)

train_raw = pd.read_csv("train_set.csv")
test_raw = pd.read_csv("test_set.csv")
val_raw = pd.read_csv("val_set.csv")

kp_mu  = torch.tensor(train_raw[KP_COLS ].mean().values, dtype=torch.float32)
kp_std = torch.tensor(train_raw[KP_COLS ].std ().values + 1e-8, dtype=torch.float32)
cnn_mu = torch.tensor(train_raw[CNN_COLS].mean().values, dtype=torch.float32)
cnn_std= torch.tensor(train_raw[CNN_COLS].std ().values + 1e-8, dtype=torch.float32)

train_dl = DataLoader(make_tensor_ds("train_set.csv"), batch_size=32, shuffle=True)
val_dl   = DataLoader(make_tensor_ds("val_set.csv"),   batch_size=32)
test_dl   = DataLoader(make_tensor_ds("test_set.csv"),   batch_size=32)

model = TwoTokenTransformer(num_classes=47)
model.to(device := ("cuda" if torch.cuda.is_available() else "cpu"))

opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", patience=3)
loss_fn = torch.nn.CrossEntropyLoss()

history, best_epoch = train_loop(model, train_dl, val_dl, opt, loss_fn, 25, 47,verbose=True)

# for epoch in range(25):
#     model.train()
#     for kp, cnn, y in train_dl:
#         kp, cnn, y = kp.to(device), cnn.to(device), y.to(device)
#         logits = model(kp, cnn)
#         loss   = F.cross_entropy(logits, y)

#         opt.zero_grad(); loss.backward(); opt.step()

#     # --- quick val pass -------------------------------------------
#     model.eval(); correct = total = 0
#     with torch.inference_mode():
#         for kp, cnn, y in val_dl:
#             kp, cnn, y = kp.to(device), cnn.to(device), y.to(device)
#             pred = model(kp, cnn).argmax(1)
#             correct += (pred == y).sum().item()
#             total   += y.size(0)
#     acc = correct/total
#     sched.step(acc)
#     print(f"epoch {epoch:02d} | val acc {acc:.3f}")