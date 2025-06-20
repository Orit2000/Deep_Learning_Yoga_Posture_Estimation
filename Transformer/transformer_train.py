import torch, torch.nn.functional as F
from torch.utils.data import DataLoader
from TwoTokenTransformer import TwoTokenTransformer
from YogaPairDataset import YogaPairDataset
import pandas as pd
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Pose_Keypoints')))
from train_loop import train_loop

FULL_DF   = pd.read_parquet("combo_features.parquet")      # read once
KP_COLS   = [c for c in FULL_DF.columns if c.startswith("kp_")]
CNN_COLS  = [c for c in FULL_DF.columns if c.startswith("cnn_")]

print("KP_COLS len :", len(KP_COLS))
print("CNN_COLS len:", len(CNN_COLS))

train_set = YogaPairDataset("combo_features.parquet", KP_COLS, CNN_COLS, train=True)
val_set   = YogaPairDataset("combo_features.parquet", KP_COLS, CNN_COLS, train=False,
                            kp_mu=train_set.kp_mu, kp_std=train_set.kp_std,
                            cnn_mu=train_set.cnn_mu, cnn_std=train_set.cnn_std)

# train_dl = DataLoader(train_set, batch_size=32, shuffle=True)
# val_dl   = DataLoader(val_set,   batch_size=32)

# model = TwoTokenTransformer(num_classes=len(train_set.df["label_idx"].unique()))
# model.to(device := ("cuda" if torch.cuda.is_available() else "cpu"))

# opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
# sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", patience=3)
# loss_fn = torch.nn.CrossEntropyLoss()

# train_loop(model, train_dl, val_dl, opt, loss_fn, 25, 47,verbose=True)

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