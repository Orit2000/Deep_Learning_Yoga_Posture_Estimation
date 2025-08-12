import torch, torch.nn.functional as F
from torch.utils.data import DataLoader
from TwoTokenTransformer import TwoTokenTransformer
import pandas as pd
import sys
import os
from transformer_train_loop import train_loop, test_step
from torch.utils.data import TensorDataset, DataLoader
from torchmetrics import Accuracy, F1Score, ConfusionMatrix


def make_tensor_ds(csv_path, kp_mu, kp_std, cnn_mu, cnn_std):
    df   = pd.read_csv(csv_path)

    kp   = torch.tensor(df[KP_COLS ].values, dtype=torch.float32)
    cnn  = torch.tensor(df[CNN_COLS].values, dtype=torch.float32)
    y    = torch.tensor(df["label_idx"].values, dtype=torch.long)

    # z-score with the *training* statistics
    kp   = (kp  - kp_mu ) / kp_std
    cnn  = (cnn - cnn_mu) / cnn_std
    return TensorDataset(kp, cnn, y)
def count_parameters(model, verbose=True):
    """
    Counts and optionally prints the number of parameters in the model.

    Args:
        model (torch.nn.Module): The model to analyze.
        verbose (bool): If True, prints param count per submodule.

    Returns:
        num_total_params (int): Total number of parameters.
        num_trainable_params (int): Number of trainable parameters (requires_grad=True).
    """
    num_total_params = sum(p.numel() for p in model.parameters())
    num_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    if verbose:
        print(f"Total parameters: {num_total_params:,}")
        print(f"Trainable parameters: {num_trainable_params:,}")
        print(f"Percentage trainable: {100 * num_trainable_params / num_total_params:.2f}%")
        print("\nBreakdown by module:")
        for name, param in model.named_parameters():
            print(f"{name:30}: {param.numel():>10} | {'trainable' if param.requires_grad else 'frozen'}")

    return num_total_params, num_trainable_params

train_raw = pd.read_csv("train_set_updated.csv")
test_raw= pd.read_csv("val_set_updated.csv")
val_raw = pd.read_csv("test_set_updated.csv")
KP_COLS   = [c for c in train_raw.columns if c.startswith("kp_")]
CNN_COLS  = [c for c in train_raw.columns if c.startswith("cnn_")]
kp_mu  = torch.tensor(train_raw[KP_COLS ].mean().values, dtype=torch.float32)
kp_std = torch.tensor(train_raw[KP_COLS ].std ().values + 1e-8, dtype=torch.float32)
cnn_mu = torch.tensor(train_raw[CNN_COLS].mean().values, dtype=torch.float32)
cnn_std= torch.tensor(train_raw[CNN_COLS].std ().values + 1e-8, dtype=torch.float32)

#train_dl = DataLoader(make_tensor_ds("train_set_updated.csv", kp_mu, kp_std, cnn_mu, cnn_std), batch_size=64, shuffle=True)
test_dl   = DataLoader(make_tensor_ds("val_set_updated.csv", kp_mu, kp_std, cnn_mu, cnn_std),   batch_size=64)
#val_dl  = DataLoader(make_tensor_ds("test_set_updated.csv", kp_mu, kp_std, cnn_mu, cnn_std),   batch_size=64)

model = TwoTokenTransformer(kp_dim=34,
                 cnn_dim=512,
                 d_model=256,
                 nhead=1,
                 depth=1,
                 num_classes= 47)
model.load_state_dict(torch.load("best.pth"))

model.eval()
print(f"The model is defined!")
model.to(device := ("cuda" if torch.cuda.is_available() else "cpu"))

opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", patience=5)
loss_fn = torch.nn.CrossEntropyLoss()
accuracy_score = Accuracy(task="multiclass", num_classes=47).to(device)
f1_score       = F1Score (task="multiclass", num_classes=47).to(device)
    
print(f"We are entering train loop...")

(test_loss, test_acc, test_f1)= test_step(model, test_dl, loss_fn, device, accuracy_score, f1_score)
print(test_loss, test_acc, test_f1)
print(count_parameters(model))
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