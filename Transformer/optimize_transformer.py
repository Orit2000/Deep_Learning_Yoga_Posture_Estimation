import optuna
import json
import torch
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter
from TwoTokenTransformer import TwoTokenTransformer
from transformer_train_loop import train_loop
import matplotlib.pyplot as plt

# Load combined dataset from Parquet instead of CSV
#FULL_DF = pd.read_parquet("combo_features.parquet")
df = pd.read_parquet("Transformer/combo_features.parquet")
df.to_csv("Transformer/combo_features.csv", index=False)
FULL_DF = pd.read_csv("Transformer/combo_features.csv")

KP_COLS = [c for c in FULL_DF.columns if c.startswith("kp_")]
CNN_COLS = [c for c in FULL_DF.columns if c.startswith("cnn_")]

# Load train/val/test splits
train_raw = pd.read_csv("Transformer/train_set_cleaned.csv")
val_raw = pd.read_csv("Transformer/val_set_cleaned.csv")
test_raw = pd.read_csv("Transformer/test_set.csv")

# === Convert to numeric to prevent dtype errors ===
train_raw[KP_COLS] = train_raw[KP_COLS].apply(pd.to_numeric, errors="coerce")
val_raw[KP_COLS] = val_raw[KP_COLS].apply(pd.to_numeric, errors="coerce")
train_raw[CNN_COLS] = train_raw[CNN_COLS].apply(pd.to_numeric, errors="coerce")
val_raw[CNN_COLS] = val_raw[CNN_COLS].apply(pd.to_numeric, errors="coerce")


def plot_class_distribution(df, set_name):
    counts = df["label_str"].value_counts().sort_index()
    plt.figure(figsize=(12, 4))
    counts.plot(kind="bar")
    plt.title(f"{set_name} Set Class Distribution (Augmented)")
    plt.xlabel("Class Label")
    plt.ylabel("Sample Count")
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.grid(axis='y', linestyle='--', linewidth=0.5)
    plt.show()

plot_class_distribution(train_raw, "Train")
plot_class_distribution(val_raw, "Validation")

# Compute normalization statistics from train set
kp_mu = torch.tensor(train_raw[KP_COLS].mean().values, dtype=torch.float32)
kp_std = torch.tensor(train_raw[KP_COLS].std().values + 1e-8, dtype=torch.float32)
cnn_mu = torch.tensor(train_raw[CNN_COLS].mean().values, dtype=torch.float32)
cnn_std = torch.tensor(train_raw[CNN_COLS].std().values + 1e-8, dtype=torch.float32)

# Helper: Create normalized tensor dataset
def make_tensor_ds(df):
    kp = torch.tensor(df[KP_COLS].values, dtype=torch.float32)
    cnn = torch.tensor(df[CNN_COLS].values, dtype=torch.float32)
    y = torch.tensor(df["label_idx"].values, dtype=torch.long)
    kp = (kp - kp_mu) / kp_std
    cnn = (cnn - cnn_mu) / cnn_std
    return TensorDataset(kp, cnn, y)

train_ds = make_tensor_ds(train_raw)
val_ds = make_tensor_ds(val_raw)

def objective(trial):
    d_model = trial.suggest_categorical("d_model", [128, 256, 512])
    nhead = trial.suggest_categorical("nhead", [2, 4, 8])
    depth = trial.suggest_int("depth", 2, 6)
    lr = trial.suggest_float("lr", 1e-5, 1e-3, log=True)
    wd = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)

    model = TwoTokenTransformer(
        kp_dim=len(KP_COLS),
        cnn_dim=len(CNN_COLS),
        d_model=d_model,
        nhead=nhead,
        depth=depth,
        num_classes=47
    )
    model.to(device := ("cuda" if torch.cuda.is_available() else "cpu"))

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    loss_fn = torch.nn.CrossEntropyLoss()

    writer = SummaryWriter(log_dir=f"runs/optuna_trial_{trial.number}")
    history, best_epoch = train_loop(
        model, train_loader, val_loader, opt, loss_fn, epochs=25,
        num_classes=47, verbose=False
    )

    for epoch in range(len(history["train_loss"])):
        writer.add_scalar("Loss/Train", history["train_loss"][epoch], epoch)
        writer.add_scalar("Accuracy/Train", history["train_accuracy"][epoch], epoch)
        writer.add_scalar("F1/Train", history["train_f1"][epoch], epoch)
        writer.add_scalar("Loss/Val", history["test_loss"][epoch], epoch)
        writer.add_scalar("Accuracy/Val", history["test_accuracy"][epoch], epoch)
        writer.add_scalar("F1/Val", history["test_f1"][epoch], epoch)

    writer.close()

    best_val_acc = max(history["test_accuracy"])
    trial.set_user_attr("history", history)
    trial.set_user_attr("best_epoch", best_epoch)
    return best_val_acc

if __name__ == "__main__":
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=20)

    best_trial = study.best_trial
    with open("best_trial_history.json", "w") as f:
        json.dump(best_trial.user_attrs["history"], f, indent=2)
    with open("best_hyperparameters.json", "w") as f:
        json.dump(best_trial.params, f, indent=2)

    print("Best hyperparameters:")
    print(best_trial.params)
    print("Best validation accuracy:", best_trial.value)
