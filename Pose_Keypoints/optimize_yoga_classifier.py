import optuna
import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from train_loop import train_loop
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner
from optuna.exceptions import TrialPruned
import json

# ----------------------------------------------
# Persistent HPO setup
# ----------------------------------------------
STUDY_NAME = "yoga_keypoints_hpo"
STORAGE_PATH = "sqlite:///optuna_yoga_keypoints.db"

study = optuna.create_study(
    study_name=STUDY_NAME,
    storage=STORAGE_PATH,
    direction="maximize",
    sampler=TPESampler(n_startup_trials=10),
    pruner=MedianPruner(n_warmup_steps=5),
    load_if_exists=True
)

# ----------------------------------------------
# Load and prepare dataset
# ----------------------------------------------
df = pd.read_csv("yolo_keypoints_dataset.csv").dropna()
# Use the numeric label directly

# Only keep rows where e0–e33 are all numeric
keypoint_columns = [f"e{i}" for i in range(34)]

# Try converting all keypoint columns to numeric, coerce errors to NaN
df[keypoint_columns] = df[keypoint_columns].apply(pd.to_numeric, errors='coerce')

# Drop any rows with NaN in the keypoint columns
df = df.dropna(subset=keypoint_columns)

X = df[keypoint_columns].values
y = df["label_idx"].values
num_classes = len(df["label_idx"].unique())
input_length = X.shape[1]

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1)

X_train = torch.tensor(X_train, dtype=torch.float32)
X_val   = torch.tensor(X_val,   dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)
y_val   = torch.tensor(y_val,   dtype=torch.long)

train_ds = TensorDataset(X_train, y_train)
val_ds   = TensorDataset(X_val, y_val)

train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
val_loader   = DataLoader(val_ds,   batch_size=32)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_classes = len(set(y))
input_length = X.shape[1]
best_trial_info = {"trial_number": None, "history": None, "best_epoch": None}

# ----------------------------------------------
# Define objective function for Optuna
# ----------------------------------------------
def objective(trial):
    hidden_dim = trial.suggest_categorical("hidden_dim", [32, 64, 128, 256])  # To change the value-space to [64, 96, 128, 160, 192, 256] new study is required
    num_layers = trial.suggest_categorical("num_layers", [2, 3]) # Let Optuna decide how many layers to take
    lr         = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64, 128])
    norm_type  = trial.suggest_categorical("norm_type", ["none", "batch", "layer"]) # Let Optuna decide which norm type to use
    opt_name   = trial.suggest_categorical("optimizer", ["Adam", "RMSprop", "SGD"]) # Let Optuna decide which optimizer type to use
    dropout    = 0.271   # Instead of - trial.suggest_float("dropout", 0.1, 0.5) we fixed the value

    # hidden_dim = 256
    # num_layers = 2
    # lr         = 0.00197
    # batch_size = 32
    # norm_type  = "layer"
    # opt_name   = "Adam"
    # dropout    = 0.271  
    
    # Dataloaders with new batch size
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size)

    def make_norm(dim):
        if norm_type == "batch":
            return nn.BatchNorm1d(dim)
        elif norm_type == "layer":
            return nn.LayerNorm(dim)
        else:
            return nn.Identity()
    
    class OptimizedYogaClassifier(nn.Module):
        def __init__(self, input_length, num_classes, hidden_dim, dropout, num_layers, norm_type):
            super().__init__()
            self.layer1 = nn.Linear(input_length, hidden_dim)
            self.norm1 = make_norm(hidden_dim)
            self.act = nn.ReLU()
            self.dropout = nn.Dropout(dropout)
            self.layer2 = nn.Linear(hidden_dim, hidden_dim)
            self.norm2 = make_norm(hidden_dim)
            if num_layers == 3:
                self.layer3 = nn.Linear(hidden_dim, hidden_dim)
            else:
                self.layer3 = None

            self.out = nn.Linear(hidden_dim, num_classes)

        def forward(self, x):
            x = self.act(self.layer1(x))
            x = self.act(self.norm1(x))
            x = self.dropout(x)
            x = self.act(self.layer2(x))
            x = self.act(self.norm2(x))
            if self.layer3 is not None:
                x = self.act(self.layer3(x))
            x = self.out(x)
            return x

    model = OptimizedYogaClassifier(input_length, num_classes, hidden_dim, dropout, num_layers, norm_type).to(device)

    # Optimizer selection
    if opt_name == "Adam":
         optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    elif opt_name == "RMSprop":
        optimizer = torch.optim.RMSprop(model.parameters(), lr=lr, momentum=0.9)
    else:  # SGD
        momentum = trial.suggest_float("momentum", 0.0, 0.95)
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum)
        
    loss_fn = nn.CrossEntropyLoss()

    history, best_epoch = train_loop(
        model=model,
        trainloader=train_loader,
        testloader=val_loader,
        optimizer=optimizer,
        loss_fn=loss_fn,
        epochs=60,  # tuning epochs
        num_classes=num_classes,
        verbose=False,
        trial=trial,
        patience =5
    )
    print(f"Train Loss: {history['train_loss'][best_epoch]:.4f}, "
      f"Val Loss: {history['test_loss'][best_epoch]:.4f}, "
      f"Train Acc: {history['train_accuracy'][best_epoch]:.4f}, "
      f"Val Acc: {history['test_accuracy'][best_epoch]:.4f}")
    
    if trial.number == study.best_trial.number:
        best_trial_info["trial_number"] = trial.number
        best_trial_info["history"] = history
        best_trial_info["best_epoch"] = best_epoch

    return history["test_accuracy"][best_epoch]  # maximize this
# ----------------------------------------------
# Run Optuna HPO
# ----------------------------------------------
study.optimize(objective, n_trials=30)

# ----------------------------------------------
# Output best trial and export to CSV
# ----------------------------------------------
print("✅ Best trial:")
print(f"  Value (Best Accuracy): {study.best_trial.value:.4f}")
print(f"  Params: {study.best_trial.params}")

# Export all trials to CSV
df_trials = study.trials_dataframe()
df_trials.to_csv("optuna_yoga_keypoints.csv", index=False)
print("📄 Trials saved to 'optuna_yoga_keypoints.csv'")

with open("best_trial_history.json", "w") as f:
    json.dump(best_trial_info, f)