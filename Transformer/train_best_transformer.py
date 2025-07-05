import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
from TwoTokenTransformer import TwoTokenTransformer
from transformer_train_loop import train_loop

# ----------------------- CONFIG ------------------------
BATCH_SIZE = 16
d_model = 512
nhead = 2
depth = 4
lr = 1.9e-5
weight_decay = 9.4e-5
epochs = 20

# ----------------------- LOAD DATA ---------------------
print("🔄 Loading cleaned data...")
train_df = pd.read_csv('Transformer/train_set_cleaned.csv')
val_df = pd.read_csv('Transformer/val_set_cleaned.csv')

# Identify column types
kp_cols = [c for c in train_df.columns if c.startswith('kp_e')]
cnn_cols = [c for c in train_df.columns if c.startswith('cnn_e')]

# Extract tensors
X_train_kp = torch.tensor(train_df[kp_cols].astype(np.float32).values)
X_train_cnn = torch.tensor(train_df[cnn_cols].astype(np.float32).values)
y_train = torch.tensor(train_df['label_idx'].values, dtype=torch.long)

X_val_kp = torch.tensor(val_df[kp_cols].astype(np.float32).values)
X_val_cnn = torch.tensor(val_df[cnn_cols].astype(np.float32).values)
y_val = torch.tensor(val_df['label_idx'].values, dtype=torch.long)

# ----------------------- DATA LOADERS ---------------------
train_dataset = TensorDataset(X_train_kp, X_train_cnn, y_train)
val_dataset = TensorDataset(X_val_kp, X_val_cnn, y_val)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# ----------------------- MODEL INIT ---------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = TwoTokenTransformer(d_model=d_model, nhead=nhead, depth=depth,
                            dim_feedforward=None, dropout=0.5, num_classes=47).to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
criterion = nn.CrossEntropyLoss()

# ----------------------- TRAIN ---------------------
print("🚀 Starting training...")
train_loop(
    model,
    train_loader,     # matches 'trainloader'
    val_loader,       # matches 'testloader'
    optimizer,
    criterion,        # matches 'loss_fn'
    25,               # epochs
    47,               # num_classes (update if needed)
    verbose=True
)

print("✅ Training complete. Model saved to 'best_transformer.pth'")
