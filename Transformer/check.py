import pandas as pd
import torch

# === Load cleaned CSVs ===
print("🔍 Loading CSVs...")
train_df = pd.read_csv("Transformer/train_set_cleaned.csv")
val_df = pd.read_csv("Transformer/val_set_cleaned.csv")

# === Detect KP & CNN columns ===
kp_cols = [c for c in train_df.columns if c.startswith("kp_")]
cnn_cols = [c for c in train_df.columns if c.startswith("cnn_")]

# === Force numeric conversion (fixes object dtype errors) ===
train_df[kp_cols] = train_df[kp_cols].apply(pd.to_numeric, errors="coerce")
val_df[kp_cols] = val_df[kp_cols].apply(pd.to_numeric, errors="coerce")
train_df[cnn_cols] = train_df[cnn_cols].apply(pd.to_numeric, errors="coerce")
val_df[cnn_cols] = val_df[cnn_cols].apply(pd.to_numeric, errors="coerce")

# === Class count check ===
print("\n📊 Label checks:")
print(f"Train classes: {train_df['label_idx'].nunique()}")
print(f"Train label range: {train_df['label_idx'].min()} - {train_df['label_idx'].max()}")
print(f"Val classes: {val_df['label_idx'].nunique()}")
print(f"Val label range: {val_df['label_idx'].min()} - {val_df['label_idx'].max()}")

# === Feature count check ===
print(f"\n📈 Found {len(kp_cols)} keypoint features, {len(cnn_cols)} CNN features")

# === NaN check ===
print("\n❓ NaN check:")
print("Any NaNs in KP features?", train_df[kp_cols].isnull().values.any() or val_df[kp_cols].isnull().values.any())
print("Any NaNs in CNN features?", train_df[cnn_cols].isnull().values.any() or val_df[cnn_cols].isnull().values.any())

# === Standard deviation check ===
kp_std = train_df[kp_cols].std().mean()
cnn_std = train_df[cnn_cols].std().mean()
print(f"\n📏 Feature std check:")
print(f"KP std mean: {kp_std} min: {train_df[kp_cols].std().min()}")
print(f"CNN std mean: {cnn_std} min: {train_df[cnn_cols].std().min()}")

# === Sample feature printout ===
print("\n🔬 Sample normalized features:")
i = 0
sample_kp = torch.tensor(train_df.loc[i, kp_cols].values.astype(float), dtype=torch.float32)
sample_cnn = torch.tensor(train_df.loc[i, cnn_cols].values.astype(float), dtype=torch.float32)

# Simulate normalization
kp_mu = torch.tensor(train_df[kp_cols].mean().values, dtype=torch.float32)
kp_std = torch.tensor(train_df[kp_cols].std().values + 1e-8, dtype=torch.float32)
cnn_mu = torch.tensor(train_df[cnn_cols].mean().values, dtype=torch.float32)
cnn_std = torch.tensor(train_df[cnn_cols].std().values + 1e-8, dtype=torch.float32)

norm_kp = (sample_kp - kp_mu) / kp_std
norm_cnn = (sample_cnn - cnn_mu) / cnn_std

print("Normalized KP sample (first 5):", norm_kp[:5])
print("Normalized CNN sample (first 5):", norm_cnn[:5])
