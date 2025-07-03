import pandas as pd
import numpy as np

# === CONFIGURATION ===
train_csv = "Transformer/train_set_augmented_final.csv"
val_csv = "Transformer/val_set_augmented_final.csv"
out_train_csv = "Transformer/train_set_cleaned.csv"
out_val_csv = "Transformer/val_set_cleaned.csv"
placeholder_value = -1.0

# === Load data ===
print("🔍 Loading data...")
train_df = pd.read_csv(train_csv)
val_df = pd.read_csv(val_csv)

# === Detect KP and CNN columns ===
kp_cols = [col for col in train_df.columns if col.startswith("kp_")]
cnn_cols = [col for col in train_df.columns if col.startswith("cnn_")]

print(f"📌 Found {len(kp_cols)} keypoint features, {len(cnn_cols)} CNN features")

# === Function to replace NaNs ===
def replace_nans(df, cols, value):
    nan_counts = df[cols].isna().sum().sum()
    print(f"⚠️ Found {nan_counts} NaNs in {len(cols)} columns")
    df[cols] = df[cols].fillna(value)
    return df

# === Replace NaNs in keypoints only ===
train_df = replace_nans(train_df, kp_cols, placeholder_value)
val_df = replace_nans(val_df, kp_cols, placeholder_value)

# === Confirm changes ===
print(f"✅ Any remaining NaNs in KP (train)? {train_df[kp_cols].isna().any().any()}")
print(f"✅ Any remaining NaNs in KP (val)? {val_df[kp_cols].isna().any().any()}")

# === Save cleaned files ===
train_df.to_csv(out_train_csv, index=False)
val_df.to_csv(out_val_csv, index=False)
print(f"💾 Cleaned train saved to: {out_train_csv}")
print(f"💾 Cleaned val saved to:   {out_val_csv}")
