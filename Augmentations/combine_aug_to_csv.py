import os
import pandas as pd

# === Load original CSVs ===
train_df = pd.read_csv("Augmentations/train_set.csv")
val_df = pd.read_csv("Augmentations/val_set.csv")

# === Build mapping: (label_str, filename) → "train"/"val"
stem_to_split = {}

def update_mapping(df, split_name):
    for _, row in df.iterrows():
        label = str(row["label_str"]).lower()
        filename = os.path.basename(row["image_path"]).lower()
        key = (label, filename)
        if key in stem_to_split:
            print(f"⚠️ Duplicate key in both sets: {key}")
        stem_to_split[key] = split_name

update_mapping(train_df, "train")
update_mapping(val_df, "val")

# === Load augmented feature data ===
aug_df = pd.read_csv("augmented_keypoints_with_cnn.csv")

augmented_train_rows = []
augmented_val_rows = []

# === Match each augmented image ===
for idx, row in aug_df.iterrows():
    aug_path = row["image_path"]
    aug_filename = os.path.basename(aug_path)
    label = str(row["label_str"]).lower()

    # Get base image name (e.g., File5 from File5_aug1.png)
    base_filename = aug_filename.split("_aug")[0] + os.path.splitext(aug_filename)[1]
    base_filename = base_filename.lower()

    key = (label, base_filename)

    if key not in stem_to_split:
        print(f"⚠️ No match for: {aug_filename} → key: {key}")
        continue

    if stem_to_split[key] == "train":
        augmented_train_rows.append(row)
    elif stem_to_split[key] == "val":
        augmented_val_rows.append(row)

# === Save final CSVs ===
final_train_df = pd.concat([train_df, pd.DataFrame(augmented_train_rows)], ignore_index=True)
final_val_df = pd.concat([val_df, pd.DataFrame(augmented_val_rows)], ignore_index=True)

print(f"✅ Matched {len(augmented_train_rows)} augmentations to TRAIN set")
print(f"✅ Matched {len(augmented_val_rows)} augmentations to VAL set")

final_train_df.to_csv("train_set_augmented_final.csv", index=False)
final_val_df.to_csv("val_set_augmented_final.csv", index=False)

print("✅ Saved both final CSV files with correct matching by (class + filename).")
