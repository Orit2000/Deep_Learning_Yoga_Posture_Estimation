import pandas as pd
import os

# === Paths to your combined CSVs ===
input_files = {
    "train": "Pose_keypoints/train_set.csv",
    "val": "Pose_keypoints/val_set.csv",
    "test": "Pose_keypoints/test_set.csv"  # change if needed
}

# === Output folder and file naming ===
output_folder = "Keypoints_Only_CSVs"
os.makedirs(output_folder, exist_ok=True)

# === Keypoint and label column names ===
keypoint_cols = [f"kp_e{i}" for i in range(34)]
meta_cols = ["image_path", "label_idx", "label_str"]

# === Process each split ===
for split, file_path in input_files.items():
    print(f"Processing {split} set...")
    df = pd.read_csv(file_path)

    # Extract keypoints and metadata only
    df_keypoints = df[keypoint_cols + meta_cols]

    # Save to new CSV
    out_path = os.path.join(output_folder, f"{split}_keypoints_only.csv")
    df_keypoints.to_csv(out_path, index=False)
    print(f"Saved to: {out_path}")
