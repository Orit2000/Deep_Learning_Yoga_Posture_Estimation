import os
import pandas as pd
from tqdm import tqdm
from ultralytics import YOLO

# Load YOLOv8 pose model
model_yolo = YOLO("yolo11x-pose.pt")

# CSV files
csv_paths = {
    "train": r"C:\Users\safit\OneDrive\GitHub\Deep_Learning_Yoga_Posture_Estimation\Augmentations\train_set.csv",
    "val": r"C:\Users\safit\OneDrive\GitHub\Deep_Learning_Yoga_Posture_Estimation\Augmentations\val_set.csv"
}

# Output
all_augmented_keypoints = []

for split, csv_path in csv_paths.items():
    print(f"📌 Extracting YOLO keypoints for: {split}_set")
    df = pd.read_csv(csv_path)

    for _, row in tqdm(df.iterrows(), total=len(df)):
        image_path = row["image_path"]
        label_idx = row["label_idx"]
        label_str = row["label_str"]

        folder, filename = os.path.split(image_path)
        name_only, ext = os.path.splitext(filename)

        for i in range(1, 4):  # aug1 to aug3
            aug_filename = f"{name_only}_aug{i}{ext}"
            aug_path = os.path.join(folder, aug_filename)

            if not os.path.exists(aug_path):
                print(f"❌ Missing augmented image: {aug_path}")
                continue

            results = model_yolo.predict(aug_path, boxes=False, verbose=False)
            r = results[0]

            if not r.keypoints or r.keypoints.xyn is None:
                print(f"⚠️ No keypoints detected in: {aug_path}")
                continue

            keypoints = r.keypoints.xyn.cpu().numpy()[0]
            keypoints_flat = keypoints.reshape(-1).tolist()

            all_augmented_keypoints.append({
                "image_path": aug_path,
                "label_idx": label_idx,
                "label_str": label_str,
                **{f"kp_e{j}": val for j, val in enumerate(keypoints_flat)}
            })

# Save YOLO keypoints CSV
output_df = pd.DataFrame(all_augmented_keypoints)
output_df.to_csv("augmented_keypoints_only.csv", index=False)
print("✅ Saved to 'augmented_keypoints_only.csv'")

