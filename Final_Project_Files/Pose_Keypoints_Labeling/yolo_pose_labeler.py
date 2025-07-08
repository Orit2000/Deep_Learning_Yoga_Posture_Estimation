"""
yolo_pose_labeler.py

This script processes a dataset of yoga posture images, detects keypoints using a YOLOv8 pose model,
and generates a labeled dataset with keypoints coordinates and metadata. It supports optional saving
of annotated images and keypoints JSON files for each image.

Main output: A CSV file `yolo_keypoints_dataset.csv` containing keypoints, image paths, and class labels.

Dependencies:
- ultralytics (YOLOv8)
- OpenCV (cv2)
- pandas
- scikit-learn
- json
- os
"""

from ultralytics import YOLO
import os
import pandas as pd 
import cv2
import json
from sklearn.preprocessing import LabelEncoder

def data_label(dataset_folder, saving_flag=False):
    """
    Process a dataset of images, detect pose keypoints using YOLOv8, and return labeled keypoint data.

    Args:
        dataset_folder (str): Path to the root dataset folder. The folder should contain subfolders 
                              named by class labels, each with images of that class.
        saving_flag (bool): If True, saves annotated images and keypoint JSON files to disk.

    Returns:
        list of list: Each inner list contains:
            - 34 float values: keypoint (x,y) normalized coordinates flattened as [x0,y0,x1,y1,...]
                               or [0.0]*34 if no keypoints detected.
            - str: image path
            - str: class label
            - int: class index (incremental integer assigned during processing)
    """
    valid_ext = (".jpg", ".jpeg", ".png", ".bmp")
    model     = YOLO("yolo11x-pose.pt")
    rows      = []
    counter   = 0

    for label in sorted(os.listdir(dataset_folder)):
        if label.lower() == "poses.json":
            continue  # skip meta file

        class_dir = os.path.join(dataset_folder, label)
        if not os.path.isdir(class_dir):
            continue

        key_dir = os.path.join(class_dir, "keypoints")
        ann_dir = os.path.join(class_dir, "annotated")
        os.makedirs(key_dir, exist_ok=True)
        os.makedirs(ann_dir, exist_ok=True)

        for img_name in sorted(os.listdir(class_dir)):
            img_path = os.path.join(class_dir, img_name)
            if not (os.path.isfile(img_path) and img_path.lower().endswith(valid_ext)):
                continue  # skip non-image files

            # YOLO pose prediction
            results = model.predict(img_path, boxes=False, verbose=False)
            r = results[0]

            if r.keypoints is None or r.keypoints.xyn.numel() == 0:
                key = [0.0] * 34  # fallback if no keypoints detected
            else:
                key = r.keypoints.xyn.cpu()[0].view(-1).tolist()

            key.extend([img_path, label, counter])
            rows.append(key)

            if saving_flag:
                cv2.imwrite(os.path.join(ann_dir, img_name), r.plot(boxes=False))
                base, _ = os.path.splitext(img_name)
                with open(os.path.join(key_dir, f"{base}.json"), "w") as f:
                    json.dump({
                        "image path": img_path,
                        "label_str": label,
                        "label_idx": counter,
                        "keypoints": key
                    }, f, indent=2)

        counter += 1

    return rows


# Main script execution
dataset_folder = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "yoga_kaggle_dataset"))
data = data_label(dataset_folder)

# Convert to DataFrame
n_feats = len(data[0])  # 34 keypoint values + 3 metadata fields
cols = [f"e{i}" for i in range(n_feats)]
cols[-3] = "image_path"
cols[-2] = "label_str"
cols[-1] = "label_idx"
df = pd.DataFrame(data, columns=cols)

# Final touches and save
df = df.fillna(0.0)
df["label_idx"] = df["label_idx"].astype(float).astype(int)
df.to_csv("yolo_keypoints_dataset.csv", index=False)
print("Saved", len(df), "rows")
