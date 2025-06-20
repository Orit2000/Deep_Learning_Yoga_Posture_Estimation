from ultralytics import YOLO
import os
import pandas as pd 
import cv2
import json
from sklearn.preprocessing import LabelEncoder
def data_label(dataset_folder, saving_flag=False):
    valid_ext = (".jpg", ".jpeg", ".png", ".bmp")
    model     = YOLO("yolo11x-pose.pt")
    rows      = []
    counter = int(0)
    for label in sorted(os.listdir(dataset_folder)):
        if label.lower() == "poses.json":
            continue                                     # skip meta file

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
                continue                                 # not an image

            # ---------- YOLO pose prediction --------------------------------
            results = model.predict(img_path, boxes=False, verbose=False)

            r   = results[0]
            key = r.keypoints.xyn.cpu()[0].view(-1).tolist()  # 34 floats
            key.extend([img_path, label, int(counter)])                     # + path + label
            rows.append(key)

            # ---------- optional saving -------------------------------------
            if saving_flag:
                cv2.imwrite(os.path.join(ann_dir, img_name), r.plot(boxes=False))
                base, _ = os.path.splitext(img_name)
                with open(os.path.join(key_dir, f"{base}.json"), "w") as f:
                    json.dump({"image path": img_path,
                               "label_str": label,
                               "label_idx": int(counter),
                               "keypoints": key}, f, indent=2)
        counter = int(counter + 1)
    return rows

dataset_folder = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "yoga_kaggle_dataset")
)
data = data_label(dataset_folder)
# ---- DataFrame -----------------------------------------------------------
n_feats = len(data[0])        # 34 + 2
cols = [f"e{i}" for i in range(n_feats)]
cols[-3] = "image_path"
cols[-2] = "label_str"
cols[-1] = "label_idx"
df = pd.DataFrame(data, columns=cols)
# df = df.rename({cols[-2]: "image_path", cols[-1]: "label_str"}, axis=1)

# # ---- alphabetical class ↦ integer ---------------------------------------
# alpha_classes = sorted(df["label_str"].unique())        # A → Z
# alpha2idx     = {c: i for i, c in enumerate(alpha_classes)}
# df["label_idx"] = df["label_str"].map(alpha2idx)

# # ---- order rows for determinism -----------------------------------------
# df = df.sort_values(["label_idx", "image_path"]).reset_index(drop=True)

# ---- save ---------------------------------------------------------------
df = df.fillna(0.0)                           # or any sentinel value
df["label_idx"] = df["label_idx"].astype(float).astype(int)
df.to_csv("yolo_keypoints_dataset.csv", index=False)
print("✅ Saved", len(df), "rows")
