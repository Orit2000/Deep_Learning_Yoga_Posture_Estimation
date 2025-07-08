# This Python code snippet is performing the following tasks:
import pandas as pd

kp  = pd.read_csv("yolo_keypoints_dataset.csv")
cnn = pd.read_csv("resnet18_embeddings.csv")

# 1️⃣  Rename the feature columns so they’re unique
kp_feat  = [c for c in kp.columns  if c.startswith("e")]
cnn_feat = [c for c in cnn.columns if c.startswith("e")]

kp.rename (columns={c: f"kp_{c}"  for c in kp_feat }, inplace=True)
cnn.rename(columns={c: f"cnn_{c}" for c in cnn_feat}, inplace=True)

# 2️⃣  Merge on the bookkeeping keys
keys   = ["image_path", "label_idx", "label_str"]
cnn["image_path"] = kp["image_path"]        # copy from kp if row order matches
cnn["label_idx"]  = kp["label_idx"]
cnn["label_str"]  = kp["label_str"]
combo  = kp.merge(cnn)    # inner → keep only rows present in both
front = [c for c in combo.columns if c not in keys]+keys
combo = combo[front]
# 3️⃣  (Optional) sanity check
# assert combo.filter(regex=r"^kp_").shape[1]  == len(kp_feat)   # 34
# assert combo.filter(regex=r"^cnn_").shape[1] == len(cnn_feat)  # 512
#assert combo.duplicated("image_path").sum() == 0               # one row per picture

# 4️⃣  Save – Parquet is faster & safer, but CSV is fine too
combo.to_parquet("combo_features.parquet", index=False)
combo.to_csv("combo_features.csv",index=False)