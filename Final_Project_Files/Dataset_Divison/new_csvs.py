# # simple_update_kp_into_splits.py
# import re
# import pandas as pd
# from pathlib import Path

# # ==== EDIT THESE ====
# KP_CSV = "yolo_keypoints_confidence_dataset.csv"
# SPLITS = [
#     "train_set_updated.csv",
#     "val_set_updated.csv",
#     "test_set_updated.csv",
# ]
# OUT_SUFFIX = "_kpconf"   # new files: <name>_kpconf.csv
# # ====================

# # 17 joints -> (x,y,c)
# KP_TRIPLETS = [f"kp{j}_{a}" for j in range(17) for a in ("x", "y", "c")]

# def load_kp_map(kp_csv: str) -> pd.DataFrame:
#     kp = pd.read_csv(kp_csv)
#     need = ["image_path"] + KP_TRIPLETS
#     missing = [c for c in need if c not in kp.columns]
#     if missing:
#         raise ValueError(f"KP CSV missing columns: {missing}")
#     # keep one row per image_path
#     kp = kp.drop_duplicates(subset=["image_path"], keep="first")
#     return kp[need].copy()

# def drop_old_kp_cols(df: pd.DataFrame) -> pd.DataFrame:
#     cols = df.columns.tolist()
#     to_drop = []
#     # new-style triplets
#     to_drop += [c for c in cols if re.fullmatch(r"kp\d+_[xyc]", c)]
#     # old single-vector styles (common in earlier scripts)
#     to_drop += [c for c in cols if re.fullmatch(r"(e|kp_e|pose_e)\d+", c)]
#     to_drop = sorted(set(to_drop))
#     if to_drop:
#         df = df.drop(columns=to_drop)
#     return df

# def update_one_split(split_path: str, kp_map: pd.DataFrame, out_suffix: str):
#     print(f"The path is {split_path}")
#     df = pd.read_csv(split_path)
#     if "image_path" not in df.columns:
#         raise ValueError(f"{split_path}: 'image_path' column is required")

#     # remove any existing KP cols (if present)
#     df = drop_old_kp_cols(df)

#     # left-merge to keep same rows/order
#     merged = df.merge(kp_map, on="image_path", how="left", validate="one_to_one")

#     # if some images didn't get KP (not found), fill with zeros so shapes match
#     if merged[KP_TRIPLETS].isna().any().any():
#         missing = merged.loc[merged[KP_TRIPLETS].isna().any(axis=1), "image_path"]
#         print(f"[WARN] {split_path}: {len(missing)} rows missing KP; filling zeros. Example:",
#               list(missing.head(3)))
#         merged[KP_TRIPLETS] = merged[KP_TRIPLETS].fillna(0.0)

#     # keep label_idx integer if exists
#     if "label_idx" in merged.columns:
#         merged["label_idx"] = merged["label_idx"].astype(int)

#     p = Path(split_path)
#     out_path = p.with_name(p.stem + out_suffix + p.suffix)
#     merged.to_csv(out_path, index=False)
#     print(f"✔ Saved {out_path}  (rows: {len(merged)})")

# def main():
#     kp_map = load_kp_map(KP_CSV)
#     for sp in SPLITS:
#         update_one_split(sp, kp_map, OUT_SUFFIX)

# if __name__ == "__main__":
#     main()
# update_splits_legacy_interleaved_conf.py
import re
import pandas as pd
from pathlib import Path

# ==== EDIT THESE ====
KP_CSV = "yolo_keypoints_confidence_dataset.csv"
SPLITS = [
    "train_set_updated.csv",
    "val_set_updated.csv",
    "test_set_updated.csv",
]
OUT_SUFFIX = "_kp_conf"   # -> <name>_kp_legacy_interconf.csv
# ====================

N_JOINTS = 17
# Legacy coord names (x0,y0,...,x16,y16)
KP_E = [f"kp_e{i}" for i in range(2 * N_JOINTS)]
# Confidence names
KP_C = [f"kp_c{j}" for j in range(N_JOINTS)]
# Interleaved pose order: e0,e1,c0, e2,e3,c1, ... e32,e33,c16
POSE_INTERLEAVED = [f for j in range(N_JOINTS) for f in (f"kp_e{2*j}", f"kp_e{2*j+1}", f"kp_c{j}")]
# CNN pattern and meta order (keep image/labels columns position)
CNN_RE = re.compile(r"cnn_e(\d+)")
META_FINAL = ["image_path", "label_idx", "label_str"]  # exact order you asked

def load_kp_triplets(kp_csv: str) -> pd.DataFrame:
    """Load triplet KP CSV (kp{j}_x/kp{j}_y/kp{j}_c) and return:
       image_path + kp_e0..kp_e33 + kp_c0..kp_c16 (we'll interleave later)
    """
    df = pd.read_csv(kp_csv)
    trip_x = [f"kp{j}_x" for j in range(N_JOINTS)]
    trip_y = [f"kp{j}_y" for j in range(N_JOINTS)]
    trip_c = [f"kp{j}_c" for j in range(N_JOINTS)]
    need = ["image_path"] + trip_x + trip_y + trip_c
    missing = [c for c in need if c not in df.columns]
    if missing:
        raise ValueError(f"KP CSV missing columns: {missing}")

    # dedup by image_path
    df = df.drop_duplicates(subset=["image_path"], keep="first")

    out = pd.DataFrame({"image_path": df["image_path"].values})
    # build legacy coord vector e0..e33 (x0,y0,x1,y1,...)
    e_cols = []
    for j in range(N_JOINTS):
        e_cols.append(df[trip_x[j]].astype(float).rename(f"kp_e{2*j}"))
        e_cols.append(df[trip_y[j]].astype(float).rename(f"kp_e{2*j+1}"))
    e_df = pd.concat(e_cols, axis=1)

    # confidences c0..c16
    c_df = pd.DataFrame({f"kp_c{j}": df[trip_c[j]].astype(float).values for j in range(N_JOINTS)})

    out = out.merge(e_df, left_index=True, right_index=True)
    out = out.merge(c_df, left_index=True, right_index=True)
    # return with all pose cols (we'll reorder to interleaved at the very end)
    return out[["image_path"] + KP_E + KP_C]

def drop_old_pose_cols(df: pd.DataFrame) -> pd.DataFrame:
    """Remove any existing pose/conf columns (legacy or triplets)."""
    cols = df.columns.tolist()
    to_drop = set()
    to_drop.update([c for c in cols if re.fullmatch(r"kp_e\d+", c)])     # legacy coords
    to_drop.update([c for c in cols if re.fullmatch(r"kp_c\d+", c)])     # old confidences
    to_drop.update([c for c in cols if re.fullmatch(r"kp\d+_[xyc]", c)]) # triplets
    if to_drop:
        df = df.drop(columns=sorted(to_drop))
    return df

def natural_sort_cnn(cols: list[str]) -> list[str]:
    pairs = []
    for c in cols:
        m = CNN_RE.fullmatch(c)
        if m:
            pairs.append((int(m.group(1)), c))
    return [c for _, c in sorted(pairs)]

def reorder_columns_interleaved(merged: pd.DataFrame) -> pd.DataFrame:
    """Final order: (e0,e1,c0, e2,e3,c1, ... e32,e33,c16), cnn_e0.., (others), image_path,label_idx,label_str."""
    # Pose (interleaved)
    pose_present = [c for c in POSE_INTERLEAVED if c in merged.columns]

    # CNN next
    cnn_cols = natural_sort_cnn([c for c in merged.columns if CNN_RE.fullmatch(c)])

    # Meta last in requested order
    meta_present = [c for c in META_FINAL if c in merged.columns]

    # Others in between CNN and META (exclude pose/CNN/meta)
    exclude = set(pose_present) | set(cnn_cols) | set(meta_present)
    others = [c for c in merged.columns if c not in exclude]

    final = pose_present + cnn_cols + others + meta_present
    return merged[final]

def update_one_split(split_path: str, kp_legacy: pd.DataFrame, out_suffix: str):
    print(f"Updating: {split_path}")
    df = pd.read_csv(split_path)
    if "image_path" not in df.columns:
        raise ValueError(f"{split_path}: missing 'image_path'")

    # drop any old pose/conf columns
    df = drop_old_pose_cols(df)

    # merge new legacy pose (coords+conf) by image_path
    merged = df.merge(kp_legacy, on="image_path", how="left", validate="one_to_one")

    # fill missing pose values with zeros (if some images lack KP)
    pose_cols = KP_E + KP_C
    if merged[pose_cols].isna().any().any():
        miss = merged.loc[merged[pose_cols].isna().any(axis=1), "image_path"]
        print(f"[WARN] {split_path}: {len(miss)} rows missing KP; filling zeros. Example:", list(miss.head(3)))
        merged[pose_cols] = merged[pose_cols].fillna(0.0)

    # keep label_idx integer if present
    if "label_idx" in merged.columns:
        merged["label_idx"] = merged["label_idx"].astype(int)

    # final order: interleaved pose -> cnn -> others -> image_path,label_idx,label_str
    merged = reorder_columns_interleaved(merged)

    p = Path(split_path)
    out_path = p.with_name(p.stem + out_suffix + p.suffix)
    merged.to_csv(out_path, index=False)
    print(f"✔ Saved {out_path}  (rows: {len(merged)})")

def main():
    kp_legacy = load_kp_triplets(KP_CSV)
    for sp in SPLITS:
        update_one_split(sp, kp_legacy, OUT_SUFFIX)

if __name__ == "__main__":
    main()
