
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
# config
split_ratios = dict(train=0.8, val=0.1, test=0.1)
num_classes  = 47       # if classes are 0…N
np.random.seed(42)                                 # reproducible shuffle

# STEP 1: Dividing the dataset for each class 80-20
lengths_per_class = np.load("counter_classes.npy")
combo  = pd.read_csv("combo_features.csv")

sub_dfs = dict(train=[], val=[], test=[])
start=1
for cls, n_rows in enumerate(lengths_per_class):
    grp   = combo.iloc[start:start+n_rows-1]                  # rows for class <cls>
    # sanity-check: all rows should have the same label
    assert (grp["label_idx"] == cls).all()
    
    
    idx = np.random.permutation(n_rows-1)
    n_train = int(split_ratios["train"] * n_rows)
    n_val   = int(split_ratios["val"]   * n_rows)
    
    train_idx = idx[:n_train]
    val_idx   = idx[n_train:n_train + n_val]
    test_idx  = idx[n_train + n_val:]

    # append the three pieces
    sub_dfs["train"].append(grp.iloc[train_idx])
    sub_dfs["val"]  .append(grp.iloc[val_idx])
    sub_dfs["test"] .append(grp.iloc[test_idx])

    start = start+n_rows                                # advance to next class slice


# concatenate the per-class pieces
train_set = pd.concat(sub_dfs["train"]).reset_index(drop=True)
val_set   = pd.concat(sub_dfs["val"  ]).reset_index(drop=True)
test_set  = pd.concat(sub_dfs["test" ]).reset_index(drop=True)

# -------------------------------- save / inspect ---------------------------------
train_set.to_csv("train_set.csv", index=False)
val_set  .to_csv("val_set.csv"  , index=False)
test_set .to_csv("test_set.csv" , index=False)

print(f"train: {len(train_set)}  val: {len(val_set)}  test: {len(test_set)}")
# STEP 2: