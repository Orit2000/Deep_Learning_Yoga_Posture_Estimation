import pandas as pd
from pathlib import Path

def replace_cnn_embeddings(set_csv_path, new_emb_csv_path, out_path):
    df_set = pd.read_csv(set_csv_path)
    df_new = pd.read_csv(new_emb_csv_path)

    # sanity checks
    assert df_set.shape[0] == df_new.shape[0], f"Row count mismatch: {set_csv_path} vs {new_emb_csv_path}"

    # columns: old CNN embeddings in the set, new embeddings 0..511 in the new file
    cnn_cols = [c for c in df_set.columns if c.startswith("cnn_e")]
    new_cols = [str(i) for i in range(512)]

    assert len(cnn_cols) == 512, f"Expected 512 cnn_e* columns in {set_csv_path}, found {len(cnn_cols)}"
    assert all(c in df_new.columns for c in new_cols), "New embeddings file must have columns '0'..'511'"

    # replace by row order
    df_set.loc[:, cnn_cols] = df_new[new_cols].values

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    df_set.to_csv(out_path, index=False)
    print(f"Saved: {out_path}")

# Example usage:
replace_cnn_embeddings("train_set_updated_kp_conf.csv", "resnet18_train_embeddings_half_fine_tune.csv", "train_set_half_fine_tune_kp_conf.csv")
replace_cnn_embeddings("val_set_updated_kp_conf.csv",   "resnet18_val_embeddings_half_fine_tune.csv",   "val_set_half_fine_tune_kp_conf.csv")
replace_cnn_embeddings("test_set_updated_kp_conf.csv",  "resnet18_test_embeddings_half_fine_tune.csv",  "test_set_half_fine_tune_kp_conf.csv")
