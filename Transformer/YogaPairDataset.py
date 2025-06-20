import pandas as pd
import torch
from torch.utils.data import Dataset
class YogaPairDataset(Dataset):
    def __init__(self, parquet_path, kp_cols, cnn_cols, train=True, split=0.8,kp_mu=None, kp_std=None, cnn_mu=None, cnn_std=None):
        df_full = pd.read_parquet(parquet_path)

        # --- simple train/val split ----------------------------------
        perm = torch.randperm(len(df_full))
        cut  = int(split*len(df_full))
        self.df = df_full.iloc[perm[:cut]] if train else df_full.iloc[perm[cut:]]
        
        self.train_df = df_full.iloc[perm[:cut]].reset_index(drop=True)
        self.val_df   = df_full.iloc[perm[cut:]].reset_index(drop=True)
        #self.train_kp_cols = kp_cols.iloc[perm[:cut]].reset_index(drop=True)
        #self.train_cnn_cols   = cnn_cols.iloc[perm[cut:]].reset_index(drop=True)
        # --- z-score each modality on *train* split only -------------
        if train:
            self.kp_mu  = self.train_df[kp_cols].mean().values
            self.kp_std = self.train_df[kp_cols].std().values + 1e-8
            self.cnn_mu  = self.train_df[cnn_cols].mean().values
            self.cnn_std = self.train_df[cnn_cols].std().values + 1e-8
        else:
            # expect caller to pass the scalers from the train split
            self.kp_mu,  self.kp_std  = kp_mu,  kp_std
            self.cnn_mu, self.cnn_std = cnn_mu, cnn_std

    def __len__(self): return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        kp  = torch.tensor((row[kp_cols ].values - self.kp_mu ) / self.kp_std , dtype=torch.float32)
        cnn = torch.tensor((row[cnn_cols].values - self.cnn_mu) / self.cnn_std, dtype=torch.float32)
        y   = torch.tensor(row["label_idx"], dtype=torch.long)
        return kp, cnn, y


