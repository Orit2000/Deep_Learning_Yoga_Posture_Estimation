import pandas as pd
import torch
from torch.utils.data import Dataset
class YogaPairDataset(Dataset):
    def __init__(self, path, kp_cols, cnn_cols, split=0.8):
        df_full = pd.read_csv(path)
        print(len(df_full))

        # --- simple train/val split ----------------------------------
        for i in range(0,num_classes+1):
            
        perm = torch.randperm(len(df_full))
        print(perm)
        cut  = int(split*len(df_full))
        # rows for each split
        train_idx = perm[:cut]
        val_idx   = perm[cut:]
        # full row sets
        cols_to_keep = kp_cols + cnn_cols + ["image_path"]+["label_str"]+["label_idx"]
        self.train_df = (
            df_full
            .iloc[train_idx][cols_to_keep]       
            .reset_index(drop=True)
        )
        self.val_df = (
            df_full
            .iloc[val_idx][cols_to_keep]
            .reset_index(drop=True)
        )
        self.kp_cols=kp_cols
        self.cnn_cols=cnn_cols
        #self.train_kp_cols = kp_cols.iloc[perm[:cut]].reset_index(drop=True)
        #self.train_cnn_cols   = cnn_cols.iloc[perm[cut:]].reset_index(drop=True)
        # --- z-score each modality on *train* split only -------------
        
        self.kp_mu  = self.train_df[kp_cols].mean().values
        self.kp_std = self.train_df[kp_cols].std().values + 1e-8
        self.cnn_mu  = self.train_df[cnn_cols].mean().values
        self.cnn_std = self.train_df[cnn_cols].std().values + 1e-8


    # def __len__(self): return len(self.df)

    # def __getitem__(self, idx):
    #     row = self.df.iloc[idx]
    #     kp  = torch.tensor((row[self.kp_cols].values - self.kp_mu ) / self.kp_std , dtype=torch.float32)
    #     cnn = torch.tensor((row[self.cnn_cols].values - self.cnn_mu) / self.cnn_std, dtype=torch.float32)
    #     y   = torch.tensor(row["label_idx"], dtype=torch.long)
    #     return kp, cnn, y


