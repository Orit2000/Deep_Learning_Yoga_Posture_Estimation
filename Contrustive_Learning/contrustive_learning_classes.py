import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class ContrastiveMLP(nn.Module):
    def __init__(self, in_dim, embed_dim=128):
        super().__init__()
        self.projector = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, embed_dim)
        )

    def forward(self, x):
        return self.projector(x)


# Full Model
class ContrastiveModel(nn.Module):
    def __init__(self, keypoint_dim=34, cnn_dim=128):
        super().__init__()
        self.MLP_keypoint = ContrastiveMLP(keypoint_dim)
        self.MLP_cnn = ContrastiveMLP(cnn_dim)

    def forward(self, keypoints, f_images):
        # 3. Projection MLPs (your contrastive model)
        z_keypoint = F.normalize(self.MLP_keypoint(keypoints), dim=1)      # → [B, D]
        z_cnn      = F.normalize(self.MLP_cnn(f_images), dim=1)       # → [B, D]

        return z_keypoint, z_cnn
    
