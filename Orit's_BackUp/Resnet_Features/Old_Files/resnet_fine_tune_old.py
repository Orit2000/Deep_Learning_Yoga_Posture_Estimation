import torch.nn as nn
from torchvision import models

class ResNetFineTune(nn.Module):
    def __init__(self, backbone_name='resnet18', embeddim=10, freeze_backbone=True):
        super().__init__()
        # 1. Load the pretrained backbone
        backbone = getattr(models, backbone_name)(pretrained=True)
        
        # 2. Remove its original classifier
        #    resnet18.fc is nn.Linear(512, 1000)
        in_features = backbone.fc.in_features
        backbone.fc = nn.Identity()
        
        self.backbone = backbone   # outputs a [B, in_features] feature vector
        
        # 3. New classifier head
        self.classifier = nn.Linear(in_features, embeddim)
        
        # 4. Optionally freeze all backbone parameters
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

    def forward(self, x):
        # x: [B, 3, H, W]
        feats = self.backbone(x)         # → [B, in_features]
        logits = self.classifier(feats)  # → [B, embeddim]
        return logits