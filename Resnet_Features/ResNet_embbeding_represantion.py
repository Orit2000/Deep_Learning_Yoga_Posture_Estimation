# ResNet embbeding represantion
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import torch, torch.nn as nn
from torchvision import models
from torch.utils.data import random_split
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Pose_Keypoints')))
from train_loop import train_loop

IMG_SIZE = 224
train_tf = transforms.Compose([
        transforms.Resize((IMG_SIZE,IMG_SIZE)),
        # transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        # transforms.Normalize(
        #     mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
])
val_tf   = transforms.Compose([
        transforms.Resize((IMG_SIZE,IMG_SIZE)),
        transforms.ToTensor(),
        # transforms.Normalize(
        #     mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
])
root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'yoga_kaggle_dataset'))
print("Resolved dataset path:", root)
print("Exists?", os.path.isdir(root))
#root = r"../../yoga_kaggle_dataset"         # same folder tree you use in data_keypoints_labeling.py :contentReference[oaicite:0]{index=0}
dataset = datasets.ImageFolder(root, transform=train_tf)  # all with train_tf, override later
dataset_dl = DataLoader(dataset, batch_size=32, shuffle=True)
num_classes = len(dataset.classes)



device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)

# 1) Model ----------------------------------------------------------------
model = models.resnet18(weights="IMAGENET1K_V1")
in_feat          = model.fc.in_features
model.fc         = nn.Identity()
model.to(device)

# 2) --------------------------------------------------------------
all_embeddings = []
all_labels = []

model.eval()
with torch.no_grad():
    for images, labels in dataset_dl:
        images = images.to(device)
        feats = model(images)  # [B, 512]
        all_embeddings.append(feats.cpu())
        all_labels.append(labels.cpu())

embeddings = torch.cat(all_embeddings)  # [N, 512]
labels     = torch.cat(all_labels)      # [N]

torch.save({'embeddings': embeddings, 'labels': labels}, 'resnet18_embeddings.pt')
