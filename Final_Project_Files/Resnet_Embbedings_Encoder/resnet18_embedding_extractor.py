"""
resnet18_embedding_extractor.py

This script extracts 512-dimensional ResNet-18 embeddings from yoga posture images
and saves them to a `.csv` file containing the embeddings, image paths, and class labels.

Main output:
- `resnet18_embeddings.csv`: A CSV file with one row per image containing embeddings, image path, label string, and label index.

Dependencies:
- torch, torchvision
- pandas
- os, sys
"""

from torchvision import transforms
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
from torchvision import models
from torchvision.datasets import ImageFolder
import pandas as pd
import sys
import os

# Allow importing custom modules from Pose_Keypoints directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Pose_Keypoints')))

def ignore_annotated(path: str) -> bool:
    """
    Filter function: Accepts the file only if it is not in an "annotated" sub-folder
    and ends with a supported image extension.

    Args:
        path (str): The full file path.

    Returns:
        bool: True if the file is valid (non-annotated and correct extension), else False.
    """
    return (
        "annotated" not in path.lower() and
        path.lower().endswith((".jpg", ".jpeg", ".png"))
    )

class ImageFolderWithPaths(ImageFolder):
    """
    Custom ImageFolder that:
    1. Filters out annotated files using `ignore_annotated`.
    2. Returns (image, label, path) in __getitem__.
    """
    def __init__(self, root, transform=None, target_transform=None):
        super().__init__(
            root,
            transform=transform,
            target_transform=target_transform,
            is_valid_file=ignore_annotated
        )

    def __getitem__(self, index):
        """
        Get an item from the dataset.

        Args:
            index (int): Index of the sample.

        Returns:
            tuple: (image tensor, label index, image path)
        """
        img, label = super().__getitem__(index)
        path = self.samples[index][0]
        return img, label, path

# Parameters
IMG_SIZE = 224
train_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor()
])

# Dataset setup
root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'yoga_kaggle_dataset'))
print("Resolved dataset path:", root)
print("Exists?", os.path.isdir(root))

dataset = ImageFolderWithPaths(root, transform=train_tf)
print("Number of images:", len(dataset))

dataset_dl = DataLoader(dataset, batch_size=32, shuffle=True)
num_classes = len(dataset.classes)
class_names = dataset.classes  

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

# Model
model = models.resnet18(weights="IMAGENET1K_V1")
in_feat = model.fc.in_features
model.fc = nn.Identity()
model.to(device)

# Embedding extraction
all_embeddings = []
all_inx = []
all_names = []
all_paths = []

model.eval()
with torch.no_grad():
    for images, labels, paths in dataset_dl:
        images = images.to(device)
        feats = model(images)
        all_embeddings.append(feats.cpu())
        all_inx.append(labels.cpu())
        all_names.extend([class_names[i] for i in labels])
        all_paths.extend(paths)

embeddings = torch.cat(all_embeddings)
labels_idx = torch.cat(all_inx)

# Build DataFrame and save CSV
cols = [f"e{i}" for i in range(embeddings.size(1))]
df = pd.DataFrame(embeddings.numpy(), columns=cols)
df["image_path"] = all_paths
df["label_str"] = all_names

# Create alphabetical mapping
alpha_classes = sorted(df["label_str"].unique())
alpha2idx = {cls: i for i, cls in enumerate(alpha_classes)}
df["label_idx"] = df["label_str"].map(alpha2idx)

# Sort rows
df = df.sort_values(["label_idx", "image_path"]).reset_index(drop=True)

df.to_csv("resnet18_embeddings.csv", index=False)
print(f"Saved {len(df)} rows to resnet18_embeddings.csv")
