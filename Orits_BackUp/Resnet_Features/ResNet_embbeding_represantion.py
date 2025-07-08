# ResNet embbeding represantion
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import torch, torch.nn as nn
from torchvision import models
from torchvision.datasets import ImageFolder
import pandas as pd
from torch.utils.data import random_split
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Pose_Keypoints')))


def ignore_annotated(path: str) -> bool:
    """
    Accept the file only if it is NOT in an “annotated” sub-folder and
    it ends with an image extension.
    """
    return (
        "annotated" not in path.lower()
        and path.lower().endswith((".jpg", ".jpeg", ".png"))
    )

class ImageFolderWithPaths(ImageFolder):
    """ImageFolder that (1) drops annotated files and (2) returns the file path."""
    def __init__(self, root, transform=None, target_transform=None):
        super().__init__(
            root,
            transform=transform,
            target_transform=target_transform,
            is_valid_file=ignore_annotated        # <<< filter applied here
        )

    def __getitem__(self, index):
        # grab (image_tensor, label_idx) from the parent class
        img, label = super().__getitem__(index)
        # fetch the corresponding file path
        path = self.samples[index][0]
        return img, label, path

IMG_SIZE = 224
train_tf = transforms.Compose([
        transforms.Resize((IMG_SIZE,IMG_SIZE)),
        # transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        # transforms.Normalize(
        #     mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
])

root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'yoga_kaggle_dataset'))
print("Resolved dataset path:", root)
print("Exists?", os.path.isdir(root))
#root = r"../../yoga_kaggle_dataset"         # same folder tree you use in data_keypoints_labeling.py :contentReference[oaicite:0]{index=0}
dataset = ImageFolderWithPaths(root, transform=train_tf) 
print("Number of images:", len(dataset))  # dataset = datasets.ImageFolder(...)

dataset_dl = DataLoader(dataset, batch_size=32, shuffle=True)
num_classes = len(dataset.classes)
class_names = dataset.classes  


device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)

# 1) Model --------------------------------------------------------
model = models.resnet18(weights="IMAGENET1K_V1")
in_feat          = model.fc.in_features
model.fc         = nn.Identity()
model.to(device)

# 2) --------------------------------------------------------------
all_embeddings = []
all_labels = []
all_paths = []
all_inx = []
all_names = []
model.eval()
with torch.no_grad():
    for images, labels, paths in dataset_dl:
        images = images.to(device)
        feats = model(images)  # [B, 512]
        all_embeddings.append(feats.cpu())
        all_inx.append(labels.cpu())
        all_names.extend([class_names[i] for i in labels])
        all_paths.extend(paths)                     # strings
        
#print(all_inx.shape)
embeddings = torch.cat(all_embeddings)  # [N, 512]
labels_idx     = torch.cat(all_inx)      # [N]
#print(labels.shape)

torch.save(
    {"embeddings": embeddings,
     "label_idx" : labels_idx,
     "label_str" : all_names,
     "paths"     : all_paths},
    "resnet18_embeddings.pt"
)
print(embeddings.shape)
# --------------------------------- Build DataFrame -------------------------
cols = [f"e{i}" for i in range(embeddings.size(1))]   # e0 … e511
df   = pd.DataFrame(embeddings.numpy(), columns=cols)
df["image_path"] = all_paths
df["label_str"]  = all_names
# --------- ❶ make an alpha-order mapping ----------------------------------
alpha_classes = sorted(df["label_str"].unique())          # A → Z
alpha2idx     = {cls: i for i, cls in enumerate(alpha_classes)}

df["label_idx"] = df["label_str"].map(alpha2idx)          # new numbers
# --------- ❷ sort rows by that alphabetical order -------------------------
df = df.sort_values(["label_idx", "image_path"]).reset_index(drop=True)

df.to_csv("resnet18_embeddings.csv", index=False)
print("✅ Saved", len(df), "rows to resnet18_embeddings.csv")