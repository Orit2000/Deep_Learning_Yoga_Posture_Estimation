# ResNet Fine Tuning Flow
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
train_size = int(0.8 * len(dataset))
val_size   = len(dataset) - train_size
train_ds, val_ds = random_split(dataset, [train_size, val_size])

# train_ds = datasets.ImageFolder(root, transform=train_tf)
# val_ds   = datasets.ImageFolder(root, transform=val_tf)
train_dl = DataLoader(train_ds, batch_size=32, shuffle=True)
val_dl   = DataLoader(val_ds,   batch_size=32, shuffle=False)

num_classes = len(dataset.classes)

print(train_ds.__sizeof__())


device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)

# 1) Model ----------------------------------------------------------------
model = models.resnet18(weights="IMAGENET1K_V1")
in_feat          = model.fc.in_features
model.fc         = nn.Linear(in_feat, num_classes)
model.to(device)

loss_fn = nn.CrossEntropyLoss()

# -------- Phase A: head only --------------------------------------------
for p in model.parameters():          # freeze all
    p.requires_grad_(False)
for p in model.fc.parameters():       # unfreeze head
    p.requires_grad_(True)

opt = torch.optim.Adam(model.fc.parameters(), lr=1e-3)

history, best_epoch = train_loop(model, train_dl, val_dl, opt, loss_fn,
           epochs=5, num_classes=num_classes)

# -------- Phase B: head + last block ------------------------------------
for p in model.layer4.parameters():   # unfreeze
    p.requires_grad_(True)

opt = torch.optim.Adam([
        {"params": model.layer4.parameters(), "lr": 1e-4},
        {"params": model.fc.parameters(),     "lr": 5e-4},
])

history, best_epoch =train_loop(model, train_dl, val_dl, opt, loss_fn,
           epochs=10, num_classes=num_classes)

torch.save(model.state_dict(), "resnet18_yoga.pt")

