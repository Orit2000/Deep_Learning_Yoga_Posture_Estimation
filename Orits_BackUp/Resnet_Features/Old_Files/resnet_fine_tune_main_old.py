import torch, os, json
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from Resnet_Features.resnet_fine_tune_old import ResNetFineTune   # your wrapper

# ─── 1.  Dataset & transforms ────────────────────────────────────────────────
root_dir = r"../yoga_kaggle_dataset"    # <— all images live under class folders

base_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),              # still in [0,1]
])

full_ds   = datasets.ImageFolder(root_dir, transform=base_transform)
print("Classes:", full_ds.classes)

# ─── 2.  Compute μ, σ on *all* images (since no split) ───────────────────────
loader = DataLoader(full_ds, batch_size=64, shuffle=False, num_workers=4)

sum_, sum_sq, total = torch.zeros(3), torch.zeros(3), 0
for imgs, _ in loader:
    B, C, H, W = imgs.shape
    sum_    += imgs.sum(dim=[0, 2, 3])
    sum_sq  += (imgs**2).sum(dim=[0, 2, 3])
    total   += B * H * W

mean = sum_ / total
std  = torch.sqrt(sum_sq / total - mean**2)
print("μ:", mean.tolist(), "\nσ:", std.tolist())

norm_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean.tolist(), std.tolist()),
])
full_ds.transform = norm_transform           # update in-place

# ─── 3.  Loader (no split) ──────────────────────────────────────────────────
batch_sz   = 32
train_loader = DataLoader(full_ds, batch_size=batch_sz,
                          shuffle=True, num_workers=4)

# ─── 4.  Model, loss, optimiser ─────────────────────────────────────────────
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(len(full_ds.classes))
model  = ResNetFineTune(embeddim=512,
                        freeze_backbone=False).to(device)

optimizer = optim.Adam([
    {"params": model.backbone.parameters(),   "lr": 1e-5},
    {"params": model.classifier.parameters(), "lr": 1e-3},
])
criterion   = nn.CrossEntropyLoss()
epochs      = 30

# ─── 5.  Training loop ──────────────────────────────────────────────────────
for epoch in range(1, epochs+1):
    model.train()
    epoch_loss = 0.0

    for imgs, labels in train_loader:
        imgs, labels = imgs.to(device), labels.to(device)

        optimizer.zero_grad()
        logits = model(imgs)
        loss   = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item() * imgs.size(0)

    print(f"Epoch {epoch:02d} – loss {epoch_loss/len(full_ds):.4f}")

# ─── 6.  Extract & save embeddings for every image ──────────────────────────
model.eval()

# helper: forward pass that returns *just* the 512-D backbone vector
def get_features(x):
    with torch.no_grad():
        feats = model.backbone(x)          # [B, 512, 1, 1]
        return torch.flatten(feats, 1)     # [B, 512]

all_feats, all_paths, all_labels = [], [], []

for imgs, labels in DataLoader(full_ds, batch_size=64,
                               shuffle=False, num_workers=4):
    imgs   = imgs.to(device)
    feats  = get_features(imgs).cpu()
    all_feats.append(feats)
    all_labels.extend(labels.numpy().tolist())

    # recover original file paths directly from dataset
    start_idx = len(all_paths)
    for i in range(imgs.size(0)):
        img_path, _ = full_ds.samples[start_idx + i]
        all_paths.append(img_path)

all_feats = torch.cat(all_feats, 0)        # (N, 512)

# ─── 7.  Persist to disk ────────────────────────────────────────────────────
torch.save({"features": all_feats,         # Tensor [N, 512]
            "labels"  : all_labels,        # List[int]
            "paths"   : all_paths}, 
           "yoga_resnet18_embeddings.pt")

print("Saved embeddings to yoga_resnet18_embeddings.pt")
