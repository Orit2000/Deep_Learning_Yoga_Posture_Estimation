import os
import torch
import pandas as pd
from tqdm import tqdm
from PIL import Image
from torchvision import models, transforms

# === Load keypoints CSV ===
keypoints_csv = "augmented_keypoints_only.csv"  # Output from the previous script
df = pd.read_csv(keypoints_csv)

# === ResNet model setup ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
resnet = models.resnet18(weights="IMAGENET1K_V1")
resnet.fc = torch.nn.Identity()
resnet.to(device)
resnet.eval()

# === Image transform ===
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# === Extract CNN features ===
all_features = []
kept_rows = []

print(f"📂 Extracting CNN features for {len(df)} augmented images")

for idx, row in tqdm(df.iterrows(), total=len(df)):
    path = row["image_path"]

    if not os.path.exists(path):
        print(f"❌ Missing file: {path}")
        continue

    try:
        pil_image = Image.open(path).convert("RGB")
        img = transform(pil_image).unsqueeze(0).to(device)

        with torch.no_grad():
            embedding = resnet(img).squeeze().cpu().numpy().tolist()

        all_features.append(embedding)
        kept_rows.append(row)

    except Exception as e:
        print(f"⚠️ Error at {path}: {e}")
        continue

# === Construct final DataFrame ===
cnn_cols = [f"cnn_e{i}" for i in range(len(all_features[0]))] if all_features else []
cnn_df = pd.DataFrame(all_features, columns=cnn_cols)
meta_df = pd.DataFrame(kept_rows).reset_index(drop=True)

final_df = pd.concat([meta_df, cnn_df], axis=1)

# === Save final output ===
final_df.to_csv("augmented_keypoints_with_cnn.csv", index=False)
print("✅ Saved to 'augmented_keypoints_with_cnn.csv'")
