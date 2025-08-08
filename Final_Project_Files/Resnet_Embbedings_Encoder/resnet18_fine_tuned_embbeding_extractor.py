import torch
import torch.nn as nn
import pandas as pd
from torchvision import models, transforms
from PIL import Image
import numpy as np
import os
from tqdm import tqdm

# ----------------------------
# Load fine-tuned ResNet-18
# ----------------------------
class ResNet18FeatureExtractor(nn.Module):
    def __init__(self, weights_path):
        super().__init__()
        # 1. Match your trained config (47 classes)
        model = models.resnet18(num_classes=47)

        # 2. Load trained weights
        state_dict = torch.load(weights_path, map_location='cpu')
        model.load_state_dict(state_dict)

        # 3. Remove classifier to extract 512-dim features
        model.fc = nn.Identity()

        # 4. Freeze
        for param in model.parameters():
            param.requires_grad = False

        self.model = model
        self.eval()

    def forward(self, x):
        return self.model(x)


# ----------------------------
# Image preprocessing
# ----------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # ResNet input size
    transforms.ToTensor(),
    # transforms.Normalize(mean=[0.485, 0.456, 0.406],  # ImageNet stats
    #                      std=[0.229, 0.224, 0.225])
])

def load_and_preprocess_image(path):
    image = Image.open(path).convert("RGB")
    return transform(image)

# ----------------------------
# Extract embeddings
# ----------------------------
def extract_embeddings(model, csv_path, output_file):
    df = pd.read_csv(csv_path)
    paths = df["image_path"].tolist()
    labels = df["label_idx"].tolist()

    features = []
    model.eval()
    for path in tqdm(paths):
        img = load_and_preprocess_image(path)
        img = img.unsqueeze(0)  # [1, 3, 224, 224]
        with torch.no_grad():
            vec = model(img)  # [1, 512]
        features.append(vec.squeeze(0).numpy())

    # Save as CSV
    out_df = pd.DataFrame(features)
    out_df["label"] = labels
    out_df.to_csv(output_file, index=False)
    print(f"Saved embeddings to {output_file}")

# ----------------------------
# Main
# ----------------------------
if __name__ == "__main__":
    model_path = "best_fine_tune_resnet18.pth"  # path to your saved model
    extractor = ResNet18FeatureExtractor(model_path)

    extract_embeddings(extractor, "train_set.csv", "resnet18_train_embeddings.csv")
    extract_embeddings(extractor, "val_set.csv", "resnet18_val_embeddings.csv")
    extract_embeddings(extractor, "test_set.csv", "resnet18_test_embeddings.csv")
