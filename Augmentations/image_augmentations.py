import os
import cv2
import pandas as pd
from tqdm import tqdm
import albumentations as A

# Parameters
num_augmentations = 3

# Define augmentation pipeline
augmentation = A.Compose([
    A.Affine(scale=(0.9, 1.1), rotate=(-12, 12), shear=(-5, 5), translate_percent=(0.02, 0.05), p=0.7),
    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
    A.MotionBlur(blur_limit=3, p=0.2),
    A.GaussNoise(var_limit=(10.0, 50.0), p=0.2),  
    A.ImageCompression(quality_lower=80, quality_upper=95, p=0.3), 
])

# Target CSV files to augment (train + val only)
csv_paths = {
    "train": r"C:\Users\safit\OneDrive\GitHub\Deep_Learning_Yoga_Posture_Estimation\Augmentations\train_set.csv",
    "val": r"C:\Users\safit\OneDrive\GitHub\Deep_Learning_Yoga_Posture_Estimation\Augmentations\val_set.csv"
}

# Process each CSV
for name, csv_path in csv_paths.items():
    print(f"📁 Processing {name}_set.csv...")

    df = pd.read_csv(csv_path)
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        original_path = row["image_path"]

        if not os.path.exists(original_path):
            print(f"❌ Skipping missing file: {original_path}")
            continue

        image = cv2.imread(original_path)
        if image is None:
            print(f"❌ Could not load image: {original_path}")
            continue

        folder, filename = os.path.split(original_path)
        name_only, ext = os.path.splitext(filename)

        for i in range(num_augmentations):
            augmented = augmentation(image=image)["image"]
            aug_filename = f"{name_only}_aug{i+1}{ext}"
            aug_path = os.path.join(folder, aug_filename)

            cv2.imwrite(aug_path, augmented)

    print(f"✅ Augmentations completed for {name}_set.csv")
