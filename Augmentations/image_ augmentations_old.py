import os
import cv2
import albumentations as A
from tqdm import tqdm

def create_augmentations(
    root_dir: str,
    splits: list = ['train', "val"],
    n_augments: int = 3,
    image_extensions: tuple = ('.jpg', '.jpeg', '.png')
):
    """
    Applies image augmentations to a dataset organized as:
    root_dir/class_x/split/*.jpg
    
    Parameters:
    - root_dir (str): Root folder of the dataset
    - splits (list): List of subfolder names to augment (e.g., ['train'])
    - n_augments (int): Number of augmented versions to generate per image
    - image_extensions (tuple): File extensions considered as images
    """

    # Define augmentation pipeline using Albumentations
    augmentation_pipeline = A.Compose([
        A.Affine(scale=(0.9, 1.1), rotate=(-12, 12), shear=(-5, 5), translate_percent=(0.02, 0.05), p=0.7),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
        A.MotionBlur(blur_limit=3, p=0.2),
        A.GaussNoise(var_limit=(10.0, 50.0), p=0.2),  # ✅ Corrected: only `var_limit` allowed
        A.ImageCompression(quality_lower=80, quality_upper=95, p=0.3),  # ✅ Corrected
    ])

    # Loop over class directories
    for class_name in tqdm(os.listdir(root_dir), desc="Processing classes"):
        class_path = os.path.join(root_dir, class_name)

        # Skip if not a directory (e.g., accidentally placed files)
        if not os.path.isdir(class_path):
            continue

        # Process each split: train / val / test
        for split in splits:
            split_path = os.path.join(class_path, split)

            # Skip if this split doesn't exist for the class
            if not os.path.isdir(split_path):
                continue

            # Iterate over all image files in the split
            for filename in os.listdir(split_path):
                if not filename.lower().endswith(image_extensions):
                    continue

                image_path = os.path.join(split_path, filename)

                # Read and convert to RGB
                image = cv2.imread(image_path)
                if image is None:
                    print(f"Warning: Unable to read image {image_path}")
                    continue
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                # Create N augmented versions
                for i in range(n_augments):
                    augmented = augmentation_pipeline(image=image)['image']
                    augmented_bgr = cv2.cvtColor(augmented, cv2.COLOR_RGB2BGR)

                    # Construct new filename
                    base_name, ext = os.path.splitext(filename)
                    new_filename = f"{base_name}_aug{i}{ext}"
                    new_path = os.path.join(split_path, new_filename)

                    # Save augmented image
                    cv2.imwrite(new_path, augmented_bgr)

    print("✅ Augmentation completed.")

# Example usage
if __name__ == "__main__":
    create_augmentations(
        root_dir=r'C:\Users\safit\OneDrive\Datasets\yoga_kaggle_dataset_aug',
        splits=["train", "val"],             # Only augment training set by default
        n_augments=3,                 # Generate 3 augmented copies per image
        image_extensions=('.jpg', '.jpeg', '.png')  # Add more if needed
    )
