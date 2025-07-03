import os
import shutil
import random
from tqdm import tqdm

def split_dataset_by_class(
    input_root: str,
    output_root: str,
    split_ratios: dict = {'train': 0.8, 'val': 0.1, 'test': 0.1},
    seed: int = 42,
    image_extensions: tuple = ('.jpg', '.jpeg', '.png')
):
    """
    Splits a dataset organized as input_root/class_x/*.jpg into:
    output_root/class_x/train/val/test/*.jpg

    Parameters:
    - input_root (str): Path to source dataset directory.
    - output_root (str): Path to output split dataset.
    - split_ratios (dict): Dictionary with ratios for 'train', 'val', and 'test'.
    - seed (int): Random seed for reproducibility.
    - image_extensions (tuple): Allowed image file extensions.
    """

    assert abs(sum(split_ratios.values()) - 1.0) < 1e-6, "Split ratios must sum to 1."

    random.seed(seed)
    os.makedirs(output_root, exist_ok=True)

    # Loop through each class folder
    for class_name in tqdm(os.listdir(input_root), desc="Processing classes"):
        class_dir = os.path.join(input_root, class_name)
        if not os.path.isdir(class_dir):
            continue

        # Collect image paths
        image_files = [f for f in os.listdir(class_dir)
                       if f.lower().endswith(image_extensions)]

        # Shuffle for reproducibility
        random.shuffle(image_files)
        total = len(image_files)

        # Compute split indices
        n_train = int(split_ratios['train'] * total)
        n_val   = int(split_ratios['val'] * total)
        n_test  = total - n_train - n_val

        split_map = {
            'train': image_files[:n_train],
            'val'  : image_files[n_train:n_train+n_val],
            'test' : image_files[n_train+n_val:]
        }

        # Copy files to output structure
        for split, files in split_map.items():
            dest_dir = os.path.join(output_root, class_name, split)
            os.makedirs(dest_dir, exist_ok=True)

            for fname in files:
                src = os.path.join(class_dir, fname)
                dst = os.path.join(dest_dir, fname)
                shutil.copy2(src, dst)

    print("✅ Dataset successfully split.")

# Example usage
if __name__ == "__main__":
    split_dataset_by_class(
        input_root=r'C:\Users\safit\OneDrive\Datasets\yoga_kaggle_dataset',     # Replace with your input folder
        output_root=r'C:\Users\safit\OneDrive\Datasets\yoga_kaggle_dataset_aug',      # Desired output structure
        split_ratios={'train': 0.8, 'val': 0.1, 'test': 0.1},
        seed=42
    )
