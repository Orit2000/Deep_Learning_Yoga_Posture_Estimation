import pandas as pd
import os

# Define the conversion rule
def convert_path(unix_path: str) -> str:
    # Extract the pose folder and file name from the original path
    parts = unix_path.strip().split('/')
    if len(parts) < 2:
        raise ValueError(f"Unexpected image_path format: {unix_path}")
    
    pose_folder = parts[-2]
    filename = parts[-1]

    # Build the new Windows-style path
    return os.path.join("C:\\Users\\safit\\OneDrive\\Datasets\\yoga_kaggle_dataset", pose_folder, filename)

# Files to process
csv_files = ["Augmentations/train_set.csv", "Augmentations/val_set.csv","Augmentations/test_set.csv"]

for csv_file in csv_files:
    print(f"Processing {csv_file}...")
    
    df = pd.read_csv(csv_file)

    if "image_path" not in df.columns:
        raise KeyError(f"'image_path' column not found in {csv_file}")
    
    # Apply conversion
    df["image_path"] = df["image_path"].apply(convert_path)
    
    # Save as <original_name>_updated.csv in the same directory
    dir_name = os.path.dirname(csv_file)
    base_name = os.path.basename(csv_file).replace(".csv", "_updated.csv")
    updated_file = os.path.join(dir_name, base_name)

    df.to_csv(updated_file, index=False)
    print(f"✅ Saved updated file: {updated_file}")
