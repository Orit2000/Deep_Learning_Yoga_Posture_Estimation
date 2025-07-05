import os

# Set this to your dataset directory path
dataset_dir = r"C:\Users\safit\OneDrive\Datasets\yoga_kaggle_dataset"

# Loop through subfolders in the dataset directory
for folder_name in os.listdir(dataset_dir):
    folder_path = os.path.join(dataset_dir, folder_name)

    # Only process directories that contain underscores
    if os.path.isdir(folder_path) and "_" in folder_name:
        new_folder_name = folder_name.replace("_", " ")
        new_folder_path = os.path.join(dataset_dir, new_folder_name)

        if not os.path.exists(new_folder_path):
            os.rename(folder_path, new_folder_path)
            print(f"✅ Renamed: '{folder_name}' → '{new_folder_name}'")
        else:
            print(f"⚠️ Skipped: '{new_folder_name}' already exists.")
    else:
        print(f"⏩ Skipped: '{folder_name}' (no underscores or not a folder)")

print("🎉 Folder renaming complete.")
