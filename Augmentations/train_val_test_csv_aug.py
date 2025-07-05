import os
import pandas as pd

def extend_csv_with_augmentations(csv_path, dataset_root='yoga_kaggle_dataset'):
    """
    Reads an original CSV (e.g., train_set.csv) and appends rows for any _aug*.jpg images found
    in the same directory as their original. Outputs a new CSV with '_augmented' suffix.
    """
    df = pd.read_csv(csv_path)
    augmented_rows = []

    for _, row in df.iterrows():
        original_path = row['filename']
        class_folder, split_folder, base_file = original_path.split('/')
        label = row['label_idx']

        dir_path = os.path.join(dataset_root, class_folder, split_folder)
        base_name, ext = os.path.splitext(base_file)

        # Find all augmentations of the form *_augX.jpg
        for fname in os.listdir(dir_path):
            if fname.startswith(base_name + '_aug') and fname.endswith(ext):
                rel_path = f"{class_folder}/{split_folder}/{fname}"
                augmented_rows.append({'filename': rel_path, 'label_idx': label})

    # Combine original + augmentations
    df_augmented = pd.concat([df, pd.DataFrame(augmented_rows)], ignore_index=True)

    # Save to new CSV
    out_path = csv_path.replace('.csv', '_augmented.csv')
    df_augmented.to_csv(out_path, index=False)
    print(f"✅ Saved: {out_path} ({len(df_augmented)} rows)")

# Run for all three splits
extend_csv_with_augmentations('Transformer/train_set.csv', dataset_root=r'C:\Users\safit\OneDrive\Datasets\yoga_kaggle_dataset_aug')
extend_csv_with_augmentations('Transformer/val_set.csv', dataset_root=r'C:\Users\safit\OneDrive\Datasets\yoga_kaggle_dataset_aug')
extend_csv_with_augmentations('Transformer/test_set.csv', dataset_root=r'C:\Users\safit\OneDrive\Datasets\yoga_kaggle_dataset_aug')
