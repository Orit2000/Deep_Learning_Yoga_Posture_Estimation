import pandas as pd
import os

# val_df = pd.read_csv("Augmentations/val_set.csv")
# val_filenames = val_df["image_path"].apply(lambda x: os.path.basename(x))
# print(val_filenames.head(10))

augmented_df = pd.read_csv("augmented_keypoints_with_cnn.csv")

aug_filenames = augmented_df["image_path"].apply(lambda x: os.path.basename(x))
base_names = aug_filenames.apply(lambda x: x.split("_aug")[0])
print(base_names.value_counts().head(10))