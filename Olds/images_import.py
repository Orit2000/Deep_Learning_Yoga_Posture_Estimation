import os
import shutil

# CONFIGURE THIS
dataset_root = os.path.normpath("./mnt/data/Yoga-82/yoga_dataset_links")        # folder containing original image folders
output_root = os.path.normpath("./images")     # new base directory to create organized folders
train_txt = os.path.normpath("./mnt/data/Yoga-82/yoga_train.txt")
test_txt = os.path.normpath("./mnt/data/Yoga-82/yoga_test.txt")

def copy_images(txt_file, split):
    with open(txt_file, 'r') as f:
        for line in f:
            parts = line.strip().split(',')
            if len(parts) < 1:
                continue
            image_path = parts[0]  # e.g., Akarna_Dhanurasana/16.jpg
            class_name, filename = image_path.split('/')
            
            # Create target dir: e.g., output_root/Akarna_Dhanurasana/train/
            target_dir = os.path.join(output_root, class_name, split)
            os.makedirs(target_dir, exist_ok=True)

            src = os.path.join(dataset_root, image_path)
            print(src)
            dst = os.path.join(target_dir, filename)
            print(dst)

            # Copy the image if it exists
            if os.path.exists(src):
                shutil.copy(src, dst)
            else:
                print(f"Missing image: {src}")

# Process train and test sets
copy_images(train_txt, "train")
copy_images(test_txt, "test")

print("✅ All images organized successfully.")
