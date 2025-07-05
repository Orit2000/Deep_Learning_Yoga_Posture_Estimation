import os
import requests
from tqdm import tqdm

pose_txt_dir = os.path.normpath("./mnt/data/Yoga-82/yoga_dataset_links")  # where the 82 .txt files are
image_output_dir = os.path.normpath("./images_not_splitted")   # we'll store images inside same structure

# Loop over all class text files
for txt_file in os.listdir(pose_txt_dir):
    if not txt_file.endswith(".txt"):
        continue
    
    class_txt_path = os.path.join(pose_txt_dir, txt_file)
    
    with open(class_txt_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    for line in tqdm(lines, desc=f"Processing {txt_file}"):
        try:
            image_rel_path, url = line.strip().split()
            image_rel_path = os.path.normpath(image_rel_path)
            image_abs_path = os.path.join(image_output_dir, image_rel_path)

            # Create directory if needed
            os.makedirs(os.path.dirname(image_abs_path), exist_ok=True)

            # Skip if already exists
            if os.path.exists(image_abs_path):
                continue

            # Download
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                with open(image_abs_path, 'wb') as out_file:
                    out_file.write(response.content)
            else:
                print(f"❌ Failed: {url} - Status {response.status_code}")
        except Exception as e:
            print(f"⚠️ Error in line: {line.strip()} - {e}")
