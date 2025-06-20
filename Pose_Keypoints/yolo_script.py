import os
import cv2
import json
from ultralytics import YOLO
from pathlib import Path

# === Paths ===
# image_folder = "./images_not_splitted/Akarna_Dhanurasana"           # Folder with input images
# keypoints_folder = "./images_not_splitted/Akarna_Dhanurasana/output_keypoints"   # Where to save the .json files
# annotated_folder = "./images_not_splitted/Akarna_Dhanurasana/output_images"      # Where to save the visualized images
image_folder = r"C:\Users\Orit\Deep_Learning_Project\Boat_Pose_or_Paripurna_Navasana_"
keypoints_folder = r"C:\Users\Orit\Deep_Learning_Project\Boat_Pose_or_Paripurna_Navasana_\output_keypoints"   # Where to save the .json files
annotated_folder = r"C:\Users\Orit\Deep_Learning_Project\Boat_Pose_or_Paripurna_Navasana_\output_images"      # Where to save the visualized images

os.makedirs(keypoints_folder, exist_ok=True)
os.makedirs(annotated_folder, exist_ok=True)

# === Load YOLOv8-pose model ===
model = YOLO("yolo11x-pose.pt")  # or your own .pt model

# === Process all images ===
image_paths = list(Path(image_folder).glob("*.jpg")) + list(Path(image_folder).glob("*.png"))

for img_path in image_paths:
    print(str(img_path))
    img = cv2.imread(str(img_path))
    results = model(str(img_path))

    for result_id, result in enumerate(results):
        image_name = img_path.stem
        annotated_img = result.plot()  # draws keypoints

        # Save annotated image
        cv2.imwrite(os.path.join(annotated_folder, f"{image_name}.jpg"), annotated_img)

        # Extract and save keypoints
        keypoints = []
        if result.keypoints is not None:
            for person_kpts in result.keypoints.data.cpu().numpy():  # shape: [n_keypoints, 3]
                kp_list = []
                for kpt in person_kpts:
                    x, y, conf = map(float, kpt)
                    kp_list.append({"x": x, "y": y, "confidence": conf})
                keypoints.append(kp_list)

        # Save to JSON
        with open(os.path.join(keypoints_folder, f"{image_name}.json"), "w") as f:
            json.dump({"image": image_name, "keypoints": keypoints}, f, indent=2)

        print(f"✅ Processed: {image_name}")

print("🎉 All images processed.")


# # Access the results
# for result in results:
# xy = result.keypoints.xy  # x and y coordinates
# xyn = result.keypoints.xyn  # normalized
# kpts = result.keypoints.data  # x, y, visibility (if available)
    
