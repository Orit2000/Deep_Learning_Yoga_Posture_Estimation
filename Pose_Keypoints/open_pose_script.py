import os
import cv2
import sys
import json
import numpy as np
from sys import platform

# === Configure your OpenPose installation path ===
dir_path = os.path.dirname(os.path.realpath(__file__))
openpose_path = "./openpose-master/openpose"  # Change this to your OpenPose root directory
sys.path.append(os.path.join(openpose_path, "build/python"))
os.environ["PATH"] += os.pathsep + os.path.join(openpose_path, "build/x64/Release")
os.environ["PATH"] += os.pathsep + os.path.join(openpose_path, "build/bin")

from openpose import pyopenpose as op
# %%
# === Configure input/output folders ===
image_folder = "./images_not_splitted/Akarna_Dhanurasana"           # Folder with input images
keypoints_folder = "./images_not_splitted/Akarna_Dhanurasana/output_keypoints"   # Where to save the .json files
annotated_folder = "./images_not_splitted/Akarna_Dhanurasana/output_images"      # Where to save the visualized images

os.makedirs(keypoints_folder, exist_ok=True)
os.makedirs(annotated_folder, exist_ok=True)

# === OpenPose parameters ===
params = {
    "model_folder": os.path.join(openpose_path, "models"),
    "hand": False,
    "face": False,
    "display": 0,
    "render_threshold": 0.05
}

# === Start OpenPose ===
opWrapper = op.WrapperPython()
opWrapper.configure(params)
opWrapper.start()

# === Process images ===
for img_name in os.listdir(image_folder):
    if not img_name.lower().endswith((".png", ".jpg", ".jpeg")):
        continue

    image_path = os.path.join(image_folder, img_name)
    image = cv2.imread(image_path)

    datum = op.Datum()
    datum.cvInputData = image
    opWrapper.emplaceAndPop([datum])

    # Save keypoints to JSON
    keypoints = datum.poseKeypoints.tolist() if datum.poseKeypoints is not None else []
    json_path = os.path.join(keypoints_folder, img_name.rsplit('.', 1)[0] + ".json")
    with open(json_path, 'w') as f:
        json.dump({"keypoints": keypoints}, f)

    # Save visualized image
    annotated_path = os.path.join(annotated_folder, img_name)
    cv2.imwrite(annotated_path, datum.cvOutputData)

    print(f"✅ Processed: {img_name} | 👣 Keypoints: {len(keypoints)}")

print("🎉 All images processed.")
