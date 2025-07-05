from ultralytics import YOLO
import os
import pandas as pd 
import cv2
import json
from sklearn.preprocessing import LabelEncoder

def data_label(dateset_folder, saving_flag=False):
    valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp')
    model_yolo = YOLO("yolo11x-pose.pt")
    data = []
    # Loop for each label
    for label in os.listdir(dateset_folder):
        
        # Loop for each image per label
        if label == "Poses.json":
            continue
        image_dir = os.path.join(dateset_folder,label)
        print(image_dir)
        
        keypoints_folder = os.path.join(image_dir,"keypoints")
        annotated_folder =os.path.join(image_dir,"annotated")
        print(keypoints_folder)
        print(annotated_folder)
        os.makedirs(keypoints_folder, exist_ok=True)
        os.makedirs(annotated_folder, exist_ok=True)
        for img in os.listdir(image_dir):
            image_path = os.path.join(image_dir, img)
            # skip if not a file or not an image
            if not os.path.isfile(image_path) or not image_path.lower().endswith(valid_extensions):
                print(f"Skipping: {image_path}")
                continue
            #image = os.path.join(dateset_folder,image_dir,img)
            image = os.path.join(image_dir,img)
            print(image)
            # Extracting keypoint with YOLOv8
            results = model_yolo.predict(image, boxes=False, verbose=False)
            
            #Save
            r = results[0]  # first result
            annotated_img = r.plot(boxes=False)  # draws keypoints
            if(saving_flag):
                cv2.imwrite(os.path.join(annotated_folder, f"{img}"), annotated_img)
            
            for r in results:
                keypoints = r.keypoints.xyn.cpu().numpy()[0]
                keypoints = keypoints.reshape((1, keypoints.shape[0]*keypoints.shape[1]))[0].tolist()
                #Save# Save to JSON
                keypoints.append(image) #insert image path
                keypoints.append(label) #insert image label

                data.append(keypoints)
              
            name, _ = os.path.splitext(img)
            print(name)
            if(saving_flag):
                with open(os.path.join(keypoints_folder, f"{name}.json"), "w") as f:
                    json.dump({"image": name, "label": label, "keypoints": keypoints}, f, indent=2)
    return data

# Main
dateset_folder =  r"../yoga_kaggle_dataset"
data_csv_path = r"yolo_keypoints_dataset.csv"
print(data_csv_path)
if os.path.exists(data_csv_path):
    df = pd.read_csv("yolo_keypoints_dataset.csv")
    df = df.dropna() #delete undetected pose 
    df = df.iloc[:, 2:]

    print(f"Total features {len(df.columns)-2}")
    df.head()
    df.to_csv("pose_features.csv", index=False)

    le = LabelEncoder()
    df['label'] = le.fit_transform(df['label'])

    classes_dict = {key:le.inverse_transform([key])[0] for key in range(len(df['label'].unique()))}
    num_classes = len(classes_dict)

    print(f"Total classes: {num_classes} ")
    print(classes_dict)

    X = df.drop(["label","image_path"], axis=1).values
    y = df['label'].values

    print(X.shape)
    print(y.shape)
else:
    dateset_folder =  r"../yoga_kaggle_dataset"
    data = data_label(dateset_folder)
    total_features = len(data[0])
    df = pd.DataFrame(
        data=data, 
        columns=[f"x{i}" for i in range(total_features)]
        ).rename({
            "x34":"image_path", "x35":"label"
            }, axis=1)
    df.to_csv("yolo_keypoints_dataset.csv", index=False)

