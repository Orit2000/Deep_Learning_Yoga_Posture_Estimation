from train_loop import train_loop
import torch
import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
from YogaClassifier import YogaClassifier
from torch import nn
from torchsummary import summary
from torchmetrics import Accuracy, F1Score, ConfusionMatrix
from torch.utils.data import DataLoader, TensorDataset
from tqdm.auto import tqdm

from ultralytics import YOLO
from PIL import Image
import warnings

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
# Load the keypoints labels
df = pd.read_csv(r"yolo_keypoints_dataset.csv")
df = df.dropna() #delete undetected pose 
df = df.iloc[:, 2:]

print(f"Total features {len(df.columns)-2}")
df.head()

le = LabelEncoder()
df['label'] = le.fit_transform(df['label'])

classes_dict = {key:le.inverse_transform([key])[0] for key in range(len(df['label'].unique()))}
num_classes = len(classes_dict)

print(f"Total classes: {num_classes} ")
print(classes_dict)

X = df.drop(["label","image_path"], axis=1).values
y = df['label'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

X_train, X_test = torch.tensor(X_train, dtype=torch.float32), torch.tensor(X_test, dtype=torch.float32)
y_train, y_test = torch.tensor(y_train, dtype=torch.long), torch.tensor(y_test, dtype=torch.long)

train_tensor = TensorDataset(X_train, y_train)
test_tensor = TensorDataset(X_test, y_test) 
BATCH_SIZE = 32

train_dataloader = DataLoader(dataset=train_tensor, batch_size=BATCH_SIZE, shuffle=True)
test_dataloader = DataLoader(dataset=test_tensor, batch_size=BATCH_SIZE, shuffle=False)

input_length = X.shape[1]
model = YogaClassifier(num_classes=num_classes, input_length=input_length).to(device)
summary(model, input_size=(X.shape)) 

optimizer = torch.optim.Adam(lr=0.001, params=model.parameters())
loss_fn = nn.CrossEntropyLoss()
epochs = 100

result, best = train_loop(
      model=model,
      trainloader=train_dataloader,
      testloader=test_dataloader,
      optimizer=optimizer,
      loss_fn=loss_fn,
      epochs=epochs,
      num_classes=num_classes,
      verbose=True,
      )

