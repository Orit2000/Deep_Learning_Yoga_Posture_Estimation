import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from YogaClassifier import YogaClassifier
from torch import nn
from torchsummary import summary
from torchmetrics import Accuracy, F1Score, ConfusionMatrix
from torch.utils.data import DataLoader, TensorDataset
import train_loop

#from train_loop import train_loop
print("✅ Loaded train_loop from:", train_loop.__file__)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
# Load the keypoints labels
# df = pd.read_csv("yolo_keypoints_dataset.csv")
# df = df.dropna() #delete undetected pose 
# df = df.iloc[:, 2:]

# print(f"Total features {len(df.columns)-2}")
# df.head()

# le = LabelEncoder()
# df['label'] = le.fit_transform(df['label'])

# classes_dict = {key:le.inverse_transform([key])[0] for key in range(len(df['label'].unique()))}
# num_classes = len(classes_dict)

# print(f"Total classes: {num_classes} ")
# print(classes_dict)

# X = df.drop(["label","image_path"], axis=1).values
# y = df['label'].values
df = pd.read_csv("yolo_keypoints_dataset.csv").dropna()
# Use the numeric label directly

# Only keep rows where e0–e33 are all numeric
keypoint_columns = [f"e{i}" for i in range(34)]

# Try converting all keypoint columns to numeric, coerce errors to NaN
df[keypoint_columns] = df[keypoint_columns].apply(pd.to_numeric, errors='coerce')

# Drop any rows with NaN in the keypoint columns
df = df.dropna(subset=keypoint_columns)

X = df[keypoint_columns].values
y = df["label_idx"].values
num_classes = len(df["label_idx"].unique())
input_length = X.shape[1]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

X_train, X_test = torch.tensor(X_train, dtype=torch.float32), torch.tensor(X_test, dtype=torch.float32)
y_train, y_test = torch.tensor(y_train, dtype=torch.long), torch.tensor(y_test, dtype=torch.long)

train_tensor = TensorDataset(X_train, y_train)
test_tensor = TensorDataset(X_test, y_test) 
BATCH_SIZE = 32

train_dataloader = DataLoader(dataset=train_tensor, batch_size=BATCH_SIZE, shuffle=True)
test_dataloader = DataLoader(dataset=test_tensor, batch_size=BATCH_SIZE, shuffle=False)

hidden_dim = 256
lr         = 0.00197
batch_size = 32
opt_name   = "Adam"
dropout    = 0.271  

model = YogaClassifier(input_length, num_classes, hidden_dim, dropout).to(device)
summary(model, input_size=(X.shape)) 

optimizer = torch.optim.Adam(model.parameters(), lr=lr)
loss_fn = nn.CrossEntropyLoss()
epochs = 30
result, best = train_loop.train_loop(
      model=model,
      trainloader=train_dataloader,
      testloader=test_dataloader,
      optimizer=optimizer,
      loss_fn=loss_fn,
      epochs=epochs,
      num_classes=num_classes,
      verbose=True,
      trial=None,
      patience=5
      )

