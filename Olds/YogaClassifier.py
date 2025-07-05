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
import YogaClassifier
from torch import nn
from torchsummary import summary
from torchmetrics import Accuracy, F1Score, ConfusionMatrix
from torch.utils.data import DataLoader, TensorDataset
from tqdm.auto import tqdm

from ultralytics import YOLO
from PIL import Image
import warnings
class YogaClassifier(nn.Module):
    def __init__(self, num_classes, input_length):
        super().__init__()
        self.layer1 = nn.Linear(in_features=input_length, out_features=64)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        self.layer2 = nn.Linear(in_features=64, out_features=64)
        self.outlayer = nn.Linear(in_features=64, out_features=num_classes)

    def forward(self, x):
        x = self.layer1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.layer2(x)
        x = self.activation(x)
        x = self.outlayer(x)
        return x

