import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

import torch
from torch import nn
from torchsummary import summary
from torchmetrics import Accuracy, F1Score, ConfusionMatrix
from torch.utils.data import DataLoader, TensorDataset
from tqdm.auto import tqdm

from ultralytics import YOLO
from PIL import Image
import warnings

def _reset_metrics(*metrics):
    for m in metrics:
        m.reset()

# def train_step(model, dataloader, optimizer, loss_fn,device,accuracy_score,f1_score):
#     model.train()
#     train_loss, train_acc, train_f1 = 0, 0, 0
#     for batch, (X, y) in enumerate(dataloader):
#         X, y = X.to(device), y.to(device)

#         y_pred = model(X)

#         #loss
#         loss = loss_fn(y_pred, y)
#         train_loss += loss.item()

#         #accuracy
#         logits = torch.softmax(y_pred, dim=1)
#         class_prediction = torch.argmax(logits, dim=1)
#         acc = accuracy_score(class_prediction, y)
#         f1 = f1_score(class_prediction, y)
#         train_acc += acc.item()
#         train_f1 += f1.item()

#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()

#     train_loss /= len(dataloader)
#     train_acc /= len(dataloader)
#     train_f1 /= len(dataloader)

#     return train_loss, train_acc, train_f1
def train_step(model, dataloader, optimizer, loss_fn,
               device, accuracy_score, f1_score):
    model.train()
    running_loss = running_acc = running_f1 = 0.0

    _reset_metrics(accuracy_score, f1_score)

    for kp, cnn, y in dataloader:                  # ← 3-tuple
        kp, cnn, y = kp.to(device), cnn.to(device), y.to(device)

        logits = model(kp, cnn)                    # forward
        loss   = loss_fn(logits, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * y.size(0)
        running_acc  += accuracy_score(logits, y).item() * y.size(0)
        running_f1   += f1_score     (logits, y).item() * y.size(0)

    n = len(dataloader.dataset)
    return (running_loss / n,
            running_acc  / n,
            running_f1   / n)


# def test_step(model, dataloader, loss_fn,device,accuracy_score,f1_score):
#     model.eval()
#     test_loss, test_acc, test_f1 = 0, 0, 0
#     with torch.inference_mode():
#         for batch, (X,y) in enumerate(dataloader):
#             X, y = X.to(device), y.to(device)
#             y_pred = model(X)

#             #loss
#             loss = loss_fn(y_pred, y)
#             test_loss += loss.item()

#             #accuracy
#             logits = torch.softmax(y_pred, dim=1)
#             class_prediction = torch.argmax(logits, dim=1)
#             acc = accuracy_score(class_prediction, y)
#             f1 = f1_score(class_prediction, y)
#             test_acc += acc.item()
#             test_f1 += f1.item()

#     test_loss /= len(dataloader)
#     test_acc /= len(dataloader)
#     test_f1 /= len(dataloader)
#     return test_loss, test_acc, test_f1

@torch.no_grad()
def test_step(model, dataloader, loss_fn,
              device, accuracy_score, f1_score):
    model.eval()
    running_loss = running_acc = running_f1 = 0.0

    _reset_metrics(accuracy_score, f1_score)

    for kp, cnn, y in dataloader:                  # ← 3-tuple
        kp, cnn, y = kp.to(device), cnn.to(device), y.to(device)

        logits = model(kp, cnn)
        loss   = loss_fn(logits, y)

        running_loss += loss.item() * y.size(0)
        running_acc  += accuracy_score(logits, y).item() * y.size(0)
        running_f1   += f1_score     (logits, y).item() * y.size(0)

    n = len(dataloader.dataset)
    return (running_loss / n,
            running_acc  / n,
            running_f1   / n)
# ───────────────────────────── loop ────────────────────────────────
def train_loop(model,
               trainloader,
               testloader,
               optimizer,
               loss_fn,
               epochs,
               num_classes,
               verbose=True):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    accuracy = Accuracy(task="multiclass", num_classes=num_classes).to(device)
    f1       = F1Score (task="multiclass", num_classes=num_classes).to(device)

    history = {k: [] for k in
               ("train_loss","train_accuracy","train_f1",
                "test_loss","test_accuracy","test_f1")}

    best_acc   = 0.0
    best_epoch = -1

    for epoch in tqdm(range(epochs), desc="epoch"):
        tl, ta, tf = train_step(model, trainloader,
                                optimizer, loss_fn, device,
                                accuracy, f1)

        vl, va, vf = test_step (model, testloader,
                                loss_fn, device,
                                accuracy, f1)

        # record
        history["train_loss"].append(tl);  history["train_accuracy"].append(ta); history["train_f1"].append(tf)
        history["test_loss"] .append(vl);  history["test_accuracy"] .append(va); history["test_f1"] .append(vf)

        # checkpoint
        improved = va > best_acc
        if improved:
            best_acc   = va
            best_epoch = epoch
            torch.save(model.state_dict(), "best.pth")

        if verbose:
            print(f"Epoch {epoch:02d}"
                  f" | train loss {tl:.4f}  acc {ta:.3f}  f1 {tf:.3f}"
                  f" | val loss {vl:.4f}  acc {va:.3f}  f1 {vf:.3f}"
                  f" | {'↑ saved' if improved else f'best @ {best_epoch}'}")

    print(f"Best val acc = {best_acc:.3f} at epoch {best_epoch}")
    return history, best_epoch
# def train_loop(model, trainloader, testloader, optimizer, loss_fn, epochs, num_classes,verbose=True,):
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     accuracy_score = Accuracy(task="multiclass", num_classes=num_classes).to(device)
#     f1_score = F1Score(task="multiclass", num_classes=num_classes).to(device)
#     history = {
#         "train_loss": [],
#         "train_accuracy": [],
#         "train_f1": [],
#         "test_loss": [],
#         "test_accuracy": [],
#         "test_f1": [],
#     }

#     for epoch in tqdm(range(epochs)):
#         train_loss, train_acc, train_f1 = train_step(
#             model=model,
#             dataloader=trainloader,
#             optimizer=optimizer,
#             loss_fn=loss_fn,
#             device=device,
#             accuracy_score=accuracy_score,
#             f1_score=f1_score
#         )

#         test_loss, test_acc, test_f1 = test_step(
#             model=model,
#             dataloader=testloader,
#             loss_fn=loss_fn,
#             device=device,
#             accuracy_score=accuracy_score,
#             f1_score=f1_score
#         )
       
#         history['train_loss'].append(train_loss)
#         history['train_accuracy'].append(train_acc)
#         history['train_f1'].append(train_f1)
#         history['test_loss'].append(test_loss)
#         history['test_accuracy'].append(test_acc)
#         history['test_f1'].append(test_f1)
    
#         # Checkpoint
#         best = max(history['test_accuracy'])
#         best_epoch = history['test_accuracy'].index(best) 
   
#         if test_acc < best:
#             status = f"Accuracy not improved from epoch {best_epoch}"
#         else: 
#             status = f"Accuracy improved, saving weight....."
#             torch.save(model.state_dict(), 'best.pth')

#         if verbose:
#             print(f"Epoch {epoch}")
#             print(f"train loss: {train_loss} | test loss: {test_loss}")
#             print(f"train accuracy: {train_acc} | test accuracy: {test_acc}")
#             print(f"train f1: {train_f1} | test f1: {test_f1}")
#             print(status)
#             print("-------------------------------------------------")

#     print(f"Best accuracy on epoch: {best_epoch}, accuracy: {best}")
#     return history, best_epoch