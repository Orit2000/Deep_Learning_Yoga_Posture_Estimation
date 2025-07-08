"""
train_loop.py

This module provides a training loop with accuracy, F1, and mean Average Precision (mAP) tracking
for multiclass classification tasks using PyTorch and torchmetrics.

Functions:
- train_step: Single epoch training phase.
- test_step: Single epoch validation/test phase.
- train_loop: Full training loop over multiple epochs.

Metrics:
- Accuracy (top-1)
- F1 Score (macro)
- mean Average Precision (mAP)

Dependencies:
- torch
- torchmetrics
- tqdm
- sklearn (for report, confusion matrix — optional)
"""

import os
import numpy as np
import pandas as pd
import torch
from torch import nn
from torchmetrics import Accuracy, F1Score, AveragePrecision
from tqdm.auto import tqdm
from torch import nn


def train_step(model, dataloader, optimizer, loss_fn, device, accuracy_score, f1_score, map_score):
    """
    Executes one training epoch:
    - Updates model parameters
    - Computes loss, accuracy, F1, AP, and mAP

    Returns:
    - Average loss, accuracy, F1, mAP, and AP array for the epoch
    """
    model.train()
    train_loss, train_acc, train_f1 = 0, 0, 0

    all_logits = []  # to collect softmax scores for AP/mAP
    all_labels = []  # to collect ground truth labels

    for X, y in dataloader:
        X, y = X.to(device), y.to(device)

        # Forward pass
        y_pred = model(X)

        # Compute loss
        loss = loss_fn(y_pred, y)
        train_loss += loss.item()

        # Compute softmax scores for AP/mAP and prediction
        logits = torch.softmax(y_pred, dim=1)
        class_prediction = torch.argmax(logits, dim=1)

        # Update metrics
        acc = accuracy_score(class_prediction, y)
        f1 = f1_score(class_prediction, y)
        train_acc += acc.item()
        train_f1 += f1.item()

        # Accumulate for AP/mAP
        all_logits.append(logits.detach().cpu())
        all_labels.append(y.detach().cpu())

        # Backpropagation and optimizer step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Compute epoch averages
    train_loss /= len(dataloader)
    train_acc /= len(dataloader)
    train_f1 /= len(dataloader)

    # Compute AP and mAP across all collected outputs
    all_logits = torch.cat(all_logits).to(device)
    all_labels = torch.cat(all_labels).to(device)
    ap_per_class = map_score(all_logits, all_labels)
    mean_ap = ap_per_class.mean().item()

    return train_loss, train_acc, train_f1, mean_ap, ap_per_class.cpu().numpy()

def test_step(model, dataloader, loss_fn, device, accuracy_score, f1_score, map_score):
    """
    Executes one evaluation epoch:
    - Does not update model parameters
    - Computes loss, accuracy, F1, AP, and mAP

    Returns:
    - Average loss, accuracy, F1, mAP, and AP array for the epoch
    """
    model.eval()
    test_loss, test_acc, test_f1 = 0, 0, 0

    all_logits = []  # to collect softmax scores for AP/mAP
    all_labels = []  # to collect ground truth labels

    with torch.inference_mode():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)

            # Forward pass
            y_pred = model(X)

            # Compute loss
            loss = loss_fn(y_pred, y)
            test_loss += loss.item()

            # Compute softmax scores for AP/mAP and prediction
            logits = torch.softmax(y_pred, dim=1)
            class_prediction = torch.argmax(logits, dim=1)

            # Update metrics
            acc = accuracy_score(class_prediction, y)
            f1 = f1_score(class_prediction, y)
            test_acc += acc.item()
            test_f1 += f1.item()

            # Accumulate for AP/mAP
            all_logits.append(logits.detach().cpu())
            all_labels.append(y.detach().cpu())

    # Compute epoch averages
    test_loss /= len(dataloader)
    test_acc /= len(dataloader)
    test_f1 /= len(dataloader)

    # Compute AP and mAP across all collected outputs
    all_logits = torch.cat(all_logits).to(device)
    all_labels = torch.cat(all_labels).to(device)
    ap_per_class = map_score(all_logits, all_labels)
    mean_ap = ap_per_class.mean().item()

    return test_loss, test_acc, test_f1, mean_ap, ap_per_class.cpu().numpy()

def train_loop(model, trainloader, testloader, optimizer, loss_fn, epochs, num_classes, verbose=True):
    """
    Full training loop:
    - Runs multiple epochs of train + test
    - Tracks loss, accuracy, F1, AP (per class), and mAP
    - Saves the best model by accuracy

    Args:
        model: PyTorch model
        trainloader: DataLoader for training data
        testloader: DataLoader for validation/test data
        optimizer: Optimizer
        loss_fn: Loss function
        epochs: Number of epochs to train
        num_classes: Number of target classes
        verbose: Whether to print progress

    Returns:
        history (dict): tracked metrics for each epoch
        best_epoch (int): epoch with the highest test accuracy
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize metric trackers
    accuracy_score = Accuracy(task="multiclass", num_classes=num_classes).to(device)
    f1_score = F1Score(task="multiclass", num_classes=num_classes).to(device)
    map_score = AveragePrecision(task="multiclass", num_classes=num_classes).to(device)

    # Initialize history storage
    history = {
        "train_loss": [],
        "train_accuracy": [],
        "train_f1": [],
        "train_mAP": [],
        "train_AP": [],  # per class AP
        "test_loss": [],
        "test_accuracy": [],
        "test_f1": [],
        "test_mAP": [],
        "test_AP": [],  # per class AP
    }

    best_epoch = -1
    best_acc = 0

    for epoch in tqdm(range(epochs)):
        # Run training phase
        train_loss, train_acc, train_f1, train_map, train_ap_array = train_step(
            model, trainloader, optimizer, loss_fn, device,
            accuracy_score, f1_score, map_score
        )

        # Run validation/test phase
        test_loss, test_acc, test_f1, test_map, test_ap_array = test_step(
            model, testloader, loss_fn, device,
            accuracy_score, f1_score, map_score
        )

        # Save metrics
        history["train_loss"].append(train_loss)
        history["train_accuracy"].append(train_acc)
        history["train_f1"].append(train_f1)
        history["train_mAP"].append(train_map)
        history["train_AP"].append(train_ap_array)

        history["test_loss"].append(test_loss)
        history["test_accuracy"].append(test_acc)
        history["test_f1"].append(test_f1)
        history["test_mAP"].append(test_map)
        history["test_AP"].append(test_ap_array)

        # Save best model by accuracy
        if test_acc > best_acc:
            best_acc = test_acc
            best_epoch = epoch
            torch.save(model.state_dict(), "best.pth")
            status = "Accuracy improved, saving weight..."
        else:
            status = f"Accuracy not improved from epoch {best_epoch}"

        # Verbose output
        if verbose:
            print(f"Epoch {epoch}")
            print(f"Train Loss: {train_loss:.4f} | Val Loss: {test_loss:.4f}")
            print(f"Train Acc: {train_acc:.4f} | Val Acc: {test_acc:.4f}")
            print(f"Train F1: {train_f1:.4f} | Val F1: {test_f1:.4f}")
            print(f"Train mAP: {train_map:.4f} | Val mAP: {test_map:.4f}")
            print(f"Train AP: {train_ap_array} | Val AP: {test_ap_array}")
            print(status)
            print("-" * 50)

    print(f"Best accuracy on epoch: {best_epoch}, accuracy: {best_acc:.4f}")
    return history, best_epoch
