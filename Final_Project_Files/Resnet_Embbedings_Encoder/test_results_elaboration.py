# simple_eval.py
import pandas as pd
from torchvision import transforms, models
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import classification_report
import torch.nn as nn
from collections import defaultdict
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
class YogaCSVLoader(Dataset):
    def __init__(self, csv_file, transform=None):
        """
        Dataset that loads images and labels from a CSV file.

        Args:
            csv_file (str): Path to CSV file containing image paths and labels.
            transform (callable, optional): Transformations to apply to the images.
        """
        print(f"Tha path: {csv_file}")
        self.df = pd.read_csv(csv_file)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_path = self.df.iloc[idx]['image_path']
        label = self.df.iloc[idx]['label_idx']

        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label
def evaluate_per_class(model, weights_pth, test_dl, batch_size=256):
    """
    model:       an instantiated model (e.g., TwoTokenTransformer(...))
                 whose forward is model(kp, cnn) -> logits [B, C]
    weights_pth: path to your saved state_dict (.pth)
    test_csv:    CSV with columns kp_*, cnn_*, label_idx
    """
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # state = torch.load(weights_pth, map_location="cpu")
    # model.load_state_dict(state, strict=False)
    # model.to(device).eval()

    # # --- load test split ---
    # df = pd.read_csv(test_csv)
    # assert "label_idx" in df.columns, "CSV must have label_idx"
    # KP_COLS  = [c for c in df.columns if c.startswith("kp_")]
    # CNN_COLS = [c for c in df.columns if c.startswith("cnn_")]
    # assert KP_COLS and CNN_COLS, "Need kp_* and cnn_* columns in CSV"

    # kp  = torch.tensor(df[KP_COLS ].values, dtype=torch.float32)
    # cnn = torch.tensor(df[CNN_COLS].values, dtype=torch.float32)
    # y   = torch.tensor(df["label_idx"].values, dtype=torch.long)

    # loader = DataLoader(TensorDataset(kp, cnn, y), batch_size=batch_size, shuffle=False)

    # # --- forward ---
    # preds, gts = [], []
    # with torch.no_grad():
    #     for b_kp, b_cnn, b_y in loader:
    #         b_kp, b_cnn = b_kp.to(device), b_cnn.to(device)
    #         logits = model(b_kp, b_cnn)
    #         preds.append(logits.argmax(1).cpu())
    #         gts.append(b_y)
    # y_pred = torch.cat(preds).numpy()
    # y_true = torch.cat(gts).numpy()

    # # --- per-class metrics ---
    # report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    # num_classes = int(y_true.max() + 1)

    # print(f"{'cls':>4} {'prec':>7} {'rec':>7} {'f1':>7} {'support':>8}")
    # for c in range(num_classes):
    #     d = report[str(c)]
    #     print(f"{c:>4} {d['precision']:7.3f} {d['recall']:7.3f} {d['f1-score']:7.3f} {int(d['support']):8d}")

    # return report  # dict you can reuse if you want

# Build the SAME model shape you trained
# model = TwoTokenTransformer(kp_dim=34, cnn_dim=512, d_model=256,
#                            nhead=1, depth=1, num_classes=47)
#def evaluate_per_class(model, weights_pth, test_dl):
    """
    Evaluates the model's performance on the test dataset, providing per-class accuracy
    and overall accuracy.

    Args:
        model (torch.nn.Module): The neural network model to evaluate.
        weights_pth (str): Path to the saved model weights (.pth file).
        test_dl (torch.utils.data.DataLoader): DataLoader for the test dataset.
    """
    # Determine the device to use (GPU if available, otherwise CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load the pre-trained weights into the model
    try:
        model.load_state_dict(torch.load(weights_pth, map_location=device))
        print(f"Successfully loaded model weights from: {weights_pth}")
    except FileNotFoundError:
        print(f"Error: Weights file not found at {weights_pth}. Please check the path.")
        return
    except Exception as e:
        print(f"Error loading weights: {e}")
        return

    # Set the model to evaluation mode
    # This disables dropout and batch normalization updates,
    # ensuring consistent behavior during evaluation.
    model.eval()
    model.to(device) # Move the model to the chosen device

    # Initialize dictionaries to store correct predictions and total predictions for each class
    correct_predictions_per_class = defaultdict(int)
    total_predictions_per_class = defaultdict(int)
    total_correct_predictions = 0
    total_samples = 0

    print("Starting evaluation...")
    # Disable gradient calculations during evaluation to save memory and speed up computation
    with torch.no_grad():
        for images, labels in test_dl:
            # Move images and labels to the appropriate device
            images = images.to(device)
            labels = labels.to(device)

            # Forward pass: get model outputs (logits)
            outputs = model(images)

            # Get the predicted class by finding the index with the maximum log-probability
            _, predictions = torch.max(outputs, 1)

            # Update overall correct predictions and total samples
            total_correct_predictions += (predictions == labels).sum().item()
            total_samples += labels.size(0)

            # Iterate through each sample in the batch to update per-class statistics
            for label, prediction in zip(labels, predictions):
                label = label.item() # Convert tensor to Python int
                prediction = prediction.item() # Convert tensor to Python int

                total_predictions_per_class[label] += 1
                if label == prediction:
                    correct_predictions_per_class[label] += 1

    print("\n--- Per-Class Accuracy ---")
    # Get class names if available from the dataset (assuming test_dl.dataset has an idx_to_label mapping)
    # This part depends on how your YogaCSVLoader maps indices to actual class names.
    # If not available, it will just print class indices.
    class_names = getattr(test_dl.dataset, 'idx_to_label', {})
    if not class_names:
        # If idx_to_label is not found, try to infer class IDs
        # This assumes labels are consecutive integers starting from 0
        all_labels = sorted(list(set(list(total_predictions_per_class.keys()))))
        class_names = {i: f"Class {i}" for i in all_labels}

    per_class_accuracies = {}
    # Print per-class accuracy
    for class_idx in sorted(total_predictions_per_class.keys()):
        class_name = class_names.get(class_idx, f"Unknown Class {class_idx}")
        total = total_predictions_per_class[class_idx]
        correct = correct_predictions_per_class[class_idx]
        accuracy = (correct / total) * 100 if total > 0 else 0.0
        per_class_accuracies[class_idx] = accuracy
        print(f"{class_name} (Class {class_idx}): Correct {correct}/{total} -> Accuracy: {accuracy:.2f}%")

    print("\n--------------------------")
    # Calculate and print overall accuracy
    overall_accuracy = (total_correct_predictions / total_samples) * 100 if total_samples > 0 else 0.0
    print(f"Overall Accuracy: Correct {total_correct_predictions}/{total_samples} -> {overall_accuracy:.2f}%")
    return per_class_accuracies, overall_accuracy

def plot_per_class_accuracies(val_accuracies, val_overall_acc, test_accuracies, test_overall_acc):
    """
    Generates a grouped bar chart for per-class accuracies of validation and test sets.

    Args:
        val_accuracies (dict): Dictionary of per-class accuracies for validation set.
        val_overall_acc (float): Overall accuracy for the validation set.
        test_accuracies (dict): Dictionary of per-class accuracies for test set.
        test_overall_acc (float): Overall accuracy for the test set.
    """
    # Ensure all classes from 0 to 46 are present in both dicts, filling with 0 if missing
    all_class_ids = sorted(list(set(val_accuracies.keys()) | set(test_accuracies.keys())))
    
    val_acc_list = [val_accuracies.get(i, 0.0) for i in all_class_ids]
    test_acc_list = [test_accuracies.get(i, 0.0) for i in all_class_ids]
    class_labels = [str(i) for i in all_class_ids] # Labels for x-axis

    index = np.arange(len(class_labels))
    bar_width = 0.35

    plt.figure(figsize=(20, 9))

    # Plotting Validation bars
    plt.bar(index - bar_width/2, val_acc_list, bar_width, 
            label=f'Validation Set (Overall Accuracy: {val_overall_acc:.2f}%)', 
            color='#63B8FF') # Blue shade

    # Plotting Test bars
    plt.bar(index + bar_width/2, test_acc_list, bar_width, 
            label=f'Test Set (Overall Accuracy: {test_overall_acc:.2f}%)', 
            color='#FF6347') # Red shade

    plt.xlabel('Class', fontsize=14)
    plt.ylabel('Accuracy (%)', fontsize=14)
    plt.title('Per-Class Accuracy: Validation vs. Test Set', fontsize=16)
    plt.xticks(index, class_labels, rotation=90, fontsize=10)
    plt.yticks(np.arange(0, 101, 10), fontsize=10)
    plt.ylim(0, 100)
    plt.legend(fontsize=12)
    plt.grid(axis='y', linestyle=':', alpha=0.6)
    plt.tight_layout()
    plt.show()
    plt.savefig("Hist.jpg")


model = models.resnet18(weights="IMAGENET1K_V1")
in_feat = model.fc.in_features
model.fc = nn.Linear(in_feat, 47)

# train_csv = r"../Dataset_Divison/train_set.csv"
val_csv = r"../Dataset_Divison/test_set.csv"
test_csv = r"../Dataset_Divison/val_set.csv"
IMG_SIZE = 224
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor()
])

# train_ds = YogaCSVLoader(train_csv, transform=transform)
val_ds = YogaCSVLoader(val_csv, transform=transform)
test_ds = YogaCSVLoader(test_csv, transform=transform)

#train_dl = DataLoader(train_ds, batch_size=32, shuffle=True)
val_dl = DataLoader(val_ds, batch_size=32, shuffle=False)
test_dl = DataLoader(test_ds, batch_size=32, shuffle=False)

num_classes = len(pd.concat([test_ds.df])["label_idx"].unique())
# Evaluate
test_per_class_accuracies, test_overall_accuracy = evaluate_per_class(model,
                   weights_pth="best_half_fine_tune.pth",
                   test_dl=test_dl)

val_per_class_accuracies, val_overall_accuracy = evaluate_per_class(model,
                   weights_pth="best_half_fine_tune.pth",
                   test_dl=val_dl)

# Generate and display the plot
plot_per_class_accuracies(
    val_accuracies=val_per_class_accuracies,
    val_overall_acc=val_overall_accuracy,
    test_accuracies=test_per_class_accuracies,
    test_overall_acc=test_overall_accuracy
)
