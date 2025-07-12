# evaluate_test_set.py
import argparse
import torch
import pandas as pd
import numpy as np
from torchmetrics.classification import MulticlassF1Score, MulticlassAveragePrecision
from Pose_Keypoints.YogaClassifier import YogaClassifier
from Transformer.TwoTokenTransformer import TwoTokenTransformer

# === Helper: Load model and weights ===
def load_model(model_type, model_path, num_classes, device):
    if model_type == "keypoints":
        # TODO: update with actual best HP values
        model = YogaClassifier(
            input_dim=34,  # 17 keypoints * 2 (x, y)
            hidden_dim=128,
            num_layers=2,
            num_classes=num_classes,
            dropout=0.3,
            norm_type='batch',
        )
    elif model_type == "transformer":
        model = TwoTokenTransformer(
            input_dim=586,  # 34 keypoints + 552 CNN features
            d_model=128,
            nhead=4,
            depth=4,
            dim_feedforward=256,
            dropout=0.2,
            num_classes=num_classes
        )
    elif model_type == "cnn":
        raise NotImplementedError("CNN model evaluation will be added later.")
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

# === Helper: Load test set ===
def load_test_dataloader(model_type, batch_size):
    from torch.utils.data import DataLoader
    from Pose_Keypoints.train_loop import CustomKeypointsDataset
    from Transformer.transformer_train import TransformerDataset

    if model_type == "keypoints":
        df = pd.read_csv("Pose_Keypoints/yolo_keypoints_dataset.csv")
        from sklearn.model_selection import train_test_split
        _, test_df = train_test_split(df, test_size=0.1, random_state=42, stratify=df["label_idx"])
        test_dataset = CustomKeypointsDataset(test_df)
        return DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    elif model_type == "transformer":
        test_dataset = TransformerDataset("Transformer/test_set.csv")
        return DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    else:
        raise ValueError(f"No dataloader available for model type: {model_type}")

# === Evaluate ===
def evaluate_model(model, test_loader, num_classes, device):
    f1_metric = MulticlassF1Score(num_classes=num_classes, average="macro").to(device)
    map_metric = MulticlassAveragePrecision(num_classes=num_classes, average="macro").to(device)

    y_true_all = []
    y_probs_all = []

    with torch.no_grad():
        for batch in test_loader:
            X = batch["x"].to(device)
            y = batch["y"].to(device)

            logits = model(X)
            probs = torch.softmax(logits, dim=1)

            f1_metric.update(probs, y)
            map_metric.update(probs, y)

            y_probs_all.append(probs.cpu().numpy())
            y_true_all.append(y.cpu().numpy())

    y_true = np.concatenate(y_true_all)
    y_probs = np.concatenate(y_probs_all)

    f1 = f1_metric.compute().item()
    mAP = map_metric.compute().item()

    return y_true, y_probs, f1, mAP

# === Save predictions for plotting ===
def save_preds_npz(y_true, y_probs, out_path):
    np.savez(out_path, y_true=y_true, y_probs=y_probs)
    print(f"✅ Saved predictions to: {out_path}")

# === Update training history CSV ===
def update_history_csv(history_csv, f1, mAP):
    df = pd.read_csv(history_csv)
    df.at[len(df) - 1, "test_f1"] = f1
    df.at[len(df) - 1, "test_map"] = mAP
    df.to_csv(history_csv, index=False)
    print(f"📈 Updated test metrics in {history_csv}")

# === Main Entry ===
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", type=str, required=True, choices=["keypoints", "transformer", "cnn"])
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--history_csv", type=str, required=True)
    parser.add_argument("--output_npz", type=str, required=True)
    parser.add_argument("--num_classes", type=int, default=6)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    model = load_model(args.model_type, args.model_path, args.num_classes, args.device)
    test_loader = load_test_dataloader(args.model_type, args.batch_size)

    y_true, y_probs, test_f1, test_map = evaluate_model(model, test_loader, args.num_classes, args.device)

    print(f"\n✅ {args.model_type.upper()} Test Set Results:")
    print(f"   F1 Score = {test_f1:.3f}")
    print(f"   mAP      = {test_map:.3f}\n")

    save_preds_npz(y_true, y_probs, args.output_npz)
    update_history_csv(args.history_csv, test_f1, test_map)
