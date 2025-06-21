import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import optuna
from optuna.visualization import plot_param_importances
import json

# Load CSV
df = pd.read_csv("optuna_yoga_keypoints.csv")
print("📄 Loaded 'optuna_yoga_keypoints.csv'")

# Rename for consistency
df_clean = df.rename(columns={
    "value": "Accuracy",
    "params_hidden_dim": "Hidden Dim",
    "params_dropout": "Dropout",
    "params_lr": "Learning Rate"
})

# Keep only relevant columns
plot_cols = ["number", "Accuracy", "Hidden Dim", "Dropout", "Learning Rate"]
df_clean = df_clean[plot_cols]

# ----------------------------------------------
# Plot 1: Accuracy vs Hidden Dim
# ----------------------------------------------
plt.figure(figsize=(8, 5))
sns.boxplot(x="Hidden Dim", y="Accuracy", data=df_clean)
plt.title("Accuracy vs Hidden Dimension")
plt.show()

# ----------------------------------------------
# Plot 2: Accuracy vs Dropout
# ----------------------------------------------
plt.figure(figsize=(8, 5))
sns.scatterplot(x="Dropout", y="Accuracy", data=df_clean)
plt.title("Accuracy vs Dropout")
plt.show()

# ----------------------------------------------
# Plot 3: Accuracy vs Learning Rate
# ----------------------------------------------
plt.figure(figsize=(8, 5))
sns.scatterplot(x="Learning Rate", y="Accuracy", data=df_clean)
plt.xscale("log")
plt.title("Accuracy vs Learning Rate")
plt.show()

# ----------------------------------------------
# Plot 4: Accuracy per Trial
# ----------------------------------------------
plt.figure(figsize=(8, 5))
sns.lineplot(x="number", y="Accuracy", data=df_clean, marker="o")
plt.title("Accuracy per Trial")
plt.xlabel("Trial Number")
plt.ylabel("Accuracy")
plt.show()

# ----------------------------------------------
# Plot 5: Correlation Heatmap
# ----------------------------------------------
corr_matrix = df_clean.drop(columns="number").corr()
plt.figure(figsize=(8, 6))
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation between Hyperparameters and Accuracy")
plt.show()

# ----------------------------------------------
# Plot 6: Parameter Importances
# ----------------------------------------------
# Load the study from the local SQLite DB
study = optuna.load_study(study_name="yoga_keypoints_hpo", storage="sqlite:///optuna_yoga_keypoints.db")
fig = plot_param_importances(study)
fig.show()

# Plot loss history -------------------------------------------------
with open("best_trial_history.json", "r") as f:
    best = json.load(f)

history = best["history"]
best_epoch = best["best_epoch"]

# Plotting
plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.plot(history["train_loss"], label="train")
plt.plot(history["test_loss"], label="val")
plt.title("Loss")
plt.legend()
plt.xlabel("epoch")

plt.subplot(1,2,2)
plt.plot(history["train_accuracy"], label="train")
plt.plot(history["test_accuracy"], label="val")
plt.title("Accuracy")
plt.legend()
plt.xlabel("epoch")

plt.suptitle(f"Best trial #{best['trial_number']} | Best epoch: {best_epoch}")
plt.tight_layout()
plt.show()