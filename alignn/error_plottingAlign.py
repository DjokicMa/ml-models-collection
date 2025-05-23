import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Set constants
BATCH_SIZE = 16
NUM_TRAIN_SAMPLES = 8648  # Replace with your actual number
NUM_VAL_SAMPLES = 1081  # Replace with your actual number

# Load history files
with open("history_train.json", "r") as f:
    train_history = json.load(f)
with open("history_val.json", "r") as f:
    val_history = json.load(f)

# Extract total losses per epoch
train_total_losses = [entry[0] for entry in train_history]
val_total_losses = [entry[0] for entry in val_history]

# Normalize losses to per-sample average
train_losses = np.array(train_total_losses) / NUM_TRAIN_SAMPLES
val_losses = np.array(val_total_losses) / NUM_VAL_SAMPLES

# Create epochs array
epochs = np.arange(len(train_losses))

# Save as CSV
df = pd.DataFrame(
    {
        "Epoch": epochs,
        "Train Loss (MSE)": train_losses,
        "Validation Loss (MSE)": val_losses,
        "Train MAE": np.sqrt(train_losses),
        "Validation MAE": np.sqrt(val_losses),
    }
)
df.to_csv("training_summary_alignn.csv", index=False)

# Plot Loss
plt.figure(figsize=(8, 6))
plt.plot(epochs, train_losses, label="Train", linewidth=2)
plt.plot(epochs, val_losses, label="Validation", linewidth=2)
plt.xlabel("Epoch")
plt.ylabel("Loss (MSE)")
plt.title("Training vs Validation Loss - ALIGNN")
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig("loss_alignn.png", dpi=300, bbox_inches="tight")
plt.savefig("loss_alignn.svg", bbox_inches="tight")
plt.close()

# Plot MAE (sqrt of MSE)
train_mae = np.sqrt(train_losses)
val_mae = np.sqrt(val_losses)

plt.figure(figsize=(8, 6))
plt.plot(epochs, train_mae, label="Train", linewidth=2)
plt.plot(epochs, val_mae, label="Validation", linewidth=2)
plt.xlabel("Epoch")
plt.ylabel("MAE")
plt.title("Training vs Validation MAE - ALIGNN")
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig("mae_alignn.png", dpi=300, bbox_inches="tight")
plt.savefig("mae_alignn.svg", bbox_inches="tight")
plt.close()

# Print final epoch summary
print(f"Final epoch {len(train_losses) - 1}:")
print(f"  Train Loss = {train_losses[-1]:.6f}, MAE = {train_mae[-1]:.6f}")
print(f"  Val Loss = {val_losses[-1]:.6f}, MAE = {val_mae[-1]:.6f}")
print("\nPlots saved: loss_alignn.png, mae_alignn.png")
