import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load history files
with open("history_train.json", "r") as f:
    train_history = json.load(f)

with open("history_val.json", "r") as f:
    val_history = json.load(f)

# Load MAE history files
with open("history_train_mae.json", "r") as f:
    train_mae_history = json.load(f)

with open("history_val_mae.json", "r") as f:
    val_mae_history = json.load(f)

# Load test MAE
with open("test_mae.json", "r") as f:
    test_mae_data = json.load(f)
    test_mae = test_mae_data["graph"]

# Extract losses (first element of each entry)
train_losses = [entry[0] for entry in train_history]
val_losses = [entry[0] for entry in val_history]

# Extract MAE values (graph component)
train_mae = [entry["graph"] for entry in train_mae_history]
val_mae = [entry["graph"] for entry in val_mae_history]

# Create epochs array
epochs = np.arange(len(train_losses))

# Create DataFrame for saving
df = pd.DataFrame(
    {
        "Epoch": epochs,
        "Train Loss": train_losses,
        "Validation Loss": val_losses,
        "Train MAE": train_mae,
        "Validation MAE": val_mae,
    }
)
df.to_csv("training_summary_alignn.csv", index=False)

# Plot Loss
plt.figure(figsize=(10, 6))
plt.plot(epochs, train_losses, label="Train", linewidth=2, color="blue")
plt.plot(epochs, val_losses, label="Validation", linewidth=2, color="orange")
plt.xlabel("Epoch", fontsize=12)
plt.ylabel("Loss", fontsize=12)
plt.title("Training vs Validation Loss - ALIGNN", fontsize=14)
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.savefig("loss_alignn.png", dpi=300, bbox_inches="tight")
plt.savefig("loss_alignn.svg", bbox_inches="tight")
plt.close()

# Plot MAE with test MAE as dashed line
plt.figure(figsize=(10, 6))
plt.plot(epochs, train_mae, label="Train", linewidth=2, color="blue")
plt.plot(epochs, val_mae, label="Validation", linewidth=2, color="orange")
plt.axhline(
    y=test_mae,
    color="red",
    linestyle="--",
    linewidth=2,
    label=f"Test MAE = {test_mae:.4f}",
)
plt.xlabel("Epoch", fontsize=12)
plt.ylabel("MAE", fontsize=12)
plt.title("Training vs Validation MAE - ALIGNN", fontsize=14)
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.ylim(bottom=0)  # Start y-axis at 0 for better visualization
plt.savefig("mae_alignn.png", dpi=300, bbox_inches="tight")
plt.savefig("mae_alignn.svg", bbox_inches="tight")
plt.close()

# Find best epoch (minimum validation MAE)
best_epoch = np.argmin(val_mae)
best_val_mae = val_mae[best_epoch]

# Print summary statistics
print(f"Training Summary:")
print(f"================")
print(f"Total epochs: {len(train_losses)}")
print(f"\nFinal epoch {len(train_losses) - 1}:")
print(f"  Train Loss = {train_losses[-1]:.4f}, MAE = {train_mae[-1]:.4f}")
print(f"  Val Loss = {val_losses[-1]:.4f}, MAE = {val_mae[-1]:.4f}")
print(f"\nBest epoch (lowest val MAE): {best_epoch}")
print(
    f"  Train Loss = {train_losses[best_epoch]:.4f}, MAE = {train_mae[best_epoch]:.4f}"
)
print(f"  Val Loss = {val_losses[best_epoch]:.4f}, MAE = {val_mae[best_epoch]:.4f}")
print(f"\nTest MAE = {test_mae:.4f}")
print(f"\nPlots saved: loss_alignn.png, mae_alignn.png")
print(f"Data saved: training_summary_alignn.csv")

# Optional: Create a zoomed-in MAE plot for better visualization
plt.figure(figsize=(10, 6))
# Only plot after epoch 10 for better visualization of convergence
if len(epochs) > 10:
    epochs_zoom = epochs[10:]
    train_mae_zoom = train_mae[10:]
    val_mae_zoom = val_mae[10:]

    plt.plot(epochs_zoom, train_mae_zoom, label="Train", linewidth=2, color="blue")
    plt.plot(epochs_zoom, val_mae_zoom, label="Validation", linewidth=2, color="orange")
    plt.axhline(
        y=test_mae,
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"Test MAE = {test_mae:.4f}",
    )
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("MAE", fontsize=12)
    plt.title("Training vs Validation MAE - ALIGNN (Zoomed)", fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.savefig("mae_alignn_zoomed.png", dpi=300, bbox_inches="tight")
    plt.savefig("mae_alignn_zoomed.svg", bbox_inches="tight")
    plt.close()
    print("Additional plot saved: mae_alignn_zoomed.png")
