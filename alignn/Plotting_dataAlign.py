import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error


def plot_predictions(data_file, data_type="test", use_csv=False):
    """
    Plot predictions vs targets for ALIGNN results

    Args:
        data_file: path to results file (JSON or CSV)
        data_type: 'train', 'val', or 'test' for plot labeling
        use_csv: True if using CSV file, False for JSON
    """

    if use_csv:
        # Load CSV file
        df = pd.read_csv(data_file)
        targets = df["target"].values
        predictions = df["prediction"].values
    else:
        # Load JSON file
        with open(data_file, "r") as f:
            data = json.load(f)

        # Extract targets and predictions
        targets = []
        predictions = []

        for entry in data:
            # Each entry has multiple outputs - we'll use the mean
            target_mean = np.mean(entry["target_out"])
            pred_mean = np.mean(entry["pred_out"])
            targets.append(target_mean)
            predictions.append(pred_mean)

        targets = np.array(targets)
        predictions = np.array(predictions)

    # Calculate metrics
    r2 = r2_score(targets, predictions)
    rmse = np.sqrt(mean_squared_error(targets, predictions))
    mae = mean_absolute_error(targets, predictions)

    print(f"{data_type.capitalize()} set metrics:")
    print(f"  R²   = {r2:.4f}")
    print(f"  RMSE = {rmse:.4f}")
    print(f"  MAE  = {mae:.4f}")

    # Determine plot range
    max_val = max(targets.max(), predictions.max())
    min_val = min(targets.min(), predictions.min())

    # Create y=x line
    x_line = np.linspace(min_val, max_val, 100)

    # Create scatter plot
    plt.figure(figsize=(8, 8), dpi=300)

    # Hexbin plot
    hb = plt.hexbin(targets, predictions, gridsize=50, mincnt=1, cmap="plasma")

    # Plot y=x line
    plt.plot(x_line, x_line, "-", color="green", linewidth=2, label="y = x")

    # Colorbar
    cb = plt.colorbar(hb)
    cb.set_label("Counts")

    # Annotate metrics
    plt.text(
        0.05,
        0.95,
        f"$R^2$ = {r2:.3f}\nRMSE = {rmse:.3f}\nMAE  = {mae:.3f}",
        transform=plt.gca().transAxes,
        va="top",
        ha="left",
        bbox=dict(facecolor="white", alpha=0.8, edgecolor="gray"),
    )

    # Labels and formatting
    plt.xlabel("Target Band Gap (eV)")
    plt.ylabel("Predicted Band Gap (eV)")
    plt.title(f"ALIGNN {data_type.capitalize()} Set Results")

    # Set equal aspect ratio
    plt.gca().set_aspect("equal", "box")

    # Set axis limits with some padding
    padding = (max_val - min_val) * 0.05
    plt.xlim(min_val - padding, max_val + padding)
    plt.ylim(min_val - padding, max_val + padding)

    plt.legend(loc="lower right")
    plt.tight_layout()

    # Save files
    plt.savefig(f"{data_type}_results_alignn.png", dpi=300)
    plt.savefig(f"{data_type}_results_alignn.svg")
    plt.close()

    print(
        f"Plots saved: {data_type}_results_alignn.png, {data_type}_results_alignn.svg\n"
    )


# Plot validation results from JSON
print("Processing validation set from JSON...")
plot_predictions("Val_results.json", "val", use_csv=False)

print("Processing test set from JSON...")
plot_predictions("Test_results.json", "test", use_csv=False)

print("Processing training set from CSV...")
plot_predictions("prediction_results_test_set.csv", "test_csv", use_csv=True)

print("Processing train set from JSON...")
plot_predictions("Train_results.json", "train", use_csv=False)

print("Processing training set from CSV...")
plot_predictions("prediction_results_train_set.csv", "train_csv", use_csv=True)
