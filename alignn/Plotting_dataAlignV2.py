#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plot ALIGNN predictions vs targets with consistent style
"""

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
        if "target" in df.columns and "prediction" in df.columns:
            targets = df["target"].values
            predictions = df["prediction"].values
        else:
            # Assume no header, columns are: id, target, prediction
            df = pd.read_csv(
                data_file, header=None, names=["id", "target", "prediction"]
            )
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
            # Handle different JSON structures
            if "target_out" in entry:
                # Each entry might have multiple outputs - use the mean
                target_val = entry["target_out"]
                pred_val = entry["pred_out"]

                # Handle single values or lists
                if isinstance(target_val, list):
                    target_mean = np.mean(target_val)
                    pred_mean = np.mean(pred_val)
                else:
                    target_mean = target_val
                    pred_mean = pred_val

                targets.append(target_mean)
                predictions.append(pred_mean)

        targets = np.array(targets)
        predictions = np.array(predictions)

    # Ensure numeric types
    targets = pd.to_numeric(targets, errors="raise")
    predictions = pd.to_numeric(predictions, errors="raise")

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

    # Create y=x line over full range
    x_line = np.linspace(0, 8, 100)  # Match the 0-8 range from your style

    # Create figure with high DPI
    plt.figure(dpi=300)

    # Hexbin scatter
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
    plt.xlabel("DFT Band-Gap (eV)")
    plt.ylabel("ML Predicted Band-Gap (eV)")

    # Set axis limits to match your style
    plt.xlim(0, 8)
    plt.ylim(0, 8)

    # Ensure square aspect ratio
    plt.gca().set_aspect("equal", "box")

    plt.legend(loc="lower right")
    plt.tight_layout()

    # Save files
    plt.savefig(f"{data_type}_results_alignn.png", dpi=300)
    plt.savefig(f"{data_type}_results_alignn.svg")
    plt.close()

    print(
        f"Plots saved: {data_type}_results_alignn.png, {data_type}_results_alignn.svg\n"
    )


# Main execution
if __name__ == "__main__":
    # Plot validation results from JSON
    print("Processing validation set from JSON...")
    plot_predictions("Val_results.json", "val", use_csv=False)

    # Plot test results from JSON
    print("Processing test set from JSON...")
    plot_predictions("Test_results.json", "test", use_csv=False)

    # Plot train results from JSON
    print("Processing training set from JSON...")
    plot_predictions("Train_results.json", "train", use_csv=False)

    # If CSV files exist, plot them too
    try:
        print("Processing test set from CSV...")
        plot_predictions("prediction_results_test_set.csv", "test_csv", use_csv=True)
    except:
        print("Test CSV not found, skipping...")

    try:
        print("Processing train set from CSV...")
        plot_predictions("prediction_results_train_set.csv", "train_csv", use_csv=True)
    except:
        print("Train CSV not found, skipping...")
