#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Comprehensive plotting script for ALIGNN outputs
Handles both training curves and prediction scatter plots
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import os


def plot_training_curves():
    """Plot training and validation loss/MAE curves"""

    # Load history files
    with open("history_train.json", "r") as f:
        train_data = json.load(f)

    with open("history_val.json", "r") as f:
        val_data = json.load(f)

    # Extract data
    train_losses = train_data["loss"]
    train_mae = train_data["mae"]
    val_losses = val_data["loss"]
    val_mae = val_data["mae"]

    # Create epochs array
    epochs = np.arange(len(train_losses))

    # Find best epoch (minimum validation MAE)
    best_epoch = np.argmin(val_mae)
    best_val_mae = val_mae[best_epoch]

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
    plt.figure(figsize=(10, 6), dpi=300)
    plt.plot(epochs, train_losses, label="Train", linewidth=2, color="blue")
    plt.plot(epochs, val_losses, label="Validation", linewidth=2, color="orange")
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Loss", fontsize=12)
    plt.title("Training vs Validation Loss - ALIGNN", fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("loss_alignn.png", dpi=300)
    plt.savefig("loss_alignn.svg")
    plt.close()

    # Plot MAE
    plt.figure(figsize=(10, 6), dpi=300)
    plt.plot(epochs, train_mae, label="Train", linewidth=2, color="blue")
    plt.plot(epochs, val_mae, label="Validation", linewidth=2, color="orange")

    # Add best epoch marker
    plt.scatter(
        best_epoch,
        best_val_mae,
        color="red",
        s=100,
        zorder=5,
        label=f"Best Val MAE = {best_val_mae:.4f}",
    )

    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("MAE (eV)", fontsize=12)
    plt.title("Training vs Validation MAE - ALIGNN", fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.ylim(bottom=0)
    plt.tight_layout()
    plt.savefig("mae_alignn.png", dpi=300)
    plt.savefig("mae_alignn.svg")
    plt.close()

    # Create zoomed MAE plot (after epoch 20)
    if len(epochs) > 20:
        plt.figure(figsize=(10, 6), dpi=300)
        epochs_zoom = epochs[20:]
        train_mae_zoom = train_mae[20:]
        val_mae_zoom = val_mae[20:]

        plt.plot(epochs_zoom, train_mae_zoom, label="Train", linewidth=2, color="blue")
        plt.plot(
            epochs_zoom, val_mae_zoom, label="Validation", linewidth=2, color="orange"
        )

        # Add best epoch marker if it's in the zoomed range
        if best_epoch >= 20:
            plt.scatter(
                best_epoch,
                best_val_mae,
                color="red",
                s=100,
                zorder=5,
                label=f"Best Val MAE = {best_val_mae:.4f}",
            )

        plt.xlabel("Epoch", fontsize=12)
        plt.ylabel("MAE (eV)", fontsize=12)
        plt.title("Training vs Validation MAE - ALIGNN (Zoomed)", fontsize=14)
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig("mae_alignn_zoomed.png", dpi=300)
        plt.savefig("mae_alignn_zoomed.svg")
        plt.close()

    # Print summary
    print("Training Summary:")
    print("=" * 50)
    print(f"Total epochs: {len(train_losses)}")
    print(f"\nFinal epoch {len(train_losses) - 1}:")
    print(f"  Train Loss = {train_losses[-1]:.4f}, MAE = {train_mae[-1]:.4f}")
    print(f"  Val Loss = {val_losses[-1]:.4f}, MAE = {val_mae[-1]:.4f}")
    print(f"\nBest epoch (lowest val MAE): {best_epoch}")
    print(
        f"  Train Loss = {train_losses[best_epoch]:.4f}, MAE = {train_mae[best_epoch]:.4f}"
    )
    print(f"  Val Loss = {val_losses[best_epoch]:.4f}, MAE = {val_mae[best_epoch]:.4f}")
    print("\nTraining curves saved!")
    print("-" * 50)


def plot_predictions_scatter(file_path, output_name, is_csv=False, has_header=True):
    """
    Plot predictions vs targets scatter plot

    Args:
        file_path: path to results file (JSON or CSV)
        output_name: base name for output files
        is_csv: True if CSV file, False if JSON
        has_header: True if CSV has header row
    """

    if is_csv:
        if has_header:
            df = pd.read_csv(file_path)
            # Check column names
            if "target" in df.columns and "prediction" in df.columns:
                targets = df["target"].values
                predictions = df["prediction"].values
            elif len(df.columns) == 2:
                # Assume first is target, second is prediction
                targets = df.iloc[:, 0].values
                predictions = df.iloc[:, 1].values
            elif len(df.columns) == 3:
                # Assume columns are: id, target, prediction
                targets = df.iloc[:, 1].values
                predictions = df.iloc[:, 2].values
        else:
            # No header
            df = pd.read_csv(file_path, header=None)
            if len(df.columns) == 2:
                targets = df.iloc[:, 0].values
                predictions = df.iloc[:, 1].values
            elif len(df.columns) == 3:
                targets = df.iloc[:, 1].values
                predictions = df.iloc[:, 2].values
    else:
        # Load JSON
        with open(file_path, "r") as f:
            data = json.load(f)

        targets = []
        predictions = []

        for entry in data:
            if "target_out" in entry and "pred_out" in entry:
                target_val = entry["target_out"]
                pred_val = entry["pred_out"]

                # Handle single values or lists
                if isinstance(target_val, list):
                    targets.extend(target_val)
                    predictions.extend(pred_val)
                else:
                    targets.append(target_val)
                    predictions.append(pred_val)

    # Convert to numpy arrays
    targets = np.array(targets, dtype=float)
    predictions = np.array(predictions, dtype=float)

    # Calculate metrics
    r2 = r2_score(targets, predictions)
    rmse = np.sqrt(mean_squared_error(targets, predictions))
    mae = mean_absolute_error(targets, predictions)

    print(f"\n{output_name} metrics:")
    print(f"  R²   = {r2:.4f}")
    print(f"  RMSE = {rmse:.4f}")
    print(f"  MAE  = {mae:.4f}")

    # Create figure
    plt.figure(dpi=300)

    # Hexbin scatter
    hb = plt.hexbin(targets, predictions, gridsize=50, mincnt=1, cmap="plasma")

    # Plot y=x line
    x_line = np.linspace(0, 8, 100)
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

    # Labels and limits
    plt.xlabel("DFT Band-Gap (eV)")
    plt.ylabel("ML Predicted Band-Gap (eV)")
    plt.xlim(0, 8)
    plt.ylim(0, 8)
    plt.gca().set_aspect("equal", "box")
    plt.legend(loc="lower right")
    plt.tight_layout()

    # Save files
    plt.savefig(f"{output_name}.png", dpi=300)
    plt.savefig(f"{output_name}.svg")
    plt.close()

    print(f"Plots saved: {output_name}.png, {output_name}.svg")

    return mae, rmse, r2


def create_summary_table(results_dict):
    """Create a summary table of all results"""

    # Create DataFrame
    df = pd.DataFrame(results_dict).T
    df = df.round(4)

    # Save to CSV
    df.to_csv("alignn_metrics_summary.csv")

    # Print table
    print("\n" + "=" * 60)
    print("ALIGNN Performance Summary")
    print("=" * 60)
    print(df.to_string())
    print("=" * 60)

    # Create bar plot of MAE values
    plt.figure(figsize=(8, 6), dpi=300)
    sets = list(results_dict.keys())
    mae_values = [results_dict[s]["MAE"] for s in sets]

    bars = plt.bar(sets, mae_values, color=["blue", "orange", "green"])

    # Add value labels on bars
    for bar, mae in zip(bars, mae_values):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.005,
            f"{mae:.3f}",
            ha="center",
            va="bottom",
        )

    plt.ylabel("MAE (eV)", fontsize=12)
    plt.title("ALIGNN MAE Comparison Across Sets", fontsize=14)
    plt.ylim(0, max(mae_values) * 1.2)
    plt.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig("mae_comparison.png", dpi=300)
    plt.savefig("mae_comparison.svg")
    plt.close()

    print("\nSummary plots saved!")


# Main execution
if __name__ == "__main__":
    print("ALIGNN Results Plotting Script")
    print("=" * 60)

    # Plot training curves
    try:
        plot_training_curves()
    except Exception as e:
        print(f"Could not plot training curves: {e}")

    # Dictionary to store results
    results_summary = {}

    # Process test set CSV
    if os.path.exists("prediction_results_test_set.csv"):
        print("\nProcessing test set predictions (CSV)...")
        mae, rmse, r2 = plot_predictions_scatter(
            "prediction_results_test_set.csv",
            "test_results_alignn",
            is_csv=True,
            has_header=True,
        )
        results_summary["Test"] = {"MAE": mae, "RMSE": rmse, "R²": r2}

    # Process train set CSV
    if os.path.exists("prediction_results_train_set.csv"):
        print("\nProcessing train set predictions (CSV)...")
        mae, rmse, r2 = plot_predictions_scatter(
            "prediction_results_train_set.csv",
            "train_results_alignn",
            is_csv=True,
            has_header=True,
        )
        results_summary["Train"] = {"MAE": mae, "RMSE": rmse, "R²": r2}

    # Process JSON files if they exist
    json_files = {
        "Test_results.json": "test_results_json",
        "Val_results.json": "val_results_alignn",
        "Train_results.json": "train_results_json",
    }

    for json_file, output_name in json_files.items():
        if os.path.exists(json_file):
            print(f"\nProcessing {json_file}...")
            mae, rmse, r2 = plot_predictions_scatter(
                json_file, output_name, is_csv=False
            )
            set_name = json_file.split("_")[0]
            if set_name not in results_summary:
                results_summary[set_name] = {"MAE": mae, "RMSE": rmse, "R²": r2}

    # Create summary table if we have results
    if results_summary:
        create_summary_table(results_summary)

    print("\nAll plots completed successfully!")
    print("Files created:")
    print("  - Training curves: loss_alignn.png, mae_alignn.png")
    print("  - Scatter plots: *_results_alignn.png")
    print("  - Summary: alignn_metrics_summary.csv, mae_comparison.png")
