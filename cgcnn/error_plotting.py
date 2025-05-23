import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

log_file = 'cgcnn_train_normalized.out'  # Update path

# Regex patterns
train_line     = re.compile(
    r"Epoch: \[(?P<epoch>\d+)\]\[(?P<batch>\d+)/(\d+)\].*?Loss [\d.]+ \((?P<loss_avg>[\d.]+)\).*?MAE [\d.]+ \((?P<mae_avg>[\d.]+)\)"
)
val_line       = re.compile(
    r"Test: \[(?P<batch>\d+)/(\d+)\].*?Loss [\d.]+ \((?P<loss_avg>[\d.]+)\).*?MAE [\d.]+ \((?P<mae_avg>[\d.]+)\)"
)
sep_final_test = re.compile(r"-+Evaluate Model on Test Set-+")
summary_mae    = re.compile(r"\*+ MAE (?P<mae>[\d.]+)")
summary_loss   = re.compile(r"\*+ Loss (?P<loss>[\d.]+)")

# Data containers
train_loss_avgs, train_mae_avgs = [], []
val_loss_avgs,   val_mae_avgs   = [], []
final_test_loss_avgs, final_test_mae_avgs = [], []
validation_summary_mae = None
final_test_summary_mae = None
validation_summary_loss = None
final_test_summary_loss = None

in_first_test = False
in_final_test = False

# Parse log
with open(log_file) as f:
    for line in f:
        # Detect start of first validation
        if line.startswith("Test:") and not in_first_test and not in_final_test:
            in_first_test = True

        # Capture summary MAE/Loss of first validation block
        if in_first_test and summary_mae.search(line) and validation_summary_mae is None:
            validation_summary_mae = float(summary_mae.search(line).group("mae"))
        if in_first_test and summary_loss.search(line) and validation_summary_loss is None:
            validation_summary_loss = float(summary_loss.search(line).group("loss"))

        # Detect final test block
        if sep_final_test.search(line):
            in_final_test = True
            in_first_test = False
            continue

        # Parse final test running averages
        if in_final_test:
            m = val_line.search(line)
            if m:
                batch = int(m.group('batch'))
                total = int(m.group(2))
                if batch == total - 1:
                    final_test_loss_avgs.append(float(m.group('loss_avg')))
                    final_test_mae_avgs.append(float(m.group('mae_avg')))
            # Capture summary markers if present
            if summary_mae.search(line) and final_test_summary_mae is None:
                final_test_summary_mae = float(summary_mae.search(line).group("mae"))
            if summary_loss.search(line) and final_test_summary_loss is None:
                final_test_summary_loss = float(summary_loss.search(line).group("loss"))
            continue

        # Parse training running averages
        m = train_line.search(line)
        if m:
            batch = int(m.group('batch'))
            total = int(m.group(3))
            if batch == total - 1:
                train_loss_avgs.append(float(m.group('loss_avg')))
                train_mae_avgs.append(float(m.group('mae_avg')))
            continue

        # Parse validation running averages for each epoch
        if in_first_test:
            m = val_line.search(line)
            if m:
                batch = int(m.group('batch'))
                total = int(m.group(2))
                if batch == total - 1:
                    val_loss_avgs.append(float(m.group('loss_avg')))
                    val_mae_avgs.append(float(m.group('mae_avg')))
            continue

# Sanity check
if len(train_loss_avgs) != len(val_loss_avgs):
    raise RuntimeError(f"Epoch count mismatch: train={len(train_loss_avgs)}, val={len(val_loss_avgs)}. Please check log parsing.")

# Fallback to running-average if summary missing
if validation_summary_mae is None and val_mae_avgs:
    validation_summary_mae = val_mae_avgs[-1]
if validation_summary_loss is None and val_loss_avgs:
    validation_summary_loss = val_loss_avgs[-1]
if final_test_summary_mae is None and final_test_mae_avgs:
    final_test_summary_mae = final_test_mae_avgs[-1]
if final_test_summary_loss is None and final_test_loss_avgs:
    final_test_summary_loss = final_test_loss_avgs[-1]

# Save running averages
df = pd.DataFrame({
    'Epoch': np.arange(len(train_loss_avgs)),
    'Train Loss': train_loss_avgs,
    'Validation Loss': val_loss_avgs,
    'Train MAE': train_mae_avgs,
    'Validation MAE': val_mae_avgs
})
df.to_csv('training_summary_running.csv', index=False)

# Plot utility
def plot_curve(train_vals, val_vals, ylabel, title, fname, val_summary=None, test_summary=None):
    epochs = np.arange(len(train_vals))
    plt.figure(figsize=(8,6))
    plt.plot(epochs, train_vals, label='Train')
    plt.plot(epochs, val_vals,   label='Validation')
    #if val_summary is not None:
    #    plt.hlines(val_summary, epochs[0], epochs[-1], linestyles='--', label=f'Val summary {ylabel}: {val_summary:.3f}')
    if test_summary is not None:
        plt.hlines(test_summary, epochs[0], epochs[-1], linestyles=':',  label=f'Test summary {ylabel}: {test_summary:.3f}')
    plt.xlabel('Epoch')
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.savefig(f'{fname}.png', dpi=300)
    plt.savefig(f'{fname}.svg')
    plt.close()

# Plot Loss
plot_curve(
    train_loss_avgs, val_loss_avgs,
    ylabel='Loss',
    title='Training vs Validation Loss (running avg)',
    fname='loss_running',
    val_summary=validation_summary_loss,
    test_summary=final_test_summary_loss
)

# Plot MAE
plot_curve(
    train_mae_avgs, val_mae_avgs,
    ylabel='MAE',
    title='Training vs Validation MAE (running avg)',
    fname='mae_running',
    val_summary=validation_summary_mae,
    test_summary=final_test_summary_mae
)

print(f"Done. Validation summary Loss = {validation_summary_loss:.3f}, MAE = {validation_summary_mae:.3f}")
print(f"      Final test summary Loss = {final_test_summary_loss:.3f}, MAE = {final_test_summary_mae:.3f}")
