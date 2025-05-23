#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Evaluate final test set predictions:
  - R²
  - RMSE
  - MAE
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# Path to your test-results CSV
test_csv = 'denormalized_test_results.csv'

# Load (no header; columns: name, DFT, ML)
df = pd.read_csv(test_csv, header=None, names=['name','DFT','ML'])

# Ensure numeric types
for col in ['DFT', 'ML']:
    df[col] = pd.to_numeric(df[col], errors='raise')

# Compute metrics
r2   = r2_score(df['DFT'], df['ML'])
rmse = np.sqrt(mean_squared_error(df['DFT'], df['ML']))
mae  = mean_absolute_error(df['DFT'], df['ML'])

print(f"Final test metrics:\n"
      f"  R²   = {r2:.4f}\n"
      f"  RMSE = {rmse:.4f}\n"
      f"  MAE  = {mae:.4f}")

# Determine plot range
max_val = max(df['DFT'].max(), df['ML'].max())
min_val = min(df['DFT'].min(), df['ML'].min())

# Create y=x line over full range
x_line = np.linspace(min_val, max_val*1.5, 100)

after = plt.figure(dpi=300)
# Hexbin scatter
hb = plt.hexbin(df['DFT'], df['ML'], gridsize=50, mincnt=1, cmap='plasma')
# Plot y=x line spanning full range
plt.plot(x_line, x_line, '-', color='green', linewidth=2, label='y = x')
# Colorbar
cb = plt.colorbar(hb)
cb.set_label('Counts')

# Annotate metrics
plt.text(0.05, 0.95,
         f"$R^2$ = {r2:.3f}\nRMSE = {rmse:.3f}\nMAE  = {mae:.3f}",
         transform=plt.gca().transAxes,
         va='top', ha='left',
         bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray'))

# Labels and limits
plt.xlabel("DFT Band-Gap (eV)")
plt.ylabel("ML Predicted Band-Gap (eV)")
plt.xlim(0,8)#min_val, max_val)
plt.ylim(0,8)#min_val, max_val)
plt.gca().set_aspect('equal', 'box')  # Ensure square
plt.legend(loc='lower right')
plt.tight_layout()

# Save files
plt.savefig('TestResults.png', dpi=300)
plt.savefig('TestResults.svg')
print('Plots saved: TestResults.png, TestResults.svg')
