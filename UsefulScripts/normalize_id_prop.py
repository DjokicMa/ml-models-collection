#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  8 13:44:39 2025

@author: marcus
"""

import pandas as pd

# === Load your original id_prop.csv ===
# It must have headers: 'material_id' and 'target'
df = pd.read_csv('../datasets/relaxed_structures_hse_bg/id_prop.csv', names=['material_id', 'target'])

# === Compute normalization parameters (z-score normalization) ===
mean_val = df['target'].mean()
std_val = df['target'].std()

# === Apply normalization ===
df['target_normalized'] = (df['target'] - mean_val) / std_val

# === Save historical CSV (raw + normalized + normalization params) ===
# Append normalization details as extra rows
history_df = df.copy()

# Add a dummy row for saving normalization params (optional but helpful)
norm_details = pd.DataFrame({
    'material_id': ['__norm_mean__', '__norm_std__'],
    'target': [mean_val, std_val],
    'target_normalized': [None, None]
})

# Append details
history_df = pd.concat([history_df, norm_details], ignore_index=True)

# Save
history_df.to_csv('id_prop_historical.csv', index=False)

# === Save CGCNN-ready id_prop.csv (normalized, 2 columns, no headers) ===
df[['material_id', 'target_normalized']].to_csv('id_prop_normalized.csv', index=False, header=False)
