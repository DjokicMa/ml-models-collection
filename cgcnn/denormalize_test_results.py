#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  8 13:51:48 2025

@author: marcus
"""

import pandas as pd

# === Load normalization parameters from id_prop_historical.csv ===
history_df = pd.read_csv('id_prop_historical.csv')

# Get the mean and std from the last rows (where we saved them)
norm_mean = history_df.loc[history_df['material_id'] == '__norm_mean__', 'target'].values[0]
norm_std = history_df.loc[history_df['material_id'] == '__norm_std__', 'target'].values[0]

# === Load test results CSV ===
# It has no headers, so assign them manually
test_df = pd.read_csv('test_results.csv', names=['material_id', 'real_target_norm', 'predicted_target_norm'])

# === Denormalize both real and predicted targets ===
test_df['real_target'] = test_df['real_target_norm'] * norm_std + norm_mean
test_df['predicted_target'] = test_df['predicted_target_norm'] * norm_std + norm_mean

# === Save denormalized results (headless) ===
test_df[['material_id', 'real_target', 'predicted_target']].to_csv(
    'denormalized_test_results.csv', index=False, header=False
)
