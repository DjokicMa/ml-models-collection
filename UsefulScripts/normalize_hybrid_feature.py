#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  8 14:01:24 2025

@author: marcus
"""

import pandas as pd
from sklearn.preprocessing import StandardScaler

# === Load the structure descriptors CSV ===
input_path = '../datasets/2D_HSE/structure_descriptors.csv'
df = pd.read_csv(input_path)

# === Separate material_id and features ===
material_ids = df['material_id']
features = df.drop(columns=['material_id'])

# === Normalize features using StandardScaler (zero mean, unit variance) ===
scaler = StandardScaler()
features_normalized = scaler.fit_transform(features)

# === Create DataFrame with same headers ===
df_normalized = pd.DataFrame(features_normalized, columns=features.columns)

# Add back the material_id column (as first column)
df_normalized.insert(0, 'material_id', material_ids)

# === Save to new CSV (normalized, keeping headers) ===
output_path = '../datasets/2D_HSE/structure_descriptors_normalized.csv'
df_normalized.to_csv(output_path, index=False)
