# ALIGNN Usage Guide

This guide explains how to use the ALIGNN (Atomistic Line Graph Neural Network) implementation for materials property prediction. Two versions are available: the standard implementation and an enhanced version with LMDB caching and improved reporting.

## Environment Setup

Activate the alignn_exact environment:
```bash
conda activate alignn_exact
```

## Installation Requirements

### Basic Dependencies
```bash
conda create -n alignn_env python=3.8
conda activate alignn_env
conda install -c conda-forge pytorch cudatoolkit
conda install -c conda-forge dgl-cuda
pip install jarvis-tools
pip install ase
pip install torch-geometric
pip install tqdm  # For progress bars (V2 version)
```

## Data Preparation

ALIGNN requires your dataset to be organized with:
- A collection of structure files (CIF format recommended)
- An `id_prop.csv` file that maps structure filenames to target properties

Example `id_prop.csv`:
```csv
structure_001.cif,1.234
structure_002.cif,2.345
structure_003.cif,3.456
```

## Configuration Setup

Create a configuration JSON file (e.g., `run1.json`) with your model parameters:

```json
{
  "batch_size": 32,
  "epochs": 300,
  "learning_rate": 0.001,
  "cutoff": 8.0,
  "atom_features": "cgcnn",
  "neighbor_strategy": "k-nearest",
  "max_neighbors": 12,
  "keep_data_order": true,
  "output_dir": "/path/to/output",
  "train_ratio": 0.8,
  "val_ratio": 0.1,
  "test_ratio": 0.1,
  "pin_memory": true,
  "num_workers": 8,
  "use_lmdb": true,
  "write_predictions": true,
  "save_dataloader": true,
  "model": {
    "name": "alignn",
    "alignn_layers": 4,
    "gcn_layers": 4,
    "atom_embedding_size": 64,
    "edge_embedding_size": 64,
    "triplet_embedding_size": 64,
    "embedding_features": 256,
    "hidden_features": 256,
    "output_features": 1
  }
}
```

## Standard ALIGNN (`train_alignn.py`)

### Basic Training

```bash
CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python train_alignn.py \
  --root_dir "/path/to/dataset/relaxed_structures_hse_bg/" \
  --config_name "/path/to/runs/run1.json" \
  --file_format "cif" \
  --batch_size 32 \
  --output_dir "/path/to/runs/run1" >> align-log.out
```

### Multi-GPU Training

For multi-GPU training, remove the `CUDA_VISIBLE_DEVICES=0` flag:

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python train_alignn.py \
  --root_dir "/path/to/dataset/" \
  --config_name "/path/to/config.json" \
  --file_format "cif" \
  --batch_size 64 \
  --output_dir "/path/to/output/" >> align-log-multigpu.out
```

### Model Restarting

To continue training from a checkpoint:

```bash
CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python train_alignn.py \
  --root_dir "/path/to/dataset/" \
  --config_name "/path/to/config.json" \
  --file_format "cif" \
  --batch_size 32 \
  --restart_model_path "/path/to/best_model.pt" \
  --output_dir "/path/to/continued_run/" >> align-log-continued.out
```

## Enhanced ALIGNN V2 (`train_alignnV2.py`)

The V2 version includes LMDB cache management, MAE reporting, and normalized loss tracking.

### New Features in V2
- **LMDB Cache Management**: Create cache once, reuse for multiple experiments
- **MAE Reporting**: Track Mean Absolute Error for all outputs at each epoch
- **Normalized Loss**: Scale losses by dataset size for fair comparison
- **Progress Tracking**: Real-time progress bars with detailed metrics

### Step 1: Create LMDB Cache (One-time Setup)

```bash
CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python train_alignnV2.py \
  --root_dir "/path/to/dataset/relaxed_structures_hse_bg/" \
  --config_name "/path/to/config.json" \
  --file_format "cif" \
  --batch_size 16 \
  --output_dir "/path/to/cache_dir" \
  --dry_run
```

This will:
- Process all structure files and create LMDB cache
- Display dataset statistics
- Exit without training
- Save cache metadata for verification

### Step 2: Train with Enhanced Features

```bash
CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python train_alignnV2.py \
  --root_dir "/path/to/dataset/relaxed_structures_hse_bg/" \
  --config_name "/path/to/config.json" \
  --file_format "cif" \
  --batch_size 16 \
  --output_dir "/path/to/runs/run1V2/" \
  --lmdb_cache_path "/path/to/cache_dir/" \
  --report_mae \
  --normalize_loss \
  --skip_cache_verification >> alignV2-log.out
```

### Step 3: Run Multiple Experiments with Same Cache

You can now run different configurations without recreating the cache:

```bash
# Experiment 1: Different batch size
CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python train_alignnV2.py \
  --root_dir "/path/to/dataset/" \
  --config_name "/path/to/config.json" \
  --file_format "cif" \
  --batch_size 32 \
  --output_dir "/path/to/runs/run_bs32" \
  --lmdb_cache_path "/path/to/cache_dir/" \
  --report_mae \
  --normalize_loss >> align-log-bs32.out

# Experiment 2: Different learning rate
CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python train_alignnV2.py \
  --root_dir "/path/to/dataset/" \
  --config_name "/path/to/config_lr001.json" \
  --file_format "cif" \
  --batch_size 16 \
  --output_dir "/path/to/runs/run_lr001" \
  --lmdb_cache_path "/path/to/cache_dir/" \
  --report_mae >> align-log-lr001.out
```

## Advanced Features

### Force and Stress Prediction

To train with forces and stresses, modify your configuration file:

```json
{
  "model": {
    "name": "alignn_atomwise",
    "calculate_gradient": true,
    "gradwise_weight": 1.0,
    "stresswise_weight": 0.1,
    "atomwise_output_features": 3
  }
}
```

And provide the key names in your command:

```bash
CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python train_alignnV2.py \
  --root_dir "/path/to/dataset/" \
  --config_name "/path/to/config_forces.json" \
  --file_format "cif" \
  --batch_size 8 \
  --target_key "total_energy" \
  --force_key "forces" \
  --stresswise_key "stresses" \
  --output_dir "/path/to/runs/forces_run" \
  --report_mae \
  --normalize_loss >> align-log-forces.out
```

## Output Files and Analysis

### Standard Version Output
- `best_model.pt`: The trained model with best validation performance
- `checkpoint/`: Directory containing training checkpoints
- `train_stats.json`: Training statistics including loss and metrics over time
- `val_stats.json`: Validation statistics
- `test_stats.json`: Test set performance metrics
- `config.json`: Copy of the configuration used
- `preds.csv`: Predictions on the test set

### Enhanced V2 Version Output
Additional files in V2:
- `history_train_mae.json`: MAE values for each component during training
- `history_val_mae.json`: MAE values for each component during validation
- `test_mae.json`: Final test set MAE for all components
- `cache_metadata.pkl`: Metadata about the LMDB cache

### Enhanced Console Output (V2)
```
================================================================================
Epoch 25 Summary:
================================================================================
Time - Train: 45.23s, Val: 12.45s

Loss (normalized):
  Total    - Train: 0.0234, Val: 0.0198
  Graph    - Train: 0.0156, Val: 0.0134
  Gradient - Train: 0.0078, Val: 0.0064

MAE:
  graph      - Train: 0.0891, Val: 0.0823
  gradient   - Train: 0.0234, Val: 0.0198

Saving best model
================================================================================
```

## Analyzing Results

### Basic Analysis
```python
import json
import matplotlib.pyplot as plt
import numpy as np

run_dir = '/path/to/runs/run1'

# Load training stats
with open(f'{run_dir}/train_stats.json', 'r') as f:
    train_stats = json.load(f)

with open(f'{run_dir}/val_stats.json', 'r') as f:
    val_stats = json.load(f)

# Plot training and validation loss
plt.figure(figsize=(10, 6))
plt.plot(train_stats['loss'], label='Training Loss')
plt.plot(val_stats['loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.savefig('loss_curve.png')
```

### Enhanced Analysis (V2)
```python
# Load MAE history (V2 specific)
with open(f'{run_dir}/history_train_mae.json', 'r') as f:
    train_mae = json.load(f)

with open(f'{run_dir}/history_val_mae.json', 'r') as f:
    val_mae = json.load(f)

# Plot training curves
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

# Loss plot
ax1.plot([h[0] for h in train_history], label='Train Loss')
ax1.plot([h[0] for h in val_history], label='Val Loss')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss')
ax1.set_title('Training and Validation Loss')
ax1.legend()

# MAE plot (V2 specific)
ax2.plot([h['graph'] for h in train_mae if h['graph'] is not None], label='Train MAE')
ax2.plot([h['graph'] for h in val_mae if h['graph'] is not None], label='Val MAE')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('MAE')
ax2.set_title('Mean Absolute Error')
ax2.legend()

plt.tight_layout()
plt.savefig(f'{run_dir}/training_curves.png')
```

## Memory Optimization

The script automatically limits GPU memory usage to 50% by default. To adjust:

### Modify Memory Fraction
```python
torch.cuda.set_per_process_memory_fraction(0.7, 0)  # Use 70% instead
```

### Use Gradient Checkpointing
Add to config for very large models:
```json
{
  "model": {
    "use_gradient_checkpointing": true
  }
}
```

### Mixed Precision Training
```json
{
  "dtype": "float16"
}
```

## When to Use Which Version

### Use Standard (`train_alignn.py`) when:
- Simple, straightforward training
- Don't need detailed MAE tracking
- Working with smaller datasets
- Want minimal complexity

### Use Enhanced V2 (`train_alignnV2.py`) when:
- Need detailed performance metrics (MAE)
- Running multiple experiments on same dataset
- Want optimized caching for faster data loading
- Need normalized loss for fair comparison
- Working with large datasets

## Troubleshooting

### Cache Verification Failed
If you see "Cache mismatch" errors:
- Delete the old cache directory
- Create a new cache with `--dry_run`
- Ensure your dataset hasn't changed between cache creation and usage

### Out of Memory Errors
- Reduce batch size
- Enable gradient accumulation
- Use mixed precision training
- Enable gradient checkpointing

### Slow Data Loading
- Increase `num_workers` in config
- Ensure LMDB cache is on fast storage (SSD preferred)
- Use `pin_memory: true` for GPU training

## Example Workflow for Hyperparameter Optimization

```bash
# 1. Create cache once
python train_alignnV2.py --config run2.json --dry_run --output_dir cache/

# 2. Try different learning rates
for lr in 0.001 0.0005 0.0001; do
    python train_alignnV2.py --config run2_lr${lr}.json \
        --lmdb_cache_path cache/ \
        --output_dir runs/lr_${lr} \
        --report_mae >> logs/lr_${lr}.log
done

# 3. Try different architectures
for layers in 2 4 6; do
    python train_alignnV2.py --config run2_layers${layers}.json \
        --lmdb_cache_path cache/ \
        --output_dir runs/layers_${layers} \
        --report_mae >> logs/layers_${layers}.log
done
```

## Best Practices

- **Always create cache first**: Use `--dry_run` before starting experiments
- **Monitor MAE trends**: MAE is often more interpretable than loss
- **Use normalized loss**: Especially important with imbalanced train/val/test splits
- **Save configurations**: Keep different JSON configs for each experiment
- **Log everything**: Use descriptive log file names for each run