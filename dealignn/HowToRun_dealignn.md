# DE-ALIGNN Usage Guide

This guide explains how to use the hybridized ALIGNN (de-ALIGNN) implementation for materials property prediction. De-ALIGNN combines neural representations with human-designed descriptors for enhanced performance.

## Environment Setup

Activate the alignn_exact environment:
```bash
conda activate alignn_exact
```

## ⚠️ Critical Installation Requirement

**REQUIRED**: Install the `degraphs.py` script in your ALIGNN environment:

```bash
cp degraphs.py /home/marcus/anaconda3/envs/alignn_exact/lib/python3.8/site-packages/jarvis/core/
# Or for your specific environment path:
cp degraphs.py /path/to/your/conda/envs/alignn_exact/lib/python3.8/site-packages/jarvis/core/
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
pip install tqdm
```

## ⚠️ Prerequisites: Feature Preprocessing

**IMPORTANT**: De-ALIGNN requires preprocessing to generate structure descriptors before training.

### Step 1: Generate Structure Descriptors
```bash
python id_propFeature.py  # Creates structure_descriptors.csv
```

### Step 2: Normalize Features
```bash
python normalize_hybrid_feature.py  # Creates structure_descriptors_normalized.csv
```

## Data Preparation

De-ALIGNN requires your dataset to be organized with:
- A collection of structure files (CIF format recommended)
- An `id_prop.csv` file that maps structure filenames to target properties
- A `structure_descriptors_normalized.csv` file with human-designed features

Example directory structure:
```
datasets/relaxed_structures_hse_bg/
├── structure_files/          # CIF files
├── id_prop.csv              # Property labels
└── structure_descriptors_normalized.csv  # Required: Normalized features
```

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
  "batch_size": 16,
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
    "output_features": 1,
    "hybridize_features": true,
    "descriptor_dim": 128
  }
}
```

## Basic Training

### Standard Training Command

```bash
CUDA_VISIBLE_DEVICES=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python fine-tuning.py \
  --root_dir "/path/to/dataset/relaxed_structures_hse_bg/" \
  --config_name "/path/to/runs/run1.json" \
  --file_format "cif" \
  --batch_size 16 \
  --output_dir "/path/to/runs/run1" \
  >> dealignn-log.out
```

### Multi-GPU Training

For multi-GPU training, remove the `CUDA_VISIBLE_DEVICES=1` flag:

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python fine-tuning.py \
  --root_dir "/path/to/dataset/" \
  --config_name "/path/to/config.json" \
  --file_format "cif" \
  --batch_size 32 \
  --output_dir "/path/to/output/" \
  >> dealignn-log-multigpu.out
```

### Model Restarting

To continue training from a checkpoint:

```bash
CUDA_VISIBLE_DEVICES=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python fine-tuning.py \
  --root_dir "/path/to/dataset/" \
  --config_name "/path/to/config.json" \
  --file_format "cif" \
  --batch_size 16 \
  --restart_model_path "/path/to/best_model.pt" \
  --output_dir "/path/to/continued_run/" \
  >> dealignn-log-continued.out
```

## Advanced Features

### Code Modifications Required

Before running, you need to:

1. **Edit `fine-tuning.py`**: Specify the feature list and path of the descriptors file at the beginning
2. **Verify descriptor path**: Ensure the path to `structure_descriptors_normalized.csv` is correctly set
3. **Feature selection**: Choose which descriptors to include in the hybridization

Example modification in `fine-tuning.py`:
```python
# At the beginning of fine-tuning.py
DESCRIPTOR_FILE = "/path/to/structure_descriptors_normalized.csv"
SELECTED_FEATURES = [
    'density', 'volume', 'packing_fraction',
    'lattice_a', 'lattice_b', 'lattice_c',
    # Add your selected features here
]
```

### Force and Stress Prediction

To train with forces and stresses, modify your configuration file:

```json
{
  "model": {
    "name": "alignn_atomwise",
    "calculate_gradient": true,
    "gradwise_weight": 1.0,
    "stresswise_weight": 0.1,
    "atomwise_output_features": 3,
    "hybridize_features": true
  }
}
```

And provide the key names in your command:

```bash
CUDA_VISIBLE_DEVICES=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python fine-tuning.py \
  --root_dir "/path/to/dataset/" \
  --config_name "/path/to/config_forces.json" \
  --file_format "cif" \
  --batch_size 8 \
  --target_key "total_energy" \
  --force_key "forces" \
  --stresswise_key "stresses" \
  --output_dir "/path/to/runs/forces_run" \
  >> dealignn-log-forces.out
```

## Key Differences from Standard ALIGNN

### Hybridized Architecture
- **Input**: Crystal structures + human-designed descriptors
- **Preprocessing**: Requires `structure_descriptors_normalized.csv`
- **Enhanced Performance**: Combines neural and traditional features
- **Special Installation**: Requires `degraphs.py` in JARVIS environment

### File Requirements
```
Your Project/
├── dealignn/
│   ├── fine-tuning.py       # Main training script
│   ├── degraphs.py          # Must copy to JARVIS environment
│   └── runs/
│       └── run1.json       # Configuration file
├── datasets/
│   └── relaxed_structures_hse_bg/
│       ├── *.cif           # Structure files  
│       ├── id_prop.csv     # Property labels
│       └── structure_descriptors_normalized.csv  # Features
```

## Output Files and Analysis

### Standard Output Files
- `best_model.pt`: The trained model with best validation performance
- `checkpoint/`: Directory containing training checkpoints
- `train_stats.json`: Training statistics including loss and metrics over time
- `val_stats.json`: Validation statistics
- `test_stats.json`: Test set performance metrics
- `config.json`: Copy of the configuration used
- `preds.csv`: Predictions on the test set
- `hybrid_feature_importance.json`: Importance of different descriptor features

### Analyzing Results

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
plt.title('De-ALIGNN Training Progress')
plt.savefig('dealignn_loss_curve.png')

# Analyze feature importance (if available)
try:
    with open(f'{run_dir}/hybrid_feature_importance.json', 'r') as f:
        feature_importance = json.load(f)
    
    features = list(feature_importance.keys())
    importance = list(feature_importance.values())
    
    plt.figure(figsize=(12, 6))
    plt.bar(features, importance)
    plt.xlabel('Features')
    plt.ylabel('Importance')
    plt.title('Hybrid Feature Importance')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('feature_importance.png')
except:
    print("Feature importance analysis not available")
```

## Memory Optimization

The script automatically limits GPU memory usage. To adjust:

### Modify Memory Settings
- Reduce batch size for memory constraints
- Use gradient accumulation for effective larger batch sizes
- Enable mixed precision training if supported

### Configuration Adjustments
```json
{
  "dtype": "float16",
  "accumulate_grad_batches": 4,
  "max_memory_fraction": 0.7
}
```

## Troubleshooting

### Common Issues

**Missing `degraphs.py`:**
```
ImportError: cannot import name 'degraph' from 'jarvis.core'
```
**Solution**: Copy `degraphs.py` to the correct JARVIS path

**Missing descriptor file:**
```
FileNotFoundError: structure_descriptors_normalized.csv not found
```
**Solution**: Run preprocessing steps first

**Feature dimension mismatch:**
```
RuntimeError: size mismatch for hybrid feature layers
```
**Solution**: Check feature list specification matches descriptor file dimensions

**Memory issues with hybridized features:**
**Solution**: Reduce batch size, hybridized models use more memory than base ALIGNN

### Preprocessing Issues

**`id_propFeature.py` fails:**
- Check crystal structure file formats
- Verify property file format consistency
- Ensure all required dependencies are installed

**Normalization fails:**
- Check for NaN values in descriptors
- Verify descriptor file format
- Ensure consistent feature columns across all structures

### Installation Issues

**JARVIS path incorrect:**
- Check your actual conda environment path
- Verify Python version matches environment
- Ensure `degraphs.py` has correct permissions

## Performance Comparison

### Comparing with Base ALIGNN

```python
# Load results from both models
with open('/path/to/alignn/runs/test_stats.json', 'r') as f:
    alignn_results = json.load(f)

with open('/path/to/dealignn/runs/test_stats.json', 'r') as f:
    dealignn_results = json.load(f)

print("Performance Comparison:")
print(f"ALIGNN MAE: {alignn_results['mae']:.4f}")
print(f"De-ALIGNN MAE: {dealignn_results['mae']:.4f}")
improvement = ((alignn_results['mae'] - dealignn_results['mae']) / alignn_results['mae']) * 100
print(f"Improvement: {improvement:.2f}%")
```

## Best Practices

- **Preprocessing First**: Always run feature generation and normalization before training
- **Feature Selection**: Carefully choose which descriptors to include for your specific property
- **Installation Verification**: Verify `degraphs.py` is correctly installed before training
- **Memory Monitoring**: Hybridized models require more memory than base ALIGNN
- **Baseline Comparison**: Compare with base ALIGNN to verify improvement from hybridization
- **Feature Analysis**: Use feature importance analysis to understand which descriptors are most valuable
- **Configuration Management**: Keep detailed records of feature selections and configurations for reproducibility

## Example Complete Workflow

```bash
# 1. Install degraphs.py
cp degraphs.py /path/to/conda/envs/alignn_exact/lib/python3.8/site-packages/jarvis/core/

# 2. Generate features
python id_propFeature.py
python normalize_hybrid_feature.py

# 3. Edit fine-tuning.py to specify features and paths
# (Manual step - edit the file)

# 4. Train de-ALIGNN
CUDA_VISIBLE_DEVICES=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python fine-tuning.py \
  --root_dir "/path/to/dataset/" \
  --config_name "/path/to/config.json" \
  --file_format "cif" \
  --batch_size 16 \
  --output_dir "/path/to/runs/dealignn_run1" \
  >> dealignn-log.out

# 5. Analyze results
python analyze_dealignn_results.py
```

This workflow ensures proper setup and execution of the hybridized de-ALIGNN model for enhanced materials property prediction.
