# DE-CGCNN Usage Guide

This guide explains how to use the hybridized Crystal Graph Convolutional Neural Networks (de-CGCNN) implementation. De-CGCNN combines neural representations with human-designed descriptors for enhanced materials property prediction.

## Environment Setup

Activate the cgcnn-gpu environment:
```bash
conda activate cgcnn-gpu
```

## ⚠️ Prerequisites: Feature Preprocessing

**IMPORTANT**: De-CGCNN requires preprocessing to generate structure descriptors before training.

### Step 1: Generate Structure Descriptors
```bash
python id_propFeature.py  # Creates structure_descriptors.csv
```

### Step 2: Normalize Features
```bash
python normalize_hybrid_feature.py  # Creates structure_descriptors_normalized.csv
```

## Standard DE-CGCNN (`main.py`)

### Basic Usage

For standard training with manually specified hyperparameters:

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONWARNINGS=ignore python main.py \
  --batch-size 128 --n-conv 5 --n-h 1 \
  --train-ratio 0.8 --val-ratio 0.1 --test-ratio 0.1 \
  --workers 10 --epochs 300 --print-freq 1 \
  ../datasets/relaxed_structures_hse_bg_decgcnn \
  ../datasets/relaxed_structures_hse_bg_decgcnn/structure_descriptors.csv >> decgcnn_train.out
```

### Important Notes for Standard Version
- **Feature List**: Specify the feature list at the beginning of `main.py` and `predict.py`
- **Descriptor Path**: Input the path of the descriptors file into both scripts
- **Dataset Path**: Use the decgcnn-specific dataset folder

## Enhanced DE-CGCNN (`main_optimized.py`)

The optimized version includes LMDB caching and comprehensive hyperparameter optimization with Optuna.

### Additional Requirements

Install optimization dependencies:
```bash
conda install -c conda-forge optuna
conda install -c conda-forge msgpack-python
pip install msgpack-numpy
pip install lmdb
```

### Step 1: Create LMDB Cache (One-time Setup)

```bash
CUDA_VISIBLE_DEVICES=1 python main_optimized.py \
  --create-lmdb-cache \
  --lmdb-path ./lmdbCache \
  --lmdb-map-size 1e11 \
  ../datasets/relaxed_structures_hse_bg_decgcnn \
  ../datasets/relaxed_structures_hse_bg_decgcnn/structure_descriptors_normalized.csv
```

### Step 2: Hyperparameter Optimization

```bash
CUDA_VISIBLE_DEVICES=1 PYTHONWARNINGS=ignore python main_optimized.py \
  --train-ratio 0.8 --val-ratio 0.1 --test-ratio 0.1 \
  --workers 8 --epochs 300 --print-freq 1 \
  --hp-opt --n-trials 150 --trial-epochs 75 \
  --scheduler-opt \
  --lmdb-path ./lmdbCache \
  ../datasets/relaxed_structures_hse_bg_decgcnn \
  ../datasets/relaxed_structures_hse_bg_decgcnn/structure_descriptors_normalized.csv \
  >> decgcnn_optimize_comprehensive.out
```

### Step 3: Train Final Model with Best Parameters

After optimization completes:
```bash
CUDA_VISIBLE_DEVICES=1 PYTHONWARNINGS=ignore python main_optimized.py \
  --batch-size <best_batch_size> --n-conv <best_n_conv> --n-h <best_n_h> \
  --lr <best_lr> --optim <best_optim> --scheduler <best_scheduler> \
  --atom-fea-len <best_atom_fea_len> --h-fea-len <best_h_fea_len> \
  --weight-decay <best_weight_decay> --lr-gamma <best_gamma> \
  --train-ratio 0.8 --val-ratio 0.1 --test-ratio 0.1 \
  --workers 10 --epochs 300 \
  --lmdb-path ./lmdbCache \
  ../datasets/relaxed_structures_hse_bg_decgcnn \
  ../datasets/relaxed_structures_hse_bg_decgcnn/structure_descriptors_normalized.csv
```

## Enhanced Hyperparameter Search Features

The optimized version performs comprehensive parameter search:

### Network Architecture Parameters
- Batch sizes: [16, 32, 64, 128, 256]
- Atom feature lengths: [32, 64, 128, 256, 512]
- Hidden feature lengths: [64, 128, 256, 512, 1024]
- Convolutional layers: 1-7
- Hidden layers: 1-5

### Optimization Parameters
- Learning rates: 5e-5 to 5e-1 (log scale)
- Optimizers: SGD, Adam
- SGD momentum: 0.5 to 0.99
- Weight decay: 1e-7 to 1e-2 (log scale)

### Learning Rate Schedulers
- Step scheduler with configurable milestones and gamma
- Cosine annealing scheduler
- Exponential scheduler with configurable decay rate

## Key Differences from Standard CGCNN

### Hybridized Features
- **Input**: Crystal structures + human-designed descriptors
- **Preprocessing**: Requires `structure_descriptors_normalized.csv`
- **Enhanced Performance**: Combines neural and traditional features

### File Structure Requirements
```
datasets/relaxed_structures_hse_bg_decgcnn/
├── structure_files/          # CIF files
├── id_prop.csv              # Property labels
└── structure_descriptors_normalized.csv  # Required: Normalized features
```

### Code Modifications Required
Before running the standard version:

1. **Edit `main.py`**: Specify feature list at the beginning
2. **Edit `predict.py`**: Specify feature list and descriptor file path
3. **Verify paths**: Ensure descriptor file path is correctly set

## Output Files

### Standard Version
- `model_best.pth.tar`: Best model checkpoint
- `checkpoint.pth.tar`: Latest checkpoint  
- `test_results.csv`: Test set predictions and targets

### Optimized Version
- `best_params.json`: Optimal hyperparameters found
- `all_trials.json`: Complete trial data
- `trial_timing.csv`: Detailed timing information
- `optimization_summary.json`: Overall statistics

## Analyzing Results

Analyze optimization results with Optuna:

```python
import optuna
from optuna.visualization import plot_param_importances, plot_optimization_history

study = optuna.load_study(study_name="decgcnn_study", storage="sqlite:///decgcnn_study.db")
plot_param_importances(study)
plot_optimization_history(study)
```

## Troubleshooting

### Common Issues

**Missing descriptor file:**
```
FileNotFoundError: structure_descriptors_normalized.csv not found
```
**Solution**: Run preprocessing steps first

**Feature dimension mismatch:**
```
RuntimeError: size mismatch for feature layers
```
**Solution**: Check feature list specification in code matches descriptor file

**Memory issues with hybridized features:**
**Solution**: Reduce batch size, increase swap space, or use gradient checkpointing

### Preprocessing Issues

**`id_propFeature.py` fails:**
- Check crystal structure file formats
- Verify property file format
- Ensure all dependencies are installed

**Normalization fails:**
- Check for NaN values in descriptors
- Verify descriptor file format
- Ensure consistent feature columns

## When to Use Which Version

### Use Standard (`main.py`) when:
- You have predetermined hyperparameters
- Quick experiments with known configurations
- Limited computational resources
- Simple baseline comparisons

### Use Optimized (`main_optimized.py`) when:
- Starting new property prediction tasks
- Want systematic hyperparameter optimization
- Have computational resources for extensive search
- Working with large datasets (LMDB caching benefits)
- Need reproducible hyperparameter selection

## Tips

- **Preprocessing**: Always run feature generation/normalization first
- **Feature Selection**: Carefully choose which descriptors to include
- **Cache Management**: Create LMDB cache once, reuse for experiments
- **Memory Monitoring**: Hybridized models use more memory than base CGCNN
- **Validation**: Compare with base CGCNN to verify improvement from hybridization