# CGCNN Usage Guide

This guide explains how to use the Crystal Graph Convolutional Neural Networks (CGCNN) implementation for materials property prediction. Two versions are available: the standard implementation and an enhanced version with LMDB caching and Optuna optimization.

## Environment Setup

Activate the cgcnn-gpu environment:
```bash
conda activate cgcnn-gpu
```

## Standard CGCNN (`main.py`)

### Basic Usage

For standard training with manually specified hyperparameters:

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONWARNINGS=ignore python main.py \
  --batch-size 128 --n-conv 5 --n-h 1 \
  --train-ratio 0.8 --val-ratio 0.1 --test-ratio 0.1 \
  --workers 10 --epochs 300 --print-freq 1 \
  data/relaxed_structures_hse_bg >> cgcnn_train.out
```

### Key Parameters

- `--batch-size`: Training batch size (default: 256)
- `--n-conv`: Number of convolutional layers (default: 3)
- `--n-h`: Number of hidden layers (default: 1)
- `--epochs`: Number of training epochs
- `--workers`: Number of data loading workers
- `--train-ratio`, `--val-ratio`, `--test-ratio`: Data split ratios

## Enhanced CGCNN (`main_optimized.py`)

The optimized version includes LMDB caching for faster data loading and comprehensive hyperparameter optimization with Optuna.

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
CUDA_VISIBLE_DEVICES=0 python main_optimized.py \
  --create-lmdb-cache \
  --lmdb-path data/relaxed_structures_hse_bg/cif_data \
  --lmdb-map-size 1e11 \
  data/relaxed_structures_hse_bg
```

### Step 2: Hyperparameter Optimization

```bash
CUDA_VISIBLE_DEVICES=0 python main_optimized.py \
  --train-ratio 0.8 --val-ratio 0.1 --test-ratio 0.1 \
  --workers 8 --epochs 300 --print-freq 1 \
  --hp-opt --n-trials 150 --trial-epochs 75 \
  --scheduler-opt \
  --lmdb-path data/relaxed_structures_hse_bg/cif_data \
  data/relaxed_structures_hse_bg >> cgcnn_optimize_comprehensive.out
```

### Step 3: Train Final Model with Best Parameters

After optimization completes, use the best parameters:
```bash
CUDA_VISIBLE_DEVICES=0 python main_optimized.py \
  --batch-size <best_batch_size> --n-conv <best_n_conv> --n-h <best_n_h> \
  --lr <best_lr> --optim <best_optim> --scheduler <best_scheduler> \
  --atom-fea-len <best_atom_fea_len> --h-fea-len <best_h_fea_len> \
  --weight-decay <best_weight_decay> --lr-gamma <best_gamma> \
  --train-ratio 0.8 --val-ratio 0.1 --test-ratio 0.1 \
  --workers 10 --epochs 300 \
  --lmdb-path data/relaxed_structures_hse_bg/cif_data \
  data/relaxed_structures_hse_bg
```

## Enhanced Hyperparameter Search Features

The optimized version performs comprehensive parameter search:

### Network Architecture Parameters
- Batch sizes: [16, 32, 64, 128, 256]
- Atom feature lengths: [32, 64, 128, 256, 512]
- Hidden feature lengths: [64, 128, 256, 512, 1024]
- Convolutional layers: 1-7 (expanded from 1-5)
- Hidden layers: 1-5 (expanded from 1-3)

### Optimization Parameters
- Learning rates: 5e-5 to 5e-1 (log scale)
- Optimizers: SGD, Adam
- SGD momentum: 0.5 to 0.99
- Weight decay: 1e-7 to 1e-2 (wider range, log scale)

### Learning Rate Schedulers
- Step scheduler with configurable milestones and gamma
- Cosine annealing scheduler
- Exponential scheduler with configurable decay rate

## Output Files

### Standard Version
- `model_best.pth.tar`: Best model checkpoint
- `checkpoint.pth.tar`: Latest checkpoint
- `test_results.csv`: Test set predictions and targets

### Optimized Version
- `best_params.json`: Optimal hyperparameters found
- `all_trials.json`: Complete trial data including parameters and performance
- `trial_timing.csv`: Detailed epoch-by-epoch timing information
- `trial_durations.csv`: Summary of each trial's total duration
- `optimization_summary.json`: Overall optimization statistics

## Analyzing Results

For the optimized version, you can analyze results with Optuna visualizations:

```python
import optuna
from optuna.visualization import plot_param_importances, plot_optimization_history

study = optuna.load_study(study_name="cgcnn_study", storage="sqlite:///cgcnn_study.db")
plot_param_importances(study)
plot_optimization_history(study)
```

## When to Use Which Version

### Use Standard (`main.py`) when:
- You have known good hyperparameters
- Quick prototyping or testing
- Limited computational resources
- Simple baseline experiments

### Use Optimized (`main_optimized.py`) when:
- Starting a new dataset/property prediction task
- Want to find optimal hyperparameters systematically
- Have computational resources for extensive search
- Need reproducible, documented hyperparameter selection
- Working with large datasets (benefits from LMDB caching)

## Tips

- **LMDB Cache**: Create once, reuse for multiple experiments
- **Memory**: Monitor GPU memory usage, reduce batch size if needed
- **Workers**: Adjust `--workers` based on your CPU cores
- **Trials**: Start with fewer trials (50-100) for initial exploration
- **Epochs**: Use shorter `--trial-epochs` for optimization, full epochs for final training