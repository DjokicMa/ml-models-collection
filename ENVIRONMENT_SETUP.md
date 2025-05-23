# Environment Setup Guide

This project requires two conda environments for different components. Follow the instructions below to set them up.

## Prerequisites

- [Anaconda](https://www.anaconda.com/products/distribution) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html) installed
- Git (to clone this repository)

## Environment Setup

### Option 1: Create from Environment Files (Recommended)

```bash
# Create alignn_exact environment
conda env create -f alignn_exact_environment.yml

# Create cgcnn-gpu environment
conda env create -f cgcnn_gpu_environment.yml
```

### Option 2: Create with Custom Names

If you want to use different environment names:

```bash
# Create with custom names
conda env create -f alignn_exact_environment.yml -n my_alignn_env
conda env create -f cgcnn_gpu_environment.yml -n my_cgcnn_env
```

## Activating Environments

```bash
# Activate alignn_exact environment
conda activate alignn_exact

# Activate cgcnn-gpu environment  
conda activate cgcnn-gpu
```

## Verification

After creating the environments, verify they work correctly:

```bash
# Test alignn_exact environment
conda activate alignn_exact
python -c "import alignn; print('ALIGNN environment ready!')"

# Test cgcnn-gpu environment
conda activate cgcnn-gpu
python -c "import torch; print(f'PyTorch GPU available: {torch.cuda.is_available()}')"
```

## Troubleshooting

### Environment Creation Fails

If environment creation fails due to package conflicts:

1. **Try creating with conda-forge channel:**
   ```bash
   conda env create -f environment.yml -c conda-forge
   ```

2. **Update conda first:**
   ```bash
   conda update conda
   ```

3. **Create environment step by step:**
   - Create empty environment: `conda create -n env_name python=3.x`
   - Install packages manually from the .yml file

### GPU Issues (cgcnn-gpu environment)

- Ensure NVIDIA drivers are installed
- Verify CUDA compatibility with your GPU
- Check PyTorch CUDA version matches your system CUDA

### Platform Differences

If you encounter platform-specific issues:

1. **Remove platform-specific lines** from the .yml file (lines starting with `- _libgcc_mutex`, etc.)
2. **Use the `--no-builds` export option** when creating environment files

## Alternative: Manual Installation

If automated environment creation doesn't work, you can install packages manually:

```bash
# Create base environment
conda create -n alignn_exact python=3.9
conda activate alignn_exact

# Install main packages (check alignn_exact_environment.yml for versions)
conda install numpy pandas matplotlib scikit-learn
pip install alignn

# Repeat similar process for cgcnn-gpu environment
```

## Updating Environments

To update the environment files after making changes:

```bash
# Re-export environments
conda activate alignn_exact
conda env export --no-builds > alignn_exact_environment.yml

conda activate cgcnn-gpu  
conda env export --no-builds > cgcnn_gpu_environment.yml
```

## Need Help?

- Check the conda documentation: https://docs.conda.io/
- For GPU setup issues, refer to PyTorch installation guide: https://pytorch.org/get-started/locally/
- Create an issue in this repository if you encounter setup problems