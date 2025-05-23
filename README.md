# Crystal Structure Deep Learning Models

This repository contains datasets and trained models for the paper: **"Improving deep representation learning for crystal structures by learning and hybridizing human-designed descriptors"**

## Project Structure

```
├── alignn/                     # ALIGNN model implementation
├── cgcnn/                      # CGCNN model implementation  
├── dealignn/                   # Hybridized ALIGNN (de-ALIGNN)
├── decgcnn/                    # Hybridized CGCNN (de-CGCNN)
├── deE3NN/                     # Hybridized E3NN (de-E3NN)
├── dematformer/                # Hybridized Matformer (de-Matformer)
├── deMEGNet/                   # Hybridized MEGNet (de-MEGNet) [In Development]
├── datasets/                   # All datasets organized by property
├── trained_models/             # Pre-trained models
└── ENVIRONMENT_SETUP.md        # Environment setup instructions
```

## Quick Start

1. **Set up environments** following `ENVIRONMENT_SETUP.md`
2. **Choose your model** from the available implementations
3. **Check the model-specific folder** for `HowToRun*.txt` files with detailed instructions

## Model Overview

### Base Models
- **ALIGNN**: Advanced graph neural network for crystal property prediction
- **CGCNN**: Crystal Graph Convolutional Neural Networks

### Hybridized Models (de- prefix)
Hybridized versions that combine neural representations with human-designed descriptors:
- **de-ALIGNN**: Hybridized ALIGNN 
- **de-CGCNN**: Hybridized CGCNN
- **de-E3NN**: Hybridized E3NN
- **de-Matformer**: Hybridized Matformer
- **de-MEGNet**: Hybridized MEGNet *(work in progress)*

## Running the Models

### Base Models

**CGCNN:**
```bash
conda activate cgcnn-gpu
cd cgcnn/
python main.py                    # Standard version
python main_optimized.py          # With caching/Optuna improvements
```

**ALIGNN:**
```bash
conda activate alignn_exact  
cd alignn/
python train_alignn.py            # Standard version
python train_alignnV2.py          # With improvements
```

### Hybridized Models

**⚠️ Important: Hybridized models require preprocessing**

1. **Generate structure descriptors:**
   ```bash
   python id_propFeature.py        # Creates structure_descriptors.csv
   ```

2. **Normalize hybrid features:**
   ```bash
   python normalize_hybrid_feature.py
   ```

3. **Run hybridized models:**

**de-CGCNN:**
```bash
conda activate cgcnn-gpu
cd decgcnn/
# Specify feature list at the beginning of main.py and predict.py
# Input the path of descriptors file into the scripts
python main.py
```

**de-ALIGNN:**
```bash
conda activate alignn_exact
cd dealignn/
# ⚠️ REQUIRED: Copy degraph.py to your ALIGNN environment
cp degraph.py /home/marcus/anaconda3/envs/alignn_exact/lib/python3.8/site-packages/jarvis/core/
# Specify feature list and descriptors file path in fine-tuning.py
python fine-tuning.py
```

**de-E3NN:**
```bash
cd deE3NN/
# Specify feature list and descriptors file path in load_data.py
python train.py  # (check HowToRun file for exact script)
```

**de-Matformer:**
```bash
cd dematformer/
# Specify feature list and descriptors file path in train.py
python train.py
```

## Important Notes

### Dataset Formats
- Some models expect different `id_prop.csv` formats
- **ALIGNN**: Expects full filename with extension (e.g., `qmof-0a0e6b0.cif`)
- **Other models**: May expect just basename (e.g., `qmof-0a0e6b0`)
- Duplicate databases exist with only `id_prop.csv` differences to accommodate these requirements

### Environment Requirements
- **cgcnn-gpu**: For CGCNN and de-CGCNN models
- **alignn_exact**: For ALIGNN and de-ALIGNN models
- See `ENVIRONMENT_SETUP.md` for detailed installation instructions

### Code Modifications
- **cgcnn/decgcnn**: 
  - `main.py`: Unmodified versions
  - `main_optimized.py`: Enhanced with caching and Optuna optimizations
- **alignn**: 
  - `train_alignn.py`: Standard version
  - `train_alignnV2.py`: Improved version
- **dealignn**: Requires `degraph.py` script placement in JARVIS environment

## Datasets

The `datasets/` folder contains all datasets organized by property. Each dataset includes:
- Crystal structure files (`.cif` format)
- Property labels (`id_prop.csv`)
- Training/validation/test splits (where applicable)

## Pre-trained Models

The `trained_models/` folder contains pre-trained models following the naming convention:
- `(model_name)_(prop_name).pt`: Trained model for predicting `prop_name` using `model_name`
- These correspond to the results shown in Figure 4 of the paper

## Getting Started Checklist

- [ ] Install conda environments using `ENVIRONMENT_SETUP.md`
- [ ] Choose your target property and model
- [ ] For hybridized models: Run preprocessing scripts
- [ ] Check model-specific `HowToRun*.txt` files
- [ ] Configure feature lists and file paths in model scripts
- [ ] For de-ALIGNN: Install `degraph.py` in JARVIS environment
- [ ] Run training or inference

## Troubleshooting

1. **Environment issues**: See `ENVIRONMENT_SETUP.md` troubleshooting section
2. **Model-specific issues**: Check `HowToRun*.txt` files in each model folder
3. **Missing degraph.py**: Ensure it's copied to the correct JARVIS path for de-ALIGNN
4. **Dataset format errors**: Verify your model expects the correct `id_prop.csv` format

## Status

- ✅ ALIGNN, CGCNN, de-ALIGNN, de-CGCNN: Ready to use
- ✅ de-E3NN, de-Matformer: Available 
- 🚧 de-MEGNet: Work in progress (migrating to PyTorch-DGL version)

---

**Need help?** Check the model-specific `HowToRun*.txt` files in each subfolder for detailed instructions.
