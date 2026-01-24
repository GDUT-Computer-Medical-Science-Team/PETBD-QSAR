# PETBD-QSAR

<div align="center">

# 🧠 Blood-Brain Barrier Permeability Prediction

**Machine Learning QSAR Models Based on PET Tracer Biodistribution Dataset**

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![RDKit](https://img.shields.io/badge/Powered%20by-RDKit-3838ff.svg?logo=data:image/svg+xml;base64,PHN2ZyBpZD0ibWFpbiIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIiB2aWV3Qm94PSIwIDAgMTAwIDEwMCI+PGNpcmNsZSBjeD0iNTAiIGN5PSI1MCIgcj0iNDAiIHN0cm9rZT0id2hpdGUiIHN0cm9rZS13aWR0aD0iNCIgZmlsbD0iYmx1ZSIvPjwvc3ZnPg==)](https://www.rdkit.org/)
[![Contributions Welcome](https://img.shields.io/badge/Contributions-Welcome-brightgreen.svg)](CONTRIBUTING.md)

</div>

---

## 🌟 Highlights

<table>
<tr>
<td width="50%">

### 📊 Novel Dataset
- **1,058 PET tracers** with in vivo biodistribution data
- **14 organ distributions** at 60 min post-injection
- First comprehensive PET-based BBB dataset

</td>
<td width="50%">

### 🎯 Advanced Models
- **6 ML algorithms** with 18F resampling
- **Test R² up to 0.85+** for logBB prediction
- **Multi-feature integration**: Morgan FP + Mordred + Metadata

</td>
</tr>
<tr>
<td width="50%">

### 🔄 Isotope Balancing
- **18F resampling** to handle class imbalance
- **3 strategies**: Oversample, Undersample, Combined
- Improved model generalization

</td>
<td width="50%">

### ✅ External Validation
- **B3DB dataset** validation (7,807 compounds)
- Proven real-world applicability
- Cross-dataset performance metrics

</td>
</tr>
</table>

---

## 📖 Table of Contents

- [Project Overview](#-project-overview)
- [Quick Start](#-quick-start)
- [Installation](#-installation)
- [Dataset](#-dataset)
- [Model Architecture](#-model-architecture)
- [Usage](#-usage)
- [Utilities](#-utilities)
- [Results](#-results)
- [Citation](#-citation)
- [Contributing](#-contributing)
- [License](#-license)

---

## 🎯 Project Overview

This repository implements **state-of-the-art machine learning models** for predicting blood-brain barrier (BBB) permeability using a novel **PET tracer biodistribution dataset (PETBD)**. 

### Key Features

🔬 **Dual Prediction Targets**
- **logBB**: Blood-brain barrier penetration coefficient
- **Cbrain**: Brain drug concentration at 60 minutes

🧬 **Comprehensive Feature Engineering**
- **Morgan Fingerprints** (1024-bit, radius=2)
- **Mordred Descriptors** (2D molecular descriptors)
- **Experimental Metadata** (species, gender, weight, dosage)

⚡ **Advanced Training Techniques**
- **18F Isotope Resampling** for class balance
- **Optuna Hyperparameter Optimization** (100 trials)
- **10-Fold Cross-Validation** on training set
- **Stratified Train/Val/Test Split** (81%/9%/10%)

📊 **Robust Evaluation**
- Multiple metrics (R², RMSE, MAE, MAPE)
- External validation on B3DB dataset
- Comprehensive performance reports (JSON + CSV)

---

## 🚀 Quick Start

```bash
# Clone repository
git clone https://github.com/GDUT-Computer-Medical-Science-Team/PETBD-QSAR.git
cd PETBD-QSAR

# Setup environment
conda create -n PETBD python=3.8
conda activate PETBD

# Install dependencies
pip install -r requirements.txt

# Train your first model
cd MachineLearningModels/logBBModel
python XGBoost_FP_18F_Resample.py

# Make predictions
cd ../..
python predict_logBB.py --smiles "CCO" --model xgboost
```

---

## 🛠️ Installation

### Prerequisites

- Python 3.8+
- Conda (recommended) or pip
- Java 8+ (for PaDEL descriptors)

### Option 1: Conda (Recommended)

```bash
# Create environment
conda create -n PETBD python=3.8
conda activate PETBD

# Install core dependencies
conda install -c conda-forge rdkit numpy pandas scikit-learn matplotlib seaborn

# Install ML frameworks
pip install xgboost lightgbm catboost optuna

# Install descriptor tools
pip install mordred padelpy

# Install utilities
pip install joblib colorlog tqdm PyYAML openpyxl xlrd
```

### Option 2: Pip

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install all dependencies
pip install -r requirements.txt
```

### Verify Installation

```bash
python -c "import rdkit; import xgboost; import lightgbm; print('Installation successful!')"
```

---

## 📊 Dataset

### PETBD Dataset (PTBD_v20240912.csv)

| Property | Value |
|----------|-------|
| **Total Compounds** | 1,058 |
| **18F Tracers** | 723 (68.3%) |
| **Non-18F Tracers** | 335 (31.7%) |
| **Organs Measured** | 14 (brain, blood, liver, lung, etc.) |
| **Data Source** | Published PET tracer studies |

**Key Features:**
- `compound index`: Compound name with isotope label
- `SMILES`: Molecular structure representation
- `logBB`: Blood-brain barrier permeability
- `brain mean60min`: Brain concentration (60 min)
- `Metadata`: Animal type, gender, weight, dosage
- `Organ data`: 14 organ distributions

### OrganDataAt60min.csv

Brain-specific concentration data for **Cbrain prediction models**.

### B3DB External Validation Set

- **7,807 compounds** for external validation
- Used to test model generalization
- Performance comparison with literature

---

## 🏗️ Model Architecture

### Supported Algorithms

<table>
<tr>
<th>Algorithm</th>
<th>Type</th>
<th>LogBB</th>
<th>Cbrain</th>
<th>Key Advantages</th>
</tr>
<tr>
<td><b>XGBoost</b></td>
<td>Gradient Boosting</td>
<td>✅</td>
<td>✅</td>
<td>Fast, regularized, handles missing values</td>
</tr>
<tr>
<td><b>Random Forest</b></td>
<td>Ensemble</td>
<td>✅</td>
<td>✅</td>
<td>Robust, feature importance, low overfitting</td>
</tr>
<tr>
<td><b>LightGBM</b></td>
<td>Gradient Boosting</td>
<td>✅</td>
<td>✅</td>
<td>Extremely fast, low memory usage</td>
</tr>
<tr>
<td><b>CatBoost</b></td>
<td>Gradient Boosting</td>
<td>✅</td>
<td>✅</td>
<td>Handles categorical features natively</td>
</tr>
<tr>
<td><b>MLP</b></td>
<td>Neural Network</td>
<td>✅</td>
<td>✅</td>
<td>Non-linear relationships, flexible architecture</td>
</tr>
<tr>
<td><b>SVM</b></td>
<td>Kernel Method</td>
<td>✅</td>
<td>✅</td>
<td>Effective in high-dimensional spaces</td>
</tr>
</table>

### 18F Isotope Resampling

**Problem**: Dataset imbalance (68% 18F vs 32% non-18F)

**Solution**: Intelligent resampling strategies

```python
# Three resampling approaches
1. Oversample:  Keep all + replicate minority to match majority
2. Undersample: Random sample majority to match minority  
3. Combined:    Keep all original + add minority samples (RECOMMENDED)
```

**Benefits**:
- ✅ Prevents model bias toward majority class
- ✅ Improves generalization to underrepresented isotopes
- ✅ Maintains all original data (Combined strategy)

---

## 💻 Usage

### 1. Train LogBB Models

```bash
cd MachineLearningModels/logBBModel

# Train individual models with 18F resampling
python XGBoost_FP_18F_Resample.py
python RF_FP_18F_Resample.py
python LightGBM_PETBD_18F_Resample.py
python CatBoost_PETBD_18F_Resample.py
python MLP_PETBD_18F_Resample_Fixed.py
python SVM_PETBD_18F_Resample.py

# Train combined RF+XGB model
python RF_XGB_FP_18F_Resample_Combined.py

# Run comparison with same test set
python Compare_RF_XGB_same_testset.py
```

### 2. Train Cbrain Models

```bash
cd MachineLearningModels/CbrainModel

# Train brain concentration models
python XGBoost_Cbrain_18F_Resample.py
python RF_Cbrain_18F_Resample.py
python LightGBM_Cbrain_18F_Resample.py
python CatBoost_Cbrain_18F_Resample.py
python MLP_Cbrain_18F_Resample.py
python SVM_Cbrain_18F_Resample.py

# Quick complete training
python Quick_Complete_Cbrain.py
```

### 3. Make Predictions

#### Single Compound Prediction

```bash
python predict_logBB.py --smiles "CCO" --model xgboost

# Output:
# Predicted logBB: -0.2341
# BBB Permeability: Moderate
# Interpretation: Moderate BBB permeability (-0.5 < logBB < 0)
```

#### Batch Prediction

```bash
python predict_logBB.py \
    --input compounds.csv \
    --output predictions.csv \
    --model xgboost \
    --smiles-column "SMILES"
```

#### List Available Models

```bash
python predict_logBB.py --list-models
```

### 4. External Validation

```bash
cd MachineLearningModels/logBBModel

# B3DB validation with Random Forest
python B3DB_RF_18F_Combined_Full.py

# B3DB validation with XGBoost
python B3DB_XGB_18F_Combined_Full.py

# Combined validation
python B3DB_XGB_18F_Combined_Validation.py
```

---

## 🔧 Utilities

### Data Analysis Tools

```bash
# Analyze 18F isotope distribution
python analyze_18F_sampling_simple.py

# Calculate combined dataset size
python calculate_combined_dataset_size.py

# Create balanced dataset
python make_balanced_18F_dataset.py \
    --csv dataset_PETBD/PTBD_v20240912.csv \
    --method oversample \
    --out result/balanced_dataset.csv
```

### Visualization Tools

```bash
# Create scatter plots for all models
python create_scatter_plots.py

# Plot best performing models
python plot_best_models.py

# Generate separate scatter plots
python create_separate_scatter_plots.py

# Comprehensive model analysis
python comprehensive_model_analysis.py
```

### Feature Engineering Tools

```bash
# Get Mordred descriptor names
python get_mordred_names.py

# Convert Mordred indices to names
python convert_mordred_names.py

# Generate prediction CSVs
python generate_prediction_csvs.py
```

---

## 📁 Project Structure

```
PETBD-QSAR/
│
├── 📂 MachineLearningModels/
│   ├── 📂 logBBModel/              # LogBB prediction models
│   │   ├── *_FP_18F_Resample.py   # Fingerprint-based models
│   │   ├── *_PETBD_18F_Resample.py # Full feature models
│   │   ├── B3DB_*.py               # B3DB validation scripts
│   │   ├── Compare_RF_XGB_*.py     # Model comparison
│   │   └── logbbModel/             # Trained models (.joblib)
│   │
│   ├── 📂 CbrainModel/             # Brain concentration models
│   │   ├── *_Cbrain_18F_Resample.py
│   │   ├── Quick_Complete_Cbrain.py
│   │   └── cbrainmodel/            # Trained models (.joblib)
│   │
│   └── 📂 VivoValidation/          # In vivo validation
│
├── 📂 MedicalDatasetsMerger/       # Dataset integration tools
│   ├── DataMerger.py
│   └── utils/
│
├── 📂 dataset_PETBD/               # Main datasets
│   ├── PTBD_v20240912.csv         # LogBB dataset (1,058 compounds)
│   └── PETBD20240906.csv          # Legacy dataset
│
├── 📂 data/
│   ├── 📂 logBB_data/             # LogBB related data
│   │   ├── OrganDataAt60min.csv   # Organ distribution data
│   │   ├── RF_XGB_FP_18F_balanced_features.csv
│   │   └── cbrain_xgb_petbd_18F_balanced_features.csv
│   ├── 📂 test/                   # Test datasets
│   └── 📂 train/                  # Training datasets
│
├── 📂 result/                     # Model outputs
│   └── 📂 vivoresult/            # Validation results
│
├── 📂 utils/                      # Utility modules
│   ├── DataLogger.py              # Logging system
│   └── datasets_loader.py         # Data loading utilities
│
├── 📄 predict_logBB.py            # 🌟 Prediction CLI tool
├── 📄 make_balanced_18F_dataset.py # Dataset balancing
├── 📄 analyze_18F_sampling_simple.py # Dataset analysis
├── 📄 create_scatter_plots.py     # Visualization
├── 📄 comprehensive_model_analysis.py # Model comparison
│
└── 📄 README.md                   # This file
```

---

## 📊 Results

### LogBB Prediction Performance

| Model | Train R² | Val R² | Test R² | Test RMSE | Test MAE |
|-------|----------|--------|---------|-----------|----------|
| **XGBoost** | 0.95 | 0.87 | **0.85** | 0.31 | 0.24 |
| **Random Forest** | 0.93 | 0.86 | **0.84** | 0.33 | 0.25 |
| **LightGBM** | 0.94 | 0.86 | **0.83** | 0.34 | 0.26 |
| CatBoost | 0.92 | 0.84 | 0.81 | 0.36 | 0.27 |
| MLP | 0.89 | 0.82 | 0.79 | 0.38 | 0.29 |
| SVM | 0.87 | 0.80 | 0.77 | 0.40 | 0.31 |

*Results based on 18F resampled models with combined features (Morgan FP + Mordred + Metadata)*

### Cbrain Prediction Performance

| Model | Train R² | Val R² | Test R² | Test RMSE |
|-------|----------|--------|---------|-----------|
| **XGBoost** | 0.92 | 0.85 | **0.82** | 0.45 |
| **Random Forest** | 0.91 | 0.84 | **0.81** | 0.47 |
| LightGBM | 0.90 | 0.83 | 0.80 | 0.48 |

### B3DB External Validation

| Model | B3DB R² | B3DB RMSE | Compounds |
|-------|---------|-----------|-----------|
| XGBoost | 0.71 | 0.52 | 7,807 |
| Random Forest | 0.69 | 0.54 | 7,807 |

---

## 📈 Evaluation Metrics

All models are evaluated using comprehensive metrics:

| Metric | Formula | Description |
|--------|---------|-------------|
| **R²** | 1 - (SS_res / SS_tot) | Coefficient of determination |
| **Adjusted R²** | 1 - [(1-R²)(n-1)/(n-k-1)] | Adjusted for feature count |
| **RMSE** | √(Σ(y-ŷ)²/n) | Root Mean Squared Error |
| **MAE** | Σ\|y-ŷ\|/n | Mean Absolute Error |
| **MSE** | Σ(y-ŷ)²/n | Mean Squared Error |
| **MAPE** | (Σ\|(y-ŷ)/y\|/n)×100 | Mean Absolute Percentage Error |

---

## 🔍 Feature Engineering

### 1. Morgan Fingerprints
```python
# RDKit implementation
- Radius: 2
- Bits: 1024
- Circular fingerprints capturing local structure
```

### 2. Mordred Descriptors
```python
# 2D molecular descriptors
- Constitutional descriptors
- Topological indices
- Physicochemical properties
- Automated correlation filtering
```

### 3. Experimental Metadata
```python
# PET experimental conditions
- Animal species (mouse/rat)
- Gender (male/female)
- Body weight (g)
- Injection dosage (μCi)
```

### 4. Feature Selection

**Stratified Selection Strategy:**
```python
# Maintain representation from all feature types
Example: 50 features total
  - 21 Morgan fingerprint bits (42%)
  - 25 Mordred descriptors (50%)
  - 4 Metadata features (8%)
```

---

## ⚙️ Hyperparameter Optimization

All models use **Optuna** for automatic hyperparameter tuning:

```python
# Optimization settings
- n_trials: 100
- cv_folds: 10
- objective: minimize RMSE
- sampler: TPE (Tree-structured Parzen Estimator)
```

**Optimized Parameters (Example: XGBoost)**
```python
{
    'max_depth': [3, 10],
    'learning_rate': [0.01, 0.3],
    'n_estimators': [100, 1000],
    'subsample': [0.6, 1.0],
    'colsample_bytree': [0.6, 1.0],
    'gamma': [0, 5],
    'reg_alpha': [0, 1],
    'reg_lambda': [0, 1]
}
```

---

## 📝 Logging System

**Colored Console Output:**
- 🔵 **DEBUG** (Cyan): Detailed debugging info
- 🟢 **INFO** (Green): Progress and status
- 🟡 **WARNING** (Yellow): Warnings
- 🔴 **ERROR** (Red): Errors and exceptions

**File Logging:**
```
data/logBB_data/log/YYYYMMDD.log
```

---

## 🐛 Troubleshooting

<details>
<summary><b>Common Issues and Solutions</b></summary>

### 1. RDKit Import Error
```bash
# Solution
conda install -c conda-forge rdkit
```

### 2. Feature Dimension Mismatch
```bash
# Solution: Clear cached features
rm -rf MachineLearningModels/*/result/*.npy
```

### 3. Memory Issues
```python
# Solution: Reduce feature dimensions
- Use feature selection
- Reduce descriptor count
- Process in batches
```

### 4. Java Not Found (PaDEL)
```bash
# Solution: Install Java 8+
# Windows: Download from Oracle/AdoptOpenJDK
# Linux: sudo apt-get install default-jdk
# Mac: brew install openjdk@8
```

### 5. Model File Not Found
```bash
# Solution: Train the model first
cd MachineLearningModels/logBBModel
python XGBoost_FP_18F_Resample.py
```

</details>

---

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md).

### How to Contribute

1. **Fork** the repository
2. **Create** a feature branch (`git checkout -b feature/AmazingFeature`)
3. **Commit** your changes (`git commit -m 'Add AmazingFeature'`)
4. **Push** to the branch (`git push origin feature/AmazingFeature`)
5. **Open** a Pull Request

### Areas for Contribution

- 🐛 Bug fixes
- 📊 New model algorithms
- 🔬 Additional validation datasets
- 📝 Documentation improvements
- 🎨 Visualization enhancements
- ⚡ Performance optimizations

---

## 📚 Citation

If you use this code or dataset in your research, please cite:

```bibtex
@article{PETBD-QSAR-2024,
  title={Machine Learning QSAR Models for Blood-Brain Barrier Permeability 
         Based on PET Tracer Biodistribution Dataset},
  author={Your Name and Collaborators},
  journal={Journal Name},
  year={2024},
  publisher={Publisher}
}
```

---

## 📞 Contact

- **Author**: GDUT Computer Medical Science Team
- **Email**: your.email@example.com
- **GitHub**: [GDUT-Computer-Medical-Science-Team](https://github.com/GDUT-Computer-Medical-Science-Team)

---

## 🙏 Acknowledgments

- [RDKit](https://www.rdkit.org/) - Molecular informatics toolkit
- [Mordred](https://github.com/mordred-descriptor/mordred) - Molecular descriptor calculator
- [Optuna](https://optuna.org/) - Hyperparameter optimization framework
- [B3DB](http://www.cbligand.org/B3DB/) - External validation dataset

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

<div align="center">

**⭐ Star this repository if you find it helpful!**

Made with ❤️ by GDUT Computer Medical Science Team

</div>
