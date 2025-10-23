# PETBD-QSAR

<div align="center">
Machine Learning QSAR Prediction Models for Blood-Brain Barrier Permeability Based on PET Tracer Datasets

![Python](https://img.shields.io/badge/Python-3.8-blue)
![License](https://img.shields.io/badge/License-MIT-green)

</div>

## 📖 Project Overview

This project builds QSAR prediction models for blood-brain barrier permeability using machine learning methods based on a novel PET tracer biodistribution dataset (PETBD). Key features:

- 📊 **First PET Tracer Dataset**: Contains in vivo biodistribution data for 816 small molecule PET tracers
- 🧬 **Multi-Organ Distribution Data**: Includes drug concentration data for 14 typical organs at 60 minutes post-intravenous injection
- 🧠 **BBB Permeability Prediction**: Predicts blood-brain barrier penetration coefficient (logBB) and brain drug concentration
- 🔄 **Data Integration Tools**: Tools for integrating PET tracer research data from published literature

## 🛠️ Environment Setup

```bash
# Create and activate environment
conda create -n MDM python=3.8
conda activate MDM

# Install core dependencies
conda install numpy pandas scikit-learn rdkit

# Install machine learning frameworks
pip install optuna xgboost lightgbm catboost

# Install molecular descriptor tools
pip install mordred padelpy

# Install data processing tools
pip install openpyxl==3.0.10
pip install "pandas<2.0.0"
pip install PyYAML==6.0
pip install tqdm==4.64.1
pip install xlrd==2.0.1
pip install colorlog joblib
```

## 📁 Project Structure

```
├── MachineLearningModels/    # Machine learning models
│   ├── logBBModel/          # logBB prediction models
│   │   ├── *_PETBD_18F_Resample.py    # 18F isotope resampling models
│   │   ├── *_FP_18F_Resample.py       # Fingerprint + 18F models
│   │   └── B3DB_*.py                  # B3DB validation scripts
│   ├── CbrainModel/         # Brain concentration prediction models
│   │   └── *_Cbrain_18F_Resample.py   # Cbrain 18F models
│   └── VivoValidation/      # In vivo validation
├── MedicalDatasetsMerger/   # PET dataset integration tool
├── utils/                   # Utility classes
├── preprocess/              # Data preprocessing
├── data/                    # Data storage
├── model/                   # Pre-trained model storage
└── result/                  # Output results
```

## 🔬 Model Algorithms

The project implements six machine learning algorithms with 18F isotope resampling:

| Algorithm | LogBB Model | Cbrain Model | Features |
|-----------|-------------|--------------|----------|
| XGBoost | ✓ | ✓ | Gradient boosting |
| Random Forest | ✓ | ✓ | Ensemble learning |
| LightGBM | ✓ | ✓ | Fast gradient boosting |
| CatBoost | ✓ | ✓ | Categorical boosting |
| MLP | ✓ | ✓ | Neural network |
| SVM | ✓ | ✓ | Support vector machine |

## 🧪 18F Isotope Resampling

The project implements isotope-based dataset balancing to address class imbalance:

**Resampling Strategies:**
- **Oversample**: Replicate minority class samples to match majority class
- **Undersample**: Reduce majority class samples to match minority class
- **Combined** (Recommended): Keep all original data + add oversampled minority samples

**Key Implementation Files:**
- `XGBoost_PETBD_18F_Resample.py` - XGBoost with 18F resampling
- `RF_PETBD_18F_Resample.py` - Random Forest with 18F resampling
- `LightGBM_PETBD_18F_Resample.py` - LightGBM with 18F resampling
- `CatBoost_PETBD_18F_Resample.py` - CatBoost with 18F resampling
- `MLP_PETBD_18F_Resample.py` - Neural network with 18F resampling
- `SVM_PETBD_18F_Resample.py` - SVM with 18F resampling

## 📊 Evaluation Metrics

Model performance is evaluated using the following metrics:

| Metric | Description |
|--------|-------------|
| MSE | Mean Squared Error |
| RMSE | Root Mean Squared Error |
| R² | Coefficient of Determination |
| Adjusted R² | Adjusted R² |
| MAE | Mean Absolute Error |
| MAPE | Mean Absolute Percentage Error |

## 💻 Usage

### Data Preparation
1. Place data files (PETBD_v20240912.csv/OrganDataAt60min.csv) in the data/ directory
2. Data files must contain the following columns:
   - SMILES: Molecular SMILES representation
   - compound index: Compound name with isotope information
   - logBB at60min: Blood-brain barrier penetration coefficient
   - brain mean60min: Brain concentration value (for Cbrain models)
   - Metadata: animal type, gender, animal weight (g), injection dosage (μCi)

### Model Training

**LogBB Models:**
```bash
cd MachineLearningModels/logBBModel

# Train with 18F resampling
python XGBoost_PETBD_18F_Resample.py
python RF_PETBD_18F_Resample.py
python LightGBM_PETBD_18F_Resample.py
python CatBoost_PETBD_18F_Resample.py
python MLP_PETBD_18F_Resample.py
python SVM_PETBD_18F_Resample.py
```

**Cbrain Models:**
```bash
cd MachineLearningModels/CbrainModel

# Train brain concentration models
python XGBoost_Cbrain_18F_Resample.py
python RF_Cbrain_18F_Resample.py
python LightGBM_Cbrain_18F_Resample.py
python CatBoost_Cbrain_18F_Resample.py
python MLP_Cbrain_18F_Resample.py
python SVM_Cbrain_18F_Resample.py
```

**External Validation:**
```bash
cd MachineLearningModels/VivoValidation

# Validate with experimental data
python Vivo_XGB_FP.py
python vivo_xgb_Mordred.py
python Vivo_LightGBM_Mordred.py
```

### Output Files
- Prediction results saved in result/ directory
- Log files saved in data/log/ directory
- Model files saved in model/ or *Model/ directories
- JSON reports with comprehensive metrics

## 🔍 Feature Engineering

### Molecular Fingerprints
- Morgan fingerprints generated using RDKit (radius=2, 1024-bit)
- PaDEL fingerprints via padelpy wrapper

### Mordred Descriptors
- 2D molecular descriptors
- Includes topological features, physicochemical properties
- Automatic removal of highly correlated features

### Metadata Features
- Animal type (mouse/rat encoding)
- Gender (male/female encoding)
- Animal weight (g, median imputation)
- Injection dosage (μCi, median imputation)

### Combined Feature Models
- Stratified feature selection maintaining representation from all types
- Example: 21 Morgan + 25 Mordred + 4 Metadata = 50 features

## ⚙️ Hyperparameter Optimization

All models use Optuna for automatic hyperparameter tuning (typically 100 trials):

**Common Parameters:**
- Learning rate
- Tree depth (for tree-based models)
- Feature sampling ratio
- Regularization parameters
- Number of iterations/estimators

**Cross-Validation:**
- 10-fold cross-validation on training data
- Separate train/validation/test split for evaluation

## 📝 Logging System

The system uses a custom DataLogger class with colored output:
- **DEBUG** (cyan): Detailed debugging information
- **INFO** (green): General information and progress
- **WARNING** (yellow): Warning messages
- **ERROR** (red): Error conditions
- **File logging**: Saves to `data/*/log/{YYYYMMDD}.log`

## 🔧 Troubleshooting

Common issues and solutions:

1. **Feature Dimension Mismatch**
   - Ensure feature selection maintains consistent dimensions
   - Check preprocessing steps are correct
   - Delete cached feature files to force recalculation

2. **Missing Model Files**
   - Ensure model files are in correct path
   - Check model filenames are correct
   - Verify .joblib files are not corrupted

3. **Memory Issues**
   - Reduce batch processing size
   - Use feature selection to reduce dimensionality
   - Monitor memory during descriptor calculation

4. **Environment Issues**
   - Always activate conda environment: `conda activate MDM`
   - Verify Python version is 3.8: `python --version`
   - Check all dependencies: `pip install -r requirements.txt`

5. **PaDEL Calculation Errors**
   - Ensure Java 8+ is installed and in system PATH
   - Verify SMILES strings are valid
   - Check RDKit can parse all molecules

## 📄 License

MIT License
