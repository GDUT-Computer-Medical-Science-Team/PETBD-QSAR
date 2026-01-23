"""
RF and XGBoost logBB models training on identical 18F-balanced data.

This script runs both Random Forest and XGBoost models on the same 18F-balanced
dataset using the complete pipeline from RF_FP_18F_Resample.py. Both models share
the same data splits, feature selection, and preprocessing steps to ensure fair comparison.
"""

import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime
import os
import sys
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, mean_absolute_percentage_error
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
import optuna
import tempfile
from padelpy import padeldescriptor
import joblib
import json
from tqdm import tqdm

sys.path.append('../../')
from utils.DataLogger import DataLogger

log = DataLogger(log_file=f"../../data/logBB_data/log/{datetime.now().strftime('%Y%m%d')}.log").getlog("rf_xgb_fp_18F_resample_combined")

ALLOWED_ISOTOPES = {"18F", "11C", "125I", "131I", "123I", "77Br", "76Br"}

def contains_18F(compound_index: str) -> bool:
    """Check if compound index contains 18F isotope."""
    if not isinstance(compound_index, str):
        return False
    text = compound_index.strip()
    return '18F' in text

def balance_18F_dataset(df: pd.DataFrame, method: str = "combined", seed: int = 42):
    """
    Balance 18F dataset using combined strategy (oversample minority class).
    
    Args:
        df: Input dataframe
        method: Balancing method ("combined", "oversample", "undersample")
        seed: Random seed
    
    Returns:
        Balanced dataframe
    """
    if method == "combined":
        df_copy = df.copy()
        df_copy["has_18F"] = df_copy["compound index"].apply(contains_18F)
        df_copy["label_18F"] = df_copy["has_18F"].astype(int)

        df_18f = df_copy[df_copy["has_18F"] == True]
        df_non_18f = df_copy[df_copy["has_18F"] == False]

        n_18f = len(df_18f)
        n_non_18f = len(df_non_18f)

        log.info(f"Original data distribution:")
        log.info(f"  Total samples: {len(df)}")
        log.info(f"  18F samples: {n_18f}")
        log.info(f"  Non-18F samples: {n_non_18f}")

        if n_18f < n_non_18f:
            extra_needed = n_non_18f - n_18f
            df_18f_extra = df_18f.sample(n=extra_needed, replace=True, random_state=seed)
            log.info(f"Generated additional 18F samples: {extra_needed}")
            combined = pd.concat([df_copy, df_18f_extra], ignore_index=True)
        elif n_non_18f < n_18f:
            extra_needed = n_18f - n_non_18f
            df_non_18f_extra = df_non_18f.sample(n=extra_needed, replace=True, random_state=seed)
            log.info(f"Generated additional non-18F samples: {extra_needed}")
            combined = pd.concat([df_copy, df_non_18f_extra], ignore_index=True)
        else:
            combined = df_copy
            log.info("Data already balanced, no additional sampling needed")

        combined = combined.sample(frac=1.0, random_state=seed).reset_index(drop=True)
        log.info(f"Total samples after combined strategy: {len(combined)} (original: {len(df)}, added: {len(combined) - len(df)})")

        return combined
    else:
        df = df.copy()
        df["has_18F"] = df["compound index"].apply(contains_18F)
        df["label_18F"] = df["has_18F"].astype(int)

        pos = df[df["label_18F"] == 1]
        neg = df[df["label_18F"] == 0]

        n_pos, n_neg = len(pos), len(neg)
        log.info(f"Original data distribution - 18F samples: {n_pos}, Non-18F samples: {n_neg}")

        if n_pos == 0 or n_neg == 0:
            raise ValueError("Cannot balance: one class has 0 samples")

        if method == "undersample":
            target = min(n_pos, n_neg)
            pos_bal = pos.sample(n=target, replace=False, random_state=seed)
            neg_bal = neg.sample(n=target, replace=False, random_state=seed)
        elif method == "oversample":
            target = max(n_pos, n_neg)
            pos_bal = pos.sample(n=target, replace=True, random_state=seed)
            neg_bal = neg.sample(n=target, replace=True, random_state=seed)
        else:
            raise ValueError("method must be 'undersample' or 'oversample'")

        balanced = pd.concat([pos_bal, neg_bal], ignore_index=True)
        balanced = balanced.sample(frac=1.0, random_state=seed).reset_index(drop=True)

        log.info(f"Balanced data distribution - 18F samples: {len(pos_bal)}, Non-18F samples: {len(neg_bal)}")

        return balanced

def calculate_padel_fingerprints(smiles_list):
    """Calculate PaDEL fingerprints for SMILES list."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.smi', delete=False) as temp_file:
        for smi in smiles_list:
            temp_file.write(f"{smi}\n")
        temp_smi_file = temp_file.name

    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            output_file = os.path.join(temp_dir, "fingerprints.csv")

            padeldescriptor(
                mol_dir=temp_smi_file,
                d_file=output_file,
                fingerprints=True,
                descriptortypes=None,
                detectaromaticity=True,
                standardizenitro=True,
                standardizetautomers=True,
                threads=2,
                removesalt=True,
                log=False
            )

            df = pd.read_csv(output_file)
            if 'Name' in df.columns:
                df = df.drop('Name', axis=1)
            return df
    except Exception as e:
        log.error(f"PaDEL Error: {str(e)}")
        return pd.DataFrame()
    finally:
        if os.path.exists(temp_smi_file):
            os.remove(temp_smi_file)

def calculate_morgan_fingerprints(smiles_list, radius=2, n_bits=1024):
    """Calculate Morgan fingerprints as fallback."""
    fingerprints = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            log.warning(f"Invalid SMILES: {smi}")
            fingerprints.append(np.zeros(n_bits))
            continue
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
        arr = np.zeros((n_bits,))
        DataStructs.ConvertToNumpyArray(fp, arr)
        fingerprints.append(arr)

    fp_df = pd.DataFrame(fingerprints)
    fp_df.columns = [f'Morgan_FP_{i}' for i in range(n_bits)]
    return fp_df

def adjusted_r2(y_true, y_pred, num_features):
    """Calculate adjusted R2 score."""
    n = len(y_true)
    r2 = r2_score(y_true, y_pred)
    if n - num_features - 1 == 0:
        return r2
    return 1 - (1 - r2) * (n - 1) / (n - num_features - 1)

def objective_rf(trial, X, y):
    """Optuna objective function for Random Forest."""
    param = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
        'max_depth': trial.suggest_int('max_depth', 3, 20),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 20),
        'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2']),
        'bootstrap': trial.suggest_categorical('bootstrap', [True, False]),
        'random_state': 42
    }

    X_train_split, X_valid_split, y_train_split, y_valid_split = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestRegressor(**param)
    model.fit(X_train_split, y_train_split)

    y_pred = model.predict(X_valid_split)
    r2 = r2_score(y_valid_split, y_pred)

    return r2

def objective_xgb(trial, X, y):
    """Optuna objective function for XGBoost."""
    param = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 2000),
        'max_depth': trial.suggest_int('max_depth', 3, 12),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 0, 50),
        'reg_lambda': trial.suggest_float('reg_lambda', 0, 50),
        'random_state': 42
    }

    X_train_split, X_valid_split, y_train_split, y_valid_split = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = XGBRegressor(**param)
    model.fit(X_train_split, y_train_split)

    y_pred = model.predict(X_valid_split)
    r2 = r2_score(y_valid_split, y_pred)

    return r2

def calculate_metrics(y_true, y_pred, n_features):
    """Calculate all regression metrics."""
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    mape = mean_absolute_percentage_error(y_true, y_pred)
    adj_r2 = adjusted_r2(y_true, y_pred, n_features)

    return {
        'mse': mse, 'rmse': rmse, 'mae': mae,
        'r2': r2, 'mape': mape, 'adj_r2': adj_r2
    }

def main():
    log.info("=============== Starting RF and XGBoost FP 18F Resample Training (Combined) ===============")

    # File paths
    petbd_data_file = "../../data/PTBD_v20240912.csv"
    petbd_features_file = "../../data/logBB_data/RF_XGB_FP_18F_balanced_features.csv"
    feature_index_file = "../../data/logBB_data/RF_XGB_FP_18F_feature_index.txt"

    os.makedirs("../../data/logBB_data/log", exist_ok=True)
    os.makedirs("./logbbModel", exist_ok=True)
    os.makedirs("./result", exist_ok=True)
    
    # Create output directory for RF and XGBoost results
    output_dir = './result/rfxgb'
    os.makedirs(output_dir, exist_ok=True)
    log.info(f"Output directory created: {output_dir}")

    if not os.path.exists(petbd_data_file):
        log.error("Missing PETBD dataset file")
        raise FileNotFoundError("Missing PETBD dataset")

    # Load data
    log.info("Reading PETBD dataset")
    df = pd.read_csv(petbd_data_file, encoding='utf-8')
    log.info(f"Original data shape: {df.shape}")

    df = df.dropna(subset=['logBB at60min']).reset_index(drop=True)
    log.info(f"Data shape after removing logBB missing values: {df.shape}")

    df_original = df.copy()
    log.info(f"Original dataset saved: {df_original.shape}")

    # Balance dataset
    log.info("Starting 18F combined strategy processing (original + 18F oversampling)")
    df_balanced = balance_18F_dataset(df, method="combined", seed=42)
    log.info(f"Training data shape after combined strategy: {df_balanced.shape}")

    df_train = df_balanced

    # Generate or load features
    log.info("Generating/loading features for training dataset (balanced)")

    if os.path.exists(petbd_features_file):
        log.info("Reading existing training dataset feature file")
        df_train_features = pd.read_csv(petbd_features_file, encoding='utf-8')

        required_columns = ['logBB at60min']
        missing_columns = [col for col in required_columns if col not in df_train_features.columns]

        if missing_columns or len(df_train_features) != len(df_train):
            log.warning(f"Feature file incomplete or size mismatch, regenerating features")
            need_regenerate = True
        else:
            X_train_full = df_train_features.drop(['SMILES', 'logBB at60min', 'compound index', 'has_18F', 'label_18F'], 
                                                  axis=1, errors='ignore')
            y_train_full = df_train_features['logBB at60min']
            need_regenerate = False
    else:
        need_regenerate = True

    if need_regenerate:
        log.info("Generating training dataset features")
        SMILES_train = df_train['SMILES']
        y_train_full = df_train['logBB at60min']

        log.info("Calculating PaDEL fingerprints (balanced dataset)")
        X_train_full = calculate_padel_fingerprints(SMILES_train)

        if X_train_full.empty:
            log.warning("PaDEL failed, using Morgan fingerprints")
            X_train_full = calculate_morgan_fingerprints(SMILES_train)

        log.info(f"Generated training feature matrix shape: {X_train_full.shape}")

        train_feature_data = pd.concat([
            df_train[['compound index', 'SMILES', 'logBB at60min', 'has_18F', 'label_18F']],
            X_train_full
        ], axis=1)

        train_feature_data.to_csv(petbd_features_file, encoding='utf-8', index=False)
        log.info(f"Features saved: {petbd_features_file}")

    log.info("Checking data consistency")
    log.info(f"Training data dimensions: X_train_full {X_train_full.shape}, y_train_full {len(y_train_full)}")

    if len(y_train_full) != X_train_full.shape[0]:
        log.warning(f"Dimension mismatch: X_train_full {X_train_full.shape[0]} vs y_train_full {len(y_train_full)}")
        min_len = min(len(y_train_full), X_train_full.shape[0])
        X_train_full = X_train_full.iloc[:min_len]
        y_train_full = y_train_full.iloc[:min_len]
        log.info(f"Adjusted training data to consistent dimension: {min_len}")

    # Process features
    log.info("Processing non-numeric features")
    X_train_full = X_train_full.apply(pd.to_numeric, errors='coerce')
    X_train_full = X_train_full.fillna(0)
    X_train_full.columns = X_train_full.columns.astype(str)

    log.info(f"Training feature matrix shape: {X_train_full.shape}")

    # Feature selection
    if not os.path.exists(feature_index_file):
        log.info("Performing feature selection")
        from sklearn.feature_selection import RFE

        estimator = RandomForestRegressor(n_estimators=100, random_state=42)
        selector = RFE(estimator, n_features_to_select=min(50, X_train_full.shape[1]), step=1)
        selector.fit(X_train_full, y_train_full)

        selected_indices = np.where(selector.support_)[0]
        np.savetxt(feature_index_file, selected_indices, fmt='%d', delimiter=',')
        X_train_full = X_train_full.iloc[:, selected_indices]
        log.info(f"Feature selection complete, Training set: {X_train_full.shape}")
    else:
        log.info("Loading existing feature indices")
        selected_indices = np.loadtxt(feature_index_file, dtype=int, delimiter=',')

        valid_indices = selected_indices[selected_indices < X_train_full.shape[1]]
        if len(valid_indices) != len(selected_indices):
            log.warning(f"Feature indices out of range, adjusting")
            selected_indices = valid_indices

        X_train_full = X_train_full.iloc[:, selected_indices]
        log.info(f"Feature selection applied, Training set: {X_train_full.shape}")

    # Dataset split
    log.info("Dataset split (balanced training data)")

    X_train_val, X_test_balanced, y_train_val, y_test_balanced = train_test_split(
        X_train_full, y_train_full, test_size=0.1, random_state=42
    )
    log.info(f"Train+Val set: {len(X_train_val)} samples, Test set: {len(X_test_balanced)} samples")

    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=0.1, random_state=42
    )
    log.info(f"Training set: {len(X_train)} samples, Validation set: {len(X_val)} samples")

    # Feature normalization
    log.info("Feature normalization")
    scaler = MinMaxScaler()
    X_train_val_scaled = scaler.fit_transform(X_train_val)
    X_test_balanced_scaled = scaler.transform(X_test_balanced)

    X_train_val = pd.DataFrame(X_train_val_scaled, columns=X_train_val.columns)
    X_test_balanced = pd.DataFrame(X_test_balanced_scaled, columns=X_test_balanced.columns)

    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=0.1, random_state=42
    )

    y_train_val = y_train_val.reset_index(drop=True)
    y_train = y_train.reset_index(drop=True)
    y_val = y_val.reset_index(drop=True)
    y_test_balanced = y_test_balanced.reset_index(drop=True)

    # =============== Train Random Forest ===============
    log.info("\n=============== Training Random Forest Model ===============")
    
    # Perform Random Forest hyperparameter optimization
    log.info("Starting RF hyperparameter search (100 trials)...")
    study_rf = optuna.create_study(direction='maximize')
    study_rf.optimize(lambda trial: objective_rf(trial, X_train_val, y_train_val), n_trials=100)

    best_params_rf = study_rf.best_params
    best_params_rf['random_state'] = 42
    log.info(f"Best RF hyperparameters: {best_params_rf}")
    log.info(f"Best RF R?: {study_rf.best_value:.4f}")

    # Train final RF model
    log.info("Training final RF model")
    final_model_rf = RandomForestRegressor(**best_params_rf)
    final_model_rf.fit(X_train, y_train)

    y_pred_train_rf = final_model_rf.predict(X_train)
    y_pred_val_rf = final_model_rf.predict(X_val)
    y_pred_test_rf = final_model_rf.predict(X_test_balanced)

    train_metrics_rf = calculate_metrics(y_train, y_pred_train_rf, X_train.shape[1])
    val_metrics_rf = calculate_metrics(y_val, y_pred_val_rf, X_val.shape[1])
    test_metrics_rf = calculate_metrics(y_test_balanced, y_pred_test_rf, X_test_balanced.shape[1])

    log.info("======== RF Test Set Results ========")
    log.info(f"MSE: {test_metrics_rf['mse']:.4f}")
    log.info(f"RMSE: {test_metrics_rf['rmse']:.4f}")
    log.info(f"MAE: {test_metrics_rf['mae']:.4f}")
    log.info(f"R?: {test_metrics_rf['r2']:.4f}")
    log.info(f"Adjusted R?: {test_metrics_rf['adj_r2']:.4f}")
    log.info(f"MAPE: {test_metrics_rf['mape']:.2f}%")

    # Save RF model
    model_path_rf = "./logbbModel/rf_fp_18F_combined_model.joblib"
    joblib.dump(final_model_rf, model_path_rf)
    log.info(f"RF Model saved to: {model_path_rf}")

    # =============== Train XGBoost ===============
    log.info("\n=============== Training XGBoost Model ===============")
    
    # Perform XGBoost hyperparameter optimization
    log.info("Starting XGBoost hyperparameter search (100 trials)...")
    study_xgb = optuna.create_study(direction='maximize')
    study_xgb.optimize(lambda trial: objective_xgb(trial, X_train_val, y_train_val), n_trials=100)

    best_params_xgb = study_xgb.best_params
    best_params_xgb['random_state'] = 42
    log.info(f"Best XGBoost hyperparameters: {best_params_xgb}")
    log.info(f"Best XGBoost R?: {study_xgb.best_value:.4f}")

    # Train final XGBoost model
    log.info("Training final XGBoost model")
    final_model_xgb = XGBRegressor(**best_params_xgb)
    final_model_xgb.fit(X_train, y_train)

    y_pred_train_xgb = final_model_xgb.predict(X_train)
    y_pred_val_xgb = final_model_xgb.predict(X_val)
    y_pred_test_xgb = final_model_xgb.predict(X_test_balanced)

    train_metrics_xgb = calculate_metrics(y_train, y_pred_train_xgb, X_train.shape[1])
    val_metrics_xgb = calculate_metrics(y_val, y_pred_val_xgb, X_val.shape[1])
    test_metrics_xgb = calculate_metrics(y_test_balanced, y_pred_test_xgb, X_test_balanced.shape[1])

    log.info("======== XGBoost Test Set Results ========")
    log.info(f"MSE: {test_metrics_xgb['mse']:.4f}")
    log.info(f"RMSE: {test_metrics_xgb['rmse']:.4f}")
    log.info(f"MAE: {test_metrics_xgb['mae']:.4f}")
    log.info(f"R?: {test_metrics_xgb['r2']:.4f}")
    log.info(f"Adjusted R?: {test_metrics_xgb['adj_r2']:.4f}")
    log.info(f"MAPE: {test_metrics_xgb['mape']:.2f}%")

    # Save XGBoost model
    model_path_xgb = "./logbbModel/xgb_fp_18F_combined_model.joblib"
    joblib.dump(final_model_xgb, model_path_xgb)
    log.info(f"XGBoost Model saved to: {model_path_xgb}")

    # =============== Save Predictions ===============
    log.info("\n=============== Saving Predictions ===============")

    # RF predictions
    rf_train_df = pd.DataFrame({
        'True_Values': y_train.values,
        'Predicted_Values': y_pred_train_rf,
        'Dataset': 'Train',
        'Model': 'RandomForest'
    })
    rf_val_df = pd.DataFrame({
        'True_Values': y_val.values,
        'Predicted_Values': y_pred_val_rf,
        'Dataset': 'Validation',
        'Model': 'RandomForest'
    })
    rf_test_df = pd.DataFrame({
        'True_Values': y_test_balanced.values,
        'Predicted_Values': y_pred_test_rf,
        'Dataset': 'Test',
        'Model': 'RandomForest'
    })

    # XGBoost predictions
    xgb_train_df = pd.DataFrame({
        'True_Values': y_train.values,
        'Predicted_Values': y_pred_train_xgb,
        'Dataset': 'Train',
        'Model': 'XGBoost'
    })
    xgb_val_df = pd.DataFrame({
        'True_Values': y_val.values,
        'Predicted_Values': y_pred_val_xgb,
        'Dataset': 'Validation',
        'Model': 'XGBoost'
    })
    xgb_test_df = pd.DataFrame({
        'True_Values': y_test_balanced.values,
        'Predicted_Values': y_pred_test_xgb,
        'Dataset': 'Test',
        'Model': 'XGBoost'
    })

    # Save Random Forest predictions separately (train, val, test)
    rf_train_df.to_csv(f'{output_dir}/rf_18F_train_predictions.csv', index=False)
    rf_val_df.to_csv(f'{output_dir}/rf_18F_val_predictions.csv', index=False)
    rf_test_df.to_csv(f'{output_dir}/rf_18F_test_predictions.csv', index=False)
    log.info("Random Forest predictions saved:")
    log.info(f"  - {output_dir}/rf_18F_train_predictions.csv")
    log.info(f"  - {output_dir}/rf_18F_val_predictions.csv")
    log.info(f"  - {output_dir}/rf_18F_test_predictions.csv")
    
    # Save XGBoost predictions separately (train, val, test)
    xgb_train_df.to_csv(f'{output_dir}/xgb_18F_train_predictions.csv', index=False)
    xgb_val_df.to_csv(f'{output_dir}/xgb_18F_val_predictions.csv', index=False)
    xgb_test_df.to_csv(f'{output_dir}/xgb_18F_test_predictions.csv', index=False)
    log.info("XGBoost predictions saved:")
    log.info(f"  - {output_dir}/xgb_18F_train_predictions.csv")
    log.info(f"  - {output_dir}/xgb_18F_val_predictions.csv")
    log.info(f"  - {output_dir}/xgb_18F_test_predictions.csv")
    
    # Combine all predictions
    all_predictions = pd.concat([
        rf_train_df, rf_val_df, rf_test_df,
        xgb_train_df, xgb_val_df, xgb_test_df
    ], ignore_index=True)
    
    all_predictions.to_csv(f'{output_dir}/rf_xgb_18F_combined_all_predictions.csv', index=False)
    log.info(f"Combined predictions saved: {output_dir}/rf_xgb_18F_combined_all_predictions.csv")

    # =============== Generate Scatter Plots ===============
    log.info("Generating scatter plots for RF and XGBoost")
    
    # Random Forest scatter plot
    fig_rf, axes_rf = plt.subplots(1, 3, figsize=(18, 5))
    
    # RF Training set
    axes_rf[0].scatter(y_train, y_pred_train_rf, alpha=0.6, edgecolors='k', linewidths=0.5)
    axes_rf[0].plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], 'r--', lw=2)
    axes_rf[0].set_xlabel('True logBB', fontsize=12)
    axes_rf[0].set_ylabel('Predicted logBB', fontsize=12)
    axes_rf[0].set_title(f'RF Training Set (R?={train_metrics_rf["r2"]:.3f})', fontsize=12)
    axes_rf[0].grid(True, alpha=0.3)
    
    # RF Validation set
    axes_rf[1].scatter(y_val, y_pred_val_rf, alpha=0.6, edgecolors='k', linewidths=0.5, color='green')
    axes_rf[1].plot([y_val.min(), y_val.max()], [y_val.min(), y_val.max()], 'r--', lw=2)
    axes_rf[1].set_xlabel('True logBB', fontsize=12)
    axes_rf[1].set_ylabel('Predicted logBB', fontsize=12)
    axes_rf[1].set_title(f'RF Validation Set (R?={val_metrics_rf["r2"]:.3f})', fontsize=12)
    axes_rf[1].grid(True, alpha=0.3)
    
    # RF Test set
    axes_rf[2].scatter(y_test_balanced, y_pred_test_rf, alpha=0.6, edgecolors='k', linewidths=0.5, color='red')
    axes_rf[2].plot([y_test_balanced.min(), y_test_balanced.max()], 
                    [y_test_balanced.min(), y_test_balanced.max()], 'r--', lw=2)
    axes_rf[2].set_xlabel('True logBB', fontsize=12)
    axes_rf[2].set_ylabel('Predicted logBB', fontsize=12)
    axes_rf[2].set_title(f'RF Test Set (R?={test_metrics_rf["r2"]:.3f})', fontsize=12)
    axes_rf[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/rf_18F_scatter_plots.png', dpi=300, bbox_inches='tight')
    plt.close()
    log.info(f"Random Forest scatter plots saved: {output_dir}/rf_18F_scatter_plots.png")
    
    # XGBoost scatter plot
    fig_xgb, axes_xgb = plt.subplots(1, 3, figsize=(18, 5))
    
    # XGBoost Training set
    axes_xgb[0].scatter(y_train, y_pred_train_xgb, alpha=0.6, edgecolors='k', linewidths=0.5)
    axes_xgb[0].plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], 'r--', lw=2)
    axes_xgb[0].set_xlabel('True logBB', fontsize=12)
    axes_xgb[0].set_ylabel('Predicted logBB', fontsize=12)
    axes_xgb[0].set_title(f'XGBoost Training Set (R?={train_metrics_xgb["r2"]:.3f})', fontsize=12)
    axes_xgb[0].grid(True, alpha=0.3)
    
    # XGBoost Validation set
    axes_xgb[1].scatter(y_val, y_pred_val_xgb, alpha=0.6, edgecolors='k', linewidths=0.5, color='green')
    axes_xgb[1].plot([y_val.min(), y_val.max()], [y_val.min(), y_val.max()], 'r--', lw=2)
    axes_xgb[1].set_xlabel('True logBB', fontsize=12)
    axes_xgb[1].set_ylabel('Predicted logBB', fontsize=12)
    axes_xgb[1].set_title(f'XGBoost Validation Set (R?={val_metrics_xgb["r2"]:.3f})', fontsize=12)
    axes_xgb[1].grid(True, alpha=0.3)
    
    # XGBoost Test set
    axes_xgb[2].scatter(y_test_balanced, y_pred_test_xgb, alpha=0.6, edgecolors='k', linewidths=0.5, color='red')
    axes_xgb[2].plot([y_test_balanced.min(), y_test_balanced.max()], 
                     [y_test_balanced.min(), y_test_balanced.max()], 'r--', lw=2)
    axes_xgb[2].set_xlabel('True logBB', fontsize=12)
    axes_xgb[2].set_ylabel('Predicted logBB', fontsize=12)
    axes_xgb[2].set_title(f'XGBoost Test Set (R?={test_metrics_xgb["r2"]:.3f})', fontsize=12)
    axes_xgb[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/xgb_18F_scatter_plots.png', dpi=300, bbox_inches='tight')
    plt.close()
    log.info(f"XGBoost scatter plots saved: {output_dir}/xgb_18F_scatter_plots.png")

    # =============== Generate Comparison Report ===============
    comparison_report = {
        'experiment_info': {
            'dataset': 'PTBD_v20240912.csv',
            'training_method': '18F balanced resampling',
            'features': 'Molecular_fingerprints (PaDEL/Morgan)',
            'n_samples_original': len(df_original),
            'n_samples_balanced_training': len(df_train),
            'n_features': X_train_full.shape[1],
            'seed': 42
        },
        'random_forest': {
            'best_params': best_params_rf,
            'train_metrics': train_metrics_rf,
            'validation_metrics': val_metrics_rf,
            'test_metrics': test_metrics_rf
        },
        'xgboost': {
            'best_params': best_params_xgb,
            'train_metrics': train_metrics_xgb,
            'validation_metrics': val_metrics_xgb,
            'test_metrics': test_metrics_xgb
        }
    }

    with open(f'{output_dir}/rf_xgb_18F_combined_report.json', 'w', encoding='utf-8') as f:
        json.dump(comparison_report, f, ensure_ascii=False, indent=2)

    log.info("======== Experiment Complete ========")
    log.info(f"All results saved to {output_dir}/:")
    log.info("  Random Forest:")
    log.info("    - rf_18F_train_predictions.csv")
    log.info("    - rf_18F_val_predictions.csv")
    log.info("    - rf_18F_test_predictions.csv")
    log.info("    - rf_18F_scatter_plots.png")
    log.info("  XGBoost:")
    log.info("    - xgb_18F_train_predictions.csv")
    log.info("    - xgb_18F_val_predictions.csv")
    log.info("    - xgb_18F_test_predictions.csv")
    log.info("    - xgb_18F_scatter_plots.png")
    log.info("  Combined:")
    log.info("    - rf_xgb_18F_combined_all_predictions.csv")
    log.info("    - rf_xgb_18F_combined_report.json")

    # Print comparison summary
    print("\n" + "="*70)
    print("Model Comparison Summary (Test Set)")
    print("="*70)
    print(f"{'Metric':<15} {'Random Forest':>20} {'XGBoost':>20}")
    print("-"*70)
    print(f"{'R?':<15} {test_metrics_rf['r2']:>20.4f} {test_metrics_xgb['r2']:>20.4f}")
    print(f"{'Adjusted R?':<15} {test_metrics_rf['adj_r2']:>20.4f} {test_metrics_xgb['adj_r2']:>20.4f}")
    print(f"{'RMSE':<15} {test_metrics_rf['rmse']:>20.4f} {test_metrics_xgb['rmse']:>20.4f}")
    print(f"{'MAE':<15} {test_metrics_rf['mae']:>20.4f} {test_metrics_xgb['mae']:>20.4f}")
    print(f"{'MAPE (%)':<15} {test_metrics_rf['mape']:>20.2f} {test_metrics_xgb['mape']:>20.2f}")
    print("="*70)

if __name__ == '__main__':
    main()

