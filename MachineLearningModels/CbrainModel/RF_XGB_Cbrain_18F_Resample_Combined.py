# -*- coding: utf-8 -*-
"""
RF and XGBoost Cbrain models training on identical 18F-balanced data.

This script runs both Random Forest and XGBoost models on the same Cbrain
18F-resampled dataset using a unified pipeline from RF_Cbrain_18F_Resample.py.
Both models share the same data splits, feature selection, and preprocessing 
steps to ensure fair comparison.
"""

import sys
import os
from time import time
import numpy as np
import pandas as pd
import optuna
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, mean_absolute_percentage_error
from sklearn.model_selection import train_test_split, KFold
from tqdm import tqdm
from datetime import datetime
import joblib
import json
import matplotlib.pyplot as plt
from typing import Optional

# Add project path to Python path
sys.path.append('../../')
from preprocess.data_preprocess.FeatureExtraction import FeatureExtraction
from preprocess.data_preprocess.data_preprocess_utils import calculate_Mordred_desc
from utils.DataLogger import DataLogger

# Create necessary directories
os.makedirs("../../data/logBB_data/log", exist_ok=True)
os.makedirs("../../data/logBB_data", exist_ok=True)
os.makedirs("./cbrainmodel", exist_ok=True)
os.makedirs("./result", exist_ok=True)
os.makedirs("./result/rfxgb", exist_ok=True)

# Initialize logger
log = DataLogger(log_file=f"../../data/logBB_data/log/{datetime.now().strftime('%Y%m%d')}.log").getlog(
    "cbrain_rf_xgb_18F_combined")

# Allowed isotopes list (for 18F detection)
ALLOWED_ISOTOPES = {"18F", "11C", "125I", "131I", "123I", "77Br", "76Br"}


def extract_18F_simple(compound_index: str) -> Optional[str]:
    """
    Extract isotope information from compound index (simplified version).
    Simplified to only recognize 18F, more direct and efficient.
    """
    if not isinstance(compound_index, str):
        return None

    text = compound_index.strip()

    # Check if contains 18F
    if "18F" in text:
        return "18F"
    else:
        return "non-18F"


def balance_18F_dataset(df: pd.DataFrame, method: str = "oversample", seed: int = 42):
    """
    Balance dataset based on 18F labeling - original data + minority class resampling strategy.
    1. Keep all original data
    2. Only resample the minority class (non-18F)
    3. Merge original data with resampled minority class data
    """
    log.info(f"Original data samples: {len(df)}")

    # 1. Keep complete original data
    df_original = df.copy()

    # 2. Extract isotope information
    isotopes = [extract_18F_simple(x) for x in df["compound index"].fillna("")]
    df_work = df.copy()
    df_work["isotope"] = isotopes

    # Only keep rows with recognizable isotopes for analysis
    df_isotope = df_work[df_work["isotope"].notna()].reset_index(drop=True)

    if len(df_isotope) == 0:
        log.warning("No recognizable isotope samples, returning original data")
        return df_original

    # Create 18F binary label
    df_isotope["label_18F"] = (df_isotope["isotope"] == "18F").astype(int)

    pos = df_isotope[df_isotope["label_18F"] == 1]  # 18F samples
    neg = df_isotope[df_isotope["label_18F"] == 0]  # non-18F samples

    n_pos, n_neg = len(pos), len(neg)
    log.info(f"Isotope distribution - 18F samples: {n_pos}, non-18F samples: {n_neg}")

    if n_pos == 0 or n_neg == 0:
        log.warning("One class has 0 samples, returning original data")
        return df_original

    # 3. Determine which minority class needs resampling
    if n_pos > n_neg:
        minority_samples = neg
        target_size = n_pos
        minority_type = "non-18F"
    else:
        minority_samples = pos
        target_size = n_neg
        minority_type = "18F"

    # Calculate number of additional samples needed
    current_size = len(minority_samples)
    extra_needed = target_size - current_size

    if extra_needed <= 0:
        log.info("Data already balanced, returning original data")
        return df_original

    # 4. Resample minority class
    extra_samples = minority_samples.sample(n=extra_needed, replace=True, random_state=seed)
    extra_clean = extra_samples.drop(['isotope', 'label_18F'], axis=1, errors='ignore')

    log.info(f"Generated additional samples for {minority_type} class: {extra_needed}")

    # 5. Merge original data with additional samples
    combined_df = pd.concat([df_original, extra_clean], ignore_index=True)
    combined_df = combined_df.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    # Add isotope labels to final dataset
    combined_isotopes = [extract_18F_simple(x) for x in combined_df["compound index"].fillna("")]
    combined_df["isotope"] = combined_isotopes
    combined_df["label_18F"] = (combined_df["isotope"] == "18F").astype(int)

    log.info(f"Final combined data: {len(combined_df)} samples")
    log.info(f"  - Original data: {len(df_original)} samples")
    log.info(f"  - Additional {minority_type} samples: {extra_needed} samples")

    return combined_df


def adjusted_r2_score(r2, n, k):
    """Calculate adjusted R-squared value"""
    if n - k - 1 <= 0:
        return r2
    return 1 - (1 - r2) * (n - 1) / (n - k - 1)


def calculate_metrics(y_true, y_pred, n_features):
    """Calculate all regression metrics"""
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    mape = mean_absolute_percentage_error(y_true, y_pred)
    adj_r2 = adjusted_r2_score(r2, len(y_true), n_features)

    return {
        'mse': float(mse), 'rmse': float(rmse), 'mae': float(mae),
        'r2': float(r2), 'mape': float(mape), 'adj_r2': float(adj_r2)
    }


def objective_rf(trial, X, y, seed):
    """Optuna objective function - Random Forest"""
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.1, random_state=seed
    )

    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
        'max_depth': trial.suggest_int('max_depth', 3, 20),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 20),
        'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2']),
        'bootstrap': trial.suggest_categorical('bootstrap', [True, False]),
        'random_state': seed
    }

    model = RandomForestRegressor(**params)
    model.fit(X_tr, y_tr)
    y_pred = model.predict(X_te)

    return r2_score(y_te, y_pred)


def objective_xgb(trial, X, y, seed):
    """Optuna objective function - XGBoost"""
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.1, random_state=seed
    )

    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 3000),
        'max_depth': trial.suggest_int('max_depth', 3, 12),
        'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.3, log=True),
        'min_child_weight': trial.suggest_float('min_child_weight', 1e-8, 100.0, log=True),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'gamma': trial.suggest_float('gamma', 1e-8, 1.0, log=True),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
        'random_state': seed
    }

    model = XGBRegressor(**params)
    model.fit(X_tr, y_tr)
    y_pred = model.predict(X_te)

    return r2_score(y_te, y_pred)


def main():
    log.info("=" * 80)
    log.info("Starting Cbrain dataset RF and XGBoost 18F resampling training (combined version)")
    log.info("=" * 80)

    # File path configuration - Use PETBD dataset for 18F resampling training
    cbrain_data_file = "../../data/PTBD_v20240912.csv"
    cbrain_features_file = "../../data/logBB_data/cbrain_rf_xgb_petbd_18F_balanced_features.csv"
    feature_index_file = "../../data/logBB_data/cbrain_rf_xgb_petbd_18F_feature_index.txt"

    # Model parameters
    smile_column_name = 'SMILES'
    pred_column_name = 'brain at60min'
    RFE_features_to_select = 50
    n_optuna_trial = 100
    cv_times = 10
    seed = int(time())  # Dynamic random seed (same as single model versions)

    # Output directory for RF and XGBoost results
    output_dir = './result/rfxgb'
    log.info(f"Output directory: {output_dir}")

    # Read PETBD data
    if not os.path.exists(cbrain_data_file):
        raise FileNotFoundError(f"Missing PTBD dataset: {cbrain_data_file}")

    log.info("Reading PTBD dataset")
    df = pd.read_csv(cbrain_data_file, encoding='utf-8')
    log.info(f"Original data shape: {df.shape}")

    # Remove rows with missing brain at60min and SMILES
    df = df.dropna(subset=[pred_column_name, smile_column_name])
    log.info(f"Data shape after removing {pred_column_name} and {smile_column_name} missing values: {df.shape}")

    # Perform 18F resampling balancing
    log.info("Starting 18F resampling balancing process")
    df_balanced = balance_18F_dataset(df, method="oversample", seed=seed)
    log.info(f"Data shape after 18F resampling: {df_balanced.shape}")

    # Feature extraction or loading
    if os.path.exists(cbrain_features_file):
        log.info("Feature file exists, loading")
        features_df = pd.read_csv(cbrain_features_file, encoding='utf-8')

        # Ensure feature file matches balanced data
        if len(features_df) != len(df_balanced):
            log.info("Feature file does not match balanced data, regenerating features")
            os.remove(cbrain_features_file)
            if os.path.exists(feature_index_file):
                os.remove(feature_index_file)
        else:
            y = features_df[pred_column_name]
            X = features_df.drop([smile_column_name, pred_column_name], axis=1)
            # Remove label columns
            label_cols = ['isotope', 'label_18F', 'compound index']
            for col in label_cols:
                if col in X.columns:
                    X = X.drop(col, axis=1)

    # If feature file does not exist or needs regeneration
    if not os.path.exists(cbrain_features_file):
        log.info("Starting feature generation work (based on 18F resampled data)")

        y = df_balanced[pred_column_name]
        SMILES = df_balanced[smile_column_name]

        log.info("Calculating Mordred molecular descriptors")
        X_mordred = calculate_Mordred_desc(SMILES)

        log.info("Using only Mordred descriptors")
        X = X_mordred

        # Remove SMILES column if exists
        if smile_column_name in X.columns:
            X = X.drop(smile_column_name, axis=1)

        log.info(f"Feature matrix shape: {X.shape}")

        # Save feature data
        if 'isotope' not in df_balanced.columns:
            isotopes = [extract_18F_simple(x) for x in df_balanced["compound index"].fillna("")]
            df_balanced["isotope"] = isotopes
            df_balanced["label_18F"] = (df_balanced["isotope"] == "18F").astype(int)

        feature_data = pd.concat([
            df_balanced[['compound index', smile_column_name, pred_column_name, 'isotope', 'label_18F']],
            X
        ], axis=1)

        feature_data.to_csv(cbrain_features_file, encoding='utf-8', index=False)
        log.info(f"Feature data saved to: {cbrain_features_file}")

    # Ensure all features are numeric
    log.info("Processing non-numeric features")
    X = X.apply(pd.to_numeric, errors='coerce')
    X = X.fillna(0)
    X.columns = X.columns.astype(str)
    log.info(f"Feature matrix shape after processing: {X.shape}")

    # Feature selection
    if not os.path.exists(feature_index_file):
        log.info("Performing feature selection")
        log.info(f"Feature matrix shape before selection: {X.shape}")

        feature_extractor = FeatureExtraction(
            X, y,
            VT_threshold=0.02,
            RFE_features_to_select=RFE_features_to_select
        )

        selected_indices = feature_extractor.feature_extraction(returnIndex=True, index_dtype=int)

        try:
            np.savetxt(feature_index_file, selected_indices, fmt='%d')
            X = X.iloc[:, selected_indices]
            log.info(f"Feature selection complete, selected matrix shape: {X.shape}")
            log.info(f"Feature indices saved to: {feature_index_file}")
        except Exception as e:
            log.error(f"Feature selection failed: {e}")
            if os.path.exists(feature_index_file):
                os.remove(feature_index_file)
            sys.exit()
    else:
        log.info("Loading existing feature indices")
        selected_indices = np.loadtxt(feature_index_file, dtype=int, delimiter=',').tolist()
        X = X.iloc[:, selected_indices]
        log.info(f"Feature matrix shape after selection: {X.shape}")

    # Feature normalization
    log.info("Feature normalization")
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    X = pd.DataFrame(X_scaled, columns=X.columns)

    # Dataset split
    log.info("Dataset split")
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=0.1, random_state=seed, stratify=df_balanced['label_18F']
    )
    log.info(f"Train+Val set: {len(X_train_val)} samples, Test set: {len(X_test)} samples")

    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=0.1, random_state=seed
    )
    log.info(f"Train set: {len(X_train)} samples, Validation set: {len(X_val)} samples")

    # Re-normalize
    scaler_final = MinMaxScaler()
    X_train_val_scaled = scaler_final.fit_transform(X_train_val)
    X_test_scaled = scaler_final.transform(X_test)

    X_train_val = pd.DataFrame(X_train_val_scaled, columns=X.columns)
    X_test = pd.DataFrame(X_test_scaled, columns=X.columns)

    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=0.1, random_state=seed
    )

    # Reset indices
    y_train_val = y_train_val.reset_index(drop=True)
    y_train = y_train.reset_index(drop=True)
    y_val = y_val.reset_index(drop=True)
    y_test = y_test.reset_index(drop=True)

    # =============== Train Random Forest ===============
    log.info("\n" + "=" * 80)
    log.info("Training Random Forest model")
    log.info("=" * 80)

    # Check if best parameters exist in report file
    report_file = f'{output_dir}/rf_xgb_cbrain_18F_combined_report.json'
    
    if os.path.exists(report_file):
        try:
            with open(report_file, 'r', encoding='utf-8') as f:
                existing_report = json.load(f)
            if 'random_forest' in existing_report and 'best_params' in existing_report['random_forest']:
                log.info("Loading existing RF best parameters from report file")
                best_params_rf = existing_report['random_forest']['best_params'].copy()
                # Remove random_state from loaded params and set to current seed
                if 'random_state' in best_params_rf:
                    del best_params_rf['random_state']
                best_params_rf['random_state'] = seed
                log.info(f"Loaded RF parameters from JSON: {best_params_rf}")
            else:
                raise ValueError("RF parameters not found in report file")
        except Exception as e:
            log.warning(f"Failed to load RF parameters from JSON: {e}")
            log.info("Using default RF parameters")
            # Fallback to default parameters
            best_params_rf = {
                'n_estimators': 321,
                'max_depth': 18,
                'min_samples_split': 5,
                'min_samples_leaf': 1,
                'max_features': 'sqrt',
                'bootstrap': False,
                'random_state': seed
            }
    else:
        log.info("Report file not found, using default RF parameters")
        best_params_rf = {
            'n_estimators': 321,
            'max_depth': 18,
            'min_samples_split': 5,
            'min_samples_leaf': 1,
            'max_features': 'sqrt',
            'bootstrap': False,
            'random_state': seed
        }
    
    log.info(f"Using RF parameters: {best_params_rf}")

    # Cross-validation
    log.info(f"Performing {cv_times}-fold cross-validation with best parameters")
    best_model_rf = RandomForestRegressor(**best_params_rf)
    cv = KFold(n_splits=cv_times, random_state=seed, shuffle=True)

    cv_scores_rf = {'rmse': [], 'mse': [], 'mae': [], 'r2': [], 'adj_r2': []}

    for fold, (train_idx, val_idx) in tqdm(enumerate(cv.split(X_train_val, y_train_val)),
                                           desc="RF Cross-validation", total=cv_times):
        X_fold_train = X_train_val.iloc[train_idx]
        X_fold_val = X_train_val.iloc[val_idx]
        y_fold_train = y_train_val.iloc[train_idx]
        y_fold_val = y_train_val.iloc[val_idx]

        best_model_rf.fit(X_fold_train, y_fold_train)
        y_pred_fold = best_model_rf.predict(X_fold_val)

        cv_scores_rf['rmse'].append(np.sqrt(mean_squared_error(y_fold_val, y_pred_fold)))
        cv_scores_rf['mse'].append(mean_squared_error(y_fold_val, y_pred_fold))
        cv_scores_rf['mae'].append(mean_absolute_error(y_fold_val, y_pred_fold))
        cv_scores_rf['r2'].append(r2_score(y_fold_val, y_pred_fold))
        cv_scores_rf['adj_r2'].append(adjusted_r2_score(
            r2_score(y_fold_val, y_pred_fold), len(y_fold_val), X_fold_val.shape[1]
        ))

    log.info("========RF Cross-validation Results========")
    log.info(f"RMSE: {np.mean(cv_scores_rf['rmse']):.3f}+/-{np.std(cv_scores_rf['rmse']):.3f}")
    log.info(f"MSE: {np.mean(cv_scores_rf['mse']):.3f}+/-{np.std(cv_scores_rf['mse']):.3f}")
    log.info(f"MAE: {np.mean(cv_scores_rf['mae']):.3f}+/-{np.std(cv_scores_rf['mae']):.3f}")
    log.info(f"R2: {np.mean(cv_scores_rf['r2']):.3f}+/-{np.std(cv_scores_rf['r2']):.3f}")
    log.info(f"Adjusted R2: {np.mean(cv_scores_rf['adj_r2']):.3f}+/-{np.std(cv_scores_rf['adj_r2']):.3f}")

    # Train final RF model
    log.info("Training final RF model")
    final_model_rf = RandomForestRegressor(**best_params_rf)
    final_model_rf.fit(X_train, y_train)

    y_pred_train_rf = final_model_rf.predict(X_train)
    y_pred_val_rf = final_model_rf.predict(X_val)
    y_pred_test_rf = final_model_rf.predict(X_test)

    train_metrics_rf = calculate_metrics(y_train, y_pred_train_rf, X_train.shape[1])
    val_metrics_rf = calculate_metrics(y_val, y_pred_val_rf, X_val.shape[1])
    test_metrics_rf = calculate_metrics(y_test, y_pred_test_rf, X_test.shape[1])

    log.info("========RF Final Test Set Results========")
    log.info(f"MSE: {test_metrics_rf['mse']:.4f}")
    log.info(f"RMSE: {test_metrics_rf['rmse']:.4f}")
    log.info(f"MAE: {test_metrics_rf['mae']:.4f}")
    log.info(f"R2: {test_metrics_rf['r2']:.4f}")
    log.info(f"Adjusted R2: {test_metrics_rf['adj_r2']:.4f}")
    log.info(f"MAPE: {test_metrics_rf['mape']:.2f}%")

    # Save RF model
    model_path_rf = "./cbrainmodel/rf_cbrain_18F_combined_model.joblib"
    joblib.dump(final_model_rf, model_path_rf)
    log.info(f"RF model saved to: {model_path_rf}")

    # =============== Train XGBoost ===============
    log.info("\n" + "=" * 80)
    log.info("Training XGBoost model")
    log.info("=" * 80)

    # Use best parameters from single XGBoost_Cbrain_18F_Resample.py (Test R2=0.6496)
    best_params_xgb = {
        'n_estimators': 2056,
        'learning_rate': 0.026770813803895732,
        'max_depth': 8,
        'min_child_weight': 0.03816155733788662,
        'subsample': 0.5144527768523542,
        'colsample_bytree': 0.7453780342281893,
        'gamma': 0.0,  # Not in original params, using default
        'reg_alpha': 0.29433953275823077,
        'reg_lambda': 18.460100836131488,
        'random_state': seed
    }
    log.info("Using pre-optimized XGBoost parameters from single model training")
    log.info(f"Best XGBoost parameters: {best_params_xgb}")

    # Cross-validation
    log.info(f"Performing {cv_times}-fold cross-validation with best parameters")
    cv_scores_xgb = {'rmse': [], 'mse': [], 'mae': [], 'r2': [], 'adj_r2': []}

    for fold, (train_idx, val_idx) in tqdm(enumerate(cv.split(X_train_val, y_train_val)),
                                           desc="XGBoost Cross-validation", total=cv_times):
        X_fold_train = X_train_val.iloc[train_idx]
        X_fold_val = X_train_val.iloc[val_idx]
        y_fold_train = y_train_val.iloc[train_idx]
        y_fold_val = y_train_val.iloc[val_idx]

        xgb_model = XGBRegressor(**best_params_xgb)
        xgb_model.fit(X_fold_train, y_fold_train)
        y_pred_fold = xgb_model.predict(X_fold_val)

        cv_scores_xgb['rmse'].append(np.sqrt(mean_squared_error(y_fold_val, y_pred_fold)))
        cv_scores_xgb['mse'].append(mean_squared_error(y_fold_val, y_pred_fold))
        cv_scores_xgb['mae'].append(mean_absolute_error(y_fold_val, y_pred_fold))
        cv_scores_xgb['r2'].append(r2_score(y_fold_val, y_pred_fold))
        cv_scores_xgb['adj_r2'].append(adjusted_r2_score(
            r2_score(y_fold_val, y_pred_fold), len(y_fold_val), X_fold_val.shape[1]
        ))

    log.info("========XGBoost Cross-validation Results========")
    log.info(f"RMSE: {np.mean(cv_scores_xgb['rmse']):.3f}+/-{np.std(cv_scores_xgb['rmse']):.3f}")
    log.info(f"MSE: {np.mean(cv_scores_xgb['mse']):.3f}+/-{np.std(cv_scores_xgb['mse']):.3f}")
    log.info(f"MAE: {np.mean(cv_scores_xgb['mae']):.3f}+/-{np.std(cv_scores_xgb['mae']):.3f}")
    log.info(f"R2: {np.mean(cv_scores_xgb['r2']):.3f}+/-{np.std(cv_scores_xgb['r2']):.3f}")
    log.info(f"Adjusted R2: {np.mean(cv_scores_xgb['adj_r2']):.3f}+/-{np.std(cv_scores_xgb['adj_r2']):.3f}")

    # Train final XGBoost model
    log.info("Training final XGBoost model")
    final_model_xgb = XGBRegressor(**best_params_xgb)
    final_model_xgb.fit(X_train, y_train)

    y_pred_train_xgb = final_model_xgb.predict(X_train)
    y_pred_val_xgb = final_model_xgb.predict(X_val)
    y_pred_test_xgb = final_model_xgb.predict(X_test)

    train_metrics_xgb = calculate_metrics(y_train, y_pred_train_xgb, X_train.shape[1])
    val_metrics_xgb = calculate_metrics(y_val, y_pred_val_xgb, X_val.shape[1])
    test_metrics_xgb = calculate_metrics(y_test, y_pred_test_xgb, X_test.shape[1])

    log.info("========XGBoost Final Test Set Results========")
    log.info(f"MSE: {test_metrics_xgb['mse']:.4f}")
    log.info(f"RMSE: {test_metrics_xgb['rmse']:.4f}")
    log.info(f"MAE: {test_metrics_xgb['mae']:.4f}")
    log.info(f"R2: {test_metrics_xgb['r2']:.4f}")
    log.info(f"Adjusted R2: {test_metrics_xgb['adj_r2']:.4f}")
    log.info(f"MAPE: {test_metrics_xgb['mape']:.2f}%")

    # Save XGBoost model
    model_path_xgb = "./cbrainmodel/xgb_cbrain_18F_combined_model.joblib"
    joblib.dump(final_model_xgb, model_path_xgb)
    log.info(f"XGBoost model saved to: {model_path_xgb}")

    # =============== Save prediction results ===============
    log.info("\n" + "=" * 80)
    log.info("Saving prediction results")
    log.info("=" * 80)

    # Save Random Forest predictions separately (train, val, test)
    rf_train_df = pd.DataFrame({
        'True_Values': y_train.values,
        'Predicted_Values': y_pred_train_rf,
        'Dataset': 'Train'
    })
    rf_val_df = pd.DataFrame({
        'True_Values': y_val.values,
        'Predicted_Values': y_pred_val_rf,
        'Dataset': 'Validation'
    })
    rf_test_df = pd.DataFrame({
        'True_Values': y_test.values,
        'Predicted_Values': y_pred_test_rf,
        'Dataset': 'Test'
    })

    rf_train_df.to_csv(f'{output_dir}/rf_cbrain_18F_train_predictions.csv', index=False)
    rf_val_df.to_csv(f'{output_dir}/rf_cbrain_18F_val_predictions.csv', index=False)
    rf_test_df.to_csv(f'{output_dir}/rf_cbrain_18F_test_predictions.csv', index=False)
    log.info("Random Forest predictions saved:")
    log.info(f"  - {output_dir}/rf_cbrain_18F_train_predictions.csv")
    log.info(f"  - {output_dir}/rf_cbrain_18F_val_predictions.csv")
    log.info(f"  - {output_dir}/rf_cbrain_18F_test_predictions.csv")

    # Save XGBoost predictions separately (train, val, test)
    xgb_train_df = pd.DataFrame({
        'True_Values': y_train.values,
        'Predicted_Values': y_pred_train_xgb,
        'Dataset': 'Train'
    })
    xgb_val_df = pd.DataFrame({
        'True_Values': y_val.values,
        'Predicted_Values': y_pred_val_xgb,
        'Dataset': 'Validation'
    })
    xgb_test_df = pd.DataFrame({
        'True_Values': y_test.values,
        'Predicted_Values': y_pred_test_xgb,
        'Dataset': 'Test'
    })

    xgb_train_df.to_csv(f'{output_dir}/xgb_cbrain_18F_train_predictions.csv', index=False)
    xgb_val_df.to_csv(f'{output_dir}/xgb_cbrain_18F_val_predictions.csv', index=False)
    xgb_test_df.to_csv(f'{output_dir}/xgb_cbrain_18F_test_predictions.csv', index=False)
    log.info("XGBoost predictions saved:")
    log.info(f"  - {output_dir}/xgb_cbrain_18F_train_predictions.csv")
    log.info(f"  - {output_dir}/xgb_cbrain_18F_val_predictions.csv")
    log.info(f"  - {output_dir}/xgb_cbrain_18F_test_predictions.csv")

    # Combined prediction results
    combined_predictions = pd.DataFrame({
        'True_Values': list(y_train.values) + list(y_val.values) + list(y_test.values),
        'RF_Predicted': list(y_pred_train_rf) + list(y_pred_val_rf) + list(y_pred_test_rf),
        'XGB_Predicted': list(y_pred_train_xgb) + list(y_pred_val_xgb) + list(y_pred_test_xgb),
        'Dataset': ['Train'] * len(y_train) + ['Validation'] * len(y_val) + ['Test'] * len(y_test)
    })
    combined_predictions.to_csv(f'{output_dir}/rf_xgb_cbrain_18F_combined_predictions.csv', index=False)
    log.info(f"Combined prediction results saved: {output_dir}/rf_xgb_cbrain_18F_combined_predictions.csv")

    # =============== Generate Scatter Plots ===============
    log.info("Generating scatter plots for RF and XGBoost")

    # Random Forest scatter plot
    fig_rf, axes_rf = plt.subplots(1, 3, figsize=(18, 5))

    # RF Training set
    axes_rf[0].scatter(y_train, y_pred_train_rf, alpha=0.6, edgecolors='k', linewidths=0.5)
    axes_rf[0].plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], 'r--', lw=2)
    axes_rf[0].set_xlabel('True Cbrain', fontsize=12)
    axes_rf[0].set_ylabel('Predicted Cbrain', fontsize=12)
    axes_rf[0].set_title(f'RF Training Set (R?={train_metrics_rf["r2"]:.3f})', fontsize=12)
    axes_rf[0].grid(True, alpha=0.3)

    # RF Validation set
    axes_rf[1].scatter(y_val, y_pred_val_rf, alpha=0.6, edgecolors='k', linewidths=0.5, color='green')
    axes_rf[1].plot([y_val.min(), y_val.max()], [y_val.min(), y_val.max()], 'r--', lw=2)
    axes_rf[1].set_xlabel('True Cbrain', fontsize=12)
    axes_rf[1].set_ylabel('Predicted Cbrain', fontsize=12)
    axes_rf[1].set_title(f'RF Validation Set (R?={val_metrics_rf["r2"]:.3f})', fontsize=12)
    axes_rf[1].grid(True, alpha=0.3)

    # RF Test set
    axes_rf[2].scatter(y_test, y_pred_test_rf, alpha=0.6, edgecolors='k', linewidths=0.5, color='red')
    axes_rf[2].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    axes_rf[2].set_xlabel('True Cbrain', fontsize=12)
    axes_rf[2].set_ylabel('Predicted Cbrain', fontsize=12)
    axes_rf[2].set_title(f'RF Test Set (R?={test_metrics_rf["r2"]:.3f})', fontsize=12)
    axes_rf[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{output_dir}/rf_cbrain_18F_scatter_plots.png', dpi=300, bbox_inches='tight')
    plt.close()
    log.info(f"Random Forest scatter plots saved: {output_dir}/rf_cbrain_18F_scatter_plots.png")

    # XGBoost scatter plot
    fig_xgb, axes_xgb = plt.subplots(1, 3, figsize=(18, 5))

    # XGBoost Training set
    axes_xgb[0].scatter(y_train, y_pred_train_xgb, alpha=0.6, edgecolors='k', linewidths=0.5)
    axes_xgb[0].plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], 'r--', lw=2)
    axes_xgb[0].set_xlabel('True Cbrain', fontsize=12)
    axes_xgb[0].set_ylabel('Predicted Cbrain', fontsize=12)
    axes_xgb[0].set_title(f'XGBoost Training Set (R?={train_metrics_xgb["r2"]:.3f})', fontsize=12)
    axes_xgb[0].grid(True, alpha=0.3)

    # XGBoost Validation set
    axes_xgb[1].scatter(y_val, y_pred_val_xgb, alpha=0.6, edgecolors='k', linewidths=0.5, color='green')
    axes_xgb[1].plot([y_val.min(), y_val.max()], [y_val.min(), y_val.max()], 'r--', lw=2)
    axes_xgb[1].set_xlabel('True Cbrain', fontsize=12)
    axes_xgb[1].set_ylabel('Predicted Cbrain', fontsize=12)
    axes_xgb[1].set_title(f'XGBoost Validation Set (R?={val_metrics_xgb["r2"]:.3f})', fontsize=12)
    axes_xgb[1].grid(True, alpha=0.3)

    # XGBoost Test set
    axes_xgb[2].scatter(y_test, y_pred_test_xgb, alpha=0.6, edgecolors='k', linewidths=0.5, color='red')
    axes_xgb[2].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    axes_xgb[2].set_xlabel('True Cbrain', fontsize=12)
    axes_xgb[2].set_ylabel('Predicted Cbrain', fontsize=12)
    axes_xgb[2].set_title(f'XGBoost Test Set (R?={test_metrics_xgb["r2"]:.3f})', fontsize=12)
    axes_xgb[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{output_dir}/xgb_cbrain_18F_scatter_plots.png', dpi=300, bbox_inches='tight')
    plt.close()
    log.info(f"XGBoost scatter plots saved: {output_dir}/xgb_cbrain_18F_scatter_plots.png")

    # =============== Generate comparison report ===============
    comparison_report = {
        'experiment_info': {
            'dataset': 'PTBD_v20240912.csv',
            'balancing_method': 'oversample_18F',
            'features': 'Mordred_descriptors',
            'target': 'brain at60min',
            'n_samples_original': len(df),
            'n_samples_balanced': len(df_balanced),
            'n_features': X.shape[1],
            'seed': seed
        },
        'random_forest': {
            'best_params': best_params_rf,
            'cross_validation': {
                'cv_folds': cv_times,
                'rmse': f"{np.mean(cv_scores_rf['rmse']):.3f}+/-{np.std(cv_scores_rf['rmse']):.3f}",
                'r2': f"{np.mean(cv_scores_rf['r2']):.3f}+/-{np.std(cv_scores_rf['r2']):.3f}",
                'adj_r2': f"{np.mean(cv_scores_rf['adj_r2']):.3f}+/-{np.std(cv_scores_rf['adj_r2']):.3f}"
            },
            'final_results': {
                'train': train_metrics_rf,
                'validation': val_metrics_rf,
                'test': test_metrics_rf
            }
        },
        'xgboost': {
            'best_params': best_params_xgb,
            'cross_validation': {
                'cv_folds': cv_times,
                'rmse': f"{np.mean(cv_scores_xgb['rmse']):.3f}+/-{np.std(cv_scores_xgb['rmse']):.3f}",
                'r2': f"{np.mean(cv_scores_xgb['r2']):.3f}+/-{np.std(cv_scores_xgb['r2']):.3f}",
                'adj_r2': f"{np.mean(cv_scores_xgb['adj_r2']):.3f}+/-{np.std(cv_scores_xgb['adj_r2']):.3f}"
            },
            'final_results': {
                'train': train_metrics_xgb,
                'validation': val_metrics_xgb,
                'test': test_metrics_xgb
            }
        }
    }

    with open(f'{output_dir}/rf_xgb_cbrain_18F_combined_report.json', 'w', encoding='utf-8') as f:
        json.dump(comparison_report, f, ensure_ascii=False, indent=2)

    log.info(f"Comparison report saved to: {output_dir}/rf_xgb_cbrain_18F_combined_report.json")

    # =============== Print comparison summary ===============
    print("\n" + "=" * 80)
    print("Model Comparison Summary (Test Set)")
    print("=" * 80)
    print(f"{'Metric':<20} {'Random Forest':>25} {'XGBoost':>25}")
    print("-" * 80)
    print(f"{'R2':<20} {test_metrics_rf['r2']:>25.4f} {test_metrics_xgb['r2']:>25.4f}")
    print(f"{'Adjusted R2':<20} {test_metrics_rf['adj_r2']:>25.4f} {test_metrics_xgb['adj_r2']:>25.4f}")
    print(f"{'RMSE':<20} {test_metrics_rf['rmse']:>25.4f} {test_metrics_xgb['rmse']:>25.4f}")
    print(f"{'MAE':<20} {test_metrics_rf['mae']:>25.4f} {test_metrics_xgb['mae']:>25.4f}")
    print(f"{'MAPE (%)':<20} {test_metrics_rf['mape']:>25.2f} {test_metrics_xgb['mape']:>25.2f}")
    print("=" * 80)

    log.info("\n" + "=" * 80)
    log.info("Experiment completed!")
    log.info(f"All results saved to {output_dir}/:")
    log.info("  Random Forest:")
    log.info("    - rf_cbrain_18F_train_predictions.csv")
    log.info("    - rf_cbrain_18F_val_predictions.csv")
    log.info("    - rf_cbrain_18F_test_predictions.csv")
    log.info("    - rf_cbrain_18F_scatter_plots.png")
    log.info("  XGBoost:")
    log.info("    - xgb_cbrain_18F_train_predictions.csv")
    log.info("    - xgb_cbrain_18F_val_predictions.csv")
    log.info("    - xgb_cbrain_18F_test_predictions.csv")
    log.info("    - xgb_cbrain_18F_scatter_plots.png")
    log.info("  Combined:")
    log.info("    - rf_xgb_cbrain_18F_combined_predictions.csv")
    log.info("    - rf_xgb_cbrain_18F_combined_report.json")
    log.info("=" * 80)

    print(f"\nExperiment completed!")
    print(f"RF  - Test set R2 = {test_metrics_rf['r2']:.4f}, RMSE = {test_metrics_rf['rmse']:.4f}")
    print(f"XGB - Test set R2 = {test_metrics_xgb['r2']:.4f}, RMSE = {test_metrics_xgb['rmse']:.4f}")


if __name__ == '__main__':
    main()
