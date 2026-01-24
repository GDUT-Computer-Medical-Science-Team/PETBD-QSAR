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
from xgboost.sklearn import XGBRegressor
import optuna
import tempfile
from padelpy import padeldescriptor
import joblib
import json
import re
from typing import Optional
from tqdm import tqdm

sys.path.append('../../')
from utils.DataLogger import DataLogger

log = DataLogger(log_file=f"../../data/logBB_data/log/{datetime.now().strftime('%Y%m%d')}.log").getlog("xgb_fp_18F_resample")

ALLOWED_ISOTOPES = {"18F", "11C", "125I", "131I", "123I", "77Br", "76Br"}

def contains_18F(compound_index: str) -> bool:
    if not isinstance(compound_index, str):
        return False

    text = compound_index.strip()

    return '18F' in text

def balance_18F_dataset(df: pd.DataFrame, method: str = "combined", seed: int = 42):
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

def calculate_morgan_fingerprints(smiles_list, radius=2, n_bits=1024):
    fingerprints = []

    for smiles in smiles_list:
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
            arr = np.zeros((1,))
            DataStructs.ConvertToNumpyArray(fp, arr)
            fingerprints.append(arr)
        else:

            fingerprints.append(np.zeros(n_bits))

    return np.array(fingerprints)

def create_objective_xgb(X_train_scaled, y_train, X_val_scaled, y_val):
    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 2000),
            'max_depth': trial.suggest_int('max_depth', 3, 12),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 0, 50),
            'reg_lambda': trial.suggest_float('reg_lambda', 0, 50),
            'random_state': 42
        }

        model = XGBRegressor(**params)
        model.fit(X_train_scaled, y_train)

        y_pred = model.predict(X_val_scaled)
        mse = mean_squared_error(y_val, y_pred)

        return mse
    return objective

def main():
    log.info("=============== Starting XGBoost FP 18F Resample Training ===============")

    np.random.seed(42)

    log.info("Loading PETBD dataset")
    df = pd.read_csv("../../dataset_PETBD/PTBD_v20240912.csv", encoding='utf-8')
    log.info(f"Original data shape: {df.shape}")

    df = df.dropna(subset=['logBB'])
    log.info(f"Data shape after removing logBB missing values: {df.shape}")

    df_original = df.copy()
    log.info(f"Saved original dataset for final prediction: {df_original.shape}")

    log.info("Starting 18F combined balancing: original data + 18F resampling")
    df_balanced = balance_18F_dataset(df, method="combined")
    log.info(f"Training data shape after combined strategy: {df_balanced.shape}")

    log.info("Calculating molecular fingerprints for original dataset")

    original_feature_file = f"../../data/logBB_data/xgb_fp_18F_original_features.csv"

    if os.path.exists(original_feature_file):
        log.info("Loading existing original dataset features")
        original_features_df = pd.read_csv(original_feature_file)

        if 'logBB' in original_features_df.columns:
            X_original = original_features_df.drop(['logBB'], axis=1, errors='ignore').values
            y_original = original_features_df['logBB'].values
        else:

            log.info("logBB column missing from saved features, recalculating...")
            X_morgan_original = calculate_morgan_fingerprints(df_original['SMILES'].tolist())
            X_original = X_morgan_original
            y_original = df_original['logBB'].values
    else:
        log.info("Calculating features for original dataset")

        X_morgan_original = calculate_morgan_fingerprints(df_original['SMILES'].tolist())

        X_original = X_morgan_original
        y_original = df_original['logBB'].values

        original_features_df = pd.DataFrame(X_original)
        original_features_df['logBB'] = y_original
        original_features_df.to_csv(original_feature_file, index=False)
        log.info(f"Original dataset features saved: {original_feature_file}")

    log.info("Calculating molecular fingerprints for training dataset")

    train_feature_file = f"../../data/logBB_data/xgb_fp_18F_train_features.csv"

    if os.path.exists(train_feature_file):
        log.info("Loading existing training dataset features")
        train_features_df = pd.read_csv(train_feature_file)

        if 'logBB' in train_features_df.columns:
            X_train_full = train_features_df.drop(['logBB', 'has_18F', 'label_18F'], axis=1, errors='ignore').values
            y_train_full = train_features_df['logBB'].values
        else:

            log.info("logBB column missing from saved feature file, recalculating...")
            X_morgan_train = calculate_morgan_fingerprints(df_balanced['SMILES'].tolist())
            X_train_full = X_morgan_train
            y_train_full = df_balanced['logBB'].values
    else:
        log.info("Calculating features for training dataset")

        X_morgan_train = calculate_morgan_fingerprints(df_balanced['SMILES'].tolist())

        X_train_full = X_morgan_train
        y_train_full = df_balanced['logBB'].values

        train_features_df = pd.DataFrame(X_train_full)
        train_features_df['logBB'] = y_train_full
        if 'has_18F' in df_balanced.columns:
            train_features_df['has_18F'] = df_balanced['has_18F'].values
        if 'label_18F' in df_balanced.columns:
            train_features_df['label_18F'] = df_balanced['label_18F'].values
        train_features_df.to_csv(train_feature_file, index=False)
        log.info(f"Training dataset features saved: {train_feature_file}")

    log.info("Checking data consistency[Chinese text removed]")

    log.info(f"Training data dimension check: X_train_full {X_train_full.shape}, y_train_full {len(y_train_full)}")
    log.info(f"Original data dimension check: X_original {X_original.shape}, y_original {len(y_original)}")

    X_train_full = np.where(np.isinf(X_train_full), np.nan, X_train_full)
    X_original = np.where(np.isinf(X_original), np.nan, X_original)

    from sklearn.impute import SimpleImputer
    imputer = SimpleImputer(strategy='mean')
    X_train_full = imputer.fit_transform(X_train_full)
    X_original = imputer.transform(X_original)

    if len(y_train_full) != X_train_full.shape[0]:
        log.warning(f"[Chinese text removed]Dimension mismatch: X_train_full {X_train_full.shape[0]} vs y_train_full {len(y_train_full)}")

        if os.path.exists(train_feature_file):
            log.info("Deleting inconsistent feature file, recalculating...")
            os.remove(train_feature_file)
            X_morgan_train = calculate_morgan_fingerprints(df_balanced['SMILES'].tolist())
            X_train_full = X_morgan_train
            y_train_full = df_balanced['logBB'].values
            log.info(f"Dimension after recalculation: X_train_full {X_train_full.shape}, y_train_full {len(y_train_full)}")
        else:

            min_len = min(len(y_train_full), X_train_full.shape[0])
            X_train_full = X_train_full[:min_len]
            y_train_full = y_train_full[:min_len]
            log.info(f"Adjusted to consistent dimension: {min_len}")

    if len(y_original) != X_original.shape[0]:
        log.warning(f"[Chinese text removed]Dimension mismatch: X_original {X_original.shape[0]} vs y_original {len(y_original)}")

        if os.path.exists(original_feature_file):
            log.info("Deleting inconsistent original feature file, recalculating...")
            os.remove(original_feature_file)
            X_morgan_original = calculate_morgan_fingerprints(df_original['SMILES'].tolist())
            X_original = X_morgan_original
            y_original = df_original['logBB'].values
            log.info(f"Dimension after recalculation: X_original {X_original.shape}, y_original {len(y_original)}")
        else:

            min_len = min(len(y_original), X_original.shape[0])
            X_original = X_original[:min_len]
            y_original = y_original[:min_len]
            log.info(f"Adjusted original data to consistent dimension: {min_len}")

    log.info(f"Training data shape after feature calculation: {X_train_full.shape}")
    log.info(f"Original data shape after feature calculation: {X_original.shape}")

    log.info("Loading existing selected features")
    feature_file = f"../../data/logBB_data/xgb_fp_18F_feature_index.txt"

    if os.path.exists(feature_file):
        log.info("Loading existing feature indices")
        with open(feature_file, 'r') as f:
            selected_indices = [int(line.strip()) for line in f.readlines()]
    else:
        log.info("Selecting features using ExtraTreesRegressor")
        from sklearn.ensemble import ExtraTreesRegressor
        from sklearn.feature_selection import SelectFromModel

        selector = ExtraTreesRegressor(n_estimators=100, random_state=42)
        selector.fit(X_train_full, y_train_full)

        sel = SelectFromModel(selector, prefit=True, max_features=50)
        X_train_full = sel.transform(X_train_full)
        X_original = sel.transform(X_original)

        selected_indices = sel.get_support(indices=True)
        with open(feature_file, 'w') as f:
            for idx in selected_indices:
                f.write(f"{idx}\n")
        log.info(f"Feature indices saved: {feature_file}")

    max_idx = max(selected_indices) if selected_indices else 0
    if max_idx < X_train_full.shape[1]:
        X_train_full = X_train_full[:, selected_indices]
        X_original = X_original[:, selected_indices]
    else:

        log.info("Selected indices out of bounds, regenerating feature selection")
        from sklearn.ensemble import ExtraTreesRegressor
        from sklearn.feature_selection import SelectFromModel

        selector = ExtraTreesRegressor(n_estimators=100, random_state=42)
        selector.fit(X_train_full, y_train_full)

        sel = SelectFromModel(selector, prefit=True, max_features=50)
        X_train_full = sel.transform(X_train_full)
        X_original = sel.transform(X_original)

        selected_indices = sel.get_support(indices=True)
        with open(feature_file, 'w') as f:
            for idx in selected_indices:
                f.write(f"{idx}\n")

    log.info(f"Data shape after feature selection training: {X_train_full.shape}")
    log.info(f"Data shape after feature selection original: {X_original.shape}")

    log.info("Dataset split (balanced training data)")
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X_train_full, y_train_full, test_size=0.1, random_state=42
    )

    log.info(f"Train+Val set: {X_train_val.shape[0]} samples, Test set: {X_test.shape[0]} samples")

    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=0.1, random_state=42
    )

    log.info(f"Training: {X_train.shape[0]} samples, Validation: {X_val.shape[0]} samples")

    log.info(f"Original full dataset for final evaluation: {X_original.shape[0]} samples")

    log.info("Data standardization")
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    X_original_scaled = scaler.transform(X_original)

    log.info("Starting hyperparameter optimization (100 trials)...")

    objective_xgb = create_objective_xgb(X_train_scaled, y_train, X_val_scaled, y_val)
    study = optuna.create_study(direction='minimize')
    study.optimize(objective_xgb, n_trials=100, show_progress_bar=True)

    best_params = study.best_params
    log.info(f"Best hyperparameters: {best_params}")

    log.info("Performing 10-fold cross-validation with best parameters")

    cv = KFold(n_splits=10, shuffle=True, random_state=42)
    cv_scores_r2 = []
    cv_scores_rmse = []
    cv_scores_mae = []
    cv_scores_adj_r2 = []

    for fold, (train_idx, val_idx) in enumerate(tqdm(cv.split(X_train_val), desc="Cross-validation")):
        X_cv_train, X_cv_val = X_train_val[train_idx], X_train_val[val_idx]
        y_cv_train, y_cv_val = y_train_val[train_idx], y_train_val[val_idx]

        cv_scaler = MinMaxScaler()
        X_cv_train_scaled = cv_scaler.fit_transform(X_cv_train)
        X_cv_val_scaled = cv_scaler.transform(X_cv_val)

        model = XGBRegressor(**best_params)
        model.fit(X_cv_train_scaled, y_cv_train)

        y_cv_pred = model.predict(X_cv_val_scaled)

        r2 = r2_score(y_cv_val, y_cv_pred)
        rmse = np.sqrt(mean_squared_error(y_cv_val, y_cv_pred))
        mae = mean_absolute_error(y_cv_val, y_cv_pred)

        n = len(y_cv_val)
        p = X_cv_val.shape[1]
        adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)

        cv_scores_r2.append(r2)
        cv_scores_rmse.append(rmse)
        cv_scores_mae.append(mae)
        cv_scores_adj_r2.append(adj_r2)

        log.info(f"Fold {fold+1}: R²={r2:.4f}, RMSE={rmse:.4f}, MAE={mae:.4f}")

    log.info("======== Cross-validation results ========")
    log.info(f"RMSE: {np.mean(cv_scores_rmse):.3f}±{np.std(cv_scores_rmse):.3f}")
    log.info(f"MSE: {np.mean(np.array(cv_scores_rmse)**2):.3f}±{np.std(np.array(cv_scores_rmse)**2):.3f}")
    log.info(f"MAE: {np.mean(cv_scores_mae):.3f}±{np.std(cv_scores_mae):.3f}")
    log.info(f"R²: {np.mean(cv_scores_r2):.3f}±{np.std(cv_scores_r2):.3f}")
    log.info(f"Adjusted R²: {np.mean(cv_scores_adj_r2):.3f}±{np.std(cv_scores_adj_r2):.3f}")

    log.info("Training final model")
    final_model = XGBRegressor(**best_params)
    final_model.fit(X_train_scaled, y_train)

    log.info("Evaluating on test set from balanced training data")
    y_test_pred = final_model.predict(X_test_scaled)

    test_balanced_metrics = {
        'mse': mean_squared_error(y_test, y_test_pred),
        'rmse': np.sqrt(mean_squared_error(y_test, y_test_pred)),
        'mae': mean_absolute_error(y_test, y_test_pred),
        'r2': r2_score(y_test, y_test_pred),
        'mape': mean_absolute_percentage_error(y_test, y_test_pred) * 100,
        'adj_r2': 1 - (1 - r2_score(y_test, y_test_pred)) * (len(y_test) - 1) / (len(y_test) - X_test.shape[1] - 1)
    }

    log.info(f"Test R²: {test_balanced_metrics['r2']:.4f}")
    log.info(f"Test RMSE: {test_balanced_metrics['rmse']:.4f}")
    log.info(f"Test MAE: {test_balanced_metrics['mae']:.4f}")

    log.info("Predicting on balanced full dataset")
    log.info(f"X_train_full shape: {X_train_full.shape}")
    log.info(f"y_train_full length: {len(y_train_full)}")

    if len(y_train_full) != X_train_full.shape[0]:
        log.warning(f"Dimension mismatch: X_train_full {X_train_full.shape[0]} vs y_train_full {len(y_train_full)}")
        min_len = min(len(y_train_full), X_train_full.shape[0])
        X_train_full = X_train_full[:min_len]
        y_train_full = y_train_full[:min_len]
        log.info(f"Adjusted dimension: X_train_full {X_train_full.shape}, y_train_full {len(y_train_full)}")

    X_train_full_scaled = scaler.transform(X_train_full)
    y_balanced_full_pred = final_model.predict(X_train_full_scaled)

    balanced_full_dataset_metrics = {
        'mse': mean_squared_error(y_train_full, y_balanced_full_pred),
        'rmse': np.sqrt(mean_squared_error(y_train_full, y_balanced_full_pred)),
        'mae': mean_absolute_error(y_train_full, y_balanced_full_pred),
        'r2': r2_score(y_train_full, y_balanced_full_pred),
        'mape': mean_absolute_percentage_error(y_train_full, y_balanced_full_pred) * 100,
        'adj_r2': 1 - (1 - r2_score(y_train_full, y_balanced_full_pred)) * (len(y_train_full) - 1) / (len(y_train_full) - X_train_full.shape[1] - 1)
    }

    log.info("======== Balanced test set results ========")
    log.info(f"MSE: {test_balanced_metrics['mse']:.4f}")
    log.info(f"RMSE: {test_balanced_metrics['rmse']:.4f}")
    log.info(f"MAE: {test_balanced_metrics['mae']:.4f}")
    log.info(f"R²: {test_balanced_metrics['r2']:.4f}")
    log.info(f"Adjusted R²: {test_balanced_metrics['adj_r2']:.4f}")
    log.info(f"MAPE: {test_balanced_metrics['mape']:.2f}%")

    log.info(f"======== Balanced full dataset results ========")
    log.info(f"MSE: {balanced_full_dataset_metrics['mse']:.4f}")
    log.info(f"RMSE: {balanced_full_dataset_metrics['rmse']:.4f}")
    log.info(f"MAE: {balanced_full_dataset_metrics['mae']:.4f}")
    log.info(f"R²: {balanced_full_dataset_metrics['r2']:.4f}")
    log.info(f"Adjusted R²: {balanced_full_dataset_metrics['adj_r2']:.4f}")
    log.info(f"MAPE: {balanced_full_dataset_metrics['mape']:.2f}%")

    model_file = f"./logbbModel/xgb_fp_18F_model.joblib"
    joblib.dump(final_model, model_file)
    log.info(f"Model saved: {model_file}")

    scaler_file = f"./logbbModel/xgb_fp_18F_scaler.joblib"
    joblib.dump(scaler, scaler_file)

    feature_array_file = f"./logbbModel/xgb_fp_18F_features.npy"
    np.save(feature_array_file, selected_indices)

    y_train_pred = final_model.predict(X_train_scaled)
    train_df = pd.DataFrame({
        'True_Values': y_train,
        'Predicted_Values': y_train_pred,
        'Dataset': 'Train'
    })
    train_file = f"./result/xgb_18F_train_predictions.csv"
    train_df.to_csv(train_file, index=False)
    log.info(f"Training set predictions saved: {train_file}")

    y_val_pred = final_model.predict(X_val_scaled)
    val_df = pd.DataFrame({
        'True_Values': y_val,
        'Predicted_Values': y_val_pred,
        'Dataset': 'Validation'
    })
    val_file = f"./result/xgb_18F_validation_predictions.csv"
    val_df.to_csv(val_file, index=False)
    log.info(f"Validation set predictions saved: {val_file}")

    test_df = pd.DataFrame({
        'True_Values': y_test,
        'Predicted_Values': y_test_pred,
        'Dataset': 'Test'
    })
    test_file = f"./result/xgb_18F_test_predictions.csv"
    test_df.to_csv(test_file, index=False)
    log.info(f"Test set predictions saved: {test_file}")

    all_results_df = pd.concat([train_df, val_df, test_df], ignore_index=True)

    results_file = f"./result/xgb_18F_all_predictions.csv"
    all_results_df.to_csv(results_file, index=False)
    log.info(f"Combined predictions saved: {results_file}")

    plt.figure(figsize=(10, 8))

    train_data = all_results_df[all_results_df['Dataset'] == 'Train']
    val_data = all_results_df[all_results_df['Dataset'] == 'Validation']
    test_data = all_results_df[all_results_df['Dataset'] == 'Test']

    plt.scatter(train_data['True_Values'], train_data['Predicted_Values'], 
               alpha=0.5, label='Training Set', color='blue', s=20)
    plt.scatter(val_data['True_Values'], val_data['Predicted_Values'], 
               alpha=0.7, label='Validation Set', color='green', s=20)
    plt.scatter(test_data['True_Values'], test_data['Predicted_Values'], 
               alpha=0.8, label='Test Set', color='red', s=30)

    min_val = min(all_results_df['True_Values'].min(), all_results_df['Predicted_Values'].min())
    max_val = max(all_results_df['True_Values'].max(), all_results_df['Predicted_Values'].max())
    plt.plot([min_val, max_val], [min_val, max_val], 'k--', lw=2, alpha=0.8)

    plt.xlabel('Experimental logBB')
    plt.ylabel('Predicted logBB')
    plt.title('XGBoost FP 18F-Resampled Model: Predicted vs Experimental logBB')
    plt.legend()

    train_r2 = r2_score(train_data['True_Values'], train_data['Predicted_Values'])
    val_r2 = r2_score(val_data['True_Values'], val_data['Predicted_Values'])
    test_r2 = r2_score(test_data['True_Values'], test_data['Predicted_Values'])

    plt.text(0.05, 0.95, f'Train R² = {train_r2:.3f}', 
             transform=plt.gca().transAxes, fontsize=10)
    plt.text(0.05, 0.90, f'Val R² = {val_r2:.3f}', 
             transform=plt.gca().transAxes, fontsize=10)
    plt.text(0.05, 0.85, f'Test R² = {test_r2:.3f}', 
             transform=plt.gca().transAxes, fontsize=10)

    plt.tight_layout()

    plot_file = f"./result/xgb_18F_scatter.png"
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    plt.close()

    train_metrics = {
        'mse': mean_squared_error(y_train, y_train_pred),
        'rmse': np.sqrt(mean_squared_error(y_train, y_train_pred)),
        'mae': mean_absolute_error(y_train, y_train_pred),
        'r2': r2_score(y_train, y_train_pred),
        'mape': mean_absolute_percentage_error(y_train, y_train_pred) * 100,
        'adj_r2': 1 - (1 - r2_score(y_train, y_train_pred)) * (len(y_train) - 1) / (len(y_train) - X_train.shape[1] - 1)
    }

    val_metrics = {
        'mse': mean_squared_error(y_val, y_val_pred),
        'rmse': np.sqrt(mean_squared_error(y_val, y_val_pred)),
        'mae': mean_absolute_error(y_val, y_val_pred),
        'r2': r2_score(y_val, y_val_pred),
        'mape': mean_absolute_percentage_error(y_val, y_val_pred) * 100,
        'adj_r2': 1 - (1 - r2_score(y_val, y_val_pred)) * (len(y_val) - 1) / (len(y_val) - X_val.shape[1] - 1)
    }

    report = {
        "experiment_info": {
            "dataset": "PTBD_v20240912.csv",
            "training_method": "18F balanced resampling",
            "evaluation_method": f"Full original dataset ({len(y_original)} samples)",
            "features": "Molecular_fingerprints (Morgan)",
            "model": "XGBoost",
            "n_samples_original": len(df_original),
            "n_samples_balanced_training": len(df_balanced),
            "n_features": len(selected_indices),
            "seed": 42
        },
        "best_params": best_params,
        "cross_validation": {
            "cv_folds": 10,
            "rmse": f"{np.mean(cv_scores_rmse):.3f}±{np.std(cv_scores_rmse):.3f}",
            "r2": f"{np.mean(cv_scores_r2):.3f}±{np.std(cv_scores_r2):.3f}",
            "adj_r2": f"{np.mean(cv_scores_adj_r2):.3f}±{np.std(cv_scores_adj_r2):.3f}"
        },
        "final_results": {
            "train": train_metrics,
            "validation": val_metrics,
            "test_balanced": test_balanced_metrics,
            "balanced_full_dataset": balanced_full_dataset_metrics
        }
    }

    report_file = f"./result/xgb_fp_18F_report.json"
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2)

    log.info("======== Experiment completed ========")
    log.info(f"Predictions saved: {results_file}")
    log.info(f"Scatter plot saved: {plot_file}")
    log.info(f"Experiment report saved: {report_file}")

    print(f"Balanced test R² = {test_balanced_metrics['r2']:.4f}, RMSE = {test_balanced_metrics['rmse']:.4f}")
    print(f"Balanced full dataset R² = {balanced_full_dataset_metrics['r2']:.4f}, RMSE = {balanced_full_dataset_metrics['rmse']:.4f}")

if __name__ == "__main__":
    main()