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

log = DataLogger(log_file=f"../../data/logBB_data/log/{datetime.now().strftime('%Y%m%d')}.log").getlog("rf_fp_18F_resample")

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
        log.info(f"  [Chinese text removed]18F samples: {n_non_18f}")

        if n_18f < n_non_18f:

            extra_needed = n_non_18f - n_18f
            df_18f_extra = df_18f.sample(n=extra_needed, replace=True, random_state=seed)
            log.info(f"[Chinese text removed]18F samples: {extra_needed}")
            combined = pd.concat([df_copy, df_18f_extra], ignore_index=True)
        elif n_non_18f < n_18f:

            extra_needed = n_18f - n_non_18f
            df_non_18f_extra = df_non_18f.sample(n=extra_needed, replace=True, random_state=seed)
            log.info(f"[Chinese text removed]18F samples: {extra_needed}")
            combined = pd.concat([df_copy, df_non_18f_extra], ignore_index=True)
        else:

            combined = df_copy
            log.info("Data already balanced, no additional sampling needed")

        combined = combined.sample(frac=1.0, random_state=seed).reset_index(drop=True)

        log.info(f"[Chinese text removed]Total samples[Chinese text removed]: {len(combined)} (original: {len(df)}, added: {len(combined) - len(df)})")

        return combined

    else:

        df = df.copy()
        df["has_18F"] = df["compound index"].apply(contains_18F)

        df["label_18F"] = df["has_18F"].astype(int)

        pos = df[df["label_18F"] == 1]
        neg = df[df["label_18F"] == 0]

        n_pos, n_neg = len(pos), len(neg)
        log.info(f"Original data distribution - 18F samples: {n_pos}, [Chinese text removed]18F samples: {n_neg}")

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

        log.info(f"Balanced data distribution - 18F samples: {len(pos_bal)}, [Chinese text removed]18F samples: {len(neg_bal)}")

        return balanced

def calculate_padel_fingerprints(smiles_list):
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
        log.error(f"[Chinese text removed]PaDEL[Chinese text removed]Error: {str(e)}")

        return pd.DataFrame()
    finally:
        if os.path.exists(temp_smi_file):
            os.remove(temp_smi_file)

def calculate_morgan_fingerprints(smiles_list, radius=2, n_bits=1024):
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
    n = len(y_true)
    r2 = r2_score(y_true, y_pred)

    if n - num_features - 1 == 0:
        return r2
    return 1 - (1 - r2) * (n - 1) / (n - num_features - 1)

def plot_scatter(y_train, y_pred_train, y_val, y_pred_val, y_test, y_pred_test, save_path):
    plt.figure(figsize=(10, 8))

    plt.scatter(y_train, y_pred_train, alpha=0.5, label='Training Set', color='blue', s=20)
    plt.scatter(y_val, y_pred_val, alpha=0.7, label='Validation Set', color='green', s=20)
    plt.scatter(y_test, y_pred_test, alpha=0.8, label='Test Set', color='red', s=30)

    all_min = min(min(y_train), min(y_val), min(y_test))
    all_max = max(max(y_train), max(y_val), max(y_test))
    plt.plot([all_min, all_max], [all_min, all_max], 'k--', lw=2, alpha=0.8)

    plt.xlabel('Experimental logBB')
    plt.ylabel('Predicted logBB')
    plt.title('Random Forest FP 18F-Resampled Model: Predicted vs Experimental logBB')
    plt.legend()

    train_r2 = r2_score(y_train, y_pred_train)
    val_r2 = r2_score(y_val, y_pred_val)
    test_r2 = r2_score(y_test, y_pred_test)

    plt.text(0.05, 0.95, f'Train R² = {train_r2:.3f}', 
             transform=plt.gca().transAxes, fontsize=10)
    plt.text(0.05, 0.90, f'Val R² = {val_r2:.3f}', 
             transform=plt.gca().transAxes, fontsize=10)
    plt.text(0.05, 0.85, f'Test R² = {test_r2:.3f}', 
             transform=plt.gca().transAxes, fontsize=10)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def objective(trial, X, y):
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

def main():
    log.info("===============Starting RandomForest FP 18Fresampling training[Chinese text removed]===============")

    petbd_data_file = "../../data/PTBD_v20240912.csv"
    petbd_features_file = "../../data/logBB_data/RF_FP_18F_balanced_features.csv"
    feature_index_file = "../../data/logBB_data/RF_FP_18F_feature_index.txt"

    os.makedirs("../../data/logBB_data/log", exist_ok=True)
    os.makedirs("./logbbModel", exist_ok=True)
    os.makedirs("./result", exist_ok=True)

    if not os.path.exists(petbd_data_file):
        log.error("Missing PETBD dataset file")
        raise FileNotFoundError("Missing PETBD dataset")

    log.info("Reading PETBD dataset")
    df = pd.read_csv(petbd_data_file, encoding='utf-8')
    log.info(f"original[Chinese text removed]: {df.shape}")

    df = df.dropna(subset=['logBB at60min']).reset_index(drop=True)
    log.info(f"Data shape after removing logBB missing values: {df.shape}")

    df_original = df.copy()
    log.info(f"[Chinese text removed]original[Chinese text removed]: {df_original.shape}")

    log.info("Starting 18F combined strategy processing（original[Chinese text removed] + 18F[Chinese text removed]）")
    df_balanced = balance_18F_dataset(df, method="combined", seed=42)
    log.info(f"Training data shape after combined strategy: {df_balanced.shape}")

    df_train = df_balanced

    log.info("[Chinese text removed]original[Chinese text removed]features")
    original_features_file = "../../data/logBB_data/RF_FP_original_features.csv"

    if os.path.exists(original_features_file):
        log.info("[Chinese text removed]original[Chinese text removed]features[Chinese text removed]")
        df_original_features = pd.read_csv(original_features_file, encoding='utf-8')

        required_columns = ['logBB at60min']
        missing_columns = [col for col in required_columns if col not in df_original_features.columns]

        if missing_columns:
            log.warning(f"features[Chinese text removed]in[Chinese text removed]: {missing_columns}，regenerating features")
            need_regenerate_original = True
        else:
            X_original = df_original_features.drop(['SMILES', 'logBB at60min', 'compound index', 'has_18F', 'label_18F'], 
                                                   axis=1, errors='ignore')
            y_original = df_original_features['logBB at60min']
            need_regenerate_original = False
    else:
        need_regenerate_original = True

    if need_regenerate_original:
        log.info("[Chinese text removed]original[Chinese text removed]features")
        SMILES_original = df_original['SMILES']
        y_original = df_original['logBB at60min']

        log.info("[Chinese text removed]PaDEL[Chinese text removed]（original[Chinese text removed]）")
        X_original = calculate_padel_fingerprints(SMILES_original)
        if X_original.empty:
            log.warning("PaDELFailed，[Chinese text removed]Morgan fingerprints")
            X_original = calculate_morgan_fingerprints(SMILES_original)

        df_original['has_18F'] = df_original["compound index"].apply(contains_18F)
        df_original['label_18F'] = df_original['has_18F'].astype(int)

        original_feature_data = pd.concat([
            df_original[['compound index', 'SMILES', 'logBB at60min', 'has_18F', 'label_18F']],
            X_original
        ], axis=1)
        original_feature_data.to_csv(original_features_file, encoding='utf-8', index=False)
        log.info(f"original[Chinese text removed]Features saved: {original_features_file}")

    log.info("Generating features for training dataset (balanced)")

    if os.path.exists(petbd_features_file):
        log.info("Reading training dataset feature file")
        df_train_features = pd.read_csv(petbd_features_file, encoding='utf-8')

        required_columns = ['logBB at60min']
        missing_columns = [col for col in required_columns if col not in df_train_features.columns]

        if missing_columns:
            log.warning(f"[Chinese text removed]features[Chinese text removed]in[Chinese text removed]: {missing_columns}，regenerating features")
            need_regenerate = True

        elif len(df_train_features) != len(df_train):
            log.warning(f"Feature file sample count mismatch, regenerating")
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

        log.info("[Chinese text removed]PaDEL[Chinese text removed]（[Chinese text removed]）")
        X_train_full = calculate_padel_fingerprints(SMILES_train)

        if X_train_full.empty:
            log.warning("PaDELFailed，[Chinese text removed]Morgan fingerprints")
            X_train_full = calculate_morgan_fingerprints(SMILES_train)

        log.info(f"Generated training feature matrix shape: {X_train_full.shape}")

        train_feature_data = pd.concat([
            df_train[['compound index', 'SMILES', 'logBB at60min', 'has_18F', 'label_18F']],
            X_train_full
        ], axis=1)

        train_feature_data.to_csv(petbd_features_file, encoding='utf-8', index=False)
        log.info(f"[Chinese text removed]Features saved[Chinese text removed]: {petbd_features_file}")

    log.info("Checking data consistency")
    log.info(f"Training data dimensions: X_train_full {X_train_full.shape}, y_train_full {len(y_train_full)}")
    log.info(f"original[Chinese text removed]: X_original {X_original.shape}, y_original {len(y_original)}")

    if len(y_train_full) != X_train_full.shape[0]:
        log.warning(f"[Chinese text removed]Dimension mismatch: X_train_full {X_train_full.shape[0]} vs y_train_full {len(y_train_full)}")
        min_len = min(len(y_train_full), X_train_full.shape[0])
        X_train_full = X_train_full.iloc[:min_len]
        y_train_full = y_train_full.iloc[:min_len]
        log.info(f"Adjusted training data to consistent dimension: {min_len}")

    if len(y_original) != X_original.shape[0]:
        log.warning(f"original[Chinese text removed]Dimension mismatch: X_original {X_original.shape[0]} vs y_original {len(y_original)}")
        min_len = min(len(y_original), X_original.shape[0])
        X_original = X_original.iloc[:min_len]
        y_original = y_original.iloc[:min_len]
        log.info(f"Adjusted original data to consistent dimension: {min_len}")

    log.info("Processing non-numeric features")
    X_train_full = X_train_full.apply(pd.to_numeric, errors='coerce')
    X_train_full = X_train_full.fillna(0)
    X_original = X_original.apply(pd.to_numeric, errors='coerce')
    X_original = X_original.fillna(0)

    X_train_full.columns = X_train_full.columns.astype(str)
    X_original.columns = X_original.columns.astype(str)

    common_cols = X_train_full.columns.intersection(X_original.columns)
    X_train_full = X_train_full[common_cols]
    X_original = X_original[common_cols]

    log.info(f"Training feature matrix shape: {X_train_full.shape}")
    log.info(f"originalfeatures[Chinese text removed]: {X_original.shape}")

    if not os.path.exists(feature_index_file):
        log.info("[Chinese text removed]features[Chinese text removed]")
        from sklearn.feature_selection import RFE

        estimator = RandomForestRegressor(n_estimators=100, random_state=42)
        selector = RFE(estimator, n_features_to_select=min(50, X_train_full.shape[1]), step=1)
        selector.fit(X_train_full, y_train_full)

        selected_indices = np.where(selector.support_)[0]
        np.savetxt(feature_index_file, selected_indices, fmt='%d', delimiter=',')
        X_train_full = X_train_full.iloc[:, selected_indices]
        X_original = X_original.iloc[:, selected_indices]
        log.info(f"Feature selection complete，Training set[Chinese text removed]: {X_train_full.shape}")
        log.info(f"Feature selection complete，original[Chinese text removed]: {X_original.shape}")
    else:
        log.info("[Chinese text removed]features[Chinese text removed]")
        selected_indices = np.loadtxt(feature_index_file, dtype=int, delimiter=',')

        valid_indices = selected_indices[selected_indices < min(X_train_full.shape[1], X_original.shape[1])]
        if len(valid_indices) != len(selected_indices):
            log.warning(f"[Chinese text removed]Feature indices out of range，[Chinese text removed]")
            selected_indices = valid_indices

        X_train_full = X_train_full.iloc[:, selected_indices]
        X_original = X_original.iloc[:, selected_indices]
        log.info(f"features[Chinese text removed]Training set[Chinese text removed]: {X_train_full.shape}")
        log.info(f"features[Chinese text removed]original[Chinese text removed]: {X_original.shape}")

    log.info("Dataset split (balanced training data)")

    X_train_val, X_test_balanced, y_train_val, y_test_balanced = train_test_split(
        X_train_full, y_train_full, test_size=0.1, random_state=42
    )
    log.info(f"[Chinese text removed]Validation set: {len(X_train_val)}samples, Test set: {len(X_test_balanced)}samples")

    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=0.1, random_state=42
    )
    log.info(f"Training set: {len(X_train)}samples, Validation set: {len(X_val)}samples")

    log.info(f"original[Chinese text removed]: {len(X_original)}samples")

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

    log.info("Starting hyperparameter search (100 trials)...")
    study = optuna.create_study(direction='maximize')
    study.optimize(lambda trial: objective(trial, X_train_val, y_train_val), n_trials=100)

    best_params = study.best_params
    log.info(f"Best hyperparameters: {best_params}")
    log.info(f"Best R²: {study.best_value:.4f}")

    log.info("[Chinese text removed]Best parameters[Chinese text removed]10folds")
    kf = KFold(n_splits=10, shuffle=True, random_state=42)
    cv_scores = {'rmse': [], 'mse': [], 'mae': [], 'r2': [], 'adj_r2': []}

    fold_details = []
    best_fold = {'fold': -1, 'r2': -999}

    for fold, (train_idx, val_idx) in tqdm(enumerate(kf.split(X_train_val, y_train_val)), 
                                          desc="Cross-validation", total=10):
        X_fold_train = X_train_val.iloc[train_idx]
        X_fold_val = X_train_val.iloc[val_idx]
        y_fold_train = y_train_val.iloc[train_idx]
        y_fold_val = y_train_val.iloc[val_idx]

        model = RandomForestRegressor(**best_params)
        model.fit(X_fold_train, y_fold_train)

        y_pred_fold_train = model.predict(X_fold_train)

        y_pred_fold = model.predict(X_fold_val)

        rmse = np.sqrt(mean_squared_error(y_fold_val, y_pred_fold))
        mse = mean_squared_error(y_fold_val, y_pred_fold)
        mae = mean_absolute_error(y_fold_val, y_pred_fold)
        r2 = r2_score(y_fold_val, y_pred_fold)
        adj_r2 = adjusted_r2(y_fold_val, y_pred_fold, X_fold_val.shape[1])

        cv_scores['rmse'].append(rmse)
        cv_scores['mse'].append(mse)
        cv_scores['mae'].append(mae)
        cv_scores['r2'].append(r2)
        cv_scores['adj_r2'].append(adj_r2)

        fold_info = {
            'fold': fold + 1,
            'r2': r2,
            'rmse': rmse,
            'mae': mae,
            'X_train': X_fold_train,
            'X_val': X_fold_val,
            'y_train': y_fold_train,
            'y_val': y_fold_val,
            'y_pred_train': y_pred_fold_train,
            'y_pred_val': y_pred_fold,
            'model': model
        }
        fold_details.append(fold_info)

        if r2 > best_fold['r2']:
            best_fold = fold_info

        log.info(f"Fold {fold + 1}: R²={r2:.4f}, RMSE={rmse:.4f}, MAE={mae:.4f}")

    log.info("========Cross-validation[Chinese text removed]========")
    log.info(f"RMSE: {np.mean(cv_scores['rmse']):.3f}±{np.std(cv_scores['rmse']):.3f}")
    log.info(f"MSE: {np.mean(cv_scores['mse']):.3f}±{np.std(cv_scores['mse']):.3f}")
    log.info(f"MAE: {np.mean(cv_scores['mae']):.3f}±{np.std(cv_scores['mae']):.3f}")
    log.info(f"R²: {np.mean(cv_scores['r2']):.3f}±{np.std(cv_scores['r2']):.3f}")
    log.info(f"Adjusted R²: {np.mean(cv_scores['adj_r2']):.3f}±{np.std(cv_scores['adj_r2']):.3f}")

    log.info(f"\n========[Chinese text removed]（[Chinese text removed] {best_fold['fold']} [Chinese text removed]）========")
    log.info(f"[Chinese text removed] R²: {best_fold['r2']:.4f}")
    log.info(f"[Chinese text removed] RMSE: {best_fold['rmse']:.4f}")
    log.info(f"[Chinese text removed] MAE: {best_fold['mae']:.4f}")

    best_model = best_fold['model']
    X_test_scaled = scaler.transform(X_test_balanced)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test_balanced.columns)
    y_pred_test_best_fold = best_model.predict(X_test_scaled)

    test_r2_best = r2_score(y_test_balanced, y_pred_test_best_fold)
    test_rmse_best = np.sqrt(mean_squared_error(y_test_balanced, y_pred_test_best_fold))
    test_mae_best = mean_absolute_error(y_test_balanced, y_pred_test_best_fold)

    log.info(f"[Chinese text removed]Test set[Chinese text removed]:")
    log.info(f"Test set R²: {test_r2_best:.4f}")
    log.info(f"Test set RMSE: {test_rmse_best:.4f}")
    log.info(f"Test set MAE: {test_mae_best:.4f}")

    best_fold_train_df = pd.DataFrame({
        'True_Values': best_fold['y_train'].values,
        'Predicted_Values': best_fold['y_pred_train'],
        'Dataset': 'Train_Best_Fold'
    })
    best_fold_train_df.to_csv('./result/rf_fp_18F_best_fold_train.csv', index=False)

    best_fold_val_df = pd.DataFrame({
        'True_Values': best_fold['y_val'].values,
        'Predicted_Values': best_fold['y_pred_val'],
        'Dataset': 'Val_Best_Fold'
    })
    best_fold_val_df.to_csv('./result/rf_fp_18F_best_fold_val.csv', index=False)

    best_fold_test_df = pd.DataFrame({
        'True_Values': y_test_balanced.values,
        'Predicted_Values': y_pred_test_best_fold,
        'Dataset': 'Test_Best_Fold'
    })
    best_fold_test_df.to_csv('./result/rf_fp_18F_best_fold_test.csv', index=False)

    best_fold_all_df = pd.concat([best_fold_train_df, best_fold_val_df, best_fold_test_df], 
                                 ignore_index=True)
    best_fold_all_df.to_csv('./result/rf_fp_18F_best_fold_all.csv', index=False)

    log.info("[Chinese text removed]CSV[Chinese text removed]:")
    log.info("  - rf_fp_18F_best_fold_train.csv")
    log.info("  - rf_fp_18F_best_fold_val.csv")
    log.info("  - rf_fp_18F_best_fold_test.csv")
    log.info("  - rf_fp_18F_best_fold_all.csv")

    plt.figure(figsize=(10, 8))
    plt.scatter(y_test_balanced, y_pred_test_best_fold, alpha=0.5, s=30, color='red')

    min_val = min(min(y_test_balanced), min(y_pred_test_best_fold))
    max_val = max(max(y_test_balanced), max(y_pred_test_best_fold))
    plt.plot([min_val, max_val], [min_val, max_val], 'k--', lw=2, alpha=0.8, label='Perfect prediction')

    plt.xlabel('Experimental logBB', fontsize=12)
    plt.ylabel('Predicted logBB', fontsize=12)
    plt.title(f'RF FP 18F-Resample: Best Fold ({best_fold["fold"]}) Test Set Predictions', fontsize=14)

    plt.text(0.05, 0.95, f'R² = {test_r2_best:.4f}', 
             transform=plt.gca().transAxes, fontsize=12, 
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    plt.text(0.05, 0.88, f'RMSE = {test_rmse_best:.4f}', 
             transform=plt.gca().transAxes, fontsize=12,
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    plt.text(0.05, 0.81, f'MAE = {test_mae_best:.4f}', 
             transform=plt.gca().transAxes, fontsize=12,
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    plt.text(0.05, 0.74, f'N = {len(y_test_balanced)}', 
             transform=plt.gca().transAxes, fontsize=12,
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    best_fold_scatter_path = './result/rf_fp_18F_best_fold_test_scatter.png'
    plt.savefig(best_fold_scatter_path, dpi=300, bbox_inches='tight')
    plt.close()

    log.info(f"[Chinese text removed]Test setScatter plot saved to: {best_fold_scatter_path}")

    log.info("Training final model")
    final_model = RandomForestRegressor(**best_params)
    final_model.fit(X_train, y_train)

    y_pred_train = final_model.predict(X_train)
    y_pred_val = final_model.predict(X_val)
    y_pred_test_balanced = final_model.predict(X_test_balanced)

    log.info("[Chinese text removed]")
    X_balanced_full_scaled = scaler.transform(X_train_full)
    X_balanced_full_scaled = pd.DataFrame(X_balanced_full_scaled, columns=X_train_full.columns)
    y_pred_balanced_full = final_model.predict(X_balanced_full_scaled)

    def calculate_metrics(y_true, y_pred, n_features):
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

    train_metrics = calculate_metrics(y_train, y_pred_train, X_train.shape[1])
    val_metrics = calculate_metrics(y_val, y_pred_val, X_val.shape[1])
    test_balanced_metrics = calculate_metrics(y_test_balanced, y_pred_test_balanced, X_test_balanced.shape[1])
    balanced_full_metrics = calculate_metrics(y_train_full, y_pred_balanced_full, X_train_full.shape[1])

    log.info("========[Chinese text removed]Test set[Chinese text removed]========")
    log.info(f"MSE: {test_balanced_metrics['mse']:.4f}")
    log.info(f"RMSE: {test_balanced_metrics['rmse']:.4f}")
    log.info(f"MAE: {test_balanced_metrics['mae']:.4f}")
    log.info(f"R²: {test_balanced_metrics['r2']:.4f}")
    log.info(f"Adjusted R²: {test_balanced_metrics['adj_r2']:.4f}")
    log.info(f"MAPE: {test_balanced_metrics['mape']:.2f}%")

    log.info("========[Chinese text removed]========")
    log.info(f"MSE: {balanced_full_metrics['mse']:.4f}")
    log.info(f"RMSE: {balanced_full_metrics['rmse']:.4f}")
    log.info(f"MAE: {balanced_full_metrics['mae']:.4f}")
    log.info(f"R²: {balanced_full_metrics['r2']:.4f}")
    log.info(f"Adjusted R²: {balanced_full_metrics['adj_r2']:.4f}")
    log.info(f"MAPE: {balanced_full_metrics['mape']:.2f}%")

    model_path = "./logbbModel/rf_fp_18F_model.joblib"
    joblib.dump(final_model, model_path)
    log.info(f"Model saved to: {model_path}")

    results = pd.DataFrame({
        'True_Values': y_train_full,
        'Predicted_Values': y_pred_balanced_full
    })
    results.to_csv('./result/rf_fp_18F_balanced_full_predictions.csv', index=False)

    balanced_results = pd.DataFrame({
        'True_Values': y_test_balanced,
        'Predicted_Values': y_pred_test_balanced
    })
    balanced_results.to_csv('./result/rf_fp_18F_balanced_test_predictions.csv', index=False)

    train_predictions_df = pd.DataFrame({
        'True_Values': y_train.values,
        'Predicted_Values': y_pred_train,
        'Dataset': 'Train'
    })
    train_predictions_df.to_csv('./result/rf_18F_train_predictions.csv', index=False)
    log.info("Training set predictions saved: ./result/rf_18F_train_predictions.csv")

    val_predictions_df = pd.DataFrame({
        'True_Values': y_val.values,
        'Predicted_Values': y_pred_val,
        'Dataset': 'Validation'
    })
    val_predictions_df.to_csv('./result/rf_18F_validation_predictions.csv', index=False)
    log.info("Validation set predictions saved: ./result/rf_18F_validation_predictions.csv")

    test_predictions_df = pd.DataFrame({
        'True_Values': y_test_balanced.values,
        'Predicted_Values': y_pred_test_balanced,
        'Dataset': 'Test'
    })
    test_predictions_df.to_csv('./result/rf_18F_test_predictions.csv', index=False)
    log.info("Test set predictions saved: ./result/rf_18F_test_predictions.csv")

    all_predictions_df = pd.concat([train_predictions_df, val_predictions_df, test_predictions_df], 
                                   ignore_index=True)
    all_predictions_df.to_csv('./result/rf_18F_all_predictions.csv', index=False)
    log.info("Combined predictions saved: ./result/rf_18F_all_predictions.csv")

    log.info("[Chinese text removed]CSV[Chinese text removed]")

    plot_scatter(y_train, y_pred_train, y_val, y_pred_val, y_test_balanced, y_pred_test_balanced, 
                './result/rf_18F_scatter.png')

    experiment_report = {
        'experiment_info': {
            'dataset': 'PTBD_v20240912.csv',
            'training_method': '18F balanced resampling',
            'evaluation_method': 'Full balanced dataset',
            'features': 'Molecular_fingerprints (PaDEL/Morgan)',
            'model': 'Random Forest',
            'n_samples_original': len(df_original),
            'n_samples_balanced_training': len(df_train),
            'n_features': X_train_full.shape[1],
            'seed': 42
        },
        'best_params': best_params,
        'cross_validation': {
            'cv_folds': 10,
            'rmse': f"{np.mean(cv_scores['rmse']):.3f}±{np.std(cv_scores['rmse']):.3f}",
            'r2': f"{np.mean(cv_scores['r2']):.3f}±{np.std(cv_scores['r2']):.3f}",
            'adj_r2': f"{np.mean(cv_scores['adj_r2']):.3f}±{np.std(cv_scores['adj_r2']):.3f}"
        },
        'final_results': {
            'train': train_metrics,
            'validation': val_metrics,
            'test_balanced': test_balanced_metrics,
            'balanced_full_dataset': balanced_full_metrics
        }
    }

    with open('./result/rf_fp_18F_report.json', 'w', encoding='utf-8') as f:
        json.dump(experiment_report, f, ensure_ascii=False, indent=2)

    log.info("========ExperimentComplete========")
    log.info(f"[Chinese text removed]: ./result/rf_fp_18F_results.csv")
    log.info(f"Scatter plot saved to: ./result/rf_fp_18F_scatter.png")
    log.info(f"Experiment[Chinese text removed]: ./result/rf_fp_18F_report.json")

    print(f"\nExperimentComplete！")
    print(f"[Chinese text removed]Test set R² = {test_balanced_metrics['r2']:.4f}, RMSE = {test_balanced_metrics['rmse']:.4f}")
    print(f"[Chinese text removed] R² = {balanced_full_metrics['r2']:.4f}, RMSE = {balanced_full_metrics['rmse']:.4f}")

if __name__ == '__main__':
    main()