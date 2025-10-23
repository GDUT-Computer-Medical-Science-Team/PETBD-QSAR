import sys
import os
from time import time
import numpy as np
import pandas as pd
import optuna
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, mean_absolute_percentage_error
from sklearn.model_selection import train_test_split, KFold
from tqdm import tqdm
from datetime import datetime
import joblib
import json
import matplotlib.pyplot as plt
import re
from typing import Optional
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs
from padelpy import padeldescriptor
import tempfile

sys.path.append('../../')
from preprocess.data_preprocess.FeatureExtraction import FeatureExtraction
from preprocess.data_preprocess.data_preprocess_utils import calculate_Mordred_desc
from utils.DataLogger import DataLogger

os.makedirs("../../data/logBB_data/log", exist_ok=True)
os.makedirs("./logbbModel", exist_ok=True)
os.makedirs("./result", exist_ok=True)

log = DataLogger(log_file=f"../../data/logBB_data/log/{datetime.now().strftime('%Y%m%d')}.log").getlog("petbd_svm_18F_undersample_training")

ALLOWED_ISOTOPES = {"18F", "11C", "125I", "131I", "123I", "77Br", "76Br"}

def extract_isotope_strict(compound_index: str) -> Optional[str]:
    if not isinstance(compound_index, str):
        return None

    text = compound_index.strip()

    if "18F" in text:
        return "18F"

    return None

def balance_18F_dataset_undersample(df: pd.DataFrame, method: str = "combined", seed: int = 42):

    df = df.copy()
    df["isotope"] = df["compound index"].apply(extract_isotope_strict)

    df["label_18F"] = (df["isotope"] == "18F").astype(int)

    pos = df[df["label_18F"] == 1]
    neg = df[df["label_18F"] == 0]

    n_pos, n_neg = len(pos), len(neg)
    log.info(f"Original data distribution - 18F samples: {n_pos}, [Chinese text removed]18F samples: {n_neg}")

    if n_pos == 0 or n_neg == 0:
        raise ValueError("Cannot balance: one class has 0 samples")

    rng = np.random.default_rng(seed)

    if method == "combined":

        if n_pos < n_neg:

            extra_needed = n_neg - n_pos
            pos_extra = pos.sample(n=extra_needed, replace=True, random_state=seed)
            pos_bal = pd.concat([pos, pos_extra], ignore_index=True)
            neg_bal = neg
        elif n_neg < n_pos:

            extra_needed = n_pos - n_neg
            neg_extra = neg.sample(n=extra_needed, replace=True, random_state=seed)
            neg_bal = pd.concat([neg, neg_extra], ignore_index=True)
            pos_bal = pos
        else:

            pos_bal = pos
            neg_bal = neg
    elif method == "undersample":
        target = min(n_pos, n_neg)
        pos_bal = pos.sample(n=target, replace=False, random_state=seed)
        neg_bal = neg.sample(n=target, replace=False, random_state=seed)
    elif method == "oversample":
        target = max(n_pos, n_neg)
        pos_bal = pos.sample(n=target, replace=True, random_state=seed)
        neg_bal = neg.sample(n=target, replace=True, random_state=seed)
    else:
        raise ValueError("method [Chinese text removed] 'combined', 'undersample' [Chinese text removed] 'oversample'")

    balanced = pd.concat([pos_bal, neg_bal], ignore_index=True)
    balanced = balanced.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    log.info(f"Balanced data distribution - 18F samples: {len(pos_bal)}, [Chinese text removed]18F samples: {len(neg_bal)}")

    return balanced

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

def calculate_padel_fingerprints(smiles_list):
    with tempfile.NamedTemporaryFile(mode='w', suffix='.smi', delete=False) as temp_file:
        for smi in smiles_list:
            temp_file.write(f"{smi}\\n")
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

def adjusted_r2_score(r2, n, k):
    return 1 - (1 - r2) * (n - 1) / (n - k - 1)

if __name__ == '__main__':
    log.info("===============StartingPETBD[Chinese text removed]SVM 18F[Chinese text removed]===============")

    petbd_data_file = "../../data/PTBD_v20240912.csv"
    petbd_features_file = "../../data/PETBD_SVM_18F_undersample_features.csv"
    feature_index_file = "../../data/PETBD_SVM_18F_undersample_feature_index.txt"

    smile_column_name = 'SMILES'
    pred_column_name = 'logBB at60min'
    RFE_features_to_select = 50
    n_optuna_trial = 100
    cv_times = 10
    seed = 42

    if not os.path.exists(petbd_data_file):
        raise FileNotFoundError(f"Missing PETBD dataset: {petbd_data_file}")

    log.info("Reading PETBD dataset")
    df = pd.read_csv(petbd_data_file, encoding='utf-8')
    log.info(f"original[Chinese text removed]: {df.shape}")

    df = df.dropna(subset=[pred_column_name])
    log.info(f"Data shape after removing logBB missing values: {df.shape}")

    log.info("Starting18F[Chinese text removed]（[Chinese text removed]）")
    df_balanced = balance_18F_dataset_undersample(df, method="combined", seed=seed)
    log.info(f"18F[Chinese text removed]: {df_balanced.shape}")

    if os.path.exists(petbd_features_file):
        log.info("Feature file exists, reading")
        features_df = pd.read_csv(petbd_features_file, encoding='utf-8')

        if len(features_df) != len(df_balanced):
            log.info("features[Chinese text removed]，regenerating features")
            should_regenerate_features = True
        elif pred_column_name not in features_df.columns:
            log.info("features[Chinese text removed]in[Chinese text removed]，regenerating features")
            should_regenerate_features = True
        else:
            should_regenerate_features = False

        if not should_regenerate_features:
            y = features_df[pred_column_name]
            X = features_df.drop([smile_column_name, pred_column_name], axis=1)

            label_cols = ['isotope', 'label_18F', 'compound index']
            for col in label_cols:
                if col in X.columns:
                    X = X.drop(col, axis=1)

    if should_regenerate_features:
        log.info("Startingfeatures[Chinese text removed]")

        y = df_balanced[pred_column_name]
        SMILES = df_balanced[smile_column_name]

        log.info("[Chinese text removed]Mordred[Chinese text removed]")
        X_mordred = calculate_Mordred_desc(SMILES)

        log.info("[Chinese text removed]Morgan[Chinese text removed]")
        X_morgan = calculate_morgan_fingerprints(SMILES)

        log.info("[Chinese text removed]PaDEL[Chinese text removed]，[Chinese text removed]Mordred descriptors[Chinese text removed]Morgan fingerprints")

        feature_dfs = [X_mordred, X_morgan]

        log.info("[Chinese text removed]features")
        X = pd.concat(feature_dfs, axis=1)

        if smile_column_name in X.columns:
            X = X.drop(smile_column_name, axis=1)

        log.info(f"[Chinese text removed]features[Chinese text removed]: {X.shape}")

        feature_data = pd.concat([
            df_balanced[['compound index', smile_column_name, pred_column_name, 'isotope', 'label_18F']],
            X
        ], axis=1)

        feature_data.to_csv(petbd_features_file, encoding='utf-8', index=False)
        log.info(f"features[Chinese text removed]: {petbd_features_file}")

    log.info("Processing non-numeric features")
    X = X.apply(pd.to_numeric, errors='coerce')
    X = X.fillna(0)

    X.columns = X.columns.astype(str)

    log.info(f"features[Chinese text removed]: {X.shape}")

    if not os.path.exists(feature_index_file):
        log.info("[Chinese text removed]features[Chinese text removed]")
        log.info(f"features[Chinese text removed]: {X.shape}")

        feature_extractor = FeatureExtraction(
            X, y,
            VT_threshold=0.02,
            RFE_features_to_select=RFE_features_to_select
        )

        selected_indices = feature_extractor.feature_extraction(returnIndex=True, index_dtype=int)

        try:
            np.savetxt(feature_index_file, selected_indices, fmt='%d')
            X = X.iloc[:, selected_indices]
            log.info(f"Feature selection complete，[Chinese text removed]: {X.shape}")
            log.info(f"Feature indices saved[Chinese text removed]: {feature_index_file}")
        except Exception as e:
            log.error(f"features[Chinese text removed]Failed: {e}")
            if os.path.exists(feature_index_file):
                os.remove(feature_index_file)
            sys.exit()
    else:
        log.info("[Chinese text removed]features[Chinese text removed]")
        selected_indices = np.loadtxt(feature_index_file, dtype=int, delimiter=',').tolist()
        X = X.iloc[:, selected_indices]
        log.info(f"features[Chinese text removed]: {X.shape}")

    log.info("Feature normalization")
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    X = pd.DataFrame(X_scaled, columns=X.columns)

    log.info("Dataset split")

    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=0.1, random_state=42, stratify=df_balanced['label_18F']
    )
    log.info(f"[Chinese text removed]Validation set: {len(X_train_val)}samples, Test set: {len(X_test)}samples")

    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=0.111, random_state=seed
    )
    log.info(f"Training set: {len(X_train)}samples, Validation set: {len(X_val)}samples")

    scaler_final = MinMaxScaler()
    X_train_val_scaled = scaler_final.fit_transform(X_train_val)
    X_test_scaled = scaler_final.transform(X_test)

    X_train_val = pd.DataFrame(X_train_val_scaled, columns=X.columns)
    X_test = pd.DataFrame(X_test_scaled, columns=X.columns)

    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=0.111, random_state=seed
    )

    y_train_val = y_train_val.reset_index(drop=True)
    y_train = y_train.reset_index(drop=True)
    y_val = y_val.reset_index(drop=True)
    y_test = y_test.reset_index(drop=True)

    log.info("StartingSVM[Chinese text removed]parameter[Chinese text removed]")

    def objective(trial):
        X_tr, X_te, y_tr, y_te = train_test_split(
            X_train_val, y_train_val, test_size=0.1, random_state=seed
        )

        params = {
            'kernel': trial.suggest_categorical('kernel', ['rbf', 'linear', 'poly']),
            'C': trial.suggest_float('C', 1e-3, 1e3, log=True),
            'epsilon': trial.suggest_float('epsilon', 1e-3, 1.0, log=True),
        }

        if params['kernel'] == 'poly':
            params['degree'] = trial.suggest_int('degree', 2, 5)

        if params['kernel'] in ['rbf', 'poly']:
            gamma_type = trial.suggest_categorical('gamma_type', ['scale', 'auto', 'float'])
            if gamma_type == 'float':
                params['gamma'] = trial.suggest_float('gamma_value', 1e-5, 1e-1, log=True)
            else:
                params['gamma'] = gamma_type
        else:
            params['gamma'] = 'scale'

        model = SVR(**params)
        model.fit(X_tr, y_tr)
        y_pred = model.predict(X_te)

        return r2_score(y_te, y_pred)

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_jobs=4, n_trials=n_optuna_trial)

    log.info(f"Best parameters: {study.best_params}")
    log.info(f"[Chinese text removed]R²[Chinese text removed]: {study.best_value}")

    log.info(f"[Chinese text removed]Best parameters[Chinese text removed]{cv_times}folds")

    svm_params = {k: v for k, v in study.best_params.items() 
                  if k not in ['gamma_type']}
    best_model = SVR(**svm_params)
    cv = KFold(n_splits=cv_times, random_state=seed, shuffle=True)

    cv_scores = {'rmse': [], 'mse': [], 'mae': [], 'r2': [], 'adj_r2': []}

    for fold, (train_idx, val_idx) in tqdm(enumerate(cv.split(X_train_val, y_train_val)), 
                                          desc="Cross-validation", total=cv_times):
        X_fold_train = X_train_val.iloc[train_idx]
        X_fold_val = X_train_val.iloc[val_idx]
        y_fold_train = y_train_val.iloc[train_idx]
        y_fold_val = y_train_val.iloc[val_idx]

        best_model.fit(X_fold_train, y_fold_train)

        y_pred_fold = best_model.predict(X_fold_val)

        rmse = np.sqrt(mean_squared_error(y_fold_val, y_pred_fold))
        mse = mean_squared_error(y_fold_val, y_pred_fold)
        mae = mean_absolute_error(y_fold_val, y_pred_fold)
        r2 = r2_score(y_fold_val, y_pred_fold)
        adj_r2 = adjusted_r2_score(r2, len(y_fold_val), X_fold_val.shape[1])

        cv_scores['rmse'].append(rmse)
        cv_scores['mse'].append(mse)
        cv_scores['mae'].append(mae)
        cv_scores['r2'].append(r2)
        cv_scores['adj_r2'].append(adj_r2)

    log.info("========Cross-validation[Chinese text removed]========")
    log.info(f"RMSE: {np.mean(cv_scores['rmse']):.3f}±{np.std(cv_scores['rmse']):.3f}")
    log.info(f"MSE: {np.mean(cv_scores['mse']):.3f}±{np.std(cv_scores['mse']):.3f}")
    log.info(f"MAE: {np.mean(cv_scores['mae']):.3f}±{np.std(cv_scores['mae']):.3f}")
    log.info(f"R²: {np.mean(cv_scores['r2']):.3f}±{np.std(cv_scores['r2']):.3f}")
    log.info(f"Adjusted R²: {np.mean(cv_scores['adj_r2']):.3f}±{np.std(cv_scores['adj_r2']):.3f}")

    log.info("Training final model")
    final_model = SVR(**svm_params)
    final_model.fit(X_train, y_train)

    y_pred_train = final_model.predict(X_train)
    y_pred_val = final_model.predict(X_val)
    y_pred_test = final_model.predict(X_test)

    def calculate_metrics(y_true, y_pred, n_features):
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        mape = mean_absolute_percentage_error(y_true, y_pred)
        adj_r2 = adjusted_r2_score(r2, len(y_true), n_features)

        return {
            'mse': mse, 'rmse': rmse, 'mae': mae,
            'r2': r2, 'mape': mape, 'adj_r2': adj_r2
        }

    train_metrics = calculate_metrics(y_train, y_pred_train, X_train.shape[1])
    val_metrics = calculate_metrics(y_val, y_pred_val, X_val.shape[1])
    test_metrics = calculate_metrics(y_test, y_pred_test, X_test.shape[1])

    log.info("========[Chinese text removed]Test set[Chinese text removed]========")
    log.info(f"MSE: {test_metrics['mse']:.4f}")
    log.info(f"RMSE: {test_metrics['rmse']:.4f}")
    log.info(f"MAE: {test_metrics['mae']:.4f}")
    log.info(f"R²: {test_metrics['r2']:.4f}")
    log.info(f"Adjusted R²: {test_metrics['adj_r2']:.4f}")
    log.info(f"MAPE: {test_metrics['mape']:.2f}%")

    model_path = "./logbbModel/svm_petbd_18F_undersample_model.joblib"
    joblib.dump(final_model, model_path)
    log.info(f"Model saved to: {model_path}")

    results = pd.DataFrame({
        'True_Values': y_test,
        'Predicted_Values': y_pred_test
    })
    results.to_csv('./result/svm_petbd_18F_undersample_results.csv', index=False)

    train_predictions_df = pd.DataFrame({
        'True_Values': y_train.values,
        'Predicted_Values': y_pred_train,
        'Dataset': 'Train'
    })
    train_predictions_df.to_csv('./result/svm_petbd_18F_undersample_train_predictions.csv', index=False)

    val_predictions_df = pd.DataFrame({
        'True_Values': y_val.values,
        'Predicted_Values': y_pred_val,
        'Dataset': 'Validation'
    })
    val_predictions_df.to_csv('./result/svm_petbd_18F_undersample_val_predictions.csv', index=False)

    test_predictions_df = pd.DataFrame({
        'True_Values': y_test.values,
        'Predicted_Values': y_pred_test,
        'Dataset': 'Test'
    })
    test_predictions_df.to_csv('./result/svm_petbd_18F_undersample_test_predictions.csv', index=False)

    all_predictions_df = pd.concat([train_predictions_df, val_predictions_df, test_predictions_df], 
                                   ignore_index=True)
    all_predictions_df.to_csv('./result/svm_petbd_18F_undersample_all_predictions.csv', index=False)

    log.info("[Chinese text removed]CSV[Chinese text removed]")

    plt.figure(figsize=(10, 8))
    plt.scatter(y_train, y_pred_train, alpha=0.5, label='Training Set', color='blue', s=20)
    plt.scatter(y_val, y_pred_val, alpha=0.7, label='Validation Set', color='green', s=20)
    plt.scatter(y_test, y_pred_test, alpha=0.8, label='Test Set', color='red', s=30)

    all_min = min(min(y_train), min(y_val), min(y_test))
    all_max = max(max(y_train), max(y_val), max(y_test))
    plt.plot([all_min, all_max], [all_min, all_max], 'k--', lw=2, alpha=0.8)

    plt.xlabel('Experimental logBB')
    plt.ylabel('Predicted logBB')
    plt.title('SVM PETBD 18F-Undersampled Model: Predicted vs Experimental logBB')
    plt.legend()

    plt.text(0.05, 0.95, f'Train R² = {train_metrics["r2"]:.3f}', 
             transform=plt.gca().transAxes, fontsize=10)
    plt.text(0.05, 0.90, f'Val R² = {val_metrics["r2"]:.3f}', 
             transform=plt.gca().transAxes, fontsize=10)
    plt.text(0.05, 0.85, f'Test R² = {test_metrics["r2"]:.3f}', 
             transform=plt.gca().transAxes, fontsize=10)

    plt.tight_layout()
    plt.savefig('./result/svm_petbd_18F_undersample_scatter.png', dpi=300, bbox_inches='tight')
    plt.close()

    experiment_report = {
        'experiment_info': {
            'dataset': 'PTBD_v20240912.csv',
            'balancing_method': 'undersample_balanced',
            'features': 'Mordred_descriptors + Morgan_fingerprints',
            'model': 'SVM',
            'n_samples_original': len(df),
            'n_samples_balanced': len(df_balanced),
            'n_features': X.shape[1],
            'seed': seed
        },
        'dataset_split': {
            'train_samples': len(X_train),
            'validation_samples': len(X_val),
            'test_samples': len(X_test),
            'total_samples': len(df_balanced)
        },
        'best_params': study.best_params,
        'cross_validation': {
            'cv_folds': cv_times,
            'rmse': f"{np.mean(cv_scores['rmse']):.3f}±{np.std(cv_scores['rmse']):.3f}",
            'r2': f"{np.mean(cv_scores['r2']):.3f}±{np.std(cv_scores['r2']):.3f}",
            'adj_r2': f"{np.mean(cv_scores['adj_r2']):.3f}±{np.std(cv_scores['adj_r2']):.3f}"
        },
        'final_results': {
            'train': train_metrics,
            'validation': val_metrics,
            'test': test_metrics
        }
    }

    with open('./result/svm_petbd_18F_undersample_report.json', 'w', encoding='utf-8') as f:
        json.dump(experiment_report, f, ensure_ascii=False, indent=2)

    log.info("========ExperimentComplete========")
    log.info(f"[Chinese text removed]: ./result/svm_petbd_18F_undersample_results.csv")
    log.info(f"Scatter plot saved to: ./result/svm_petbd_18F_undersample_scatter.png") 
    log.info(f"Experiment[Chinese text removed]: ./result/svm_petbd_18F_undersample_report.json")

    print(f"\\nExperimentComplete！Test set R² = {test_metrics['r2']:.4f}, RMSE = {test_metrics['rmse']:.4f}")
    print(f"[Chinese text removed]: Training set={len(X_train)}, Validation set={len(X_val)}, Test set={len(X_test)}, [Chinese text removed]={len(df_balanced)}samples")
    print(f"18F[Chinese text removed]: [Chinese text removed]{len(df_balanced)}samples（original{len(df)}samples）")