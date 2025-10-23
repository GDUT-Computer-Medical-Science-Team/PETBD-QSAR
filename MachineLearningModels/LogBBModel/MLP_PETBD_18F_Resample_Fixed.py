import sys
import os
from time import time
import numpy as np
import pandas as pd
import optuna
from sklearn.preprocessing import MinMaxScaler
from sklearn.neural_network import MLPRegressor
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
import warnings
warnings.filterwarnings('ignore')

sys.path.append('../../')
from preprocess.data_preprocess.FeatureExtraction import FeatureExtraction
from preprocess.data_preprocess.data_preprocess_utils import calculate_Mordred_desc
from utils.DataLogger import DataLogger

os.makedirs("../../data/logBB_data/log", exist_ok=True)
os.makedirs("./logbbModel", exist_ok=True)
os.makedirs("./result", exist_ok=True)

log = DataLogger(log_file=f"../../data/logBB_data/log/{datetime.now().strftime('%Y%m%d')}.log").getlog("petbd_mlp_18F_training_fixed")

def contains_18F(compound_index: str) -> bool:
    if not isinstance(compound_index, str):
        return False

    text = str(compound_index).strip()

    return '18F' in text

def balance_18F_dataset(df: pd.DataFrame, method: str = "oversample", seed: int = 42):

    df = df.copy()
    df["has_18F"] = df["compound index"].apply(contains_18F)

    log.info(f"Total samples with logBB: {len(df)}")

    df["label_18F"] = df["has_18F"].astype(int)

    pos = df[df["label_18F"] == 1]
    neg = df[df["label_18F"] == 0]

    n_pos, n_neg = len(pos), len(neg)
    log.info(f"Original distribution - 18F samples: {n_pos}, Non-18F samples: {n_neg}")

    if n_pos == 0 or n_neg == 0:
        raise ValueError("Cannot balance: one class has 0 samples")

    rng = np.random.default_rng(seed)

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

    log.info(f"Balanced distribution - 18F samples: {len(pos_bal)}, Non-18F samples: {len(neg_bal)}")
    log.info(f"Total samples after balancing: {len(balanced)}")

    return balanced

def calculate_morgan_fingerprints(smiles_list, radius=2, n_bits=1024):
    fingerprints = []
    for smiles in tqdm(smiles_list, desc="Calculating Morgan fingerprints"):
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
            arr = np.zeros((1,))
            DataStructs.ConvertToNumpyArray(fp, arr)
            fingerprints.append(arr)
        else:
            fingerprints.append(np.zeros(n_bits))

    df = pd.DataFrame(fingerprints, columns=[f'Morgan_{i}' for i in range(n_bits)])
    return df

def calculate_padel_fingerprints(smiles_list, fingerprint_type='PubchemFingerprinter'):
    try:
        import tempfile

        with tempfile.NamedTemporaryFile(mode='w', suffix='.smi', delete=False) as f:
            for i, smiles in enumerate(smiles_list):
                f.write(f'{smiles}\tmol_{i}\n')
            smiles_file = f.name

        output_file = tempfile.mktemp(suffix='.csv')

        padeldescriptor(
            mol=smiles_file,
            d_file=output_file,
            fingerprints=True,
            descriptors=False,
            sp=fingerprint_type,
            threads=-1,
            removesalt=True,
            detectaromaticity=True,
            standardizenitro=True
        )

        df = pd.read_csv(output_file)
        df = df.drop(['Name'], axis=1)

        os.unlink(smiles_file)
        os.unlink(output_file)

        return df
    except Exception as e:
        log.error(f"PaDEL fingerprint calculation failed: {e}")
        return pd.DataFrame()

def train_mlp_18F_resample_fixed():

    petbd_data_file = "../../data/PTBD_v20240912.csv"
    petbd_features_file = "../../data/PETBD_mlp_w_features_fixed.csv"
    feature_index_file = "../../data/PETBD_mlp_feature_index_fixed.txt"

    smile_column_name = 'SMILES'
    pred_column_name = 'logBB at60min'
    RFE_features_to_select = 50
    n_optuna_trial = 50
    cv_times = 5
    seed = int(time())

    if not os.path.exists(petbd_data_file):
        raise FileNotFoundError(f"Missing PETBD dataset: {petbd_data_file}")

    log.info("Reading PETBD dataset")
    df = pd.read_csv(petbd_data_file, encoding='utf-8')
    log.info(f"Original data shape: {df.shape}")

    df = df.dropna(subset=[pred_column_name])
    log.info(f"Data shape after removing logBB missing values: {df.shape}")

    log.info("Starting 18F resampling")
    df_balanced = balance_18F_dataset(df, method="oversample", seed=seed)
    log.info(f"Data shape after 18F resampling: {df_balanced.shape}")

    if os.path.exists(petbd_features_file):
        log.info("Loading existing feature file")
        features_df = pd.read_csv(petbd_features_file, encoding='utf-8')

        if len(features_df) != len(df_balanced):
            log.info("Feature file doesn't match balanced data, regenerating features")
            raise FileNotFoundError("Feature file needs regeneration")

        y = features_df[pred_column_name]
        X = features_df.drop([smile_column_name, pred_column_name], axis=1)

        label_cols = ['has_18F', 'label_18F', 'compound index']
        for col in label_cols:
            if col in X.columns:
                X = X.drop(col, axis=1)

    else:
        log.info("Feature file doesn't exist, starting feature generation")

        y = df_balanced[pred_column_name]
        SMILES = df_balanced[smile_column_name]

        log.info("Calculating Mordred molecular descriptors")
        X_mordred = calculate_Mordred_desc(SMILES)

        log.info("Calculating Morgan fingerprints")
        X_morgan = calculate_morgan_fingerprints(SMILES)

        log.info("Trying to calculate PaDEL fingerprints")
        X_padel = calculate_padel_fingerprints(SMILES)

        feature_dfs = [X_mordred, X_morgan]
        if not X_padel.empty:
            log.info(f"PaDEL fingerprints calculated successfully, features: {X_padel.shape[1]}")
            feature_dfs.append(X_padel)
        else:
            log.info("PaDEL fingerprint calculation failed, using only Morgan fingerprints")

        log.info("Combining molecular descriptors and fingerprints")
        X = pd.concat(feature_dfs, axis=1)

        if smile_column_name in X.columns:
            X = X.drop(smile_column_name, axis=1)

        log.info(f"Combined feature matrix shape: {X.shape}")

        feature_data = pd.concat([
            df_balanced[['compound index', smile_column_name, pred_column_name, 'has_18F', 'label_18F']],
            X
        ], axis=1)

        feature_data.to_csv(petbd_features_file, encoding='utf-8', index=False)
        log.info(f"Feature data saved to: {petbd_features_file}")

    log.info("Processing non-numeric features")
    X = X.apply(pd.to_numeric, errors='coerce')
    X = X.fillna(0)

    X.columns = X.columns.astype(str)

    log.info(f"Feature matrix shape after processing: {X.shape}")

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
            log.info(f"Feature selection complete, shape after selection: {X.shape}")
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

    log.info("Feature normalization")
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    X = pd.DataFrame(X_scaled, columns=X.columns)

    log.info("Dataset split")

    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=0.1, random_state=42, stratify=df_balanced['label_18F']
    )
    log.info(f"Train+Val set: {len(X_train_val)} samples, Test set: {len(X_test)} samples")

    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=0.1, random_state=42
    )
    log.info(f"Train set: {len(X_train)} samples, Validation set: {len(X_val)} samples")

    scaler_final = MinMaxScaler()
    X_train_val_scaled = scaler_final.fit_transform(X_train_val)
    X_test_scaled = scaler_final.transform(X_test)

    X_train_val = pd.DataFrame(X_train_val_scaled, columns=X.columns)
    X_test = pd.DataFrame(X_test_scaled, columns=X.columns)

    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=0.1, random_state=42
    )

    y_train_val = y_train_val.reset_index(drop=True)
    y_train = y_train.reset_index(drop=True)
    y_val = y_val.reset_index(drop=True)
    y_test = y_test.reset_index(drop=True)

    log.info("Starting MLP hyperparameter optimization")

    def objective(trial):
        X_tr, X_te, y_tr, y_te = train_test_split(
            X_train_val, y_train_val, test_size=0.1, random_state=seed
        )

        hidden_layer_sizes = tuple([
            trial.suggest_int(f'n_units_l{i}', 50, 200) 
            for i in range(trial.suggest_int('n_layers', 1, 3))
        ])

        params = {
            'hidden_layer_sizes': hidden_layer_sizes,
            'activation': trial.suggest_categorical('activation', ['relu', 'tanh']),
            'solver': trial.suggest_categorical('solver', ['adam', 'lbfgs']),
            'alpha': trial.suggest_float('alpha', 1e-5, 1e-1, log=True),
            'learning_rate': trial.suggest_categorical('learning_rate', ['constant', 'adaptive']),
            'learning_rate_init': trial.suggest_float('learning_rate_init', 1e-4, 1e-2, log=True),
            'max_iter': 2000,
            'early_stopping': True,
            'validation_fraction': 0.1,
            'random_state': seed
        }

        model = MLPRegressor(**params)
        model.fit(X_tr, y_tr)
        y_pred = model.predict(X_te)
        rmse = np.sqrt(mean_squared_error(y_te, y_pred))

        return rmse

    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=n_optuna_trial, show_progress_bar=True)

    best_params = study.best_params

    n_layers = best_params.pop('n_layers')
    hidden_layer_sizes = tuple([best_params.pop(f'n_units_l{i}') for i in range(n_layers)])
    best_params['hidden_layer_sizes'] = hidden_layer_sizes
    best_params['max_iter'] = 2000
    best_params['early_stopping'] = True
    best_params['validation_fraction'] = 0.1
    best_params['random_state'] = seed

    log.info(f"Best hyperparameters: {best_params}")

    log.info("Performing cross-validation")
    model_cv = MLPRegressor(**best_params)

    kf = KFold(n_splits=cv_times, shuffle=True, random_state=seed)
    cv_rmse_scores = []
    cv_r2_scores = []

    for train_idx, test_idx in tqdm(kf.split(X_train_val), total=cv_times, desc="Cross-validation"):
        X_cv_train, X_cv_test = X_train_val.iloc[train_idx], X_train_val.iloc[test_idx]
        y_cv_train, y_cv_test = y_train_val.iloc[train_idx], y_train_val.iloc[test_idx]

        model_cv.fit(X_cv_train, y_cv_train)
        y_cv_pred = model_cv.predict(X_cv_test)

        cv_rmse = np.sqrt(mean_squared_error(y_cv_test, y_cv_pred))
        cv_r2 = r2_score(y_cv_test, y_cv_pred)

        cv_rmse_scores.append(cv_rmse)
        cv_r2_scores.append(cv_r2)

    cv_rmse = f"{np.mean(cv_rmse_scores):.3f}±{np.std(cv_rmse_scores):.3f}"
    cv_r2 = f"{np.mean(cv_r2_scores):.3f}±{np.std(cv_r2_scores):.3f}"

    log.info(f"Cross-validation RMSE: {cv_rmse}")
    log.info(f"Cross-validation R²: {cv_r2}")

    log.info("Training final model")
    final_model = MLPRegressor(**best_params)
    final_model.fit(X_train, y_train)

    y_pred_train = final_model.predict(X_train)
    y_pred_val = final_model.predict(X_val)
    y_pred_test = final_model.predict(X_test)

    def calculate_metrics(y_true, y_pred):
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        mape = mean_absolute_percentage_error(y_true, y_pred)
        n = len(y_true)
        p = X_train.shape[1]
        adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1) if n > p + 1 else r2

        return {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'mape': mape,
            'adj_r2': adj_r2
        }

    train_metrics = calculate_metrics(y_train, y_pred_train)
    val_metrics = calculate_metrics(y_val, y_pred_val)
    test_metrics = calculate_metrics(y_test, y_pred_test)

    log.info("Training metrics:")
    for key, value in train_metrics.items():
        log.info(f"  {key}: {value:.4f}")

    log.info("Validation metrics:")
    for key, value in val_metrics.items():
        log.info(f"  {key}: {value:.4f}")

    log.info("Test metrics:")
    for key, value in test_metrics.items():
        log.info(f"  {key}: {value:.4f}")

    model_filename = "./logbbModel/mlp_petbd_18F_fixed_model.joblib"
    scaler_filename = "./logbbModel/mlp_petbd_18F_fixed_scaler.joblib"
    joblib.dump(final_model, model_filename)
    joblib.dump(scaler_final, scaler_filename)
    log.info(f"Model saved to: {model_filename}")
    log.info(f"Scaler saved to: {scaler_filename}")

    train_predictions_df = pd.DataFrame({
        'True_Values': y_train.values,
        'Predicted_Values': y_pred_train,
        'Dataset': 'Train'
    })
    train_predictions_df.to_csv('./result/mlp_petbd_18F_fixed_train_predictions.csv', index=False)

    val_predictions_df = pd.DataFrame({
        'True_Values': y_val.values,
        'Predicted_Values': y_pred_val,
        'Dataset': 'Validation'
    })
    val_predictions_df.to_csv('./result/mlp_petbd_18F_fixed_val_predictions.csv', index=False)

    test_predictions_df = pd.DataFrame({
        'True_Values': y_test.values,
        'Predicted_Values': y_pred_test,
        'Dataset': 'Test'
    })
    test_predictions_df.to_csv('./result/mlp_petbd_18F_fixed_test_predictions.csv', index=False)

    all_predictions_df = pd.concat([train_predictions_df, val_predictions_df, test_predictions_df])
    all_predictions_df.to_csv('./result/mlp_petbd_18F_fixed_all_predictions.csv', index=False)
    log.info("Predictions saved to CSV files")

    report = {
        'experiment_info': {
            'dataset': 'PTBD_v20240912.csv',
            'balancing_method': 'oversample',
            'features': 'Mordred_descriptors + Molecular_fingerprints',
            'model': 'MLP (Neural Network)',
            'n_samples_original': len(df),
            'n_samples_balanced': len(df_balanced),
            'n_features': X_train.shape[1],
            'seed': seed
        },
        'best_params': {k: str(v) if isinstance(v, tuple) else v for k, v in best_params.items()},
        'cross_validation': {
            'cv_folds': cv_times,
            'rmse': cv_rmse,
            'r2': cv_r2,
            'adj_r2': f"{np.mean([1 - (1 - r) * (len(X_train_val) - 1) / (len(X_train_val) - X_train.shape[1] - 1) for r in cv_r2_scores]):.3f}±{np.std([1 - (1 - r) * (len(X_train_val) - 1) / (len(X_train_val) - X_train.shape[1] - 1) for r in cv_r2_scores]):.3f}"
        },
        'final_results': {
            'train': train_metrics,
            'validation': val_metrics,
            'test': test_metrics
        }
    }

    with open('./result/mlp_petbd_18F_fixed_report.json', 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2)

    log.info("Experiment report saved to: ./result/mlp_petbd_18F_fixed_report.json")

    return final_model, test_metrics

if __name__ == "__main__":
    model, metrics = train_mlp_18F_resample_fixed()
    print(f"\nFinal test R²: {metrics['r2']:.4f}")
    print(f"Final test RMSE: {metrics['rmse']:.4f}")
    print(f"Expected samples after balancing: ~784 (392*2)")