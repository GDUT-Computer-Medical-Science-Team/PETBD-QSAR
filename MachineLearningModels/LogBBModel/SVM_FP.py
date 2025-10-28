import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime
import os
import matplotlib.pyplot as plt
from utils.DataLogger import DataLogger
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, mean_absolute_percentage_error
from sklearn.svm import SVR
import optuna
import tempfile
from padelpy import padeldescriptor
import joblib
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestRegressor  # [Chinese text removed]features[Chinese text removed]

# [Chinese text removed]
log = DataLogger(log_file=f"../../data/logBB_data/log/{datetime.now().strftime('%Y%m%d')}.log").getlog("svm_feature_selection")

# [Chinese text removed]
model_dir = "logbbModel"
if not os.path.exists(model_dir):
    os.makedirs(model_dir)


def calculate_padel_fingerprints(smiles_list):
    """
    [Chinese text removed]PaDEL[Chinese text removed]
    :param smiles_list: SMILES [Chinese text removed]
    :return: DataFrame，[Chinese text removed]
    """
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
            df.insert(0, 'SMILES', smiles_list)
            return df
    except Exception as e:
        log.error(f"[Chinese text removed]PaDEL[Chinese text removed]Error: {str(e)}")
        raise
    finally:
        if os.path.exists(temp_smi_file):
            os.remove(temp_smi_file)


def adjusted_r2(y_true, y_pred, num_features):
    """
    Calculate adjusted R²
    :param y_true: true values
    :param y_pred: predicted values
    :param num_features: number of features used by the model
    :return: adjusted R²
    """
    n = len(y_true)
    r2 = r2_score(y_true, y_pred)
    # avoid division by zero
    if n - num_features - 1 == 0:
        return r2
    return 1 - (1 - r2) * (n - 1) / (n - num_features - 1)


def plot_scatter(y_true_train, y_pred_train, y_true_test, y_pred_test):
    """
    [Chinese text removed]
    """
    plt.figure(figsize=(12, 6))

    # [Chinese text removed]
    plt.subplot(1, 2, 1)
    plt.scatter(y_true_train, y_pred_train, color='blue', alpha=0.5, label='Train')
    plt.plot([y_true_train.min(), y_true_train.max()],
             [y_true_train.min(), y_true_train.max()],
             color='red', lw=2)
    plt.title('Train Set: True vs Predicted')
    plt.xlabel('True Values')
    plt.ylabel('Predicted Values')
    plt.legend()

    # [Chinese text removed]
    plt.subplot(1, 2, 2)
    plt.scatter(y_true_test, y_pred_test, color='green', alpha=0.5, label='Test')
    plt.plot([y_true_test.min(), y_true_test.max()],
             [y_true_test.min(), y_true_test.max()],
             color='red', lw=2)
    plt.title('Test Set: True vs Predicted')
    plt.xlabel('True Values')
    plt.ylabel('Predicted Values')
    plt.legend()

    plt.tight_layout()
    plt.show()


# --------------------------------------------------------------------
# 1. [Chinese text removed]parameter[Chinese text removed]（[Chinese text removed]Cross-validation，[Chinese text removed]/[Chinese text removed]，[Chinese text removed]）
def objective(trial, X, y):
    param = {
        'C': trial.suggest_float('C', 0.1, 10.0),
        'epsilon': trial.suggest_float('epsilon', 0.01, 1.0),
        'kernel': trial.suggest_categorical('kernel', ['linear', 'poly', 'rbf', 'sigmoid']),
    }
    
    # [Chinese text removed] 80/20 [Chinese text removed]
    X_train_split, X_valid_split, y_train_split, y_valid_split = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = SVR(**param)
    model.fit(X_train_split, y_train_split)
    y_valid_pred = model.predict(X_valid_split)
    rmse = np.sqrt(mean_squared_error(y_valid_split, y_valid_pred))
    return rmse


# --------------------------------------------------------------------
def main():
    logBB_data_file = "../../data/logBB_data/logBB.csv"
    logBB_desc_file = "../../data/logBB_data/logBB_w_desc_fp.csv"
    feature_index_path = "../../data/logBB_data/logBB_w_desc_fp_index2.txt"  # features[Chinese text removed]
    log.info("===============Starting SVM features[Chinese text removed]===============")

    # [Chinese text removed]
    if not os.path.exists(logBB_data_file):
        log.error("[Chinese text removed]logBB[Chinese text removed]")
        raise FileNotFoundError("[Chinese text removed]logBB[Chinese text removed]")

    # [Chinese text removed]
    df = pd.read_csv(logBB_data_file, encoding='utf-8')
    df = df.dropna(subset=['logBB']).reset_index(drop=True)

    # [Chinese text removed] Compound index [Chinese text removed]
    if 'Compound index' in df.columns:
        df = df.drop(columns=['Compound index'])
        log.info("[Chinese text removed] Compound index [Chinese text removed]")

    SMILES = df['SMILES']
    log.info(f"[Chinese text removed] {len(SMILES)} [Chinese text removed] SMILES [Chinese text removed]")

    # [Chinese text removed]
    df_desc = calculate_padel_fingerprints(SMILES)
    log.info(f"[Chinese text removed] {df_desc.shape[0]} [Chinese text removed]，[Chinese text removed] {df_desc.shape[1]} [Chinese text removed]")

    # [Chinese text removed]original[Chinese text removed]
    df_final = pd.concat([df, df_desc.drop('SMILES', axis=1)], axis=1)
    df_final.to_csv(logBB_desc_file, encoding='utf-8', index=False)
    log.info(f"[Chinese text removed]: {logBB_desc_file}")
    log.info(f"[Chinese text removed]: {df_final.shape}")

    # features[Chinese text removed]
    X_initial = df_final.drop(['SMILES', 'logBB'], axis=1)
    y = df_final['logBB']

    # [Chinese text removed]features
    log.info(f"[Chinese text removed]features[Chinese text removed]: {X_initial.shape[1]}")
    log.info(f"features[Chinese text removed]: {X_initial.columns.tolist()}")

    # [Chinese text removed]features[Chinese text removed]
    if os.path.exists(feature_index_path):
        log.info("features[Chinese text removed]，[Chinese text removed]features")
        desc_index = np.loadtxt(feature_index_path, dtype=int, delimiter=',')
        
        # [Chinese text removed]
        valid_indices = desc_index[desc_index < X_initial.shape[1]]
        if len(valid_indices) != len(desc_index):
            log.warning(f"[Chinese text removed]featuresIndex out of range，[Chinese text removed]")
            desc_index = valid_indices
        
        X = X_initial.iloc[:, desc_index]  # [Chinese text removed]
        log.info(f"[Chinese text removed]features[Chinese text removed]，[Chinese text removed]features[Chinese text removed]: {X.shape[1]}")
        log.info(f"[Chinese text removed]features[Chinese text removed]: {X.columns.tolist()}")
    else:
        log.error("features[Chinese text removed]，[Chinese text removed]features")
        return

    # Dataset split
    X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
    log.info(f"[Chinese text removed]9:1[Chinese text removed]({len(X_train_val)})[Chinese text removed]samples[Chinese text removed]({len(X_test)})[Chinese text removed]samples")

    # [Chinese text removed] (10%)
    X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.1, random_state=42)
    log.info(f"[Chinese text removed]9:1[Chinese text removed]({len(X_train)})[Chinese text removed]samples[Chinese text removed]({len(X_val)})[Chinese text removed]samples")

    # [Chinese text removed]Cross-validation
    kf = KFold(n_splits=10, shuffle=True, random_state=42)
    metrics_list = []

    for fold_num, (train_index, val_index) in enumerate(kf.split(X)):
        X_train_cv, X_val_cv = X.iloc[train_index], X.iloc[val_index]
        y_train_cv, y_val_cv = y.iloc[train_index], y.iloc[val_index]

        # [Chinese text removed]parameter[Chinese text removed]
        study = optuna.create_study(direction='minimize')
        study.optimize(lambda trial: objective(trial, X_train_cv, y_train_cv), n_trials=10)
        best_params = study.best_params

        # Training final model
        model_final = SVR(**best_params)
        model_final.fit(X_train_cv, y_train_cv)

        # [Chinese text removed]
        y_val_pred = model_final.predict(X_val_cv)

        # Calculate evaluation metrics
        mse_val = mean_squared_error(y_val_cv, y_val_pred)
        rmse_val = np.sqrt(mse_val)
        r2_val = r2_score(y_val_cv, y_val_pred)
        mae_val = mean_absolute_error(y_val_cv, y_val_pred)
        mape_val = mean_absolute_percentage_error(y_val_cv, y_val_pred)
        adj_r2_val = adjusted_r2(y_val_cv, y_val_pred, X_val_cv.shape[1])

        metrics_list.append({
            'fold': fold_num + 1,
            'mse': mse_val,
            'rmse': rmse_val,
            'r2': r2_val,
            'mae': mae_val,
            'mape': mape_val,
            'adj_r2': adj_r2_val
        })

        log.info(f"Fold {fold_num + 1} - MSE: {mse_val:.4f}, RMSE: {rmse_val:.4f}, R2: {r2_val:.4f}, Adjusted R2: {adj_r2_val:.4f}, MAE: {mae_val:.4f}, MAPE: {mape_val:.2f}%")

    # [Chinese text removed]
    avg_metrics = {key: np.mean([m[key] for m in metrics_list]) for key in metrics_list[0] if key != 'fold'}
    log.info("10[Chinese text removed]Cross-validation[Chinese text removed]：")
    log.info(f"MSE: {avg_metrics['mse']:.4f}")
    log.info(f"RMSE: {avg_metrics['rmse']:.4f}")
    log.info(f"R2: {avg_metrics['r2']:.4f}")
    log.info(f"Adjusted R2: {avg_metrics['adj_r2']:.4f}")
    log.info(f"MAE: {avg_metrics['mae']:.4f}")
    log.info(f"MAPE: {avg_metrics['mape']:.2f}%")

    # [Chinese text removed]
    y_test_pred = model_final.predict(X_test)
    mse_test = mean_squared_error(y_test, y_test_pred)
    rmse_test = np.sqrt(mse_test)
    r2_test = r2_score(y_test, y_test_pred)
    mae_test = mean_absolute_error(y_test, y_test_pred)
    mape_test = mean_absolute_percentage_error(y_test, y_test_pred)
    adj_r2_test = adjusted_r2(y_test, y_test_pred, X_test.shape[1])

    log.info("[Chinese text removed]：")
    log.info(f"MSE: {mse_test:.4f}, RMSE: {rmse_test:.4f}, R2: {r2_test:.4f}, Adjusted R2: {adj_r2_test:.4f}, MAE: {mae_test:.4f}, MAPE: {mape_test:.2f}%")

    # [Chinese text removed] CSV [Chinese text removed]
    train_results = pd.DataFrame({
        'True Values': y_train_val,
        'Predicted Values': model_final.predict(X_train_val)
    })
    test_results = pd.DataFrame({
        'True Values': y_test,
        'Predicted Values': y_test_pred
    })
    train_results.to_csv('./result/svm_fp_train_results.csv', index=False)
    test_results.to_csv('./result/svm_fp_test_results.csv', index=False)
    log.info("[Chinese text removed] './result/svm_fp_train_results.csv' [Chinese text removed] './result/svm_fp_test_results.csv'")

    # Save model
    model_save_path = os.path.join(model_dir, "svm_fp_model.joblib")
    joblib.dump(model_final, model_save_path)
    log.info(f"Model saved[Chinese text removed] {model_save_path}")


if __name__ == '__main__':
    main()

