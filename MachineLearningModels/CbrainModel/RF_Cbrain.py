import datetime
import os
import time
import traceback
import pandas as pd
import numpy as np
import torch
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from sklearn.model_selection import KFold
from tqdm import tqdm
from xgboost.sklearn import XGBRegressor
import utils.datasets_loader as loader

from preprocess.MedicalDatasetsHandler import MedicalDatasetsHandler
from utils.DataLogger import DataLogger
from sklearn.ensemble import RandomForestRegressor
import optuna
import joblib
import matplotlib.pyplot as plt

# [Chinese text removed]
cur_time = time.localtime()
result_parent_dir = f"../../result/{time.strftime('%Y%m%d', cur_time)}"
result_dir = f"{result_parent_dir}/{time.strftime('%H%M%S', cur_time)}"
os.makedirs(result_dir, exist_ok=True)

main_log = DataLogger(f"{result_dir}/run.log").getlog(disable_console_output=True)
log = DataLogger().getlog("run")

def check_datasets_exist(parent_folder: str):
    flag = False
    if os.path.exists(parent_folder):
        if not os.path.isdir(parent_folder):
            raise NotADirectoryError(f"Error：{parent_folder}[Chinese text removed]")
        files = os.listdir(parent_folder)
        for file in files:
            if file.endswith("_dataset.pt"):
                flag = True
                break
    return flag

def check_data_exist(merge_filepath, organ_names_list, certain_time,
                     train_dir_path, test_dir_path,
                     FP=False, overwrite=False):
    """
    [Chinese text removed]，[Chinese text removed]
    :return:
    """
    try:
        flag = check_datasets_exist(train_dir_path) and check_datasets_exist(test_dir_path)
    except NotADirectoryError as e:
        log.error(traceback.format_exc())
        flag = False

    if not overwrite and flag:
        log.info(f"[Chinese text removed]TensorDatasets[Chinese text removed]，[Chinese text removed]")
    else:
        log.info(f"[Chinese text removed]TensorDatasets[Chinese text removed]，Starting[Chinese text removed]")
        if not os.path.exists(merge_filepath):
            raise FileNotFoundError(f"[Chinese text removed]\"{merge_filepath}\"[Chinese text removed]")
        md = MedicalDatasetsHandler()
        md.read_merged_datafile(merged_filepath=merge_filepath,
                                organ_names=organ_names_list,
                                certain_time=certain_time,
                                overwrite=overwrite)
        md.transform_organ_time_data_to_tensor_dataset(test_size=0.1,
                                                       double_index=False,
                                                       FP=FP,
                                                       overwrite=overwrite)
        log.info(f"[Chinese text removed]Complete")

def objective(trial, X, y):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
        'max_depth': trial.suggest_int('max_depth', 3, 30),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
        'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
        'bootstrap': trial.suggest_categorical('bootstrap', [True, False]),
        'max_leaf_nodes': trial.suggest_int('max_leaf_nodes', 10, 1000),
        'min_weight_fraction_leaf': trial.suggest_float('min_weight_fraction_leaf', 0.0, 0.5),
    }

    # [Chinese text removed] bootstrap [Chinese text removed] True [Chinese text removed] max_samples
    if params['bootstrap']:
        params['max_samples'] = trial.suggest_float('max_samples', 0.5, 1.0)
    else:
        params['max_samples'] = None  # [Chinese text removed] max_samples

    rf = RandomForestRegressor(**params)
    cv = KFold(n_splits=10, shuffle=True)
    mse_scores = []

    for train_idx, val_idx in tqdm(cv.split(X), desc="Training Random Forest"):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        rf.fit(X_train, y_train)
        preds = rf.predict(X_val)
        mse_scores.append(mean_squared_error(y_val, preds))

    return np.mean(mse_scores)

def adjusted_r2_score(r2, n, k):
    """
    [Chinese text removed]R[Chinese text removed]
    :param r2: R[Chinese text removed]
    :param n: samples[Chinese text removed]
    :param k: [Chinese text removed]
    :return: [Chinese text removed]R[Chinese text removed]
    """
    return 1 - (1 - r2) * (n - 1) / (n - k - 1)

def train_random_forest(organ_name):
    X, y = loader.get_sklearn_data('../../data/train/train_organ_df.npy', organ_name)
    X_test, y_test = loader.get_sklearn_data('../../data/test/test_organ_df.npy', organ_name)

    model_dir = "model"
    os.makedirs(model_dir, exist_ok=True)

    log.info("Starting[Chinese text removed]parameter[Chinese text removed]")
    study = optuna.create_study(direction='minimize')
    study.optimize(lambda trial: objective(trial, X, y), n_trials=50)

    best_params = study.best_params
    log.info(f"[Chinese text removed]parameter: {best_params}")

    rf = RandomForestRegressor(**best_params)
    r2_list = []
    adj_r2_list = []
    mse_list = []
    mae_list = []
    mape_list = []
    rmse_list = []

    log.info("[Chinese text removed]")
    cv = KFold(n_splits=10, shuffle=True, random_state=42)
    y_train_all, y_train_pred_all = [], []
    y_val_all, y_val_pred_all = [], []

    for train_idx, val_idx in tqdm(cv.split(X), desc="Cross-validation"):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        rf.fit(X_train, y_train)
        preds = rf.predict(X_val)

        y_train_all.extend(y_train)
        y_train_pred_all.extend(rf.predict(X_train))
        y_val_all.extend(y_val)
        y_val_pred_all.extend(preds)

        r2 = r2_score(y_val, preds)
        adj_r2 = adjusted_r2_score(r2, X_val.shape[0], X_val.shape[1])

        r2_list.append(r2)
        adj_r2_list.append(adj_r2)
        mse_list.append(mean_squared_error(y_val, preds))
        mae_list.append(mean_absolute_error(y_val, preds))
        mape_list.append(mean_absolute_percentage_error(y_val, preds))
        rmse_list.append(np.sqrt(mean_squared_error(y_val, preds)))

    log.info(f"Cross-validation R2: {np.mean(r2_list)}, [Chinese text removed]R2: {np.mean(adj_r2_list)}, MSE: {np.mean(mse_list)}, MAE: {np.mean(mae_list)}, MAPE: {np.mean(mape_list)}, RMSE: {np.mean(rmse_list)}")

    # [Chinese text removed]Cross-validation Results
    log.info("\n========[Chinese text removed]Cross-validation Results========")
    log.info(f"[Chinese text removed]R2: {np.mean(r2_list):.3f} (±{np.std(r2_list):.3f})")
    log.info(f"[Chinese text removed]R2: {np.mean(adj_r2_list):.3f} (±{np.std(adj_r2_list):.3f})")
    log.info(f"[Chinese text removed]MSE: {np.mean(mse_list):.3f} (±{np.std(mse_list):.3f})")
    log.info(f"[Chinese text removed]MAE: {np.mean(mae_list):.3f} (±{np.std(mae_list):.3f})")
    log.info(f"[Chinese text removed]MAPE: {np.mean(mape_list):.3f}% (±{np.std(mape_list):.3f}%)")
    log.info(f"[Chinese text removed]RMSE: {np.mean(rmse_list):.3f} (±{np.std(rmse_list):.3f})")

    rf.fit(X, y)
    test_preds = rf.predict(X_test)
    test_r2 = r2_score(y_test, test_preds)
    test_adj_r2 = adjusted_r2_score(test_r2, X_test.shape[0], X_test.shape[1])
    test_mse = mean_squared_error(y_test, test_preds)
    test_mae = mean_absolute_error(y_test, test_preds)
    test_mape = mean_absolute_percentage_error(y_test, test_preds)
    test_rmse = np.sqrt(mean_squared_error(y_test, test_preds))

    log.info(f"[Chinese text removed] R2: {test_r2}, [Chinese text removed]R2: {test_adj_r2}, MSE: {test_mse}, MAE: {test_mae}, MAPE: {test_mape}, RMSE: {test_rmse}")

    # Save model
    model_save_path = f"{model_dir}/rf_model.joblib"
    joblib.dump(rf, model_save_path)
    log.info(f"Model saved[Chinese text removed] {model_save_path}")

    # [Chinese text removed]
    train_results = pd.DataFrame({
        'Actual': y,
        'Predicted': rf.predict(X)
    })
    train_results.to_csv(f'{result_dir}/rf_train_predictions.csv', index=False)

    # [Chinese text removed]
    test_results = pd.DataFrame({
        'Actual': y_test,
        'Predicted': test_preds
    })
    test_results.to_csv(f'{result_dir}/rf_test_predictions.csv', index=False)

    # [Chinese text removed]
    fig, ax = plt.subplots(1, 2, figsize=(16, 6))

    ax[0].scatter(y_train_all, y_train_pred_all, edgecolors=(0, 0, 0), color='blue')
    ax[0].plot([min(y_train_all), max(y_train_all)], [min(y_train_all), max(y_train_all)], 'k--', lw=4)
    ax[0].set_xlabel('True')
    ax[0].set_ylabel('Predict')
    ax[0].set_title('cbrain rf Training Set: True vs Predict')

    ax[1].scatter(y_val_all, y_val_pred_all, edgecolors=(0, 0, 0), color='green')
    ax[1].plot([min(y_val_all), max(y_val_all)], [min(y_val_all), max(y_val_all)], 'k--', lw=4)
    ax[1].set_xlabel('True')
    ax[1].set_ylabel('Predict')
    ax[1].set_title('cbrain rf Validation Set: True vs Predict')

    plt.show()

    # [Chinese text removed]
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, test_preds, color='green', label='Test Data')
    plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'k--', lw=4)
    plt.title('Random Forest Regression: True vs. Predicted (Test Data)')
    plt.xlabel('True')
    plt.ylabel('Predict')
    plt.legend(loc='upper left')
    plt.grid(True)
    plt.show()

    # [Chinese text removed]
    best_metrics = {
        'test_rmse': test_rmse,
        'test_r2': test_r2,
        'test_adj_r2': test_adj_r2,
        'test_mae': test_mae,
        'test_mape': test_mape,
        'test_mse': test_mse
    }

    log.info("\n========[Chinese text removed]========")
    log.info(f"[Chinese text removed]RMSE: {test_rmse:.3f}")
    log.info(f"[Chinese text removed]R2: {test_r2:.3f}")
    log.info(f"[Chinese text removed]R2: {test_adj_r2:.3f}")
    log.info(f"[Chinese text removed]MAE: {test_mae:.3f}")
    log.info(f"[Chinese text removed]MAPE: {test_mape:.3f}%")
    log.info(f"[Chinese text removed]MSE: {test_mse:.3f}")

    # [Chinese text removed]Test Set Results
    print("\nTest Set Results:")
    print(f"R2[Chinese text removed]: {test_r2:.3f}")
    print(f"[Chinese text removed]R2[Chinese text removed]: {test_adj_r2:.3f}")
    print(f"MSE[Chinese text removed]: {test_mse:.3f}")
    print(f"MAE[Chinese text removed]: {test_mae:.3f}")
    print(f"MAPE[Chinese text removed]: {test_mape:.3f}%")
    print(f"RMSE[Chinese text removed]: {test_rmse:.3f}")

    return {
        'model': rf,
        'cv_metrics': {
            'r2': np.mean(r2_list),
            'adj_r2': np.mean(adj_r2_list),
            'mse': np.mean(mse_list),
            'mae': np.mean(mae_list),
            'mape': np.mean(mape_list),
            'rmse': np.mean(rmse_list)
        },
        'test_metrics': {
            'r2': test_r2,
            'adj_r2': test_adj_r2,
            'mse': test_mse,
            'mae': test_mae,
            'mape': test_mape,
            'rmse': test_rmse
        }
    }

if __name__ == '__main__':
    organ_name = 'brain'
    merge_filepath = "../../data/[Chinese text removed].xlsx"
    organ_names_list = ['blood', 'bone', 'brain', 'fat', 'heart',
                        'intestine', 'kidney', 'liver', 'lung', 'muscle',
                        'pancreas', 'spleen', 'stomach', 'uterus']
    certain_time = 60
    train_datasets_dir = "../../data/train/datasets"
    test_datasets_dir = "../../data/test/datasets"
    overwrite = False
    FP = False
    if FP:
        log.info("[Chinese text removed]features[Chinese text removed]：[Chinese text removed]")
    else:
        log.info("[Chinese text removed]features[Chinese text removed]：[Chinese text removed]")
    check_data_exist(merge_filepath, organ_names_list, certain_time,
                     train_datasets_dir, test_datasets_dir,
                     FP=FP, overwrite=overwrite)

    results = train_random_forest(organ_name)

