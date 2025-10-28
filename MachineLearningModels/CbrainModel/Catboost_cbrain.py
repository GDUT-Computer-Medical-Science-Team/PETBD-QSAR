import numpy as np
import os
import time
import pandas as pd
import optuna
from catboost import CatBoostRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import joblib
import utils.datasets_loader as loader
from utils.DataLogger import DataLogger
from preprocess.MedicalDatasetsHandler import MedicalDatasetsHandler
from tqdm import tqdm

# [Chinese text removed]
cur_time = time.localtime()
result_parent_dir = f"../../result/{time.strftime('%Y%m%d', cur_time)}"
result_dir = f"{result_parent_dir}/{time.strftime('%H%M%S', cur_time)}"
os.makedirs(result_dir, exist_ok=True)
main_log = DataLogger(f"{result_dir}/run.log").getlog(disable_console_output=True)
log = DataLogger().getlog("run")

def check_datasets_exist(parent_folder: str):
    if os.path.exists(parent_folder):
        if not os.path.isdir(parent_folder):
            raise NotADirectoryError(f"Error：{parent_folder}[Chinese text removed]")
        return any(file.endswith("_dataset.pt") for file in os.listdir(parent_folder))
    return False

def check_data_exist(merge_filepath, organ_names_list, certain_time,
                     train_dir_path, test_dir_path,
                     FP=False, overwrite=False):
    try:
        flag = check_datasets_exist(train_dir_path) and check_datasets_exist(test_dir_path)
    except NotADirectoryError:
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

def calculate_mape(y_true, y_pred):
    epsilon = 1e-8  # [Chinese text removed]0
    return np.mean(np.abs((y_true - y_pred) / (y_true + epsilon))) * 100

def adjusted_r2_score(r2, n, k):
    """
    [Chinese text removed]R[Chinese text removed]
    :param r2: R[Chinese text removed]
    :param n: samples[Chinese text removed]
    :param k: [Chinese text removed]
    :return: [Chinese text removed]R[Chinese text removed]
    """
    return 1 - (1 - r2) * (n - 1) / (n - k - 1)

def objective(trial, X, y):
    params = {
        'iterations': trial.suggest_int('iterations', 100, 2000),
        'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.1),
        'depth': trial.suggest_int('depth', 4, 10),
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1e-5, 1e-1),
        'random_strength': trial.suggest_float('random_strength', 0, 2),
        'min_child_samples': trial.suggest_int('min_child_samples', 1, 20),
        'verbose': False,
        'random_seed': int(time.time())
    }

    cb = CatBoostRegressor(**params)
    cv = KFold(n_splits=5, shuffle=True)
    mse_scores = []

    for train_idx, val_idx in cv.split(X, y):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        cb.fit(X_train, y_train, eval_set=[(X_val, y_val)], early_stopping_rounds=100)
        preds = cb.predict(X_val)
        mse_scores.append(mean_squared_error(y_val, preds))

    return np.mean(mse_scores)

def train_catboost(organ_name):
    X, y = loader.get_sklearn_data('../../data/train/train_organ_df.npy', organ_name)
    X_test, y_test = loader.get_sklearn_data('../../data/test/test_organ_df.npy', organ_name)

    model_dir = "model"
    os.makedirs(model_dir, exist_ok=True)

    log.info("StartingCatBoost[Chinese text removed]parameter[Chinese text removed]")
    study = optuna.create_study(direction='minimize')
    study.optimize(lambda trial: objective(trial, X, y), n_trials=50)

    best_params = study.best_params
    log.info(f"[Chinese text removed]parameter: {best_params}")

    cb = CatBoostRegressor(**best_params)
    cv = KFold(n_splits=10, shuffle=True)

    fold_mse = []
    fold_r2 = []
    fold_adj_r2 = []
    fold_mae = []
    fold_mape = []
    fold_rmse = []

    for train_idx, val_idx in tqdm(cv.split(X, y), desc="Training CatBoost"):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        cb.fit(X_train, y_train, eval_set=[(X_val, y_val)], early_stopping_rounds=100)
        preds = cb.predict(X_val)
        mse = mean_squared_error(y_val, preds)
        r2 = r2_score(y_val, preds)
        adj_r2 = adjusted_r2_score(r2, X_val.shape[0], X_val.shape[1])
        mae = mean_absolute_error(y_val, preds)
        mape = calculate_mape(y_val, preds)
        rmse = np.sqrt(mse)

        fold_mse.append(mse)
        fold_r2.append(r2)
        fold_adj_r2.append(adj_r2)
        fold_mae.append(mae)
        fold_mape.append(mape)
        fold_rmse.append(rmse)

    avg_mse = np.mean(fold_mse)
    avg_r2 = np.mean(fold_r2)
    avg_adj_r2 = np.mean(fold_adj_r2)
    avg_mae = np.mean(fold_mae)
    avg_mape = np.mean(fold_mape)
    avg_rmse = np.mean(fold_rmse)

    log.info(f"[Chinese text removed]Cross-validation[Chinese text removed]MSE: {avg_mse}, [Chinese text removed]R2: {avg_r2}, [Chinese text removed]R2: {avg_adj_r2}, [Chinese text removed]MAE: {avg_mae}, [Chinese text removed]MAPE: {avg_mape}, [Chinese text removed]RMSE: {avg_rmse}")

    test_preds = cb.predict(X_test)
    test_mse = mean_squared_error(y_test, test_preds)
    test_r2 = r2_score(y_test, test_preds)
    test_adj_r2 = adjusted_r2_score(test_r2, X_test.shape[0], X_test.shape[1])
    test_mae = mean_absolute_error(y_test, test_preds)
    test_mape = calculate_mape(y_test, test_preds)
    test_rmse = np.sqrt(test_mse)

    log.info(f"[Chinese text removed]R2: {test_r2}, [Chinese text removed]R2: {test_adj_r2}, MSE: {test_mse}, MAE: {test_mae}, MAPE: {test_mape}%, RMSE: {test_rmse}")

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
    log.info(f"RMSE: {best_metrics['test_rmse']:.3f}")
    log.info(f"R2: {best_metrics['test_r2']:.3f}")
    log.info(f"[Chinese text removed]R2: {best_metrics['test_adj_r2']:.3f}")
    log.info(f"MAE: {best_metrics['test_mae']:.3f}")
    log.info(f"MAPE: {best_metrics['test_mape']:.3f}%")
    log.info(f"MSE: {best_metrics['test_mse']:.3f}")


    # Save model
    model_save_path = f"{model_dir}/catboost_model.cbm"
    cb.save_model(model_save_path)
    log.info(f"Model saved[Chinese text removed] {model_save_path}")

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
    FP = True
    if FP:
        log.info("[Chinese text removed]features[Chinese text removed]：[Chinese text removed]")
    else:
        log.info("[Chinese text removed]features[Chinese text removed]：[Chinese text removed]")
    check_data_exist(merge_filepath, organ_names_list, certain_time,
                     train_datasets_dir, test_datasets_dir,
                     FP=FP, overwrite=overwrite)
    train_catboost(organ_name)
