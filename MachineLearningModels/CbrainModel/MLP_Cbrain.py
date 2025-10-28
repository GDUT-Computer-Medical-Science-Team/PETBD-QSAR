import os
import time
import traceback
import optuna
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import KFold
from sklearn.neural_network import MLPRegressor
from tqdm import tqdm
import joblib
import utils.datasets_loader as loader
from preprocess.MedicalDatasetsHandler import MedicalDatasetsHandler
from utils.DataLogger import DataLogger

# [Chinese text removed]
cur_time = time.localtime()
result_parent_dir = f"../../result/{time.strftime('%Y%m%d', cur_time)}"
result_dir = f"{result_parent_dir}/{time.strftime('%H%M%S', cur_time)}"
os.makedirs(result_dir, exist_ok=True)

main_log = DataLogger(f"{result_dir}/run.log").getlog(disable_console_output=True)
log = DataLogger().getlog("run")


def check_datasets_exist(parent_folder: str):
    if not os.path.isdir(parent_folder):
        raise NotADirectoryError(f"Error：{parent_folder}[Chinese text removed]")
    return any(file.endswith("_dataset.pt") for file in os.listdir(parent_folder))


def check_data_exist(merge_filepath, organ_names_list, certain_time,
                     train_dir_path, test_dir_path, FP=False, overwrite=False):
    """
    [Chinese text removed]，[Chinese text removed]
    :return:
    """
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
        md.read_merged_datafile(merged_filepath=merge_filepath, organ_names=organ_names_list, certain_time=certain_time,
                                overwrite=overwrite)
        md.transform_organ_time_data_to_tensor_dataset(test_size=0.1, double_index=False, FP=FP, overwrite=overwrite)
        log.info(f"[Chinese text removed]Complete")


def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    nonzero_indices = y_true != 0
    return np.mean(np.abs((y_true[nonzero_indices] - y_pred[nonzero_indices]) / y_true[nonzero_indices])) * 100


def adjusted_r2_score(r2, n, k):
    """
    [Chinese text removed] R^2 [Chinese text removed]。
    r2: original R^2 [Chinese text removed]
    n: samples[Chinese text removed]
    k: features[Chinese text removed]
    """
    return 1 - (1 - r2) * ((n - 1) / (n - k - 1))


def objective(trial, X, y):
    params = {
        'hidden_layer_sizes': tuple(
            trial.suggest_int(f'n_units_l{i}', 32, 512) 
            for i in range(trial.suggest_int('n_layers', 1, 5))
        ),
        'activation': trial.suggest_categorical('activation', ['relu', 'tanh', 'logistic']),
        'solver': trial.suggest_categorical('solver', ['adam', 'sgd', 'lbfgs']),
        'alpha': trial.suggest_float('alpha', 1e-5, 1e-2, log=True),
        'learning_rate': trial.suggest_categorical('learning_rate', ['constant', 'invscaling', 'adaptive']),
        'learning_rate_init': trial.suggest_float('learning_rate_init', 1e-5, 1e-2, log=True),
        'max_iter': trial.suggest_int('max_iter', 200, 2000),
        'batch_size': trial.suggest_categorical('batch_size', ['auto', 32, 64, 128, 256]),
        'beta_1': trial.suggest_float('beta_1', 0.8, 0.999),
        'beta_2': trial.suggest_float('beta_2', 0.8, 0.999),
        'epsilon': trial.suggest_float('epsilon', 1e-8, 1e-7, log=True),
        'n_iter_no_change': trial.suggest_int('n_iter_no_change', 10, 50),
    }

    cv = KFold(n_splits=10, shuffle=True)
    mse_scores = []
    
    for train_idx, val_idx in tqdm(cv.split(X, y), desc="Training MLP"):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        mlp = MLPRegressor(**params)
        mlp.fit(X_train, y_train)
        preds = mlp.predict(X_val)
        mse_scores.append(mean_squared_error(y_val, preds))

    return np.mean(mse_scores)


if __name__ == '__main__':
    organ_name = 'brain'
    merge_filepath = "../../data/[Chinese text removed].xlsx"
    organ_names_list = ['blood', 'bone', 'brain', 'fat', 'heart', 'intestine', 'kidney', 'liver', 'lung', 'muscle',
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

    check_data_exist(merge_filepath, organ_names_list, certain_time, train_datasets_dir, test_datasets_dir, FP=FP,
                     overwrite=overwrite)

    # [Chinese text removed]
    X, y = loader.get_sklearn_data('../../data/train/train_organ_df.npy', organ_name)

    # Optuna [Chinese text removed]parameter[Chinese text removed]
    study = optuna.create_study(direction='minimize')
    study.optimize(lambda trial: objective(trial, X, y), n_trials=50)

    log.info(f"Bestparameter: {study.best_params}")
    log.info(f"BestMSE: {study.best_value}")

    # [Chinese text removed]Cross-validation[Chinese text removed]
    mse_scores, r2_scores, mae_scores, rmse_scores, mape_scores = [], [], [], [], []
    for trial in study.trials:
        mse_scores.append(trial.value)
        r2_scores.extend(trial.user_attrs['r2_scores'])
        mae_scores.extend(trial.user_attrs['mae_scores'])
        rmse_scores.extend(trial.user_attrs['rmse_scores'])
        mape_scores.extend(trial.user_attrs['mape_scores'])

    n = len(y)
    k = X.shape[1]
    avg_r2 = np.mean(r2_scores)
    avg_adjusted_r2 = adjusted_r2_score(avg_r2, n, k)

    log.info("\n========[Chinese text removed]Cross-validation Results========")
    log.info(f"[Chinese text removed]MSE: {np.mean(mse_scores):.3f} (±{np.std(mse_scores):.3f})")
    log.info(f"[Chinese text removed]R2: {avg_r2:.3f} (±{np.std(r2_scores):.3f})")
    log.info(f"[Chinese text removed]R2: {avg_adjusted_r2:.3f} (±{np.std(r2_scores):.3f})")
    log.info(f"[Chinese text removed]MAE: {np.mean(mae_scores):.3f} (±{np.std(mae_scores):.3f})")
    log.info(f"[Chinese text removed]RMSE: {np.mean(rmse_scores):.3f} (±{np.std(rmse_scores):.3f})")
    log.info(f"[Chinese text removed]MAPE: {np.mean(mape_scores):.3f}% (±{np.std(mape_scores):.3f}%)")

    # [Chinese text removed]trials[Chinese text removed]
    best_trial = study.best_trial
    best_model = best_trial.user_attrs['best_model']

    # [Chinese text removed]
    X_test, y_test = loader.get_sklearn_data('../../data/test/test_organ_df.npy', organ_name)
    preds = best_model.predict(X_test)

    test_mse = mean_squared_error(y_test, preds)
    test_r2 = r2_score(y_test, preds)
    test_adjusted_r2 = adjusted_r2_score(test_r2, len(y_test), X_test.shape[1])
    test_mae = mean_absolute_error(y_test, preds)
    test_rmse = np.sqrt(test_mse)
    test_mape = mean_absolute_percentage_error(y_test, preds)

    log.info("\n========[Chinese text removed]========")
    log.info(f"[Chinese text removed]RMSE: {test_rmse:.3f}")
    log.info(f"[Chinese text removed]R2: {test_r2:.3f}")
    log.info(f"[Chinese text removed]R2: {test_adjusted_r2:.3f}")
    log.info(f"[Chinese text removed]MAE: {test_mae:.3f}")
    log.info(f"[Chinese text removed]MAPE: {test_mape:.3f}%")
    log.info(f"[Chinese text removed]MSE: {test_mse:.3f}")

    # [Chinese text removed]Test Set Results
    print("\nTest Set Results:")
    print(f"R2[Chinese text removed]: {test_r2:.3f}")
    print(f"[Chinese text removed]R2[Chinese text removed]: {test_adjusted_r2:.3f}")
    print(f"MSE[Chinese text removed]: {test_mse:.3f}")
    print(f"MAE[Chinese text removed]: {test_mae:.3f}")
    print(f"MAPE[Chinese text removed]: {test_mape:.3f}%")
    print(f"RMSE[Chinese text removed]: {test_rmse:.3f}")

    # [Chinese text removed]
    best_metrics = {
        'test_rmse': test_rmse,
        'test_r2': test_r2,
        'test_adjusted_r2': test_adjusted_r2,
        'test_mae': test_mae,
        'test_mape': test_mape,
        'test_mse': test_mse
    }

    # [Chinese text removed]
    os.makedirs("model", exist_ok=True)
    model_save_path = "model/mlp_mo_model.joblib"
    joblib.dump(best_model, model_save_path)
    log.info(f"Model saved[Chinese text removed] {model_save_path}")
