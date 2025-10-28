import optuna
import lightgbm as lgb
import datetime
import os
import time
import traceback
import pandas as pd
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from sklearn.model_selection import KFold
import joblib
import utils.datasets_loader as loader
from preprocess.MedicalDatasetsHandler import MedicalDatasetsHandler
from utils.DataLogger import DataLogger
from tqdm import tqdm

# [Chinese text removed]
cur_time = time.localtime()
result_parent_dir = f"../../result/{time.strftime('%Y%m%d', cur_time)}"
result_dir = f"{result_parent_dir}/{time.strftime('%H%M%S', cur_time)}"
os.makedirs(result_dir, exist_ok=True)
main_log = DataLogger(f"{result_dir}/run.log").getlog(disable_console_output=True)
log = DataLogger().getlog("run")


def read_merged_datafile(self,
                         merged_filepath,
                         organ_names: list,
                         certain_time: int,
                         overwrite=False,
                         is_sd=False):
    """
    [Chinese text removed]original[Chinese text removed]，[Chinese text removed]
    :param merged_filepath: [Chinese text removed]
    :param organ_names: [Chinese text removed]
    :param certain_time: [Chinese text removed]，[Chinese text removed]
    :param overwrite: [Chinese text removed]
    :param is_sd: [Chinese text removed](sd)，[Chinese text removed]False，[Chinese text removed](mean)
    """
    self.__merged_filepath = merged_filepath
    # [Chinese text removed]original[Chinese text removed]
    if self.__merged_filepath is not None and len(self.__merged_filepath) > 0:
        self.__saving_folder = os.path.split(self.__merged_filepath)[0]
    else:
        raise ValueError("parametermerged_filepathError")

    organ_data_filepath = os.path.join(self.__saving_folder, "OrganData.csv")
    self.__organ_time_data_filepath = os.path.join(self.__saving_folder, f"OrganDataAt{certain_time}min.csv")
    log.info("[Chinese text removed]original[Chinese text removed]，[Chinese text removed]")

    # [Chinese text removed]
    if overwrite or not os.path.exists(organ_data_filepath):
        log.info(f"[Chinese text removed]: {organ_names} [Chinese text removed]: ")
        save_organ_data_by_names(root_filepath=self.__merged_filepath,
                                 target_filepath=organ_data_filepath,
                                 organ_names=organ_names,
                                 is_sd=is_sd)
    else:
        log.info(f"[Chinese text removed]Complete[Chinese text removed]csv[Chinese text removed]: {organ_data_filepath}，[Chinese text removed]")
    # [Chinese text removed]
    if overwrite or not os.path.exists(self.__organ_time_data_filepath):
        df = pd.DataFrame()
        log.info(f"[Chinese text removed]，[Chinese text removed]: {certain_time}min")
        # [Chinese text removed]
        for organ_name in tqdm(organ_names, desc=f"[Chinese text removed]{certain_time}min[Chinese text removed]: "):
            df = pd.concat([df, get_certain_time_organ_data(root_filepath=organ_data_filepath,
                                                            organ_name=organ_name,
                                                            certain_time=certain_time,
                                                            is_sd=is_sd)],
                           axis=1)
        log.info(f"[Chinese text removed]Complete，[Chinese text removed]csv[Chinese text removed]{self.__organ_time_data_filepath}")
        df.to_csv(self.__organ_time_data_filepath, encoding='utf-8')
    else:
        log.info(f"[Chinese text removed]Complete[Chinese text removed]csv[Chinese text removed]: {self.__organ_time_data_filepath}，[Chinese text removed]")


def transform_organ_time_data_to_tensor_dataset(self,
                                                test_size=0.2,
                                                double_index=False,
                                                FP=False,
                                                overwrite=False):
    """
    [Chinese text removed]csv[Chinese text removed]，[Chinese text removed]，[Chinese text removed]、[Chinese text removed]，[Chinese text removed]TensorDataset[Chinese text removed]
    :param test_size: [Chinese text removed]，[Chinese text removed][0.0, 1.0)
    :param double_index: [Chinese text removed]features[Chinese text removed]
    :param FP: [Chinese text removed]
    :param overwrite: [Chinese text removed]npy[Chinese text removed]
    """
    if test_size < 0.0 or test_size >= 1.0:
        raise ValueError("parametertest_size[Chinese text removed][0.0, 1.0)")
    npy_file_path = self.__transform_organs_data(FP=FP,
                                                 double_index=double_index,
                                                 overwrite=overwrite)
    self.__split_df2TensorDataset(npy_file_path, test_size=test_size)


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


def adjusted_r2_score(r2, n, k):
    """
    [Chinese text removed] R^2 [Chinese text removed]。
    r2: original R^2 [Chinese text removed]
    n: samples[Chinese text removed]
    k: features[Chinese text removed]
    """
    return 1 - (1 - r2) * ((n - 1) / (n - k - 1))


def objective(trial, X, y):
    param = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 3000),
        'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.3, log=True),
        'num_leaves': trial.suggest_int('num_leaves', 20, 300),
        'max_depth': trial.suggest_int('max_depth', 3, 12),
        'min_child_samples': trial.suggest_int('min_child_samples', 1, 100),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
        'min_split_gain': trial.suggest_float('min_split_gain', 1e-8, 1.0, log=True),
        'min_child_weight': trial.suggest_float('min_child_weight', 1e-8, 1.0, log=True),
    }

    cv = KFold(n_splits=10, shuffle=True, random_state=42)
    cv_scores = {
        'rmse': [],
        'r2': [],
        'mae': [],
        'mape': [],
        'adjusted_r2': []
    }

    for fold, (train_idx, val_idx) in enumerate(tqdm(cv.split(X), desc="Cross-validation")):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        model = lgb.LGBMRegressor(**param)
        model.fit(X_train, y_train)
        preds = model.predict(X_val)

        r2 = r2_score(y_val, preds)
        rmse = np.sqrt(mean_squared_error(y_val, preds))
        mae = mean_absolute_error(y_val, preds)
        mape = mean_absolute_percentage_error(y_val, preds)
        adj_r2 = adjusted_r2_score(r2, len(y_val), X_val.shape[1])

        cv_scores['rmse'].append(rmse)
        cv_scores['r2'].append(r2)
        cv_scores['mae'].append(mae)
        cv_scores['mape'].append(mape)
        cv_scores['adjusted_r2'].append(adj_r2)

    # [Chinese text removed]Cross-validation Results，[Chinese text removed]
    trial.set_user_attr('cv_scores', cv_scores)

    return np.mean(cv_scores['rmse'])


def train_lightgbm(organ_name, FP=False):
    model_type = "Fingerprint" if FP else "Mordred Descriptor"
    log.info(f"Training LightGBM model with {model_type} features")

    X, y = loader.get_sklearn_data('../../data/train/train_organ_df.npy', organ_name)
    X_test, y_test = loader.get_sklearn_data('../../data/test/test_organ_df.npy', organ_name)

    # [Chinese text removed] X [Chinese text removed] y Converting[Chinese text removed] NumPy [Chinese text removed]Error
    X = np.array(X)
    y = np.array(y)

    study = optuna.create_study(direction='minimize')
    study.optimize(lambda trial: objective(trial, X, y), n_trials=50)

    best_params = study.best_params
    best_gbm = lgb.LGBMRegressor(**best_params)
    best_gbm.fit(X, y)

    # [Chinese text removed]trial[Chinese text removed]Cross-validation Results
    best_cv_scores = study.best_trial.user_attrs['cv_scores']

    # [Chinese text removed]
    preds = best_gbm.predict(X_test)
    test_r2 = r2_score(y_test, preds)
    test_mse = mean_squared_error(y_test, preds)
    test_mae = mean_absolute_error(y_test, preds)
    test_mape = mean_absolute_percentage_error(y_test, preds)
    test_rmse = np.sqrt(test_mse)
    test_adjusted_r2 = adjusted_r2_score(test_r2, len(y_test), X_test.shape[1])

    # [Chinese text removed]
    log.info(f"\n========{model_type} [Chinese text removed]========")
    
    # [Chinese text removed]Cross-validation Results
    log.info("\n[Chinese text removed]Cross-validation Results:")
    log.info(f"[Chinese text removed]RMSE: {np.mean(best_cv_scores['rmse']):.3f} (±{np.std(best_cv_scores['rmse']):.3f})")
    log.info(f"[Chinese text removed]R2: {np.mean(best_cv_scores['r2']):.3f} (±{np.std(best_cv_scores['r2']):.3f})")
    log.info(f"[Chinese text removed]R2: {np.mean(best_cv_scores['adjusted_r2']):.3f} (±{np.std(best_cv_scores['adjusted_r2']):.3f})")
    log.info(f"[Chinese text removed]MAE: {np.mean(best_cv_scores['mae']):.3f} (±{np.std(best_cv_scores['mae']):.3f})")
    log.info(f"[Chinese text removed]MAPE: {np.mean(best_cv_scores['mape']):.3f}% (±{np.std(best_cv_scores['mape']):.3f}%)")

    # [Chinese text removed]Test Set Results
    log.info("\nTest Set Results:")
    log.info(f"RMSE: {test_rmse:.3f}")
    log.info(f"R2: {test_r2:.3f}")
    log.info(f"[Chinese text removed]R2: {test_adjusted_r2:.3f}")
    log.info(f"MAE: {test_mae:.3f}")
    log.info(f"MAPE: {test_mape:.3f}%")
    log.info(f"MSE: {test_mse:.3f}")

    # Save model
    model_dir = "cbrainModel"
    os.makedirs(model_dir, exist_ok=True)
    model_save_path = f"{model_dir}/lightgbm_{model_type}_model.joblib"
    joblib.dump(best_gbm, model_save_path)
    log.info(f"\nModel saved[Chinese text removed] {model_save_path}")

    return {
        'model': best_gbm,
        'cv_metrics': {
            'rmse': np.mean(best_cv_scores['rmse']),
            'r2': np.mean(best_cv_scores['r2']),
            'adjusted_r2': np.mean(best_cv_scores['adjusted_r2']),
            'mae': np.mean(best_cv_scores['mae']),
            'mape': np.mean(best_cv_scores['mape'])
        },
        'test_metrics': {
            'rmse': test_rmse,
            'r2': test_r2,
            'adjusted_r2': test_adjusted_r2,
            'mae': test_mae,
            'mape': test_mape,
            'mse': test_mse
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

    # [Chinese text removed]
    model_dir = "model"
    os.makedirs(model_dir, exist_ok=True)
    overwrite = False
    
    # [Chinese text removed]
    results = {}

    # [Chinese text removed]
    FP = True
    check_data_exist(merge_filepath, organ_names_list, certain_time,
                     train_datasets_dir, test_datasets_dir,
                     FP=FP, overwrite=overwrite)
    results['fingerprint'] = train_lightgbm(organ_name, FP=FP)

    # [Chinese text removed]
    FP = False
    check_data_exist(merge_filepath, organ_names_list, certain_time,
                     train_datasets_dir, test_datasets_dir,
                     FP=FP, overwrite=overwrite)
    results['mordred'] = train_lightgbm(organ_name, FP=FP)

    # [Chinese text removed]
    log.info("\n========= LightGBM [Chinese text removed] =========")
    
    # [Chinese text removed]
    log.info("\n【[Chinese text removed]】")
    log.info("[Chinese text removed]Cross-validation Results:")
    cv_metrics = results['fingerprint']['cv_metrics']
    log.info(f"[Chinese text removed]RMSE: {cv_metrics['rmse']:.3f}")
    log.info(f"[Chinese text removed]R2: {cv_metrics['r2']:.3f}")
    log.info(f"[Chinese text removed]R2: {cv_metrics['adjusted_r2']:.3f}")
    log.info(f"[Chinese text removed]MAE: {cv_metrics['mae']:.3f}")
    log.info(f"[Chinese text removed]MAPE: {cv_metrics['mape']:.3f}%")
    
    log.info("\nTest Set Results:")
    test_metrics = results['fingerprint']['test_metrics']
    log.info(f"RMSE: {test_metrics['rmse']:.3f}")
    log.info(f"R2: {test_metrics['r2']:.3f}")
    log.info(f"[Chinese text removed]R2: {test_metrics['adjusted_r2']:.3f}")
    log.info(f"MAE: {test_metrics['mae']:.3f}")
    log.info(f"MAPE: {test_metrics['mape']:.3f}%")
    log.info(f"MSE: {test_metrics['mse']:.3f}")

    # [Chinese text removed]
    log.info("\n【[Chinese text removed]】")
    log.info("[Chinese text removed]Cross-validation Results:")
    cv_metrics = results['mordred']['cv_metrics']
    log.info(f"[Chinese text removed]RMSE: {cv_metrics['rmse']:.3f}")
    log.info(f"[Chinese text removed]R2: {cv_metrics['r2']:.3f}")
    log.info(f"[Chinese text removed]R2: {cv_metrics['adjusted_r2']:.3f}")
    log.info(f"[Chinese text removed]MAE: {cv_metrics['mae']:.3f}")
    log.info(f"[Chinese text removed]MAPE: {cv_metrics['mape']:.3f}%")
    
    log.info("\nTest Set Results:")
    test_metrics = results['mordred']['test_metrics']
    log.info(f"RMSE: {test_metrics['rmse']:.3f}")
    log.info(f"R2: {test_metrics['r2']:.3f}")
    log.info(f"[Chinese text removed]R2: {test_metrics['adjusted_r2']:.3f}")
    log.info(f"MAE: {test_metrics['mae']:.3f}")
    log.info(f"MAPE: {test_metrics['mape']:.3f}%")
    log.info(f"MSE: {test_metrics['mse']:.3f}")
