import lightgbm as lgb
import datetime
import os
import time
import traceback
import pandas as pd
import math
import numpy as np
import torch
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from sklearn.model_selection import KFold
from tqdm import tqdm
from xgboost import XGBRegressor
import utils.datasets_loader as loader
from preprocess.MedicalDatasetsHandler import MedicalDatasetsHandler
from utils.DataLogger import DataLogger
import optuna
from optuna import Trial
from optuna.samplers import TPESampler
from sklearn.svm import SVR
import joblib

# 确保定义 adjusted_r2_score 函数
def adjusted_r2_score(r2, n, k):
    """
    计算调整后的 R^2 值。
    r2: 原始 R^2 值
    n: 样本数量
    k: 特征数量
    """
    return 1 - (1 - r2) * ((n - 1) / (n - k - 1))

# 初始化结果保存目录
cur_time = time.localtime()
result_parent_dir = f"result\\{time.strftime('%Y%m%d', cur_time)}"
result_dir = f"{result_parent_dir}\\{time.strftime('%H%M%S', cur_time)}"
if not os.path.exists(result_parent_dir):
    os.mkdir(result_parent_dir)
if not os.path.exists(result_dir):
    os.mkdir(result_dir)
main_log = DataLogger(f"{result_dir}\\run.log").getlog(disable_console_output=True)
log = DataLogger().getlog("run")

def check_datasets_exist(parent_folder: str):
    flag = False
    if os.path.exists(parent_folder):
        if not os.path.isdir(parent_folder):
            raise NotADirectoryError(f"错误：{parent_folder}不是目录")
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
    检查是否有数据，无数据则重新生成数据
    :return:
    """
    try:
        flag = check_datasets_exist(train_dir_path) and check_datasets_exist(test_dir_path)
    except NotADirectoryError as e:
        log.error(traceback.format_exc())
        flag = False

    if not overwrite and flag:
        log.info(f"存在TensorDatasets数据，无须进行数据获取操作")
    else:
        log.info(f"不存在TensorDatasets数据，开始进行数据获取操作")
        if not os.path.exists(merge_filepath):
            raise FileNotFoundError(f"数据表文件\"{merge_filepath}\"未找到")
        md = MedicalDatasetsHandler()

        md.read_merged_datafile(merged_filepath=merge_filepath,
                                organ_names=organ_names_list,
                                certain_time=certain_time,
                                overwrite=overwrite)
        md.transform_organ_time_data_to_tensor_dataset(test_size=0.1,
                                                       double_index=False,
                                                       FP=FP,
                                                       overwrite=overwrite)
        log.info(f"数据获取完成")

def objective(trial, X, y):
    params = {
        'kernel': trial.suggest_categorical('kernel', ['rbf', 'linear', 'poly', 'sigmoid']),
        'C': trial.suggest_float('C', 1e-3, 1e3, log=True),
        'gamma': trial.suggest_categorical('gamma', ['scale', 'auto']),
        'epsilon': trial.suggest_float('epsilon', 1e-3, 1.0, log=True),
        'shrinking': trial.suggest_categorical('shrinking', [True, False]),
        'tol': trial.suggest_float('tol', 1e-5, 1e-3, log=True),
        'cache_size': trial.suggest_categorical('cache_size', [100, 200, 500, 1000]),
    }
    
    if params['kernel'] == 'poly':
        params['degree'] = trial.suggest_int('degree', 2, 5)
        params['coef0'] = trial.suggest_float('coef0', -1.0, 1.0)
    elif params['kernel'] == 'sigmoid':
        params['coef0'] = trial.suggest_float('coef0', -1.0, 1.0)

    svm = SVR(**params)

    cv = KFold(n_splits=10, shuffle=True)
    r2_scores = []

    for train_idx, val_idx in tqdm(cv.split(X), desc="Cross-validation"):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        svm.fit(X_train, y_train)
        preds = svm.predict(X_val)
        r2_scores.append(r2_score(y_val, preds))

    return np.mean(r2_scores)

def train_svm(organ_name):
    # 加载数据
    X, y = loader.get_sklearn_data('../../data/train/train_organ_df.npy', organ_name)
    X_test, y_test = loader.get_sklearn_data('../../data/test/test_organ_df.npy', organ_name)

    # Optuna 超参数优化
    study = optuna.create_study(direction='maximize', sampler=TPESampler())
    study.optimize(lambda trial: objective(trial, X, y), n_trials=50)

    best_params = study.best_params
    best_svm = SVR(**best_params)

    # 进行十折交叉验证
    cv = KFold(n_splits=10, shuffle=True)
    cv_scores = {
        'r2': [],
        'adjusted_r2': [],
        'mse': [],
        'mae': [],
        'mape': [],
        'rmse': []
    }

    for train_idx, val_idx in tqdm(cv.split(X), desc="十折交叉验证"):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        best_svm.fit(X_train, y_train)
        preds = best_svm.predict(X_val)

        r2 = r2_score(y_val, preds)
        mse = mean_squared_error(y_val, preds)
        mae = mean_absolute_error(y_val, preds)
        mape = mean_absolute_percentage_error(y_val, preds)
        rmse = np.sqrt(mse)
        adj_r2 = adjusted_r2_score(r2, len(y_val), X_val.shape[1])

        cv_scores['r2'].append(r2)
        cv_scores['adjusted_r2'].append(adj_r2)
        cv_scores['mse'].append(mse)
        cv_scores['mae'].append(mae)
        cv_scores['mape'].append(mape)
        cv_scores['rmse'].append(rmse)

    # 打印十折交叉验证结果
    log.info("\n========十折交叉验证结果========")
    log.info(f"平均R2: {np.mean(cv_scores['r2']):.3f} (±{np.std(cv_scores['r2']):.3f})")
    log.info(f"平均调整后R2: {np.mean(cv_scores['adjusted_r2']):.3f} (±{np.std(cv_scores['adjusted_r2']):.3f})")
    log.info(f"平均MSE: {np.mean(cv_scores['mse']):.3f} (±{np.std(cv_scores['mse']):.3f})")
    log.info(f"平均MAE: {np.mean(cv_scores['mae']):.3f} (±{np.std(cv_scores['mae']):.3f})")
    log.info(f"平均MAPE: {np.mean(cv_scores['mape']):.3f}% (±{np.std(cv_scores['mape']):.3f}%)")
    log.info(f"平均RMSE: {np.mean(cv_scores['rmse']):.3f} (±{np.std(cv_scores['rmse']):.3f})")

    # 在全部训练数据上训练最终模型
    best_svm.fit(X, y)
    preds = best_svm.predict(X_test)

    # 计算测试集指标
    test_r2 = r2_score(y_test, preds)
    test_mse = mean_squared_error(y_test, preds)
    test_mae = mean_absolute_error(y_test, preds)
    test_mape = mean_absolute_percentage_error(y_test, preds)
    test_rmse = np.sqrt(test_mse)
    test_adjusted_r2 = adjusted_r2_score(test_r2, len(y_test), X_test.shape[1])

    # 打印测试集结果
    log.info("\n========测试集结果========")
    log.info(f"R2: {test_r2:.3f}")
    log.info(f"调整后R2: {test_adjusted_r2:.3f}")
    log.info(f"MSE: {test_mse:.3f}")
    log.info(f"MAE: {test_mae:.3f}")
    log.info(f"MAPE: {test_mape:.3f}%")
    log.info(f"RMSE: {test_rmse:.3f}")

    # 保存模型
    model_dir = "model"
    os.makedirs(model_dir, exist_ok=True)
    model_save_path = f"{model_dir}/svm_model.joblib"
    joblib.dump(best_svm, model_save_path)
    log.info(f"\n模型已保存至 {model_save_path}")

    return {
        'model': best_svm,
        'cv_metrics': {
            'r2': np.mean(cv_scores['r2']),
            'adjusted_r2': np.mean(cv_scores['adjusted_r2']),
            'mse': np.mean(cv_scores['mse']),
            'mae': np.mean(cv_scores['mae']),
            'mape': np.mean(cv_scores['mape']),
            'rmse': np.mean(cv_scores['rmse'])
        },
        'test_metrics': {
            'r2': test_r2,
            'adjusted_r2': test_adjusted_r2,
            'mse': test_mse,
            'mae': test_mae,
            'mape': test_mape,
            'rmse': test_rmse
        }
    }

if __name__ == '__main__':
    organ_name = 'brain'
    merge_filepath = "../../data/数据表汇总.xlsx"
    organ_names_list = ['blood', 'bone', 'brain', 'fat', 'heart',
                        'intestine', 'kidney', 'liver', 'lung', 'muscle',
                        'pancreas', 'spleen', 'stomach', 'uterus']
    certain_time = 60
    train_datasets_dir = "../../data/train/datasets"
    test_datasets_dir = "../../data/test/datasets"

    # 创建模型保存目录
    model_dir = "model"
    os.makedirs(model_dir, exist_ok=True)

    overwrite = False
    FP = True
    if FP:
        log.info("目标特征为：分子指纹")
    else:
        log.info("目标特征为：分子描述符")
    check_data_exist(merge_filepath, organ_names_list, certain_time,
                     train_datasets_dir, test_datasets_dir,
                     FP=FP, overwrite=overwrite)
    results = train_svm(organ_name)  # 调用训练 SVM 的函数

    # 保存模型
    model_save_path = f"{model_dir}/svm_fp_model.joblib"
    joblib.dump(results['model'], model_save_path)
    log.info(f"模型已保存至 {model_save_path}")
