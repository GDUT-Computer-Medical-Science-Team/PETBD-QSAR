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

# 初始化结果保存目录
cur_time = time.localtime()
result_parent_dir = f"../../result/{time.strftime('%Y%m%d', cur_time)}"
result_dir = f"{result_parent_dir}/{time.strftime('%H%M%S', cur_time)}"
os.makedirs(result_dir, exist_ok=True)
main_log = DataLogger(f"{result_dir}/run.log").getlog(disable_console_output=True)
log = DataLogger().getlog("run")

def check_datasets_exist(parent_folder: str):
    if os.path.exists(parent_folder):
        if not os.path.isdir(parent_folder):
            raise NotADirectoryError(f"错误：{parent_folder}不是目录")
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

def calculate_mape(y_true, y_pred):
    epsilon = 1e-8  # 防止分母为0
    return np.mean(np.abs((y_true - y_pred) / (y_true + epsilon))) * 100

def adjusted_r2_score(r2, n, k):
    """
    计算调整后的R方
    :param r2: R方值
    :param n: 样本数量
    :param k: 自变量数量
    :return: 调整后的R方值
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

    log.info("开始CatBoost模型的超参数优化")
    study = optuna.create_study(direction='minimize')
    study.optimize(lambda trial: objective(trial, X, y), n_trials=50)

    best_params = study.best_params
    log.info(f"最佳参数: {best_params}")

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

    log.info(f"十折交叉验证平均MSE: {avg_mse}, 平均R2: {avg_r2}, 平均调整后R2: {avg_adj_r2}, 平均MAE: {avg_mae}, 平均MAPE: {avg_mape}, 平均RMSE: {avg_rmse}")

    test_preds = cb.predict(X_test)
    test_mse = mean_squared_error(y_test, test_preds)
    test_r2 = r2_score(y_test, test_preds)
    test_adj_r2 = adjusted_r2_score(test_r2, X_test.shape[0], X_test.shape[1])
    test_mae = mean_absolute_error(y_test, test_preds)
    test_mape = calculate_mape(y_test, test_preds)
    test_rmse = np.sqrt(test_mse)

    log.info(f"测试集R2: {test_r2}, 调整后R2: {test_adj_r2}, MSE: {test_mse}, MAE: {test_mae}, MAPE: {test_mape}%, RMSE: {test_rmse}")

    # 在测试集评估后添加
    best_metrics = {
        'test_rmse': test_rmse,
        'test_r2': test_r2,
        'test_adj_r2': test_adj_r2,
        'test_mae': test_mae,
        'test_mape': test_mape,
        'test_mse': test_mse
    }

    log.info("\n========在测试集上的表现========")
    log.info(f"RMSE: {best_metrics['test_rmse']:.3f}")
    log.info(f"R2: {best_metrics['test_r2']:.3f}")
    log.info(f"整后R2: {best_metrics['test_adj_r2']:.3f}")
    log.info(f"MAE: {best_metrics['test_mae']:.3f}")
    log.info(f"MAPE: {best_metrics['test_mape']:.3f}%")
    log.info(f"MSE: {best_metrics['test_mse']:.3f}")


    # 保存模型
    model_save_path = f"{model_dir}/catboost_model.cbm"
    cb.save_model(model_save_path)
    log.info(f"模型已保存至 {model_save_path}")

if __name__ == '__main__':
    organ_name = 'brain'
    merge_filepath = "../../data/数据表汇总.xlsx"
    organ_names_list = ['blood', 'bone', 'brain', 'fat', 'heart',
                        'intestine', 'kidney', 'liver', 'lung', 'muscle',
                        'pancreas', 'spleen', 'stomach', 'uterus']
    certain_time = 60
    train_datasets_dir = "../../data/train/datasets"
    test_datasets_dir = "../../data/test/datasets"

    overwrite = False
    FP = True
    if FP:
        log.info("目标特征为：分子指纹")
    else:
        log.info("目标特征为：分子描述符")
    check_data_exist(merge_filepath, organ_names_list, certain_time,
                     train_datasets_dir, test_datasets_dir,
                     FP=FP, overwrite=overwrite)
    train_catboost(organ_name)
