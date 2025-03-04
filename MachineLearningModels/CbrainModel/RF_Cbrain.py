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

# 初始化结果保存目录
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
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
        'max_depth': trial.suggest_int('max_depth', 3, 30),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
        'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
        'bootstrap': trial.suggest_categorical('bootstrap', [True, False]),
        'max_leaf_nodes': trial.suggest_int('max_leaf_nodes', 10, 1000),
        'min_weight_fraction_leaf': trial.suggest_float('min_weight_fraction_leaf', 0.0, 0.5),
    }

    # 仅在 bootstrap 为 True 时设置 max_samples
    if params['bootstrap']:
        params['max_samples'] = trial.suggest_float('max_samples', 0.5, 1.0)
    else:
        params['max_samples'] = None  # 或者不设置 max_samples

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
    计算调整后的R方
    :param r2: R方值
    :param n: 样本数量
    :param k: 自变量数量
    :return: 调整后的R方值
    """
    return 1 - (1 - r2) * (n - 1) / (n - k - 1)

def train_random_forest(organ_name):
    X, y = loader.get_sklearn_data('../../data/train/train_organ_df.npy', organ_name)
    X_test, y_test = loader.get_sklearn_data('../../data/test/test_organ_df.npy', organ_name)

    model_dir = "model"
    os.makedirs(model_dir, exist_ok=True)

    log.info("开始随机森林模型的超参数优化")
    study = optuna.create_study(direction='minimize')
    study.optimize(lambda trial: objective(trial, X, y), n_trials=50)

    best_params = study.best_params
    log.info(f"最佳参数: {best_params}")

    rf = RandomForestRegressor(**best_params)
    r2_list = []
    adj_r2_list = []
    mse_list = []
    mae_list = []
    mape_list = []
    rmse_list = []

    log.info("进行随机森林模型训练")
    cv = KFold(n_splits=10, shuffle=True, random_state=42)
    y_train_all, y_train_pred_all = [], []
    y_val_all, y_val_pred_all = [], []

    for train_idx, val_idx in tqdm(cv.split(X), desc="交叉验证"):
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

    log.info(f"交叉验证 R2: {np.mean(r2_list)}, 调整后R2: {np.mean(adj_r2_list)}, MSE: {np.mean(mse_list)}, MAE: {np.mean(mae_list)}, MAPE: {np.mean(mape_list)}, RMSE: {np.mean(rmse_list)}")

    # 打印十折交叉验证结果
    log.info("\n========十折交叉验证结果========")
    log.info(f"平均R2: {np.mean(r2_list):.3f} (±{np.std(r2_list):.3f})")
    log.info(f"平均调整后R2: {np.mean(adj_r2_list):.3f} (±{np.std(adj_r2_list):.3f})")
    log.info(f"平均MSE: {np.mean(mse_list):.3f} (±{np.std(mse_list):.3f})")
    log.info(f"平均MAE: {np.mean(mae_list):.3f} (±{np.std(mae_list):.3f})")
    log.info(f"平均MAPE: {np.mean(mape_list):.3f}% (±{np.std(mape_list):.3f}%)")
    log.info(f"平均RMSE: {np.mean(rmse_list):.3f} (±{np.std(rmse_list):.3f})")

    rf.fit(X, y)
    test_preds = rf.predict(X_test)
    test_r2 = r2_score(y_test, test_preds)
    test_adj_r2 = adjusted_r2_score(test_r2, X_test.shape[0], X_test.shape[1])
    test_mse = mean_squared_error(y_test, test_preds)
    test_mae = mean_absolute_error(y_test, test_preds)
    test_mape = mean_absolute_percentage_error(y_test, test_preds)
    test_rmse = np.sqrt(mean_squared_error(y_test, test_preds))

    log.info(f"测试集 R2: {test_r2}, 调整后R2: {test_adj_r2}, MSE: {test_mse}, MAE: {test_mae}, MAPE: {test_mape}, RMSE: {test_rmse}")

    # 保存模型
    model_save_path = f"{model_dir}/rf_model.joblib"
    joblib.dump(rf, model_save_path)
    log.info(f"模型已保存至 {model_save_path}")

    # 保存训练集预测结果
    train_results = pd.DataFrame({
        'Actual': y,
        'Predicted': rf.predict(X)
    })
    train_results.to_csv(f'{result_dir}/rf_train_predictions.csv', index=False)

    # 保存测试集预测结果
    test_results = pd.DataFrame({
        'Actual': y_test,
        'Predicted': test_preds
    })
    test_results.to_csv(f'{result_dir}/rf_test_predictions.csv', index=False)

    # 绘制训练集和测试集性能散点图
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

    # 绘制测试数据的散点图
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, test_preds, color='green', label='Test Data')
    plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'k--', lw=4)
    plt.title('Random Forest Regression: True vs. Predicted (Test Data)')
    plt.xlabel('True')
    plt.ylabel('Predict')
    plt.legend(loc='upper left')
    plt.grid(True)
    plt.show()

    # 在测试集评估后添加
    best_metrics = {
        'test_rmse': test_rmse,
        'test_r2': test_r2,
        'test_adj_r2': test_adj_r2,
        'test_mae': test_mae,
        'test_mape': test_mape,
        'test_mse': test_mse
    }

    log.info("\n========最佳模型在测试集上的表现========")
    log.info(f"最佳RMSE: {test_rmse:.3f}")
    log.info(f"最佳R2: {test_r2:.3f}")
    log.info(f"最佳调整后R2: {test_adj_r2:.3f}")
    log.info(f"最佳MAE: {test_mae:.3f}")
    log.info(f"最佳MAPE: {test_mape:.3f}%")
    log.info(f"最佳MSE: {test_mse:.3f}")

    # 输出测试集结果
    print("\n测试集结果:")
    print(f"R2值: {test_r2:.3f}")
    print(f"调整后R2值: {test_adj_r2:.3f}")
    print(f"MSE值: {test_mse:.3f}")
    print(f"MAE值: {test_mae:.3f}")
    print(f"MAPE值: {test_mape:.3f}%")
    print(f"RMSE值: {test_rmse:.3f}")

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
    merge_filepath = "../../data/数据表汇总.xlsx"
    organ_names_list = ['blood', 'bone', 'brain', 'fat', 'heart',
                        'intestine', 'kidney', 'liver', 'lung', 'muscle',
                        'pancreas', 'spleen', 'stomach', 'uterus']
    certain_time = 60
    train_datasets_dir = "../../data/train/datasets"
    test_datasets_dir = "../../data/test/datasets"
    overwrite = False
    FP = False
    if FP:
        log.info("目标特征为：分子指纹")
    else:
        log.info("目标特征为：分子描述符")
    check_data_exist(merge_filepath, organ_names_list, certain_time,
                     train_datasets_dir, test_datasets_dir,
                     FP=FP, overwrite=overwrite)

    results = train_random_forest(organ_name)

