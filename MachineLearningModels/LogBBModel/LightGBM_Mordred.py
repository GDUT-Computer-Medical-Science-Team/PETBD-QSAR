import sys
from time import time
import numpy as np
import optuna
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split, KFold
import pandas as pd
from datetime import datetime
from preprocess.data_preprocess.FeatureExtraction import FeatureExtraction
from preprocess.data_preprocess.data_preprocess_utils import calculate_Mordred_desc
import os
from utils.DataLogger import DataLogger
import lightgbm as lgb
from tqdm import tqdm

# MAPE calculation function
def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    non_zero_index = (y_true != 0)
    return np.mean(np.abs((y_true[non_zero_index] - y_pred[non_zero_index]) / y_true[non_zero_index])) * 100

def adjusted_r2_score(r2, n, k):
    """
    计算调整后的R方
    :param r2: R方值
    :param n: 样本数量
    :param k: 自变量数量
    :return: 调整后的R方值
    """
    if n <= k + 1:
        raise ValueError("样本数量必须大于自变量数量加1")
    return 1 - (1 - r2) * (n - 1) / (n - k - 1)

# 创建必要的目录
os.makedirs("../../data/logBB_data/log", exist_ok=True)
os.makedirs("model", exist_ok=True)

log = DataLogger(log_file=f"../../data/logBB_data/log/{datetime.now().strftime('%Y%m%d')}.log").getlog("ratio_lightgbm_training")

os.makedirs("./logbbModel", exist_ok=True)

def preprocess_data(X, y):
    # 1. 移除常量特征
    constant_features = [col for col in X.columns if X[col].nunique() == 1]
    X = X.drop(columns=constant_features)
    
    # 2. 处理异常值
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.mean())
    
    # 3. 处理偏态分布
    numeric_features = X.select_dtypes(include=[np.number]).columns
    for col in numeric_features:
        if X[col].skew() > 1:
            X[col] = np.log1p(X[col] - X[col].min() + 1e-6)
    
    return X, y

def objective(trial, X, y, seed):
    param = {
        'objective': 'regression',
        'metric': 'rmse',
        'boosting_type': 'gbdt',
        'num_leaves': trial.suggest_int('num_leaves', 20, 300),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2),
        'feature_fraction': trial.suggest_float('feature_fraction', 0.6, 1.0),
        'bagging_fraction': trial.suggest_float('bagging_fraction', 0.6, 1.0),
        'bagging_freq': trial.suggest_int('bagging_freq', 0, 10),
        'min_child_samples': trial.suggest_int('min_child_samples', 10, 50),
        'min_child_weight': trial.suggest_float('min_child_weight', 1e-3, 0.1),
        'min_split_gain': trial.suggest_float('min_split_gain', 1e-3, 0.1),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-3, 1.0),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-3, 1.0),
        'n_estimators': trial.suggest_int('n_estimators', 500, 3000),
        'max_depth': trial.suggest_int('max_depth', 5, 15),
        'random_state': seed,
        'verbose': -1
    }

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=seed, test_size=0.1)
    model = lgb.LGBMRegressor(**param)
    
    # Remove verbose from fit_params
    fit_params = {
        'eval_set': [(X_test, y_test)],
        'callbacks': [lgb.early_stopping(stopping_rounds=100)]
    }
    
    model.fit(X_train, y_train, **fit_params)
    
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    return r2

if __name__ == '__main__':
    logBB_data_file = "../../data/logBB_data/logBB.csv"
    logBB_desc_file = "../../data/logBB_data/logBB_w_desc.csv"
    logBB_desc_index_file = "../../data/logBB_data/desc_index.txt"
    log.info("===============启动LightGBM调参及训练工作===============")

    smile_column_name = 'SMILES'
    pred_column_name = 'logBB'
    RFE_features_to_select = 50
    n_optuna_trial = 100
    cv_times = 10
    seed = int(time())

    if not os.path.exists(logBB_data_file):
        raise FileNotFoundError("缺失logBB数据集")

    if os.path.exists(logBB_desc_file):
        log.info("存在特征文件，进行读取")
        df = pd.read_csv(logBB_desc_file, encoding='utf-8')
        y = df[pred_column_name]
        X = df.drop([smile_column_name, pred_column_name], axis=1)
    else:
        log.info("特征文件不存在，执行特征生成工作")
        df = pd.read_csv(logBB_data_file, encoding='utf-8')
        df = df.dropna(subset=[pred_column_name])
        df = df.reset_index(drop=True)

        y = df[pred_column_name]
        SMILES = df[smile_column_name]

        # 生成特征
        X = calculate_Mordred_desc(SMILES)
        log.info(f"保存特征数据到csv文件 {logBB_desc_file} 中")
        pd.concat([X, y], axis=1).to_csv(logBB_desc_file, encoding='utf-8', index=False)
        X = X.drop(smile_column_name, axis=1)

    # 特征选择与数据准备
    X = df.drop(['SMILES', 'logBB'], axis=1)

    # 在特征选择之前排除 blood mean60min
    if 'blood mean60min' in X.columns:
        X = X.drop('blood mean60min', axis=1)
        log.info("已排除特征: blood mean60min")

    # 确保所有特征都是数值类型
    X = X.apply(pd.to_numeric, errors='coerce')  # 将非数值数据转换为 NaN
    X = X.fillna(0)  # 用0填充NaN值

    # 读取特征索引文件
    if os.path.exists(logBB_desc_index_file):
        with open(logBB_desc_index_file, 'r') as f:
            selected_features = f.read().splitlines()
        log.info(f"加载了 {len(selected_features)} 个特征索引")
        # 只保留选择的特征
        X = X[selected_features]
    else:
        log.warning("特征索引文件不存在，将使用所有生成的特征")

    # 先进行整体归一化
    sc = MinMaxScaler()
    sc.fit(X)
    X = pd.DataFrame(sc.transform(X))

    # 预处理
    X, y = preprocess_data(X, y)

    # 模型调参
    log.info("进行LightGBM调参")
    study = optuna.create_study(direction='maximize')
    study.optimize(lambda trial: objective(trial, X, y, seed), 
                  n_jobs=4, n_trials=n_optuna_trial)

    log.info(f"最佳参数: {study.best_params}")
    log.info(f"最佳预测结果: {study.best_value}")

    # 最优参数投入使用
    model = lgb.LGBMRegressor(**study.best_params)

    # 数据集划分
    # 首先分出独立测试集 (10%)
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=0.1, random_state=42
    )
    log.info(f"将数据集按9:1划分为训练验证集({len(X_train_val)}个样本)和测试集({len(X_test)}个样本)")

    # 在剩余数据中分出验证集 (10%)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=0.1, random_state=42
    )
    log.info(f"将训练验证集按9:1划分为训练集({len(X_train)}个样本)和验证集({len(X_val)}个样本)")

    # 训练模型
    model.fit(X_train, y_train,
             eval_set=[(X_val, y_val)],
             callbacks=[lgb.early_stopping(stopping_rounds=50)])

    # 在所有数据集上进行预测
    y_pred_test = model.predict(X_test)

    # 计算测试集指标
    mse_test = mean_squared_error(y_test, y_pred_test)
    rmse_test = np.sqrt(mse_test)
    r2_test = r2_score(y_test, y_pred_test)
    mae_test = mean_absolute_error(y_test, y_pred_test)
    mape_test = mean_absolute_percentage_error(y_test, y_pred_test)
    adj_r2_test = adjusted_r2_score(r2_test, len(y_test), X_test.shape[1])

    # 输出评估结果
    log.info("最终测试集评估指标：")
    log.info(f"测试集 -> MSE: {mse_test:.4f}, RMSE: {rmse_test:.4f}, R2: {r2_test:.4f}, Adjusted R2: {adj_r2_test:.4f}, MAE: {mae_test:.4f}, MAPE: {mape_test:.2f}%")

    # 只保存测试集结果
    test_results = pd.DataFrame({
        'True Values': y_test, 
        'Predicted Values': y_pred_test
    })

    # 保存训练集和测试集的预测结果到 CSV 文件
    train_results = pd.DataFrame({
        'True Values': y_train,
        'Predicted Values': model.predict(X_train)
    })
    train_results.to_csv('./result/lightgbm_mordred_train_results.csv', index=False)
    test_results.to_csv('./result/lightgbm_mordred_test_results.csv', index=False)
    log.info("训练集和测试集的预测结果已保存到 './result/lightgbm_mordred_train_results.csv' 和 './result/lightgbm_mordred_test_results.csv'")

    # 定义模型保存路径
    model_save_path = os.path.join("./logbbModel", "lightgbm_mordred_model.joblib")
    model.booster_.save_model(model_save_path)
    log.info(f"模型已保存至 {model_save_path}")

    log.info("\n最终数据集划分:")
    log.info(f"训练集: {len(X_train)}个样本 ({len(X_train)/len(X)*100:.1f}%) [应约为81%]")
    log.info(f"验证集: {len(X_val)}个样本 ({len(X_val)/len(X)*100:.1f}%) [应约为9%]")
    log.info(f"测试集: {len(X_test)}个样本 ({len(X_test)/len(X)*100:.1f}%) [应约为10%]")
