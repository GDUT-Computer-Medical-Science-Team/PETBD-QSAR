import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, mean_absolute_percentage_error
from sklearn.model_selection import KFold, train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime
import os
import optuna
from utils.DataLogger import DataLogger
from preprocess.data_preprocess.FeatureExtraction import FeatureExtraction
from preprocess.data_preprocess.data_preprocess_utils import calculate_Mordred_desc
import joblib

# 创建必要的目录
os.makedirs("../../data/logBB_data/log", exist_ok=True)
os.makedirs("model", exist_ok=True)

log = DataLogger(log_file=f"../../data/logBB_data/log/{datetime.now().strftime('%Y%m%d')}.log").getlog("mlp_cross_validation")

def calculate_mape(y_true, y_pred):
    epsilon = 1e-8  # 添加平滑项，避免分母为零
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / (y_true + epsilon))) * 100

def objective(trial):
    # 定义超参数搜索空间
    param = {
        'hidden_layer_sizes': trial.suggest_categorical('hidden_layer_sizes', [(50,), (100,), (150,), (100, 50)]),
        'activation': trial.suggest_categorical('activation', ['relu', 'tanh']),
        'solver': trial.suggest_categorical('solver', ['adam', 'sgd']),
        'alpha': trial.suggest_loguniform('alpha', 1e-5, 1e-1),
        'learning_rate': trial.suggest_categorical('learning_rate', ['constant', 'invscaling', 'adaptive']),
        'learning_rate_init': trial.suggest_loguniform('learning_rate_init', 1e-5, 1e-1),
        'random_state': seed
    }

    model = MLPRegressor(**param)
    kf = KFold(n_splits=5, shuffle=True, random_state=seed)
    rmse_list = []

    for train_index, test_index in kf.split(X_scaled):
        X_train, X_test = X_scaled[train_index], X_scaled[test_index]
        y_train, y_test = y[train_index], y[test_index]

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        rmse_list.append(rmse)

    return np.mean(rmse_list)

def adjusted_r2(y_true, y_pred, n_features):
    """
    计算调整后的R方
    :param y_true: 真实值
    :param y_pred: 预测值
    :param n_features: 特征数量
    :return: 调整后的R方值
    """
    r2 = r2_score(y_true, y_pred)
    n = len(y_true)
    return 1 - (1 - r2) * (n - 1) / (n - n_features - 1)

if __name__ == '__main__':
    # 需要的文件路径
    logBB_data_file = "../../data/logBB_data/logBB.csv"
    logBB_desc_file = "../../data/logBB_data/logBB_w_desc.csv"
    logBB_desc_index_file = "../../data/logBB_data/desc_index.txt"
    model_dir = "./logbbModel"

    log.info("===============启动MLP交叉验证工作===============")

    # 变量初始化
    smile_column_name = 'SMILES'
    pred_column_name = 'logBB'
    seed = 42  # 设置随机种子以确保结果可复现

    if not os.path.exists(logBB_data_file):
        raise FileNotFoundError("缺失logBB数据集")

    # 检查特征文件是否存在并加载数据
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

        # 在特征选择之前排除 blood mean60min
        if 'blood mean60min' in df.columns:
            df = df.drop('blood mean60min', axis=1)
            log.info("已排除特征: blood mean60min")

        # 确保所有特征都是数值类型
        X = df.drop(['SMILES', 'logBB'], axis=1)
        X = X.apply(pd.to_numeric, errors='coerce')  # 将非数值数据转换为 NaN
        X = X.fillna(0)  # 用0填充NaN值

        log.info(f"保存特征数据到csv文件 {logBB_desc_file} 中")
        pd.concat([X, y], axis=1).to_csv(logBB_desc_file, encoding='utf-8', index=False)

    # 特征筛选
    if not os.path.exists(logBB_desc_index_file):
        log.info("不存在特征索引文件，进行特征筛选")
        log.info(f"筛选前的特征矩阵形状为：{X.shape}")
        desc_index = FeatureExtraction(X, y, VT_threshold=0.02, RFE_features_to_select=50).feature_extraction(returnIndex=True, index_dtype=int)
        np.savetxt(logBB_desc_index_file, desc_index, fmt='%d')
        X = X.iloc[:, desc_index]
        log.info(f"特征筛选完成，筛选后的特征矩阵形状为：{X.shape}, 筛选得到的特征索引保存到：{logBB_desc_index_file}")
    else:
        log.info("存在特征索引文件，进行读取")
        desc_index = np.loadtxt(logBB_desc_index_file, dtype=int).tolist()
        X = X.iloc[:, desc_index]
        log.info(f"读取特征索引完成，筛选后的特征矩阵形状为：{X.shape}")

    # 特征归一化
    log.info("归一化特征数据")
    sc = MinMaxScaler()
    X_scaled = sc.fit_transform(X)

    # 首先分出独立测试集 (10%)
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X_scaled, y, test_size=0.1, random_state=42
    )
    log.info(f"将数据集按9:1划分为训练验证集({len(X_train_val)}个样本)和测试集({len(X_test)}个样本)")

    # 在剩余数据中分出验证集 (10%)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=0.1, random_state=42
    )
    log.info(f"将训练验证集按9:1划分为训练集({len(X_train)}个样本)和验证集({len(X_val)}个样本)")

    # Optuna 超参数优化
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=100)

    log.info(f"最优参数: {study.best_params}")
    log.info(f"最优RMSE: {study.best_value}")

    # 使用最佳参数重新训练模型并进行评估
    best_params = study.best_params
    model = MLPRegressor(**best_params)

    # 设置交叉验证
    cv = KFold(n_splits=10, shuffle=True, random_state=seed)

    # 存储每折的性能指标
    rmse_scores = []
    r2_scores = []
    adj_r2_scores = []
    mae_scores = []
    mse_scores = []
    mape_scores = []

    # 执行交叉验证
    for train_idx, test_idx in cv.split(X_train):
        X_train_cv, X_test_cv = X_train[train_idx], X_train[test_idx]
        y_train_cv, y_test_cv = y_train.iloc[train_idx], y_train.iloc[test_idx]

        # 训练模型
        model.fit(X_train_cv, y_train_cv)

        # 在验证集上进行预测
        y_pred_cv = model.predict(X_test_cv)

        # 计算性能指标
        rmse = np.sqrt(mean_squared_error(y_test_cv, y_pred_cv))
        r2 = r2_score(y_test_cv, y_pred_cv)
        adj_r2 = adjusted_r2(y_test_cv, y_pred_cv, X_train_cv.shape[1])
        mae = mean_absolute_error(y_test_cv, y_pred_cv)
        mse = mean_squared_error(y_test_cv, y_pred_cv)
        mape = calculate_mape(y_test_cv, y_pred_cv)

        # 存储性能指标
        rmse_scores.append(rmse)
        r2_scores.append(r2)
        adj_r2_scores.append(adj_r2)
        mae_scores.append(mae)
        mse_scores.append(mse)
        mape_scores.append(mape)

    # 打印平均性能指标
    log.info("========十折交叉验证结果========")
    log.info(f"RMSE: {np.mean(rmse_scores):.3f}±{np.std(rmse_scores):.3f}")
    log.info(f"R2: {np.mean(r2_scores):.3f}±{np.std(r2_scores):.3f}")
    log.info(f"Adjusted R2: {np.mean(adj_r2_scores):.3f}±{np.std(adj_r2_scores):.3f}")
    log.info(f"MAE: {np.mean(mae_scores):.3f}±{np.std(mae_scores):.3f}")
    log.info(f"MSE: {np.mean(mse_scores):.3f}±{np.std(mse_scores):.3f}")
    log.info(f"MAPE: {np.mean(mape_scores):.3f}±{np.std(mape_scores):.3f}%")

    # 在测试集上进行最终评估
    model.fit(X_train, y_train)
    y_pred_test = model.predict(X_test)

    # 计算测试集指标
    mse_test = mean_squared_error(y_test, y_pred_test)
    rmse_test = np.sqrt(mse_test)
    r2_test = r2_score(y_test, y_pred_test)
    mae_test = mean_absolute_error(y_test, y_pred_test)
    mape_test = mean_absolute_percentage_error(y_test, y_pred_test)
    adj_r2_test = 1 - (1 - r2_test) * (len(y_test) - 1) / (len(y_test) - X_test.shape[1] - 1)

    # 在计算测试集指标后添加
    log.info("========测试集结果========")
    log.info(f"RMSE: {rmse_test:.3f}")
    log.info(f"R2: {r2_test:.3f}")
    log.info(f"Adjusted R2: {adj_r2_test:.3f}")
    log.info(f"MAE: {mae_test:.3f}")
    log.info(f"MSE: {mse_test:.3f}")
    log.info(f"MAPE: {mape_test:.3f}%")

    # 在测试集评估后添加
    joblib.dump(model, os.path.join(model_dir, "mlp_mordred_model.joblib"))
    log.info(f"模型已保存至 {os.path.join(model_dir, 'mlp_mordred_model.joblib')}")

    # 只保存测试集结果
    test_results = pd.DataFrame({
        'True Values': y_test, 
        'Predicted Values': y_pred_test
    })

    os.makedirs('./result', exist_ok=True)
    test_results.to_csv('./result/mlp_mordred_test_results.csv', index=False)

    log.info("\n最终数据集划分:")
    log.info(f"训练集: {len(X_train)}个样本 ({len(X_train)/len(X)*100:.1f}%) [应约为81%]")
    log.info(f"验证集: {len(X_val)}个样本 ({len(X_val)/len(X)*100:.1f}%) [应约为9%]")
    log.info(f"测试集: {len(X_test)}个样本 ({len(X_test)/len(X)*100:.1f}%) [应约为10%]")

    # 保存训练集和测试集的预测结果到 CSV 文件
    train_results = pd.DataFrame({
        'True Values': y_train,
        'Predicted Values': model.predict(X_train)
    })
    train_results.to_csv('./result/mlp_mordred_train_results.csv', index=False)
    test_results.to_csv('./result/mlp_mordred_test_results.csv', index=False)
    log.info("训练集和测试集的预测结果已保存到 './result/mlp_mordred_train_results.csv' 和 './result/mlp_mordred_test_results.csv'")

    # 定义模型保存路径
    model_save_path = os.path.join(model_dir, "mlp_mordred_model.joblib")
