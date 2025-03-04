import sys
from time import time

import numpy
import optuna
from sklearn.preprocessing import MinMaxScaler
from xgboost.sklearn import XGBRegressor
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, mean_absolute_percentage_error
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.model_selection import KFold
from tqdm import tqdm
from datetime import datetime
from preprocess.data_preprocess.FeatureExtraction import FeatureExtraction
from preprocess.data_preprocess.data_preprocess_utils import calculate_Mordred_desc
import joblib
import os
from utils.DataLogger import DataLogger
import matplotlib.pyplot as plt

# 创建必要的目录
os.makedirs("../../data/logBB_data/log", exist_ok=True)
os.makedirs("model", exist_ok=True)

log = DataLogger(log_file=f"../../data/logBB_data/log/{datetime.now().strftime('%Y%m%d')}.log").getlog("ratio_xgboost_training")

os.makedirs("./logbbModel", exist_ok=True)

def adjusted_r2_score(r2, n, k):
    """
    计算调整后的R方
    :param r2: R方值
    :param n: 样本数量
    :param k: 自变量数量
    :return: 调整后的R方值
    """
    return 1 - (1 - r2) * (n - 1) / (n - k - 1)

if __name__ == '__main__':
    # 需要的文件路径
    logBB_data_file = "../../data/logBB_data/logBB.csv"
    logBB_desc_file = "../../data/logBB_data/logBB_w_desc.csv"
    logBB_desc_index_file = "../../data/logBB_data/desc_index.txt"
    test_data_file = "../../data/logBB_data/logBB_test.csv"
    log.info("===============启动XGBoost调参及训练工作===============")
    # 变量初始化
    smile_column_name = 'SMILES'
    pred_column_name = 'logBB'
    RFE_features_to_select = 50
    n_optuna_trial = 100
    cv_times = 10
    seed = int(time())

    if not os.path.exists(logBB_data_file):
        raise FileNotFoundError("缺失logBB数据集")
    # 特征读取或获取
    if os.path.exists(logBB_desc_file):
        log.info("存在特征文件，进行读取")
        df = pd.read_csv(logBB_desc_file, encoding='utf-8')
        y = df[pred_column_name]
        X = df.drop([smile_column_name, pred_column_name], axis=1)
    # 描述符数据不存在，读取原始数据，生成描述符并保存
    else:
        log.info("特征文件不存在，执行特征生成工作")
        df = pd.read_csv(logBB_data_file, encoding='utf-8')
        # 根据logBB列，删除logBB列为空的行
        df = df.dropna(subset=[pred_column_name])
        # 重置因为删除行导致顺序不一致的索引
        df = df.reset_index()

        y = df[pred_column_name]
        SMILES = df[smile_column_name]

        X = calculate_Mordred_desc(SMILES)
        log.info(f"保存特征数据到csv文件 {logBB_desc_file} 中")
        pd.concat([X, y], axis=1).to_csv(logBB_desc_file, encoding='utf-8', index=False)
        X = X.drop(smile_column_name, axis=1)


    # 确保所有特征都是数值类型
    X = X.apply(pd.to_numeric, errors='coerce')  # 将非数值数据转换为 NaN
    X = X.fillna(0)  # 用0填充NaN值

    # 特征筛选
    if not os.path.exists(logBB_desc_index_file):
        log.info("不存在特征索引文件，进行特征筛选")
        log.info(f"筛选前的特征矩阵形状为：{X.shape}")
        desc_index = (FeatureExtraction(X,
                                      y,
                                      VT_threshold=0.02,
                                      RFE_features_to_select=RFE_features_to_select).
                      feature_extraction(returnIndex=True, index_dtype=int))
        try:
            np.savetxt(logBB_desc_index_file, desc_index, fmt='%d')
            X = X.iloc[:, desc_index]  # 使用 iloc 而不是直接索引
            log.info(f"特征筛选完成，筛选后的特征矩阵形状为：{X.shape}, 筛选得到的特征索引保存到：{logBB_desc_index_file}")
            # 打印筛选后的特征名称
            log.info(f"筛选后的特征名称: {X.columns.tolist()}")
        except (TypeError, KeyError) as e:
            log.error(e)
            os.remove(logBB_desc_index_file)
            sys.exit()
    else:
        log.info("存在特征索引文件，进行读取")
        desc_index = np.loadtxt(logBB_desc_index_file, dtype=int, delimiter=',').tolist()
        X = X.iloc[:, desc_index]  # 使用 iloc 而不是直接索引
        log.info(f"读取特征索引完成，筛选后的特征矩阵形状为：{X.shape}")
        # 打印筛选后的特征名称
        log.info(f"筛选后的特征名称: {X.columns.tolist()}")

    # 数据归一化
    log.info("归一化特征数据")
    sc = MinMaxScaler()
    X_scaled = sc.fit_transform(X)
    X = pd.DataFrame(X_scaled, columns=X.columns)

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

    # 数据归一化 - 使用训练验证集拟合scaler
    sc = MinMaxScaler()
    X_train_val_scaled = sc.fit_transform(X_train_val)
    X_test_scaled = sc.transform(X_test)

    X_train_val = pd.DataFrame(X_train_val_scaled, columns=X.columns)
    X_test = pd.DataFrame(X_test_scaled, columns=X.columns)

    # 在超参数搜索和交叉验证时使用训练验证集
    X = X_train_val
    y = y_train_val
    log.info("使用训练验证集进行超参数优化和交叉验证")

    # 在加载数据后，确保索引一致
    y = y.reset_index(drop=True)

    # 计算MAPE的函数
    def calculate_mape(y_true, y_pred):
        y_true, y_pred = np.array(y_true), np.array(y_pred)
        return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

    # 模型调参
    def objective(trial):
        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=seed, test_size=0.1)
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 3000),
            'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.03),
            'max_depth': trial.suggest_int('max_depth', 3, 15),
            'min_child_weight': trial.suggest_float('min_child_weight', 1e-5, 1e2),
            'subsample': trial.suggest_float('subsample', 0.1, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.1, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-5, 1e2),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-5, 1e2),
            'random_state': seed,
            'objective': 'reg:squarederror'  # 回归任务
        }
        model = XGBRegressor(**params)

        # 训练模型
        model.fit(X_train, y_train,
                 eval_set=[(X_val, y_val)],
                 early_stopping_rounds=50,
                 verbose=False)  # XGBoost 支持 verbose 参数

        # 在测试集上进行预测
        y_pred = model.predict(X_test)

        # 计算误差
        # rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        # mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        return r2


    log.info("进行XGBoost调参")
    # study = optuna.create_study(direction='minimize')
    # 最大化R2结果
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_jobs=4, n_trials=n_optuna_trial)

    log.info(f"最佳参数: {study.best_params}")
    log.info(f"最佳预测结果: {study.best_value}")

    # 最优参数投入使用
    model = XGBRegressor(**study.best_params)

    # 训练集交叉验证训练
    cv = KFold(n_splits=cv_times, random_state=seed, shuffle=True)
    rmse_result_list = []
    r2_result_list = []
    mse_result_list = []  # 移到循环外
    mae_result_list = []  # 移到循环外
    adj_r2_result_list = []  # 移到循环外
    log.info(f"使用最佳参数进行{cv_times}折交叉验证")

    for idx, (train_idx, test_idx) in tqdm(enumerate(cv.split(X, y)), desc="交叉验证: ", total=cv_times):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        # 训练模型
        model.fit(X_train, y_train, eval_set=[(X_test, y_test)], early_stopping_rounds=100, verbose=False)

        # 在测试集上进行预测
        y_pred = model.predict(X_test)

        # 计算各种评估指标
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        adj_r2 = adjusted_r2_score(r2, len(y_test), X_test.shape[1])

        rmse_result_list.append(rmse)
        r2_result_list.append(r2)
        mse_result_list.append(mse)
        mae_result_list.append(mae)
        adj_r2_result_list.append(adj_r2)

    log.info(f"随机种子: {seed}")
    log.info("========十折交叉验证结果========")
    log.info(f"RMSE: {round(np.mean(rmse_result_list), 3)}±{round(np.std(rmse_result_list), 3)}")
    log.info(f"MSE: {round(np.mean(mse_result_list), 3)}±{round(np.std(mse_result_list), 3)}")
    log.info(f"MAE: {round(np.mean(mae_result_list), 3)}±{round(np.std(mae_result_list), 3)}")
    log.info(f"R2: {round(np.mean(r2_result_list), 3)}±{round(np.std(r2_result_list), 3)}")
    log.info(f"Adjusted R2: {round(np.mean(adj_r2_result_list), 3)}±{round(np.std(adj_r2_result_list), 3)}")

    # 验证集验证
    y_pred = model.predict(X_val)
    rmse = np.sqrt(mean_squared_error(y_val, y_pred))
    mse = mean_squared_error(y_val, y_pred)
    mae = mean_absolute_error(y_val, y_pred)
    r2 = r2_score(y_val, y_pred)
    adj_r2 = adjusted_r2_score(r2, len(y_val), X_val.shape[1])

    log.info("========验证集结果========")
    log.info(f"RMSE: {round(rmse, 3)}")
    log.info(f"MSE: {round(mse, 3)}")
    log.info(f"MAE: {round(mae, 3)}")
    log.info(f"R2: {round(r2, 3)}")
    log.info(f"Adjusted R2: {round(adj_r2, 3)}")

    # TODO: B3DB验证

    model_save_path = "./logbbModel/xgb_mordred_model.joblib"
    joblib.dump(model, model_save_path)
    log.info(f"模型已保存至 {model_save_path}")

    # 最终模型训练和评估
    model_final = XGBRegressor(**study.best_params)
    model_final.fit(X_train, y_train)

    # 在所有数据集上进行预测
    y_pred_train = model_final.predict(X_train)
    y_pred_val = model_final.predict(X_val)
    y_pred_test = model_final.predict(X_test)

    # 在预测之后添加评估指标计算
    # 计算训练集指标
    mse_train = mean_squared_error(y_train, y_pred_train)
    rmse_train = np.sqrt(mse_train)
    r2_train = r2_score(y_train, y_pred_train)
    mae_train = mean_absolute_error(y_train, y_pred_train)
    mape_train = mean_absolute_percentage_error(y_train, y_pred_train)
    adj_r2_train = adjusted_r2_score(r2_train, len(y_train), X_train.shape[1])

    # 计算验证集指标
    mse_val = mean_squared_error(y_val, y_pred_val)
    rmse_val = np.sqrt(mse_val)
    r2_val = r2_score(y_val, y_pred_val)
    mae_val = mean_absolute_error(y_val, y_pred_val)
    mape_val = mean_absolute_percentage_error(y_val, y_pred_val)
    adj_r2_val = adjusted_r2_score(r2_val, len(y_val), X_val.shape[1])

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

    os.makedirs('./result', exist_ok=True)
    test_results.to_csv('./result/xgb_mordred_test_results.csv', index=False)

    log.info("\n最终数据集划分:")
    log.info(f"训练集: {len(X_train)}个样本 ({len(X_train)/len(X)*100:.1f}%) [应约为81%]")
    log.info(f"验证集: {len(X_val)}个样本 ({len(X_val)/len(X)*100:.1f}%) [应约为9%]")
    log.info(f"测试集: {len(X_test)}个样本 ({len(X_test)/len(X)*100:.1f}%) [应约为10%]")

    # 修改绘制散点图的函数
    def plot_scatter(y_train, y_pred_train, y_test, y_pred_test, save_path='./result/xgb_mordred_scatter.png'):
        """
        在同一张图中绘制训练集和测试集的散点图
        """
        plt.figure(figsize=(8, 8))

        # 绘制训练集散点
        plt.scatter(y_train, y_pred_train, alpha=0.5, label='Training Set', color='blue')

        # 绘制测试集散点
        plt.scatter(y_test, y_pred_test, alpha=0.5, label='Test Set', color='red')

        # 绘制对角线
        all_min = min(min(y_train), min(y_test))
        all_max = max(max(y_train), max(y_test))
        plt.plot([all_min, all_max], [all_min, all_max], 'k--', lw=2)

        plt.xlabel('Experimental logBB')
        plt.ylabel('Predicted logBB')
        plt.title('Predicted vs Experimental logBB')

        # 添加R²到图中
        r2_train = r2_score(y_train, y_pred_train)
        r2_test = r2_score(y_test, y_pred_test)
        plt.text(0.05, 0.95, f'Training R² = {r2_train:.3f}', transform=plt.gca().transAxes)
        plt.text(0.05, 0.90, f'Test R² = {r2_test:.3f}', transform=plt.gca().transAxes)

        plt.legend()
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()

    # 在评估完测试集后添加绘图代码
    plot_scatter(y_train, y_pred_train, y_test, y_pred_test)
    log.info("训练集和测试集散点图已保存到 './result/xgb_mordred_scatter.png'")