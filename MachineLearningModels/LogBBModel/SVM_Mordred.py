import sys
from time import time
import numpy as np
import optuna
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, mean_absolute_percentage_error
from sklearn.model_selection import KFold
import pandas as pd
from datetime import datetime
from preprocess.data_preprocess.FeatureExtraction import FeatureExtraction
from preprocess.data_preprocess.data_preprocess_utils import calculate_Mordred_desc
import joblib
import os
from utils.DataLogger import DataLogger
from sklearn.svm import SVR  # 导入支持向量机回归器
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

log = DataLogger(log_file=f"../../data/logBB_data/log/{datetime.now().strftime('%Y%m%d')}.log").getlog("ratio_svm_training")

os.makedirs("./logbbModel", exist_ok=True)


def calculate_mape(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    # 避免除以零的情况
    mask = y_true != 0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100


def adjusted_r2_score(r2, n, k):
    """
    计算调整后的R方
    :param r2: R方值
    :param n: 样本数量
    :param k: 自变量数量
    :return: 调整后的R方值
    """
    return 1 - (1 - r2) * (n - 1) / (n - k - 1)


def objective(trial):
    df = pd.read_csv(logBB_desc_file, encoding='utf-8')
    y = df[pred_column_name]
    X = df.drop([smile_column_name, pred_column_name], axis=1)

    # 在特征选择之前排除 blood mean60min
    if 'blood mean60min' in X.columns:
        X = X.drop('blood mean60min', axis=1)
        log.info("已排除特征: blood mean60min")

    sc = MinMaxScaler()
    sc.fit(X)
    X = pd.DataFrame(sc.transform(X))

    # 定义超参数搜索空间
    param = {
        'kernel': trial.suggest_categorical('kernel', ['linear', 'poly', 'rbf', 'sigmoid']),
        'C': trial.suggest_loguniform('C', 1e-3, 1e2),
        'epsilon': trial.suggest_loguniform('epsilon', 1e-3, 1e1)
    }

    kf = KFold(n_splits=10, shuffle=True, random_state=seed)
    rmse_list = []

    for train_index, test_index in kf.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y[train_index], y[test_index]

        model = SVR(**param)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        rmse_list.append(rmse)

    return np.mean(rmse_list)


if __name__ == '__main__':
    logBB_data_file = "../../data/logBB_data/logBB.csv"
    logBB_desc_file = "../../data/logBB_data/logBB_w_desc.csv"
    logBB_desc_index_file = "../../data/logBB_data/desc_index.txt"
    log.info("===============启动SVM调参及训练工作===============")

    smile_column_name = 'SMILES'
    pred_column_name = 'logBB'
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

        X = calculate_Mordred_desc(SMILES)
        log.info(f"保存特征数据到csv文件 {logBB_desc_file} 中")
        pd.concat([X, y], axis=1).to_csv(logBB_desc_file, encoding='utf-8', index=False)

    # 特征筛选
    if not os.path.exists(logBB_desc_index_file):
        log.info("不存在特征索引文件，进行特征筛选")
        log.info(f"筛选前的特征矩阵形状为：{X.shape}")
        desc_index = (FeatureExtraction(X, y, VT_threshold=0.02, RFE_features_to_select=50)
                      .feature_extraction(returnIndex=True, index_dtype=int))
        try:
            np.savetxt(logBB_desc_index_file, desc_index, fmt='%d')
            X = X.iloc[:, desc_index]
            log.info(f"特征筛选完成，筛选后的特征矩阵形状为：{X.shape}, 筛选得到的特征索引保存到：{logBB_desc_index_file}")
        except (TypeError, KeyError) as e:
            log.error(e)
            os.remove(logBB_desc_index_file)
            sys.exit()
    else:
        log.info("存在特征索引文件，进行读取")
        desc_index = np.loadtxt(logBB_desc_index_file, dtype=int, delimiter=',').tolist()
        X = X.iloc[:, desc_index]
        log.info(f"读取特征索引完成，筛选后的特征矩阵形状为：{X.shape}")

    log.info("归一化特征数据")
    sc = MinMaxScaler()
    sc.fit(X)
    X = pd.DataFrame(sc.transform(X))

    # Optuna 超参数优化
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=100)

    log.info(f"最优参数: {study.best_params}")
    log.info(f"最优RMSE: {study.best_value}")

    # 使用最佳参数重新训练模型并进行交叉验证
    best_params = study.best_params

    kf = KFold(n_splits=10, shuffle=True, random_state=seed)
    rmse_list = []
    mse_list = []
    r2_list = []
    adj_r2_list = []
    mae_list = []
    mape_list = []

    for train_index, test_index in kf.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y[train_index], y[test_index]

        model = SVR(**best_params)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)
        adj_r2 = adjusted_r2_score(r2, X_test.shape[0], X_test.shape[1])
        mae = mean_absolute_error(y_test, y_pred)
        mape = calculate_mape(y_test, y_pred)

        rmse_list.append(rmse)
        mse_list.append(mse)
        r2_list.append(r2)
        adj_r2_list.append(adj_r2)
        mae_list.append(mae)
        mape_list.append(mape)

    avg_rmse = np.mean(rmse_list)
    std_rmse = np.std(rmse_list)
    avg_mse = np.mean(mse_list)
    std_mse = np.std(mse_list)
    avg_r2 = np.mean(r2_list)
    std_r2 = np.std(r2_list)
    avg_adj_r2 = np.mean(adj_r2_list)
    std_adj_r2 = np.std(adj_r2_list)
    avg_mae = np.mean(mae_list)
    std_mae = np.std(mae_list)
    avg_mape = np.mean(mape_list)
    std_mape = np.std(mape_list)

    log.info("========SVM交叉验证结果========")
    log.info(f"Avg RMSE: {round(avg_rmse, 3)} ± {round(std_rmse, 3)}")
    log.info(f"Avg MSE: {round(avg_mse, 3)} ± {round(std_mse, 3)}")
    log.info(f"Avg R2: {round(avg_r2, 3)} ± {round(std_r2, 3)}")
    log.info(f"Avg 调整后R2: {round(avg_adj_r2, 3)} ± {round(std_adj_r2, 3)}")
    log.info(f"Avg MAE: {round(avg_mae, 3)} ± {round(std_mae, 3)}")
    log.info(f"Avg MAPE: {round(avg_mape, 3)}% ± {round(std_mape, 3)}%")

    # 使用最佳参数在全部训练集上训练模型，并在测试集上评估
    log.info("========在测试集上评估模型性能========")
    # 添加数据集划分代码
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

    # 在数据集划分后添加模型训练和评估代码
    # 训练最终模型
    model_final = SVR(**best_params)
    model_final.fit(X_train, y_train)

    # 在所有数据集上进行预测
    y_pred_train = model_final.predict(X_train)
    y_pred_test = model_final.predict(X_test)

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
    test_results.to_csv('./result/svm_mordred_test_results.csv', index=False)

    log.info("\n最终数据集划分:")
    log.info(f"训练集: {len(X_train)}个样本 ({len(X_train)/len(X)*100:.1f}%) [应约为81%]")
    log.info(f"验证集: {len(X_val)}个样本 ({len(X_val)/len(X)*100:.1f}%) [应约为9%]")
    log.info(f"测试集: {len(X_test)}个样本 ({len(X_test)/len(X)*100:.1f}%) [应约为10%]")

    # 绘制散点图
    def plot_scatter(y_train, y_pred_train, y_test, y_pred_test, save_path='./result/svm_mordred_scatter.png'):
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

    # 绘制散点图
    plot_scatter(y_train, y_pred_train, y_test, y_pred_test)
    log.info("训练集和测试集散点图已保存到 './result/svm_mordred_scatter.png'")

    # 保存模型
    model_save_path = "./logbbModel/svm_mordred_model.joblib"
    joblib.dump(model_final, model_save_path)
    log.info(f"模型已保存至 {model_save_path}")
