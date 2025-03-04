import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime
import os
import matplotlib.pyplot as plt
from utils.DataLogger import DataLogger
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, mean_absolute_percentage_error
from sklearn.svm import SVR
import optuna
import tempfile
from padelpy import padeldescriptor
import joblib
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestRegressor  # 使用随机森林进行特征选择

# 设置日志
log = DataLogger(log_file=f"../../data/logBB_data/log/{datetime.now().strftime('%Y%m%d')}.log").getlog("svm_feature_selection")

# 定义模型保存目录
model_dir = "logbbModel"
if not os.path.exists(model_dir):
    os.makedirs(model_dir)


def calculate_padel_fingerprints(smiles_list):
    """
    计算分子的PaDEL指纹
    :param smiles_list: SMILES 字符串列表
    :return: DataFrame，包含所有计算的指纹
    """
    with tempfile.NamedTemporaryFile(mode='w', suffix='.smi', delete=False) as temp_file:
        for smi in smiles_list:
            temp_file.write(f"{smi}\n")
        temp_smi_file = temp_file.name

    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            output_file = os.path.join(temp_dir, "fingerprints.csv")
            
            padeldescriptor(
                mol_dir=temp_smi_file,
                d_file=output_file,
                fingerprints=True,
                descriptortypes=None,
                detectaromaticity=True,
                standardizenitro=True,
                standardizetautomers=True,
                threads=2,
                removesalt=True,
                log=False
            )
            
            df = pd.read_csv(output_file)
            if 'Name' in df.columns:
                df = df.drop('Name', axis=1)
            df.insert(0, 'SMILES', smiles_list)
            return df
    except Exception as e:
        log.error(f"计算PaDEL指纹时发生错误: {str(e)}")
        raise
    finally:
        if os.path.exists(temp_smi_file):
            os.remove(temp_smi_file)


def adjusted_r2(y_true, y_pred, num_features):
    """
    计算调整后的 R^2
    :param y_true: 真实值
    :param y_pred: 预测值
    :param num_features: 模型使用的特征数量
    :return: 调整后的 R^2
    """
    n = len(y_true)
    r2 = r2_score(y_true, y_pred)
    # 避免除零
    if n - num_features - 1 == 0:
        return r2
    return 1 - (1 - r2) * (n - 1) / (n - num_features - 1)


def plot_scatter(y_true_train, y_pred_train, y_true_test, y_pred_test):
    """
    绘制训练集和测试集散点图
    """
    plt.figure(figsize=(12, 6))

    # 训练集
    plt.subplot(1, 2, 1)
    plt.scatter(y_true_train, y_pred_train, color='blue', alpha=0.5, label='Train')
    plt.plot([y_true_train.min(), y_true_train.max()],
             [y_true_train.min(), y_true_train.max()],
             color='red', lw=2)
    plt.title('Train Set: True vs Predicted')
    plt.xlabel('True Values')
    plt.ylabel('Predicted Values')
    plt.legend()

    # 测试集
    plt.subplot(1, 2, 2)
    plt.scatter(y_true_test, y_pred_test, color='green', alpha=0.5, label='Test')
    plt.plot([y_true_test.min(), y_true_test.max()],
             [y_true_test.min(), y_true_test.max()],
             color='red', lw=2)
    plt.title('Test Set: True vs Predicted')
    plt.xlabel('True Values')
    plt.ylabel('Predicted Values')
    plt.legend()

    plt.tight_layout()
    plt.show()


# --------------------------------------------------------------------
# 1. 超参数搜索（不使用交叉验证，采用简单的训练/验证划分，并搜索轮数少）
def objective(trial, X, y):
    param = {
        'C': trial.suggest_float('C', 0.1, 10.0),
        'epsilon': trial.suggest_float('epsilon', 0.01, 1.0),
        'kernel': trial.suggest_categorical('kernel', ['linear', 'poly', 'rbf', 'sigmoid']),
    }
    
    # 使用 80/20 的划分
    X_train_split, X_valid_split, y_train_split, y_valid_split = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = SVR(**param)
    model.fit(X_train_split, y_train_split)
    y_valid_pred = model.predict(X_valid_split)
    rmse = np.sqrt(mean_squared_error(y_valid_split, y_valid_pred))
    return rmse


# --------------------------------------------------------------------
def main():
    logBB_data_file = "../../data/logBB_data/logBB.csv"
    logBB_desc_file = "../../data/logBB_data/logBB_w_desc_fp.csv"
    feature_index_path = "../../data/logBB_data/logBB_w_desc_fp_index2.txt"  # 特征索引文件路径
    log.info("===============启动 SVM 特征筛选及训练工作===============")

    # 检查数据文件
    if not os.path.exists(logBB_data_file):
        log.error("缺失logBB数据集文件")
        raise FileNotFoundError("缺失logBB数据集")

    # 读取数据
    df = pd.read_csv(logBB_data_file, encoding='utf-8')
    df = df.dropna(subset=['logBB']).reset_index(drop=True)

    # 删除 Compound index 列
    if 'Compound index' in df.columns:
        df = df.drop(columns=['Compound index'])
        log.info("已删除 Compound index 列")

    SMILES = df['SMILES']
    log.info(f"读取 {len(SMILES)} 条 SMILES 数据")

    # 生成分子指纹
    df_desc = calculate_padel_fingerprints(SMILES)
    log.info(f"生成了 {df_desc.shape[0]} 条指纹数据，每个指纹有 {df_desc.shape[1]} 位")

    # 合并指纹与原始数据
    df_final = pd.concat([df, df_desc.drop('SMILES', axis=1)], axis=1)
    df_final.to_csv(logBB_desc_file, encoding='utf-8', index=False)
    log.info(f"分子指纹生成并保存在: {logBB_desc_file}")
    log.info(f"生成的数据框形状: {df_final.shape}")

    # 特征选择与数据准备
    X_initial = df_final.drop(['SMILES', 'logBB'], axis=1)
    y = df_final['logBB']

    # 打印特征
    log.info(f"初始特征数量: {X_initial.shape[1]}")
    log.info(f"特征名称: {X_initial.columns.tolist()}")

    # 检查特征索引文件是否存在
    if os.path.exists(feature_index_path):
        log.info("特征索引文件存在，加载已选择的特征")
        desc_index = np.loadtxt(feature_index_path, dtype=int, delimiter=',')
        
        # 确保索引在有效范围内
        valid_indices = desc_index[desc_index < X_initial.shape[1]]
        if len(valid_indices) != len(desc_index):
            log.warning(f"部分特征索引超出范围，将只使用有效索引")
            desc_index = valid_indices
        
        X = X_initial.iloc[:, desc_index]  # 使用有效的索引
        log.info(f"使用特征索引文件，选择的特征数量: {X.shape[1]}")
        log.info(f"选择的特征名称: {X.columns.tolist()}")
    else:
        log.error("特征索引文件不存在，无法加载特征")
        return

    # 数据集划分
    X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
    log.info(f"将数据集按9:1划分为训练验证集({len(X_train_val)})个样本和测试集({len(X_test)})个样本")

    # 在剩余数据中分出验证集 (10%)
    X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.1, random_state=42)
    log.info(f"将训练验证集按9:1划分为训练集({len(X_train)})个样本和验证集({len(X_val)})个样本")

    # 十折交叉验证
    kf = KFold(n_splits=10, shuffle=True, random_state=42)
    metrics_list = []

    for fold_num, (train_index, val_index) in enumerate(kf.split(X)):
        X_train_cv, X_val_cv = X.iloc[train_index], X.iloc[val_index]
        y_train_cv, y_val_cv = y.iloc[train_index], y.iloc[val_index]

        # 超参数搜索
        study = optuna.create_study(direction='minimize')
        study.optimize(lambda trial: objective(trial, X_train_cv, y_train_cv), n_trials=10)
        best_params = study.best_params

        # 训练最终模型
        model_final = SVR(**best_params)
        model_final.fit(X_train_cv, y_train_cv)

        # 预测
        y_val_pred = model_final.predict(X_val_cv)

        # 计算评估指标
        mse_val = mean_squared_error(y_val_cv, y_val_pred)
        rmse_val = np.sqrt(mse_val)
        r2_val = r2_score(y_val_cv, y_val_pred)
        mae_val = mean_absolute_error(y_val_cv, y_val_pred)
        mape_val = mean_absolute_percentage_error(y_val_cv, y_val_pred)
        adj_r2_val = adjusted_r2(y_val_cv, y_val_pred, X_val_cv.shape[1])

        metrics_list.append({
            'fold': fold_num + 1,
            'mse': mse_val,
            'rmse': rmse_val,
            'r2': r2_val,
            'mae': mae_val,
            'mape': mape_val,
            'adj_r2': adj_r2_val
        })

        log.info(f"Fold {fold_num + 1} - MSE: {mse_val:.4f}, RMSE: {rmse_val:.4f}, R2: {r2_val:.4f}, Adjusted R2: {adj_r2_val:.4f}, MAE: {mae_val:.4f}, MAPE: {mape_val:.2f}%")

    # 平均指标
    avg_metrics = {key: np.mean([m[key] for m in metrics_list]) for key in metrics_list[0] if key != 'fold'}
    log.info("10折交叉验证平均指标：")
    log.info(f"MSE: {avg_metrics['mse']:.4f}")
    log.info(f"RMSE: {avg_metrics['rmse']:.4f}")
    log.info(f"R2: {avg_metrics['r2']:.4f}")
    log.info(f"Adjusted R2: {avg_metrics['adj_r2']:.4f}")
    log.info(f"MAE: {avg_metrics['mae']:.4f}")
    log.info(f"MAPE: {avg_metrics['mape']:.2f}%")

    # 测试集评估
    y_test_pred = model_final.predict(X_test)
    mse_test = mean_squared_error(y_test, y_test_pred)
    rmse_test = np.sqrt(mse_test)
    r2_test = r2_score(y_test, y_test_pred)
    mae_test = mean_absolute_error(y_test, y_test_pred)
    mape_test = mean_absolute_percentage_error(y_test, y_test_pred)
    adj_r2_test = adjusted_r2(y_test, y_test_pred, X_test.shape[1])

    log.info("测试集评估指标：")
    log.info(f"MSE: {mse_test:.4f}, RMSE: {rmse_test:.4f}, R2: {r2_test:.4f}, Adjusted R2: {adj_r2_test:.4f}, MAE: {mae_test:.4f}, MAPE: {mape_test:.2f}%")

    # 保存训练集和测试集的预测结果到 CSV 文件
    train_results = pd.DataFrame({
        'True Values': y_train_val,
        'Predicted Values': model_final.predict(X_train_val)
    })
    test_results = pd.DataFrame({
        'True Values': y_test,
        'Predicted Values': y_test_pred
    })
    train_results.to_csv('./result/svm_fp_train_results.csv', index=False)
    test_results.to_csv('./result/svm_fp_test_results.csv', index=False)
    log.info("训练集和测试集的预测结果已保存到 './result/svm_fp_train_results.csv' 和 './result/svm_fp_test_results.csv'")

    # 保存模型
    model_save_path = os.path.join(model_dir, "svm_fp_model.joblib")
    joblib.dump(model_final, model_save_path)
    log.info(f"模型已保存至 {model_save_path}")


if __name__ == '__main__':
    main()

