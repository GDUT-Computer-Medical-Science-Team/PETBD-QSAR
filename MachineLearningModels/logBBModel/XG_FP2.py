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
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import xgboost as xgb
import optuna
import tempfile
from padelpy import padeldescriptor
import joblib
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestRegressor  # 确保导入 RandomForestRegressor

# 设置日志
log = DataLogger(log_file=f"../../data/logBB_data/log/{datetime.now().strftime('%Y%m%d')}.log").getlog(
    "xgboost_feature_selection")

# 计算评估指标的函数
def calculate_metrics(y_train, y_pred_train, y_val, y_pred_val, y_test, y_pred_test, num_features):
    """
    计算模型在训练集、验证集和测试集上的各项评估指标
    """
    metrics = {}
    
    # 训练集指标
    metrics['train_mse'] = mean_squared_error(y_train, y_pred_train)
    metrics['train_rmse'] = np.sqrt(metrics['train_mse'])
    metrics['train_r2'] = r2_score(y_train, y_pred_train)
    metrics['train_mae'] = mean_absolute_error(y_train, y_pred_train)
    metrics['train_mape'] = mean_absolute_percentage_error(y_train, y_pred_train)
    metrics['train_adj_r2'] = adjusted_r2(y_train, y_pred_train, num_features)
    
    # 验证集指标
    metrics['val_mse'] = mean_squared_error(y_val, y_pred_val)
    metrics['val_rmse'] = np.sqrt(metrics['val_mse'])
    metrics['val_r2'] = r2_score(y_val, y_pred_val)
    metrics['val_mae'] = mean_absolute_error(y_val, y_pred_val)
    metrics['val_mape'] = mean_absolute_percentage_error(y_val, y_pred_val)
    metrics['val_adj_r2'] = adjusted_r2(y_val, y_pred_val, num_features)
    
    # 测试集指标
    metrics['test_mse'] = mean_squared_error(y_test, y_pred_test)
    metrics['test_rmse'] = np.sqrt(metrics['test_mse'])
    metrics['test_r2'] = r2_score(y_test, y_pred_test)
    metrics['test_mae'] = mean_absolute_error(y_test, y_pred_test)
    metrics['test_mape'] = mean_absolute_percentage_error(y_test, y_pred_test)
    metrics['test_adj_r2'] = adjusted_r2(y_test, y_pred_test, num_features)

    return metrics

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


def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    non_zero_index = (y_true != 0)
    return np.mean(np.abs((y_true[non_zero_index] - y_pred[non_zero_index]) / y_true[non_zero_index])) * 100


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


def plot_scatter(y_train, y_pred_train, y_test, y_pred_test, save_path=None):
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
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()


def objective(trial, X, y):
    """
    Optuna 的目标函数，用于超参数优化
    """
    param = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 300),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 7),
        'random_state': 42
    }
    
    # 使用 80/20 的划分
    X_train_split, X_valid_split, y_train_split, y_valid_split = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    model = xgb.XGBRegressor(**param)
    model.fit(
        X_train_split, 
        y_train_split,
        eval_set=[(X_valid_split, y_valid_split)],
        early_stopping_rounds=50,
        verbose=False
    )
    
    y_pred = model.predict(X_valid_split)
    rmse = np.sqrt(mean_squared_error(y_valid_split, y_pred))
    return rmse


def main():
    # 定义基础参数
    base_params = {
        'n_estimators': 100,
        'max_depth': 10,
        'learning_rate': 0.1,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'random_state': 42
    }

    # 文件路径设置
    logBB_data_file = "../../data/logBB_data/logBB.csv"
    logBB_desc_file = "../../data/logBB_data/logBB_w_desc_fp.csv"
    feature_index_path = "../../data/logBB_data/logBB_w_desc_fp_index2.txt"  # 特征索引文件路径
    log.info("===============启动 XGBoost 特征筛选及训练工作===============")

    # 检查数据文件
    if not os.path.exists(logBB_data_file):
        log.error("缺失logBB数据集文件")
        raise FileNotFoundError("缺失logBB数据集")

    # 读取数据
    df = pd.read_csv(logBB_data_file, encoding='utf-8')
    df = df.dropna(subset=['logBB']).reset_index(drop=True)
    SMILES = df['SMILES']
    log.info(f"读取 {len(SMILES)} 条 SMILES 数据")

    # 生成分子指纹
    df_desc = calculate_padel_fingerprints(SMILES)
    log.info(f"生成了 {df_desc.shape[0]} 条指纹数据，每个指纹有 {df_desc.shape[1]} 位")

    # 合并指纹与原始数据
    df_final = pd.concat([df, df_desc.drop('SMILES', axis=1)], axis=1)

    log.info(f"生成的数据框形状: {df_final.shape}")

    # 特征选择与数据准备
    X_initial = df_final.drop(['SMILES', 'logBB'], axis=1)


    y = df_final['logBB']
    log.info(f"读取 {len(X_initial)} 个特征")

    # 确保所有特征都是数值类型
    X_initial = X_initial.apply(pd.to_numeric, errors='coerce')  # 将非数值数据转换为 NaN
    X_initial = X_initial.fillna(0)  # 用0填充NaN值

    # 特征筛选
    if not os.path.exists(feature_index_path):
        log.info("不存在特征索引文件，进行特征筛选")
        
        # 使用RFE进行特征选择
        model_rfe = xgb.XGBRegressor()
        rfe = RFE(estimator=model_rfe, n_features_to_select=50)  # 选择50个特征
        rfe.fit(X_initial, y)
        
        # 获取选择的特征索引
        selected_features = X_initial.columns[rfe.support_]
        log.info(f"选择的特征: {selected_features.tolist()}")
        
        # 保存特征索引
        desc_index = np.where(rfe.support_)[0]
        np.savetxt(feature_index_path, desc_index, fmt='%d')
        log.info(f"特征索引已保存至：{feature_index_path}")
        
        X = X_initial[selected_features]  # 使用选择的特征
        log.info(f"筛选后的特征矩阵形状为：{X.shape}")
        log.info(f"选择的特征名称: {X.columns.tolist()}")
    else:
        log.info("存在特征索引文件，进行读取")
        desc_index = np.loadtxt(feature_index_path, dtype=int, delimiter=',').tolist()
        log.info(f"读取特征索引完成，索引内容: {desc_index}")

        # Check if desc_index is valid
        if any(i >= X_initial.shape[1] for i in desc_index):
            log.error("特征索引包含无效的列索引，无法进行筛选。")
            raise IndexError("特征索引包含无效的列索引。")

        X = X_initial.iloc[:, desc_index]  # Use iloc to select columns by index
        log.info(f"筛选后的特征矩阵形状为：{X.shape}")
        log.info(f"选择的特征名称: {X.columns.tolist()}")

    # 数据集划分
    X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
    log.info(f"将数据集按9:1划分为训练验证集({len(X_train_val)}个样本)和测试集({len(X_test)}个样本)")

    # 在剩余数据中分出验证集 (10%)
    X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.1, random_state=42)
    log.info(f"将训练验证集按9:1划分为训练集({len(X_train)}个样本)和验证集({len(X_val)}个样本)")

    # 数据归一化
    sc = MinMaxScaler()
    X_train_scaled = sc.fit_transform(X_train)
    X_val_scaled = sc.transform(X_val)
    X_test_scaled = sc.transform(X_test)

    # 转换回DataFrame以保持列名
    X_train = pd.DataFrame(X_train_scaled, columns=X_train.columns)
    X_val = pd.DataFrame(X_val_scaled, columns=X_val.columns)
    X_test = pd.DataFrame(X_test_scaled, columns=X_test.columns)

    # 在超参数搜索和交叉验证时使用训练验证集
    X = X_train_val
    y = y_train_val
    log.info("使用训练验证集进行超参数优化和交叉验证")

    # 首先进行完整的超参数搜索（100轮）
    log.info("开始进行超参数搜索（100轮）...")
    study = optuna.create_study(direction='minimize')
    study.optimize(lambda trial: objective(trial, X, y), n_trials=100)
    
    best_params = study.best_params
    log.info(f"最优超参数: {best_params}")
    log.info(f"最优 RMSE: {study.best_value:.4f}")

    # 使用找到的最佳参数进行十折交叉验证
    kf = KFold(n_splits=10, shuffle=True, random_state=42)
    metrics_list = []

    for fold_num, (train_index, val_index) in enumerate(kf.split(X)):
        X_train_cv, X_val_cv = X.iloc[train_index], X.iloc[val_index]
        y_train_cv, y_val_cv = y.iloc[train_index], y.iloc[val_index]

        # 使用最佳参数训练模型
        model_final = xgb.XGBRegressor(**best_params)
        model_final.fit(X_train_cv, y_train_cv)

        # 预测
        y_val_pred = model_final.predict(X_val_cv)

        # 计算评估指标
        mse_val = mean_squared_error(y_val_cv, y_val_pred)
        rmse_val = np.sqrt(mse_val)
        r2_val = r2_score(y_val_cv, y_val_pred)
        mae_val = mean_absolute_error(y_val_cv, y_val_pred)
        mape_val = mean_absolute_percentage_error(y_val_cv, y_val_pred)

        metrics_list.append({
            'fold': fold_num + 1,
            'mse': mse_val,
            'rmse': rmse_val,
            'r2': r2_val,
            'mae': mae_val,
            'mape': mape_val
        })

        log.info(f"Fold {fold_num + 1} - MSE: {mse_val:.4f}, RMSE: {rmse_val:.4f}, R2: {r2_val:.4f}, MAE: {mae_val:.4f}, MAPE: {mape_val:.2f}%")

    # 平均指标
    avg_metrics = {key: np.mean([m[key] for m in metrics_list]) for key in metrics_list[0]}
    log.info("10折交叉验证平均指标：")
    log.info(f"MSE: {avg_metrics['mse']:.4f}")
    log.info(f"RMSE: {avg_metrics['rmse']:.4f}")
    log.info(f"R2: {avg_metrics['r2']:.4f}")
    log.info(f"MAE: {avg_metrics['mae']:.4f}")
    log.info(f"MAPE: {avg_metrics['mape']:.2f}%")

    # ----------------------------------------------------------------
    # 4. 修改最终评估部分
    # 训练最终模型
    model_final = xgb.XGBRegressor(**best_params)
    model_final.fit(X_train, y_train,
                    eval_set=[(X_val, y_val)],
                    early_stopping_rounds=50,
                    verbose=False)

    # 在所有数据集上进行预测
    y_pred_train = model_final.predict(X_train)
    y_pred_val = model_final.predict(X_val)
    y_pred_test = model_final.predict(X_test)

    # 计算各个集合的评估指标
    # 训练集指标
    mse_train = mean_squared_error(y_train, y_pred_train)
    rmse_train = np.sqrt(mse_train)
    r2_train = r2_score(y_train, y_pred_train)
    mae_train = mean_absolute_error(y_train, y_pred_train)
    mape_train = mean_absolute_percentage_error(y_train, y_pred_train)
    adj_r2_train = adjusted_r2(y_train, y_pred_train, X_train.shape[1])

    # 验证集指标
    mse_val = mean_squared_error(y_val, y_pred_val)
    rmse_val = np.sqrt(mse_val)
    r2_val = r2_score(y_val, y_pred_val)
    mae_val = mean_absolute_error(y_val, y_pred_val)
    mape_val = mean_absolute_percentage_error(y_val, y_pred_val)
    adj_r2_val = adjusted_r2(y_val, y_pred_val, X_val.shape[1])

    # 测试集指标
    mse_test = mean_squared_error(y_test, y_pred_test)
    rmse_test = np.sqrt(mse_test)
    r2_test = r2_score(y_test, y_pred_test)
    mae_test = mean_absolute_error(y_test, y_pred_test)
    mape_test = mean_absolute_percentage_error(y_test, y_pred_test)
    adj_r2_test = adjusted_r2(y_test, y_pred_test, X_test.shape[1])

    log.info("最终评估指标：")
    log.info(f"测试集 -> MSE: {mse_test:.4f}, RMSE: {rmse_test:.4f}, R2: {r2_test:.4f}, Adjusted R2: {adj_r2_test:.4f}, MAE: {mae_test:.4f}, MAPE: {mape_test:.2f}%")

    # 绘制散点图
    plot_scatter(y_train, y_pred_train, y_test, y_pred_test)

    # 计算并打印单独划分的评估指标
    mse_test = mean_squared_error(y_test, y_pred_test)
    rmse_test = np.sqrt(mse_test)
    r2_test = r2_score(y_test, y_pred_test)
    mae_test = mean_absolute_error(y_test, y_pred_test)
    mape_test = mean_absolute_percentage_error(y_test, y_pred_test)
    adj_r2_test = adjusted_r2(y_test, y_pred_test, X_test.shape[1])

    log.info("最终测试集评估指标：")
    log.info(f"测试集 -> MSE: {mse_test:.4f}, RMSE: {rmse_test:.4f}, R2: {r2_test:.4f}, Adjusted R2: {adj_r2_test:.4f}, MAE: {mae_test:.4f}, MAPE: {mape_test:.2f}%")

    # 保存训练集和测试集的预测结果到 CSV 文件
    train_results = pd.DataFrame({'True Values': y_train, 'Predicted Values': y_pred_train})
    test_results = pd.DataFrame({'True Values': y_test, 'Predicted Values': y_pred_test})
    
    # 创建 result 目录（如果不存在）
    os.makedirs('./result', exist_ok=True)
    
    train_results.to_csv('./result/xgb_fp_train_results.csv', index=False)
    test_results.to_csv('./result/xgb_fp_test_results.csv', index=False)
    log.info("训练集和测试集的预测结果已保存到 './result/xgb_fp_train_results.csv' 和 './result/xgb_fp_test_results.csv'")

    # 定义模型保存目录
    model_dir = "logbbModel"
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    # 定义模型保存路径
    model_save_path = os.path.join(model_dir, "xgb_fp_model.joblib")

    # 保存模型
    joblib.dump(model_final, model_save_path)
    log.info(f"模型已保存到: {model_save_path}")

    # 加载模型
    loaded_model = joblib.load(model_save_path)
    # 进行预测
    y_pred_test_loaded = loaded_model.predict(X_test)

    # 验证加载的模型与原始模型是否一致
    assert np.allclose(y_pred_test, y_pred_test_loaded), "加载的模型与原始模型预测结果不一致！"
    log.info("模型加载成功，并且预测结果一致！")


if __name__ == '__main__':
    main()