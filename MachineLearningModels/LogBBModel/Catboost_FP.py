import joblib
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
from catboost import CatBoostRegressor
import optuna
import tempfile
from padelpy import padeldescriptor
from tqdm import tqdm
from preprocess.data_preprocess.FeatureExtraction import FeatureExtraction
from sklearn.feature_selection import RFE  # Import RFE
from sklearn.ensemble import RandomForestRegressor

# 设置日志
log = DataLogger(log_file=f"../../data/logBB_data/log/{datetime.now().strftime('%Y%m%d')}.log").getlog(
    "catboost_feature_selection")


def calculate_padel_fingerprints(smiles_list):
    """
    计算分子的PaDEL指纹
    :param smiles_list: SMILES 字符串列表
    :return: DataFrame，包含所有计算的指纹
    """
    # 创建临时文件来存储SMILES
    with tempfile.NamedTemporaryFile(mode='w', suffix='.smi', delete=False) as temp_file:
        for smi in smiles_list:
            temp_file.write(f"{smi}\n")
        temp_smi_file = temp_file.name

    try:
        # 创建临时目录存储结果
        with tempfile.TemporaryDirectory() as temp_dir:
            output_file = os.path.join(temp_dir, "fingerprints.csv")
            
            # 直接计算所有可用的指纹
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
            # 读取结果
            df = pd.read_csv(output_file)
            # 删除Name列（如果存在）
            if 'Name' in df.columns:
                df = df.drop('Name', axis=1)
            # 添加SMILES列
            df.insert(0, 'SMILES', smiles_list)
            return df
    except Exception as e:
        log.error(f"计算PaDEL指纹时发生错误: {str(e)}")
        raise
    finally:
        # 清理临时文件
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


# --------------------------------------------------------------------
# 1. 超参数搜索（不使用交叉验证，采用简单的训练/验证划分，并搜索轮数少）
def objective(trial, X_train, y_train):
    param = {
        'iterations': trial.suggest_int('iterations', 100, 1000),
        'depth': trial.suggest_int('depth', 4, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1, 10),
        'random_seed': 42,
        'verbose': False
    }
    
    # Split data for validation
    X_t, X_v, y_t, y_v = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
    
    # Train model
    model = CatBoostRegressor(**param)
    model.fit(X_t, y_t, eval_set=(X_v, y_v), early_stopping_rounds=50, verbose=False)
    
    # Predict and calculate RMSE
    y_pred = model.predict(X_v)
    rmse = np.sqrt(mean_squared_error(y_v, y_pred))
    
    return rmse  # Return RMSE as the objective to minimize


# --------------------------------------------------------------------
def main():
    # 文件路径设置
    logBB_data_file = "../../data/logBB_data/logBB.csv"
    logBB_desc_file = "../../data/logBB_data/logBB_w_desc_fp.csv"
    logBB_desc_index_file = "../../data/logBB_data/logBB_w_desc_fp_index2.txt"
    log.info("===============启动 CatBoost 特征筛选及训练工作===============")

    # 检查数据文件
    if not os.path.exists(logBB_data_file):
        log.error("缺失logBB数据集文件")
        raise FileNotFoundError("缺失logBB数据集")

    # 读取数据
    df = pd.read_csv(logBB_data_file, encoding='utf-8')
    df = df.dropna(subset=['logBB']).reset_index(drop=True)
    
    # 首先删除 Compound index 列
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
    log.info(f"部分生成的指纹数据：\n{df_final.head()}")

    # 特征选择与数据准备
    X = df_final.drop(['SMILES', 'logBB'], axis=1)

    # 在特征选择之前排除 blood mean60min
    if 'blood mean60min' in X.columns:
        X = X.drop('blood mean60min', axis=1)
        log.info("已排除特征: blood mean60min")

    # 确保所有特征都是数值类型
    X = X.apply(pd.to_numeric, errors='coerce')  # 将非数值数据转换为 NaN
    X = X.fillna(0)  # 用0填充NaN值

    y = df_final['logBB']
    log.info(f"读取 {len(X)} 条特征数据")

    # 检查特征索引文件是否存在
    if os.path.exists(logBB_desc_index_file):
        log.info("特征索引文件存在，加载已选择的特征")
        desc_index = np.loadtxt(logBB_desc_index_file, dtype=int, delimiter=',')
        
        # 确保索引在有效范围内
        valid_indices = desc_index[desc_index < X.shape[1]]
        if len(valid_indices) != len(desc_index):
            log.warning(f"部分特征索引超出范围，将只使用有效索引")
            desc_index = valid_indices
        
        X = X.iloc[:, desc_index]  # 使用有效的索引
        log.info(f"使用特征索引文件，选择的特征数量: {X.shape[1]}")
        log.info(f"选择的特征名称: {X.columns.tolist()}")
    else:
        # 使用随机森林进行特征选择
        model_rfe = RandomForestRegressor(n_estimators=100, random_state=42)
        rfe = RFE(estimator=model_rfe, n_features_to_select=50)
        rfe.fit(X, y)

        # 获取选择的特征索引
        selected_features = X.columns[rfe.support_]
        log.info(f"选择的特征: {selected_features.tolist()}")

        # 保存特征索引
        desc_index = np.where(rfe.support_)[0]
        np.savetxt(logBB_desc_index_file, desc_index, fmt='%d')
        log.info(f"特征索引已保存至: {logBB_desc_index_file}")

        X = X[selected_features]

    # 数据集划分
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
    log.info(f"将数据集按9:1划分为训练集({len(X_train)}个样本)和测试集({len(X_test)}个样本)")

    # 十折交叉验证
    kf = KFold(n_splits=10, shuffle=True, random_state=42)
    metrics_list = []

    for fold_num, (train_index, val_index) in enumerate(kf.split(X_train)):
        X_fold_train, X_fold_val = X_train.iloc[train_index], X_train.iloc[val_index]
        y_fold_train, y_fold_val = y_train.iloc[train_index], y_train.iloc[val_index]

        # 超参数优化
        study = optuna.create_study(direction='minimize')
        study.optimize(lambda trial: objective(trial, X_fold_train, y_fold_train), n_trials=10)
        best_params = study.best_params

        # 训练最终模型
        model_fold = CatBoostRegressor(**best_params, verbose=0)
        model_fold.fit(X_fold_train, y_fold_train)

        # 预测
        y_fold_val_pred = model_fold.predict(X_fold_val)

        # 计算评估指标
        mse_val = mean_squared_error(y_fold_val, y_fold_val_pred)
        rmse_val = np.sqrt(mse_val)
        r2_val = r2_score(y_fold_val, y_fold_val_pred)
        mae_val = mean_absolute_error(y_fold_val, y_fold_val_pred)
        mape_val = mean_absolute_percentage_error(y_fold_val, y_fold_val_pred)
        adj_r2_val = adjusted_r2(y_fold_val, y_fold_val_pred, X_fold_val.shape[1])

        metrics_list.append({
            'fold': fold_num + 1,
            'mse': mse_val,
            'rmse': rmse_val,
            'r2': r2_val,
            'mae': mae_val,
            'mape': mape_val,
            'adj_r2': adj_r2_val
        })

        log.info(f"Fold {fold_num + 1} - MSE: {mse_val:.4f}, RMSE: {rmse_val:.4f}, R2: {r2_val:.4f}, MAE: {mae_val:.4f}, MAPE: {mape_val:.2f}%, Adjusted R2: {adj_r2_val:.4f}")

    # 平均指标
    avg_metrics = {key: np.mean([m[key] for m in metrics_list]) for key in metrics_list[0] if key != 'fold'}
    log.info("10折交叉验证平均指标：")
    log.info(f"MSE: {avg_metrics['mse']:.4f}")
    log.info(f"RMSE: {avg_metrics['rmse']:.4f}")
    log.info(f"R2: {avg_metrics['r2']:.4f}")
    log.info(f"MAE: {avg_metrics['mae']:.4f}")
    log.info(f"MAPE: {avg_metrics['mape']:.2f}%")
    log.info(f"Adjusted R2: {avg_metrics['adj_r2']:.4f}")

    # 使用最佳参数在整个训练集上训练最终模型
    study = optuna.create_study(direction='minimize')
    study.optimize(lambda trial: objective(trial, X_train, y_train), n_trials=10)
    best_params = study.best_params

    model_final = CatBoostRegressor(**best_params, verbose=0)
    model_final.fit(X_train, y_train)

    # 在测试集上评估最终模型
    y_pred_test = model_final.predict(X_test)
    test_metrics = {
        'test_mse': mean_squared_error(y_test, y_pred_test),
        'test_rmse': np.sqrt(mean_squared_error(y_test, y_pred_test)),
        'test_r2': r2_score(y_test, y_pred_test),
        'test_mae': mean_absolute_error(y_test, y_pred_test),
        'test_mape': mean_absolute_percentage_error(y_test, y_pred_test),
        'test_adj_r2': adjusted_r2(y_test, y_pred_test, X_test.shape[1])
    }

    # 输出测试集评估结果
    log.info("\n测试集评估指标：")
    log.info(f"MSE: {test_metrics['test_mse']:.4f}")
    log.info(f"RMSE: {test_metrics['test_rmse']:.4f}")
    log.info(f"R²: {test_metrics['test_r2']:.4f}")
    log.info(f"Adjusted R²: {test_metrics['test_adj_r2']:.4f}")
    log.info(f"MAE: {test_metrics['test_mae']:.4f}")
    log.info(f"MAPE: {test_metrics['test_mape']:.2f}%")

    # 保存测试集结果
    test_results = pd.DataFrame({
        'True Values': y_test,
        'Predicted Values': y_pred_test
    })

    os.makedirs('./result', exist_ok=True)
    test_results.to_csv('./result/catboost_fp_test_results.csv', index=False)

    # 定义模型保存目录
    model_dir = "logbbModel"
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    # 定义模型保存路径
    model_path = os.path.join(model_dir, "catboost_fp_model.joblib")

    # 保存模型
    joblib.dump(model_final, model_path)
    log.info(f"模型已保存到: {model_path}")

    # 加载模型
    loaded_model = joblib.load(model_path)
    # 进行预测 - 使用测试集
    y_pred_test_loaded = loaded_model.predict(X_test)

    # 验证加载的模型与原始模型是否一致
    original_pred = model_final.predict(X_test)
    assert np.allclose(original_pred, y_pred_test_loaded), "加载的模型与原始模型预测结果不一致！"
    log.info("模型加载成功，并且预测结果一致！")

    log.info("\n最终数据集划分:")
    log.info(f"训练集: {len(X_train)}个样本 ({len(X_train)/len(X)*100:.1f}%)")
    log.info(f"测试集: {len(X_test)}个样本 ({len(X_test)/len(X)*100:.1f}%)")

    # 绘制散点图
    plot_scatter(y_train, model_final.predict(X_train), y_test, y_pred_test, save_path='./result/catboost_fp_scatter.png')
    log.info("训练集和测试集散点图已保存到 './result/catboost_fp_scatter.png'")

    # 保存训练集和测试集的预测结果到 CSV 文件
    train_results = pd.DataFrame({
        'True Values': y_train,
        'Predicted Values': model_final.predict(X_train)
    })
    train_results.to_csv('./result/catboost_fp_train_results.csv', index=False)
    test_results.to_csv('./result/catboost_fp_test_results.csv', index=False)
    log.info("训练集和测试集的预测结果已保存到 './result/catboost_fp_train_results.csv' 和 './result/catboost_fp_test_results.csv'")


if __name__ == '__main__':
    main()

