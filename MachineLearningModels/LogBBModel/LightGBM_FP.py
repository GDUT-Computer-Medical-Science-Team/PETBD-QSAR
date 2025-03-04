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
import lightgbm as lgb
import optuna
from padelpy import padeldescriptor
import tempfile
import joblib
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression  # 可以使用其他模型
from tqdm import tqdm
import random

# 设置日志
log = DataLogger(log_file=f"../../data/logBB_data/log/{datetime.now().strftime('%Y%m%d')}.log").getlog(
    "lightgbm_feature_selection")

# 在文件开头添加随机种子设置
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

def calculate_morgan_fingerprints(smiles_list, radius=2, n_bits=1024):
    """
    计算分子的Morgan指纹
    :param smiles_list: SMILES 字符串列表
    :param radius: 指纹半径，默认为2
    :param n_bits: 指纹位数，默认为1024
    :return: numpy 数组，每一行是对应分子的指纹
    """
    fingerprints = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            log.warning(f"无效的 SMILES: {smi}")
            continue
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
        arr = np.zeros((1,), dtype=int)
        DataStructs.ConvertToNumpyArray(fp, arr)
        fingerprints.append(arr)
    return np.array(fingerprints)


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
                fingerprints=True,  # 启用指纹计算
                descriptortypes=None,  # 不指定XML文件，使用默认设置
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
        log.error(f"计算PaDEL指纹时出错: {str(e)}")
        raise
    
    finally:
        # 清理临时文件
        if os.path.exists(temp_smi_file):
            os.remove(temp_smi_file)


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


def mean_absolute_percentage_error(y_true, y_pred):
    """
    计算平均绝对百分比误差
    """
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    non_zero_index = (y_true != 0)
    return np.mean(np.abs((y_true[non_zero_index] - y_pred[non_zero_index]) / y_true[non_zero_index])) * 100


def adjusted_r2(y_true, y_pred, num_features):
    """
    计算调整后的 R²
    :param y_true: 真实值
    :param y_pred: 预测值
    :param num_features: 模型使用的特征数量
    :return: 调整后的 R²
    """
    n = len(y_true)
    r2 = r2_score(y_true, y_pred)
    # 当样本数量小于或等于特征数量加1时，返回原始R²
    if n <= num_features + 1:
        return r2
    return 1 - ((1 - r2) * (n - 1) / (n - num_features - 1))


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
def objective(trial, X, y):
    param = {
        'num_leaves': trial.suggest_int('num_leaves', 20, 3000),
        'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.1),
        'n_estimators': trial.suggest_int('n_estimators', 100, 3000),
        'max_depth': trial.suggest_int('max_depth', 3, 30),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
        'min_split_gain': trial.suggest_float('min_split_gain', 0.0, 1.0),
        'max_bin': trial.suggest_int('max_bin', 200, 300),
        'boosting_type': trial.suggest_categorical('boosting_type', ['gbdt', 'dart']),
        'random_state': RANDOM_SEED,
        'objective': 'regression'
    }

    # 使用简单的训练/验证划分进行评估
    X_train_split, X_valid_split, y_train_split, y_valid_split = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_SEED
    )

    # 训练模型
    model = lgb.LGBMRegressor(**param)
    model.fit(X_train_split, y_train_split, 
              eval_set=[(X_valid_split, y_valid_split)], 
              callbacks=[lgb.early_stopping(stopping_rounds=50)])

    # 预测并计算评估指标
    y_pred = model.predict(X_valid_split)
    rmse = np.sqrt(mean_squared_error(y_valid_split, y_pred))

    return rmse  # 返回 RMSE 作为优化目标


def feature_selection(X, y, n_features=50):
    """
    使用 RFE 进行特征选择
    :param X: 特征矩阵
    :param y: 目标变量
    :param n_features: 要选择的特征数量
    :return: 选择的特征索引列表
    """
    model = lgb.LGBMRegressor()  # 也可以使用其他模型，如 LogisticRegression
    rfe = RFE(estimator=model, n_features_to_select=n_features)
    
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=RANDOM_SEED)
    rfe.fit(X_train, y_train)
    selected_indices = np.where(rfe.support_)[0]

    log.info(f"选择的特征索引: {selected_indices.tolist()}")
    log.info(f"选择的特征数量: {len(selected_indices)}")
    
    return selected_indices.tolist()


def evaluate_model(X, y, selected_indices):
    """
    评估模型性能
    :param X: 特征矩阵
    :param y: 目标变量
    :param selected_indices: 选择的特征索引
    :return: 评估指标
    """
    X_selected = X.iloc[:, selected_indices]  # 选择特征
    X_train, X_val, y_train, y_val = train_test_split(X_selected, y, test_size=0.2, random_state=RANDOM_SEED)

    model = lgb.LGBMRegressor()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_val)

    mse = mean_squared_error(y_val, y_pred)
    r2 = r2_score(y_val, y_pred)
    mae = mean_absolute_error(y_val, y_pred)

    return mse, r2, mae


def train_lightgbm_with_selected_features(X, y, selected_indices):
    """
    使用筛选后的特征训练 LightGBM 模型
    :param X: 特征矩阵
    :param y: 目标变量
    :param selected_indices: 选择的特征索引
    :return: 训练好的模型和评估指标
    """
    # 选择特征
    X_selected = X.iloc[:, selected_indices]

    # 拆分数据集
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X_selected, y, test_size=0.1, random_state=RANDOM_SEED
    )
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=0.1, random_state=RANDOM_SEED
    )

    # 超参数搜索
    study = optuna.create_study(direction='minimize')
    study.optimize(lambda trial: objective(trial, X_train, y_train), n_trials=100)
    
    best_params = study.best_params
    
    # 训练最终模型
    model = lgb.LGBMRegressor(**best_params)
    model.fit(X_train, y_train,
              eval_set=[(X_val, y_val)],
              callbacks=[lgb.early_stopping(stopping_rounds=50)])

    # 预测并计算评估指标
    y_pred_train = model.predict(X_train)
    y_pred_val = model.predict(X_val)
    y_pred_test = model.predict(X_test)

    # 计算评估指标
    metrics = calculate_metrics(y_train, y_pred_train, y_val, y_pred_val, 
                                y_test, y_pred_test, X_train.shape[1])

    return model, metrics, (X_train, X_val, X_test)


# --------------------------------------------------------------------
def main():
    # 文件路径设置
    logBB_data_file = "../../data/logBB_data/logBB.csv"
    logBB_desc_file = "../../data/logBB_data/logBB_w_desc_fp.csv"
    feature_index_path = "../../data/logBB_data/logBB_w_desc_fp_index2.txt"  # 特征索引文件路径
    log.info("===============启动 LightGBM 特征筛选及训练工作===============")

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

    # 定义目标变量
    y = df_final['logBB']  # 确保 y 被定义

    # 特征归一化
    sc = MinMaxScaler()
    X = pd.DataFrame(sc.fit_transform(X), columns=X.columns)

    # 特征筛选
    if not os.path.exists(feature_index_path):
        log.info("不存在特征索引文件，进行特征筛选")
        
        # 使用RFE进行特征选择
        model_rfe = lgb.LGBMRegressor()
        rfe = RFE(estimator=model_rfe, n_features_to_select=50)  # 选择50个特征
        rfe.fit(X, y)
        
        # 获取选择的特征索引
        selected_features = X.columns[rfe.support_]
        log.info(f"选择的特征: {selected_features.tolist()}")
        
        # 保存特征索引
        desc_index = np.where(rfe.support_)[0]
        np.savetxt(feature_index_path, desc_index, fmt='%d')
        log.info(f"特征索引已保存至：{feature_index_path}")
        
        X = X[selected_features]  # 使用选择的特征
        log.info(f"筛选后的特征矩阵形状为：{X.shape}")
        log.info(f"选择的特征名称: {X.columns.tolist()}")
    else:
        log.info("存在特征索引文件，进行读取")
        desc_index = np.loadtxt(feature_index_path, dtype=int, delimiter=',').tolist()
        log.info(f"读取特征索引完成，索引内容: {desc_index}")

        # Check if desc_index is valid
        if any(i >= X.shape[1] for i in desc_index):
            log.error("特征索引包含无效的列索引，无法进行筛选。")
            raise IndexError("特征索引包含无效的列索引。")

        X = X.iloc[:, desc_index]  # Use iloc to select columns by index
        log.info(f"筛选后的特征矩阵形状为：{X.shape}")
        log.info(f"选择的特征名称: {X.columns.tolist()}")

    # 数据集划分
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=0.1, random_state=RANDOM_SEED
    )
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=0.1, random_state=RANDOM_SEED
    )
    log.info(f"将数据集按9:1划分为训练验证集({len(X_train_val)}个样本)和测试集({len(X_test)}个样本)")

    # 首先进行完整的超参数搜索（100轮）
    log.info("开始进行超参数搜索（100轮）...")
    study = optuna.create_study(direction='minimize')
    study.optimize(lambda trial: objective(trial, X, y), n_trials=100)
    
    best_params = study.best_params
    log.info(f"最优超参数: {best_params}")
    log.info(f"最优 RMSE: {study.best_value:.4f}")

    # 使用找到的最佳参数进行十折交叉验证
    kf = KFold(n_splits=10, shuffle=True, random_state=RANDOM_SEED)
    metrics_list = []

    for fold_num, (train_index, val_index) in enumerate(kf.split(X_train_val)):
        X_train_cv, X_val_cv = X_train_val.iloc[train_index], X_train_val.iloc[val_index]
        y_train_cv, y_val_cv = y_train_val.iloc[train_index], y_train_val.iloc[val_index]

        # 使用最佳参数训练模型
        model_final = lgb.LGBMRegressor(**best_params)
        model_final.fit(X_train_cv, y_train_cv)

        # 预测
        y_val_pred = model_final.predict(X_val_cv)

        # 计算评估指标
        mse_val = mean_squared_error(y_val_cv, y_val_pred)
        rmse_val = np.sqrt(mse_val)
        r2_val = r2_score(y_val_cv, y_val_pred)
        mae_val = mean_absolute_error(y_val_cv, y_val_pred)
        mape_val = mean_absolute_percentage_error(y_val_cv, y_val_pred)

        # 计算调整后 R²
        adj_r2 = adjusted_r2(y_val_cv, y_val_pred, X_val_cv.shape[1])

        # 计算评估指标
        metrics = {
            'fold': fold_num + 1,
            'mse': mse_val,
            'rmse': rmse_val,
            'r2': r2_val,
            'mae': mae_val,
            'mape': mape_val,
            'adj_r2': adj_r2
        }
        metrics_list.append(metrics)

        log.info(f"Fold {fold_num + 1}:")
        log.info(f"MSE: {mse_val:.4f}, RMSE: {rmse_val:.4f}, R²: {r2_val:.4f}, Adj R²: {adj_r2:.4f}, MAE: {mae_val:.4f}, MAPE: {mape_val:.2f}%")

    # 平均指标
    avg_metrics = {key: np.mean([m[key] for m in metrics_list]) for key in metrics_list[0] if key != 'fold'}
    log.info("\n10折交叉验证平均指标：")
    log.info(f"MSE: {avg_metrics['mse']:.4f}")
    log.info(f"RMSE: {avg_metrics['rmse']:.4f}")
    log.info(f"R²: {avg_metrics['r2']:.4f}")
    log.info(f"Adjusted R²: {avg_metrics['adj_r2']:.4f}")
    log.info(f"MAE: {avg_metrics['mae']:.4f}")
    log.info(f"MAPE: {avg_metrics['mape']:.2f}%")

    # 测试集评估
    y_pred_test = model_final.predict(X_test)
    test_metrics = {
        'test_mse': mean_squared_error(y_test, y_pred_test),
        'test_rmse': np.sqrt(mean_squared_error(y_test, y_pred_test)),
        'test_r2': r2_score(y_test, y_pred_test),
        'test_mae': mean_absolute_error(y_test, y_pred_test),
        'test_mape': mean_absolute_percentage_error(y_test, y_pred_test),
        'test_adj_r2': adjusted_r2(y_test, y_pred_test, X_test.shape[1])
    }

    log.info("\n测试集评估指标：")
    log.info(f"MSE: {test_metrics['test_mse']:.4f}")
    log.info(f"RMSE: {test_metrics['test_rmse']:.4f}")
    log.info(f"R²: {test_metrics['test_r2']:.4f}")
    log.info(f"Adjusted R²: {test_metrics['test_adj_r2']:.4f}")
    log.info(f"MAE: {test_metrics['test_mae']:.4f}")
    log.info(f"MAPE: {test_metrics['test_mape']:.2f}%")

    # After predicting and before creating the DataFrame
    y_train_val_pred = model_final.predict(X_train_val)
    y_test_pred = model_final.predict(X_test)

    # Check lengths
    print(f"Length of SMILES: {len(SMILES[X_train_val.index].values)}")
    print(f"Length of Experimental logBB: {len(y_train_val.values)}")
    print(f"Length of Predicted logBB: {len(y_train_val_pred)}")

    # Create DataFrame for training results
    train_results = pd.DataFrame({
        'SMILES': SMILES[X_train_val.index].values,  # 确保使用 .values
        'Experimental logBB': y_train_val.values,  # 确保使用 .values
        'Predicted logBB': y_train_val_pred  # 确保使用 .values
    })

    # Create DataFrame for test results
    test_results = pd.DataFrame({
        'SMILES': SMILES[X_test.index].values,  # 使用 X_test.index 来获取正确的索引
        'Experimental logBB': y_test.values,  # 确保使用 .values
        'Predicted logBB': y_pred_test  # 确保使用 .values
    })

    # Save results to CSV
    train_results.to_csv('./result/lightgbm_fp_train_results.csv', index=False)
    test_results.to_csv('./result/lightgbm_fp_test_results.csv', index=False)
    log.info("训练集和测试集的预测结果已保存到 './result/lightgbm_fp_train_results.csv' 和 './result/lightgbm_fp_test_results.csv'")

    # 定义模型保存路径
    model_dir = "logbbModel"
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    model_path = os.path.join(model_dir, "lightgbm_fp_model.joblib")
    joblib.dump(model_final, model_path)
    log.info(f"模型已保存到: {model_path}")

    # 加载模型并验证
    loaded_model = joblib.load(model_path)
    y_pred_test_loaded = loaded_model.predict(X_test)

    # 验证加载的模型预测结果是否与原模型一致
    assert np.allclose(y_pred_test, y_pred_test_loaded), "加载的模型与原始模型预测结果不一致！"
    log.info("模型加载验证成功：预测结果与原模型一致")

if __name__ == '__main__':
    main()
