import sys
import os
from time import time
import numpy as np
import pandas as pd
import optuna
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, mean_absolute_percentage_error
from sklearn.model_selection import train_test_split, KFold
from tqdm import tqdm
from datetime import datetime
import joblib
import json
import matplotlib.pyplot as plt
import re
from typing import Optional
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs
from padelpy import padeldescriptor
import tempfile

# 添加项目路径到Python路径
sys.path.append('../../')
from preprocess.data_preprocess.FeatureExtraction import FeatureExtraction
from preprocess.data_preprocess.data_preprocess_utils import calculate_Mordred_desc
from utils.DataLogger import DataLogger

# 创建必要的目录
os.makedirs("../../data/logBB_data/log", exist_ok=True)
os.makedirs("../../data/logBB_data", exist_ok=True)
os.makedirs("./cbrainmodel", exist_ok=True)
os.makedirs("./result", exist_ok=True)

# 初始化日志
log = DataLogger(log_file=f"../../data/logBB_data/log/{datetime.now().strftime('%Y%m%d')}.log").getlog("cbrain_svm_18F_training")

# 允许的同位素列表（用于18F检测）
ALLOWED_ISOTOPES = {"18F", "11C", "125I", "131I", "123I", "77Br", "76Br"}

def extract_18F_simple(compound_index: str) -> Optional[str]:
    """
    从化合物索引中提取同位素信息（修正版本）
    简化为只识别18F，更直接高效
    """
    if not isinstance(compound_index, str):
        return None
    
    text = compound_index.strip()
    
    # 检查是否包含18F
    if "18F" in text:
        return "18F"
    else:
        return "non-18F"  # 修复：明确标识为非18F

def balance_18F_dataset(df: pd.DataFrame, method: str = "oversample", seed: int = 42):
    """
    基于18F标记平衡数据集 - 原始数据+稀少类别重采样策略
    1. 保留所有原始数据
    2. 只对稀少的类别（非18F）进行重采样补充
    3. 合并原始数据与重采样的稀少类别数据
    """
    log.info(f"原始数据样本数: {len(df)}")
    
    # 1. 保留完整的原始数据
    df_original = df.copy()
    
    # 2. 提取同位素信息
    isotopes = [extract_18F_simple(x) for x in df["compound index"].fillna("")]
    df_work = df.copy()
    df_work["isotope"] = isotopes
    
    # 只保留能够识别同位素的行用于分析
    df_isotope = df_work[df_work["isotope"].notna()].reset_index(drop=True)
    
    if len(df_isotope) == 0:
        log.warning("没有可识别的同位素样本，返回原始数据")
        return df_original
    
    # 创建18F二进制标签
    df_isotope["label_18F"] = (df_isotope["isotope"] == "18F").astype(int)
    
    pos = df_isotope[df_isotope["label_18F"] == 1]  # 18F样本
    neg = df_isotope[df_isotope["label_18F"] == 0]  # 非18F样本
    
    n_pos, n_neg = len(pos), len(neg)
    log.info(f"同位素分布 - 18F样本: {n_pos}, 非18F样本: {n_neg}")
    
    if n_pos == 0 or n_neg == 0:
        log.warning("某一类别样本数为0，返回原始数据")
        return df_original
    
    # 3. 确定需要重采样的稀少类别
    if n_pos > n_neg:
        # 非18F是稀少类别，需要重采样
        minority_samples = neg
        target_size = n_pos
        minority_type = "非18F"
    else:
        # 18F是稀少类别，需要重采样  
        minority_samples = pos
        target_size = n_neg
        minority_type = "18F"
    
    # 计算需要额外生成的样本数
    current_size = len(minority_samples)
    extra_needed = target_size - current_size
    
    if extra_needed <= 0:
        log.info("数据已平衡，返回原始数据")
        return df_original
    
    # 4. 对稀少类别进行重采样
    extra_samples = minority_samples.sample(n=extra_needed, replace=True, random_state=seed)
    extra_clean = extra_samples.drop(['isotope', 'label_18F'], axis=1, errors='ignore')
    
    log.info(f"为{minority_type}类别生成额外样本: {extra_needed}个")
    
    # 5. 将原始数据与额外样本合并
    combined_df = pd.concat([df_original, extra_clean], ignore_index=True)
    combined_df = combined_df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    
    # 添加同位素标签到最终数据集
    combined_isotopes = [extract_18F_simple(x) for x in combined_df["compound index"].fillna("")]
    combined_df["isotope"] = combined_isotopes
    combined_df["label_18F"] = (combined_df["isotope"] == "18F").astype(int)
    
    log.info(f"最终组合数据: {len(combined_df)} 样本")
    log.info(f"  - 原始数据: {len(df_original)} 样本")
    log.info(f"  - 额外{minority_type}样本: {extra_needed} 样本")
    
    return combined_df

def calculate_morgan_fingerprints(smiles_list, radius=2, n_bits=1024):
    """
    计算分子的Morgan指纹
    """
    fingerprints = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            log.warning(f"无效的 SMILES: {smi}")
            fingerprints.append(np.zeros(n_bits))
            continue
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
        arr = np.zeros((n_bits,))
        DataStructs.ConvertToNumpyArray(fp, arr)
        fingerprints.append(arr)
    
    # 转换为DataFrame
    fp_df = pd.DataFrame(fingerprints)
    fp_df.columns = [f'Morgan_FP_{i}' for i in range(n_bits)]
    return fp_df

def calculate_padel_fingerprints(smiles_list):
    """
    计算分子的PaDEL指纹
    """
    with tempfile.NamedTemporaryFile(mode='w', suffix='.smi', delete=False) as temp_file:
        for smi in smiles_list:
            temp_file.write(f"{smi}\\n")
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
            
            return df
    except Exception as e:
        log.error(f"计算PaDEL指纹时发生错误: {str(e)}")
        return pd.DataFrame()
    finally:
        if os.path.exists(temp_smi_file):
            os.remove(temp_smi_file)

def adjusted_r2_score(r2, n, k):
    """
    计算调整后的R²值
    """
    return 1 - (1 - r2) * (n - 1) / (n - k - 1)

if __name__ == '__main__':
    log.info("===============启动Cbrain数据集SVM 18F重采样训练===============")
    
    # 文件路径配置 - 使用PETBD数据集
    cbrain_data_file = "../../data/PTBD_v20240912.csv"
    cbrain_features_file = "../../data/logBB_data/cbrain_svm_petbd_18F_balanced_features.csv"
    feature_index_file = "../../data/logBB_data/cbrain_svm_petbd_18F_feature_index.txt"
    
    # 模型参数
    smile_column_name = 'SMILES'
    pred_column_name = 'brain at60min'
    RFE_features_to_select = 50
    n_optuna_trial = 50  # SVM训练较慢，减少试验次数
    cv_times = 5         # 减少折数
    seed = int(time())
    
    # 读取Cbrain数据
    if not os.path.exists(cbrain_data_file):
        raise FileNotFoundError(f"缺失PTBD数据集: {cbrain_data_file}")
    
    log.info("读取PTBD数据集")
    df = pd.read_csv(cbrain_data_file, encoding='utf-8')
    log.info(f"原始数据形状: {df.shape}")
    
    # 删除brain at60min缺失的行
    df = df.dropna(subset=[pred_column_name])
    log.info(f"删除{pred_column_name}缺失值后数据形状: {df.shape}")
    
    # 进行18F重采样平衡
    log.info("开始18F重采样平衡处理")
    df_balanced = balance_18F_dataset(df, method="oversample", seed=seed)
    log.info(f"18F重采样后数据形状: {df_balanced.shape}")
    
    # 特征提取或读取
    if os.path.exists(cbrain_features_file):
        log.info("存在特征文件，进行读取")
        features_df = pd.read_csv(cbrain_features_file, encoding='utf-8')
        
        # 确保特征文件与平衡后的数据对应
        if len(features_df) != len(df_balanced):
            log.info("特征文件与平衡数据不匹配，重新生成特征")
            # 删除旧的特征文件，重新生成
            os.remove(cbrain_features_file)
            if os.path.exists(feature_index_file):
                os.remove(feature_index_file)
        else:
            y = features_df[pred_column_name]
            X = features_df.drop([smile_column_name, pred_column_name], axis=1)
            # 如果有标签列，也要删除
            label_cols = ['isotope', 'label_18F', 'compound index']
            for col in label_cols:
                if col in X.columns:
                    X = X.drop(col, axis=1)
                
    else:
        log.info("特征文件不存在，开始特征生成工作")
        
        y = df_balanced[pred_column_name]
        SMILES = df_balanced[smile_column_name]
        
        log.info("计算Mordred分子描述符")
        X_mordred = calculate_Mordred_desc(SMILES)
        
        log.info("仅使用Mordred描述符")
        X = X_mordred
        
        # 删除SMILES列如果存在
        if smile_column_name in X.columns:
            X = X.drop(smile_column_name, axis=1)
        
        log.info(f"特征矩阵形状: {X.shape}")
        
        # 保存特征数据
        # 确保df_balanced包含isotope和label_18F列
        if 'isotope' not in df_balanced.columns:
            isotopes = [extract_18F_simple(x) for x in df_balanced["compound index"].fillna("")]
            df_balanced["isotope"] = isotopes
            df_balanced["label_18F"] = (df_balanced["isotope"] == "18F").astype(int)
        
        feature_data = pd.concat([
            df_balanced[['compound index', smile_column_name, pred_column_name, 'isotope', 'label_18F']],
            X
        ], axis=1)
        
        feature_data.to_csv(cbrain_features_file, encoding='utf-8', index=False)
        log.info(f"特征数据已保存到: {cbrain_features_file}")
    
    # 确保所有特征都是数值型
    log.info("处理非数值特征")
    X = X.apply(pd.to_numeric, errors='coerce')
    X = X.fillna(0)
    
    # 确保所有列名都是字符串类型
    X.columns = X.columns.astype(str)
    
    log.info(f"特征处理后矩阵形状: {X.shape}")
    
    # 特征选择
    if not os.path.exists(feature_index_file):
        log.info("进行特征选择")
        log.info(f"特征选择前矩阵形状: {X.shape}")
        
        feature_extractor = FeatureExtraction(
            X, y,
            VT_threshold=0.02,
            RFE_features_to_select=RFE_features_to_select
        )
        
        selected_indices = feature_extractor.feature_extraction(returnIndex=True, index_dtype=int)
        
        try:
            np.savetxt(feature_index_file, selected_indices, fmt='%d')
            X = X.iloc[:, selected_indices]
            log.info(f"特征选择完成，选择后矩阵形状: {X.shape}")
            log.info(f"特征索引已保存到: {feature_index_file}")
        except Exception as e:
            log.error(f"特征选择失败: {e}")
            if os.path.exists(feature_index_file):
                os.remove(feature_index_file)
            sys.exit()
    else:
        log.info("读取已存在的特征索引")
        selected_indices = np.loadtxt(feature_index_file, dtype=int, delimiter=',').tolist()
        X = X.iloc[:, selected_indices]
        log.info(f"特征选择后矩阵形状: {X.shape}")
    
    # 特征归一化
    log.info("特征归一化")
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    X = pd.DataFrame(X_scaled, columns=X.columns)
    
    # 数据集划分
    log.info("数据集划分")
    # 首先划分出测试集 (10%)
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=0.1, random_state=42, stratify=df_balanced['label_18F']
    )
    log.info(f"训练验证集: {len(X_train_val)}个样本, 测试集: {len(X_test)}个样本")
    
    # 从训练验证集中划分出验证集 (10%)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=0.1, random_state=42
    )
    log.info(f"训练集: {len(X_train)}个样本, 验证集: {len(X_val)}个样本")
    
    # 重新归一化（使用训练+验证集拟合，测试集转换）
    scaler_final = MinMaxScaler()
    X_train_val_scaled = scaler_final.fit_transform(X_train_val)
    X_test_scaled = scaler_final.transform(X_test)
    
    X_train_val = pd.DataFrame(X_train_val_scaled, columns=X.columns)
    X_test = pd.DataFrame(X_test_scaled, columns=X.columns)
    
    # 更新划分
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=0.1, random_state=42
    )
    
    # 重置索引
    y_train_val = y_train_val.reset_index(drop=True)
    y_train = y_train.reset_index(drop=True)
    y_val = y_val.reset_index(drop=True)
    y_test = y_test.reset_index(drop=True)
    
    # 超参数优化
    log.info("开始SVM超参数优化")
    
    def objective(trial):
        X_tr, X_te, y_tr, y_te = train_test_split(
            X_train_val, y_train_val, test_size=0.1, random_state=seed
        )
        
        kernel = trial.suggest_categorical('kernel', ['rbf', 'linear', 'poly'])
        params = {
            'kernel': kernel,
            'C': trial.suggest_float('C', 1e-3, 1e3, log=True),
            'epsilon': trial.suggest_float('epsilon', 1e-3, 1.0, log=True),
        }
        
        # 如果是poly核，添加degree参数
        if kernel == 'poly':
            params['degree'] = trial.suggest_int('degree', 2, 5)
        
        # 如果是rbf或poly核，设置gamma参数
        if kernel in ['rbf', 'poly']:
            gamma_type = trial.suggest_categorical('gamma_type', ['scale', 'auto', 'float'])
            if gamma_type == 'float':
                params['gamma'] = trial.suggest_float('gamma', 1e-5, 1e-1, log=True)
            else:
                params['gamma'] = gamma_type
        
        model = SVR(**params)
        model.fit(X_tr, y_tr)
        y_pred = model.predict(X_te)
        
        return r2_score(y_te, y_pred)
    
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_jobs=2, n_trials=n_optuna_trial)  # SVM使用较少的并行
    
    log.info(f"最佳参数: {study.best_params}")
    log.info(f"最佳R²得分: {study.best_value}")
    
    # 使用最佳参数进行交叉验证
    log.info(f"使用最佳参数进行{cv_times}折交叉验证")
    
    # 清理参数，移除gamma_type
    clean_params = study.best_params.copy()
    if 'gamma_type' in clean_params:
        del clean_params['gamma_type']
    
    best_model = SVR(**clean_params)
    cv = KFold(n_splits=cv_times, random_state=seed, shuffle=True)
    
    cv_scores = {'rmse': [], 'mse': [], 'mae': [], 'r2': [], 'adj_r2': []}
    
    for fold, (train_idx, val_idx) in tqdm(enumerate(cv.split(X_train_val, y_train_val)), 
                                          desc="交叉验证", total=cv_times):
        X_fold_train = X_train_val.iloc[train_idx]
        X_fold_val = X_train_val.iloc[val_idx]
        y_fold_train = y_train_val.iloc[train_idx]
        y_fold_val = y_train_val.iloc[val_idx]
        
        # 训练模型
        best_model.fit(X_fold_train, y_fold_train)
        
        # 预测
        y_pred_fold = best_model.predict(X_fold_val)
        
        # 计算指标
        rmse = np.sqrt(mean_squared_error(y_fold_val, y_pred_fold))
        mse = mean_squared_error(y_fold_val, y_pred_fold)
        mae = mean_absolute_error(y_fold_val, y_pred_fold)
        r2 = r2_score(y_fold_val, y_pred_fold)
        adj_r2 = adjusted_r2_score(r2, len(y_fold_val), X_fold_val.shape[1])
        
        cv_scores['rmse'].append(rmse)
        cv_scores['mse'].append(mse)
        cv_scores['mae'].append(mae)
        cv_scores['r2'].append(r2)
        cv_scores['adj_r2'].append(adj_r2)
    
    # 输出交叉验证结果
    log.info("========交叉验证结果========")
    log.info(f"RMSE: {np.mean(cv_scores['rmse']):.3f}±{np.std(cv_scores['rmse']):.3f}")
    log.info(f"MSE: {np.mean(cv_scores['mse']):.3f}±{np.std(cv_scores['mse']):.3f}")
    log.info(f"MAE: {np.mean(cv_scores['mae']):.3f}±{np.std(cv_scores['mae']):.3f}")
    log.info(f"R²: {np.mean(cv_scores['r2']):.3f}±{np.std(cv_scores['r2']):.3f}")
    log.info(f"Adjusted R²: {np.mean(cv_scores['adj_r2']):.3f}±{np.std(cv_scores['adj_r2']):.3f}")
    
    # 最终模型训练和评估
    log.info("训练最终模型")
    
    # 清理参数，移除gamma_type
    clean_params = study.best_params.copy()
    if 'gamma_type' in clean_params:
        del clean_params['gamma_type']
    
    final_model = SVR(**clean_params)
    final_model.fit(X_train, y_train)
    
    # 各数据集预测
    y_pred_train = final_model.predict(X_train)
    y_pred_val = final_model.predict(X_val)
    y_pred_test = final_model.predict(X_test)
    
    # 计算各数据集指标
    def calculate_metrics(y_true, y_pred, n_features):
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        mape = mean_absolute_percentage_error(y_true, y_pred)
        adj_r2 = adjusted_r2_score(r2, len(y_true), n_features)
        
        return {
            'mse': mse, 'rmse': rmse, 'mae': mae,
            'r2': r2, 'mape': mape, 'adj_r2': adj_r2
        }
    
    train_metrics = calculate_metrics(y_train, y_pred_train, X_train.shape[1])
    val_metrics = calculate_metrics(y_val, y_pred_val, X_val.shape[1])
    test_metrics = calculate_metrics(y_test, y_pred_test, X_test.shape[1])
    
    # 输出最终结果
    log.info("========最终测试集结果========")
    log.info(f"MSE: {test_metrics['mse']:.4f}")
    log.info(f"RMSE: {test_metrics['rmse']:.4f}")
    log.info(f"MAE: {test_metrics['mae']:.4f}")
    log.info(f"R²: {test_metrics['r2']:.4f}")
    log.info(f"Adjusted R²: {test_metrics['adj_r2']:.4f}")
    log.info(f"MAPE: {test_metrics['mape']:.2f}%")
    
    # 保存模型
    model_path = "./cbrainmodel/svm_cbrain_18F_model.joblib"
    joblib.dump(final_model, model_path)
    log.info(f"模型已保存至: {model_path}")
    
    # 保存预测结果
    results = pd.DataFrame({
        'True_Values': y_test,
        'Predicted_Values': y_pred_test
    })
    results.to_csv('./result/svm_cbrain_18F_results.csv', index=False)
    
    # 绘制散点图
    plt.figure(figsize=(10, 8))
    plt.scatter(y_train, y_pred_train, alpha=0.5, label='Training Set', color='blue', s=20)
    plt.scatter(y_val, y_pred_val, alpha=0.7, label='Validation Set', color='green', s=20)
    plt.scatter(y_test, y_pred_test, alpha=0.8, label='Test Set', color='red', s=30)
    
    # 绘制对角线
    all_min = min(min(y_train), min(y_val), min(y_test))
    all_max = max(max(y_train), max(y_val), max(y_test))
    plt.plot([all_min, all_max], [all_min, all_max], 'k--', lw=2, alpha=0.8)
    
    plt.xlabel('Experimental Cbrain (60min)')
    plt.ylabel('Predicted Cbrain (60min)')
    plt.title('SVM Cbrain 18F-Resampled Model: Predicted vs Experimental Brain Concentration')
    plt.legend()
    
    # 添加R²信息
    plt.text(0.05, 0.95, f'Train R² = {train_metrics["r2"]:.3f}', 
             transform=plt.gca().transAxes, fontsize=10)
    plt.text(0.05, 0.90, f'Val R² = {val_metrics["r2"]:.3f}', 
             transform=plt.gca().transAxes, fontsize=10)
    plt.text(0.05, 0.85, f'Test R² = {test_metrics["r2"]:.3f}', 
             transform=plt.gca().transAxes, fontsize=10)
    
    plt.tight_layout()
    plt.savefig('./result/svm_cbrain_18F_scatter.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 保存完整实验报告
    experiment_report = {
        'experiment_info': {
            'dataset': 'PTBD_v20240912.csv',
            'balancing_method': 'oversample',
            'features': 'Mordred_descriptors + Molecular_fingerprints',
            'model': 'SVM',
            'target': 'brain at60min',
            'n_samples_original': len(df),
            'n_samples_balanced': len(df_balanced),
            'n_features': X.shape[1],
            'seed': seed
        },
        'best_params': study.best_params,
        'cross_validation': {
            'cv_folds': cv_times,
            'rmse': f"{np.mean(cv_scores['rmse']):.3f}±{np.std(cv_scores['rmse']):.3f}",
            'r2': f"{np.mean(cv_scores['r2']):.3f}±{np.std(cv_scores['r2']):.3f}",
            'adj_r2': f"{np.mean(cv_scores['adj_r2']):.3f}±{np.std(cv_scores['adj_r2']):.3f}"
        },
        'final_results': {
            'train': train_metrics,
            'validation': val_metrics,
            'test': test_metrics
        }
    }
    
    with open('./result/svm_cbrain_18F_report.json', 'w', encoding='utf-8') as f:
        json.dump(experiment_report, f, ensure_ascii=False, indent=2)
    
    log.info("========实验完成========")
    log.info(f"预测结果已保存至: ./result/svm_cbrain_18F_results.csv")
    log.info(f"散点图已保存至: ./result/svm_cbrain_18F_scatter.png") 
    log.info(f"实验报告已保存至: ./result/svm_cbrain_18F_report.json")
    
    print(f"\\nExperiment completed! Test R2 = {test_metrics['r2']:.4f}, RMSE = {test_metrics['rmse']:.4f}")