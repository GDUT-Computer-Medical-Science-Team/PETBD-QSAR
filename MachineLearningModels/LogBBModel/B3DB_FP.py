import pandas as pd
import numpy as np
import os
import joblib
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from utils.DataLogger import DataLogger
from datetime import datetime
import matplotlib.pyplot as plt
import tempfile
from padelpy import padeldescriptor
from tqdm import tqdm
from preprocess.data_preprocess.FeatureExtraction import FeatureExtraction

# 设置日志记录器
log = DataLogger(log_file=f"../../data/b3db_data/log/{datetime.now().strftime('%Y%m%d')}.log").getlog("b3db_external_validation")

def calculate_padel_fingerprints(smiles_list):
    """计算分子的PaDEL指纹"""
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
        log.error(f"计算PaDEL指纹时出错: {str(e)}")
        raise
    finally:
        if os.path.exists(temp_smi_file):
            os.remove(temp_smi_file)

def plot_scatter(y_true, y_pred, save_path=None):
    """绘制散点图"""
    plt.figure(figsize=(10, 6))
    plt.scatter(y_true, y_pred, alpha=0.5)
    
    min_val = min(min(y_true), min(y_pred))
    max_val = max(max(y_true), max(y_pred))
    plt.plot([min_val, max_val], [min_val, max_val], 'k--', lw=2)
    
    plt.xlabel('真实 logBB')
    plt.ylabel('预测 logBB')
    plt.title('B3DB External Validation')
    
    r2 = r2_score(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    
    plt.text(0.05, 0.95, f'R² = {r2:.3f}', transform=plt.gca().transAxes)
    plt.text(0.05, 0.90, f'RMSE = {rmse:.3f}', transform=plt.gca().transAxes)
    plt.text(0.05, 0.85, f'MAE = {mae:.3f}', transform=plt.gca().transAxes)
    plt.grid(True)
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def main():
    # 文件路径设置
    b3db_file = "../../data/B3DB_classification.tsv"
    desc_file = '../../data/B3DB_w_desc_fp.csv'
    desc_index_file = "../../data/logBB_data/logBB_w_desc_fp_index2.txt"
    model_path = "./logbbModel/xgb_fp_model.joblib"
    
    smile_column_name = 'SMILES'
    pred_column_name = 'logBB'
    RFE_features_to_select = 50

    # 读取数据并计算分子指纹
    log.info("开始加载B3DB数据集...")
    df = pd.read_csv(b3db_file, sep='\t')

    # 生成或读取描述符
    if not os.path.exists(desc_file):
        log.info("计算B3DB数据集的分子指纹...")
        X = calculate_padel_fingerprints(tqdm(df[smile_column_name].tolist(), desc="计算分子指纹"))
        pd.concat([df[smile_column_name], X.drop('SMILES', axis=1), df[[pred_column_name]]], axis=1).to_csv(desc_file, index=False)
        log.info(f"特征矩阵形状: {X.shape}")
    else:
        log.info("从已有文件加载分子指纹...")
        df = pd.read_csv(desc_file)
        X = df.drop([pred_column_name, smile_column_name], axis=1)
        log.info(f"特征矩阵形状: {X.shape}")

    # 使用平均值填充 NaN 值
    mean_logBB = df[pred_column_name].mean()
    df[pred_column_name].fillna(mean_logBB, inplace=True)
    y = df[pred_column_name]

    # 特征归一化
    sc = MinMaxScaler()
    X = pd.DataFrame(sc.fit_transform(X), columns=X.columns)

    # 特征筛选
    if not os.path.exists(desc_index_file):
        log.info("不存在特征索引文件，进行特征筛选")
        log.info(f"筛选前的特征矩阵形状为：{X.shape}")
        desc_index = (FeatureExtraction(X, y,
                                      VT_threshold=0.02,
                                      RFE_features_to_select=RFE_features_to_select).
                     feature_extraction(returnIndex=True, index_dtype=int))
        try:
            np.savetxt(desc_index_file, desc_index, fmt='%d')
            log.info(f"特征索引已保存至：{desc_index_file}")
        except (TypeError, KeyError) as e:
            log.error(e)
            os.remove(desc_index_file)
            raise
    else:
        log.info("存在特征索引文件，进行读取")
        desc_index = np.loadtxt(desc_index_file, dtype=int, delimiter=',').tolist()
        log.info(f"读取特征索引完成，特征索引内容: {desc_index}")

        # 检查索引有效性
        if any(i >= X.shape[1] for i in desc_index):
            log.error("特征索引包含无效的列索引，无法进行筛选。")
            raise IndexError("特征索引包含无效的列索引。")

        X = X.iloc[:, desc_index]  # 使用iloc按索引选择列
        log.info(f"筛选后的特征矩阵形状为：{X.shape}")
        log.info(f"选择的特征名称: {X.columns.tolist()}")

        # 确保特征数量为50个
        if X.shape[1] < 50:
            log.error("特征数量不足50，尝试填充特征。")
            # 填充特征，使用零填充或其他特征
            additional_features = pd.DataFrame(np.zeros((X.shape[0], 50 - X.shape[1])), columns=[f'ExtraFeature{i}' for i in range(50 - X.shape[1])])
            X = pd.concat([X, additional_features], axis=1)
            log.info(f"填充后特征矩阵形状为：{X.shape}")

        if X.shape[1] > 50:
            X = X.iloc[:, :50]  # 只保留前50个特征
            log.info(f"特征数量超过50，已调整为前50个特征，形状为：{X.shape}")

    # 重置索引
    X = X.reset_index(drop=True)
    y = y.reset_index(drop=True)

    # 加载模型并预测
    log.info("加载预训练的XGBoost模型...")
    model = joblib.load(model_path)
    
    log.info("开始进行预测...")
    y_pred = model.predict(X)
    
    # 修改保存预测结果的部分
    # 读取原始B3DB数据
    original_b3db = pd.read_csv(b3db_file, sep='\t')

    # 确保索引对齐
    if len(original_b3db) == len(y_pred):
        # 添加预测值到原始数据
        original_b3db['Predicted_logBB'] = y_pred
        
        # 保存结果
        original_b3db.to_csv('./result/B3DB_logBB_FP_Predict.csv', index=False)
        log.info("已将预测结果保存至 './result/B3DB_logBB_FP_Predict.csv'")
    else:
        log.warning(f"原始数据行数({len(original_b3db)})与预测结果行数({len(y_pred)})不匹配，无法直接合并")
        
        # 创建包含SMILES、实际值和预测值的DataFrame
        y_pred_df = pd.DataFrame({
            'SMILES': df[smile_column_name],
            'Actual_logBB': y,
            'Predicted_logBB': y_pred
        })
        y_pred_df.to_csv('./result/B3DB_logBB_FP_Predict.csv', index=False)
        log.info("已将预测结果保存至 './result/B3DB_logBB_FP_Predict.csv'")
    
    # 绘制并保存散点图
    plot_scatter(y, y_pred, save_path='./result/B3DB_fp_logBB_prediction_scatter.png')
    log.info("散点图已保存到 './result/B3DB_fp_logBB_prediction_scatter.png'")
    
    # 计算并输出评估指标
    mse = mean_squared_error(y, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y, y_pred)
    mae = mean_absolute_error(y, y_pred)
    mape = np.mean(np.abs((y - y_pred) / y)) * 100
    adj_r2 = 1 - (1 - r2) * (len(y) - 1) / (len(y) - X.shape[1] - 1)
    
    log.info("\n========外部验证结果 (B3DB数据集)========")
    log.info(f"样本数量: {len(y)}")
    log.info(f"特征数量: {X.shape[1]}")
    log.info(f"MSE: {mse:.4f}")
    log.info(f"RMSE: {rmse:.4f}")
    log.info(f"R2: {r2:.4f}")
    log.info(f"调整后R2: {adj_r2:.4f}")
    log.info(f"MAE: {mae:.4f}")
    log.info(f"MAPE: {mape:.2f}%")

if __name__ == '__main__':
    main()
