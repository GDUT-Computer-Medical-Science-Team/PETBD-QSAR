import pandas as pd
import numpy as np
import os
import joblib
from time import time
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score
from preprocess.data_preprocess.data_preprocess_utils import calculate_Mordred_desc
from utils.DataLogger import DataLogger
from datetime import datetime
import matplotlib.pyplot as plt
from preprocess.data_preprocess.FeatureExtraction import FeatureExtraction
from preprocess.data_preprocess.data_preprocess_utils import calculate_Mordred_desc
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import cross_val_predict

# 设置日志记录器
log = DataLogger(log_file=f"../../data/b3db_data/log/{datetime.now().strftime('%Y%m%d')}.log").getlog(
    "ratio_xgboost_training")

# 文件路径设置
b3db_file = "../../data/B3DB_regression.tsv"
desc_file = '../../data/B3DB_w_desc_regrssion_fp.csv'
desc_index_file = "../../data/logBB_data/logBB_w_desc_fp_index2.txt"



# 读取数据
df = pd.read_csv(b3db_file, sep='\t')
RFE_features_to_select = 50
# 检查目标列和SMILES列的名称
smile_column_name = 'SMILES'
pred_column_name = 'logBB'
seed = int(time())

# 生成或读取描述符
if not os.path.exists(desc_file):
    X = calculate_Mordred_desc(df[smile_column_name])  # 确保calculate_Mordred_desc已定义
    pd.concat([X, df[pred_column_name]], axis=1).to_csv(desc_file, index=False)
else:
    df = pd.read_csv(desc_file)
    X = df.drop([pred_column_name], axis=1)
    X = X.drop(smile_column_name, axis=1)

y = df[pred_column_name]
mean_value = df['logBB'].mean()  # 计算目标变量的均值
df['logBB'].fillna(mean_value, inplace=True)  # 用均值填充 NaN 值

# 特征归一化
sc = MinMaxScaler()
sc.fit(X)
X = pd.DataFrame(sc.transform(X))

# 特征筛选
if not os.path.exists(desc_index_file):
    log.info("不存在特征索引文件，进行特征筛选")
    log.info(f"筛选前的特征矩阵形状为：{X.shape}")
    # X = FeatureExtraction(X,
    #                       y,
    #                       VT_threshold=0.02,
    #                       RFE_features_to_select=RFE_features_to_select).feature_extraction()
    desc_index = (FeatureExtraction(X,
                                    y,
                                    VT_threshold=0.02,
                                    RFE_features_to_select=RFE_features_to_select).
                  feature_extraction(returnIndex=True, index_dtype=int))
    try:
        np.savetxt(desc_index_file, desc_index, fmt='%d')
        X = X[desc_index]
        log.info(f"特征筛选完成，筛选后的特征矩阵形状为：{X.shape}, 筛选得到的特征索引保存到：{desc_index_file}")
    except (TypeError, KeyError) as e:
        log.error(e)
        os.remove(logBB_desc_index_file)
        sys.exit()
else:
    log.info("存在特征索引文件，进行读取")
    desc_index = np.loadtxt(desc_index_file, dtype=int, delimiter=',').tolist()
    X = X[desc_index]
    log.info(f"读取特征索引完成，筛选后的特征矩阵形状为：{X.shape}")

# 分割训练集与验证集
# 分割后索引重置，否则训练时KFold出现错误
X = X.reset_index(drop=True)
y = y.reset_index(drop=True)

# 从这里开始，您可以继续使用模型训练和验证的代码
model = joblib.load('./logbbModel/xgb_mordred_model.joblib')
# 确保验证数据集 X_val 和 y_val 已经准备好
# 如果需要，您可以重新加载并预处理数据集


# 直接使用模型进行外部验证预测
y_pred = model.predict(X)

# 计算外部验证性能指标
mse = mean_squared_error(y, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y, y_pred)
mae = mean_absolute_error(y, y_pred)

# 安全计算MAPE，避免除零错误
epsilon = 1e-10
mape = np.mean(np.abs((y - y_pred) / np.maximum(np.abs(y), epsilon))) * 100

# 计算调整后的 R 平方
n = X.shape[0]
p = X.shape[1]
adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)

# 创建结果目录
os.makedirs('./result', exist_ok=True)

# 读取原始B3DB数据
original_b3db = pd.read_csv(b3db_file, sep='\t')

# 确保索引对齐
if len(original_b3db) == len(y_pred):
    # 添加预测值到原始数据
    original_b3db['Predicted_logBB'] = y_pred

    # 保存结果
    original_b3db.to_csv('./result/B3DB_logBB_Mordred_Predict.csv', index=False)
    log.info("已将预测结果保存至 './result/B3DB_logBB_Mordred_Predict.csv'")
else:
    log.warning(f"原始数据行数({len(original_b3db)})与预测结果行数({len(y_pred)})不匹配，无法直接合并")

    # 创建包含SMILES、实际值和预测值的DataFrame
    y_pred_df = pd.DataFrame({
        'SMILES': df[smile_column_name],
        'Actual_logBB': y,
        'Predicted_logBB': y_pred
    })
    y_pred_df.to_csv('./result/B3DB_logBB_Mordred_Predict.csv', index=False)
    log.info("已将预测结果保存至 './result/B3DB_logBB_Mordred_Predict.csv'")

# 打印外部验证性能指标
log.info("\n========外部验证性能评估 (B3DB数据集)========")
log.info(f"样本数量: {n}")
log.info(f"特征数量: {p}")
log.info(f"MSE: {mse:.4f}")
log.info(f"RMSE: {rmse:.4f}")
log.info(f"R²: {r2:.4f}")
log.info(f"调整后R²: {adj_r2:.4f}")
log.info(f"MAE: {mae:.4f}")
log.info(f"MAPE: {mape:.2f}%")

# 绘制散点图
plt.figure(figsize=(10, 6))
plt.scatter(y, y_pred, alpha=0.5)
plt.title('B3DB 外部验证 - 实际值 vs 预测值')
plt.xlabel('实际 logBB 值')
plt.ylabel('预测 logBB 值')
plt.grid(True)
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=2)

# 添加性能指标文本
textstr = '\n'.join((
    f'R² = {r2:.3f}',
    f'RMSE = {rmse:.3f}',
    f'MAE = {mae:.3f}'
))
plt.text(0.05, 0.95, textstr, transform=plt.gca().transAxes, fontsize=12,
         verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

plt.tight_layout()
plt.savefig('./result/B3DB_logBB_mordred_prediction_scatter.png')
plt.close()
