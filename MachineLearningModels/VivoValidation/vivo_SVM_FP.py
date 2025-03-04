import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from sklearn.impute import SimpleImputer
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.feature_selection import SelectFromModel
from sklearn.svm import SVR
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

def calculate_fingerprints(smiles_list):
    molecules = [Chem.MolFromSmiles(smi) for smi in smiles_list]
    fingerprints = [AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024) for mol in molecules]
    return np.array([np.array(fp) for fp in fingerprints])

def feature_selection(df_fingerprints, target):
    tree_clf = ExtraTreesRegressor()
    tree_clf.fit(df_fingerprints, target)
    model = SelectFromModel(tree_clf, prefit=True, max_features=29)  # 改成29个特征
    return model.transform(df_fingerprints)

# 加载数据
df = pd.read_excel("./数据表汇总.xlsx")

# 计算分子指纹
df_fingerprints = calculate_fingerprints(df['SMILES'])

# 特征选择
X_selected = feature_selection(df_fingerprints, df['cbrain'])

# 加载已经训练好的 SVM 模型
svm_model = joblib.load("./model/svm_fp_model.joblib")

# 进行预测
predictions = svm_model.predict(X_selected)
print(predictions)

# 创建包含 SMILES、预测值和原始目标值的 DataFrame
df_predictions = pd.DataFrame({
    "SMILES": df['SMILES'],
    "Predictions": predictions,
    "Original Values": df['cbrain']
})

# 保存 DataFrame 到 CSV 文件
df_predictions.to_csv("./vivoresult/svm_fp.csv", index=False)

# 绘制散点图
plt.figure(figsize=(10, 6))
plt.scatter(df_predictions["Original Values"], df_predictions["Predictions"])
plt.xlabel('Original Values')
plt.ylabel('Predictions')
plt.title('Original vs Predicted Values')
plt.grid(True)
plt.show()

# 计算评估指标
mse = mean_squared_error(df_predictions["Original Values"], df_predictions["Predictions"])
r2 = r2_score(df_predictions["Original Values"], df_predictions["Predictions"])
mae = mean_absolute_error(df_predictions["Original Values"], df_predictions["Predictions"])

# 打印评估指标
print("Mean Squared Error:", mse)
print("R-squared:", r2)
print("Mean Absolute Error:", mae)
