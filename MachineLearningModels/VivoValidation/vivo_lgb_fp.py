import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from sklearn.impute import SimpleImputer
from sklearn.svm import SVR
from sklearn.feature_selection import RFE
import lightgbm as lgb
import joblib

# 加载数据
df = pd.read_excel("./数据表汇总.xlsx")

# 计算分子指纹
molecules = [Chem.MolFromSmiles(smi) for smi in df['SMILES']]
fingerprints = [AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024) for mol in molecules]

# 将 RDKit ExplicitBitVect 转换为 NumPy 数组
df_fingerprints = np.array([np.array(fp) for fp in fingerprints])

# 使用分子指纹创建 DataFrame
df_fingerprints = pd.DataFrame(df_fingerprints, columns=[f'FP_{i}' for i in range(1, 1025)])

# 填充 NaN 值
imputer = SimpleImputer(strategy='mean')
df_fingerprints_imputed = imputer.fit_transform(df_fingerprints)

# 使用 RFE 进行特征选择，选择 50 个特征
estimator = SVR(kernel="linear")
selector = RFE(estimator, n_features_to_select=50, step=1)
selector = selector.fit(df_fingerprints_imputed, df['cbrain'])
X_selected = selector.transform(df_fingerprints_imputed)

# 加载 LightGBM 模型
model = joblib.load("./model/lgb_model_fp.joblib")

# 进行预测
predictions = model.predict(X_selected)
print(predictions)

# 创建包含 SMILES、预测值和原始目标值的 DataFrame
df_predictions = pd.DataFrame({
    "SMILES": df['SMILES'],  # 假设原始数据中 SMILES 列名为 'SMILES'
    "Predictions": predictions,
    "Original Values": df['cbrain']  # 假设原始目标值列名为 'cbrain'
})

# 保存 DataFrame 到 CSV 文件
df_predictions.to_csv("./vivoresult/lightgbm_fp.csv", index=False)
