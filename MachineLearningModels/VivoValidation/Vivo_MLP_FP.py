import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from sklearn.impute import SimpleImputer
from sklearn.svm import SVR
from sklearn.feature_selection import SelectFromModel
from sklearn.neural_network import MLPRegressor
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

# 使用 SVM 进行特征选择
svm_clf = SVR(kernel="linear")
svm_clf = svm_clf.fit(df_fingerprints_imputed, df['cbrain'])
model = SelectFromModel(svm_clf, prefit=True, max_features=50, threshold=-np.inf)
X_selected = model.transform(df_fingerprints_imputed)

# 加载 MLP 模型
mlp_model = joblib.load("./model/mlp_fp_model.joblib")

# 进行预测
predictions = mlp_model.predict(X_selected)
print(predictions)

# 创建包含 SMILES、预测值和原始目标值的 DataFrame
df_predictions = pd.DataFrame({
    "SMILES": df['SMILES'],  # 假设原始数据中 SMILES 列名为 'SMILES'
    "Predictions": predictions,
    "Original Values": df['cbrain']  # 假设原始目标值列名为 'cbrain'
})

# 保存 DataFrame 到 CSV 文件
df_predictions.to_csv("./vivoresult/mlp_fp.csv", index=False)
