import pandas as pd
from rdkit import Chem
from mordred import Calculator, descriptors
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectFromModel
import joblib

# 加载数据
df = pd.read_excel("./数据表汇总.xlsx")

# 计算 Mordred 描述符
calc = Calculator(descriptors, ignore_3D=True)
molecules = [Chem.MolFromSmiles(smi) for smi in df['SMILES']]
df_descriptors = pd.DataFrame([calc(mol) for mol in molecules])

# 移除全部是 NaN 的列
df_descriptors = df_descriptors.dropna(axis=1, how='all')

# 填充剩余的 NaN 值
imputer = SimpleImputer(strategy='mean')
df_descriptors_imputed = imputer.fit_transform(df_descriptors)

# 使用基于树的模型进行特征选择
forest_clf = RandomForestRegressor(n_estimators=100)
forest_clf = forest_clf.fit(df_descriptors_imputed, df['cbrain'])
model = SelectFromModel(forest_clf, prefit=True, max_features=50)
X_selected = model.transform(df_descriptors_imputed)

# 加载随机森林模型
rf_model = joblib.load("./model/random_forest_Mordred.joblib")

# 进行预测
predictions = rf_model.predict(X_selected)
print(predictions)

# 创建包含 SMILES、预测值和原始目标值的 DataFrame
df_predictions = pd.DataFrame({
    "SMILES": df['SMILES'],  # 假设原始数据中 SMILES 列名为 'SMILES'
    "Predictions": predictions,
    "Original Values": df['cbrain']  # 假设原始目标值列名为 'cbrain'
})

# 保存 DataFrame 到 CSV 文件
df_predictions.to_csv("./vivoresult/rf_modered.csv", index=False)
