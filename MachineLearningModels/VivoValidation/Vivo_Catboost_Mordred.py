import pandas as pd
from rdkit import Chem
from mordred import Calculator, descriptors
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectFromModel
import catboost as cb

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

# 使用 CatBoost 进行特征选择
catboost_clf = cb.CatBoostRegressor(iterations=100, depth=6, learning_rate=0.1, loss_function='RMSE', verbose=0)
catboost_clf.fit(df_descriptors_imputed, df['cbrain'])

# 获取特征重要性并选择前50个重要特征
feature_importances = catboost_clf.get_feature_importance()
important_features_indices = feature_importances.argsort()[-50:][::-1]
X_selected = df_descriptors_imputed[:, important_features_indices]

# 加载 XGBoost 模型
model = cb.CatBoostRegressor()
model.load_model("./model/catboost_mo_model.cbm")

# 进行预测
predictions = model.predict(X_selected)
print(predictions)

# 创建包含 SMILES、预测值和原始目标值的 DataFrame
df_predictions = pd.DataFrame({
    "SMILES": df['SMILES'],  # 假设原始数据中SMILES列名为 'SMILES'
    "Predictions": predictions,
    "Original Values": df['cbrain']  # 假设原始目标值列名为 'cbrain'
})

# 保存 DataFrame 到 CSV 文件
df_predictions.to_csv("./vivoresult/catboost_predictions_with_original_values_mordred.csv", index=False)
