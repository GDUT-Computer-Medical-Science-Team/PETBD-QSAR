import pandas as pd
from rdkit import Chem
from mordred import Calculator, descriptors
from sklearn.impute import SimpleImputer
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.feature_selection import SelectFromModel
from sklearn.svm import SVR
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

def calculate_descriptors(smiles_list):
    calc = Calculator(descriptors, ignore_3D=True)
    molecules = [Chem.MolFromSmiles(smi) for smi in smiles_list]
    return pd.DataFrame([calc(mol) for mol in molecules])

def feature_selection(df_descriptors, target):
    tree_clf = ExtraTreesRegressor()
    tree_clf.fit(df_descriptors, target)
    model = SelectFromModel(tree_clf, prefit=True, max_features=50)
    return model.transform(df_descriptors)

# 加载数据
df = pd.read_excel("./数据表汇总.xlsx")

# 计算 Mordred 描述符
df_descriptors = calculate_descriptors(df['SMILES'])

# 移除全部是 NaN 的列
df_descriptors = df_descriptors.dropna(axis=1, how='all')

# 填充剩余的 NaN 值
imputer = SimpleImputer(strategy='mean')
df_descriptors_imputed = imputer.fit_transform(df_descriptors)

# 特征选择
X_selected = feature_selection(df_descriptors_imputed, df['cbrain'])

# 使用 SVM 模型进行训练
svm_model = SVR(kernel='linear')
svm_model.fit(X_selected, df['cbrain'])



# 加载 SVM 模型
svm_model = joblib.load("./model/svm_mo_model.joblib")

# 再次进行特征选择，确保训练和预测时一致
X_selected = feature_selection(df_descriptors_imputed, df['cbrain'])

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
df_predictions.to_csv("./vivoresult/svm_mordred.csv", index=False)

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
