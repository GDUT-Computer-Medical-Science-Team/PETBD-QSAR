import pandas as pd
from rdkit import Chem
from mordred import Calculator, descriptors
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import RFE
import xgboost as xgb
from sklearn.model_selection import train_test_split
import numpy as np
import joblib


def main():
    try:
        # 加载数据
        df = pd.read_excel("./数据表汇总.xlsx")
        print("Data loaded with shape:", df.shape)

        # 计算 Mordred 描述符
        calc = Calculator(descriptors, ignore_3D=True)
        molecules = [Chem.MolFromSmiles(smi) for smi in df['SMILES'] if smi]  # 检查 SMILES 是否非空
        df_descriptors = pd.DataFrame([calc(mol) for mol in molecules if mol])  # 只对有效分子计算
        print("Descriptors calculated with shape:", df_descriptors.shape)

        # 移除全部是 NaN 的列
        df_descriptors.dropna(axis=1, how='all', inplace=True)

        # 移除包含任何 NaN 值的列
        df_descriptors = df_descriptors.dropna(axis=1)
        print("NaN columns dropped, remaining shape:", df_descriptors.shape)

        # 填充剩余的 NaN 值
        imputer = SimpleImputer(strategy='mean')
        df_descriptors_imputed = imputer.fit_transform(df_descriptors)
        print("NaN values imputed.")

        # 使用 RFE 选择 50 个特征
        model_for_rfe = RandomForestRegressor()  # 使用 RandomForest 作为 RFE 的基础模型
        rfe = RFE(model_for_rfe, n_features_to_select=50)
        rfe.fit(df_descriptors_imputed, df['cbrain'])
        df_descriptors_selected = rfe.transform(df_descriptors_imputed)
        print("RFE completed. Selected features shape:", df_descriptors_selected.shape)

        # 加入噪声（模拟更复杂的情况）
        noise = np.random.normal(0, 0.1, df_descriptors_selected.shape)  # 生成均值为0，标准差为0.1的噪声
        df_descriptors_selected_noisy = df_descriptors_selected + noise
        print("Noise added to the selected features.")

        # 划分训练集和测试集
        X = pd.DataFrame(df_descriptors_selected_noisy)  # 将NumPy数组转换为DataFrame
        X_train, X_test, y_train, y_test = train_test_split(X, df['cbrain'], test_size=0.2, random_state=42)

        # 训练 XGBoost 模型
        model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=1000, learning_rate=0.01, max_depth=6,
                                 random_state=42)
        model.fit(X_train, y_train)
        print("XGBoost model trained.")

        # 进行预测
        predictions = model.predict(X_test)
        print("Predictions made:", predictions[:5])  # 打印前五个预测值，以避免大量输出

        # 获取测试集对应的SMILES
        test_indices = y_test.index  # 使用y_test的索引
        test_smiles = df['SMILES'].iloc[test_indices]

        # 创建包含SMILES、预测值和原始目标值的DataFrame
        df_predictions = pd.DataFrame({
            "SMILES": test_smiles,
            "Predictions": predictions,
            "Original Values": y_test
        })
        print("DataFrame with predictions created.")

        # 保存 DataFrame 到 CSV 文件
        df_predictions.to_csv("./vivoresult/xgb_mo.csv", index=False)
        print("Predictions saved to CSV.")

        # 保存训练好的模型
        joblib.dump(model, "./model/xgb_mo.joblib")
        print("XGBoost model saved to disk.")

    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    main()
