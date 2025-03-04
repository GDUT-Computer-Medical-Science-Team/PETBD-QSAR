import pandas as pd
from rdkit import Chem
from mordred import Calculator, descriptors
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import RFE
from sklearn.neural_network import MLPRegressor
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

        # 加载 MLP 模型
        model = joblib.load("./model/mlp_mo.joblib")
        print("MLP model loaded.")

        # 进行预测
        predictions = model.predict(df_descriptors_selected)
        print("Predictions made:", predictions[:5])  # 打印前五个预测值，以避免大量输出

        # 创建包含 SMILES、预测值和原始目标值的 DataFrame
        df_predictions = pd.DataFrame({
            "SMILES": df['SMILES'],

            "Predictions": predictions,
            "Original Values": df['cbrain']
        })
        print("DataFrame with predictions created.")

        # 保存 DataFrame 到 CSV 文件
        df_predictions.to_csv("./vivoresult/mlp_mo.csv", index=False)
        print("Predictions saved to CSV.")

    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    main()
