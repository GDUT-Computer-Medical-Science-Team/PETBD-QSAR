import pandas as pd
import numpy as np
from rdkit import Chem
from mordred import Calculator, descriptors
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import RFE
import xgboost as xgb
import joblib
import matplotlib.pyplot as plt
import os

def main():
    try:
        # 创建索引保存目录
        os.makedirs("../data/cbrain/index", exist_ok=True)
        
        # 加载数据
        df = pd.read_excel("./数据表汇总.xlsx")
        print("Data loaded with shape:", df.shape)

        # 计算分子描述符
        calc = Calculator(descriptors, ignore_3D=True)
        molecules = [Chem.MolFromSmiles(smi) for smi in df['SMILES'] if smi]
        df_descriptors = pd.DataFrame([calc(mol) for mol in molecules if mol])
        print("Descriptors calculated with shape:", df_descriptors.shape)

        # 移除全部是NaN的列
        df_descriptors = df_descriptors.replace([np.inf, -np.inf], np.nan)  # 将无穷值转换为NaN
        df_descriptors.dropna(axis=1, how='all', inplace=True)
        print("After dropping all-NaN columns:", df_descriptors.shape)

        # 使用SimpleImputer填充剩余的NaN值
        imputer = SimpleImputer(strategy='mean')
        df_descriptors_imputed = pd.DataFrame(
            imputer.fit_transform(df_descriptors),
            columns=df_descriptors.columns
        )
        print("After imputation shape:", df_descriptors_imputed.shape)

        # 确保数据类型是数值型
        df_descriptors_imputed = df_descriptors_imputed.astype(float)

        # 使用RFE进行特征选择
        from sklearn.ensemble import ExtraTreesRegressor
        tree_clf = ExtraTreesRegressor(random_state=42)
        rfe = RFE(estimator=tree_clf, n_features_to_select=50)
        
        # 确保df_descriptors和目标变量的长度匹配
        y = df['cbrain'].iloc[:len(df_descriptors_imputed)]
        
        # 进行特征选择
        rfe.fit(df_descriptors_imputed, y)
        
        # 获取并保存选中特征的索引
        selected_features_mask = rfe.support_
        selected_features_indices = np.where(selected_features_mask)[0]
        selected_feature_names = df_descriptors_imputed.columns[selected_features_indices].tolist()
        
        # 保存特征索引和名称
        index_file_path = "../data/cbrain/index/fp_selected_features_index.txt"
        names_file_path = "../data/cbrain/index/fp_selected_features_names.txt"
        
        np.savetxt(index_file_path, selected_features_indices, fmt='%d')
        with open(names_file_path, 'w') as f:
            for name in selected_feature_names:
                f.write(f"{name}\n")
                
        print(f"Selected features indices saved to: {index_file_path}")
        print(f"Selected features names saved to: {names_file_path}")

        # 使用选中的特征进行转换
        X_selected = rfe.transform(df_descriptors_imputed)

        # 训练XGBoost模型
        xgb_model = xgb.XGBRegressor(
            objective='reg:squarederror',
            n_estimators=1000,
            learning_rate=0.01,
            max_depth=6,
            random_state=42
        )
        xgb_model.fit(X_selected, y)

        # 进行预测
        predictions = xgb_model.predict(X_selected)

        # 创建预测结果DataFrame
        df_predictions = pd.DataFrame({
            "SMILES": df['SMILES'].iloc[:len(predictions)],
            "Predictions": predictions,
            "Original Values": y
        })

        # 保存结果
        df_predictions.to_csv("./vivoresult/xgb_fp.csv", index=False)
        joblib.dump(xgb_model, "./model/xgb_fp_model.pkl")

        # 打印一些评估指标
        from sklearn.metrics import mean_squared_error, r2_score
        mse = mean_squared_error(y, predictions)
        r2 = r2_score(y, predictions)
        print(f"\nModel Performance:")
        print(f"Mean Squared Error: {mse:.4f}")
        print(f"R² Score: {r2:.4f}")

        # 绘制散点图
        plt.figure(figsize=(10, 6))
        plt.scatter(df_predictions["Original Values"], df_predictions["Predictions"])
        plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2)  # 添加对角线
        plt.xlabel('Original Values')
        plt.ylabel('Predictions')
        plt.title('Original vs Predicted Values')
        plt.grid(True)
        plt.savefig("./vivoresult/prediction_plot.png")
        plt.close()

        print("Processing completed successfully!")

    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        print(traceback.format_exc())

if __name__ == "__main__":
    main()
