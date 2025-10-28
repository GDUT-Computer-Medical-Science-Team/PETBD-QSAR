import pandas as pd
import numpy as np
import os
import joblib
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from utils.DataLogger import DataLogger
from datetime import datetime
import matplotlib.pyplot as plt
import tempfile
from padelpy import padeldescriptor
from tqdm import tqdm
from preprocess.data_preprocess.FeatureExtraction import FeatureExtraction

# [Chinese text removed]
log = DataLogger(log_file=f"../../data/b3db_data/log/{datetime.now().strftime('%Y%m%d')}.log").getlog(
    "b3db_external_validation")


def calculate_padel_fingerprints(smiles_list):
    """[Chinese text removed]PaDEL[Chinese text removed]"""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.smi', delete=False) as temp_file:
        for smi in smiles_list:
            temp_file.write(f"{smi}\n")
        temp_smi_file = temp_file.name

    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            output_file = os.path.join(temp_dir, "fingerprints.csv")

            padeldescriptor(
                mol_dir=temp_smi_file,
                d_file=output_file,
                fingerprints=True,
                descriptortypes=None,
                detectaromaticity=True,
                standardizenitro=True,
                standardizetautomers=True,
                threads=2,
                removesalt=True,
                log=False
            )

            df = pd.read_csv(output_file)
            if 'Name' in df.columns:
                df = df.drop('Name', axis=1)
            df.insert(0, 'SMILES', smiles_list)
            return df

    except Exception as e:
        log.error(f"[Chinese text removed]PaDEL[Chinese text removed]: {str(e)}")
        raise
    finally:
        if os.path.exists(temp_smi_file):
            os.remove(temp_smi_file)


def plot_scatter(y_true, y_pred, save_path=None):
    """Plot scatter plot"""
    plt.figure(figsize=(10, 6))
    plt.scatter(y_true, y_pred, alpha=0.5)

    min_val = min(min(y_true), min(y_pred))
    max_val = max(max(y_true), max(y_pred))
    plt.plot([min_val, max_val], [min_val, max_val], 'k--', lw=2)

    plt.xlabel('[Chinese text removed] logBB')
    plt.ylabel('[Chinese text removed] logBB')
    plt.title('B3DB External Validation')

    r2 = r2_score(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)

    plt.text(0.05, 0.95, f'R² = {r2:.3f}', transform=plt.gca().transAxes)
    plt.text(0.05, 0.90, f'RMSE = {rmse:.3f}', transform=plt.gca().transAxes)
    plt.text(0.05, 0.85, f'MAE = {mae:.3f}', transform=plt.gca().transAxes)
    plt.grid(True)

    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()


def main():
    # [Chinese text removed]
    b3db_file = "../../data/B3DB_regression.tsv"
    desc_file = '../../data/B3DB_w_desc_regrssion_fp.csv'
    desc_index_file = "../../data/logBB_data/logBB_w_desc_fp_index2.txt"
    model_path = "./logbbModel/xgb_fp_model.joblib"

    smile_column_name = 'SMILES'
    pred_column_name = 'logBB'
    RFE_features_to_select = 50

    # [Chinese text removed]
    log.info("Starting[Chinese text removed]B3DB[Chinese text removed]...")
    df = pd.read_csv(b3db_file, sep='\t')

    # [Chinese text removed]
    if not os.path.exists(desc_file):
        log.info("[Chinese text removed]B3DB[Chinese text removed]...")
        X = calculate_padel_fingerprints(tqdm(df[smile_column_name].tolist(), desc="[Chinese text removed]"))
        pd.concat([df[smile_column_name], X.drop('SMILES', axis=1), df[[pred_column_name]]], axis=1).to_csv(desc_file,
                                                                                                            index=False)
        log.info(f"features[Chinese text removed]: {X.shape}")
    else:
        log.info("[Chinese text removed]...")
        df = pd.read_csv(desc_file)
        X = df.drop([pred_column_name, smile_column_name], axis=1)
        log.info(f"features[Chinese text removed]: {X.shape}")

    # [Chinese text removed] NaN [Chinese text removed]
    mean_logBB = df[pred_column_name].mean()
    df[pred_column_name].fillna(mean_logBB, inplace=True)
    y = df[pred_column_name]

    # Feature normalization
    sc = MinMaxScaler()
    X = pd.DataFrame(sc.fit_transform(X), columns=X.columns)

    # features[Chinese text removed]
    if not os.path.exists(desc_index_file):
        log.info("[Chinese text removed]features[Chinese text removed]，[Chinese text removed]features[Chinese text removed]")
        log.info(f"[Chinese text removed]features[Chinese text removed]：{X.shape}")
        desc_index = (FeatureExtraction(X, y,
                                        VT_threshold=0.02,
                                        RFE_features_to_select=RFE_features_to_select).
                      feature_extraction(returnIndex=True, index_dtype=int))
        try:
            np.savetxt(desc_index_file, desc_index, fmt='%d')
            log.info(f"Feature indices saved[Chinese text removed]：{desc_index_file}")
        except (TypeError, KeyError) as e:
            log.error(e)
            os.remove(desc_index_file)
            raise
    else:
        log.info("[Chinese text removed]features[Chinese text removed]，[Chinese text removed]")
        desc_index = np.loadtxt(desc_index_file, dtype=int, delimiter=',').tolist()
        log.info(f"[Chinese text removed]features[Chinese text removed]Complete，features[Chinese text removed]: {desc_index}")

        # [Chinese text removed]
        if any(i >= X.shape[1] for i in desc_index):
            log.error("features[Chinese text removed]，[Chinese text removed]。")
            raise IndexError("features[Chinese text removed]。")

        X = X.iloc[:, desc_index]  # [Chinese text removed]iloc[Chinese text removed]
        log.info(f"[Chinese text removed]features[Chinese text removed]：{X.shape}")
        log.info(f"[Chinese text removed]features[Chinese text removed]: {X.columns.tolist()}")

        # [Chinese text removed]features[Chinese text removed]50[Chinese text removed]
        if X.shape[1] < 50:
            log.error("features[Chinese text removed]50，[Chinese text removed]features。")
            # [Chinese text removed]features，[Chinese text removed]features
            additional_features = pd.DataFrame(np.zeros((X.shape[0], 50 - X.shape[1])),
                                               columns=[f'ExtraFeature{i}' for i in range(50 - X.shape[1])])
            X = pd.concat([X, additional_features], axis=1)
            log.info(f"[Chinese text removed]features[Chinese text removed]：{X.shape}")

        if X.shape[1] > 50:
            X = X.iloc[:, :50]  # [Chinese text removed]50[Chinese text removed]features
            log.info(f"features[Chinese text removed]50，[Chinese text removed]50[Chinese text removed]features，[Chinese text removed]：{X.shape}")

    # [Chinese text removed]
    X = X.reset_index(drop=True)
    y = y.reset_index(drop=True)

    # [Chinese text removed]
    log.info("[Chinese text removed]XGBoost[Chinese text removed]...")
    model = joblib.load(model_path)

    log.info("Starting[Chinese text removed]...")
    y_pred = model.predict(X)

    # [Chinese text removed]Save prediction results[Chinese text removed]
    # [Chinese text removed]originalB3DB[Chinese text removed]
    original_b3db = pd.read_csv(b3db_file, sep='\t')

    # [Chinese text removed]
    if len(original_b3db) == len(y_pred):
        # [Chinese text removed]predicted values[Chinese text removed]original[Chinese text removed]
        original_b3db['Predicted_logBB'] = y_pred

        # [Chinese text removed]
        original_b3db.to_csv('./result/B3DB_logBB_FP_regression_Predict.csv', index=False)
        log.info("[Chinese text removed] './result/B3DB_logBB_FP_regression_Predict.csv'")
    else:
        log.warning(f"original[Chinese text removed]({len(original_b3db)})[Chinese text removed]({len(y_pred)})[Chinese text removed]，[Chinese text removed]")

        # [Chinese text removed]SMILES、[Chinese text removed]predicted values[Chinese text removed]DataFrame
        y_pred_df = pd.DataFrame({
            'SMILES': df[smile_column_name],
            'Actual_logBB': y,
            'Predicted_logBB': y_pred
        })
        y_pred_df.to_csv('./result/B3DB_logBB_FP_Predict.csv', index=False)
        log.info("[Chinese text removed] './result/B3DB_logBB_FP_Predict.csv'")

    # [Chinese text removed]
    plot_scatter(y, y_pred, save_path='./result/B3DB_fp_logBB_prediction_scatter.png')
    log.info("[Chinese text removed] './result/B3DB_fp_logBB_prediction_scatter.png'")

    # [Chinese text removed]
    mse = mean_squared_error(y, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y, y_pred)
    mae = mean_absolute_error(y, y_pred)
    mape = np.mean(np.abs((y - y_pred) / y)) * 100
    adj_r2 = 1 - (1 - r2) * (len(y) - 1) / (len(y) - X.shape[1] - 1)

    log.info("\n========[Chinese text removed] (B3DB[Chinese text removed])========")
    log.info(f"samples[Chinese text removed]: {len(y)}")
    log.info(f"features[Chinese text removed]: {X.shape[1]}")
    log.info(f"MSE: {mse:.4f}")
    log.info(f"RMSE: {rmse:.4f}")
    log.info(f"R2: {r2:.4f}")
    log.info(f"[Chinese text removed]R2: {adj_r2:.4f}")
    log.info(f"MAE: {mae:.4f}")
    log.info(f"MAPE: {mape:.2f}%")


if __name__ == '__main__':
    main()
