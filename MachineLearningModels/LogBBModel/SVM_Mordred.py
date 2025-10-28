import sys
from time import time
import numpy as np
import optuna
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, mean_absolute_percentage_error
from sklearn.model_selection import KFold
import pandas as pd
from datetime import datetime
from preprocess.data_preprocess.FeatureExtraction import FeatureExtraction
from preprocess.data_preprocess.data_preprocess_utils import calculate_Mordred_desc
import joblib
import os
from utils.DataLogger import DataLogger
from sklearn.svm import SVR  # [Chinese text removed]
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

log = DataLogger(log_file=f"../../data/logBB_data/log/{datetime.now().strftime('%Y%m%d')}.log").getlog("ratio_svm_training")

os.makedirs("./logbbModel", exist_ok=True)


def calculate_mape(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    # [Chinese text removed]
    mask = y_true != 0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100


def adjusted_r2_score(r2, n, k):
    """
    [Chinese text removed]R[Chinese text removed]
    :param r2: R[Chinese text removed]
    :param n: samples[Chinese text removed]
    :param k: [Chinese text removed]
    :return: [Chinese text removed]R[Chinese text removed]
    """
    return 1 - (1 - r2) * (n - 1) / (n - k - 1)


def objective(trial):
    df = pd.read_csv(logBB_desc_file, encoding='utf-8')
    y = df[pred_column_name]
    X = df.drop([smile_column_name, pred_column_name], axis=1)

    # [Chinese text removed]features[Chinese text removed] blood mean60min
    if 'blood mean60min' in X.columns:
        X = X.drop('blood mean60min', axis=1)
        log.info("[Chinese text removed]features: blood mean60min")

    sc = MinMaxScaler()
    sc.fit(X)
    X = pd.DataFrame(sc.transform(X))

    # [Chinese text removed]parameter[Chinese text removed]
    param = {
        'kernel': trial.suggest_categorical('kernel', ['linear', 'poly', 'rbf', 'sigmoid']),
        'C': trial.suggest_loguniform('C', 1e-3, 1e2),
        'epsilon': trial.suggest_loguniform('epsilon', 1e-3, 1e1)
    }

    kf = KFold(n_splits=10, shuffle=True, random_state=seed)
    rmse_list = []

    for train_index, test_index in kf.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y[train_index], y[test_index]

        model = SVR(**param)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        rmse_list.append(rmse)

    return np.mean(rmse_list)


if __name__ == '__main__':
    logBB_data_file = "../../data/logBB_data/logBB.csv"
    logBB_desc_file = "../../data/logBB_data/logBB_w_desc.csv"
    logBB_desc_index_file = "../../data/logBB_data/desc_index.txt"
    log.info("===============StartingSVM[Chinese text removed]===============")

    smile_column_name = 'SMILES'
    pred_column_name = 'logBB'
    seed = int(time())

    if not os.path.exists(logBB_data_file):
        raise FileNotFoundError("[Chinese text removed]logBB[Chinese text removed]")

    if os.path.exists(logBB_desc_file):
        log.info("[Chinese text removed]features[Chinese text removed]，[Chinese text removed]")
        df = pd.read_csv(logBB_desc_file, encoding='utf-8')
        y = df[pred_column_name]
        X = df.drop([smile_column_name, pred_column_name], axis=1)
    else:
        log.info("features[Chinese text removed]，[Chinese text removed]features[Chinese text removed]")
        df = pd.read_csv(logBB_data_file, encoding='utf-8')
        df = df.dropna(subset=[pred_column_name])
        df = df.reset_index(drop=True)

        y = df[pred_column_name]
        SMILES = df[smile_column_name]

        X = calculate_Mordred_desc(SMILES)
        log.info(f"[Chinese text removed]features[Chinese text removed]csv[Chinese text removed] {logBB_desc_file} [Chinese text removed]")
        pd.concat([X, y], axis=1).to_csv(logBB_desc_file, encoding='utf-8', index=False)

    # features[Chinese text removed]
    if not os.path.exists(logBB_desc_index_file):
        log.info("[Chinese text removed]features[Chinese text removed]，[Chinese text removed]features[Chinese text removed]")
        log.info(f"[Chinese text removed]features[Chinese text removed]：{X.shape}")
        desc_index = (FeatureExtraction(X, y, VT_threshold=0.02, RFE_features_to_select=50)
                      .feature_extraction(returnIndex=True, index_dtype=int))
        try:
            np.savetxt(logBB_desc_index_file, desc_index, fmt='%d')
            X = X.iloc[:, desc_index]
            log.info(f"features[Chinese text removed]Complete，[Chinese text removed]features[Chinese text removed]：{X.shape}, [Chinese text removed]features[Chinese text removed]：{logBB_desc_index_file}")
        except (TypeError, KeyError) as e:
            log.error(e)
            os.remove(logBB_desc_index_file)
            sys.exit()
    else:
        log.info("[Chinese text removed]features[Chinese text removed]，[Chinese text removed]")
        desc_index = np.loadtxt(logBB_desc_index_file, dtype=int, delimiter=',').tolist()
        X = X.iloc[:, desc_index]
        log.info(f"[Chinese text removed]features[Chinese text removed]Complete，[Chinese text removed]features[Chinese text removed]：{X.shape}")

    log.info("[Chinese text removed]features[Chinese text removed]")
    sc = MinMaxScaler()
    sc.fit(X)
    X = pd.DataFrame(sc.transform(X))

    # Optuna [Chinese text removed]parameter[Chinese text removed]
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=100)

    log.info(f"Bestparameter: {study.best_params}")
    log.info(f"BestRMSE: {study.best_value}")

    # [Chinese text removed]parameter[Chinese text removed]Cross-validation
    best_params = study.best_params

    kf = KFold(n_splits=10, shuffle=True, random_state=seed)
    rmse_list = []
    mse_list = []
    r2_list = []
    adj_r2_list = []
    mae_list = []
    mape_list = []

    for train_index, test_index in kf.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y[train_index], y[test_index]

        model = SVR(**best_params)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)
        adj_r2 = adjusted_r2_score(r2, X_test.shape[0], X_test.shape[1])
        mae = mean_absolute_error(y_test, y_pred)
        mape = calculate_mape(y_test, y_pred)

        rmse_list.append(rmse)
        mse_list.append(mse)
        r2_list.append(r2)
        adj_r2_list.append(adj_r2)
        mae_list.append(mae)
        mape_list.append(mape)

    avg_rmse = np.mean(rmse_list)
    std_rmse = np.std(rmse_list)
    avg_mse = np.mean(mse_list)
    std_mse = np.std(mse_list)
    avg_r2 = np.mean(r2_list)
    std_r2 = np.std(r2_list)
    avg_adj_r2 = np.mean(adj_r2_list)
    std_adj_r2 = np.std(adj_r2_list)
    avg_mae = np.mean(mae_list)
    std_mae = np.std(mae_list)
    avg_mape = np.mean(mape_list)
    std_mape = np.std(mape_list)

    log.info("========SVMCross-validation Results========")
    log.info(f"Avg RMSE: {round(avg_rmse, 3)} ± {round(std_rmse, 3)}")
    log.info(f"Avg MSE: {round(avg_mse, 3)} ± {round(std_mse, 3)}")
    log.info(f"Avg R2: {round(avg_r2, 3)} ± {round(std_r2, 3)}")
    log.info(f"Avg [Chinese text removed]R2: {round(avg_adj_r2, 3)} ± {round(std_adj_r2, 3)}")
    log.info(f"Avg MAE: {round(avg_mae, 3)} ± {round(std_mae, 3)}")
    log.info(f"Avg MAPE: {round(avg_mape, 3)}% ± {round(std_mape, 3)}%")

    # [Chinese text removed]parameter[Chinese text removed]，[Chinese text removed]
    log.info("========[Chinese text removed]========")
    # [Chinese text removed]Dataset split[Chinese text removed]
    # [Chinese text removed] (10%)
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=0.1, random_state=42
    )
    log.info(f"[Chinese text removed]9:1[Chinese text removed]({len(X_train_val)}[Chinese text removed]samples)[Chinese text removed]({len(X_test)}[Chinese text removed]samples)")

    # [Chinese text removed] (10%)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=0.1, random_state=42
    )
    log.info(f"[Chinese text removed]9:1[Chinese text removed]({len(X_train)}[Chinese text removed]samples)[Chinese text removed]({len(X_val)}[Chinese text removed]samples)")

    # [Chinese text removed]Dataset split[Chinese text removed]
    # Training final model
    model_final = SVR(**best_params)
    model_final.fit(X_train, y_train)

    # Make predictions on all datasets
    y_pred_train = model_final.predict(X_train)
    y_pred_test = model_final.predict(X_test)

    # [Chinese text removed]
    mse_test = mean_squared_error(y_test, y_pred_test)
    rmse_test = np.sqrt(mse_test)
    r2_test = r2_score(y_test, y_pred_test)
    mae_test = mean_absolute_error(y_test, y_pred_test)
    mape_test = mean_absolute_percentage_error(y_test, y_pred_test)
    adj_r2_test = adjusted_r2_score(r2_test, len(y_test), X_test.shape[1])

    # [Chinese text removed]
    log.info("[Chinese text removed]：")
    log.info(f"[Chinese text removed] -> MSE: {mse_test:.4f}, RMSE: {rmse_test:.4f}, R2: {r2_test:.4f}, Adjusted R2: {adj_r2_test:.4f}, MAE: {mae_test:.4f}, MAPE: {mape_test:.2f}%")

    # [Chinese text removed]Test Set Results
    test_results = pd.DataFrame({
        'True Values': y_test, 
        'Predicted Values': y_pred_test
    })

    os.makedirs('./result', exist_ok=True)
    test_results.to_csv('./result/svm_mordred_test_results.csv', index=False)

    log.info("\n[Chinese text removed]Dataset split:")
    log.info(f"Training set: {len(X_train)}[Chinese text removed]samples ({len(X_train)/len(X)*100:.1f}%) [[Chinese text removed]81%]")
    log.info(f"Validation set: {len(X_val)}[Chinese text removed]samples ({len(X_val)/len(X)*100:.1f}%) [[Chinese text removed]9%]")
    log.info(f"Test set: {len(X_test)}[Chinese text removed]samples ({len(X_test)/len(X)*100:.1f}%) [[Chinese text removed]10%]")

    # Plot scatter plot
    def plot_scatter(y_train, y_pred_train, y_test, y_pred_test, save_path='./result/svm_mordred_scatter.png'):
        """
        [Chinese text removed]
        """
        plt.figure(figsize=(8, 8))
        
        # [Chinese text removed]
        plt.scatter(y_train, y_pred_train, alpha=0.5, label='Training Set', color='blue')
        
        # [Chinese text removed]
        plt.scatter(y_test, y_pred_test, alpha=0.5, label='Test Set', color='red')
        
        # [Chinese text removed]
        all_min = min(min(y_train), min(y_test))
        all_max = max(max(y_train), max(y_test))
        plt.plot([all_min, all_max], [all_min, all_max], 'k--', lw=2)
        
        plt.xlabel('Experimental logBB')
        plt.ylabel('Predicted logBB')
        plt.title('Predicted vs Experimental logBB')
        
        # [Chinese text removed]R²[Chinese text removed]
        r2_train = r2_score(y_train, y_pred_train)
        r2_test = r2_score(y_test, y_pred_test)
        plt.text(0.05, 0.95, f'Training R² = {r2_train:.3f}', transform=plt.gca().transAxes)
        plt.text(0.05, 0.90, f'Test R² = {r2_test:.3f}', transform=plt.gca().transAxes)
        
        plt.legend()
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()

    # Plot scatter plot
    plot_scatter(y_train, y_pred_train, y_test, y_pred_test)
    log.info("[Chinese text removed] './result/svm_mordred_scatter.png'")

    # Save model
    model_save_path = "./logbbModel/svm_mordred_model.joblib"
    joblib.dump(model_final, model_save_path)
    log.info(f"Model saved[Chinese text removed] {model_save_path}")
