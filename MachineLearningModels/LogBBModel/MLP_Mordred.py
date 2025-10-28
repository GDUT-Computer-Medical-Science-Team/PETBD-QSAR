import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, mean_absolute_percentage_error
from sklearn.model_selection import KFold, train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime
import os
import optuna
from utils.DataLogger import DataLogger
from preprocess.data_preprocess.FeatureExtraction import FeatureExtraction
from preprocess.data_preprocess.data_preprocess_utils import calculate_Mordred_desc
import joblib

# [Chinese text removed]
os.makedirs("../../data/logBB_data/log", exist_ok=True)
os.makedirs("model", exist_ok=True)

log = DataLogger(log_file=f"../../data/logBB_data/log/{datetime.now().strftime('%Y%m%d')}.log").getlog("mlp_cross_validation")

def calculate_mape(y_true, y_pred):
    epsilon = 1e-8  # [Chinese text removed]，[Chinese text removed]
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / (y_true + epsilon))) * 100

def objective(trial):
    # [Chinese text removed]parameter[Chinese text removed]
    param = {
        'hidden_layer_sizes': trial.suggest_categorical('hidden_layer_sizes', [(50,), (100,), (150,), (100, 50)]),
        'activation': trial.suggest_categorical('activation', ['relu', 'tanh']),
        'solver': trial.suggest_categorical('solver', ['adam', 'sgd']),
        'alpha': trial.suggest_loguniform('alpha', 1e-5, 1e-1),
        'learning_rate': trial.suggest_categorical('learning_rate', ['constant', 'invscaling', 'adaptive']),
        'learning_rate_init': trial.suggest_loguniform('learning_rate_init', 1e-5, 1e-1),
        'random_state': seed
    }

    model = MLPRegressor(**param)
    kf = KFold(n_splits=5, shuffle=True, random_state=seed)
    rmse_list = []

    for train_index, test_index in kf.split(X_scaled):
        X_train, X_test = X_scaled[train_index], X_scaled[test_index]
        y_train, y_test = y[train_index], y[test_index]

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        rmse_list.append(rmse)

    return np.mean(rmse_list)

def adjusted_r2(y_true, y_pred, n_features):
    """
    [Chinese text removed]R[Chinese text removed]
    :param y_true: true values
    :param y_pred: predicted values
    :param n_features: features[Chinese text removed]
    :return: [Chinese text removed]R[Chinese text removed]
    """
    r2 = r2_score(y_true, y_pred)
    n = len(y_true)
    return 1 - (1 - r2) * (n - 1) / (n - n_features - 1)

if __name__ == '__main__':
    # [Chinese text removed]
    logBB_data_file = "../../data/logBB_data/logBB.csv"
    logBB_desc_file = "../../data/logBB_data/logBB_w_desc.csv"
    logBB_desc_index_file = "../../data/logBB_data/desc_index.txt"
    model_dir = "./logbbModel"

    log.info("===============StartingMLPCross-validation[Chinese text removed]===============")

    # [Chinese text removed]
    smile_column_name = 'SMILES'
    pred_column_name = 'logBB'
    seed = 42  # [Chinese text removed]

    if not os.path.exists(logBB_data_file):
        raise FileNotFoundError("[Chinese text removed]logBB[Chinese text removed]")

    # [Chinese text removed]features[Chinese text removed]
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

        # [Chinese text removed]features[Chinese text removed] blood mean60min
        if 'blood mean60min' in df.columns:
            df = df.drop('blood mean60min', axis=1)
            log.info("[Chinese text removed]features: blood mean60min")

        # [Chinese text removed]features[Chinese text removed]
        X = df.drop(['SMILES', 'logBB'], axis=1)
        X = X.apply(pd.to_numeric, errors='coerce')  # [Chinese text removed]Converting[Chinese text removed] NaN
        X = X.fillna(0)  # [Chinese text removed]0[Chinese text removed]NaN[Chinese text removed]

        log.info(f"[Chinese text removed]features[Chinese text removed]csv[Chinese text removed] {logBB_desc_file} [Chinese text removed]")
        pd.concat([X, y], axis=1).to_csv(logBB_desc_file, encoding='utf-8', index=False)

    # features[Chinese text removed]
    if not os.path.exists(logBB_desc_index_file):
        log.info("[Chinese text removed]features[Chinese text removed]，[Chinese text removed]features[Chinese text removed]")
        log.info(f"[Chinese text removed]features[Chinese text removed]：{X.shape}")
        desc_index = FeatureExtraction(X, y, VT_threshold=0.02, RFE_features_to_select=50).feature_extraction(returnIndex=True, index_dtype=int)
        np.savetxt(logBB_desc_index_file, desc_index, fmt='%d')
        X = X.iloc[:, desc_index]
        log.info(f"features[Chinese text removed]Complete，[Chinese text removed]features[Chinese text removed]：{X.shape}, [Chinese text removed]features[Chinese text removed]：{logBB_desc_index_file}")
    else:
        log.info("[Chinese text removed]features[Chinese text removed]，[Chinese text removed]")
        desc_index = np.loadtxt(logBB_desc_index_file, dtype=int).tolist()
        X = X.iloc[:, desc_index]
        log.info(f"[Chinese text removed]features[Chinese text removed]Complete，[Chinese text removed]features[Chinese text removed]：{X.shape}")

    # Feature normalization
    log.info("[Chinese text removed]features[Chinese text removed]")
    sc = MinMaxScaler()
    X_scaled = sc.fit_transform(X)

    # [Chinese text removed] (10%)
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X_scaled, y, test_size=0.1, random_state=42
    )
    log.info(f"[Chinese text removed]9:1[Chinese text removed]({len(X_train_val)}[Chinese text removed]samples)[Chinese text removed]({len(X_test)}[Chinese text removed]samples)")

    # [Chinese text removed] (10%)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=0.1, random_state=42
    )
    log.info(f"[Chinese text removed]9:1[Chinese text removed]({len(X_train)}[Chinese text removed]samples)[Chinese text removed]({len(X_val)}[Chinese text removed]samples)")

    # Optuna [Chinese text removed]parameter[Chinese text removed]
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=100)

    log.info(f"Bestparameter: {study.best_params}")
    log.info(f"BestRMSE: {study.best_value}")

    # [Chinese text removed]parameter[Chinese text removed]
    best_params = study.best_params
    model = MLPRegressor(**best_params)

    # [Chinese text removed]Cross-validation
    cv = KFold(n_splits=10, shuffle=True, random_state=seed)

    # [Chinese text removed]
    rmse_scores = []
    r2_scores = []
    adj_r2_scores = []
    mae_scores = []
    mse_scores = []
    mape_scores = []

    # [Chinese text removed]Cross-validation
    for train_idx, test_idx in cv.split(X_train):
        X_train_cv, X_test_cv = X_train[train_idx], X_train[test_idx]
        y_train_cv, y_test_cv = y_train.iloc[train_idx], y_train.iloc[test_idx]

        # [Chinese text removed]
        model.fit(X_train_cv, y_train_cv)

        # [Chinese text removed]
        y_pred_cv = model.predict(X_test_cv)

        # [Chinese text removed]
        rmse = np.sqrt(mean_squared_error(y_test_cv, y_pred_cv))
        r2 = r2_score(y_test_cv, y_pred_cv)
        adj_r2 = adjusted_r2(y_test_cv, y_pred_cv, X_train_cv.shape[1])
        mae = mean_absolute_error(y_test_cv, y_pred_cv)
        mse = mean_squared_error(y_test_cv, y_pred_cv)
        mape = calculate_mape(y_test_cv, y_pred_cv)

        # [Chinese text removed]
        rmse_scores.append(rmse)
        r2_scores.append(r2)
        adj_r2_scores.append(adj_r2)
        mae_scores.append(mae)
        mse_scores.append(mse)
        mape_scores.append(mape)

    # [Chinese text removed]
    log.info("========[Chinese text removed]Cross-validation Results========")
    log.info(f"RMSE: {np.mean(rmse_scores):.3f}±{np.std(rmse_scores):.3f}")
    log.info(f"R2: {np.mean(r2_scores):.3f}±{np.std(r2_scores):.3f}")
    log.info(f"Adjusted R2: {np.mean(adj_r2_scores):.3f}±{np.std(adj_r2_scores):.3f}")
    log.info(f"MAE: {np.mean(mae_scores):.3f}±{np.std(mae_scores):.3f}")
    log.info(f"MSE: {np.mean(mse_scores):.3f}±{np.std(mse_scores):.3f}")
    log.info(f"MAPE: {np.mean(mape_scores):.3f}±{np.std(mape_scores):.3f}%")

    # [Chinese text removed]
    model.fit(X_train, y_train)
    y_pred_test = model.predict(X_test)

    # [Chinese text removed]
    mse_test = mean_squared_error(y_test, y_pred_test)
    rmse_test = np.sqrt(mse_test)
    r2_test = r2_score(y_test, y_pred_test)
    mae_test = mean_absolute_error(y_test, y_pred_test)
    mape_test = mean_absolute_percentage_error(y_test, y_pred_test)
    adj_r2_test = 1 - (1 - r2_test) * (len(y_test) - 1) / (len(y_test) - X_test.shape[1] - 1)

    # [Chinese text removed]
    log.info("========Test Set Results========")
    log.info(f"RMSE: {rmse_test:.3f}")
    log.info(f"R2: {r2_test:.3f}")
    log.info(f"Adjusted R2: {adj_r2_test:.3f}")
    log.info(f"MAE: {mae_test:.3f}")
    log.info(f"MSE: {mse_test:.3f}")
    log.info(f"MAPE: {mape_test:.3f}%")

    # [Chinese text removed]
    joblib.dump(model, os.path.join(model_dir, "mlp_mordred_model.joblib"))
    log.info(f"Model saved[Chinese text removed] {os.path.join(model_dir, 'mlp_mordred_model.joblib')}")

    # [Chinese text removed]Test Set Results
    test_results = pd.DataFrame({
        'True Values': y_test, 
        'Predicted Values': y_pred_test
    })

    os.makedirs('./result', exist_ok=True)
    test_results.to_csv('./result/mlp_mordred_test_results.csv', index=False)

    log.info("\n[Chinese text removed]Dataset split:")
    log.info(f"Training set: {len(X_train)}[Chinese text removed]samples ({len(X_train)/len(X)*100:.1f}%) [[Chinese text removed]81%]")
    log.info(f"Validation set: {len(X_val)}[Chinese text removed]samples ({len(X_val)/len(X)*100:.1f}%) [[Chinese text removed]9%]")
    log.info(f"Test set: {len(X_test)}[Chinese text removed]samples ({len(X_test)/len(X)*100:.1f}%) [[Chinese text removed]10%]")

    # [Chinese text removed] CSV [Chinese text removed]
    train_results = pd.DataFrame({
        'True Values': y_train,
        'Predicted Values': model.predict(X_train)
    })
    train_results.to_csv('./result/mlp_mordred_train_results.csv', index=False)
    test_results.to_csv('./result/mlp_mordred_test_results.csv', index=False)
    log.info("[Chinese text removed] './result/mlp_mordred_train_results.csv' [Chinese text removed] './result/mlp_mordred_test_results.csv'")

    # [Chinese text removed]
    model_save_path = os.path.join(model_dir, "mlp_mordred_model.joblib")
