import sys
from time import time
import numpy as np
import optuna
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split, KFold
import pandas as pd
from datetime import datetime
from preprocess.data_preprocess.FeatureExtraction import FeatureExtraction
from preprocess.data_preprocess.data_preprocess_utils import calculate_Mordred_desc
import os
from utils.DataLogger import DataLogger
import lightgbm as lgb
from tqdm import tqdm

# MAPE calculation function
def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    non_zero_index = (y_true != 0)
    return np.mean(np.abs((y_true[non_zero_index] - y_pred[non_zero_index]) / y_true[non_zero_index])) * 100

def adjusted_r2_score(r2, n, k):
    """
    [Chinese text removed]R[Chinese text removed]
    :param r2: R[Chinese text removed]
    :param n: samples[Chinese text removed]
    :param k: [Chinese text removed]
    :return: [Chinese text removed]R[Chinese text removed]
    """
    if n <= k + 1:
        raise ValueError("samples[Chinese text removed]1")
    return 1 - (1 - r2) * (n - 1) / (n - k - 1)

# [Chinese text removed]
os.makedirs("../../data/logBB_data/log", exist_ok=True)
os.makedirs("model", exist_ok=True)

log = DataLogger(log_file=f"../../data/logBB_data/log/{datetime.now().strftime('%Y%m%d')}.log").getlog("ratio_lightgbm_training")

os.makedirs("./logbbModel", exist_ok=True)

def preprocess_data(X, y):
    # 1. [Chinese text removed]features
    constant_features = [col for col in X.columns if X[col].nunique() == 1]
    X = X.drop(columns=constant_features)
    
    # 2. [Chinese text removed]
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.mean())
    
    # 3. [Chinese text removed]
    numeric_features = X.select_dtypes(include=[np.number]).columns
    for col in numeric_features:
        if X[col].skew() > 1:
            X[col] = np.log1p(X[col] - X[col].min() + 1e-6)
    
    return X, y

def objective(trial, X, y, seed):
    param = {
        'objective': 'regression',
        'metric': 'rmse',
        'boosting_type': 'gbdt',
        'num_leaves': trial.suggest_int('num_leaves', 20, 300),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2),
        'feature_fraction': trial.suggest_float('feature_fraction', 0.6, 1.0),
        'bagging_fraction': trial.suggest_float('bagging_fraction', 0.6, 1.0),
        'bagging_freq': trial.suggest_int('bagging_freq', 0, 10),
        'min_child_samples': trial.suggest_int('min_child_samples', 10, 50),
        'min_child_weight': trial.suggest_float('min_child_weight', 1e-3, 0.1),
        'min_split_gain': trial.suggest_float('min_split_gain', 1e-3, 0.1),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-3, 1.0),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-3, 1.0),
        'n_estimators': trial.suggest_int('n_estimators', 500, 3000),
        'max_depth': trial.suggest_int('max_depth', 5, 15),
        'random_state': seed,
        'verbose': -1
    }

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=seed, test_size=0.1)
    model = lgb.LGBMRegressor(**param)
    
    # Remove verbose from fit_params
    fit_params = {
        'eval_set': [(X_test, y_test)],
        'callbacks': [lgb.early_stopping(stopping_rounds=100)]
    }
    
    model.fit(X_train, y_train, **fit_params)
    
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    return r2

if __name__ == '__main__':
    logBB_data_file = "../../data/logBB_data/logBB.csv"
    logBB_desc_file = "../../data/logBB_data/logBB_w_desc.csv"
    logBB_desc_index_file = "../../data/logBB_data/desc_index.txt"
    log.info("===============StartingLightGBM[Chinese text removed]===============")

    smile_column_name = 'SMILES'
    pred_column_name = 'logBB'
    RFE_features_to_select = 50
    n_optuna_trial = 100
    cv_times = 10
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

        # [Chinese text removed]features
        X = calculate_Mordred_desc(SMILES)
        log.info(f"[Chinese text removed]features[Chinese text removed]csv[Chinese text removed] {logBB_desc_file} [Chinese text removed]")
        pd.concat([X, y], axis=1).to_csv(logBB_desc_file, encoding='utf-8', index=False)
        X = X.drop(smile_column_name, axis=1)

    # features[Chinese text removed]
    X = df.drop(['SMILES', 'logBB'], axis=1)

    # [Chinese text removed]features[Chinese text removed] blood mean60min
    if 'blood mean60min' in X.columns:
        X = X.drop('blood mean60min', axis=1)
        log.info("[Chinese text removed]features: blood mean60min")

    # [Chinese text removed]features[Chinese text removed]
    X = X.apply(pd.to_numeric, errors='coerce')  # [Chinese text removed]Converting[Chinese text removed] NaN
    X = X.fillna(0)  # [Chinese text removed]0[Chinese text removed]NaN[Chinese text removed]

    # [Chinese text removed]features[Chinese text removed]
    if os.path.exists(logBB_desc_index_file):
        with open(logBB_desc_index_file, 'r') as f:
            selected_features = f.read().splitlines()
        log.info(f"[Chinese text removed] {len(selected_features)} [Chinese text removed]features[Chinese text removed]")
        # [Chinese text removed]features
        X = X[selected_features]
    else:
        log.warning("features[Chinese text removed]，[Chinese text removed]features")

    # [Chinese text removed]
    sc = MinMaxScaler()
    sc.fit(X)
    X = pd.DataFrame(sc.transform(X))

    # [Chinese text removed]
    X, y = preprocess_data(X, y)

    # [Chinese text removed]
    log.info("[Chinese text removed]LightGBM[Chinese text removed]")
    study = optuna.create_study(direction='maximize')
    study.optimize(lambda trial: objective(trial, X, y, seed), 
                  n_jobs=4, n_trials=n_optuna_trial)

    log.info(f"[Chinese text removed]parameter: {study.best_params}")
    log.info(f"[Chinese text removed]: {study.best_value}")

    # Bestparameter[Chinese text removed]
    model = lgb.LGBMRegressor(**study.best_params)

    # Dataset split
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

    # [Chinese text removed]
    model.fit(X_train, y_train,
             eval_set=[(X_val, y_val)],
             callbacks=[lgb.early_stopping(stopping_rounds=50)])

    # Make predictions on all datasets
    y_pred_test = model.predict(X_test)

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

    # [Chinese text removed] CSV [Chinese text removed]
    train_results = pd.DataFrame({
        'True Values': y_train,
        'Predicted Values': model.predict(X_train)
    })
    train_results.to_csv('./result/lightgbm_mordred_train_results.csv', index=False)
    test_results.to_csv('./result/lightgbm_mordred_test_results.csv', index=False)
    log.info("[Chinese text removed] './result/lightgbm_mordred_train_results.csv' [Chinese text removed] './result/lightgbm_mordred_test_results.csv'")

    # [Chinese text removed]
    model_save_path = os.path.join("./logbbModel", "lightgbm_mordred_model.joblib")
    model.booster_.save_model(model_save_path)
    log.info(f"Model saved[Chinese text removed] {model_save_path}")

    log.info("\n[Chinese text removed]Dataset split:")
    log.info(f"Training set: {len(X_train)}[Chinese text removed]samples ({len(X_train)/len(X)*100:.1f}%) [[Chinese text removed]81%]")
    log.info(f"Validation set: {len(X_val)}[Chinese text removed]samples ({len(X_val)/len(X)*100:.1f}%) [[Chinese text removed]9%]")
    log.info(f"Test set: {len(X_test)}[Chinese text removed]samples ({len(X_test)/len(X)*100:.1f}%) [[Chinese text removed]10%]")
