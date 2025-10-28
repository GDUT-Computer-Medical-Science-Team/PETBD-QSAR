import sys
from time import time
import numpy as np
import optuna
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, mean_absolute_percentage_error
from sklearn.model_selection import train_test_split, KFold
import pandas as pd
from tqdm import tqdm
from datetime import datetime
from preprocess.data_preprocess.FeatureExtraction import FeatureExtraction
from preprocess.data_preprocess.data_preprocess_utils import calculate_Mordred_desc
import joblib
import os
from utils.DataLogger import DataLogger

log = DataLogger(log_file=f"../../data/logBB_data/log/{datetime.now().strftime('%Y%m%d')}.log").getlog("ratio_rf_training")

os.makedirs("./logbbModel", exist_ok=True)

def adjusted_r2(y_true, y_pred, num_features):
    """
    Calculate adjusted R²
    :param y_true: true values
    :param y_pred: predicted values
    :param num_features: number of features used by the model
    :return: adjusted R²
    """
    n = len(y_true)
    r2 = r2_score(y_true, y_pred)
    # avoid division by zero
    if n - num_features - 1 == 0:
        return r2
    return 1 - (1 - r2) * (n - 1) / (n - num_features - 1)

if __name__ == '__main__':
    # [Chinese text removed]
    logBB_data_file = "../../data/logBB_data/logBB.csv"
    logBB_desc_file = "../../data/logBB_data/logBB_w_desc.csv"
    logBB_desc_index_file = "../../data/logBB_data/desc_index.txt"
    test_data_file = "../../data/logBB_data/logBB_test.csv"
    log.info("===============StartingRandom Forest[Chinese text removed]===============")
    
    # [Chinese text removed]
    smile_column_name = 'SMILES'
    pred_column_name = 'logBB'
    RFE_features_to_select = 50
    n_optuna_trial = 100
    cv_times = 10
    seed = int(time())

    if not os.path.exists(logBB_data_file):
        raise FileNotFoundError("[Chinese text removed]logBB[Chinese text removed]")
        
    # features[Chinese text removed]
    if os.path.exists(logBB_desc_file):
        log.info("[Chinese text removed]features[Chinese text removed]，[Chinese text removed]")
        df = pd.read_csv(logBB_desc_file, encoding='utf-8')
        y = df[pred_column_name]
        X = df.drop([smile_column_name, pred_column_name], axis=1)
    else:
        log.info("features[Chinese text removed]，[Chinese text removed]features[Chinese text removed]") 
        df = pd.read_csv(logBB_data_file, encoding='utf-8')
        df = df.dropna(subset=[pred_column_name])
        df = df.reset_index()
        
        y = df[pred_column_name]
        SMILES = df[smile_column_name]
        
        X = calculate_Mordred_desc(SMILES)
        log.info(f"[Chinese text removed]features[Chinese text removed]csv[Chinese text removed] {logBB_desc_file} [Chinese text removed]")
        pd.concat([X, y], axis=1).to_csv(logBB_desc_file, encoding='utf-8', index=False)
        X = X.drop(smile_column_name, axis=1)

    # [Chinese text removed]features[Chinese text removed] blood mean60min
    if 'blood mean60min' in X.columns:
        X = X.drop('blood mean60min', axis=1)
        log.info("[Chinese text removed]features: blood mean60min")

    # Feature normalization
    log.info("[Chinese text removed]features[Chinese text removed]")
    sc = MinMaxScaler()
    sc.fit(X)
    X = pd.DataFrame(sc.transform(X))

    # features[Chinese text removed]
    if not os.path.exists(logBB_desc_index_file):
        log.info("[Chinese text removed]features[Chinese text removed]，[Chinese text removed]features[Chinese text removed]")
        log.info(f"[Chinese text removed]features[Chinese text removed]：{X.shape}")
        desc_index = (FeatureExtraction(X,
                                      y,
                                      VT_threshold=0.02,
                                      RFE_features_to_select=RFE_features_to_select).
                     feature_extraction(returnIndex=True, index_dtype=int))
        try:
            np.savetxt(logBB_desc_index_file, desc_index, fmt='%d')
            X = X[desc_index]
            log.info(f"features[Chinese text removed]Complete，[Chinese text removed]features[Chinese text removed]：{X.shape}, [Chinese text removed]features[Chinese text removed]：{logBB_desc_index_file}")
        except (TypeError, KeyError) as e:
            log.error(e)
            os.remove(logBB_desc_index_file)
            sys.exit()
    else:
        log.info("[Chinese text removed]features[Chinese text removed]，[Chinese text removed]")
        desc_index = np.loadtxt(logBB_desc_index_file, dtype=int, delimiter=',').tolist()
        X = X[desc_index]
        log.info(f"[Chinese text removed]features[Chinese text removed]Complete，[Chinese text removed]features[Chinese text removed]：{X.shape}")

    # [Chinese text removed]
    X.columns = X.columns.astype(str)

    # 1. [Chinese text removed]Dataset split
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
    def objective(trial):
        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=seed, test_size=0.1)
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 300),
            'max_depth': trial.suggest_int('max_depth', 10, 50),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
            'random_state': seed
        }
        model = RandomForestRegressor(**params)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        return r2

    log.info("[Chinese text removed]Random Forest[Chinese text removed]")
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_jobs=4, n_trials=n_optuna_trial)

    log.info(f"[Chinese text removed]parameter: {study.best_params}")
    log.info(f"[Chinese text removed]: {study.best_value}")

    # [Chinese text removed]parameter[Chinese text removed]
    model = RandomForestRegressor(**study.best_params)

    # [Chinese text removed]Cross-validation
    cv = KFold(n_splits=cv_times, random_state=seed, shuffle=True)
    rmse_result_list = []
    r2_result_list = []
    mse_result_list = []
    mae_result_list = []
    adj_r2_result_list = []
    
    log.info(f"[Chinese text removed]parameter[Chinese text removed]{cv_times}[Chinese text removed]Cross-validation")

    for idx, (train_idx, test_idx) in tqdm(enumerate(cv.split(X, y)), desc="Cross-validation: ", total=cv_times):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        adj_r2 = 1 - (1-r2)*(len(y_test)-1)/(len(y_test)-X_test.shape[1]-1)

        rmse_result_list.append(rmse)
        r2_result_list.append(r2)
        mse_result_list.append(mse)
        mae_result_list.append(mae)
        adj_r2_result_list.append(adj_r2)

    log.info(f"[Chinese text removed]: {seed}")
    log.info("========[Chinese text removed]========")
    log.info(f"RMSE: {round(np.mean(rmse_result_list), 3)}±{round(np.std(rmse_result_list), 3)}")
    log.info(f"MSE: {round(np.mean(mse_result_list), 3)}±{round(np.std(mse_result_list), 3)}")
    log.info(f"MAE: {round(np.mean(mae_result_list), 3)}±{round(np.std(mae_result_list), 3)}")
    log.info(f"R2: {round(np.mean(r2_result_list), 3)}±{round(np.std(r2_result_list), 3)}")
    log.info(f"Adjusted R2: {round(np.mean(adj_r2_result_list), 3)}±{round(np.std(adj_r2_result_list), 3)}")

    # [Chinese text removed]
    y_pred = model.predict(X_val)
    rmse = np.sqrt(mean_squared_error(y_val, y_pred))
    mse = mean_squared_error(y_val, y_pred)
    mae = mean_absolute_error(y_val, y_pred)
    r2 = r2_score(y_val, y_pred)
    adj_r2 = 1 - (1-r2)*(len(y_val)-1)/(len(y_val)-X_val.shape[1]-1)
    
    log.info("========[Chinese text removed]========")
    log.info(f"RMSE: {round(rmse, 3)}")
    log.info(f"MSE: {round(mse, 3)}")
    log.info(f"MAE: {round(mae, 3)}")
    log.info(f"R2: {round(r2, 3)}")
    log.info(f"Adjusted R2: {round(adj_r2, 3)}")

    model_save_path = os.path.join("./logbbModel", "rf_mordred_model.joblib")
    joblib.dump(model, model_save_path)
    log.info(f"Model saved[Chinese text removed] {model_save_path}")

    # 2. [Chinese text removed]
    # Make predictions on all datasets
    y_pred_train = model.predict(X_train)
    y_pred_val = model.predict(X_val)
    y_pred_test = model.predict(X_test)

    # Calculate evaluation metrics
    mse_train = mean_squared_error(y_train, y_pred_train)
    rmse_train = np.sqrt(mse_train)
    r2_train = r2_score(y_train, y_pred_train)
    mae_train = mean_absolute_error(y_train, y_pred_train)
    mape_train = mean_absolute_percentage_error(y_train, y_pred_train)
    adj_r2_train = adjusted_r2(y_train, y_pred_train, X_train.shape[1])

    # [Chinese text removed]
    mse_val = mean_squared_error(y_val, y_pred_val)
    rmse_val = np.sqrt(mse_val)
    r2_val = r2_score(y_val, y_pred_val)
    mae_val = mean_absolute_error(y_val, y_pred_val)
    mape_val = mean_absolute_percentage_error(y_val, y_pred_val)
    adj_r2_val = adjusted_r2(y_val, y_pred_val, X_val.shape[1])

    # [Chinese text removed]
    mse_test = mean_squared_error(y_test, y_pred_test)
    rmse_test = np.sqrt(mse_test)
    r2_test = r2_score(y_test, y_pred_test)
    mae_test = mean_absolute_error(y_test, y_pred_test)
    mape_test = mean_absolute_percentage_error(y_test, y_pred_test)
    adj_r2_test = adjusted_r2(y_test, y_pred_test, X_test.shape[1])

    # [Chinese text removed]
    log.info("[Chinese text removed]：")
    log.info(f"[Chinese text removed] -> MSE: {mse_test:.4f}, RMSE: {rmse_test:.4f}, R2: {r2_test:.4f}, Adjusted R2: {adj_r2_test:.4f}, MAE: {mae_test:.4f}, MAPE: {mape_test:.2f}%")

    # [Chinese text removed]Test Set Results
    test_results = pd.DataFrame({
        'True Values': y_test, 
        'Predicted Values': y_pred_test
    })

    os.makedirs('./result', exist_ok=True)
    test_results.to_csv('./result/rf_mordred_test_results.csv', index=False)

    log.info("\n[Chinese text removed]Dataset split:")
    log.info(f"Training set: {len(X_train)}[Chinese text removed]samples ({len(X_train)/len(X)*100:.1f}%) [[Chinese text removed]81%]")
    log.info(f"Validation set: {len(X_val)}[Chinese text removed]samples ({len(X_val)/len(X)*100:.1f}%) [[Chinese text removed]9%]")
    log.info(f"Test set: {len(X_test)}[Chinese text removed]samples ({len(X_test)/len(X)*100:.1f}%) [[Chinese text removed]10%]")

    # [Chinese text removed] CSV [Chinese text removed]
    train_results = pd.DataFrame({
        'True Values': y_train,
        'Predicted Values': y_pred_train
    })
    train_results.to_csv('./result/rf_mordred_train_results.csv', index=False)
    test_results.to_csv('./result/rf_mordred_test_results.csv', index=False)
    log.info("[Chinese text removed] './result/rf_mordred_train_results.csv' [Chinese text removed] './result/rf_mordred_test_results.csv'")

    # [Chinese text removed]
    model_save_path = os.path.join("./logbbModel", "rf_mordred_model.joblib")

    # [Chinese text removed]Cross-validation
    kf = KFold(n_splits=10, shuffle=True, random_state=42)
    metrics_list = []

    for fold_num, (train_index, val_index) in enumerate(kf.split(X)):
        X_train_cv, X_val_cv = X.iloc[train_index], X.iloc[val_index]
        y_train_cv, y_val_cv = y.iloc[train_index], y.iloc[val_index]

        # [Chinese text removed]parameter[Chinese text removed]
        model_cv = RandomForestRegressor(**study.best_params)
        model_cv.fit(X_train_cv, y_train_cv)

        # [Chinese text removed]
        y_val_pred = model_cv.predict(X_val_cv)

        # Calculate evaluation metrics
        mse_val = mean_squared_error(y_val_cv, y_val_pred)
        rmse_val = np.sqrt(mse_val)
        r2_val = r2_score(y_val_cv, y_val_pred)
        mae_val = mean_absolute_error(y_val_cv, y_val_pred)
        mape_val = mean_absolute_percentage_error(y_val_cv, y_val_pred)
        adj_r2_val = adjusted_r2(y_val_cv, y_val_pred, X_val_cv.shape[1])

        metrics_list.append({
            'fold': fold_num + 1,
            'mse': mse_val,
            'rmse': rmse_val,
            'r2': r2_val,
            'mae': mae_val,
            'mape': mape_val,
            'adj_r2': adj_r2_val
        })

        log.info(f"Fold {fold_num + 1} - MSE: {mse_val:.4f}, RMSE: {rmse_val:.4f}, R2: {r2_val:.4f}, Adjusted R2: {adj_r2_val:.4f}, MAE: {mae_val:.4f}, MAPE: {mape_val:.2f}%")

    # [Chinese text removed]
    avg_metrics = {key: np.mean([m[key] for m in metrics_list]) for key in metrics_list[0] if key != 'fold'}
    log.info("10[Chinese text removed]Cross-validation[Chinese text removed]：")
    log.info(f"MSE: {avg_metrics['mse']:.4f}")
    log.info(f"RMSE: {avg_metrics['rmse']:.4f}")
    log.info(f"R2: {avg_metrics['r2']:.4f}")
    log.info(f"Adjusted R2: {avg_metrics['adj_r2']:.4f}")
    log.info(f"MAE: {avg_metrics['mae']:.4f}")
    log.info(f"MAPE: {avg_metrics['mape']:.2f}%")

    # [Chinese text removed]
    model_final = RandomForestRegressor(**study.best_params)
    model_final.fit(X_train, y_train)
    y_pred_test = model_final.predict(X_test)
    
    mse_test = mean_squared_error(y_test, y_pred_test)
    rmse_test = np.sqrt(mse_test)
    r2_test = r2_score(y_test, y_pred_test)
    mae_test = mean_absolute_error(y_test, y_pred_test)
    mape_test = mean_absolute_percentage_error(y_test, y_pred_test)
    adj_r2_test = adjusted_r2(y_test, y_pred_test, X_test.shape[1])

    log.info("[Chinese text removed]：")
    log.info(f"MSE: {mse_test:.4f}, RMSE: {rmse_test:.4f}, R2: {r2_test:.4f}, Adjusted R2: {adj_r2_test:.4f}, MAE: {mae_test:.4f}, MAPE: {mape_test:.2f}%")

    # [Chinese text removed] CSV [Chinese text removed]
    train_results = pd.DataFrame({
        'True Values': y_train,
        'Predicted Values': model_final.predict(X_train)
    })
    test_results = pd.DataFrame({
        'True Values': y_test,
        'Predicted Values': y_pred_test
    })
    
    os.makedirs('./result', exist_ok=True)
    train_results.to_csv('./result/rf_mordred_train_results.csv', index=False)
    test_results.to_csv('./result/rf_mordred_test_results.csv', index=False)
    log.info("[Chinese text removed] './result/rf_mordred_train_results.csv' [Chinese text removed] './result/rf_mordred_test_results.csv'")

    # Save model
    model_save_path = os.path.join("./logbbModel", "rf_mordred_model.joblib")
    joblib.dump(model_final, model_save_path)
    log.info(f"Model saved[Chinese text removed] {model_save_path}")

    # [Chinese text removed]
    loaded_model = joblib.load(model_save_path)
    y_pred_test_loaded = loaded_model.predict(X_test)
    
    # [Chinese text removed]
    assert np.allclose(y_pred_test, y_pred_test_loaded), "[Chinese text removed]original[Chinese text removed]！"
    log.info("[Chinese text removed]Success：[Chinese text removed]")
