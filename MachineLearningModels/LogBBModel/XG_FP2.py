import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime
import os
import matplotlib.pyplot as plt
from utils.DataLogger import DataLogger
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import xgboost as xgb
import optuna
import tempfile
from padelpy import padeldescriptor
import joblib
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestRegressor  # [Chinese text removed] RandomForestRegressor

# [Chinese text removed]
log = DataLogger(log_file=f"../../data/logBB_data/log/{datetime.now().strftime('%Y%m%d')}.log").getlog(
    "xgboost_feature_selection")

# Calculate evaluation metrics[Chinese text removed]
def calculate_metrics(y_train, y_pred_train, y_val, y_pred_val, y_test, y_pred_test, num_features):
    """
    [Chinese text removed]、[Chinese text removed]
    """
    metrics = {}
    
    # [Chinese text removed]
    metrics['train_mse'] = mean_squared_error(y_train, y_pred_train)
    metrics['train_rmse'] = np.sqrt(metrics['train_mse'])
    metrics['train_r2'] = r2_score(y_train, y_pred_train)
    metrics['train_mae'] = mean_absolute_error(y_train, y_pred_train)
    metrics['train_mape'] = mean_absolute_percentage_error(y_train, y_pred_train)
    metrics['train_adj_r2'] = adjusted_r2(y_train, y_pred_train, num_features)
    
    # [Chinese text removed]
    metrics['val_mse'] = mean_squared_error(y_val, y_pred_val)
    metrics['val_rmse'] = np.sqrt(metrics['val_mse'])
    metrics['val_r2'] = r2_score(y_val, y_pred_val)
    metrics['val_mae'] = mean_absolute_error(y_val, y_pred_val)
    metrics['val_mape'] = mean_absolute_percentage_error(y_val, y_pred_val)
    metrics['val_adj_r2'] = adjusted_r2(y_val, y_pred_val, num_features)
    
    # [Chinese text removed]
    metrics['test_mse'] = mean_squared_error(y_test, y_pred_test)
    metrics['test_rmse'] = np.sqrt(metrics['test_mse'])
    metrics['test_r2'] = r2_score(y_test, y_pred_test)
    metrics['test_mae'] = mean_absolute_error(y_test, y_pred_test)
    metrics['test_mape'] = mean_absolute_percentage_error(y_test, y_pred_test)
    metrics['test_adj_r2'] = adjusted_r2(y_test, y_pred_test, num_features)

    return metrics

def calculate_padel_fingerprints(smiles_list):
    """
    [Chinese text removed]PaDEL[Chinese text removed]
    :param smiles_list: SMILES [Chinese text removed]
    :return: DataFrame，[Chinese text removed]
    """
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
        log.error(f"[Chinese text removed]PaDEL[Chinese text removed]Error: {str(e)}")
        raise
    finally:
        if os.path.exists(temp_smi_file):
            os.remove(temp_smi_file)


def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    non_zero_index = (y_true != 0)
    return np.mean(np.abs((y_true[non_zero_index] - y_pred[non_zero_index]) / y_true[non_zero_index])) * 100


def adjusted_r2(y_true, y_pred, num_features):
    """
    [Chinese text removed] R^2
    :param y_true: true values
    :param y_pred: predicted values
    :param num_features: number of features used by the model
    :return: [Chinese text removed] R^2
    """
    n = len(y_true)
    r2 = r2_score(y_true, y_pred)
    # avoid division by zero
    if n - num_features - 1 == 0:
        return r2
    return 1 - (1 - r2) * (n - 1) / (n - num_features - 1)


def plot_scatter(y_train, y_pred_train, y_test, y_pred_test, save_path=None):
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
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()


def objective(trial, X, y):
    """
    Optuna [Chinese text removed]，[Chinese text removed]parameter[Chinese text removed]
    """
    param = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 300),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 7),
        'random_state': 42
    }
    
    # [Chinese text removed] 80/20 [Chinese text removed]
    X_train_split, X_valid_split, y_train_split, y_valid_split = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    model = xgb.XGBRegressor(**param)
    model.fit(
        X_train_split, 
        y_train_split,
        eval_set=[(X_valid_split, y_valid_split)],
        early_stopping_rounds=50,
        verbose=False
    )
    
    y_pred = model.predict(X_valid_split)
    rmse = np.sqrt(mean_squared_error(y_valid_split, y_pred))
    return rmse


def main():
    # [Chinese text removed]parameter
    base_params = {
        'n_estimators': 100,
        'max_depth': 10,
        'learning_rate': 0.1,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'random_state': 42
    }

    # [Chinese text removed]
    logBB_data_file = "../../data/logBB_data/logBB.csv"
    logBB_desc_file = "../../data/logBB_data/logBB_w_desc_fp.csv"
    feature_index_path = "../../data/logBB_data/logBB_w_desc_fp_index2.txt"  # features[Chinese text removed]
    log.info("===============Starting XGBoost features[Chinese text removed]===============")

    # [Chinese text removed]
    if not os.path.exists(logBB_data_file):
        log.error("[Chinese text removed]logBB[Chinese text removed]")
        raise FileNotFoundError("[Chinese text removed]logBB[Chinese text removed]")

    # [Chinese text removed]
    df = pd.read_csv(logBB_data_file, encoding='utf-8')
    df = df.dropna(subset=['logBB']).reset_index(drop=True)
    SMILES = df['SMILES']
    log.info(f"[Chinese text removed] {len(SMILES)} [Chinese text removed] SMILES [Chinese text removed]")

    # [Chinese text removed]
    df_desc = calculate_padel_fingerprints(SMILES)
    log.info(f"[Chinese text removed] {df_desc.shape[0]} [Chinese text removed]，[Chinese text removed] {df_desc.shape[1]} [Chinese text removed]")

    # [Chinese text removed]original[Chinese text removed]
    df_final = pd.concat([df, df_desc.drop('SMILES', axis=1)], axis=1)

    log.info(f"[Chinese text removed]: {df_final.shape}")

    # features[Chinese text removed]
    X_initial = df_final.drop(['SMILES', 'logBB'], axis=1)


    y = df_final['logBB']
    log.info(f"[Chinese text removed] {len(X_initial)} [Chinese text removed]features")

    # [Chinese text removed]features[Chinese text removed]
    X_initial = X_initial.apply(pd.to_numeric, errors='coerce')  # [Chinese text removed]Converting[Chinese text removed] NaN
    X_initial = X_initial.fillna(0)  # [Chinese text removed]0[Chinese text removed]NaN[Chinese text removed]

    # features[Chinese text removed]
    if not os.path.exists(feature_index_path):
        log.info("[Chinese text removed]features[Chinese text removed]，[Chinese text removed]features[Chinese text removed]")
        
        # [Chinese text removed]RFE[Chinese text removed]features[Chinese text removed]
        model_rfe = xgb.XGBRegressor()
        rfe = RFE(estimator=model_rfe, n_features_to_select=50)  # [Chinese text removed]50[Chinese text removed]features
        rfe.fit(X_initial, y)
        
        # [Chinese text removed]features[Chinese text removed]
        selected_features = X_initial.columns[rfe.support_]
        log.info(f"[Chinese text removed]features: {selected_features.tolist()}")
        
        # Save feature indices
        desc_index = np.where(rfe.support_)[0]
        np.savetxt(feature_index_path, desc_index, fmt='%d')
        log.info(f"Feature indices saved[Chinese text removed]：{feature_index_path}")
        
        X = X_initial[selected_features]  # [Chinese text removed]features
        log.info(f"[Chinese text removed]features[Chinese text removed]：{X.shape}")
        log.info(f"[Chinese text removed]features[Chinese text removed]: {X.columns.tolist()}")
    else:
        log.info("[Chinese text removed]features[Chinese text removed]，[Chinese text removed]")
        desc_index = np.loadtxt(feature_index_path, dtype=int, delimiter=',').tolist()
        log.info(f"[Chinese text removed]features[Chinese text removed]Complete，[Chinese text removed]: {desc_index}")

        # Check if desc_index is valid
        if any(i >= X_initial.shape[1] for i in desc_index):
            log.error("features[Chinese text removed]，[Chinese text removed]。")
            raise IndexError("features[Chinese text removed]。")

        X = X_initial.iloc[:, desc_index]  # Use iloc to select columns by index
        log.info(f"[Chinese text removed]features[Chinese text removed]：{X.shape}")
        log.info(f"[Chinese text removed]features[Chinese text removed]: {X.columns.tolist()}")

    # Dataset split
    X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
    log.info(f"[Chinese text removed]9:1[Chinese text removed]({len(X_train_val)}[Chinese text removed]samples)[Chinese text removed]({len(X_test)}[Chinese text removed]samples)")

    # [Chinese text removed] (10%)
    X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.1, random_state=42)
    log.info(f"[Chinese text removed]9:1[Chinese text removed]({len(X_train)}[Chinese text removed]samples)[Chinese text removed]({len(X_val)}[Chinese text removed]samples)")

    # [Chinese text removed]
    sc = MinMaxScaler()
    X_train_scaled = sc.fit_transform(X_train)
    X_val_scaled = sc.transform(X_val)
    X_test_scaled = sc.transform(X_test)

    # Converting[Chinese text removed]DataFrame[Chinese text removed]
    X_train = pd.DataFrame(X_train_scaled, columns=X_train.columns)
    X_val = pd.DataFrame(X_val_scaled, columns=X_val.columns)
    X_test = pd.DataFrame(X_test_scaled, columns=X_test.columns)

    # [Chinese text removed]parameter[Chinese text removed]Cross-validation[Chinese text removed]
    X = X_train_val
    y = y_train_val
    log.info("[Chinese text removed]parameter[Chinese text removed]Cross-validation")

    # [Chinese text removed]parameter[Chinese text removed]（100[Chinese text removed]）
    log.info("Starting[Chinese text removed]parameter[Chinese text removed]（100[Chinese text removed]）...")
    study = optuna.create_study(direction='minimize')
    study.optimize(lambda trial: objective(trial, X, y), n_trials=100)
    
    best_params = study.best_params
    log.info(f"Best[Chinese text removed]parameter: {best_params}")
    log.info(f"Best RMSE: {study.best_value:.4f}")

    # [Chinese text removed]parameter[Chinese text removed]Cross-validation
    kf = KFold(n_splits=10, shuffle=True, random_state=42)
    metrics_list = []

    for fold_num, (train_index, val_index) in enumerate(kf.split(X)):
        X_train_cv, X_val_cv = X.iloc[train_index], X.iloc[val_index]
        y_train_cv, y_val_cv = y.iloc[train_index], y.iloc[val_index]

        # [Chinese text removed]parameter[Chinese text removed]
        model_final = xgb.XGBRegressor(**best_params)
        model_final.fit(X_train_cv, y_train_cv)

        # [Chinese text removed]
        y_val_pred = model_final.predict(X_val_cv)

        # Calculate evaluation metrics
        mse_val = mean_squared_error(y_val_cv, y_val_pred)
        rmse_val = np.sqrt(mse_val)
        r2_val = r2_score(y_val_cv, y_val_pred)
        mae_val = mean_absolute_error(y_val_cv, y_val_pred)
        mape_val = mean_absolute_percentage_error(y_val_cv, y_val_pred)

        metrics_list.append({
            'fold': fold_num + 1,
            'mse': mse_val,
            'rmse': rmse_val,
            'r2': r2_val,
            'mae': mae_val,
            'mape': mape_val
        })

        log.info(f"Fold {fold_num + 1} - MSE: {mse_val:.4f}, RMSE: {rmse_val:.4f}, R2: {r2_val:.4f}, MAE: {mae_val:.4f}, MAPE: {mape_val:.2f}%")

    # [Chinese text removed]
    avg_metrics = {key: np.mean([m[key] for m in metrics_list]) for key in metrics_list[0]}
    log.info("10[Chinese text removed]Cross-validation[Chinese text removed]：")
    log.info(f"MSE: {avg_metrics['mse']:.4f}")
    log.info(f"RMSE: {avg_metrics['rmse']:.4f}")
    log.info(f"R2: {avg_metrics['r2']:.4f}")
    log.info(f"MAE: {avg_metrics['mae']:.4f}")
    log.info(f"MAPE: {avg_metrics['mape']:.2f}%")

    # ----------------------------------------------------------------
    # 4. [Chinese text removed]
    # Training final model
    model_final = xgb.XGBRegressor(**best_params)
    model_final.fit(X_train, y_train,
                    eval_set=[(X_val, y_val)],
                    early_stopping_rounds=50,
                    verbose=False)

    # Make predictions on all datasets
    y_pred_train = model_final.predict(X_train)
    y_pred_val = model_final.predict(X_val)
    y_pred_test = model_final.predict(X_test)

    # [Chinese text removed]
    # [Chinese text removed]
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

    log.info("[Chinese text removed]：")
    log.info(f"[Chinese text removed] -> MSE: {mse_test:.4f}, RMSE: {rmse_test:.4f}, R2: {r2_test:.4f}, Adjusted R2: {adj_r2_test:.4f}, MAE: {mae_test:.4f}, MAPE: {mape_test:.2f}%")

    # Plot scatter plot
    plot_scatter(y_train, y_pred_train, y_test, y_pred_test)

    # [Chinese text removed]
    mse_test = mean_squared_error(y_test, y_pred_test)
    rmse_test = np.sqrt(mse_test)
    r2_test = r2_score(y_test, y_pred_test)
    mae_test = mean_absolute_error(y_test, y_pred_test)
    mape_test = mean_absolute_percentage_error(y_test, y_pred_test)
    adj_r2_test = adjusted_r2(y_test, y_pred_test, X_test.shape[1])

    log.info("[Chinese text removed]：")
    log.info(f"[Chinese text removed] -> MSE: {mse_test:.4f}, RMSE: {rmse_test:.4f}, R2: {r2_test:.4f}, Adjusted R2: {adj_r2_test:.4f}, MAE: {mae_test:.4f}, MAPE: {mape_test:.2f}%")

    # [Chinese text removed] CSV [Chinese text removed]
    train_results = pd.DataFrame({'True Values': y_train, 'Predicted Values': y_pred_train})
    test_results = pd.DataFrame({'True Values': y_test, 'Predicted Values': y_pred_test})
    
    # [Chinese text removed] result [Chinese text removed]（[Chinese text removed]）
    os.makedirs('./result', exist_ok=True)
    
    train_results.to_csv('./result/xgb_fp_train_results.csv', index=False)
    test_results.to_csv('./result/xgb_fp_test_results.csv', index=False)
    log.info("[Chinese text removed] './result/xgb_fp_train_results.csv' [Chinese text removed] './result/xgb_fp_test_results.csv'")

    # [Chinese text removed]
    model_dir = "logbbModel"
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    # [Chinese text removed]
    model_save_path = os.path.join(model_dir, "xgb_fp_model.joblib")

    # Save model
    joblib.dump(model_final, model_save_path)
    log.info(f"Model saved to: {model_save_path}")

    # [Chinese text removed]
    loaded_model = joblib.load(model_save_path)
    # [Chinese text removed]
    y_pred_test_loaded = loaded_model.predict(X_test)

    # [Chinese text removed]original[Chinese text removed]
    assert np.allclose(y_pred_test, y_pred_test_loaded), "[Chinese text removed]original[Chinese text removed]！"
    log.info("[Chinese text removed]Success，[Chinese text removed]！")


if __name__ == '__main__':
    main()