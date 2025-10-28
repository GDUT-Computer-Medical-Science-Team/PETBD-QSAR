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
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, mean_absolute_percentage_error
from sklearn.ensemble import RandomForestRegressor
import optuna
import tempfile
from padelpy import padeldescriptor
import joblib
from sklearn.feature_selection import RFE

# [Chinese text removed]
log = DataLogger(log_file=f"../../data/logBB_data/log/{datetime.now().strftime('%Y%m%d')}.log").getlog("rf_feature_selection")


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


def plot_scatter(y_true_train, y_pred_train, y_true_test, y_pred_test):
    """
    [Chinese text removed]
    """
    plt.figure(figsize=(12, 6))

    # [Chinese text removed]
    plt.subplot(1, 2, 1)
    plt.scatter(y_true_train, y_pred_train, color='blue', alpha=0.5, label='Train')
    plt.plot([y_true_train.min(), y_true_train.max()],
             [y_true_train.min(), y_true_train.max()],
             color='red', lw=2)
    plt.title('Train Set: True vs Predicted')
    plt.xlabel('True Values')
    plt.ylabel('Predicted Values')
    plt.legend()

    # [Chinese text removed]
    plt.subplot(1, 2, 2)
    plt.scatter(y_true_test, y_pred_test, color='green', alpha=0.5, label='Test')
    plt.plot([y_true_test.min(), y_true_test.max()],
             [y_true_test.min(), y_true_test.max()],
             color='red', lw=2)
    plt.title('Test Set: True vs Predicted')
    plt.xlabel('True Values')
    plt.ylabel('Predicted Values')
    plt.legend()

    plt.tight_layout()
    plt.show()


# --------------------------------------------------------------------
# 1. [Chinese text removed]parameter[Chinese text removed]（[Chinese text removed]Cross-validation，[Chinese text removed]/[Chinese text removed]，[Chinese text removed]）
def objective(trial, X, y):
    param = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
        'max_depth': trial.suggest_int('max_depth', 3, 30),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
        'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2']),
        'bootstrap': trial.suggest_categorical('bootstrap', [True, False]),
        'min_weight_fraction_leaf': trial.suggest_float('min_weight_fraction_leaf', 0.0, 0.5),
        'max_leaf_nodes': trial.suggest_int('max_leaf_nodes', 10, 100),
        'random_state': 42
    }

    # [Chinese text removed]/[Chinese text removed]
    X_train_split, X_valid_split, y_train_split, y_valid_split = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # [Chinese text removed]
    model = RandomForestRegressor(**param)
    model.fit(X_train_split, y_train_split)

    # [Chinese text removed]Calculate evaluation metrics
    y_pred = model.predict(X_valid_split)
    rmse = np.sqrt(mean_squared_error(y_valid_split, y_pred))

    return rmse  # return RMSE [Chinese text removed]


# --------------------------------------------------------------------
def main():
    logBB_data_file = "../../data/logBB_data/logBB.csv"
    logBB_desc_file = "../../data/logBB_data/logBB_w_desc_fp.csv"
    feature_index_path = "../../data/logBB_data/logBB_w_desc_fp_index2.txt"  # features[Chinese text removed]
    log.info("===============Starting RandomForest features[Chinese text removed]===============")

    # [Chinese text removed]
    if not os.path.exists(logBB_data_file):
        log.error("[Chinese text removed]logBB[Chinese text removed]")
        raise FileNotFoundError("[Chinese text removed]logBB[Chinese text removed]")

    # [Chinese text removed]
    df = pd.read_csv(logBB_data_file, encoding='utf-8')
    df = df.dropna(subset=['logBB']).reset_index(drop=True)

    # [Chinese text removed] Compound index [Chinese text removed]
    if 'Compound index' in df.columns:
        df = df.drop(columns=['Compound index'])
        log.info("[Chinese text removed] Compound index [Chinese text removed]")

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

    # [Chinese text removed]features
    log.info(f"[Chinese text removed]features[Chinese text removed]: {X_initial.shape[1]}")
    log.info(f"features[Chinese text removed]: {X_initial.columns.tolist()}")

    # [Chinese text removed]features[Chinese text removed]
    if os.path.exists(feature_index_path):
        log.info("features[Chinese text removed]，[Chinese text removed]features")
        desc_index = np.loadtxt(feature_index_path, dtype=int, delimiter=',')
        
        # [Chinese text removed]
        valid_indices = desc_index[desc_index < X_initial.shape[1]]
        if len(valid_indices) != len(desc_index):
            log.warning(f"[Chinese text removed]featuresIndex out of range，[Chinese text removed]")
            desc_index = valid_indices
        
        X = X_initial.iloc[:, desc_index]  # [Chinese text removed]
        log.info(f"[Chinese text removed]features[Chinese text removed]，[Chinese text removed]features[Chinese text removed]: {X.shape[1]}")
        log.info(f"[Chinese text removed]features[Chinese text removed]: {X.columns.tolist()}")
    else:
        log.error("features[Chinese text removed]，[Chinese text removed]features")
        return

    # Dataset split
    X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
    log.info(f"[Chinese text removed]9:1[Chinese text removed]({len(X_train_val)})[Chinese text removed]samples[Chinese text removed]({len(X_test)})[Chinese text removed]samples")

    # [Chinese text removed] (10%)
    X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.1, random_state=42)
    log.info(f"[Chinese text removed]9:1[Chinese text removed]({len(X_train)})[Chinese text removed]samples[Chinese text removed]({len(X_val)})[Chinese text removed]samples")

    # [Chinese text removed]
    sc = MinMaxScaler()
    X_scaled = sc.fit_transform(X)
    X = pd.DataFrame(X_scaled, columns=X.columns)

    # [Chinese text removed]parameter[Chinese text removed]（100[Chinese text removed]）
    log.info("Starting hyperparameter search (100 trials)...")
    study = optuna.create_study(direction='minimize')
    study.optimize(lambda trial: objective(trial, X, y), n_trials=100)
    
    best_params = study.best_params
    log.info(f"Best hyperparameters: {best_params}")
    log.info(f"Best RMSE: {study.best_value:.4f}")

    # [Chinese text removed]parameter[Chinese text removed]Cross-validation
    kf = KFold(n_splits=10, shuffle=True, random_state=42)
    metrics_list = []

    for fold_num, (train_index, val_index) in enumerate(kf.split(X)):
        X_train_cv, X_val_cv = X.iloc[train_index], X.iloc[val_index]
        y_train_cv, y_val_cv = y.iloc[train_index], y.iloc[val_index]

        # [Chinese text removed]parameter[Chinese text removed]
        model_final = RandomForestRegressor(**best_params)
        model_final.fit(X_train_cv, y_train_cv)

        # [Chinese text removed]
        y_val_pred = model_final.predict(X_val_cv)

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

        log.info(f"Fold {fold_num + 1} - MSE: {mse_val:.4f}, RMSE: {rmse_val:.4f}, R2: {r2_val:.4f}, MAE: {mae_val:.4f}, MAPE: {mape_val:.2f}%, Adjusted R2: {adj_r2_val:.4f}")

    # [Chinese text removed]
    avg_metrics = {key: np.mean([m[key] for m in metrics_list]) for key in metrics_list[0] if key != 'fold'}
    log.info("10[Chinese text removed]Cross-validation[Chinese text removed]：")
    log.info(f"MSE: {avg_metrics['mse']:.4f}")
    log.info(f"RMSE: {avg_metrics['rmse']:.4f}")
    log.info(f"R2: {avg_metrics['r2']:.4f}")
    log.info(f"MAE: {avg_metrics['mae']:.4f}")
    log.info(f"MAPE: {avg_metrics['mape']:.2f}%")
    log.info(f"Adjusted R2: {avg_metrics['adj_r2']:.4f}")

    # ----------------------------------------------------------------
    # 4. Besides 10-fold CV, we also perform a separate train/test split for plotting scatter and saving predictions
    model_final = RandomForestRegressor(**best_params)
    model_final.fit(X_train, y_train)

    # Make predictions on all datasets
    y_pred_train = model_final.predict(X_train)
    y_pred_val = model_final.predict(X_val)
    y_pred_test = model_final.predict(X_test)

    # Calculate evaluation metrics
    mse_train = mean_squared_error(y_train, y_pred_train)
    rmse_train = np.sqrt(mse_train)
    r2_train = r2_score(y_train, y_pred_train)
    mae_train = mean_absolute_error(y_train, y_pred_train)
    mape_train = mean_absolute_percentage_error(y_train, y_pred_train)
    adj_r2_train = adjusted_r2(y_train, y_pred_train, X_train.shape[1])

    mse_val = mean_squared_error(y_val, y_pred_val)
    rmse_val = np.sqrt(mse_val)
    r2_val = r2_score(y_val, y_pred_val)
    mae_val = mean_absolute_error(y_val, y_pred_val)
    mape_val = mean_absolute_percentage_error(y_val, y_pred_val)
    adj_r2_val = adjusted_r2(y_val, y_pred_val, X_val.shape[1])

    mse_test = mean_squared_error(y_test, y_pred_test)
    rmse_test = np.sqrt(mse_test)
    r2_test = r2_score(y_test, y_pred_test)
    mae_test = mean_absolute_error(y_test, y_pred_test)
    mape_test = mean_absolute_percentage_error(y_test, y_pred_test)
    adj_r2_test = adjusted_r2(y_test, y_pred_test, X_test.shape[1])

    # [Chinese text removed]
    log.info("[Chinese text removed]：")
    log.info(f"[Chinese text removed] -> MSE: {mse_test:.4f}, RMSE: {rmse_test:.4f}, R2: {r2_test:.4f}, Adjusted R2: {adj_r2_test:.4f}, MAE: {mae_test:.4f}, MAPE: {mape_test:.2f}%")

    # [Chinese text removed]
    train_results = pd.DataFrame({
        'True Values': y_train, 
        'Predicted Values': y_pred_train
    })
    val_results = pd.DataFrame({
        'True Values': y_val, 
        'Predicted Values': y_pred_val
    })
    test_results = pd.DataFrame({
        'True Values': y_test, 
        'Predicted Values': y_pred_test
    })
    
    os.makedirs('./result', exist_ok=True)
    train_results.to_csv('./result/rf_fp_train_results.csv', index=False)
    val_results.to_csv('./result/rf_fp_val_results.csv', index=False)
    test_results.to_csv('./result/rf_fp_test_results.csv', index=False)

    # [Chinese text removed]
    model_dir = "logbbModel"
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    # [Chinese text removed]
    model_save_path = os.path.join(model_dir, "rf_fp_model.joblib")

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

    # Plot scatter plot
    plot_scatter(y_train, y_pred_train, y_test, y_pred_test)


if __name__ == '__main__':
    main()


