import joblib
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
from catboost import CatBoostRegressor
import optuna
import tempfile
from padelpy import padeldescriptor
from tqdm import tqdm
from preprocess.data_preprocess.FeatureExtraction import FeatureExtraction
from sklearn.feature_selection import RFE  # Import RFE
from sklearn.ensemble import RandomForestRegressor

# [Chinese text removed]
log = DataLogger(log_file=f"../../data/logBB_data/log/{datetime.now().strftime('%Y%m%d')}.log").getlog(
    "catboost_feature_selection")


def calculate_padel_fingerprints(smiles_list):
    """
    [Chinese text removed]PaDEL[Chinese text removed]
    :param smiles_list: SMILES [Chinese text removed]
    :return: DataFrame，[Chinese text removed]
    """
    # [Chinese text removed]SMILES
    with tempfile.NamedTemporaryFile(mode='w', suffix='.smi', delete=False) as temp_file:
        for smi in smiles_list:
            temp_file.write(f"{smi}\n")
        temp_smi_file = temp_file.name

    try:
        # [Chinese text removed]
        with tempfile.TemporaryDirectory() as temp_dir:
            output_file = os.path.join(temp_dir, "fingerprints.csv")
            
            # [Chinese text removed]
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
            # [Chinese text removed]
            df = pd.read_csv(output_file)
            # [Chinese text removed]Name[Chinese text removed]（[Chinese text removed]）
            if 'Name' in df.columns:
                df = df.drop('Name', axis=1)
            # [Chinese text removed]SMILES[Chinese text removed]
            df.insert(0, 'SMILES', smiles_list)
            return df
    except Exception as e:
        log.error(f"[Chinese text removed]PaDEL[Chinese text removed]Error: {str(e)}")
        raise
    finally:
        # [Chinese text removed]
        if os.path.exists(temp_smi_file):
            os.remove(temp_smi_file)


def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    non_zero_index = (y_true != 0)
    return np.mean(np.abs((y_true[non_zero_index] - y_pred[non_zero_index]) / y_true[non_zero_index])) * 100


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


# --------------------------------------------------------------------
# 1. [Chinese text removed]parameter[Chinese text removed]（[Chinese text removed]Cross-validation，[Chinese text removed]/[Chinese text removed]，[Chinese text removed]）
def objective(trial, X_train, y_train):
    param = {
        'iterations': trial.suggest_int('iterations', 100, 1000),
        'depth': trial.suggest_int('depth', 4, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1, 10),
        'random_seed': 42,
        'verbose': False
    }
    
    # Split data for validation
    X_t, X_v, y_t, y_v = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
    
    # Train model
    model = CatBoostRegressor(**param)
    model.fit(X_t, y_t, eval_set=(X_v, y_v), early_stopping_rounds=50, verbose=False)
    
    # Predict and calculate RMSE
    y_pred = model.predict(X_v)
    rmse = np.sqrt(mean_squared_error(y_v, y_pred))
    
    return rmse  # Return RMSE as the objective to minimize


# --------------------------------------------------------------------
def main():
    # [Chinese text removed]
    logBB_data_file = "../../data/logBB_data/logBB.csv"
    logBB_desc_file = "../../data/logBB_data/logBB_w_desc_fp.csv"
    logBB_desc_index_file = "../../data/logBB_data/logBB_w_desc_fp_index2.txt"
    log.info("===============Starting CatBoost features[Chinese text removed]===============")

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
    df_final.to_csv(logBB_desc_file, encoding='utf-8', index=False)
    log.info(f"[Chinese text removed]: {logBB_desc_file}")
    log.info(f"[Chinese text removed]: {df_final.shape}")
    log.info(f"[Chinese text removed]：\n{df_final.head()}")

    # features[Chinese text removed]
    X = df_final.drop(['SMILES', 'logBB'], axis=1)

    # [Chinese text removed]features[Chinese text removed] blood mean60min
    if 'blood mean60min' in X.columns:
        X = X.drop('blood mean60min', axis=1)
        log.info("[Chinese text removed]features: blood mean60min")

    # [Chinese text removed]features[Chinese text removed]
    X = X.apply(pd.to_numeric, errors='coerce')  # [Chinese text removed]Converting[Chinese text removed] NaN
    X = X.fillna(0)  # [Chinese text removed]0[Chinese text removed]NaN[Chinese text removed]

    y = df_final['logBB']
    log.info(f"[Chinese text removed] {len(X)} [Chinese text removed]features[Chinese text removed]")

    # [Chinese text removed]features[Chinese text removed]
    if os.path.exists(logBB_desc_index_file):
        log.info("features[Chinese text removed]，[Chinese text removed]features")
        desc_index = np.loadtxt(logBB_desc_index_file, dtype=int, delimiter=',')
        
        # [Chinese text removed]
        valid_indices = desc_index[desc_index < X.shape[1]]
        if len(valid_indices) != len(desc_index):
            log.warning(f"[Chinese text removed]featuresIndex out of range，[Chinese text removed]")
            desc_index = valid_indices
        
        X = X.iloc[:, desc_index]  # [Chinese text removed]
        log.info(f"[Chinese text removed]features[Chinese text removed]，[Chinese text removed]features[Chinese text removed]: {X.shape[1]}")
        log.info(f"[Chinese text removed]features[Chinese text removed]: {X.columns.tolist()}")
    else:
        # [Chinese text removed]features[Chinese text removed]
        model_rfe = RandomForestRegressor(n_estimators=100, random_state=42)
        rfe = RFE(estimator=model_rfe, n_features_to_select=50)
        rfe.fit(X, y)

        # [Chinese text removed]features[Chinese text removed]
        selected_features = X.columns[rfe.support_]
        log.info(f"[Chinese text removed]features: {selected_features.tolist()}")

        # [Chinese text removed]features[Chinese text removed]
        desc_index = np.where(rfe.support_)[0]
        np.savetxt(logBB_desc_index_file, desc_index, fmt='%d')
        log.info(f"Feature indices saved[Chinese text removed]: {logBB_desc_index_file}")

        X = X[selected_features]

    # Dataset split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
    log.info(f"[Chinese text removed]9:1[Chinese text removed]({len(X_train)}[Chinese text removed]samples)[Chinese text removed]({len(X_test)}[Chinese text removed]samples)")

    # [Chinese text removed]Cross-validation
    kf = KFold(n_splits=10, shuffle=True, random_state=42)
    metrics_list = []

    for fold_num, (train_index, val_index) in enumerate(kf.split(X_train)):
        X_fold_train, X_fold_val = X_train.iloc[train_index], X_train.iloc[val_index]
        y_fold_train, y_fold_val = y_train.iloc[train_index], y_train.iloc[val_index]

        # [Chinese text removed]parameter[Chinese text removed]
        study = optuna.create_study(direction='minimize')
        study.optimize(lambda trial: objective(trial, X_fold_train, y_fold_train), n_trials=10)
        best_params = study.best_params

        # Training final model
        model_fold = CatBoostRegressor(**best_params, verbose=0)
        model_fold.fit(X_fold_train, y_fold_train)

        # [Chinese text removed]
        y_fold_val_pred = model_fold.predict(X_fold_val)

        # Calculate evaluation metrics
        mse_val = mean_squared_error(y_fold_val, y_fold_val_pred)
        rmse_val = np.sqrt(mse_val)
        r2_val = r2_score(y_fold_val, y_fold_val_pred)
        mae_val = mean_absolute_error(y_fold_val, y_fold_val_pred)
        mape_val = mean_absolute_percentage_error(y_fold_val, y_fold_val_pred)
        adj_r2_val = adjusted_r2(y_fold_val, y_fold_val_pred, X_fold_val.shape[1])

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

    # [Chinese text removed]parameter[Chinese text removed]Training final model
    study = optuna.create_study(direction='minimize')
    study.optimize(lambda trial: objective(trial, X_train, y_train), n_trials=10)
    best_params = study.best_params

    model_final = CatBoostRegressor(**best_params, verbose=0)
    model_final.fit(X_train, y_train)

    # [Chinese text removed]
    y_pred_test = model_final.predict(X_test)
    test_metrics = {
        'test_mse': mean_squared_error(y_test, y_pred_test),
        'test_rmse': np.sqrt(mean_squared_error(y_test, y_pred_test)),
        'test_r2': r2_score(y_test, y_pred_test),
        'test_mae': mean_absolute_error(y_test, y_pred_test),
        'test_mape': mean_absolute_percentage_error(y_test, y_pred_test),
        'test_adj_r2': adjusted_r2(y_test, y_pred_test, X_test.shape[1])
    }

    # [Chinese text removed]
    log.info("\n[Chinese text removed]：")
    log.info(f"MSE: {test_metrics['test_mse']:.4f}")
    log.info(f"RMSE: {test_metrics['test_rmse']:.4f}")
    log.info(f"R²: {test_metrics['test_r2']:.4f}")
    log.info(f"Adjusted R²: {test_metrics['test_adj_r2']:.4f}")
    log.info(f"MAE: {test_metrics['test_mae']:.4f}")
    log.info(f"MAPE: {test_metrics['test_mape']:.2f}%")

    # [Chinese text removed]Test Set Results
    test_results = pd.DataFrame({
        'True Values': y_test,
        'Predicted Values': y_pred_test
    })

    os.makedirs('./result', exist_ok=True)
    test_results.to_csv('./result/catboost_fp_test_results.csv', index=False)

    # [Chinese text removed]
    model_dir = "logbbModel"
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    # [Chinese text removed]
    model_path = os.path.join(model_dir, "catboost_fp_model.joblib")

    # Save model
    joblib.dump(model_final, model_path)
    log.info(f"Model saved to: {model_path}")

    # [Chinese text removed]
    loaded_model = joblib.load(model_path)
    # [Chinese text removed] - [Chinese text removed]
    y_pred_test_loaded = loaded_model.predict(X_test)

    # [Chinese text removed]original[Chinese text removed]
    original_pred = model_final.predict(X_test)
    assert np.allclose(original_pred, y_pred_test_loaded), "[Chinese text removed]original[Chinese text removed]！"
    log.info("[Chinese text removed]Success，[Chinese text removed]！")

    log.info("\n[Chinese text removed]Dataset split:")
    log.info(f"Training set: {len(X_train)}[Chinese text removed]samples ({len(X_train)/len(X)*100:.1f}%)")
    log.info(f"Test set: {len(X_test)}[Chinese text removed]samples ({len(X_test)/len(X)*100:.1f}%)")

    # Plot scatter plot
    plot_scatter(y_train, model_final.predict(X_train), y_test, y_pred_test, save_path='./result/catboost_fp_scatter.png')
    log.info("[Chinese text removed] './result/catboost_fp_scatter.png'")

    # [Chinese text removed] CSV [Chinese text removed]
    train_results = pd.DataFrame({
        'True Values': y_train,
        'Predicted Values': model_final.predict(X_train)
    })
    train_results.to_csv('./result/catboost_fp_train_results.csv', index=False)
    test_results.to_csv('./result/catboost_fp_test_results.csv', index=False)
    log.info("[Chinese text removed] './result/catboost_fp_train_results.csv' [Chinese text removed] './result/catboost_fp_test_results.csv'")


if __name__ == '__main__':
    main()

