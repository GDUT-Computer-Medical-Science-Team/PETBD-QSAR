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
import lightgbm as lgb
import optuna
from padelpy import padeldescriptor
import tempfile
import joblib
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression  # [Chinese text removed]
from tqdm import tqdm
import random

# [Chinese text removed]
log = DataLogger(log_file=f"../../data/logBB_data/log/{datetime.now().strftime('%Y%m%d')}.log").getlog(
    "lightgbm_feature_selection")

# [Chinese text removed]
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

def calculate_morgan_fingerprints(smiles_list, radius=2, n_bits=1024):
    """
    [Chinese text removed]Morgan fingerprints
    :param smiles_list: SMILES [Chinese text removed]
    :param radius: [Chinese text removed]，[Chinese text removed]2
    :param n_bits: [Chinese text removed]，[Chinese text removed]1024
    :return: numpy [Chinese text removed]，[Chinese text removed]
    """
    fingerprints = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            log.warning(f"[Chinese text removed] SMILES: {smi}")
            continue
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
        arr = np.zeros((1,), dtype=int)
        DataStructs.ConvertToNumpyArray(fp, arr)
        fingerprints.append(arr)
    return np.array(fingerprints)


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
                fingerprints=True,  # [Chinese text removed]
                descriptortypes=None,  # [Chinese text removed]XML[Chinese text removed]，[Chinese text removed]
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
        log.error(f"[Chinese text removed]PaDEL[Chinese text removed]: {str(e)}")
        raise
    
    finally:
        # [Chinese text removed]
        if os.path.exists(temp_smi_file):
            os.remove(temp_smi_file)


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


def mean_absolute_percentage_error(y_true, y_pred):
    """
    [Chinese text removed]
    """
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    non_zero_index = (y_true != 0)
    return np.mean(np.abs((y_true[non_zero_index] - y_pred[non_zero_index]) / y_true[non_zero_index])) * 100


def adjusted_r2(y_true, y_pred, num_features):
    """
    [Chinese text removed]adjusted R²
    :param y_true: true values
    :param y_pred: predicted values
    :param num_features: number of features used by the model
    :return: adjusted R²
    """
    n = len(y_true)
    r2 = r2_score(y_true, y_pred)
    # [Chinese text removed]samples[Chinese text removed]features[Chinese text removed]1[Chinese text removed]，returnoriginalR²
    if n <= num_features + 1:
        return r2
    return 1 - ((1 - r2) * (n - 1) / (n - num_features - 1))


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
def objective(trial, X, y):
    param = {
        'num_leaves': trial.suggest_int('num_leaves', 20, 3000),
        'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.1),
        'n_estimators': trial.suggest_int('n_estimators', 100, 3000),
        'max_depth': trial.suggest_int('max_depth', 3, 30),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
        'min_split_gain': trial.suggest_float('min_split_gain', 0.0, 1.0),
        'max_bin': trial.suggest_int('max_bin', 200, 300),
        'boosting_type': trial.suggest_categorical('boosting_type', ['gbdt', 'dart']),
        'random_state': RANDOM_SEED,
        'objective': 'regression'
    }

    # [Chinese text removed]/[Chinese text removed]
    X_train_split, X_valid_split, y_train_split, y_valid_split = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_SEED
    )

    # [Chinese text removed]
    model = lgb.LGBMRegressor(**param)
    model.fit(X_train_split, y_train_split, 
              eval_set=[(X_valid_split, y_valid_split)], 
              callbacks=[lgb.early_stopping(stopping_rounds=50)])

    # [Chinese text removed]Calculate evaluation metrics
    y_pred = model.predict(X_valid_split)
    rmse = np.sqrt(mean_squared_error(y_valid_split, y_pred))

    return rmse  # return RMSE [Chinese text removed]


def feature_selection(X, y, n_features=50):
    """
    [Chinese text removed] RFE [Chinese text removed]features[Chinese text removed]
    :param X: features[Chinese text removed]
    :param y: [Chinese text removed]
    :param n_features: [Chinese text removed]features[Chinese text removed]
    :return: [Chinese text removed]features[Chinese text removed]
    """
    model = lgb.LGBMRegressor()  # [Chinese text removed]，[Chinese text removed] LogisticRegression
    rfe = RFE(estimator=model, n_features_to_select=n_features)
    
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=RANDOM_SEED)
    rfe.fit(X_train, y_train)
    selected_indices = np.where(rfe.support_)[0]

    log.info(f"[Chinese text removed]features[Chinese text removed]: {selected_indices.tolist()}")
    log.info(f"[Chinese text removed]features[Chinese text removed]: {len(selected_indices)}")
    
    return selected_indices.tolist()


def evaluate_model(X, y, selected_indices):
    """
    [Chinese text removed]
    :param X: features[Chinese text removed]
    :param y: [Chinese text removed]
    :param selected_indices: [Chinese text removed]features[Chinese text removed]
    :return: [Chinese text removed]
    """
    X_selected = X.iloc[:, selected_indices]  # [Chinese text removed]features
    X_train, X_val, y_train, y_val = train_test_split(X_selected, y, test_size=0.2, random_state=RANDOM_SEED)

    model = lgb.LGBMRegressor()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_val)

    mse = mean_squared_error(y_val, y_pred)
    r2 = r2_score(y_val, y_pred)
    mae = mean_absolute_error(y_val, y_pred)

    return mse, r2, mae


def train_lightgbm_with_selected_features(X, y, selected_indices):
    """
    [Chinese text removed]features[Chinese text removed] LightGBM [Chinese text removed]
    :param X: features[Chinese text removed]
    :param y: [Chinese text removed]
    :param selected_indices: [Chinese text removed]features[Chinese text removed]
    :return: [Chinese text removed]
    """
    # [Chinese text removed]features
    X_selected = X.iloc[:, selected_indices]

    # [Chinese text removed]
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X_selected, y, test_size=0.1, random_state=RANDOM_SEED
    )
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=0.1, random_state=RANDOM_SEED
    )

    # [Chinese text removed]parameter[Chinese text removed]
    study = optuna.create_study(direction='minimize')
    study.optimize(lambda trial: objective(trial, X_train, y_train), n_trials=100)
    
    best_params = study.best_params
    
    # Training final model
    model = lgb.LGBMRegressor(**best_params)
    model.fit(X_train, y_train,
              eval_set=[(X_val, y_val)],
              callbacks=[lgb.early_stopping(stopping_rounds=50)])

    # [Chinese text removed]Calculate evaluation metrics
    y_pred_train = model.predict(X_train)
    y_pred_val = model.predict(X_val)
    y_pred_test = model.predict(X_test)

    # Calculate evaluation metrics
    metrics = calculate_metrics(y_train, y_pred_train, y_val, y_pred_val, 
                                y_test, y_pred_test, X_train.shape[1])

    return model, metrics, (X_train, X_val, X_test)


# --------------------------------------------------------------------
def main():
    # [Chinese text removed]
    logBB_data_file = "../../data/logBB_data/logBB.csv"
    logBB_desc_file = "../../data/logBB_data/logBB_w_desc_fp.csv"
    feature_index_path = "../../data/logBB_data/logBB_w_desc_fp_index2.txt"  # features[Chinese text removed]
    log.info("===============Starting LightGBM features[Chinese text removed]===============")

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

    # [Chinese text removed]
    y = df_final['logBB']  # [Chinese text removed] y [Chinese text removed]

    # Feature normalization
    sc = MinMaxScaler()
    X = pd.DataFrame(sc.fit_transform(X), columns=X.columns)

    # features[Chinese text removed]
    if not os.path.exists(feature_index_path):
        log.info("[Chinese text removed]features[Chinese text removed]，[Chinese text removed]features[Chinese text removed]")
        
        # [Chinese text removed]RFE[Chinese text removed]features[Chinese text removed]
        model_rfe = lgb.LGBMRegressor()
        rfe = RFE(estimator=model_rfe, n_features_to_select=50)  # [Chinese text removed]50[Chinese text removed]features
        rfe.fit(X, y)
        
        # [Chinese text removed]features[Chinese text removed]
        selected_features = X.columns[rfe.support_]
        log.info(f"[Chinese text removed]features: {selected_features.tolist()}")
        
        # [Chinese text removed]features[Chinese text removed]
        desc_index = np.where(rfe.support_)[0]
        np.savetxt(feature_index_path, desc_index, fmt='%d')
        log.info(f"Feature indices saved[Chinese text removed]：{feature_index_path}")
        
        X = X[selected_features]  # [Chinese text removed]features
        log.info(f"[Chinese text removed]features[Chinese text removed]：{X.shape}")
        log.info(f"[Chinese text removed]features[Chinese text removed]: {X.columns.tolist()}")
    else:
        log.info("[Chinese text removed]features[Chinese text removed]，[Chinese text removed]")
        desc_index = np.loadtxt(feature_index_path, dtype=int, delimiter=',').tolist()
        log.info(f"[Chinese text removed]features[Chinese text removed]Complete，[Chinese text removed]: {desc_index}")

        # Check if desc_index is valid
        if any(i >= X.shape[1] for i in desc_index):
            log.error("features[Chinese text removed]，[Chinese text removed]。")
            raise IndexError("features[Chinese text removed]。")

        X = X.iloc[:, desc_index]  # Use iloc to select columns by index
        log.info(f"[Chinese text removed]features[Chinese text removed]：{X.shape}")
        log.info(f"[Chinese text removed]features[Chinese text removed]: {X.columns.tolist()}")

    # Dataset split
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=0.1, random_state=RANDOM_SEED
    )
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=0.1, random_state=RANDOM_SEED
    )
    log.info(f"[Chinese text removed]9:1[Chinese text removed]({len(X_train_val)}[Chinese text removed]samples)[Chinese text removed]({len(X_test)}[Chinese text removed]samples)")

    # [Chinese text removed]parameter[Chinese text removed]（100[Chinese text removed]）
    log.info("Starting hyperparameter search (100 trials)...")
    study = optuna.create_study(direction='minimize')
    study.optimize(lambda trial: objective(trial, X, y), n_trials=100)
    
    best_params = study.best_params
    log.info(f"Best hyperparameters: {best_params}")
    log.info(f"Best RMSE: {study.best_value:.4f}")

    # [Chinese text removed]parameter[Chinese text removed]Cross-validation
    kf = KFold(n_splits=10, shuffle=True, random_state=RANDOM_SEED)
    metrics_list = []

    for fold_num, (train_index, val_index) in enumerate(kf.split(X_train_val)):
        X_train_cv, X_val_cv = X_train_val.iloc[train_index], X_train_val.iloc[val_index]
        y_train_cv, y_val_cv = y_train_val.iloc[train_index], y_train_val.iloc[val_index]

        # [Chinese text removed]parameter[Chinese text removed]
        model_final = lgb.LGBMRegressor(**best_params)
        model_final.fit(X_train_cv, y_train_cv)

        # [Chinese text removed]
        y_val_pred = model_final.predict(X_val_cv)

        # Calculate evaluation metrics
        mse_val = mean_squared_error(y_val_cv, y_val_pred)
        rmse_val = np.sqrt(mse_val)
        r2_val = r2_score(y_val_cv, y_val_pred)
        mae_val = mean_absolute_error(y_val_cv, y_val_pred)
        mape_val = mean_absolute_percentage_error(y_val_cv, y_val_pred)

        # [Chinese text removed] R²
        adj_r2 = adjusted_r2(y_val_cv, y_val_pred, X_val_cv.shape[1])

        # Calculate evaluation metrics
        metrics = {
            'fold': fold_num + 1,
            'mse': mse_val,
            'rmse': rmse_val,
            'r2': r2_val,
            'mae': mae_val,
            'mape': mape_val,
            'adj_r2': adj_r2
        }
        metrics_list.append(metrics)

        log.info(f"Fold {fold_num + 1}:")
        log.info(f"MSE: {mse_val:.4f}, RMSE: {rmse_val:.4f}, R²: {r2_val:.4f}, Adj R²: {adj_r2:.4f}, MAE: {mae_val:.4f}, MAPE: {mape_val:.2f}%")

    # [Chinese text removed]
    avg_metrics = {key: np.mean([m[key] for m in metrics_list]) for key in metrics_list[0] if key != 'fold'}
    log.info("\n10[Chinese text removed]Cross-validation[Chinese text removed]：")
    log.info(f"MSE: {avg_metrics['mse']:.4f}")
    log.info(f"RMSE: {avg_metrics['rmse']:.4f}")
    log.info(f"R²: {avg_metrics['r2']:.4f}")
    log.info(f"Adjusted R²: {avg_metrics['adj_r2']:.4f}")
    log.info(f"MAE: {avg_metrics['mae']:.4f}")
    log.info(f"MAPE: {avg_metrics['mape']:.2f}%")

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

    log.info("\n[Chinese text removed]：")
    log.info(f"MSE: {test_metrics['test_mse']:.4f}")
    log.info(f"RMSE: {test_metrics['test_rmse']:.4f}")
    log.info(f"R²: {test_metrics['test_r2']:.4f}")
    log.info(f"Adjusted R²: {test_metrics['test_adj_r2']:.4f}")
    log.info(f"MAE: {test_metrics['test_mae']:.4f}")
    log.info(f"MAPE: {test_metrics['test_mape']:.2f}%")

    # After predicting and before creating the DataFrame
    y_train_val_pred = model_final.predict(X_train_val)
    y_test_pred = model_final.predict(X_test)

    # Check lengths
    print(f"Length of SMILES: {len(SMILES[X_train_val.index].values)}")
    print(f"Length of Experimental logBB: {len(y_train_val.values)}")
    print(f"Length of Predicted logBB: {len(y_train_val_pred)}")

    # Create DataFrame for training results
    train_results = pd.DataFrame({
        'SMILES': SMILES[X_train_val.index].values,  # [Chinese text removed] .values
        'Experimental logBB': y_train_val.values,  # [Chinese text removed] .values
        'Predicted logBB': y_train_val_pred  # [Chinese text removed] .values
    })

    # Create DataFrame for test results
    test_results = pd.DataFrame({
        'SMILES': SMILES[X_test.index].values,  # [Chinese text removed] X_test.index [Chinese text removed]
        'Experimental logBB': y_test.values,  # [Chinese text removed] .values
        'Predicted logBB': y_pred_test  # [Chinese text removed] .values
    })

    # Save results to CSV
    train_results.to_csv('./result/lightgbm_fp_train_results.csv', index=False)
    test_results.to_csv('./result/lightgbm_fp_test_results.csv', index=False)
    log.info("[Chinese text removed] './result/lightgbm_fp_train_results.csv' [Chinese text removed] './result/lightgbm_fp_test_results.csv'")

    # [Chinese text removed]
    model_dir = "logbbModel"
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    model_path = os.path.join(model_dir, "lightgbm_fp_model.joblib")
    joblib.dump(model_final, model_path)
    log.info(f"Model saved to: {model_path}")

    # [Chinese text removed]
    loaded_model = joblib.load(model_path)
    y_pred_test_loaded = loaded_model.predict(X_test)

    # [Chinese text removed]
    assert np.allclose(y_pred_test, y_pred_test_loaded), "[Chinese text removed]original[Chinese text removed]！"
    log.info("[Chinese text removed]Success：[Chinese text removed]")

if __name__ == '__main__':
    main()
