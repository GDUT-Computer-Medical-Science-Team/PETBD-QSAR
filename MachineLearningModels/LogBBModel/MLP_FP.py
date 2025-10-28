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
from sklearn.neural_network import MLPRegressor
import optuna
import tempfile
from padelpy import padeldescriptor
import joblib
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestRegressor  # [Chinese text removed]features[Chinese text removed]

# [Chinese text removed]
log = DataLogger(log_file=f"../../data/logBB_data/log/{datetime.now().strftime('%Y%m%d')}.log").getlog("mlp_feature_selection")


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
    # [Chinese text removed]，[Chinese text removed]
    param = {
        'hidden_layer_sizes': trial.suggest_categorical('hidden_layer_sizes', [(100,), (50, 50), (150,)]),
        'activation': trial.suggest_categorical('activation', ['relu', 'tanh']),
        'solver': trial.suggest_categorical('solver', ['adam', 'sgd']),
        'learning_rate': trial.suggest_categorical('learning_rate', ['constant', 'adaptive']),
        'random_state': 42
    }
    # [Chinese text removed] 80/20 [Chinese text removed]
    X_train_split, X_valid_split, y_train_split, y_valid_split = train_test_split(X, y, test_size=0.2, random_state=42)

    model = MLPRegressor(**param, max_iter=500)  # [Chinese text removed]
    model.fit(X_train_split, y_train_split)
    y_valid_pred = model.predict(X_valid_split)
    rmse = np.sqrt(mean_squared_error(y_valid_split, y_valid_pred))
    return rmse


# --------------------------------------------------------------------
def main():
    logBB_data_file = "../../data/logBB_data/logBB.csv"
    logBB_desc_file = "../../data/logBB_data/logBB_w_desc_fp.csv"
    logBB_desc_index_file = "../../data/logBB_data/logBB_w_desc_fp_index2.txt"
    log.info("===============Starting MLP features[Chinese text removed]===============")

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
    df_final = df_final.drop(columns=['Compound index'], errors='ignore')  # [Chinese text removed] Compound index [Chinese text removed]
    df_final.to_csv(logBB_desc_file, encoding='utf-8', index=False)
    log.info(f"[Chinese text removed]: {logBB_desc_file}")
    log.info(f"[Chinese text removed]: {df_final.shape}")

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
    X_initial = X.copy()  # [Chinese text removed]original[Chinese text removed]

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
        rfe.fit(X_initial, y)

        # [Chinese text removed]features[Chinese text removed]
        selected_features = X_initial.columns[rfe.support_]
        log.info(f"[Chinese text removed]features: {selected_features.tolist()}")

        # [Chinese text removed]features
        if 'blood mean60min' in selected_features:
            selected_features = selected_features[selected_features != 'blood mean60min']
            log.info("[Chinese text removed]features: blood mean60min")

        # [Chinese text removed]features[Chinese text removed]
        desc_index = np.where(rfe.support_)[0]
        np.savetxt(logBB_desc_index_file, desc_index, fmt='%d')
        log.info(f"Feature indices saved[Chinese text removed]: {logBB_desc_index_file}")

        X = X_initial[selected_features]

    # Dataset split
    X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
    log.info(f"[Chinese text removed]9:1[Chinese text removed]({len(X_train_val)}[Chinese text removed]samples)[Chinese text removed]({len(X_test)}[Chinese text removed]samples)")

    # [Chinese text removed] (10%)
    X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.1, random_state=42)
    log.info(f"[Chinese text removed]9:1[Chinese text removed]({len(X_train)}[Chinese text removed]samples)[Chinese text removed]({len(X_val)}[Chinese text removed]samples)")

    # [Chinese text removed]Cross-validation
    kf = KFold(n_splits=10, shuffle=True, random_state=42)
    metrics_list = []

    for fold_num, (train_index, val_index) in enumerate(kf.split(X_train_val)):
        X_train_cv, X_val_cv = X_train_val.iloc[train_index], X_train_val.iloc[val_index]
        y_train_cv, y_val_cv = y_train_val.iloc[train_index], y_train_val.iloc[val_index]

        # [Chinese text removed]parameter[Chinese text removed]
        study = optuna.create_study(direction='minimize')
        study.optimize(lambda trial: objective(trial, X_train_cv, y_train_cv), n_trials=10)
        best_params = study.best_params

        # Training final model
        model_final = MLPRegressor(**best_params)
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

        metrics = {
            'fold': fold_num + 1,
            'mse': mse_val,
            'rmse': rmse_val,
            'r2': r2_val,
            'mae': mae_val,
            'mape': mape_val,
            'adj_r2': adj_r2_val
        }
        metrics_list.append(metrics)

        log.info(f"Fold {fold_num + 1}:")
        log.info(f"MSE: {mse_val:.4f}, RMSE: {rmse_val:.4f}, R²: {r2_val:.4f}, Adj R²: {metrics['adj_r2']:.4f}, MAE: {mae_val:.4f}, MAPE: {mape_val:.2f}%")

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

    # Save model
    model_dir = "logbbModel"
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    model_path = os.path.join(model_dir, "mlp_fp_model.joblib")
    joblib.dump(model_final, model_path)
    log.info(f"Model saved to: {model_path}")

    # [Chinese text removed]
    loaded_model = joblib.load(model_path)
    y_pred_test_loaded = loaded_model.predict(X_test)
    
    # [Chinese text removed]
    assert np.allclose(y_pred_test, y_pred_test_loaded), "[Chinese text removed]original[Chinese text removed]！"
    log.info("[Chinese text removed]Success：[Chinese text removed]")

    # [Chinese text removed] CSV [Chinese text removed]
    train_results = pd.DataFrame({
        'True': y_train,
        'Predicted': model_final.predict(X_train)
    })
    test_results = pd.DataFrame({
        'True': y_test,
        'Predicted': y_pred_test
    })
    train_results.to_csv('./result/mlp_fp_train_results.csv', index=False)
    test_results.to_csv('./result/mlp_fp_test_results.csv', index=False)
    log.info("[Chinese text removed] './result/mlp_fp_train_results.csv' [Chinese text removed] './result/mlp_fp_test_results.csv'")


def train_mlp(organ_name, FP=False):
    model_type = "fp" if FP else "mordred"
    model_save_path = f"cbrainModel/mlp_{model_type}_model.joblib"

    # Check if the model already exists
    if os.path.exists(model_save_path):
        log.info(f"Loading existing MLP model from {model_save_path}")
        best_model = joblib.load(model_save_path)
        return {
            'model': best_model,
            'cv_metrics': {},  # You may want to load or calculate these if needed
            'test_metrics': {}  # Same as above
        }

    log.info(f"Training MLP model with {model_type} features")
    

    # Optuna [Chinese text removed]parameter[Chinese text removed]
    study = optuna.create_study(direction='minimize')
    study.optimize(lambda trial: objective(trial, X, y), n_trials=50)

    best_params = study.best_params
    best_model = study.best_trial.user_attrs['best_model']

    # [Chinese text removed]
    preds = best_model.predict(X_test)

    test_metrics = {
        'test_mse': mean_squared_error(y_test, preds),
        'test_rmse': np.sqrt(mean_squared_error(y_test, preds)),
        'test_r2': r2_score(y_test, preds),
        'test_mae': mean_absolute_error(y_test, preds),
        'test_mape': mean_absolute_percentage_error(y_test, preds)
    }
    test_metrics['test_adjusted_r2'] = adjusted_r2(y_test, preds, X_test.shape[1])

    # Save model
    joblib.dump(best_model, model_save_path)
    log.info(f"Model saved to {model_save_path}")

    return {
        'model': best_model,
        'cv_metrics': {},  # [Chinese text removed]returnCross-validation[Chinese text removed]
        'test_metrics': test_metrics
    }


if __name__ == '__main__':
    main()

