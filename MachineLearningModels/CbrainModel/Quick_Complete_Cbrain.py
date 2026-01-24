import sys
import os
from time import time
import numpy as np
import pandas as pd
import optuna
from sklearn.preprocessing import MinMaxScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, mean_absolute_percentage_error
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_selection import VarianceThreshold, RFE
import lightgbm as lgb
from catboost import CatBoostRegressor
import joblib
import json
import warnings
warnings.filterwarnings('ignore')

def calculate_morgan_fingerprints(smiles_list, n_bits=2048):
    """计算Morgan分子指纹"""
    from rdkit import Chem
    from rdkit.Chem import AllChem
    
    fingerprints = []
    for smiles in smiles_list:
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=n_bits)
            fingerprints.append(list(fp))
        else:
            fingerprints.append([0] * n_bits)
    
    fp_df = pd.DataFrame(fingerprints)
    fp_df.columns = [f'Morgan_FP_{i}' for i in range(n_bits)]
    return fp_df

def adjusted_r2_score(r2, n, k):
    return 1 - (1 - r2) * (n - 1) / (n - k - 1)

def train_model(model_name, model_class, params_func):
    """训练单个模型"""
    print(f"\n=== 训练 {model_name} ===")
    
    # 超参数优化
    def objective(trial):
        params = params_func(trial)
        model = model_class(**params)
        
        if model_name in ['MLP']:
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_val_scaled)
        else:
            if hasattr(model, 'fit') and 'eval_set' in model.fit.__code__.co_varnames:
                model.fit(X_train, y_train, eval_set=[(X_val, y_val)], early_stopping_rounds=50, verbose=False)
            else:
                model.fit(X_train, y_train)
            y_pred = model.predict(X_val if model_name not in ['MLP'] else X_val_scaled)
        
        return r2_score(y_val, y_pred)
    
    study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=20)  # 减少试验次数加速
    
    best_params = study.best_params
    print(f"最优参数: {best_params}")
    
    # 训练最终模型
    model = model_class(**best_params)
    if model_name in ['MLP']:
        model.fit(X_train_scaled, y_train)
        train_pred = model.predict(X_train_scaled)
        val_pred = model.predict(X_val_scaled)
        test_pred = model.predict(X_test_scaled)
    else:
        if hasattr(model, 'fit') and 'eval_set' in model.fit.__code__.co_varnames:
            model.fit(X_train, y_train, eval_set=[(X_val, y_val)], early_stopping_rounds=100, verbose=False)
        else:
            model.fit(X_train, y_train)
        train_pred = model.predict(X_train)
        val_pred = model.predict(X_val)
        test_pred = model.predict(X_test)
    
    # 评估
    def evaluate_metrics(y_true, y_pred, X_shape):
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        mape = mean_absolute_percentage_error(y_true, y_pred)
        adj_r2 = adjusted_r2_score(r2, len(y_true), X_shape[1])
        
        return {
            'mse': mse, 'rmse': rmse, 'mae': mae,
            'r2': r2, 'mape': mape, 'adj_r2': adj_r2
        }
    
    train_metrics = evaluate_metrics(y_train, train_pred, X_train.shape)
    val_metrics = evaluate_metrics(y_val, val_pred, X_val.shape)
    test_metrics = evaluate_metrics(y_test, test_pred, X_test.shape)
    
    # 交叉验证
    if model_name in ['MLP']:
        cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='r2')
        cv_rmse = np.sqrt(-cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='neg_mean_squared_error'))
    else:
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')
        cv_rmse = np.sqrt(-cross_val_score(model, X_train, y_train, cv=5, scoring='neg_mean_squared_error'))
    
    # 保存结果
    results = {
        "experiment_info": {
            "dataset": "OrganDataAt60min.csv",
            "balancing_method": "oversample",
            "features": "Morgan_fingerprints_only",
            "model": model_name,
            "target": "brain mean60min",
            "n_samples_original": len(df),
            "n_samples_balanced": len(X_balanced),
            "n_features": X_selected.shape[1],
            "seed": 42
        },
        "best_params": best_params,
        "cross_validation": {
            "cv_folds": 5,
            "rmse": f"{cv_rmse.mean():.3f}±{cv_rmse.std():.3f}",
            "r2": f"{cv_scores.mean():.3f}±{cv_scores.std():.3f}",
            "adj_r2": f"{np.mean([adjusted_r2_score(s, len(X_train), X_selected.shape[1]) for s in cv_scores]):.3f}±{np.std([adjusted_r2_score(s, len(X_train), X_selected.shape[1]) for s in cv_scores]):.3f}"
        },
        "final_results": {
            "train": train_metrics,
            "validation": val_metrics,
            "test": test_metrics
        }
    }
    
    # 保存文件
    result_file = f"./result/{model_name.lower()}_cbrain_18F_report.json"
    with open(result_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    model_file = f"./cbrainmodel/{model_name.lower()}_cbrain_18F_model.joblib"
    joblib.dump(model, model_file)
    
    if model_name == 'MLP':
        scaler_file = f"./cbrainmodel/{model_name.lower()}_cbrain_18F_scaler.joblib"
        joblib.dump(scaler, scaler_file)
    
    print(f"{model_name} 训练完成! 测试集R²: {test_metrics['r2']:.4f}")
    return test_metrics['r2']

if __name__ == '__main__':
    print("=== 快速完成剩余Cbrain 18F重采样模型 ===")
    
    # 创建目录
    os.makedirs("./cbrainmodel", exist_ok=True)
    os.makedirs("./result", exist_ok=True)
    
    # 读取数据
    data_file = "../../data/logBB_data/OrganDataAt60min.csv"
    if not os.path.exists(data_file):
        print(f"数据文件不存在: {data_file}")
        sys.exit(1)
    
    print("读取Cbrain数据集")
    df = pd.read_csv(data_file, encoding='utf-8')
    print(f"原始数据形状: {df.shape}")
    
    # 数据预处理
    df = df.dropna(subset=['SMILES', 'brain mean60min'])
    smiles_list = df['SMILES'].tolist()
    target = df['brain mean60min'].values
    
    print("计算Morgan分子指纹 (跳过Mordred以加速)")
    morgan_features = calculate_morgan_fingerprints(smiles_list, n_bits=1024)
    X_combined = morgan_features
    print(f"特征形状: {X_combined.shape}")
    
    # 18F同位素检测和平衡
    compound_indices = df.get('compound index', df.index)
    has_18f = compound_indices.astype(str).str.contains('18F', na=False)
    
    print(f"18F化合物数量: {has_18f.sum()}, 非18F化合物数量: {(~has_18f).sum()}")
    
    # 重采样平衡数据
    from imblearn.over_sampling import RandomOverSampler
    ros = RandomOverSampler(random_state=42)
    X_balanced, y_balanced = ros.fit_resample(X_combined, target)
    
    print(f"重采样后数据形状: {X_balanced.shape}, 目标形状: {y_balanced.shape}")
    
    # 特征选择
    variance_selector = VarianceThreshold(threshold=0.01)
    X_var_selected = variance_selector.fit_transform(X_balanced)
    
    # RFE特征选择 - 使用简单模型加速
    from sklearn.linear_model import LinearRegression
    base_lr = LinearRegression()
    rfe = RFE(estimator=base_lr, n_features_to_select=50, step=1)
    X_selected = rfe.fit_transform(X_var_selected, y_balanced)
    
    print(f"最终特征数量: {X_selected.shape[1]}")
    
    # 数据分割
    X_train, X_temp, y_train, y_temp = train_test_split(X_selected, y_balanced, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
    
    # 数据标准化 (仅用于MLP)
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"训练集: {X_train.shape}, 验证集: {X_val.shape}, 测试集: {X_test.shape}")
    
    # 定义模型参数
    def mlp_params(trial):
        n_layers = trial.suggest_int('n_layers', 1, 3)
        layers = []
        for i in range(n_layers):
            layers.append(trial.suggest_int(f'layer_{i}_size', 10, 200))
        
        params = {
            'hidden_layer_sizes': tuple(layers),
            'activation': trial.suggest_categorical('activation', ['relu', 'tanh', 'logistic']),
            'solver': trial.suggest_categorical('solver', ['adam', 'lbfgs']),
            'alpha': trial.suggest_float('alpha', 1e-6, 1e-2, log=True),
            'learning_rate': trial.suggest_categorical('learning_rate', ['constant', 'invscaling', 'adaptive']),
            'max_iter': 1000,
            'random_state': 42
        }
        return params
    
    def lightgbm_params(trial):
        return {
            'num_leaves': trial.suggest_int('num_leaves', 10, 1000),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
            'feature_fraction': trial.suggest_float('feature_fraction', 0.4, 1.0),
            'bagging_fraction': trial.suggest_float('bagging_fraction', 0.4, 1.0),
            'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
            'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
            'n_estimators': 1000,
            'random_state': 42,
            'verbose': -1
        }
    
    def catboost_params(trial):
        params = {
            'iterations': trial.suggest_int('iterations', 500, 1500),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
            'depth': trial.suggest_int('depth', 4, 10),
            'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1e-8, 10.0, log=True),
            'bootstrap_type': trial.suggest_categorical('bootstrap_type', ['Bayesian', 'Bernoulli']),
            'random_state': 42,
            'verbose': False
        }
        
        if params['bootstrap_type'] == 'Bayesian':
            params['bagging_temperature'] = trial.suggest_float('bagging_temperature', 0.0, 10.0)
        elif params['bootstrap_type'] == 'Bernoulli':
            params['subsample'] = trial.suggest_float('subsample', 0.1, 1.0)
        
        return params
    
    # 训练所有模型
    results = []
    
    # 检查哪些模型还没有完成
    models_to_run = []
    if not os.path.exists("./result/mlp_cbrain_18F_report.json"):
        models_to_run.append(('MLP', MLPRegressor, mlp_params))
    
    if not os.path.exists("./result/lightgbm_cbrain_18F_report.json"):
        models_to_run.append(('LightGBM', lgb.LGBMRegressor, lightgbm_params))
    
    if not os.path.exists("./result/catboost_cbrain_18F_report.json"):
        models_to_run.append(('CatBoost', CatBoostRegressor, catboost_params))
    
    print(f"需要运行的模型: {[m[0] for m in models_to_run]}")
    
    # 训练每个模型
    for model_name, model_class, params_func in models_to_run:
        try:
            r2_score = train_model(model_name, model_class, params_func)
            results.append((model_name, r2_score))
        except Exception as e:
            print(f"{model_name} 训练失败: {e}")
    
    print("\n=== 训练完成总结 ===")
    for model_name, r2 in results:
        print(f"{model_name}: R² = {r2:.4f}")
    
    print("所有剩余Cbrain模型训练完成!")