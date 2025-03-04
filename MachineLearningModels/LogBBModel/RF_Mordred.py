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
    计算调整后的 R^2
    :param y_true: 真实值
    :param y_pred: 预测值
    :param num_features: 模型使用的特征数量
    :return: 调整后的 R^2
    """
    n = len(y_true)
    r2 = r2_score(y_true, y_pred)
    # 避免除零
    if n - num_features - 1 == 0:
        return r2
    return 1 - (1 - r2) * (n - 1) / (n - num_features - 1)

if __name__ == '__main__':
    # 需要的文件路径
    logBB_data_file = "../../data/logBB_data/logBB.csv"
    logBB_desc_file = "../../data/logBB_data/logBB_w_desc.csv"
    logBB_desc_index_file = "../../data/logBB_data/desc_index.txt"
    test_data_file = "../../data/logBB_data/logBB_test.csv"
    log.info("===============启动Random Forest调参及训练工作===============")
    
    # 变量初始化
    smile_column_name = 'SMILES'
    pred_column_name = 'logBB'
    RFE_features_to_select = 50
    n_optuna_trial = 100
    cv_times = 10
    seed = int(time())

    if not os.path.exists(logBB_data_file):
        raise FileNotFoundError("缺失logBB数据集")
        
    # 特征读取或获取
    if os.path.exists(logBB_desc_file):
        log.info("存在特征文件，进行读取")
        df = pd.read_csv(logBB_desc_file, encoding='utf-8')
        y = df[pred_column_name]
        X = df.drop([smile_column_name, pred_column_name], axis=1)
    else:
        log.info("特征文件不存在，执行特征生成工作") 
        df = pd.read_csv(logBB_data_file, encoding='utf-8')
        df = df.dropna(subset=[pred_column_name])
        df = df.reset_index()
        
        y = df[pred_column_name]
        SMILES = df[smile_column_name]
        
        X = calculate_Mordred_desc(SMILES)
        log.info(f"保存特征数据到csv文件 {logBB_desc_file} 中")
        pd.concat([X, y], axis=1).to_csv(logBB_desc_file, encoding='utf-8', index=False)
        X = X.drop(smile_column_name, axis=1)

    # 在特征选择之前排除 blood mean60min
    if 'blood mean60min' in X.columns:
        X = X.drop('blood mean60min', axis=1)
        log.info("已排除特征: blood mean60min")

    # 特征归一化
    log.info("归一化特征数据")
    sc = MinMaxScaler()
    sc.fit(X)
    X = pd.DataFrame(sc.transform(X))

    # 特征筛选
    if not os.path.exists(logBB_desc_index_file):
        log.info("不存在特征索引文件，进行特征筛选")
        log.info(f"筛选前的特征矩阵形状为：{X.shape}")
        desc_index = (FeatureExtraction(X,
                                      y,
                                      VT_threshold=0.02,
                                      RFE_features_to_select=RFE_features_to_select).
                     feature_extraction(returnIndex=True, index_dtype=int))
        try:
            np.savetxt(logBB_desc_index_file, desc_index, fmt='%d')
            X = X[desc_index]
            log.info(f"特征筛选完成，筛选后的特征矩阵形状为：{X.shape}, 筛选得到的特征索引保存到：{logBB_desc_index_file}")
        except (TypeError, KeyError) as e:
            log.error(e)
            os.remove(logBB_desc_index_file)
            sys.exit()
    else:
        log.info("存在特征索引文件，进行读取")
        desc_index = np.loadtxt(logBB_desc_index_file, dtype=int, delimiter=',').tolist()
        X = X[desc_index]
        log.info(f"读取特征索引完成，筛选后的特征矩阵形状为：{X.shape}")

    # 确保所有列名都是字符串类型
    X.columns = X.columns.astype(str)

    # 1. 修改数据集划分
    # 首先分出独立测试集 (10%)
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=0.1, random_state=42
    )
    log.info(f"将数据集按9:1划分为训练验证集({len(X_train_val)}个样本)和测试集({len(X_test)}个样本)")

    # 在剩余数据中分出验证集 (10%)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=0.1, random_state=42
    )
    log.info(f"将训练验证集按9:1划分为训练集({len(X_train)}个样本)和验证集({len(X_val)}个样本)")

    # 模型调参
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

    log.info("进行Random Forest调参")
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_jobs=4, n_trials=n_optuna_trial)

    log.info(f"最佳参数: {study.best_params}")
    log.info(f"最佳预测结果: {study.best_value}")

    # 使用最佳参数训练模型
    model = RandomForestRegressor(**study.best_params)

    # 训练集交叉验证
    cv = KFold(n_splits=cv_times, random_state=seed, shuffle=True)
    rmse_result_list = []
    r2_result_list = []
    mse_result_list = []
    mae_result_list = []
    adj_r2_result_list = []
    
    log.info(f"使用最佳参数进行{cv_times}折交叉验证")

    for idx, (train_idx, test_idx) in tqdm(enumerate(cv.split(X, y)), desc="交叉验证: ", total=cv_times):
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

    log.info(f"随机种子: {seed}")
    log.info("========训练集结果========")
    log.info(f"RMSE: {round(np.mean(rmse_result_list), 3)}±{round(np.std(rmse_result_list), 3)}")
    log.info(f"MSE: {round(np.mean(mse_result_list), 3)}±{round(np.std(mse_result_list), 3)}")
    log.info(f"MAE: {round(np.mean(mae_result_list), 3)}±{round(np.std(mae_result_list), 3)}")
    log.info(f"R2: {round(np.mean(r2_result_list), 3)}±{round(np.std(r2_result_list), 3)}")
    log.info(f"Adjusted R2: {round(np.mean(adj_r2_result_list), 3)}±{round(np.std(adj_r2_result_list), 3)}")

    # 验证集验证
    y_pred = model.predict(X_val)
    rmse = np.sqrt(mean_squared_error(y_val, y_pred))
    mse = mean_squared_error(y_val, y_pred)
    mae = mean_absolute_error(y_val, y_pred)
    r2 = r2_score(y_val, y_pred)
    adj_r2 = 1 - (1-r2)*(len(y_val)-1)/(len(y_val)-X_val.shape[1]-1)
    
    log.info("========验证集结果========")
    log.info(f"RMSE: {round(rmse, 3)}")
    log.info(f"MSE: {round(mse, 3)}")
    log.info(f"MAE: {round(mae, 3)}")
    log.info(f"R2: {round(r2, 3)}")
    log.info(f"Adjusted R2: {round(adj_r2, 3)}")

    model_save_path = os.path.join("./logbbModel", "rf_mordred_model.joblib")
    joblib.dump(model, model_save_path)
    log.info(f"模型已保存至 {model_save_path}")

    # 2. 添加最终评估指标计算和输出
    # 在所有数据集上进行预测
    y_pred_train = model.predict(X_train)
    y_pred_val = model.predict(X_val)
    y_pred_test = model.predict(X_test)

    # 计算评估指标
    mse_train = mean_squared_error(y_train, y_pred_train)
    rmse_train = np.sqrt(mse_train)
    r2_train = r2_score(y_train, y_pred_train)
    mae_train = mean_absolute_error(y_train, y_pred_train)
    mape_train = mean_absolute_percentage_error(y_train, y_pred_train)
    adj_r2_train = adjusted_r2(y_train, y_pred_train, X_train.shape[1])

    # 验证集指标
    mse_val = mean_squared_error(y_val, y_pred_val)
    rmse_val = np.sqrt(mse_val)
    r2_val = r2_score(y_val, y_pred_val)
    mae_val = mean_absolute_error(y_val, y_pred_val)
    mape_val = mean_absolute_percentage_error(y_val, y_pred_val)
    adj_r2_val = adjusted_r2(y_val, y_pred_val, X_val.shape[1])

    # 测试集指标
    mse_test = mean_squared_error(y_test, y_pred_test)
    rmse_test = np.sqrt(mse_test)
    r2_test = r2_score(y_test, y_pred_test)
    mae_test = mean_absolute_error(y_test, y_pred_test)
    mape_test = mean_absolute_percentage_error(y_test, y_pred_test)
    adj_r2_test = adjusted_r2(y_test, y_pred_test, X_test.shape[1])

    # 输出评估结果
    log.info("最终测试集评估指标：")
    log.info(f"测试集 -> MSE: {mse_test:.4f}, RMSE: {rmse_test:.4f}, R2: {r2_test:.4f}, Adjusted R2: {adj_r2_test:.4f}, MAE: {mae_test:.4f}, MAPE: {mape_test:.2f}%")

    # 只保存测试集结果
    test_results = pd.DataFrame({
        'True Values': y_test, 
        'Predicted Values': y_pred_test
    })

    os.makedirs('./result', exist_ok=True)
    test_results.to_csv('./result/rf_mordred_test_results.csv', index=False)

    log.info("\n最终数据集划分:")
    log.info(f"训练集: {len(X_train)}个样本 ({len(X_train)/len(X)*100:.1f}%) [应约为81%]")
    log.info(f"验证集: {len(X_val)}个样本 ({len(X_val)/len(X)*100:.1f}%) [应约为9%]")
    log.info(f"测试集: {len(X_test)}个样本 ({len(X_test)/len(X)*100:.1f}%) [应约为10%]")

    # 保存训练集和测试集的预测结果到 CSV 文件
    train_results = pd.DataFrame({
        'True Values': y_train,
        'Predicted Values': y_pred_train
    })
    train_results.to_csv('./result/rf_mordred_train_results.csv', index=False)
    test_results.to_csv('./result/rf_mordred_test_results.csv', index=False)
    log.info("训练集和测试集的预测结果已保存到 './result/rf_mordred_train_results.csv' 和 './result/rf_mordred_test_results.csv'")

    # 定义模型保存路径
    model_save_path = os.path.join("./logbbModel", "rf_mordred_model.joblib")

    # 十折交叉验证
    kf = KFold(n_splits=10, shuffle=True, random_state=42)
    metrics_list = []

    for fold_num, (train_index, val_index) in enumerate(kf.split(X)):
        X_train_cv, X_val_cv = X.iloc[train_index], X.iloc[val_index]
        y_train_cv, y_val_cv = y.iloc[train_index], y.iloc[val_index]

        # 使用最佳参数训练模型
        model_cv = RandomForestRegressor(**study.best_params)
        model_cv.fit(X_train_cv, y_train_cv)

        # 预测
        y_val_pred = model_cv.predict(X_val_cv)

        # 计算评估指标
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

    # 平均指标
    avg_metrics = {key: np.mean([m[key] for m in metrics_list]) for key in metrics_list[0] if key != 'fold'}
    log.info("10折交叉验证平均指标：")
    log.info(f"MSE: {avg_metrics['mse']:.4f}")
    log.info(f"RMSE: {avg_metrics['rmse']:.4f}")
    log.info(f"R2: {avg_metrics['r2']:.4f}")
    log.info(f"Adjusted R2: {avg_metrics['adj_r2']:.4f}")
    log.info(f"MAE: {avg_metrics['mae']:.4f}")
    log.info(f"MAPE: {avg_metrics['mape']:.2f}%")

    # 测试集评估
    model_final = RandomForestRegressor(**study.best_params)
    model_final.fit(X_train, y_train)
    y_pred_test = model_final.predict(X_test)
    
    mse_test = mean_squared_error(y_test, y_pred_test)
    rmse_test = np.sqrt(mse_test)
    r2_test = r2_score(y_test, y_pred_test)
    mae_test = mean_absolute_error(y_test, y_pred_test)
    mape_test = mean_absolute_percentage_error(y_test, y_pred_test)
    adj_r2_test = adjusted_r2(y_test, y_pred_test, X_test.shape[1])

    log.info("测试集评估指标：")
    log.info(f"MSE: {mse_test:.4f}, RMSE: {rmse_test:.4f}, R2: {r2_test:.4f}, Adjusted R2: {adj_r2_test:.4f}, MAE: {mae_test:.4f}, MAPE: {mape_test:.2f}%")

    # 保存训练集和测试集的预测结果到 CSV 文件
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
    log.info("训练集和测试集的预测结果已保存到 './result/rf_mordred_train_results.csv' 和 './result/rf_mordred_test_results.csv'")

    # 保存模型
    model_save_path = os.path.join("./logbbModel", "rf_mordred_model.joblib")
    joblib.dump(model_final, model_save_path)
    log.info(f"模型已保存至 {model_save_path}")

    # 加载模型并验证
    loaded_model = joblib.load(model_save_path)
    y_pred_test_loaded = loaded_model.predict(X_test)
    
    # 验证加载的模型预测结果是否与原模型一致
    assert np.allclose(y_pred_test, y_pred_test_loaded), "加载的模型与原始模型预测结果不一致！"
    log.info("模型加载验证成功：预测结果与原模型一致")
