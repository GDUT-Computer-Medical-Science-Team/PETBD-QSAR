import optuna
import lightgbm as lgb
import datetime
import os
import time
import traceback
import pandas as pd
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from sklearn.model_selection import KFold
import joblib
import utils.datasets_loader as loader
from preprocess.MedicalDatasetsHandler import MedicalDatasetsHandler
from utils.DataLogger import DataLogger
from tqdm import tqdm

# 初始化结果保存目录
cur_time = time.localtime()
result_parent_dir = f"../../result/{time.strftime('%Y%m%d', cur_time)}"
result_dir = f"{result_parent_dir}/{time.strftime('%H%M%S', cur_time)}"
os.makedirs(result_dir, exist_ok=True)
main_log = DataLogger(f"{result_dir}/run.log").getlog(disable_console_output=True)
log = DataLogger().getlog("run")


def read_merged_datafile(self,
                         merged_filepath,
                         organ_names: list,
                         certain_time: int,
                         overwrite=False,
                         is_sd=False):
    """
    读取原始的整合数据集，选定器官和时间进行筛选并保存
    :param merged_filepath: 整合数据集路径
    :param organ_names: 器官名列表
    :param certain_time: 指定的时间，单位为分钟
    :param overwrite: 是否启用覆盖模式
    :param is_sd: 是否取方差值(sd)，默认为False，即只取平均值(mean)
    """
    self.__merged_filepath = merged_filepath
    # 默认保存目录为原始整合文件所在的根目录
    if self.__merged_filepath is not None and len(self.__merged_filepath) > 0:
        self.__saving_folder = os.path.split(self.__merged_filepath)[0]
    else:
        raise ValueError("参数merged_filepath错误")

    organ_data_filepath = os.path.join(self.__saving_folder, "OrganData.csv")
    self.__organ_time_data_filepath = os.path.join(self.__saving_folder, f"OrganDataAt{certain_time}min.csv")
    log.info("准备读取原始的整合数据集，并按照提供的器官名和时间点进行筛选")

    # 获取指定器官的全部数据
    if overwrite or not os.path.exists(organ_data_filepath):
        log.info(f"正在按照提供的器官名: {organ_names} 进行筛选: ")
        save_organ_data_by_names(root_filepath=self.__merged_filepath,
                                 target_filepath=organ_data_filepath,
                                 organ_names=organ_names,
                                 is_sd=is_sd)
    else:
        log.info(f"存在已完成器官名筛选的csv文件: {organ_data_filepath}，跳过器官名筛选步骤")
    # 获取指定时间点的全部数据
    if overwrite or not os.path.exists(self.__organ_time_data_filepath):
        df = pd.DataFrame()
        log.info(f"正在按照提供的时间点进行进一步筛选，筛选的时间点为: {certain_time}min")
        # 获取每个器官对应时间的数据并整合
        for organ_name in tqdm(organ_names, desc=f"正在从指定器官数据从提取{certain_time}min的数据: "):
            df = pd.concat([df, get_certain_time_organ_data(root_filepath=organ_data_filepath,
                                                            organ_name=organ_name,
                                                            certain_time=certain_time,
                                                            is_sd=is_sd)],
                           axis=1)
        log.info(f"数据筛选完成，以csv格式保存至{self.__organ_time_data_filepath}")
        df.to_csv(self.__organ_time_data_filepath, encoding='utf-8')
    else:
        log.info(f"存在已完成时间点筛选的csv文件: {self.__organ_time_data_filepath}，跳过时间点筛选步骤")


def transform_organ_time_data_to_tensor_dataset(self,
                                                test_size=0.2,
                                                double_index=False,
                                                FP=False,
                                                overwrite=False):
    """
    读取csv浓度数据文件，进行数据预处理，并进行训练集、测试集分割，最后保存为TensorDataset数据集
    :param test_size: 测试集大小，范围为[0.0, 1.0)
    :param double_index: 是否是双倍特征筛选量
    :param FP: 是否计算分子指纹
    :param overwrite: 是否覆盖现有的npy文件
    """
    if test_size < 0.0 or test_size >= 1.0:
        raise ValueError("参数test_size超过范围[0.0, 1.0)")
    npy_file_path = self.__transform_organs_data(FP=FP,
                                                 double_index=double_index,
                                                 overwrite=overwrite)
    self.__split_df2TensorDataset(npy_file_path, test_size=test_size)


def check_datasets_exist(parent_folder: str):
    if os.path.exists(parent_folder):
        if not os.path.isdir(parent_folder):
            raise NotADirectoryError(f"错误：{parent_folder}不是目录")
        return any(file.endswith("_dataset.pt") for file in os.listdir(parent_folder))
    return False


def check_data_exist(merge_filepath, organ_names_list, certain_time,
                     train_dir_path, test_dir_path,
                     FP=False, overwrite=False):
    try:
        flag = check_datasets_exist(train_dir_path) and check_datasets_exist(test_dir_path)
    except NotADirectoryError:
        log.error(traceback.format_exc())
        flag = False

    if not overwrite and flag:
        log.info(f"存在TensorDatasets数据，无须进行数据获取操作")
    else:
        log.info(f"不存在TensorDatasets数据，开始进行数据获取操作")
        if not os.path.exists(merge_filepath):
            raise FileNotFoundError(f"数据表文件\"{merge_filepath}\"未找到")
        md = MedicalDatasetsHandler()

        md.read_merged_datafile(merged_filepath=merge_filepath,
                                organ_names=organ_names_list,
                                certain_time=certain_time,
                                overwrite=overwrite)
        md.transform_organ_time_data_to_tensor_dataset(test_size=0.1,
                                                       double_index=False,
                                                       FP=FP,
                                                       overwrite=overwrite)
        log.info(f"数据获取完成")


def adjusted_r2_score(r2, n, k):
    """
    计算调整后的 R^2 值。
    r2: 原始 R^2 值
    n: 样本数量
    k: 特征数量
    """
    return 1 - (1 - r2) * ((n - 1) / (n - k - 1))


def objective(trial, X, y):
    param = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 3000),
        'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.3, log=True),
        'num_leaves': trial.suggest_int('num_leaves', 20, 300),
        'max_depth': trial.suggest_int('max_depth', 3, 12),
        'min_child_samples': trial.suggest_int('min_child_samples', 1, 100),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
        'min_split_gain': trial.suggest_float('min_split_gain', 1e-8, 1.0, log=True),
        'min_child_weight': trial.suggest_float('min_child_weight', 1e-8, 1.0, log=True),
    }

    cv = KFold(n_splits=10, shuffle=True, random_state=42)
    cv_scores = {
        'rmse': [],
        'r2': [],
        'mae': [],
        'mape': [],
        'adjusted_r2': []
    }

    for fold, (train_idx, val_idx) in enumerate(tqdm(cv.split(X), desc="Cross-validation")):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        model = lgb.LGBMRegressor(**param)
        model.fit(X_train, y_train)
        preds = model.predict(X_val)

        r2 = r2_score(y_val, preds)
        rmse = np.sqrt(mean_squared_error(y_val, preds))
        mae = mean_absolute_error(y_val, preds)
        mape = mean_absolute_percentage_error(y_val, preds)
        adj_r2 = adjusted_r2_score(r2, len(y_val), X_val.shape[1])

        cv_scores['rmse'].append(rmse)
        cv_scores['r2'].append(r2)
        cv_scores['mae'].append(mae)
        cv_scores['mape'].append(mape)
        cv_scores['adjusted_r2'].append(adj_r2)

    # 保存交叉验证结果，供后续使用
    trial.set_user_attr('cv_scores', cv_scores)

    return np.mean(cv_scores['rmse'])


def train_lightgbm(organ_name, FP=False):
    model_type = "Fingerprint" if FP else "Mordred Descriptor"
    log.info(f"Training LightGBM model with {model_type} features")

    X, y = loader.get_sklearn_data('../../data/train/train_organ_df.npy', organ_name)
    X_test, y_test = loader.get_sklearn_data('../../data/test/test_organ_df.npy', organ_name)

    # 将 X 和 y 转换为 NumPy 数组以避免索引错误
    X = np.array(X)
    y = np.array(y)

    study = optuna.create_study(direction='minimize')
    study.optimize(lambda trial: objective(trial, X, y), n_trials=50)

    best_params = study.best_params
    best_gbm = lgb.LGBMRegressor(**best_params)
    best_gbm.fit(X, y)

    # 获取最佳trial的交叉验证结果
    best_cv_scores = study.best_trial.user_attrs['cv_scores']

    # 计算测试集指标
    preds = best_gbm.predict(X_test)
    test_r2 = r2_score(y_test, preds)
    test_mse = mean_squared_error(y_test, preds)
    test_mae = mean_absolute_error(y_test, preds)
    test_mape = mean_absolute_percentage_error(y_test, preds)
    test_rmse = np.sqrt(test_mse)
    test_adjusted_r2 = adjusted_r2_score(test_r2, len(y_test), X_test.shape[1])

    # 打印综合结果报告
    log.info(f"\n========{model_type} 模型评估报告========")
    
    # 打印十折交叉验证结果
    log.info("\n十折交叉验证结果:")
    log.info(f"平均RMSE: {np.mean(best_cv_scores['rmse']):.3f} (±{np.std(best_cv_scores['rmse']):.3f})")
    log.info(f"平均R2: {np.mean(best_cv_scores['r2']):.3f} (±{np.std(best_cv_scores['r2']):.3f})")
    log.info(f"平均调整后R2: {np.mean(best_cv_scores['adjusted_r2']):.3f} (±{np.std(best_cv_scores['adjusted_r2']):.3f})")
    log.info(f"平均MAE: {np.mean(best_cv_scores['mae']):.3f} (±{np.std(best_cv_scores['mae']):.3f})")
    log.info(f"平均MAPE: {np.mean(best_cv_scores['mape']):.3f}% (±{np.std(best_cv_scores['mape']):.3f}%)")

    # 打印测试集结果
    log.info("\n测试集结果:")
    log.info(f"RMSE: {test_rmse:.3f}")
    log.info(f"R2: {test_r2:.3f}")
    log.info(f"调整后R2: {test_adjusted_r2:.3f}")
    log.info(f"MAE: {test_mae:.3f}")
    log.info(f"MAPE: {test_mape:.3f}%")
    log.info(f"MSE: {test_mse:.3f}")

    # 保存模型
    model_dir = "cbrainModel"
    os.makedirs(model_dir, exist_ok=True)
    model_save_path = f"{model_dir}/lightgbm_{model_type}_model.joblib"
    joblib.dump(best_gbm, model_save_path)
    log.info(f"\n模型已保存至 {model_save_path}")

    return {
        'model': best_gbm,
        'cv_metrics': {
            'rmse': np.mean(best_cv_scores['rmse']),
            'r2': np.mean(best_cv_scores['r2']),
            'adjusted_r2': np.mean(best_cv_scores['adjusted_r2']),
            'mae': np.mean(best_cv_scores['mae']),
            'mape': np.mean(best_cv_scores['mape'])
        },
        'test_metrics': {
            'rmse': test_rmse,
            'r2': test_r2,
            'adjusted_r2': test_adjusted_r2,
            'mae': test_mae,
            'mape': test_mape,
            'mse': test_mse
        }
    }


if __name__ == '__main__':
    organ_name = 'brain'
    merge_filepath = "../../data/数据表汇总.xlsx"
    organ_names_list = ['blood', 'bone', 'brain', 'fat', 'heart',
                        'intestine', 'kidney', 'liver', 'lung', 'muscle',
                        'pancreas', 'spleen', 'stomach', 'uterus']
    certain_time = 60
    train_datasets_dir = "../../data/train/datasets"
    test_datasets_dir = "../../data/test/datasets"

    # 创建模型保存目录
    model_dir = "model"
    os.makedirs(model_dir, exist_ok=True)
    overwrite = False
    
    # 存储两种模型的结果
    results = {}

    # 训练分子指纹模型
    FP = True
    check_data_exist(merge_filepath, organ_names_list, certain_time,
                     train_datasets_dir, test_datasets_dir,
                     FP=FP, overwrite=overwrite)
    results['fingerprint'] = train_lightgbm(organ_name, FP=FP)

    # 训练分子描述符模型
    FP = False
    check_data_exist(merge_filepath, organ_names_list, certain_time,
                     train_datasets_dir, test_datasets_dir,
                     FP=FP, overwrite=overwrite)
    results['mordred'] = train_lightgbm(organ_name, FP=FP)

    # 打印最终的比较结果
    log.info("\n========= LightGBM 模型最终评估报告 =========")
    
    # 分子指纹模型结果
    log.info("\n【分子指纹模型】")
    log.info("十折交叉验证结果:")
    cv_metrics = results['fingerprint']['cv_metrics']
    log.info(f"平均RMSE: {cv_metrics['rmse']:.3f}")
    log.info(f"平均R2: {cv_metrics['r2']:.3f}")
    log.info(f"平均调整后R2: {cv_metrics['adjusted_r2']:.3f}")
    log.info(f"平均MAE: {cv_metrics['mae']:.3f}")
    log.info(f"平均MAPE: {cv_metrics['mape']:.3f}%")
    
    log.info("\n测试集结果:")
    test_metrics = results['fingerprint']['test_metrics']
    log.info(f"RMSE: {test_metrics['rmse']:.3f}")
    log.info(f"R2: {test_metrics['r2']:.3f}")
    log.info(f"调整后R2: {test_metrics['adjusted_r2']:.3f}")
    log.info(f"MAE: {test_metrics['mae']:.3f}")
    log.info(f"MAPE: {test_metrics['mape']:.3f}%")
    log.info(f"MSE: {test_metrics['mse']:.3f}")

    # 分子描述符模型结果
    log.info("\n【分子描述符模型】")
    log.info("十折交叉验证结果:")
    cv_metrics = results['mordred']['cv_metrics']
    log.info(f"平均RMSE: {cv_metrics['rmse']:.3f}")
    log.info(f"平均R2: {cv_metrics['r2']:.3f}")
    log.info(f"平均调整后R2: {cv_metrics['adjusted_r2']:.3f}")
    log.info(f"平均MAE: {cv_metrics['mae']:.3f}")
    log.info(f"平均MAPE: {cv_metrics['mape']:.3f}%")
    
    log.info("\n测试集结果:")
    test_metrics = results['mordred']['test_metrics']
    log.info(f"RMSE: {test_metrics['rmse']:.3f}")
    log.info(f"R2: {test_metrics['r2']:.3f}")
    log.info(f"调整后R2: {test_metrics['adjusted_r2']:.3f}")
    log.info(f"MAE: {test_metrics['mae']:.3f}")
    log.info(f"MAPE: {test_metrics['mape']:.3f}%")
    log.info(f"MSE: {test_metrics['mse']:.3f}")
