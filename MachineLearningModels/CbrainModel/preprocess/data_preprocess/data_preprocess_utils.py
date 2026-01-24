import traceback

import pandas

import pandas as pd
import numpy as np
from deepchem import feat
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from tqdm import tqdm
from utils.DataLogger import DataLogger
"""
"""

log = DataLogger().getlog("data_preprocess_utils")


def save_organ_data_by_names(root_filepath: str, target_filepath: str, organ_names: list, is_sd=False):
    """

    """
    if root_filepath is None or len(root_filepath) == 0 or target_filepath is None or len(target_filepath) == 0:
        raise ValueError("输入或输出的文件路径错误")
    if organ_names is None or len(organ_names) == 0:
        raise ValueError("器官名列表为空")
    if root_filepath.endswith('xlsx') or root_filepath.endswith('xls'):
        raw_df = pd.read_excel(root_filepath, index_col=[0, 1], engine='openpyxl')
    elif root_filepath.endswith('csv'):
        raw_df = pd.read_csv(root_filepath, index_col=[0, 1])
    else:
        raise TypeError("输入的文件并非excel类型的格式(xlsx, xls, csv)")
    df_list = []
    log.info(f"准备获取指定器官数据，器官数量为:{len(organ_names)}")
    for organ_name in tqdm(organ_names, desc="正在获取指定器官数据: "):
        if not is_sd:
            df = raw_df.loc[:, raw_df.columns.str.startswith(f'{organ_name} mean')]
        else:
            df = raw_df.loc[:, raw_df.columns.str.startswith(f'{organ_name} sd')]
        df_list.append(df)
    df = pd.concat(df_list, axis=1)
    df.to_csv(target_filepath, encoding='utf-8')
    log.info(f"数据获取成功，以csv格式保存至{target_filepath}")


def get_certain_time_organ_data(root_filepath: str, organ_name: str, certain_time: int, is_sd=False) \
        -> pandas.DataFrame:
    """

    """
    if root_filepath is None or len(root_filepath) == 0:
        raise ValueError("参数root_filepath错误")
    if root_filepath.endswith('xlsx') or root_filepath.endswith('xls'):
        raw_df = pd.read_excel(root_filepath, index_col=[0, 1], engine='openpyxl')
    elif root_filepath.endswith('csv'):
        raw_df = pd.read_csv(root_filepath, index_col=[0, 1])
    else:
        raise TypeError("参数root_filepath错误，输入的文件并非excel类型的格式(xlsx, xls, csv)")
    if organ_name is None or len(organ_name) == 0:
        raise ValueError("参数organ_name错误")
    if is_sd:
        organ_df = raw_df.loc[:, raw_df.columns.str.startswith(f'{organ_name.lower()} sd{certain_time}min')]
    else:
        organ_df = raw_df.loc[:, raw_df.columns.str.startswith(f'{organ_name.lower()} mean{certain_time}min')]
    return organ_df


def save_max_organ_data(root_filepath: str, target_filepath: str, organ_name: str):
    """

    """
    if root_filepath.endswith('xlsx') or root_filepath.endswith('xls'):
        raw_df = pd.read_excel(root_filepath, index_col=[0, 1], engine='openpyxl')
    elif root_filepath.endswith('csv'):
        raw_df = pd.read_csv(root_filepath, index_col=[0, 1])
    else:
        raise TypeError("参数root_filepath错误，输入的文件并非excel类型的格式(xlsx, xls, csv)")
    organ_df = raw_df.loc[:, raw_df.columns.str.startswith(f'{organ_name.lower()} mean')]
    max_concentration2time = dict()

    for index, row_data in organ_df.iterrows():
        row_data = row_data.dropna()
        if row_data.empty:
            continue
        else:
            num2time = dict()
            row_data = row_data.to_frame()
            row_data = pd.DataFrame(row_data.values.T, columns=row_data.index)
            for column in row_data.columns.to_list():
                concentration_num = float(row_data[column].values[0])
                num2time[column.split(" ")[1].replace('mean', '')] = concentration_num
        sorted_data = sorted(num2time.items(), key=lambda kv: (kv[1], kv[0]), reverse=True)
        max_concentration2time[index] = sorted_data[0]
    max_data_list = []
    for key, value in max_concentration2time.items():
        index = key[0]
        smiles = key[1]
        time = value[0]
        concentration_num = value[1]
        max_data_list.append([index, smiles, concentration_num, time])
    df = pd.DataFrame(data=max_data_list,
                      columns=['Compound index', 'SMILES', 'Max Concentration', 'Reach time'])
    df.to_csv(target_filepath, index=False)


def calculate_Mordred_desc(datasrc, drop_duplicates=False):
    """
    """
    if isinstance(datasrc, str):
        df = pd.read_csv(datasrc)
    elif isinstance(datasrc, pd.DataFrame) or isinstance(datasrc, pd.Series):
        df = datasrc
    else:
        raise ValueError("错误的datasrc类型(str, DataFrame, Series)")

    if isinstance(datasrc, str) or isinstance(datasrc, pd.DataFrame):
        SMILES = df['SMILES']
    else:
        SMILES = df

    featurizer = feat.MordredDescriptors(ignore_3D=True)
    X1 = []
    for smiles in tqdm(SMILES, desc="正在计算Mordred描述符: "):
        try:
            X1.append(featurizer.featurize(smiles)[0])
        except RuntimeWarning as e:
            log.error(f"化合物 {smiles} 计算Mordred描述符出现运行时警告: ")
            log.error(traceback.format_exc())
    X = pd.DataFrame(data=X1)
    #X = pd.concat([X, X1], axis=1)

    if not X.empty:
        X = clean_desc_dataframe(X, drop_duplicates=drop_duplicates)

        df = pd.concat([df, X], axis=1)

        return df
    else:
        raise ValueError("Empty dataframe")

def get_X_y_smiles(csv_file, smile_col=2, label_col=0, desc_start_col=3):
    """
    """
    df = pd.read_csv(csv_file)
    df = clean_desc_dataframe(df)
    X = df.iloc[:, desc_start_col:]
    y = df.iloc[:, label_col]
    smiles = df.iloc[:, smile_col]
    return X, y, smiles


def split_null_from_data(df: pd.DataFrame):
    """

    """
    data_df = df.dropna(axis=0)
    empty_df = df.drop(index=data_df.index)
    return data_df.reset_index(drop=True), empty_df.reset_index(drop=True)


def clean_desc_dataframe(df: pd.DataFrame, axis=1, drop_duplicates=True) -> pd.DataFrame:
    """

    """
    log.info("执行特征Dataframe预处理")
    df.replace(["#NAME?", np.inf, -np.inf], np.nan, inplace=True)
    df.replace("#NUM!", np.nan, inplace=True)
    df = df.dropna(axis=axis)
    if drop_duplicates:
        df = df.drop_duplicates()
    return df
