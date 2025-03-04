import traceback

import pandas

import pandas as pd
import numpy as np
from deepchem import feat
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from tqdm import tqdm
from utils.DataLogger import DataLogger
"""
    提供各种数据预处理的工具方法
"""

log = DataLogger().getlog("data_preprocess_utils")


def save_organ_data_by_names(root_filepath: str, target_filepath: str, organ_names: list, is_sd=False):
    """
    从root_filepath中筛选指定器官的浓度数据，以csv格式保存到target_filepath中

    :param root_filepath: 源文件路径，其中第一列和第二列分别为compound index和SMILES
    :param target_filepath: 目标csv文件路径
    :param is_sd: 是否取方差值(sd)，默认为False，即只取平均值(mean)
    :param organ_names: 选定的器官名列表
    """
    if root_filepath is None or len(root_filepath) == 0 or target_filepath is None or len(target_filepath) == 0:
        raise ValueError("输入或输出的文件路径错误")
    if organ_names is None or len(organ_names) == 0:
        raise ValueError("器官名列表为空")
    # 根据后缀读取并包装为dataframe
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
    筛选出指定器官在指定时间的浓度数据

    :param root_filepath: 源文件路径
    :param organ_name: 选定的器官名
    :param certain_time: 选定的时间点
    :param is_sd: 是否取方差值(sd)，默认为False，即只取平均值(mean)
    :return: 筛选得到的浓度数据dataframe
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
    筛选出源文件中每个药物在指定器官的最大浓度以及触及时间，并保存到目标csv文件中

    :param root_filepath: 源文件路径
    :param target_filepath: 目标csv文件路径
    :param organ_name: 选定的器官名
    """
    # 读取数据并选择指定器官的全部浓度数据
    if root_filepath.endswith('xlsx') or root_filepath.endswith('xls'):
        raw_df = pd.read_excel(root_filepath, index_col=[0, 1], engine='openpyxl')
    elif root_filepath.endswith('csv'):
        raw_df = pd.read_csv(root_filepath, index_col=[0, 1])
    else:
        raise TypeError("参数root_filepath错误，输入的文件并非excel类型的格式(xlsx, xls, csv)")
    organ_df = raw_df.loc[:, raw_df.columns.str.startswith(f'{organ_name.lower()} mean')]
    # 保存每个药物到达最大浓度的数据
    max_concentration2time = dict()

    # 遍历每一款药物（index同时记录了文献编号和SMILES）
    for index, row_data in organ_df.iterrows():
        # 去除没有数据的列
        row_data = row_data.dropna()
        if row_data.empty:
            continue
        else:
            # 用于保存每个浓度数据与时间的对应关系
            num2time = dict()
            # 转换Series为Dataframe
            row_data = row_data.to_frame()
            row_data = pd.DataFrame(row_data.values.T, columns=row_data.index)
            for column in row_data.columns.to_list():
                concentration_num = float(row_data[column].values[0])
                # 将时间与浓度数据作为键值对保存到字典中
                num2time[column.split(" ")[1].replace('mean', '')] = concentration_num
        sorted_data = sorted(num2time.items(), key=lambda kv: (kv[1], kv[0]), reverse=True)
        # 保存药物索引与最大浓度数据
        max_concentration2time[index] = sorted_data[0]
    # 将字典转换成Dataframe所需的列表格式
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


# TODO: 将分子指纹的计算方法改为用PadelpyCall方法
def calculate_Mordred_desc(datasrc, drop_duplicates=False):
    """
    从datasrc中的SMILES计算描述符并返回带描述符的datasrc数据
    :param datasrc: 需要计算的带SMILES的数据源，类型为str（指向数据源的csv文件）或者Dataframe或Series
    :return: 带描述符的datasrc数据
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
            # TODO: 保留Mordred的特征名
            X1.append(featurizer.featurize(smiles)[0])
        except RuntimeWarning as e:
            log.error(f"化合物 {smiles} 计算Mordred描述符出现运行时警告: ")
            log.error(traceback.format_exc())
    # 将list转换为Dataframe
    X = pd.DataFrame(data=X1)
    #X = pd.concat([X, X1], axis=1)

    # 将SMILES与描述符拼接在一起
    if not X.empty:
        # 使用函数clean_desc_dataframe对描述符数据进行清理和处理
        X = clean_desc_dataframe(X, drop_duplicates=drop_duplicates)

        # 使用pd.concat将原始数据框架(df)和描述符数据(X)在列方向(axis=1)上进行拼接
        df = pd.concat([df, X], axis=1)

        # 返回包含描述符的新数据框架(df)
        return df
    else:
        # 如果描述符数据为空，则抛出值错误(ValueError)
        raise ValueError("Empty dataframe")

def get_X_y_smiles(csv_file, smile_col=2, label_col=0, desc_start_col=3):
    """
    读取只有一列回归数据的csv文件，默认第一二列为药物index和SMILES，第三列为数据标签，其后为数据特征
    :param csv_file: 只有一列回归数据的csv文件
    :param smile_col: SMILES所在列号
    :param label_col: 数据标签所在列号
    :param desc_start_col: 数据特征起始列号
    :return: 完成清洗的数据特征X, 回归数据y，SMILES
    """
    df = pd.read_csv(csv_file)
    df = clean_desc_dataframe(df)
    X = df.iloc[:, desc_start_col:]
    y = df.iloc[:, label_col]
    smiles = df.iloc[:, smile_col]
    return X, y, smiles


def split_null_from_data(df: pd.DataFrame):
    """
    将数据中的空数据与其他数据分开成两份dataframe

    :param df: 包含空数据的源dataframe
    :return: 含数据的dataframe以及含空数据的dataframe
    """
    data_df = df.dropna(axis=0)
    empty_df = df.drop(index=data_df.index)
    return data_df.reset_index(drop=True), empty_df.reset_index(drop=True)


def clean_desc_dataframe(df: pd.DataFrame, axis=1, drop_duplicates=True) -> pd.DataFrame:
    """
    清除描述符dataframe中的无效数据，避免发生报错

    :param df: 包含无效数据的Dataframe
    :param axis: axis为1时清除掉包含无效数据的列（默认），0时清除行
    :param drop_duplicates: 是否丢弃重复的行
    :return: 完成清除的Dataframe
    """
    log.info("执行特征Dataframe预处理")
    df.replace(["#NAME?", np.inf, -np.inf], np.nan, inplace=True)
    df.replace("#NUM!", np.nan, inplace=True)
    df = df.dropna(axis=axis)
    if drop_duplicates:
        df = df.drop_duplicates()
    return df
