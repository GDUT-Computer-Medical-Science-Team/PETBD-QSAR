import os

import numpy as np
import pandas as pd
from learn2learn.data import MetaDataset
from torch.utils.data import ConcatDataset, DataLoader
from utils.DataLogger import DataLogger

logger = DataLogger().getlog('datasets_loader')

"""
    加载数据集的工具类
"""


def read_tensor_datasets(base_dir, device):
    """
    读取并返回base_dir目录下的所有.pt后缀的tensor datasets
    :param base_dir:
    :param device:
    :return:
    """
    map = {}
    for path in os.listdir(base_dir):
        if path.endswith(".pt"):
            name = path.split("_")[0]
            map[name] = torch.load(os.path.join(base_dir, path), map_location=device)
    return map


def get_train_datasets(train_datasets_dir, target_organ, support_batch_size, query_batch_size, device):
    """
    获取训练集，将数据集处理成支持集与查询集
    :param train_datasets_dir: 保存训练集TensorDataset的目录
    :param target_organ: 目标器官名，将该器官的dataset脱离成查询集，其他的为支持集
    :param support_batch_size: 支持集batch大小
    :param query_batch_size: 查询集batch大小
    :param device:
    :return: 支持集数据装载器与查询集数据装载器
    """
    logger.info("读取训练集数据")
    torchDatasets = read_tensor_datasets(base_dir=train_datasets_dir, device=device)
    queryset = torchDatasets.pop(target_organ)
    supportset = torchDatasets
    meta_queryset = MetaDataset(queryset)
    meta_supportset = MetaDataset(ConcatDataset(supportset.values()))

    logger.info(f"训练集数据选择器官 {target_organ} 作为查询集")

    # # 加入外部验证环节
    # if external_validation:
    #     # 生成外部验证TensorDataset
    #     md.transform_organ_time_data_to_tensor_dataset(desc_file='merged_FP.csv',
    #                                                    concentration_csv_file="OrganDataAt60min.csv",
    #                                                    external=True)
    #     # 读取外部验证集
    #     if target_organ == 'blood':
    #         externalset = torch.load("./ExtenalDatasets/blood_12_dataset.pt", map_location=device)
    #     elif target_organ == 'brain':
    #         externalset = torch.load("./ExtenalDatasets/brain_9_dataset.pt", map_location=device)
    #     else:
    #         raise ValueError("外部数据集未找到")
    #     meta_externalset = MetaDataset(externalset)

    query_dataloader = DataLoader(meta_queryset, batch_size=query_batch_size, shuffle=False)
    support_dataloader = DataLoader(meta_supportset, batch_size=support_batch_size, shuffle=True)
    # if external_validation:
    #     external_dataloader = DataLoader(meta_externalset, batch_size=1, shuffle=True)

    return support_dataloader, query_dataloader


def get_test_datasets(test_datasets_dir, target_organ, batch_size, device):
    """
    读取测试集TensorDataset
    :param test_datasets_dir: 保存测试集TensorDataset的目录
    :param target_organ: 目标器官名，将只选择该器官的pt文件进行读取
    :param batch_size: 测试集batch大小
    :return: 测试集装载器
    """
    logger.info("读取测试集数据")
    torchDatasets = read_tensor_datasets(base_dir=test_datasets_dir, device=device)
    queryset = torchDatasets.pop(target_organ)
    # supportset = torchDatasets
    meta_queryset = MetaDataset(queryset)
    # meta_supportset = MetaDataset(ConcatDataset(supportset.values()))

    logger.info(f"测试集数据选择器官 {target_organ}")

    query_dataloader = DataLoader(meta_queryset, batch_size=batch_size, shuffle=False)

    return query_dataloader

def get_sklearn_data(npy_filename:str, organ_name:str):
    """
    读取sklearn模型输入格式的数据
    :return:
    """
    if npy_filename is None or organ_name is None:
        raise ValueError("参数错误")
    if not os.path.isfile(npy_filename):
        raise FileNotFoundError(f"{npy_filename}文件未找到")
    train_data = np.load(npy_filename, allow_pickle=True).item()
    # 调试：打印train_data中的键
    print("可用键：", train_data.keys())

    if organ_name not in train_data:
        raise KeyError(f"'{organ_name}' 不在数据中。可用键包括：{list(train_data.keys())}")

    data = train_data[organ_name]
    X, y, smiles = get_X_y_SMILES(data)
    return X, y

def get_X_y_SMILES(data):
    if not isinstance(data, pd.DataFrame):
        raise TypeError("需要输入Dataframe类型数据")
    y = data['Concentration'].ravel()
    smiles = data['SMILES']
    X = data.drop(['SMILES', 'Concentration'], axis=1)
    return X, y, smiles