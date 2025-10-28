import os

import numpy as np
import pandas as pd
from learn2learn.data import MetaDataset
from torch.utils.data import ConcatDataset, DataLoader
from utils.DataLogger import DataLogger

logger = DataLogger().getlog('datasets_loader')

"""
    [Chinese text removed]
"""


def read_tensor_datasets(base_dir, device):
    """
    [Chinese text removed]returnbase_dir[Chinese text removed].pt[Chinese text removed]tensor datasets
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
    [Chinese text removed]，[Chinese text removed]
    :param train_datasets_dir: [Chinese text removed]TensorDataset[Chinese text removed]
    :param target_organ: [Chinese text removed]，[Chinese text removed]dataset[Chinese text removed]，[Chinese text removed]
    :param support_batch_size: [Chinese text removed]batch[Chinese text removed]
    :param query_batch_size: [Chinese text removed]batch[Chinese text removed]
    :param device:
    :return: [Chinese text removed]
    """
    logger.info("[Chinese text removed]")
    torchDatasets = read_tensor_datasets(base_dir=train_datasets_dir, device=device)
    queryset = torchDatasets.pop(target_organ)
    supportset = torchDatasets
    meta_queryset = MetaDataset(queryset)
    meta_supportset = MetaDataset(ConcatDataset(supportset.values()))

    logger.info(f"[Chinese text removed] {target_organ} [Chinese text removed]")

    # # [Chinese text removed]
    # if external_validation:
    #     # [Chinese text removed]TensorDataset
    #     md.transform_organ_time_data_to_tensor_dataset(desc_file='merged_FP.csv',
    #                                                    concentration_csv_file="OrganDataAt60min.csv",
    #                                                    external=True)
    #     # [Chinese text removed]
    #     if target_organ == 'blood':
    #         externalset = torch.load("./ExtenalDatasets/blood_12_dataset.pt", map_location=device)
    #     elif target_organ == 'brain':
    #         externalset = torch.load("./ExtenalDatasets/brain_9_dataset.pt", map_location=device)
    #     else:
    #         raise ValueError("[Chinese text removed]")
    #     meta_externalset = MetaDataset(externalset)

    query_dataloader = DataLoader(meta_queryset, batch_size=query_batch_size, shuffle=False)
    support_dataloader = DataLoader(meta_supportset, batch_size=support_batch_size, shuffle=True)
    # if external_validation:
    #     external_dataloader = DataLoader(meta_externalset, batch_size=1, shuffle=True)

    return support_dataloader, query_dataloader


def get_test_datasets(test_datasets_dir, target_organ, batch_size, device):
    """
    [Chinese text removed]TensorDataset
    :param test_datasets_dir: [Chinese text removed]TensorDataset[Chinese text removed]
    :param target_organ: [Chinese text removed]，[Chinese text removed]pt[Chinese text removed]
    :param batch_size: [Chinese text removed]batch[Chinese text removed]
    :return: [Chinese text removed]
    """
    logger.info("[Chinese text removed]")
    torchDatasets = read_tensor_datasets(base_dir=test_datasets_dir, device=device)
    queryset = torchDatasets.pop(target_organ)
    # supportset = torchDatasets
    meta_queryset = MetaDataset(queryset)
    # meta_supportset = MetaDataset(ConcatDataset(supportset.values()))

    logger.info(f"[Chinese text removed] {target_organ}")

    query_dataloader = DataLoader(meta_queryset, batch_size=batch_size, shuffle=False)

    return query_dataloader

def get_sklearn_data(npy_filename:str, organ_name:str):
    """
    [Chinese text removed]sklearn[Chinese text removed]
    :return:
    """
    if npy_filename is None or organ_name is None:
        raise ValueError("parameterError")
    if not os.path.isfile(npy_filename):
        raise FileNotFoundError(f"{npy_filename}[Chinese text removed]")
    train_data = np.load(npy_filename, allow_pickle=True).item()
    # [Chinese text removed]：[Chinese text removed]train_data[Chinese text removed]
    print("[Chinese text removed]：", train_data.keys())

    if organ_name not in train_data:
        raise KeyError(f"'{organ_name}' [Chinese text removed]。[Chinese text removed]：{list(train_data.keys())}")

    data = train_data[organ_name]
    X, y, smiles = get_X_y_SMILES(data)
    return X, y

def get_X_y_SMILES(data):
    if not isinstance(data, pd.DataFrame):
        raise TypeError("[Chinese text removed]Dataframe[Chinese text removed]")
    y = data['Concentration'].ravel()
    smiles = data['SMILES']
    X = data.drop(['SMILES', 'Concentration'], axis=1)
    return X, y, smiles