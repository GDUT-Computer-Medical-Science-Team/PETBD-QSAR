import traceback

import pandas

import pandas as pd
import numpy as np
from deepchem import feat
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from tqdm import tqdm
from utils.DataLogger import DataLogger
"""
    [Chinese text removed]
"""

log = DataLogger().getlog("data_preprocess_utils")


def save_organ_data_by_names(root_filepath: str, target_filepath: str, organ_names: list, is_sd=False):
    """
    [Chinese text removed]root_filepath[Chinese text removed]，[Chinese text removed]csv[Chinese text removed]target_filepath[Chinese text removed]

    :param root_filepath: [Chinese text removed]，[Chinese text removed]compound index[Chinese text removed]SMILES
    :param target_filepath: [Chinese text removed]csv[Chinese text removed]
    :param is_sd: [Chinese text removed](sd)，[Chinese text removed]False，[Chinese text removed](mean)
    :param organ_names: [Chinese text removed]
    """
    if root_filepath is None or len(root_filepath) == 0 or target_filepath is None or len(target_filepath) == 0:
        raise ValueError("[Chinese text removed]Error")
    if organ_names is None or len(organ_names) == 0:
        raise ValueError("[Chinese text removed]")
    # [Chinese text removed]dataframe
    if root_filepath.endswith('xlsx') or root_filepath.endswith('xls'):
        raw_df = pd.read_excel(root_filepath, index_col=[0, 1], engine='openpyxl')
    elif root_filepath.endswith('csv'):
        raw_df = pd.read_csv(root_filepath, index_col=[0, 1])
    else:
        raise TypeError("[Chinese text removed]excel[Chinese text removed](xlsx, xls, csv)")
    df_list = []
    log.info(f"[Chinese text removed]，[Chinese text removed]:{len(organ_names)}")
    for organ_name in tqdm(organ_names, desc="[Chinese text removed]: "):
        if not is_sd:
            df = raw_df.loc[:, raw_df.columns.str.startswith(f'{organ_name} mean')]
        else:
            df = raw_df.loc[:, raw_df.columns.str.startswith(f'{organ_name} sd')]
        df_list.append(df)
    df = pd.concat(df_list, axis=1)
    df.to_csv(target_filepath, encoding='utf-8')
    log.info(f"[Chinese text removed]Success，[Chinese text removed]csv[Chinese text removed]{target_filepath}")


def get_certain_time_organ_data(root_filepath: str, organ_name: str, certain_time: int, is_sd=False) \
        -> pandas.DataFrame:
    """
    [Chinese text removed]

    :param root_filepath: [Chinese text removed]
    :param organ_name: [Chinese text removed]
    :param certain_time: [Chinese text removed]
    :param is_sd: [Chinese text removed](sd)，[Chinese text removed]False，[Chinese text removed](mean)
    :return: [Chinese text removed]dataframe
    """
    if root_filepath is None or len(root_filepath) == 0:
        raise ValueError("parameterroot_filepathError")
    if root_filepath.endswith('xlsx') or root_filepath.endswith('xls'):
        raw_df = pd.read_excel(root_filepath, index_col=[0, 1], engine='openpyxl')
    elif root_filepath.endswith('csv'):
        raw_df = pd.read_csv(root_filepath, index_col=[0, 1])
    else:
        raise TypeError("parameterroot_filepathError，[Chinese text removed]excel[Chinese text removed](xlsx, xls, csv)")
    if organ_name is None or len(organ_name) == 0:
        raise ValueError("parameterorgan_nameError")
    if is_sd:
        organ_df = raw_df.loc[:, raw_df.columns.str.startswith(f'{organ_name.lower()} sd{certain_time}min')]
    else:
        organ_df = raw_df.loc[:, raw_df.columns.str.startswith(f'{organ_name.lower()} mean{certain_time}min')]
    return organ_df


def save_max_organ_data(root_filepath: str, target_filepath: str, organ_name: str):
    """
    [Chinese text removed]，[Chinese text removed]csv[Chinese text removed]

    :param root_filepath: [Chinese text removed]
    :param target_filepath: [Chinese text removed]csv[Chinese text removed]
    :param organ_name: [Chinese text removed]
    """
    # [Chinese text removed]
    if root_filepath.endswith('xlsx') or root_filepath.endswith('xls'):
        raw_df = pd.read_excel(root_filepath, index_col=[0, 1], engine='openpyxl')
    elif root_filepath.endswith('csv'):
        raw_df = pd.read_csv(root_filepath, index_col=[0, 1])
    else:
        raise TypeError("parameterroot_filepathError，[Chinese text removed]excel[Chinese text removed](xlsx, xls, csv)")
    organ_df = raw_df.loc[:, raw_df.columns.str.startswith(f'{organ_name.lower()} mean')]
    # [Chinese text removed]
    max_concentration2time = dict()

    # [Chinese text removed]（index[Chinese text removed]SMILES）
    for index, row_data in organ_df.iterrows():
        # [Chinese text removed]
        row_data = row_data.dropna()
        if row_data.empty:
            continue
        else:
            # [Chinese text removed]
            num2time = dict()
            # ConvertingSeries[Chinese text removed]Dataframe
            row_data = row_data.to_frame()
            row_data = pd.DataFrame(row_data.values.T, columns=row_data.index)
            for column in row_data.columns.to_list():
                concentration_num = float(row_data[column].values[0])
                # [Chinese text removed]
                num2time[column.split(" ")[1].replace('mean', '')] = concentration_num
        sorted_data = sorted(num2time.items(), key=lambda kv: (kv[1], kv[0]), reverse=True)
        # [Chinese text removed]
        max_concentration2time[index] = sorted_data[0]
    # [Chinese text removed]Converting[Chinese text removed]Dataframe[Chinese text removed]
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


# TODO: [Chinese text removed]PadelpyCall[Chinese text removed]
def calculate_Mordred_desc(datasrc, drop_duplicates=False):
    """
    [Chinese text removed]datasrc[Chinese text removed]SMILES[Chinese text removed]return[Chinese text removed]datasrc[Chinese text removed]
    :param datasrc: [Chinese text removed]SMILES[Chinese text removed]，[Chinese text removed]str（[Chinese text removed]csv[Chinese text removed]）[Chinese text removed]Dataframe[Chinese text removed]Series
    :return: [Chinese text removed]datasrc[Chinese text removed]
    """
    if isinstance(datasrc, str):
        df = pd.read_csv(datasrc)
    elif isinstance(datasrc, pd.DataFrame) or isinstance(datasrc, pd.Series):
        df = datasrc
    else:
        raise ValueError("Error[Chinese text removed]datasrc[Chinese text removed](str, DataFrame, Series)")

    if isinstance(datasrc, str) or isinstance(datasrc, pd.DataFrame):
        SMILES = df['SMILES']
    else:
        SMILES = df

    featurizer = feat.MordredDescriptors(ignore_3D=True)
    X1 = []
    for smiles in tqdm(SMILES, desc="[Chinese text removed]Mordred descriptors: "):
        try:
            # TODO: [Chinese text removed]Mordred[Chinese text removed]features[Chinese text removed]
            X1.append(featurizer.featurize(smiles)[0])
        except RuntimeWarning as e:
            log.error(f"[Chinese text removed] {smiles} [Chinese text removed]Mordred descriptors[Chinese text removed]Warning: ")
            log.error(traceback.format_exc())
    # [Chinese text removed]listConverting[Chinese text removed]Dataframe
    X = pd.DataFrame(data=X1)
    #X = pd.concat([X, X1], axis=1)

    # [Chinese text removed]SMILES[Chinese text removed]
    if not X.empty:
        # [Chinese text removed]clean_desc_dataframe[Chinese text removed]
        X = clean_desc_dataframe(X, drop_duplicates=drop_duplicates)

        # [Chinese text removed]pd.concat[Chinese text removed]original[Chinese text removed](df)[Chinese text removed](X)[Chinese text removed](axis=1)[Chinese text removed]
        df = pd.concat([df, X], axis=1)

        # return[Chinese text removed](df)
        return df
    else:
        # [Chinese text removed]Data is empty，[Chinese text removed]Error(ValueError)
        raise ValueError("Empty dataframe")

def get_X_y_smiles(csv_file, smile_col=2, label_col=0, desc_start_col=3):
    """
    [Chinese text removed]csv[Chinese text removed]，[Chinese text removed]index[Chinese text removed]SMILES，[Chinese text removed]，[Chinese text removed]features
    :param csv_file: [Chinese text removed]csv[Chinese text removed]
    :param smile_col: SMILES[Chinese text removed]
    :param label_col: [Chinese text removed]
    :param desc_start_col: [Chinese text removed]features[Chinese text removed]
    :return: Complete[Chinese text removed]featuresX, [Chinese text removed]y，SMILES
    """
    df = pd.read_csv(csv_file)
    df = clean_desc_dataframe(df)
    X = df.iloc[:, desc_start_col:]
    y = df.iloc[:, label_col]
    smiles = df.iloc[:, smile_col]
    return X, y, smiles


def split_null_from_data(df: pd.DataFrame):
    """
    [Chinese text removed]dataframe

    :param df: [Chinese text removed]dataframe
    :return: [Chinese text removed]dataframe[Chinese text removed]dataframe
    """
    data_df = df.dropna(axis=0)
    empty_df = df.drop(index=data_df.index)
    return data_df.reset_index(drop=True), empty_df.reset_index(drop=True)


def clean_desc_dataframe(df: pd.DataFrame, axis=1, drop_duplicates=True) -> pd.DataFrame:
    """
    [Chinese text removed]dataframe[Chinese text removed]，[Chinese text removed]

    :param df: [Chinese text removed]Dataframe
    :param axis: axis[Chinese text removed]1[Chinese text removed]（[Chinese text removed]），0[Chinese text removed]
    :param drop_duplicates: [Chinese text removed]
    :return: Complete[Chinese text removed]Dataframe
    """
    log.info("[Chinese text removed]featuresDataframe[Chinese text removed]")
    df.replace(["#NAME?", np.inf, -np.inf], np.nan, inplace=True)
    df.replace("#NUM!", np.nan, inplace=True)
    df = df.dropna(axis=axis)
    if drop_duplicates:
        df = df.drop_duplicates()
    return df
