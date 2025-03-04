import os.path

from torch.utils.data import TensorDataset
from preprocess.data_preprocess.FeatureExtraction import FeatureExtraction
from preprocess.data_preprocess.PadelpyCall import PadelpyCall
from preprocess.data_preprocess.data_preprocess_utils import *
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from utils.DataLogger import DataLogger
import pandas as pd
import numpy as np
import torch
import os
import traceback

log = DataLogger().getlog("MedicalDatasetsHandler")


class MedicalDatasetsHandler:
    def __init__(self):
        """
        用于读取原始的整合浓度数据并处理成可输入到模型的Dataset

        """
        self.__merged_filepath = None
        self.__organ_time_data_filepath = None
        self.__saving_folder = None

        self.__device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.desc_num = 0
        self.__feature_select_number = 50
        self.__output_filename = None


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

    def __transform_organs_data(self,
                                desc_file='descriptors.csv',
                                FP=False,
                                double_index=True,
                                overwrite=False) -> str:
        """
        读取保存全部器官浓度数据的csv文件，根据化合物SMILES计算数据特征，筛选数据特征
        以器官名为key、浓度数据为value包装到字典中，并保存到npy文件中
        :param desc_file: 存储数据特征的csv文件名
        :param FP: 是否启动分子指纹处理
        :param double_index: 是否读取双倍的特征索引
        :param overwrite: 是否覆盖已有的npy文件
        :return: 保存所有器官及其df的总字典文件路径
        """
        # 路径初始化
        npy_file = os.path.join(self.__saving_folder, 'organ_df.npy')
        desc_file = os.path.join(self.__saving_folder, desc_file)
        log.info("读取保存每种器官的特征及标签的npy文件")

        if overwrite or not os.path.exists(npy_file):
            log.info("npy文件未找到或overwrite参数设置为True，读取筛选后的器官数据并进行数据预处理操作")
            # 读取浓度数据，并获取分子描述符
            df = pd.read_csv(self.__organ_time_data_filepath)
            # df = clean_desc_dataframe(df)
            smiles = pd.DataFrame({'SMILES': df.iloc[:, 1]})
            # 不存在保存数据特征的文件，进行特征生成
            if overwrite or not os.path.exists(desc_file):
                log.info("未找到特征文件，进行特征生成操作")
                # 计算SMILES的描述符，然后保存到mol_Desc文件中方便再次读取
                if FP:  # 计算分子指纹
                    finger_prints = ['EState', 'MACCS', 'KlekotaRoth', 'PubChem']
                    log.info(f"生成分子指纹: {finger_prints}")
                    smiles_smi = os.path.join(self.__saving_folder, "smiles.smi")
                    with open(smiles_smi, 'w', encoding='utf-8') as f:
                        for smile in smiles.values.tolist():
                            f.write(smile[0])
                            f.write("\n")
                    fp_xml_dir = "preprocess/data_preprocess/fingerprints_xml/*.xml"
                    pc = PadelpyCall(save_dir=self.__saving_folder, smi_filename=smiles_smi, fp_xml_dir=fp_xml_dir)
                    mol_Desc = pc.CalculateFP(finger_prints, overwrite=False)
                else:  # 计算分子描述符
                    log.info("生成Mordred分子描述符")
                    mol_Desc = calculate_Mordred_desc(smiles)
                mol_Desc.to_csv(desc_file, index=False, encoding='utf-8')
                log.info(f"特征生成完成，以csv格式存储至{desc_file}")
            # 存在保存数据特征的文件，直接读取
            else:
                log.info(f"存在特征文件{desc_file}，读取数据特征")
                mol_Desc = pd.read_csv(desc_file)
            log.info("正在执行特征数据归一化处理")
            # 读取纯特征部分(Mordred描述符)
            if not FP:
                mol_Desc = mol_Desc.iloc[:, 1:]
            # 预处理数据集
            # sc = StandardScaler()
            sc = MinMaxScaler()
            mol_Desc = pd.DataFrame(sc.fit_transform(mol_Desc), columns=mol_Desc.columns)
            mol_Desc = clean_desc_dataframe(mol_Desc, drop_duplicates=False)
            organs_labels = df.iloc[:, 2:]

            # 保存所有器官的描述符以及浓度数据的总字典
            datasets = {}
            log.info("进行特征筛选")
            # 处理每一种器官的浓度数据
            for index, col in tqdm(organs_labels.iteritems(), desc="正在筛选特征: ", total=organs_labels.shape[1]):
                organ_name = index.split()[0]
                concentration_data = pd.DataFrame({'Concentration': col})
                # 去除异常值，将最多占数据集4%的异常值置为0
                from sklearn.covariance import EllipticEnvelope
                try:
                    predictions1 = EllipticEnvelope(contamination=0.04, support_fraction=1) \
                        .fit_predict(concentration_data.fillna(value=0))
                    predictions1 = (predictions1 == 1)
                    concentration_data.loc[~predictions1] = np.nan
                except ValueError:
                    log.error(f"器官{organ_name}的标签去异常值存在问题，取消去除异常操作")
                    log.error(traceback.format_exc())

                """
                    进行特征筛选，获得50个或者100个特征
                """
                if not double_index:
                    x = FeatureExtraction(mol_Desc,
                                          concentration_data.fillna(value=0),
                                          RFE_features_to_select=self.__feature_select_number)\
                        .feature_extraction(TBE=True, returnIndex=False)
                else:
                    x = FeatureExtraction(mol_Desc,
                                          concentration_data.fillna(value=0),
                                          RFE_features_to_select=self.__feature_select_number * 2) \
                        .feature_extraction(TBE=True, returnIndex=False)
                # 合并SMILES、浓度数据和筛选完成的特征
                organ_df = pd.concat([smiles, concentration_data, x], axis=1)
                # 根据浓度数据列的空数据抛弃行数据
                organ_df = organ_df.dropna(subset=['Concentration'])
                organ_df.reset_index(inplace=True, drop=True)
                # 按照器官名添加到总字典中
                datasets[organ_name] = organ_df
            # 保存字典
            np.save(npy_file, datasets)
            log.info(f"全部特征筛选成功，已将数据以npy格式保存至{npy_file}")
            return npy_file
        # 总字典存在，直接读取
        else:
            log.info("npy文件存在")
            # datasets = np.load(npy_file, allow_pickle=True).item()
            return npy_file
        # self.save_df2TensorDataset(datasets)
        # log.info("已获取各器官的筛选后特征数据及数据标签")
        # return datasets

    def __split_df2TensorDataset(self, npy_file_path: str, test_size=0.2, overwrite=False):
        """
        将字典内的器官数据分别转换成对应的TensorDataset，并保存到saving_folder中
        :param npy_file_path: 保存器官与df数据的字典文件路径
        """
        """
        1. 读取npy文件，分割成train和test后创建对应的目录并保存对应的npy
        2. 分别读取对应的npy文件，转换数据为TensorDataset并保存在train和test的datasets目录里
        """
        if npy_file_path is None or len(npy_file_path) == 0:
            raise ValueError("参数npy_file_path错误")
        if not os.path.exists(npy_file_path):
            raise FileNotFoundError(f"文件 {npy_file_path} 不存在")
        if test_size < 0.0 or test_size > 1.0:
            raise ValueError("参数test_size超过范围[0.0, 1.0)")

        # 保存数据的次级目录
        train_dir = os.path.join(self.__saving_folder, 'train')
        if not os.path.exists(train_dir):
            os.mkdir(train_dir)
        train_datasets_dir = os.path.join(train_dir, 'datasets')
        if not os.path.exists(train_datasets_dir):
            os.mkdir(train_datasets_dir)

        test_dir = os.path.join(self.__saving_folder, 'test')
        if not os.path.exists(test_dir):
            os.mkdir(test_dir)
        test_datasets_dir = os.path.join(test_dir, 'datasets')
        if not os.path.exists(test_datasets_dir):
            os.mkdir(test_datasets_dir)
        # 保存分割后的数据字典
        train_data_dict = dict()
        test_data_dict = dict()


        log.info("读取器官数据总字典npy文件")
        main_df_map = np.load(npy_file_path, allow_pickle=True).item()
        # 将每个器官都按照test_size分割，并存储到对应的字典中等待保存及处理
        for organ_name, df in tqdm(main_df_map.items(), desc="正在将器官数据进行Train Test切分: "):
            train, test = train_test_split(df, test_size=test_size)
            train_data_dict[organ_name] = train
            test_data_dict[organ_name] = test
        log.info("切分完成，将数据保存为对应的npy文件")
        # 保存到npy文件中
        train_data_npy = os.path.join(train_dir, 'train_organ_df.npy')
        test_data_npy = os.path.join(test_dir, 'test_organ_df.npy')
        np.save(train_data_npy, train_data_dict)
        np.save(test_data_npy, test_data_dict)
        log.info("保存完成")
        # 读取npy文件，转换成TensorDataset
        self.__df2TensorDataset(train_data_npy, train_datasets_dir, overwrite=overwrite)
        self.__df2TensorDataset(test_data_npy, test_datasets_dir, overwrite=overwrite)

    def __df2TensorDataset(self, npy_file: str, torch_datasets_dir: str, overwrite=False):

        """
        1. 读取npy文件数据
        2. 遍历数据，存储到对应目录中
        """
        # 删除目录下的所有.pt文件
        if overwrite:
            log.info("启用覆盖模式，删除目录内的pt文件")
            for file in os.listdir(torch_datasets_dir):
                if file.endswith('.pt'):
                    os.remove(file)
        log.info("读取对应的npy文件并转换数据为TensorDataset格式")
        df_map = np.load(npy_file, allow_pickle=True).item()
        # 遍历每个器官的数据，分离出特征x与标签y，保存为TensorDataset
        for organ_name, df in tqdm(df_map.items(), desc="正在将器官数据转换为TensorDataset格式: "):
            try:
                # df = DataPreprocess.clean_desc_dataframe(df)
                x = df.iloc[:, 2:]
                y = df['Concentration']
                if x.shape[0] != y.shape[0]:
                    raise ValueError("x and y having different counts")
                count = y.shape[0]
                x = torch.tensor(x.values).to(self.__device)
                y = torch.tensor(y.values).resize_(count, 1).to(self.__device)
                dataset = TensorDataset(x, y)
                torch.save(dataset, os.path.join(torch_datasets_dir, f'{organ_name}_{count}_dataset.pt'))
            except Exception as e:
                log.error(f"转换器官 {organ_name} 的数据时出现以下错误: ")
                log.error(traceback.format_exc())
        log.info("全部数据已成功转换为TensorDataset格式")

    # def get_single_organ_tensor(self, test_size=0.1):
    #     x, y, _ = get_X_y_smiles(self.__organ_time_data_filepath)
    #     sc = StandardScaler()
    #     x = pd.DataFrame(sc.fit_transform(x))
    #
    #     x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size)
    #     sample_num, self.desc_num = x.shape[0], x.shape[1]
    #
    #     # Prepare your data as PyTorch tensors
    #     x_train, y_train = torch.Tensor(x_train.values).to(self.__device), \
    #         torch.Tensor(y_train.values).resize_(y_train.shape[0], 1).to(self.__device)
    #     x_test, y_test = torch.Tensor(x_test.values).to(self.__device), \
    #         torch.Tensor(y_test.values).resize_(y_test.shape[0], 1).to(self.__device)
    #
    #     # Create PyTorch datasets
    #     train_dataset = TensorDataset(x_train, y_train)
    #     test_dataset = TensorDataset(x_test, y_test)
    #
    #     return train_dataset, test_dataset
