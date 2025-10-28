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
        [Chinese text removed]original[Chinese text removed]Dataset

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
        [Chinese text removed]original[Chinese text removed]，[Chinese text removed]
        :param merged_filepath: [Chinese text removed]
        :param organ_names: [Chinese text removed]
        :param certain_time: [Chinese text removed]，[Chinese text removed]
        :param overwrite: [Chinese text removed]
        :param is_sd: [Chinese text removed](sd)，[Chinese text removed]False，[Chinese text removed](mean)
        """
        self.__merged_filepath = merged_filepath
        # [Chinese text removed]original[Chinese text removed]
        if self.__merged_filepath is not None and len(self.__merged_filepath) > 0:
            self.__saving_folder = os.path.split(self.__merged_filepath)[0]
        else:
            raise ValueError("parametermerged_filepathError")

        organ_data_filepath = os.path.join(self.__saving_folder, "OrganData.csv")
        self.__organ_time_data_filepath = os.path.join(self.__saving_folder, f"OrganDataAt{certain_time}min.csv")
        log.info("[Chinese text removed]original[Chinese text removed]，[Chinese text removed]")

        # [Chinese text removed]
        if overwrite or not os.path.exists(organ_data_filepath):
            log.info(f"[Chinese text removed]: {organ_names} [Chinese text removed]: ")
            save_organ_data_by_names(root_filepath=self.__merged_filepath,
                                     target_filepath=organ_data_filepath,
                                     organ_names=organ_names,
                                     is_sd=is_sd)
        else:
            log.info(f"[Chinese text removed]Complete[Chinese text removed]csv[Chinese text removed]: {organ_data_filepath}，[Chinese text removed]")
        # [Chinese text removed]
        if overwrite or not os.path.exists(self.__organ_time_data_filepath):
            df = pd.DataFrame()
            log.info(f"[Chinese text removed]，[Chinese text removed]: {certain_time}min")
            # [Chinese text removed]
            for organ_name in tqdm(organ_names, desc=f"[Chinese text removed]{certain_time}min[Chinese text removed]: "):
                df = pd.concat([df, get_certain_time_organ_data(root_filepath=organ_data_filepath,
                                                                organ_name=organ_name,
                                                                certain_time=certain_time,
                                                                is_sd=is_sd)],
                               axis=1)
            log.info(f"[Chinese text removed]Complete，[Chinese text removed]csv[Chinese text removed]{self.__organ_time_data_filepath}")
            df.to_csv(self.__organ_time_data_filepath, encoding='utf-8')
        else:
            log.info(f"[Chinese text removed]Complete[Chinese text removed]csv[Chinese text removed]: {self.__organ_time_data_filepath}，[Chinese text removed]")

    def transform_organ_time_data_to_tensor_dataset(self,
                                                    test_size=0.2,
                                                    double_index=False,
                                                    FP=False,
                                                    overwrite=False):
        """
        [Chinese text removed]csv[Chinese text removed]，[Chinese text removed]，[Chinese text removed]、[Chinese text removed]，[Chinese text removed]TensorDataset[Chinese text removed]
        :param test_size: [Chinese text removed]，[Chinese text removed][0.0, 1.0)
        :param double_index: [Chinese text removed]features[Chinese text removed]
        :param FP: [Chinese text removed]
        :param overwrite: [Chinese text removed]npy[Chinese text removed]
        """
        if test_size < 0.0 or test_size >= 1.0:
            raise ValueError("parametertest_size[Chinese text removed][0.0, 1.0)")
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
        [Chinese text removed]csv[Chinese text removed]，[Chinese text removed]SMILES[Chinese text removed]features，[Chinese text removed]features
        [Chinese text removed]key、[Chinese text removed]value[Chinese text removed]，[Chinese text removed]npy[Chinese text removed]
        :param desc_file: [Chinese text removed]features[Chinese text removed]csv[Chinese text removed]
        :param FP: [Chinese text removed]Starting[Chinese text removed]
        :param double_index: [Chinese text removed]features[Chinese text removed]
        :param overwrite: [Chinese text removed]npy[Chinese text removed]
        :return: [Chinese text removed]df[Chinese text removed]
        """
        # [Chinese text removed]
        npy_file = os.path.join(self.__saving_folder, 'organ_df.npy')
        desc_file = os.path.join(self.__saving_folder, desc_file)
        log.info("[Chinese text removed]features[Chinese text removed]npy[Chinese text removed]")

        if overwrite or not os.path.exists(npy_file):
            log.info("npy[Chinese text removed]overwriteparameter[Chinese text removed]True，[Chinese text removed]")
            # [Chinese text removed]，[Chinese text removed]
            df = pd.read_csv(self.__organ_time_data_filepath)
            # df = clean_desc_dataframe(df)
            smiles = pd.DataFrame({'SMILES': df.iloc[:, 1]})
            # [Chinese text removed]features[Chinese text removed]，[Chinese text removed]features[Chinese text removed]
            if overwrite or not os.path.exists(desc_file):
                log.info("[Chinese text removed]features[Chinese text removed]，[Chinese text removed]features[Chinese text removed]")
                # [Chinese text removed]SMILES[Chinese text removed]，[Chinese text removed]mol_Desc[Chinese text removed]
                if FP:  # [Chinese text removed]
                    finger_prints = ['EState', 'MACCS', 'KlekotaRoth', 'PubChem']
                    log.info(f"[Chinese text removed]: {finger_prints}")
                    smiles_smi = os.path.join(self.__saving_folder, "smiles.smi")
                    with open(smiles_smi, 'w', encoding='utf-8') as f:
                        for smile in smiles.values.tolist():
                            f.write(smile[0])
                            f.write("\n")
                    fp_xml_dir = "preprocess/data_preprocess/fingerprints_xml/*.xml"
                    pc = PadelpyCall(save_dir=self.__saving_folder, smi_filename=smiles_smi, fp_xml_dir=fp_xml_dir)
                    mol_Desc = pc.CalculateFP(finger_prints, overwrite=False)
                else:  # [Chinese text removed]
                    log.info("[Chinese text removed]Mordred[Chinese text removed]")
                    mol_Desc = calculate_Mordred_desc(smiles)
                mol_Desc.to_csv(desc_file, index=False, encoding='utf-8')
                log.info(f"features[Chinese text removed]Complete，[Chinese text removed]csv[Chinese text removed]{desc_file}")
            # [Chinese text removed]features[Chinese text removed]，[Chinese text removed]
            else:
                log.info(f"[Chinese text removed]features[Chinese text removed]{desc_file}，[Chinese text removed]features")
                mol_Desc = pd.read_csv(desc_file)
            log.info("[Chinese text removed]features[Chinese text removed]")
            # [Chinese text removed]features[Chinese text removed](Mordred descriptors)
            if not FP:
                mol_Desc = mol_Desc.iloc[:, 1:]
            # [Chinese text removed]
            # sc = StandardScaler()
            sc = MinMaxScaler()
            mol_Desc = pd.DataFrame(sc.fit_transform(mol_Desc), columns=mol_Desc.columns)
            mol_Desc = clean_desc_dataframe(mol_Desc, drop_duplicates=False)
            organs_labels = df.iloc[:, 2:]

            # [Chinese text removed]
            datasets = {}
            log.info("[Chinese text removed]features[Chinese text removed]")
            # [Chinese text removed]
            for index, col in tqdm(organs_labels.iteritems(), desc="[Chinese text removed]features: ", total=organs_labels.shape[1]):
                organ_name = index.split()[0]
                concentration_data = pd.DataFrame({'Concentration': col})
                # [Chinese text removed]，[Chinese text removed]4%[Chinese text removed]0
                from sklearn.covariance import EllipticEnvelope
                try:
                    predictions1 = EllipticEnvelope(contamination=0.04, support_fraction=1) \
                        .fit_predict(concentration_data.fillna(value=0))
                    predictions1 = (predictions1 == 1)
                    concentration_data.loc[~predictions1] = np.nan
                except ValueError:
                    log.error(f"[Chinese text removed]{organ_name}[Chinese text removed]，[Chinese text removed]")
                    log.error(traceback.format_exc())

                """
                    [Chinese text removed]features[Chinese text removed]，[Chinese text removed]50[Chinese text removed]100[Chinese text removed]features
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
                # [Chinese text removed]SMILES、[Chinese text removed]Complete[Chinese text removed]features
                organ_df = pd.concat([smiles, concentration_data, x], axis=1)
                # [Chinese text removed]
                organ_df = organ_df.dropna(subset=['Concentration'])
                organ_df.reset_index(inplace=True, drop=True)
                # [Chinese text removed]
                datasets[organ_name] = organ_df
            # [Chinese text removed]
            np.save(npy_file, datasets)
            log.info(f"[Chinese text removed]features[Chinese text removed]Success，[Chinese text removed]npy[Chinese text removed]{npy_file}")
            return npy_file
        # [Chinese text removed]，[Chinese text removed]
        else:
            log.info("npy[Chinese text removed]")
            # datasets = np.load(npy_file, allow_pickle=True).item()
            return npy_file
        # self.save_df2TensorDataset(datasets)
        # log.info("[Chinese text removed]features[Chinese text removed]")
        # return datasets

    def __split_df2TensorDataset(self, npy_file_path: str, test_size=0.2, overwrite=False):
        """
        [Chinese text removed]Converting[Chinese text removed]TensorDataset，[Chinese text removed]saving_folder[Chinese text removed]
        :param npy_file_path: [Chinese text removed]df[Chinese text removed]
        """
        """
        1. [Chinese text removed]npy[Chinese text removed]，[Chinese text removed]train[Chinese text removed]test[Chinese text removed]npy
        2. [Chinese text removed]npy[Chinese text removed]，Converting[Chinese text removed]TensorDataset[Chinese text removed]train[Chinese text removed]test[Chinese text removed]datasets[Chinese text removed]
        """
        if npy_file_path is None or len(npy_file_path) == 0:
            raise ValueError("parameternpy_file_pathError")
        if not os.path.exists(npy_file_path):
            raise FileNotFoundError(f"[Chinese text removed] {npy_file_path} [Chinese text removed]")
        if test_size < 0.0 or test_size > 1.0:
            raise ValueError("parametertest_size[Chinese text removed][0.0, 1.0)")

        # [Chinese text removed]
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
        # [Chinese text removed]
        train_data_dict = dict()
        test_data_dict = dict()


        log.info("[Chinese text removed]npy[Chinese text removed]")
        main_df_map = np.load(npy_file_path, allow_pickle=True).item()
        # [Chinese text removed]test_size[Chinese text removed]，[Chinese text removed]
        for organ_name, df in tqdm(main_df_map.items(), desc="[Chinese text removed]Train Test[Chinese text removed]: "):
            train, test = train_test_split(df, test_size=test_size)
            train_data_dict[organ_name] = train
            test_data_dict[organ_name] = test
        log.info("[Chinese text removed]Complete，[Chinese text removed]npy[Chinese text removed]")
        # [Chinese text removed]npy[Chinese text removed]
        train_data_npy = os.path.join(train_dir, 'train_organ_df.npy')
        test_data_npy = os.path.join(test_dir, 'test_organ_df.npy')
        np.save(train_data_npy, train_data_dict)
        np.save(test_data_npy, test_data_dict)
        log.info("[Chinese text removed]Complete")
        # [Chinese text removed]npy[Chinese text removed]，Converting[Chinese text removed]TensorDataset
        self.__df2TensorDataset(train_data_npy, train_datasets_dir, overwrite=overwrite)
        self.__df2TensorDataset(test_data_npy, test_datasets_dir, overwrite=overwrite)

    def __df2TensorDataset(self, npy_file: str, torch_datasets_dir: str, overwrite=False):

        """
        1. [Chinese text removed]npy[Chinese text removed]
        2. [Chinese text removed]，[Chinese text removed]
        """
        # [Chinese text removed].pt[Chinese text removed]
        if overwrite:
            log.info("[Chinese text removed]，[Chinese text removed]pt[Chinese text removed]")
            for file in os.listdir(torch_datasets_dir):
                if file.endswith('.pt'):
                    os.remove(file)
        log.info("[Chinese text removed]npy[Chinese text removed]Converting[Chinese text removed]TensorDataset[Chinese text removed]")
        df_map = np.load(npy_file, allow_pickle=True).item()
        # [Chinese text removed]，[Chinese text removed]featuresx[Chinese text removed]y，[Chinese text removed]TensorDataset
        for organ_name, df in tqdm(df_map.items(), desc="[Chinese text removed]Converting[Chinese text removed]TensorDataset[Chinese text removed]: "):
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
                log.error(f"Converting[Chinese text removed] {organ_name} [Chinese text removed]Error: ")
                log.error(traceback.format_exc())
        log.info("[Chinese text removed]SuccessConverting[Chinese text removed]TensorDataset[Chinese text removed]")

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
