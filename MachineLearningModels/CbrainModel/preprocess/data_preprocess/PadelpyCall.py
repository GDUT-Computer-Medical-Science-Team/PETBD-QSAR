import os.path
import glob
import traceback
import pandas as pd
from tqdm import tqdm
from utils.DataLogger import DataLogger
from padelpy import padeldescriptor

log = DataLogger().getlog('Padel')


class PadelpyCall:
    def __init__(self, save_dir, smi_filename: str = None,
                 fp_xml_dir: str = "./fingerprints_xml/*.xml", merge_csv: str = 'FP_descriptors.csv'):
        """
        调用Padelpy进行分子指纹计算

        :param smi_filename: 记录待计算分子的SMILES的smi文件
        :param save_dir: 保存计算结果的目录
        :param merge_csv: 保存的合并文件名
        """
        self.smi_dir = smi_filename
        self.save_dir = save_dir
        self.merge_csv = merge_csv
        try:
            if not os.path.exists(self.save_dir):
                os.mkdir(self.save_dir)
        except Exception:
            log.error(traceback.format_exc())
        self.index_name = 'SMILES'
        # self.xml_files = glob.glob("./fingerprints_xml/*.xml")  # 绝对路径会导致报错
        self.xml_files = glob.glob(fp_xml_dir)
        self.xml_files.sort()
        self.FP_list = ['AtomPairs2DCount',
                        'AtomPairs2D',
                        'EState',
                        'CDKextended',
                        'CDK',
                        'CDKgraphonly',
                        'KlekotaRothCount',
                        'KlekotaRoth',
                        'MACCS',
                        'PubChem',
                        'SubstructureCount',
                        'Substructure']
        self.fp = dict(zip(self.FP_list, self.xml_files))
        self.target_fingerprints = None

    def CalculateFP(self, fingerprints, overwrite=False):
        """
        遍历计算分子指纹，分别输出对应分子指纹的文件
        :param fingerprints: 需要计算的分子指纹类型，例如：['EState', 'MACCS', 'KlekotaRoth', 'PubChem']
        :return:
        """
        if self.smi_dir is not None:
            # 获取SMILES，等待后续使用
            with open(self.smi_dir, "r") as file:
                lines = file.readlines()
            # 去除每行末尾的换行符，并创建一个包含每行内容的SMILES列表
            SMILES = [line.rstrip("\n") for line in lines]
        else:
            raise ValueError("参数smi_dir为None，参数错误")
        if fingerprints is None or len(fingerprints) == 0:
            raise ValueError("参数fingerprints错误")
        self.target_fingerprints = fingerprints
        log.info(f"开始计算分子指纹")
        for fingerprint in tqdm(fingerprints, desc="分子指纹计算进度: "):
            if fingerprint not in self.FP_list:
                log.error(f"错误: {fingerprint} 不是合法指纹类型")
                continue
            # fingerprint = 'Substructure'
            fingerprint_output_file = os.path.join(self.save_dir, ''.join([fingerprint, '.csv']))
            fingerprint_descriptor_types = self.fp[fingerprint]  # 解析文件地址
            if os.path.exists(fingerprint_output_file):
                if overwrite:
                    os.remove(fingerprint_output_file)
                else:
                    continue
            padeldescriptor(mol_dir=self.smi_dir,
                            d_file=fingerprint_output_file,  # ex: 'Substructure.csv'
                            descriptortypes=fingerprint_descriptor_types,
                            detectaromaticity=True,
                            standardizenitro=True,
                            standardizetautomers=True,
                            threads=2,  # 增加该数值会报错
                            removesalt=True,
                            log=False,
                            fingerprints=True)
        # 将输出的csv文件中的SMILES由AUTOGEN_result_{}转换为正常输入的SMILES
            df = pd.read_csv(fingerprint_output_file)
            df = df.drop('Name', axis=1)
            df.insert(0, self.index_name, pd.Series(SMILES))
            df.to_csv(fingerprint_output_file, index=False)

        return self.MergeResult()

    def MergeResult(self, remove_existed_csvfiles=True, save_to_file=False):
        """
        用于将分次计算的特征文件合并成一个csv文件保存
        """
        csv_files = list()
        for fp in self.target_fingerprints:
            csv_files.append(os.path.join(self.save_dir, fp + ".csv"))
        # 存储每个特征文件的dataframe
        dataframes = []
        for csv_file in csv_files:
            df = pd.read_csv(csv_file, index_col=self.index_name)
            dataframes.append(df)
        log.info("合并分子指纹数据...")
        # 合并所有Dataframe
        merged_df = pd.concat(dataframes, axis=1, sort=False)
        if save_to_file:
            merged_df.to_csv(os.path.join(self.save_dir, self.merge_csv))
        # 删除其他指纹文件，只保留合并文件
        if remove_existed_csvfiles:
            for csv_file in csv_files:
                os.remove(csv_file)
        log.info("合并后的指纹文件维度为: " + str(merged_df.shape))
        return merged_df
