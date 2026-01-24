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
        :return:
        """
        if self.smi_dir is not None:
            with open(self.smi_dir, "r") as file:
                lines = file.readlines()
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
            fingerprint_descriptor_types = self.fp[fingerprint]
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
                            threads=2,
                            removesalt=True,
                            log=False,
                            fingerprints=True)
            df = pd.read_csv(fingerprint_output_file)
            df = df.drop('Name', axis=1)
            df.insert(0, self.index_name, pd.Series(SMILES))
            df.to_csv(fingerprint_output_file, index=False)

        return self.MergeResult()

    def MergeResult(self, remove_existed_csvfiles=True, save_to_file=False):
        """
        """
        csv_files = list()
        for fp in self.target_fingerprints:
            csv_files.append(os.path.join(self.save_dir, fp + ".csv"))
        dataframes = []
        for csv_file in csv_files:
            df = pd.read_csv(csv_file, index_col=self.index_name)
            dataframes.append(df)
        log.info("合并分子指纹数据...")
        merged_df = pd.concat(dataframes, axis=1, sort=False)
        if save_to_file:
            merged_df.to_csv(os.path.join(self.save_dir, self.merge_csv))
        if remove_existed_csvfiles:
            for csv_file in csv_files:
                os.remove(csv_file)
        log.info("合并后的指纹文件维度为: " + str(merged_df.shape))
        return merged_df
