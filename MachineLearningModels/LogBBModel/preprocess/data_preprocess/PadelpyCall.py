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
        [Chinese text removed]Padelpy[Chinese text removed]

        :param smi_filename: [Chinese text removed]SMILES[Chinese text removed]smi[Chinese text removed]
        :param save_dir: [Chinese text removed]
        :param merge_csv: [Chinese text removed]
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
        # self.xml_files = glob.glob("./fingerprints_xml/*.xml")  # [Chinese text removed]
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
        [Chinese text removed]，[Chinese text removed]
        :param fingerprints: [Chinese text removed]，[Chinese text removed]：['EState', 'MACCS', 'KlekotaRoth', 'PubChem']
        :return:
        """
        if self.smi_dir is not None:
            # [Chinese text removed]SMILES，[Chinese text removed]
            with open(self.smi_dir, "r") as file:
                lines = file.readlines()
            # [Chinese text removed]，[Chinese text removed]SMILES[Chinese text removed]
            SMILES = [line.rstrip("\n") for line in lines]
        else:
            raise ValueError("parametersmi_dir[Chinese text removed]None，parameterError")
        if fingerprints is None or len(fingerprints) == 0:
            raise ValueError("parameterfingerprintsError")
        self.target_fingerprints = fingerprints
        log.info(f"Starting[Chinese text removed]")
        for fingerprint in tqdm(fingerprints, desc="[Chinese text removed]: "):
            if fingerprint not in self.FP_list:
                log.error(f"Error: {fingerprint} [Chinese text removed]")
                continue
            # fingerprint = 'Substructure'
            fingerprint_output_file = os.path.join(self.save_dir, ''.join([fingerprint, '.csv']))
            fingerprint_descriptor_types = self.fp[fingerprint]  # [Chinese text removed]
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
                            threads=2,  # [Chinese text removed]
                            removesalt=True,
                            log=False,
                            fingerprints=True)
        # [Chinese text removed]csv[Chinese text removed]SMILES[Chinese text removed]AUTOGEN_result_{}Converting[Chinese text removed]SMILES
            df = pd.read_csv(fingerprint_output_file)
            df = df.drop('Name', axis=1)
            df.insert(0, self.index_name, pd.Series(SMILES))
            df.to_csv(fingerprint_output_file, index=False)

        return self.MergeResult()

    def MergeResult(self, remove_existed_csvfiles=True, save_to_file=False):
        """
        [Chinese text removed]features[Chinese text removed]csv[Chinese text removed]
        """
        csv_files = list()
        for fp in self.target_fingerprints:
            csv_files.append(os.path.join(self.save_dir, fp + ".csv"))
        # [Chinese text removed]features[Chinese text removed]dataframe
        dataframes = []
        for csv_file in csv_files:
            df = pd.read_csv(csv_file, index_col=self.index_name)
            dataframes.append(df)
        log.info("[Chinese text removed]...")
        # [Chinese text removed]Dataframe
        merged_df = pd.concat(dataframes, axis=1, sort=False)
        if save_to_file:
            merged_df.to_csv(os.path.join(self.save_dir, self.merge_csv))
        # [Chinese text removed]，[Chinese text removed]
        if remove_existed_csvfiles:
            for csv_file in csv_files:
                os.remove(csv_file)
        log.info("[Chinese text removed]: " + str(merged_df.shape))
        return merged_df
