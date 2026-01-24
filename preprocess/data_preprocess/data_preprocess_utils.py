import pandas as pd
from padelpy import padeldescriptor
import tempfile
import os

def calculate_padel_fingerprints(smiles_list):
    """
    """
    with tempfile.NamedTemporaryFile(mode='w', suffix='.smi', delete=False) as temp_file:
        for smi in smiles_list:
            temp_file.write(f"{smi}\n")
        temp_smi_file = temp_file.name

    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            output_file = os.path.join(temp_dir, "fingerprints.csv")
            
            padeldescriptor(
                mol_dir=temp_smi_file,
                d_file=output_file,
                fingerprints=True,
                descriptortypes=None,
                detectaromaticity=True,
                standardizenitro=True,
                standardizetautomers=True,
                threads=2,
                removesalt=True,
                log=False
            )
            df = pd.read_csv(output_file)
            if 'Name' in df.columns:
                df = df.drop('Name', axis=1)
            df.insert(0, 'SMILES', smiles_list)
            return df
    except Exception as e:
        log.error(f"计算PaDEL指纹时发生错误: {str(e)}")
        raise
    finally:
        if os.path.exists(temp_smi_file):
            os.remove(temp_smi_file) 