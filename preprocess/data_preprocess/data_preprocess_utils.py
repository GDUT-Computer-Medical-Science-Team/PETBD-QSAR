import pandas as pd
from padelpy import padeldescriptor
import tempfile
import os

def calculate_padel_fingerprints(smiles_list):
    """
    计算分子的PaDEL指纹
    :param smiles_list: SMILES 字符串列表
    :return: DataFrame，包含所有计算的指纹
    """
    # 创建临时文件来存储SMILES
    with tempfile.NamedTemporaryFile(mode='w', suffix='.smi', delete=False) as temp_file:
        for smi in smiles_list:
            temp_file.write(f"{smi}\n")
        temp_smi_file = temp_file.name

    try:
        # 创建临时目录存储结果
        with tempfile.TemporaryDirectory() as temp_dir:
            output_file = os.path.join(temp_dir, "fingerprints.csv")
            
            # 直接计算所有可用的指纹
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
            # 读取结果
            df = pd.read_csv(output_file)
            # 删除Name列（如果存在）
            if 'Name' in df.columns:
                df = df.drop('Name', axis=1)
            # 添加SMILES列
            df.insert(0, 'SMILES', smiles_list)
            return df
    except Exception as e:
        log.error(f"计算PaDEL指纹时发生错误: {str(e)}")
        raise
    finally:
        # 清理临时文件
        if os.path.exists(temp_smi_file):
            os.remove(temp_smi_file) 