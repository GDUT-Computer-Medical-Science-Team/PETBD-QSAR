#!/usr/bin/env python3
"""
获取 Mordred 描述符的实际名称
"""
import sys
sys.path.append('MachineLearningModels/logBBModel/')

from preprocess.data_preprocess.data_preprocess_utils import calculate_Mordred_desc
import pandas as pd

def get_mordred_descriptor_names():
    """
    获取 Mordred 描述符的实际名称
    """
    # 使用一个简单的 SMILES 来获取描述符名称
    test_smiles = "CCO"  # 乙醇
    df_test = pd.DataFrame({'SMILES': [test_smiles]})
    
    # 计算 Mordred 描述符
    df_with_mordred = calculate_Mordred_desc(df_test['SMILES'])
    
    # 获取描述符列名（排除 SMILES）
    mordred_cols = [col for col in df_with_mordred.columns if col not in ['SMILES']]
    
    print(f"找到 {len(mordred_cols)} 个 Mordred 描述符:")
    for i, col in enumerate(mordred_cols):
        print(f"{i}: {col}")
    
    # 保存到文件
    with open('mordred_descriptor_names.txt', 'w', encoding='utf-8') as f:
        for i, col in enumerate(mordred_cols):
            f.write(f"{i}\t{col}\n")
    
    print(f"\n描述符名称已保存到: mordred_descriptor_names.txt")
    return mordred_cols

if __name__ == "__main__":
    get_mordred_descriptor_names()

