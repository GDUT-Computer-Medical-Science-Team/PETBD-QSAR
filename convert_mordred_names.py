#!/usr/bin/env python3
"""
将 Mordred_xxx 索引转换为实际描述符名称
"""
import sys
sys.path.append('MachineLearningModels/logBBModel/')

from preprocess.data_preprocess.data_preprocess_utils import calculate_Mordred_desc
import pandas as pd

def convert_mordred_names():
    """
    将特征名称文件中的 Mordred_xxx 转换为实际名称
    """
    # 读取 XGBoost 特征名称文件
    feature_file = "MachineLearningModels/logBBModel/logbbModel/xgb_combined_18F_feature_names.txt"
    
    with open(feature_file, 'r', encoding='utf-8') as f:
        feature_names = [line.strip() for line in f.readlines()]
    
    print("原始特征名称:")
    for i, name in enumerate(feature_names):
        print(f"{i}: {name}")
    
    # 获取 Mordred 描述符的实际名称
    test_smiles = "CCO"  # 乙醇
    df_test = pd.DataFrame({'SMILES': [test_smiles]})
    df_with_mordred = calculate_Mordred_desc(df_test['SMILES'])
    mordred_cols = [col for col in df_with_mordred.columns if col not in ['SMILES']]
    
    print(f"\n找到 {len(mordred_cols)} 个 Mordred 描述符")
    
    # 转换特征名称
    converted_names = []
    for name in feature_names:
        if name.startswith('Mordred_'):
            # 提取索引
            try:
                idx = int(name.split('_')[1])
                if idx < len(mordred_cols):
                    converted_name = f"Mordred_{mordred_cols[idx]}"
                    converted_names.append(converted_name)
                    print(f"{name} -> {converted_name}")
                else:
                    converted_names.append(name)
                    print(f"{name} -> {name} (索引超出范围)")
            except (ValueError, IndexError):
                converted_names.append(name)
                print(f"{name} -> {name} (无法解析索引)")
        else:
            converted_names.append(name)
    
    # 保存转换后的名称
    output_file = "converted_feature_names.txt"
    with open(output_file, 'w', encoding='utf-8') as f:
        for name in converted_names:
            f.write(f"{name}\n")
    
    print(f"\n转换后的特征名称已保存到: {output_file}")
    
    return converted_names

if __name__ == "__main__":
    convert_mordred_names()

