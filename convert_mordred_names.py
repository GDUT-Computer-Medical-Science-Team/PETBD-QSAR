#!/usr/bin/env python3
"""
Convert Mordred_xxx indices to actual descriptor names
"""
import sys
sys.path.append('MachineLearningModels/logBBModel/')

from preprocess.data_preprocess.data_preprocess_utils import calculate_Mordred_desc
import pandas as pd

def convert_mordred_names():
    """
    Convert Mordred_xxx in feature names file to actual names
    """
    feature_file = "MachineLearningModels/logBBModel/logbbModel/xgb_combined_18F_feature_names.txt"
    
    with open(feature_file, 'r', encoding='utf-8') as f:
        feature_names = [line.strip() for line in f.readlines()]
    
    print("Original feature names:")
    for i, name in enumerate(feature_names):
        print(f"{i}: {name}")
    
    test_smiles = "CCO"
    df_test = pd.DataFrame({'SMILES': [test_smiles]})
    df_with_mordred = calculate_Mordred_desc(df_test['SMILES'])
    mordred_cols = [col for col in df_with_mordred.columns if col not in ['SMILES']]
    
    print(f"\nFound {len(mordred_cols)} Mordred descriptors")
    
    converted_names = []
    for name in feature_names:
        if name.startswith('Mordred_'):
            try:
                idx = int(name.split('_')[1])
                if idx < len(mordred_cols):
                    converted_name = f"Mordred_{mordred_cols[idx]}"
                    converted_names.append(converted_name)
                    print(f"{name} -> {converted_name}")
                else:
                    converted_names.append(name)
                    print(f"{name} -> {name} (index out of range)")
            except (ValueError, IndexError):
                converted_names.append(name)
                print(f"{name} -> {name} (unable to parse index)")
        else:
            converted_names.append(name)
    
    output_file = "converted_feature_names.txt"
    with open(output_file, 'w', encoding='utf-8') as f:
        for name in converted_names:
            f.write(f"{name}\n")
    
    print(f"\nConverted feature names saved to: {output_file}")
    
    return converted_names

if __name__ == "__main__":
    convert_mordred_names()
