#!/usr/bin/env python3
"""
Get actual names of Mordred descriptors
"""
import sys
sys.path.append('MachineLearningModels/logBBModel/')

from preprocess.data_preprocess.data_preprocess_utils import calculate_Mordred_desc
import pandas as pd

def get_mordred_descriptor_names():
    """
    Get actual names of Mordred descriptors
    """
    test_smiles = "CCO"
    df_test = pd.DataFrame({'SMILES': [test_smiles]})
    
    df_with_mordred = calculate_Mordred_desc(df_test['SMILES'])
    
    mordred_cols = [col for col in df_with_mordred.columns if col not in ['SMILES']]
    
    print(f"Found {len(mordred_cols)} Mordred descriptors:")
    for i, col in enumerate(mordred_cols):
        print(f"{i}: {col}")
    
    with open('mordred_descriptor_names.txt', 'w', encoding='utf-8') as f:
        for i, col in enumerate(mordred_cols):
            f.write(f"{i}\t{col}\n")
    
    print(f"\nDescriptor names saved to: mordred_descriptor_names.txt")
    return mordred_cols

if __name__ == "__main__":
    get_mordred_descriptor_names()
