#!/usr/bin/env python
# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import re

print("="*80)
print("计算original数据 + 18F重采样数据的组合数据集大小")
print("="*80)

# 提取同位素的函数
ALLOWED_ISOTOPES = {"18F", "11C", "125I", "131I", "123I", "77Br", "76Br"}

def extract_isotope_strict(compound_index):
    if not isinstance(compound_index, str):
        return None
    text = compound_index.strip()
    for iso in ALLOWED_ISOTOPES:
        if re.search(rf"(^|[^0-9A-Za-z]){re.escape(iso)}([^0-9A-Za-z]|$)", text):
            return iso
    return None

# ================== 1. LogBB (PETBD) 数据集 ==================
print("\n1. LogBB (PETBD) 数据集分析:")
print("-"*40)

df_petbd = pd.read_csv('dataset_PETBD/PETBD20240906.csv', encoding='utf-8')
df_petbd_valid = df_petbd.dropna(subset=['logBB'])
print(f"original有效samples数: {len(df_petbd_valid)}")

# 提取同位素信息
df_petbd_valid['isotope'] = df_petbd_valid['compound index'].apply(extract_isotope_strict)
df_with_isotope = df_petbd_valid[df_petbd_valid['isotope'].notna()]
df_no_isotope = df_petbd_valid[df_petbd_valid['isotope'].isna()]

print(f"可识别同位素的samples: {len(df_with_isotope)}")
print(f"无法识别同位素的samples: {len(df_no_isotope)}")

# 统计18F和非18F
n_18f = (df_with_isotope['isotope'] == '18F').sum()
n_non_18f = (df_with_isotope['isotope'] != '18F').sum()
print(f"  - 18F samples: {n_18f}")
print(f"  - 非18F samples: {n_non_18f}")

# 策略1：仅对可识别同位素的samples进行18F重采样
print("\n策略1：仅对可识别同位素的samples进行重采样")
# 过采样到平衡
target = max(n_18f, n_non_18f)
print(f"  - 重采样后每类: {target}")
print(f"  - 重采样后总数: {target * 2}")
print(f"  - 加上无法识别同位素的samples: {target * 2 + len(df_no_isotope)}")

# 策略2：使用全部original数据 + 18F重采样的少数类
print("\n策略2：original数据 + 18F少数类重采样（推荐）")
# 计算需要额外生成的18F samples
if n_18f < n_non_18f:
    extra_18f = n_non_18f - n_18f
    print(f"  - original数据: {len(df_petbd_valid)}")
    print(f"  - 需要额外生成的18F samples: {extra_18f}")
    print(f"  - 组合后Total samples数: {len(df_petbd_valid) + extra_18f}")
    logbb_combined = len(df_petbd_valid) + extra_18f
else:
    extra_non_18f = n_18f - n_non_18f
    print(f"  - original数据: {len(df_petbd_valid)}")
    print(f"  - 需要额外生成的非18F samples: {extra_non_18f}")
    print(f"  - 组合后Total samples数: {len(df_petbd_valid) + extra_non_18f}")
    logbb_combined = len(df_petbd_valid) + extra_non_18f

# Dataset split
print(f"\n  Dataset split（81%/9%/10%）:")
print(f"    - Training set: {int(logbb_combined * 0.81)}")
print(f"    - Validation set: {int(logbb_combined * 0.09)}")
print(f"    - Test set: {int(logbb_combined * 0.10)}")

# ================== 2. Cbrain (OrganDataAt60min) 数据集 ==================
print("\n2. Cbrain (OrganDataAt60min) 数据集分析:")
print("-"*40)

df_organ = pd.read_csv('data/logBB_data/OrganDataAt60min.csv', encoding='utf-8')
df_organ_valid = df_organ.dropna(subset=['brain mean60min'])
print(f"original有效samples数: {len(df_organ_valid)}")

# 找到compound index列
compound_col = None
for col in df_organ_valid.columns:
    if 'compound' in col.lower() and 'index' in col.lower():
        compound_col = col
        break

if compound_col:
    # 提取同位素信息
    df_organ_valid['isotope'] = df_organ_valid[compound_col].apply(extract_isotope_strict)
    df_organ_isotope = df_organ_valid[df_organ_valid['isotope'].notna()]
    df_organ_no_isotope = df_organ_valid[df_organ_valid['isotope'].isna()]
    
    print(f"可识别同位素的samples: {len(df_organ_isotope)}")
    print(f"无法识别同位素的samples: {len(df_organ_no_isotope)}")
    
    # 统计18F和非18F
    n_18f_organ = (df_organ_isotope['isotope'] == '18F').sum()
    n_non_18f_organ = (df_organ_isotope['isotope'] != '18F').sum()
    print(f"  - 18F samples: {n_18f_organ}")
    print(f"  - 非18F samples: {n_non_18f_organ}")
    
    # 策略1：仅对可识别同位素的samples进行18F重采样
    print("\n策略1：仅对可识别同位素的samples进行重采样")
    target_organ = max(n_18f_organ, n_non_18f_organ)
    print(f"  - 重采样后每类: {target_organ}")
    print(f"  - 重采样后总数: {target_organ * 2}")
    print(f"  - 加上无法识别同位素的samples: {target_organ * 2 + len(df_organ_no_isotope)}")
    
    # 策略2：使用全部original数据 + 18F重采样的少数类
    print("\n策略2：original数据 + 18F少数类重采样（推荐）")
    if n_18f_organ < n_non_18f_organ:
        extra_18f_organ = n_non_18f_organ - n_18f_organ
        print(f"  - original数据: {len(df_organ_valid)}")
        print(f"  - 需要额外生成的18F samples: {extra_18f_organ}")
        print(f"  - 组合后Total samples数: {len(df_organ_valid) + extra_18f_organ}")
        cbrain_combined = len(df_organ_valid) + extra_18f_organ
    else:
        extra_non_18f_organ = n_18f_organ - n_non_18f_organ
        print(f"  - original数据: {len(df_organ_valid)}")
        print(f"  - 需要额外生成的非18F samples: {extra_non_18f_organ}")
        print(f"  - 组合后Total samples数: {len(df_organ_valid) + extra_non_18f_organ}")
        cbrain_combined = len(df_organ_valid) + extra_non_18f_organ
    
    # Dataset split
    print(f"\n  Dataset split（81%/9%/10%）:")
    print(f"    - Training set: {int(cbrain_combined * 0.81)}")
    print(f"    - Validation set: {int(cbrain_combined * 0.09)}")
    print(f"    - Test set: {int(cbrain_combined * 0.10)}")

# ================== 3. 总结 ==================
print("\n" + "="*80)
print("【推荐方案】original数据 + 少数类18F重采样")
print("="*80)

print(f"\nLogBB (PETBD):")
print(f"  - 组合后Total samples数: {logbb_combined}")
print(f"  - Training set: {int(logbb_combined * 0.81)}")
print(f"  - Validation set: {int(logbb_combined * 0.09)}")
print(f"  - Test set: {int(logbb_combined * 0.10)}")

print(f"\nCbrain (OrganDataAt60min):")
print(f"  - 组合后Total samples数: {cbrain_combined}")
print(f"  - Training set: {int(cbrain_combined * 0.81)}")
print(f"  - Validation set: {int(cbrain_combined * 0.09)}")
print(f"  - Test set: {int(cbrain_combined * 0.10)}")

print("\n这种方法的优势：")
print("1. 充分利用所有original数据")
print("2. 通过重采样平衡18F/非18F分布")
print("3. 避免丢失有价值的samples信息")
print("4. 训练集规模更大，模型学习更充分")

