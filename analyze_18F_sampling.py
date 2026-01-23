#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
18F samples采样策略详细分析
分析PETBD和OrganDataAt60min数据集中的18F samples分布和采样策略
"""

import pandas as pd
import numpy as np
import re
from typing import Optional
import matplotlib.pyplot as plt
import seaborn as sns

# 允许的同位素列表
ALLOWED_ISOTOPES = {"18F", "11C", "125I", "131I", "123I", "77Br", "76Br"}

def extract_isotope_strict(compound_index: str) -> Optional[str]:
    """从化合物索引中严格提取同位素信息"""
    if not isinstance(compound_index, str):
        return None
    
    text = compound_index.strip()
    
    # 优先匹配完整的同位素标记
    for iso in ALLOWED_ISOTOPES:
        if re.search(rf"(^|[^0-9A-Za-z]){re.escape(iso)}([^0-9A-Za-z]|$)", text):
            return iso
    
    return None

def analyze_dataset_sampling(df: pd.DataFrame, dataset_name: str, target_column: str):
    """分析单个数据集的采样情况"""
    print(f"\n{'='*80}")
    print(f"Dataset: {dataset_name}")
    print('='*80)
    
    # 基本信息
    print(f"Total samples: {len(df)}")
    
    # 删除目标变量缺失值
    df_valid = df.dropna(subset=[target_column])
    print(f"Valid samples (with {target_column}): {len(df_valid)}")
    
    # 提取同位素信息
    if "compound index" in df_valid.columns:
        isotopes = df_valid["compound index"].apply(extract_isotope_strict)
    else:
        print("Warning: 'compound index' column not found")
        return None
    
    df_valid = df_valid.copy()
    df_valid["isotope"] = isotopes
    
    # 只保留能识别同位素的samples
    df_isotope = df_valid[df_valid["isotope"].notna()].copy()
    print(f"Samples with identified isotope: {len(df_isotope)}")
    
    # 统计各同位素分布
    print("\n--- Isotope Distribution ---")
    isotope_counts = df_isotope["isotope"].value_counts()
    for iso, count in isotope_counts.items():
        print(f"  {iso:5s}: {count:4d} samples ({count/len(df_isotope)*100:5.1f}%)")
    
    # 18F vs non-18F分析
    df_isotope["is_18F"] = df_isotope["isotope"] == "18F"
    n_18f = df_isotope["is_18F"].sum()
    n_non_18f = (~df_isotope["is_18F"]).sum()
    
    print(f"\n--- 18F Binary Classification ---")
    print(f"  18F samples:     {n_18f:4d} ({n_18f/len(df_isotope)*100:5.1f}%)")
    print(f"  Non-18F samples: {n_non_18f:4d} ({n_non_18f/len(df_isotope)*100:5.1f}%)")
    print(f"  Imbalance ratio: 1:{n_non_18f/n_18f:.2f} (18F:non-18F)")
    
    # 采样策略分析
    print(f"\n--- Sampling Strategies ---")
    
    # 1. 过采样 (Oversampling)
    oversample_target = max(n_18f, n_non_18f)
    print(f"\n1. OVERSAMPLING (match to majority class):")
    print(f"   Target samples per class: {oversample_target}")
    print(f"   Total samples after:      {oversample_target * 2}")
    
    if n_18f < n_non_18f:
        print(f"   18F samples to generate:  {oversample_target - n_18f} (replicate {(oversample_target/n_18f):.1f}x)")
        print(f"   Non-18F samples:          {oversample_target} (no change)")
    else:
        print(f"   18F samples:              {oversample_target} (no change)")  
        print(f"   Non-18F to generate:      {oversample_target - n_non_18f} (replicate {(oversample_target/n_non_18f):.1f}x)")
    
    print(f"   Data increase:            {(oversample_target * 2 - len(df_isotope))/len(df_isotope)*100:.1f}%")
    
    # 2. 欠采样 (Undersampling)
    undersample_target = min(n_18f, n_non_18f)
    print(f"\n2. UNDERSAMPLING (match to minority class):")
    print(f"   Target samples per class: {undersample_target}")
    print(f"   Total samples after:      {undersample_target * 2}")
    
    if n_18f > n_non_18f:
        print(f"   18F samples to remove:    {n_18f - undersample_target} (keep {undersample_target/n_18f*100:.1f}%)")
        print(f"   Non-18F samples:          {undersample_target} (no change)")
    else:
        print(f"   18F samples:              {undersample_target} (no change)")
        print(f"   Non-18F to remove:        {n_non_18f - undersample_target} (keep {undersample_target/n_non_18f*100:.1f}%)")
    
    print(f"   Data reduction:           {(len(df_isotope) - undersample_target * 2)/len(df_isotope)*100:.1f}%")
    
    # 3. SMOTE建议（合成少数类过采样）
    print(f"\n3. SMOTE (Synthetic Minority Over-sampling):")
    print(f"   Recommended for highly imbalanced data")
    print(f"   Current imbalance ratio: 1:{max(n_18f, n_non_18f)/min(n_18f, n_non_18f):.1f}")
    if max(n_18f, n_non_18f)/min(n_18f, n_non_18f) > 3:
        print(f"   ⚠️ High imbalance detected - SMOTE may be beneficial")
    else:
        print(f"   ✓ Moderate imbalance - simple resampling should work")
    
    return {
        'dataset': dataset_name,
        'total_samples': len(df),
        'valid_samples': len(df_valid),
        'isotope_samples': len(df_isotope),
        'n_18f': n_18f,
        'n_non_18f': n_non_18f,
        'imbalance_ratio': n_non_18f/n_18f if n_18f > 0 else float('inf'),
        'oversample_target': oversample_target,
        'undersample_target': undersample_target
    }

def main():
    """主分析函数"""
    print("="*80)
    print("18F SAMPLING STRATEGY ANALYSIS")
    print("="*80)
    
    results = []
    
    # 分析PETBD数据集
    try:
        df_petbd = pd.read_csv("data/PTBD_v20240912.csv", encoding='utf-8')
        result_petbd = analyze_dataset_sampling(df_petbd, "PETBD (LogBB)", "logBB")
        if result_petbd:
            results.append(result_petbd)
    except Exception as e:
        print(f"Error loading PETBD dataset: {e}")
    
    # 分析OrganDataAt60min数据集
    try:
        df_organ = pd.read_csv("data/OrganDataAt60min.csv", encoding='utf-8')
        result_organ = analyze_dataset_sampling(df_organ, "OrganDataAt60min (Cbrain)", "brain mean60min")
        if result_organ:
            results.append(result_organ)
    except Exception as e:
        print(f"Error loading OrganDataAt60min dataset: {e}")
    
    # 比较分析
    if len(results) == 2:
        print("\n" + "="*80)
        print("COMPARATIVE ANALYSIS")
        print("="*80)
        
        print("\n--- Dataset Comparison ---")
        print(f"{'Dataset':<25} {'Total':<8} {'Valid':<8} {'18F':<8} {'Non-18F':<8} {'Ratio':<10}")
        print("-"*70)
        for r in results:
            print(f"{r['dataset']:<25} {r['total_samples']:<8} {r['valid_samples']:<8} "
                  f"{r['n_18f']:<8} {r['n_non_18f']:<8} 1:{r['imbalance_ratio']:.2f}")
        
        print("\n--- Sampling Impact ---")
        print(f"{'Dataset':<25} {'Original':<10} {'Oversample':<12} {'Undersample':<12}")
        print("-"*60)
        for r in results:
            original = r['isotope_samples']
            oversample = r['oversample_target'] * 2
            undersample = r['undersample_target'] * 2
            print(f"{r['dataset']:<25} {original:<10} {oversample:<12} {undersample:<12}")
            print(f"{'':25} {'':10} ({(oversample-original)/original*100:+.1f}%) "
                  f"     ({(undersample-original)/original*100:+.1f}%)")
    
    # 创建可视化
    if results:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # 左图：Original data distribution
        ax1 = axes[0]
        datasets = [r['dataset'].split()[0] for r in results]
        x = np.arange(len(datasets))
        width = 0.35
        
        bars1 = ax1.bar(x - width/2, [r['n_18f'] for r in results], width, label='18F', color='blue', alpha=0.7)
        bars2 = ax1.bar(x + width/2, [r['n_non_18f'] for r in results], width, label='Non-18F', color='orange', alpha=0.7)
        
        ax1.set_xlabel('Dataset')
        ax1.set_ylabel('Number of Samples')
        ax1.set_title('Original 18F Distribution')
        ax1.set_xticks(x)
        ax1.set_xticklabels(datasets)
        ax1.legend()
        
        # 添加数值标签
        for bar in bars1:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height)}', ha='center', va='bottom')
        for bar in bars2:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height)}', ha='center', va='bottom')
        
        # 右图：采样策略对比
        ax2 = axes[1]
        strategies = ['Original', 'Oversample', 'Undersample']
        
        for i, r in enumerate(results):
            original = r['isotope_samples']
            oversample = r['oversample_target'] * 2
            undersample = r['undersample_target'] * 2
            
            ax2.plot(strategies, [original, oversample, undersample], 
                    marker='o', linewidth=2, markersize=8, 
                    label=r['dataset'].split()[0], alpha=0.8)
        
        ax2.set_xlabel('Sampling Strategy')
        ax2.set_ylabel('Total Samples')
        ax2.set_title('Sampling Strategy Comparison')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('18F_sampling_analysis.png', dpi=150, bbox_inches='tight')
        print(f"\nVisualization saved to: 18F_sampling_analysis.png")
        plt.show()
    
    # 生成建议
    print("\n" + "="*80)
    print("RECOMMENDATIONS")
    print("="*80)
    
    for r in results:
        print(f"\n{r['dataset']}:")
        ratio = r['imbalance_ratio']
        
        if ratio < 1.5:
            print("  ✓ Relatively balanced dataset")
            print("  - Consider using original data without resampling")
            print("  - If needed, slight oversampling may help")
        elif ratio < 3:
            print("  ⚠️ Moderate imbalance detected")
            print("  - Oversampling recommended for better balance")
            print("  - Consider ensemble methods with different sampling")
        else:
            print("  ⚠️ High imbalance detected")
            print("  - Strong oversampling recommended")
            print("  - Consider SMOTE or ADASYN for synthetic sample generation")
            print("  - Ensemble methods with balanced sampling highly recommended")
    
    print("\n✅ Analysis complete!")

if __name__ == "__main__":
    main()