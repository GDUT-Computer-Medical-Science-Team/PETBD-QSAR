#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
18F Sampling Analysis Script
Analyze 18F distribution and sampling strategies in PETBD dataset
"""

import pandas as pd
import numpy as np
import re
from typing import Optional
import matplotlib.pyplot as plt
import seaborn as sns

plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['axes.unicode_minus'] = False

ALLOWED_ISOTOPES = {'18F', '11C', '125I', '131I', '123I', '77Br', '76Br'}

def extract_isotope_strict(compound_index: str) -> Optional[str]:
    """Extract isotope information from compound index"""
    if not isinstance(compound_index, str):
        return None
    
    text = compound_index.strip()
    
    for iso in ALLOWED_ISOTOPES:
        if re.search(rf"(^|[^0-9A-Za-z]){re.escape(iso)}([^0-9A-Za-z]|$)", text):
            return iso
    
    return None

def analyze_dataset():
    """Analyze 18F distribution in PETBD dataset"""
    
    print("="*80)
    print("18F Sampling Analysis - PETBD Dataset")
    print("="*80)
    
    df = pd.read_csv('dataset_PETBD/PTBD_v20240912.csv', encoding='utf-8')
    print(f"\nOriginal dataset size: {len(df)} samples")
    
    df['isotope'] = df['compound index'].apply(extract_isotope_strict)
    
    df_with_isotope = df[df['isotope'].notna()]
    print(f"Samples with isotope labels: {len(df_with_isotope)}")
    
    df_valid = df_with_isotope.dropna(subset=['logBB'])
    print(f"Valid samples (isotope+logBB): {len(df_valid)}")
    
    df_valid['is_18F'] = (df_valid['isotope'] == '18F')
    
    n_18F = df_valid['is_18F'].sum()
    n_non_18F = (~df_valid['is_18F']).sum()
    
    print("\n" + "="*50)
    print("Isotope Distribution Statistics")
    print("="*50)
    
    isotope_counts = df_valid['isotope'].value_counts()
    for isotope, count in isotope_counts.items():
        percentage = count / len(df_valid) * 100
        print(f"{isotope:6s}: {count:3d} samples ({percentage:5.1f}%)")
    
    print("\n" + "="*50)
    print("18F vs Non-18F Classification")
    print("="*50)
    print(f"18F samples:     {n_18F:3d} ({n_18F/len(df_valid)*100:5.1f}%)")
    print(f"Non-18F samples: {n_non_18F:3d} ({n_non_18F/len(df_valid)*100:5.1f}%)")
    print(f"Total:           {len(df_valid):3d}")
    print(f"Ratio (18F:Non-18F): 1:{n_non_18F/n_18F:.2f}")
    
    non_18F_isotopes = df_valid[~df_valid['is_18F']]['isotope'].value_counts()
    print(f"\nNon-18F composition:")
    for isotope, count in non_18F_isotopes.items():
        print(f"  {isotope:6s}: {count:3d}")
    
    print("\n" + "="*50)
    print("Sampling Strategy Analysis")
    print("="*50)
    
    print("\n1. Oversampling Strategy - Used in scripts:")
    print("-"*40)
    target_oversample = max(n_18F, n_non_18F)
    print(f"   Target: {target_oversample} samples per class")
    print(f"   18F processing:")
    print(f"     - Original: {n_18F}")
    print(f"     - Target: {target_oversample}")
    if target_oversample > n_18F:
        print(f"     - Operation: Add {target_oversample - n_18F} replicated samples")
        print(f"     - Replication factor: {target_oversample/n_18F:.2f}x")
    else:
        print(f"     - Operation: Keep unchanged")
    
    print(f"   Non-18F processing:")
    print(f"     - Original: {n_non_18F}")
    print(f"     - Target: {target_oversample}")
    if target_oversample > n_non_18F:
        print(f"     - Operation: Add {target_oversample - n_non_18F} replicated samples")
        print(f"     - Replication factor: {target_oversample/n_non_18F:.2f}x")
    else:
        print(f"     - Operation: Keep unchanged")
    
    print(f"   Final dataset: {target_oversample * 2} samples ({target_oversample*2/len(df_valid)*100:.1f}% of original)")
    
    print("\n2. Undersampling Strategy:")
    print("-"*40)
    target_undersample = min(n_18F, n_non_18F)
    print(f"   Target: {target_undersample} samples per class")
    print(f"   18F processing:")
    print(f"     - Original: {n_18F}")
    print(f"     - Target: {target_undersample}")
    if n_18F > target_undersample:
        print(f"     - Operation: Remove {n_18F - target_undersample} samples")
        print(f"     - Retention rate: {target_undersample/n_18F*100:.1f}%")
    else:
        print(f"     - Operation: Keep unchanged")
    
    print(f"   Non-18F processing:")
    print(f"     - Original: {n_non_18F}")
    print(f"     - Target: {target_undersample}")
    if n_non_18F > target_undersample:
        print(f"     - Operation: Remove {n_non_18F - target_undersample} samples")
        print(f"     - Retention rate: {target_undersample/n_non_18F*100:.1f}%")
    else:
        print(f"     - Operation: Keep unchanged")
    
    print(f"   Final dataset: {target_undersample * 2} samples ({target_undersample*2/len(df_valid)*100:.1f}% of original)")
    
    print("\n" + "="*50)
    print("Key Findings")
    print("="*50)
    print(f"[1] 18F is majority class: {n_18F/len(df_valid)*100:.1f}%")
    print(f"[2] Non-18F is minority class: {n_non_18F/len(df_valid)*100:.1f}%")
    print(f"[3] Severe class imbalance, ratio: 1:{n_non_18F/n_18F:.2f}")
    print(f"[4] Oversampling requires {target_oversample/n_non_18F:.1f}x replication of non-18F")
    print(f"[5] Undersampling loses {(n_18F-target_undersample)/n_18F*100:.1f}% of 18F samples")
    
    create_visualizations(df_valid, n_18F, n_non_18F, isotope_counts)
    
    return df_valid

def create_visualizations(df_valid, n_18F, n_non_18F, isotope_counts):
    """Create visualization charts"""
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    ax1 = axes[0, 0]
    isotopes = list(isotope_counts.index)
    counts = list(isotope_counts.values)
    colors = ['red' if iso == '18F' else 'blue' for iso in isotopes]
    
    bars = ax1.bar(isotopes, counts, color=colors, alpha=0.7)
    ax1.set_title('Isotope Distribution', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Isotope Type')
    ax1.set_ylabel('Sample Count')
    ax1.grid(True, alpha=0.3)
    
    for bar, count in zip(bars, counts):
        ax1.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                f'{count}', ha='center', va='bottom')
    
    ax2 = axes[0, 1]
    sizes = [n_18F, n_non_18F]
    labels = [f'18F\n({n_18F}, {n_18F/(n_18F+n_non_18F)*100:.1f}%)', 
              f'Non-18F\n({n_non_18F}, {n_non_18F/(n_18F+n_non_18F)*100:.1f}%)']
    colors = ['#ff9999', '#66b3ff']
    explode = (0.05, 0.05)
    
    ax2.pie(sizes, explode=explode, labels=labels, colors=colors,
            autopct='%1.0f%%', shadow=True, startangle=90)
    ax2.set_title('18F vs Non-18F Distribution', fontsize=14, fontweight='bold')
    
    ax3 = axes[1, 0]
    strategies = ['Original', 'Oversample', 'Undersample']
    n_18F_list = [n_18F, max(n_18F, n_non_18F), min(n_18F, n_non_18F)]
    n_non_18F_list = [n_non_18F, max(n_18F, n_non_18F), min(n_18F, n_non_18F)]
    
    x = np.arange(len(strategies))
    width = 0.35
    
    bars1 = ax3.bar(x - width/2, n_18F_list, width, label='18F', color='#ff9999')
    bars2 = ax3.bar(x + width/2, n_non_18F_list, width, label='Non-18F', color='#66b3ff')
    
    ax3.set_xlabel('Sampling Strategy')
    ax3.set_ylabel('Sample Count')
    ax3.set_title('Sample Count by Strategy', fontsize=14, fontweight='bold')
    ax3.set_xticks(x)
    ax3.set_xticklabels(strategies)
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height)}', ha='center', va='bottom')
    
    ax4 = axes[1, 1]
    total_original = n_18F + n_non_18F
    total_oversample = max(n_18F, n_non_18F) * 2
    total_undersample = min(n_18F, n_non_18F) * 2
    
    strategies = ['Original', 'Oversample', 'Undersample']
    totals = [total_original, total_oversample, total_undersample]
    percentages = [100, total_oversample/total_original*100, total_undersample/total_original*100]
    
    bars = ax4.bar(strategies, totals, color=['green', 'orange', 'purple'], alpha=0.7)
    ax4.set_ylabel('Total Sample Count')
    ax4.set_title('Dataset Size Changes', fontsize=14, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    
    for bar, total, pct in zip(bars, totals, percentages):
        ax4.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                f'{int(total)}\n({pct:.0f}%)', ha='center', va='bottom')
    
    ax4.axhline(y=total_original, color='red', linestyle='--', alpha=0.5, label='Original size')
    ax4.legend()
    
    plt.suptitle('PETBD Dataset - 18F Sampling Analysis', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('18F_sampling_visualization.png', dpi=300, bbox_inches='tight')
    print(f"\nVisualization saved: 18F_sampling_visualization.png")

if __name__ == '__main__':
    df_result = analyze_dataset()
