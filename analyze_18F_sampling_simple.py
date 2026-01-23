#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
18F采样数量分析脚本
分析PETBD数据集的18F分布及采样策略
"""

import pandas as pd
import numpy as np
import re
from typing import Optional
import matplotlib.pyplot as plt
import seaborn as sns

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 允许的同位素列表
ALLOWED_ISOTOPES = {'18F', '11C', '125I', '131I', '123I', '77Br', '76Br'}

def extract_isotope_strict(compound_index: str) -> Optional[str]:
    """从化合物索引中提取同位素信息"""
    if not isinstance(compound_index, str):
        return None
    
    text = compound_index.strip()
    
    for iso in ALLOWED_ISOTOPES:
        if re.search(rf"(^|[^0-9A-Za-z]){re.escape(iso)}([^0-9A-Za-z]|$)", text):
            return iso
    
    return None

def analyze_dataset():
    """分析PETBD数据集的18F分布"""
    
    print("="*80)
    print("18F 采样数量分析 - PETBD数据集")
    print("="*80)
    
    # 读取数据
    df = pd.read_csv('dataset_PETBD/PETBD20240906.csv', encoding='utf-8')
    print(f"\noriginal数据集大小: {len(df)} 个samples")
    
    # 提取同位素信息
    df['isotope'] = df['compound index'].apply(extract_isotope_strict)
    
    # 删除没有同位素标记的samples
    df_with_isotope = df[df['isotope'].notna()]
    print(f"有同位素标记的samples: {len(df_with_isotope)} 个")
    
    # 删除没有logBB值的samples
    df_valid = df_with_isotope.dropna(subset=['logBB'])
    print(f"有效samples（同位素+logBB）: {len(df_valid)} 个")
    
    # 创建18F标签
    df_valid['is_18F'] = (df_valid['isotope'] == '18F')
    
    # 统计18F vs 非18F
    n_18F = df_valid['is_18F'].sum()
    n_non_18F = (~df_valid['is_18F']).sum()
    
    print("\n" + "="*50)
    print("同位素分布统计")
    print("="*50)
    
    # 详细的同位素分布
    isotope_counts = df_valid['isotope'].value_counts()
    for isotope, count in isotope_counts.items():
        percentage = count / len(df_valid) * 100
        print(f"{isotope:6s}: {count:3d} 个samples ({percentage:5.1f}%)")
    
    print("\n" + "="*50)
    print("18F vs 非18F 分类统计")
    print("="*50)
    print(f"18F samples:     {n_18F:3d} 个 ({n_18F/len(df_valid)*100:5.1f}%)")
    print(f"非18F samples:   {n_non_18F:3d} 个 ({n_non_18F/len(df_valid)*100:5.1f}%)")
    print(f"总计:        {len(df_valid):3d} 个")
    print(f"比例 (18F:非18F): 1:{n_non_18F/n_18F:.2f}")
    
    # 非18F的具体组成
    non_18F_isotopes = df_valid[~df_valid['is_18F']]['isotope'].value_counts()
    print(f"\n非18F samples的组成:")
    for isotope, count in non_18F_isotopes.items():
        print(f"  {isotope:6s}: {count:3d} 个")
    
    print("\n" + "="*50)
    print("采样策略分析")
    print("="*50)
    
    # 过采样策略
    print("\n1. 过采样策略 (Oversampling) - 脚本中使用的方法:")
    print("-"*40)
    target_oversample = max(n_18F, n_non_18F)
    print(f"   目标: 每类 {target_oversample} 个samples")
    print(f"   18F处理:")
    print(f"     - original: {n_18F} 个")
    print(f"     - 目标: {target_oversample} 个")
    if target_oversample > n_18F:
        print(f"     - 操作: 复制增加 {target_oversample - n_18F} 个samples")
        print(f"     - 复制倍数: {target_oversample/n_18F:.2f}x")
    else:
        print(f"     - 操作: 保持不变")
    
    print(f"   非18F处理:")
    print(f"     - original: {n_non_18F} 个")
    print(f"     - 目标: {target_oversample} 个")
    if target_oversample > n_non_18F:
        print(f"     - 操作: 复制增加 {target_oversample - n_non_18F} 个samples")
        print(f"     - 复制倍数: {target_oversample/n_non_18F:.2f}x")
    else:
        print(f"     - 操作: 保持不变")
    
    print(f"   最终数据集: {target_oversample * 2} 个samples (original的 {target_oversample*2/len(df_valid)*100:.1f}%)")
    
    # 欠采样策略
    print("\n2. 欠采样策略 (Undersampling):")
    print("-"*40)
    target_undersample = min(n_18F, n_non_18F)
    print(f"   目标: 每类 {target_undersample} 个samples")
    print(f"   18F处理:")
    print(f"     - original: {n_18F} 个")
    print(f"     - 目标: {target_undersample} 个")
    if n_18F > target_undersample:
        print(f"     - 操作: 随机删除 {n_18F - target_undersample} 个samples")
        print(f"     - 保留率: {target_undersample/n_18F*100:.1f}%")
    else:
        print(f"     - 操作: 保持不变")
    
    print(f"   非18F处理:")
    print(f"     - original: {n_non_18F} 个")
    print(f"     - 目标: {target_undersample} 个")
    if n_non_18F > target_undersample:
        print(f"     - 操作: 随机删除 {n_non_18F - target_undersample} 个samples")
        print(f"     - 保留率: {target_undersample/n_non_18F*100:.1f}%")
    else:
        print(f"     - 操作: 保持不变")
    
    print(f"   最终数据集: {target_undersample * 2} 个samples (original的 {target_undersample*2/len(df_valid)*100:.1f}%)")
    
    print("\n" + "="*50)
    print("关键发现")
    print("="*50)
    print(f"[1] 18F是多数类，占 {n_18F/len(df_valid)*100:.1f}% 的samples")
    print(f"[2] 非18F是少数类，仅占 {n_non_18F/len(df_valid)*100:.1f}% 的samples")
    print(f"[3] 数据严重不平衡，比例为 1:{n_non_18F/n_18F:.2f}")
    print(f"[4] 过采样需要将非18F samples复制 {target_oversample/n_non_18F:.1f} 倍")
    print(f"[5] 欠采样会损失 {(n_18F-target_undersample)/n_18F*100:.1f}% 的18F samples")
    
    # 创建可视化
    create_visualizations(df_valid, n_18F, n_non_18F, isotope_counts)
    
    return df_valid

def create_visualizations(df_valid, n_18F, n_non_18F, isotope_counts):
    """创建可视化图表"""
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 1. 同位素分布条形图
    ax1 = axes[0, 0]
    isotopes = list(isotope_counts.index)
    counts = list(isotope_counts.values)
    colors = ['red' if iso == '18F' else 'blue' for iso in isotopes]
    
    bars = ax1.bar(isotopes, counts, color=colors, alpha=0.7)
    ax1.set_title('同位素分布', fontsize=14, fontweight='bold')
    ax1.set_xlabel('同位素类型')
    ax1.set_ylabel('samples数量')
    ax1.grid(True, alpha=0.3)
    
    # 添加数值标签
    for bar, count in zip(bars, counts):
        ax1.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                f'{count}', ha='center', va='bottom')
    
    # 2. 18F vs 非18F饼图
    ax2 = axes[0, 1]
    sizes = [n_18F, n_non_18F]
    labels = [f'18F\n({n_18F}个, {n_18F/(n_18F+n_non_18F)*100:.1f}%)', 
              f'非18F\n({n_non_18F}个, {n_non_18F/(n_18F+n_non_18F)*100:.1f}%)']
    colors = ['#ff9999', '#66b3ff']
    explode = (0.05, 0.05)
    
    ax2.pie(sizes, explode=explode, labels=labels, colors=colors,
            autopct='%1.0f%%', shadow=True, startangle=90)
    ax2.set_title('18F vs 非18F 分布', fontsize=14, fontweight='bold')
    
    # 3. 采样策略对比
    ax3 = axes[1, 0]
    strategies = ['original', '过采样', '欠采样']
    n_18F_list = [n_18F, max(n_18F, n_non_18F), min(n_18F, n_non_18F)]
    n_non_18F_list = [n_non_18F, max(n_18F, n_non_18F), min(n_18F, n_non_18F)]
    
    x = np.arange(len(strategies))
    width = 0.35
    
    bars1 = ax3.bar(x - width/2, n_18F_list, width, label='18F', color='#ff9999')
    bars2 = ax3.bar(x + width/2, n_non_18F_list, width, label='非18F', color='#66b3ff')
    
    ax3.set_xlabel('采样策略')
    ax3.set_ylabel('samples数量')
    ax3.set_title('不同采样策略的samples数量', fontsize=14, fontweight='bold')
    ax3.set_xticks(x)
    ax3.set_xticklabels(strategies)
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 添加数值标签
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height)}', ha='center', va='bottom')
    
    # 4. 数据集大小变化
    ax4 = axes[1, 1]
    total_original = n_18F + n_non_18F
    total_oversample = max(n_18F, n_non_18F) * 2
    total_undersample = min(n_18F, n_non_18F) * 2
    
    strategies = ['original', '过采样', '欠采样']
    totals = [total_original, total_oversample, total_undersample]
    percentages = [100, total_oversample/total_original*100, total_undersample/total_original*100]
    
    bars = ax4.bar(strategies, totals, color=['green', 'orange', 'purple'], alpha=0.7)
    ax4.set_ylabel('Total samples数')
    ax4.set_title('数据集大小变化', fontsize=14, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    
    # 添加百分比标签
    for bar, total, pct in zip(bars, totals, percentages):
        ax4.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                f'{int(total)}\n({pct:.0f}%)', ha='center', va='bottom')
    
    # 添加基准线
    ax4.axhline(y=total_original, color='red', linestyle='--', alpha=0.5, label='original大小')
    ax4.legend()
    
    plt.suptitle('PETBD数据集 - 18F采样分析', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('18F_sampling_visualization.png', dpi=300, bbox_inches='tight')
    print(f"\n可视化图表已保存: 18F_sampling_visualization.png")

if __name__ == '__main__':
    df_result = analyze_dataset()

