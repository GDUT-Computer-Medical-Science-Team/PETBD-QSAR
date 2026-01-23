#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
综合模型结果分析脚本
统计所有机器学习模型的性能指标并生成详细报告
"""

import pandas as pd
import numpy as np
import os
import glob
import json
from pathlib import Path

def analyze_csv_results():
    """分析CSV结果文件"""
    result_files = []
    
    # 查找所有test结果CSV文件
    test_patterns = [
        'MachineLearningModels/**/result/*_test_results.csv',
        'MachineLearningModels/**/result/*_petbd_18F_results.csv',
    ]
    
    for pattern in test_patterns:
        result_files.extend(glob.glob(pattern, recursive=True))
    
    print(f"找到 {len(result_files)} 个测试结果CSV文件:")
    for file in result_files:
        print(f"  - {file}")
    
    model_results = []
    
    for file_path in result_files:
        try:
            # 读取CSV文件
            df = pd.read_csv(file_path)
            
            # 提取模型信息
            model_info = extract_csv_model_info(df, file_path)
            if model_info:
                model_results.append(model_info)
                print(f"Success处理: {os.path.basename(file_path)}")
                
        except Exception as e:
            print(f"处理文件 {file_path} 时出错: {e}")
    
    return model_results

def extract_csv_model_info(df, file_path):
    """从CSV数据中提取模型性能信息"""
    try:
        file_name = os.path.basename(file_path)
        
        # 确定数据集类型
        if 'logBB' in file_path.lower() or 'petbd' in file_name.lower():
            dataset = 'LogBB'
        elif 'cbrain' in file_path.lower():
            dataset = 'Cbrain'  
        else:
            dataset = 'Unknown'
        
        # 提取模型名称
        model_name = extract_model_name_from_csv(file_name)
        
        # 确定features类型
        if '_fp_' in file_name.lower() or '_fingerprint' in file_name.lower():
            feature_type = 'Fingerprint'
        elif '_mordred' in file_name.lower():
            feature_type = 'Mordred'
        elif 'petbd_18F' in file_name:
            feature_type = 'Combined'
        else:
            feature_type = 'Unknown'
        
        # 处理不同的列名格式
        actual_col = None
        predicted_col = None
        
        # 查找实际值列
        for col in df.columns:
            if any(keyword in col.lower() for keyword in ['actual', 'true', 'real']):
                actual_col = col
                break
        
        # 查找predicted values列  
        for col in df.columns:
            if any(keyword in col.lower() for keyword in ['predicted', 'pred', 'forecast']):
                predicted_col = col
                break
        
        # 计算性能指标
        if actual_col is not None and predicted_col is not None:
            actual = df[actual_col].values
            predicted = df[predicted_col].values
            
            # 计算基本指标
            mse = np.mean((actual - predicted) ** 2)
            rmse = np.sqrt(mse)
            mae = np.mean(np.abs(actual - predicted))
            
            # 计算R²
            ss_res = np.sum((actual - predicted) ** 2)
            ss_tot = np.sum((actual - np.mean(actual)) ** 2)
            r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
            
            # 计算调整R²
            n = len(actual)
            p = 50  # 假设features数为50
            adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1) if n > p + 1 else r2
            
            # 计算MAPE
            mape = np.mean(np.abs((actual - predicted) / actual)) * 100 if np.all(actual != 0) else np.nan
            
            model_info = {
                'file_name': file_name,
                'dataset': dataset,
                'model': model_name,
                'feature_type': feature_type,
                'n_samples': len(df),
                'r2': round(r2, 4),
                'rmse': round(rmse, 4),
                'mae': round(mae, 4),
                'adj_r2': round(adj_r2, 4),
                'mape': round(mape, 2) if not np.isnan(mape) else 'N/A',
                'mse': round(mse, 4)
            }
            
            return model_info
        else:
            print(f"文件 {file_name} 缺少必要的列，找到的列: {list(df.columns)}")
            return None
            
    except Exception as e:
        print(f"提取CSV模型信息时出错: {e}")
        return None

def extract_model_name_from_csv(file_name):
    """从CSV文件名提取模型名称"""
    file_lower = file_name.lower()
    
    if 'xgb' in file_lower or 'xgboost' in file_lower:
        return 'XGBoost'
    elif 'rf' in file_lower and 'random' not in file_lower:
        return 'RandomForest'  
    elif 'svm' in file_lower:
        return 'SVM'
    elif 'mlp' in file_lower:
        return 'MLP'
    elif 'lightgbm' in file_lower or 'lgb' in file_lower:
        return 'LightGBM'
    elif 'catboost' in file_lower:
        return 'CatBoost'
    else:
        return 'Unknown'

def create_comprehensive_summary(csv_results, json_results=None):
    """创建综合汇总报告"""
    
    # 合并结果
    all_results = csv_results.copy()
    if json_results:
        all_results.extend(json_results)
    
    if not all_results:
        print("没有找到可分析的结果")
        return None
    
    # 创建DataFrame
    df = pd.DataFrame(all_results)
    
    # 过滤掉无效数据
    df = df.dropna(subset=['r2', 'rmse', 'mae'])
    
    print(f"\n=== 综合模型性能汇总 ({len(df)} 个有效结果) ===")
    
    # 按性能排序
    df_sorted = df.sort_values(['dataset', 'r2'], ascending=[True, False])
    
    # 显示主要指标
    display_cols = ['model', 'dataset', 'feature_type', 'r2', 'rmse', 'mae', 'n_samples']
    print(df_sorted[display_cols].to_string(index=False))
    
    # 保存详细结果
    output_file = "comprehensive_model_results.csv"
    df_sorted.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"\n详细结果已保存到: {output_file}")
    
    # 生成分析报告
    generate_comprehensive_report(df_sorted)
    
    return df_sorted

def generate_comprehensive_report(df):
    """生成综合分析报告"""
    
    report_lines = [
        "# 机器学习模型综合性能分析报告",
        "",
        f"生成时间: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## 实验汇总",
        f"- 总实验数: {len(df)}",
        f"- 数据集: {', '.join(df['dataset'].unique())}",
        f"- 模型类型: {', '.join(df['model'].unique())}",
        f"- features类型: {', '.join(df['feature_type'].unique())}",
        "",
    ]
    
    # 最佳性能模型
    report_lines.extend([
        "## 最佳性能模型排行榜 (按R²降序)",
        ""
    ])
    
    top_models = df.nlargest(10, 'r2')[['model', 'dataset', 'feature_type', 'r2', 'rmse', 'mae', 'n_samples']]
    report_lines.extend([
        top_models.to_string(index=False),
        "",
    ])
    
    # 按数据集分组统计
    if len(df['dataset'].unique()) > 1:
        report_lines.extend([
            "## 按数据集分组的性能统计",
            ""
        ])
        
        dataset_stats = df.groupby('dataset').agg({
            'r2': ['mean', 'std', 'max', 'min', 'count'],
            'rmse': ['mean', 'std', 'min'],
            'mae': ['mean', 'std', 'min']
        }).round(4)
        
        report_lines.extend([
            dataset_stats.to_string(),
            "",
        ])
    
    # 按模型类型分组统计
    report_lines.extend([
        "## 按模型类型分组的性能统计",
        ""
    ])
    
    model_stats = df.groupby('model').agg({
        'r2': ['mean', 'std', 'max', 'min', 'count'],
        'rmse': ['mean', 'std', 'min'],
        'mae': ['mean', 'std', 'min']
    }).round(4)
    
    report_lines.extend([
        model_stats.to_string(),
        "",
    ])
    
    # 按features类型分组统计
    if len(df['feature_type'].unique()) > 1:
        report_lines.extend([
            "## 按features类型分组的性能统计",
            ""
        ])
        
        feature_stats = df.groupby('feature_type').agg({
            'r2': ['mean', 'std', 'max', 'min', 'count'],
            'rmse': ['mean', 'std', 'min'],
            'mae': ['mean', 'std', 'min']
        }).round(4)
        
        report_lines.extend([
            feature_stats.to_string(),
            "",
        ])
    
    # 模型-数据集交叉分析
    report_lines.extend([
        "## 模型在不同数据集上的表现对比",
        ""
    ])
    
    pivot_r2 = df.pivot_table(values='r2', index='model', columns='dataset', aggfunc='max').round(4)
    report_lines.extend([
        "### R² 得分:",
        pivot_r2.to_string(),
        "",
    ])
    
    pivot_rmse = df.pivot_table(values='rmse', index='model', columns='dataset', aggfunc='min').round(4)
    report_lines.extend([
        "### RMSE 得分:",
        pivot_rmse.to_string(),
        "",
    ])
    
    # 关键发现
    report_lines.extend([
        "## 关键发现",
        "",
        f"1. **最佳整体性能**: {df.loc[df['r2'].idxmax(), 'model']} 在 {df.loc[df['r2'].idxmax(), 'dataset']} 数据集上达到 R² = {df['r2'].max():.4f}",
        f"2. **平均最佳模型**: {df.groupby('model')['r2'].mean().idxmax()} (平均 R² = {df.groupby('model')['r2'].mean().max():.4f})",
        f"3. **最稳定模型**: {df.groupby('model')['r2'].std().idxmin()} (R² 标准差 = {df.groupby('model')['r2'].std().min():.4f})",
        "",
        "## 建议",
        "",
        "1. 重点关注性能最佳的模型组合进行进一步优化",
        "2. 分析features类型对模型性能的影响",
        "3. 考虑集成学习方法结合多个高性能模型",
        ""
    ])
    
    # 保存报告
    report_file = "comprehensive_analysis_report.md"
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report_lines))
    
    print(f"综合分析报告已保存到: {report_file}")

def main():
    """主函数"""
    print("Starting综合分析模型结果...")
    
    # 分析CSV结果文件
    csv_results = analyze_csv_results()
    
    # 创建综合汇总
    df = create_comprehensive_summary(csv_results)
    
    if df is not None:
        print(f"\n分析Complete! 共处理 {len(df)} 个模型结果")
    else:
        print("分析Failed，未找到有效结果")

if __name__ == "__main__":
    main()