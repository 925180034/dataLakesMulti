#!/usr/bin/env python3
"""
完整提取WebTable和OpenData数据集
确保：
1. 提取所有ground truth中涉及的表
2. 只保留有ground truth的查询
3. 生成详细统计信息
"""

import json
import pandas as pd
import os
import csv
import random
from pathlib import Path
from typing import Dict, Any, List, Set
import numpy as np

def read_csv_table(filepath: str, max_samples: int = 5, max_rows: int = 1000) -> Dict[str, Any]:
    """读取CSV表格并提取基本信息和样例值"""
    try:
        # 获取文件名作为表名
        table_name = os.path.basename(filepath)
        
        # 读取CSV文件（只读前max_rows行用于推断类型和采样）
        df = pd.read_csv(filepath, nrows=max_rows, on_bad_lines='skip', encoding='utf-8', low_memory=False)
        
        if df.empty:
            return None
        
        # 记录实际行列数
        actual_rows = len(df)
        actual_cols = len(df.columns)
        
        # 构建列信息
        columns = []
        for col in df.columns:
            col_data = df[col].dropna()
            
            # 推断数据类型（简化版，跳过日期推断以提高速度）
            dtype = 'string'  # 默认类型
            if len(col_data) > 0:
                # 尝试转换为数值
                try:
                    pd.to_numeric(col_data)
                    dtype = 'numeric'
                except:
                    dtype = 'string'
            
            # 采样值（去重后采样）
            unique_values = col_data.unique()
            if len(unique_values) > max_samples:
                sample_values = [str(v) for v in random.sample(list(unique_values), max_samples)]
            else:
                sample_values = [str(v) for v in unique_values[:max_samples]]
            
            columns.append({
                'name': str(col),
                'type': dtype,
                'sample_values': sample_values
            })
        
        return {
            'table_name': table_name,
            'columns': columns,
            'num_rows': actual_rows,
            'num_columns': actual_cols
        }
    
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        return None

def process_webtable(task: str, source_dir: str, target_dir: str, subset_size: int = 100):
    """处理WebTable数据集"""
    
    print(f"\n处理 WebTable {task.upper()} 数据...")
    
    task_dir = os.path.join(source_dir, task)
    
    # 读取ground truth
    gt_file = os.path.join(task_dir, f"webtable_{task}_ground_truth.csv")
    gt_df = pd.read_csv(gt_file)
    
    # 读取query
    query_file = os.path.join(task_dir, f"webtable_{task}_query.csv")
    query_df = pd.read_csv(query_file)
    
    # 获取所有需要的表（查询表 + 候选表）
    all_tables_needed = set(query_df['query_table'].unique())
    all_tables_needed.update(gt_df['candidate_table'].unique())
    
    print(f"  需要提取的表格总数: {len(all_tables_needed)}")
    
    # 只保留有ground truth的查询
    queries_with_gt = set(gt_df['query_table'].unique())
    valid_queries = [q for q in query_df['query_table'].unique() if q in queries_with_gt]
    
    print(f"  有效查询数（有GT）: {len(valid_queries)}")
    
    # 读取表格
    tables_dir = os.path.join(task_dir, "tables")
    tables_data = []
    table_map = {}
    
    for table_name in all_tables_needed:
        table_path = os.path.join(tables_dir, table_name)
        if os.path.exists(table_path):
            table_info = read_csv_table(table_path)
            if table_info:
                tables_data.append(table_info)
                table_map[table_name] = table_info
    
    print(f"  成功读取表格数: {len(tables_data)}")
    
    # 创建queries列表（只包含有GT的）
    queries = []
    for table_name in valid_queries:
        if table_name in table_map:
            queries.append({
                "query_table": table_name,
                "task_type": task
            })
    
    # 创建ground truth列表
    ground_truth = []
    valid_query_set = set(valid_queries)
    for _, row in gt_df.iterrows():
        if row['query_table'] in valid_query_set and row['query_table'] in table_map and row['candidate_table'] in table_map:
            ground_truth.append({
                "query_table": row['query_table'],
                "candidate_table": row['candidate_table']
            })
    
    # 创建subset和complete版本
    subset_queries = queries[:subset_size] if len(queries) > subset_size else queries
    subset_query_tables = set([q['query_table'] for q in subset_queries])
    
    # subset的ground truth和tables
    subset_gt = [gt for gt in ground_truth if gt['query_table'] in subset_query_tables]
    subset_tables_needed = subset_query_tables.copy()
    for gt in subset_gt:
        subset_tables_needed.add(gt['candidate_table'])
    subset_tables = [t for t in tables_data if t['table_name'] in subset_tables_needed]
    
    # 保存subset
    subset_dir = os.path.join(target_dir, f"{task}_subset")
    os.makedirs(subset_dir, exist_ok=True)
    
    with open(os.path.join(subset_dir, "queries.json"), 'w') as f:
        json.dump(subset_queries, f, indent=2)
    
    with open(os.path.join(subset_dir, "ground_truth.json"), 'w') as f:
        json.dump(subset_gt, f, indent=2)
    
    with open(os.path.join(subset_dir, "tables.json"), 'w') as f:
        json.dump(subset_tables, f, indent=2)
    
    print(f"  Subset: {len(subset_queries)} queries, {len(subset_tables)} tables, {len(subset_gt)} GT")
    
    # 保存complete
    complete_dir = os.path.join(target_dir, f"{task}_complete")
    os.makedirs(complete_dir, exist_ok=True)
    
    with open(os.path.join(complete_dir, "queries.json"), 'w') as f:
        json.dump(queries, f, indent=2)
    
    with open(os.path.join(complete_dir, "ground_truth.json"), 'w') as f:
        json.dump(ground_truth, f, indent=2)
    
    with open(os.path.join(complete_dir, "tables.json"), 'w') as f:
        json.dump(tables_data, f, indent=2)
    
    print(f"  Complete: {len(queries)} queries, {len(tables_data)} tables, {len(ground_truth)} GT")
    
    # 计算统计信息
    stats = {
        "subset": {
            "queries": len(subset_queries),
            "tables": len(subset_tables),
            "ground_truth": len(subset_gt),
            "coverage": 100.0,  # 因为只保留有GT的查询
            "rows": [t['num_rows'] for t in subset_tables],
            "columns": [t['num_columns'] for t in subset_tables]
        },
        "complete": {
            "queries": len(queries),
            "tables": len(tables_data),
            "ground_truth": len(ground_truth),
            "coverage": 100.0,
            "rows": [t['num_rows'] for t in tables_data],
            "columns": [t['num_columns'] for t in tables_data]
        }
    }
    
    return stats

def process_opendata(task: str, source_dir: str, target_dir: str, subset_size: int = 100):
    """处理OpenData数据集"""
    
    print(f"\n处理 OpenData {task.upper()} 数据...")
    
    task_dir = os.path.join(source_dir, task)
    
    # 读取ground truth
    gt_file = os.path.join(task_dir, f"opendata_{task}_ground_truth.csv")
    gt_df = pd.read_csv(gt_file)
    
    # 读取query
    query_file = os.path.join(task_dir, f"opendata_{task}_query.csv")
    query_df = pd.read_csv(query_file)
    
    # 获取所有需要的表
    all_tables_needed = set(query_df['query_table'].unique())
    all_tables_needed.update(gt_df['candidate_table'].unique())
    
    print(f"  需要提取的表格总数: {len(all_tables_needed)}")
    
    # 只保留有ground truth的查询
    queries_with_gt = set(gt_df['query_table'].unique())
    valid_queries = [q for q in query_df['query_table'].unique() if q in queries_with_gt]
    
    print(f"  有效查询数（有GT）: {len(valid_queries)}")
    
    # 确定表格目录
    if task == "join":
        tables_dir = os.path.join(task_dir, "tables")
    else:  # union
        tables_dir = os.path.join(task_dir, "opendata_union_query")
    
    # 读取表格
    tables_data = []
    table_map = {}
    missing_count = 0
    
    for table_name in all_tables_needed:
        table_path = os.path.join(tables_dir, table_name)
        if os.path.exists(table_path):
            table_info = read_csv_table(table_path)
            if table_info:
                tables_data.append(table_info)
                table_map[table_name] = table_info
        else:
            missing_count += 1
    
    print(f"  成功读取表格数: {len(tables_data)}")
    if missing_count > 0:
        print(f"  缺失表格数: {missing_count}")
    
    # 创建queries列表
    queries = []
    for table_name in valid_queries:
        if table_name in table_map:
            queries.append({
                "query_table": table_name,
                "task_type": task
            })
    
    # 创建ground truth列表
    ground_truth = []
    valid_query_set = set([q['query_table'] for q in queries])
    for _, row in gt_df.iterrows():
        if row['query_table'] in valid_query_set and row['query_table'] in table_map and row['candidate_table'] in table_map:
            ground_truth.append({
                "query_table": row['query_table'],
                "candidate_table": row['candidate_table']
            })
    
    # 创建subset和complete版本
    subset_queries = queries[:subset_size] if len(queries) > subset_size else queries
    subset_query_tables = set([q['query_table'] for q in subset_queries])
    
    # subset的ground truth和tables
    subset_gt = [gt for gt in ground_truth if gt['query_table'] in subset_query_tables]
    subset_tables_needed = subset_query_tables.copy()
    for gt in subset_gt:
        subset_tables_needed.add(gt['candidate_table'])
    subset_tables = [t for t in tables_data if t['table_name'] in subset_tables_needed]
    
    # 保存subset
    subset_dir = os.path.join(target_dir, f"{task}_subset")
    os.makedirs(subset_dir, exist_ok=True)
    
    with open(os.path.join(subset_dir, "queries.json"), 'w') as f:
        json.dump(subset_queries, f, indent=2)
    
    with open(os.path.join(subset_dir, "ground_truth.json"), 'w') as f:
        json.dump(subset_gt, f, indent=2)
    
    with open(os.path.join(subset_dir, "tables.json"), 'w') as f:
        json.dump(subset_tables, f, indent=2)
    
    print(f"  Subset: {len(subset_queries)} queries, {len(subset_tables)} tables, {len(subset_gt)} GT")
    
    # 保存complete
    complete_dir = os.path.join(target_dir, f"{task}_complete")
    os.makedirs(complete_dir, exist_ok=True)
    
    with open(os.path.join(complete_dir, "queries.json"), 'w') as f:
        json.dump(queries, f, indent=2)
    
    with open(os.path.join(complete_dir, "ground_truth.json"), 'w') as f:
        json.dump(ground_truth, f, indent=2)
    
    with open(os.path.join(complete_dir, "tables.json"), 'w') as f:
        json.dump(tables_data, f, indent=2)
    
    print(f"  Complete: {len(queries)} queries, {len(tables_data)} tables, {len(ground_truth)} GT")
    
    # 计算统计信息
    stats = {
        "subset": {
            "queries": len(subset_queries),
            "tables": len(subset_tables),
            "ground_truth": len(subset_gt),
            "coverage": 100.0,
            "rows": [t['num_rows'] for t in subset_tables],
            "columns": [t['num_columns'] for t in subset_tables]
        },
        "complete": {
            "queries": len(queries),
            "tables": len(tables_data),
            "ground_truth": len(ground_truth),
            "coverage": 100.0,
            "rows": [t['num_rows'] for t in tables_data],
            "columns": [t['num_columns'] for t in tables_data]
        }
    }
    
    return stats

def calculate_detailed_stats(stats_dict):
    """计算详细的统计信息"""
    for task_stats in stats_dict.values():
        for split_stats in task_stats.values():
            if 'rows' in split_stats:
                rows = split_stats['rows']
                cols = split_stats['columns']
                
                split_stats['rows_stats'] = {
                    'min': min(rows) if rows else 0,
                    'max': max(rows) if rows else 0,
                    'mean': round(np.mean(rows), 2) if rows else 0,
                    'std': round(np.std(rows), 2) if rows else 0
                }
                
                split_stats['columns_stats'] = {
                    'min': min(cols) if cols else 0,
                    'max': max(cols) if cols else 0,
                    'mean': round(np.mean(cols), 2) if cols else 0,
                    'std': round(np.std(cols), 2) if cols else 0
                }
                
                # 删除原始列表以节省空间
                del split_stats['rows']
                del split_stats['columns']
    
    return stats_dict

def main():
    """主函数"""
    
    # 清理旧的examples目录
    print("清理旧的examples目录...")
    os.system("rm -rf examples/webtable examples/opendata")
    
    # 数据源和目标
    webtable_source = "/root/autodl-tmp/datalakes/webtable"
    opendata_source = "/root/autodl-tmp/datalakes/opendata"
    webtable_target = "examples/webtable"
    opendata_target = "examples/opendata"
    
    # 创建目标目录
    os.makedirs(webtable_target, exist_ok=True)
    os.makedirs(opendata_target, exist_ok=True)
    
    all_stats = {}
    
    # 处理WebTable
    print("\n" + "="*60)
    print("处理 WebTable 数据集")
    print("="*60)
    
    webtable_stats = {}
    for task in ['join', 'union']:
        stats = process_webtable(task, webtable_source, webtable_target, subset_size=100)
        webtable_stats[task] = stats
    
    all_stats['webtable'] = calculate_detailed_stats(webtable_stats)
    
    # 处理OpenData
    print("\n" + "="*60)
    print("处理 OpenData 数据集")
    print("="*60)
    
    opendata_stats = {}
    for task in ['join', 'union']:
        stats = process_opendata(task, opendata_source, opendata_target, subset_size=100)
        opendata_stats[task] = stats
    
    all_stats['opendata'] = calculate_detailed_stats(opendata_stats)
    
    # 保存统计信息
    with open("dataset_complete_stats.json", 'w') as f:
        json.dump(all_stats, f, indent=2)
    
    print("\n" + "="*60)
    print("数据提取完成！")
    print("="*60)
    print("统计信息已保存到 dataset_complete_stats.json")
    
    # 打印汇总
    for dataset in all_stats:
        print(f"\n{dataset.upper()}:")
        for task in all_stats[dataset]:
            for split in all_stats[dataset][task]:
                s = all_stats[dataset][task][split]
                print(f"  {task} {split}: {s['queries']} queries, {s['tables']} tables, {s['ground_truth']} GT")

if __name__ == "__main__":
    main()