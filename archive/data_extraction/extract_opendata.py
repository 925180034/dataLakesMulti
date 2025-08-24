#!/usr/bin/env python
"""
提取和转换 OpenData 数据集
"""

import os
import csv
import json
import random
from pathlib import Path
from typing import List, Dict, Any
import pandas as pd
from collections import defaultdict

def read_csv_table(filepath: str, max_samples: int = 5) -> Dict[str, Any]:
    """读取 CSV 表格并提取元数据和样例值"""
    try:
        # 读取 CSV 文件
        df = pd.read_csv(filepath, low_memory=False, nrows=1000)  # 只读前1000行以提高速度
        
        columns = []
        for col_name in df.columns:
            col_data = df[col_name].dropna()
            
            # 推断数据类型
            if col_data.empty:
                data_type = "unknown"
                sample_values = []
            else:
                # 尝试转换为数值
                try:
                    pd.to_numeric(col_data)
                    data_type = "numeric"
                except:
                    # 尝试转换为日期
                    try:
                        pd.to_datetime(col_data)
                        data_type = "datetime"
                    except:
                        data_type = "string"
                
                # 获取样例值
                unique_values = col_data.unique()
                if len(unique_values) <= max_samples:
                    sample_values = [str(v) for v in unique_values]
                else:
                    # 随机选择样例
                    sample_values = [str(v) for v in random.sample(list(unique_values), max_samples)]
            
            columns.append({
                "column_name": str(col_name),
                "name": str(col_name),
                "data_type": data_type,
                "type": data_type,
                "sample_values": sample_values
            })
        
        table_name = os.path.basename(filepath)
        return {
            "table_name": table_name,
            "name": table_name,
            "columns": columns,
            "row_count": len(df),
            "file_path": filepath
        }
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        return None

def process_opendata_join(source_dir: str, target_dir: str, subset_size: int = 100):
    """处理 OpenData JOIN 数据集"""
    
    print(f"Processing OpenData JOIN from {source_dir}")
    
    # 读取 query CSV
    query_df = pd.read_csv(os.path.join(source_dir, "opendata_join_query.csv"))
    
    # 读取 ground truth CSV
    gt_df = pd.read_csv(os.path.join(source_dir, "opendata_join_ground_truth.csv"))
    
    # 获取所有唯一的查询表
    query_tables = query_df['query_table'].unique()
    
    # 获取所有涉及的表（查询表 + 候选表）
    all_tables_needed = set(query_tables)
    all_tables_needed.update(gt_df['candidate_table'].unique())
    
    # 读取表格数据
    tables_dir = os.path.join(source_dir, "tables")
    tables_data = []
    table_map = {}
    
    print(f"Reading {len(all_tables_needed)} tables...")
    for table_name in all_tables_needed:
        table_path = os.path.join(tables_dir, table_name)
        if os.path.exists(table_path):
            table_info = read_csv_table(table_path)
            if table_info:
                tables_data.append(table_info)
                table_map[table_name] = table_info
    
    print(f"Successfully read {len(tables_data)} tables")
    
    # 创建 queries 列表
    queries = []
    for table_name in query_tables:
        if table_name in table_map:
            queries.append({
                "query_table": table_name,
                "task_type": "join"
            })
    
    # 创建 ground truth 列表
    ground_truth = []
    for _, row in gt_df.iterrows():
        if row['query_table'] in table_map and row['candidate_table'] in table_map:
            ground_truth.append({
                "query_table": row['query_table'],
                "candidate_table": row['candidate_table'],
                "query_column": row.get('query_column', ''),
                "candidate_column": row.get('candidate_column', '')
            })
    
    # 创建 subset 和 complete 版本
    # Subset: 前 subset_size 个查询
    subset_queries = queries[:subset_size] if len(queries) > subset_size else queries
    subset_query_tables = set([q['query_table'] for q in subset_queries])
    
    # 筛选 subset 的 ground truth 和 tables
    subset_gt = [gt for gt in ground_truth if gt['query_table'] in subset_query_tables]
    subset_tables_needed = subset_query_tables.copy()
    for gt in subset_gt:
        subset_tables_needed.add(gt['candidate_table'])
    subset_tables = [t for t in tables_data if t['table_name'] in subset_tables_needed]
    
    # 保存 subset 数据
    subset_dir = os.path.join(target_dir, "join_subset")
    os.makedirs(subset_dir, exist_ok=True)
    
    with open(os.path.join(subset_dir, "queries.json"), 'w') as f:
        json.dump(subset_queries, f, indent=2)
    
    with open(os.path.join(subset_dir, "ground_truth.json"), 'w') as f:
        json.dump(subset_gt, f, indent=2)
    
    with open(os.path.join(subset_dir, "tables.json"), 'w') as f:
        json.dump(subset_tables, f, indent=2)
    
    print(f"Saved JOIN subset: {len(subset_queries)} queries, {len(subset_tables)} tables")
    
    # 保存 complete 数据
    complete_dir = os.path.join(target_dir, "join_complete")
    os.makedirs(complete_dir, exist_ok=True)
    
    with open(os.path.join(complete_dir, "queries.json"), 'w') as f:
        json.dump(queries, f, indent=2)
    
    with open(os.path.join(complete_dir, "ground_truth.json"), 'w') as f:
        json.dump(ground_truth, f, indent=2)
    
    with open(os.path.join(complete_dir, "tables.json"), 'w') as f:
        json.dump(tables_data, f, indent=2)
    
    print(f"Saved JOIN complete: {len(queries)} queries, {len(tables_data)} tables")
    
    return {
        "subset": {"queries": len(subset_queries), "tables": len(subset_tables), "ground_truth": len(subset_gt)},
        "complete": {"queries": len(queries), "tables": len(tables_data), "ground_truth": len(ground_truth)}
    }

def process_opendata_union(source_dir: str, target_dir: str, subset_size: int = 100):
    """处理 OpenData UNION 数据集"""
    
    print(f"Processing OpenData UNION from {source_dir}")
    
    # UNION的表格在 opendata_union_query 目录中
    tables_dir = os.path.join(source_dir, "opendata_union_query")
    if not os.path.exists(tables_dir):
        print("UNION tables not yet extracted, skipping...")
        return None
    
    # 读取 query CSV
    query_df = pd.read_csv(os.path.join(source_dir, "opendata_union_query.csv"))
    
    # 读取 ground truth CSV
    gt_df = pd.read_csv(os.path.join(source_dir, "opendata_union_ground_truth.csv"))
    
    # 获取所有唯一的查询表
    query_tables = query_df['query_table'].unique()
    
    # 获取所有涉及的表（查询表 + 候选表）
    all_tables_needed = set(query_tables)
    all_tables_needed.update(gt_df['candidate_table'].unique())
    
    # 读取表格数据
    tables_data = []
    table_map = {}
    
    print(f"Reading {len(all_tables_needed)} tables...")
    for table_name in all_tables_needed:
        table_path = os.path.join(tables_dir, table_name)
        if os.path.exists(table_path):
            table_info = read_csv_table(table_path)
            if table_info:
                tables_data.append(table_info)
                table_map[table_name] = table_info
    
    print(f"Successfully read {len(tables_data)} tables")
    
    # 创建 queries 列表
    queries = []
    for table_name in query_tables:
        if table_name in table_map:
            queries.append({
                "query_table": table_name,
                "task_type": "union"
            })
    
    # 创建 ground truth 列表
    ground_truth = []
    for _, row in gt_df.iterrows():
        if row['query_table'] in table_map and row['candidate_table'] in table_map:
            ground_truth.append({
                "query_table": row['query_table'],
                "candidate_table": row['candidate_table']
            })
    
    # 创建 subset 和 complete 版本
    # Subset: 前 subset_size 个查询
    subset_queries = queries[:subset_size] if len(queries) > subset_size else queries
    subset_query_tables = set([q['query_table'] for q in subset_queries])
    
    # 筛选 subset 的 ground truth 和 tables
    subset_gt = [gt for gt in ground_truth if gt['query_table'] in subset_query_tables]
    subset_tables_needed = subset_query_tables.copy()
    for gt in subset_gt:
        subset_tables_needed.add(gt['candidate_table'])
    subset_tables = [t for t in tables_data if t['table_name'] in subset_tables_needed]
    
    # 保存 subset 数据
    subset_dir = os.path.join(target_dir, "union_subset")
    os.makedirs(subset_dir, exist_ok=True)
    
    with open(os.path.join(subset_dir, "queries.json"), 'w') as f:
        json.dump(subset_queries, f, indent=2)
    
    with open(os.path.join(subset_dir, "ground_truth.json"), 'w') as f:
        json.dump(subset_gt, f, indent=2)
    
    with open(os.path.join(subset_dir, "tables.json"), 'w') as f:
        json.dump(subset_tables, f, indent=2)
    
    print(f"Saved UNION subset: {len(subset_queries)} queries, {len(subset_tables)} tables")
    
    # 保存 complete 数据
    complete_dir = os.path.join(target_dir, "union_complete")
    os.makedirs(complete_dir, exist_ok=True)
    
    with open(os.path.join(complete_dir, "queries.json"), 'w') as f:
        json.dump(queries, f, indent=2)
    
    with open(os.path.join(complete_dir, "ground_truth.json"), 'w') as f:
        json.dump(ground_truth, f, indent=2)
    
    with open(os.path.join(complete_dir, "tables.json"), 'w') as f:
        json.dump(tables_data, f, indent=2)
    
    print(f"Saved UNION complete: {len(queries)} queries, {len(tables_data)} tables")
    
    return {
        "subset": {"queries": len(subset_queries), "tables": len(subset_tables), "ground_truth": len(subset_gt)},
        "complete": {"queries": len(queries), "tables": len(tables_data), "ground_truth": len(ground_truth)}
    }

def main():
    """主函数"""
    
    opendata_source = "/root/autodl-tmp/datalakes/opendata"
    opendata_target = "examples/opendata"
    
    stats = {}
    
    # 处理 JOIN 数据
    join_stats = process_opendata_join(
        os.path.join(opendata_source, "join"),
        opendata_target,
        subset_size=100
    )
    if join_stats:
        stats['join'] = join_stats
    
    # 处理 UNION 数据（如果已解压）
    union_stats = process_opendata_union(
        os.path.join(opendata_source, "union"),
        opendata_target,
        subset_size=100
    )
    if union_stats:
        stats['union'] = union_stats
    
    # 保存统计信息
    with open("opendata_stats.json", 'w') as f:
        json.dump(stats, f, indent=2)
    
    print("\nOpenData extraction complete!")
    print(f"Statistics saved to opendata_stats.json")
    
    return stats

if __name__ == "__main__":
    main()