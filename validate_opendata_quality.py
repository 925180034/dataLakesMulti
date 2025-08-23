#!/usr/bin/env python3
"""
验证OpenData数据集的质量
确保：
1. 所有queries都有ground truth
2. 所有查询表都在tables中存在
3. 所有ground truth的候选表都在tables中存在
"""

import json
import pandas as pd
import os
from pathlib import Path

def validate_dataset(dataset_path: str, task: str):
    """验证单个数据集的质量"""
    print(f"\n{'='*60}")
    print(f"验证 {dataset_path} - {task}")
    print(f"{'='*60}")
    
    # 读取文件
    queries_file = os.path.join(dataset_path, "queries.json")
    tables_file = os.path.join(dataset_path, "tables.json")
    gt_file = os.path.join(dataset_path, "ground_truth.json")
    
    with open(queries_file, 'r') as f:
        queries = json.load(f)
    
    with open(tables_file, 'r') as f:
        tables = json.load(f)
    
    with open(gt_file, 'r') as f:
        ground_truth = json.load(f)
    
    # 获取所有表名
    table_names = set([t['table_name'] for t in tables])
    query_tables = set([q['query_table'] for q in queries])
    
    # 统计信息
    print(f"查询数: {len(queries)}")
    print(f"表格数: {len(tables)}")
    print(f"Ground Truth数: {len(ground_truth)}")
    
    # 检查1: 所有查询表都在tables中
    missing_query_tables = query_tables - table_names
    if missing_query_tables:
        print(f"❌ 缺失的查询表: {len(missing_query_tables)}")
        print(f"   示例: {list(missing_query_tables)[:5]}")
    else:
        print(f"✅ 所有查询表都在tables中")
    
    # 检查2: 所有查询都有ground truth
    queries_with_gt = set([gt['query_table'] for gt in ground_truth])
    queries_without_gt = query_tables - queries_with_gt
    if queries_without_gt:
        print(f"❌ 没有ground truth的查询: {len(queries_without_gt)}")
        print(f"   示例: {list(queries_without_gt)[:5]}")
    else:
        print(f"✅ 所有查询都有ground truth")
    
    # 检查3: ground truth中的候选表都在tables中
    candidate_tables = set([gt['candidate_table'] for gt in ground_truth])
    missing_candidate_tables = candidate_tables - table_names
    if missing_candidate_tables:
        print(f"❌ 缺失的候选表: {len(missing_candidate_tables)}")
        print(f"   示例: {list(missing_candidate_tables)[:5]}")
    else:
        print(f"✅ 所有候选表都在tables中")
    
    # 计算覆盖率
    coverage = len(queries_with_gt) / len(query_tables) if query_tables else 0
    print(f"\nGround Truth覆盖率: {coverage:.2%}")
    
    # 平均每个查询的ground truth数
    gt_per_query = {}
    for gt in ground_truth:
        qt = gt['query_table']
        if qt not in gt_per_query:
            gt_per_query[qt] = 0
        gt_per_query[qt] += 1
    
    if gt_per_query:
        avg_gt = sum(gt_per_query.values()) / len(gt_per_query)
        max_gt = max(gt_per_query.values())
        min_gt = min(gt_per_query.values())
        print(f"平均每查询GT数: {avg_gt:.2f} (最小: {min_gt}, 最大: {max_gt})")
    
    return {
        'queries': len(queries),
        'tables': len(tables),
        'ground_truth': len(ground_truth),
        'missing_query_tables': len(missing_query_tables),
        'queries_without_gt': len(queries_without_gt),
        'missing_candidate_tables': len(missing_candidate_tables),
        'coverage': coverage
    }

def validate_original_data():
    """验证原始数据的质量"""
    print("\n" + "="*80)
    print("验证原始OpenData数据质量")
    print("="*80)
    
    # JOIN数据验证
    print("\nJOIN原始数据:")
    join_path = "/root/autodl-tmp/datalakes/opendata/join"
    
    # 读取CSV文件
    gt_df = pd.read_csv(os.path.join(join_path, "opendata_join_ground_truth.csv"))
    query_df = pd.read_csv(os.path.join(join_path, "opendata_join_query.csv"))
    
    # 获取表格目录中的文件
    tables_dir = os.path.join(join_path, "tables")
    available_tables = set(os.listdir(tables_dir)) if os.path.exists(tables_dir) else set()
    
    print(f"Ground Truth行数: {len(gt_df)}")
    print(f"Query行数: {len(query_df)}")
    print(f"Tables目录文件数: {len(available_tables)}")
    
    # 统计唯一表
    query_tables = set(query_df['query_table'].unique())
    all_tables_in_gt = set(gt_df['query_table'].unique()) | set(gt_df['candidate_table'].unique())
    
    print(f"唯一查询表数: {len(query_tables)}")
    print(f"Ground Truth中的唯一表数: {len(all_tables_in_gt)}")
    print(f"应该提取的表数: {len(query_tables)}")  # 只需要查询表即可
    
    # UNION数据验证
    print("\nUNION原始数据:")
    union_path = "/root/autodl-tmp/datalakes/opendata/union"
    
    gt_df = pd.read_csv(os.path.join(union_path, "opendata_union_ground_truth.csv"))
    query_df = pd.read_csv(os.path.join(union_path, "opendata_union_query.csv"))
    
    # 获取表格目录中的文件
    tables_dir = os.path.join(union_path, "opendata_union_query")
    available_tables = set(os.listdir(tables_dir)) if os.path.exists(tables_dir) else set()
    
    print(f"Ground Truth行数: {len(gt_df)}")
    print(f"Query行数: {len(query_df)}")
    print(f"Tables目录文件数: {len(available_tables)}")
    
    # 统计唯一表
    query_tables = set(query_df['query_table'].unique())
    all_tables_in_gt = set(gt_df['query_table'].unique()) | set(gt_df['candidate_table'].unique())
    
    print(f"唯一查询表数: {len(query_tables)}")
    print(f"Ground Truth中的唯一表数: {len(all_tables_in_gt)}")
    print(f"应该提取的表数: {len(all_tables_in_gt)}")  # UNION需要所有表

def main():
    # 验证原始数据
    validate_original_data()
    
    print("\n" + "="*80)
    print("验证提取后的数据质量")
    print("="*80)
    
    # 验证WebTable
    stats = {}
    for dataset in ['webtable', 'opendata']:
        stats[dataset] = {}
        for task in ['join', 'union']:
            for split in ['subset', 'complete']:
                path = f"examples/{dataset}/{task}_{split}"
                if os.path.exists(path):
                    key = f"{task}_{split}"
                    stats[dataset][key] = validate_dataset(path, f"{dataset} {task} {split}")
    
    # 总结
    print("\n" + "="*80)
    print("质量总结")
    print("="*80)
    
    for dataset in stats:
        print(f"\n{dataset.upper()}:")
        for key in stats[dataset]:
            s = stats[dataset][key]
            status = "✅" if s['missing_query_tables'] == 0 and s['queries_without_gt'] == 0 else "❌"
            print(f"  {key}: {status} (覆盖率: {s['coverage']:.2%})")

if __name__ == "__main__":
    main()