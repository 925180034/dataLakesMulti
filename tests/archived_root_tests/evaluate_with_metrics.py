#!/usr/bin/env python
"""
使用真实数据集评估优化工作流，输出标准评价指标
"""

import asyncio
import json
import time
from pathlib import Path
from typing import List, Dict, Any, Set, Tuple
from collections import defaultdict

from src.core.workflow import discover_data
from src.core.models import TableInfo
from src.utils.data_parser import parse_tables_data


def load_ground_truth(file_path: str) -> Dict[str, Set[str]]:
    """加载ground truth数据"""
    with open(file_path) as f:
        ground_truth_data = json.load(f)
    
    # 组织ground truth为查询表到候选表的映射
    ground_truth = defaultdict(set)
    for item in ground_truth_data:
        query_table = item['query_table']
        candidate_table = item['candidate_table']
        ground_truth[query_table].add(candidate_table)
    
    return dict(ground_truth)


def load_queries(queries_file: str, tables_file: str) -> List[Dict[str, Any]]:
    """加载查询数据"""
    with open(queries_file) as f:
        queries_data = json.load(f)
    
    # 加载所有表数据以获取表结构
    with open(tables_file) as f:
        all_tables_data = json.load(f)
    
    # 创建表名到表数据的映射
    table_map = {table['table_name']: table for table in all_tables_data}
    
    # 提取唯一的查询表
    unique_query_tables = set()
    for query in queries_data:
        unique_query_tables.add(query['query_table'])
    
    # 创建查询列表
    queries = []
    for table_name in unique_query_tables:
        if table_name in table_map:
            queries.append({
                'table_name': table_name,
                'table_data': table_map[table_name]
            })
    
    return queries


def calculate_metrics(predictions: Dict[str, Set[str]], ground_truth: Dict[str, Set[str]]) -> Dict[str, float]:
    """计算评价指标：Precision, Recall, F1"""
    true_positives = 0
    false_positives = 0
    false_negatives = 0
    
    all_query_tables = set(predictions.keys()) | set(ground_truth.keys())
    
    for query_table in all_query_tables:
        pred_set = predictions.get(query_table, set())
        true_set = ground_truth.get(query_table, set())
        
        # 计算交集（True Positives）
        tp = len(pred_set & true_set)
        true_positives += tp
        
        # 计算False Positives（预测了但不在ground truth中）
        fp = len(pred_set - true_set)
        false_positives += fp
        
        # 计算False Negatives（在ground truth中但没有预测）
        fn = len(true_set - pred_set)
        false_negatives += fn
    
    # 计算指标
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'true_positives': true_positives,
        'false_positives': false_positives,
        'false_negatives': false_negatives
    }


async def evaluate_workflow(
    queries: List[Dict[str, Any]],
    all_tables_data: List[Dict[str, Any]],
    ground_truth: Dict[str, Set[str]],
    use_optimized: bool = True,
    top_k: int = 10
) -> Dict[str, Any]:
    """评估工作流性能"""
    predictions = defaultdict(set)
    total_time = 0
    query_times = []
    
    print(f"\n{'='*60}")
    print(f"评估 {'优化' if use_optimized else '基础'} 工作流")
    print(f"{'='*60}")
    print(f"查询数量: {len(queries)}")
    print(f"Top-K: {top_k}")
    
    # 对每个查询进行预测
    for i, query_info in enumerate(queries):
        table_name = query_info['table_name']
        table_data = query_info['table_data']
        
        print(f"\r处理查询 {i+1}/{len(queries)}: {table_name}", end='', flush=True)
        
        start_time = time.time()
        
        try:
            # 执行数据发现
            result = await discover_data(
                user_query=f"Find joinable tables for {table_name}",
                query_tables=[table_data],
                all_tables_data=all_tables_data if use_optimized and i == 0 else None,  # 只在第一次初始化
                use_optimized=use_optimized
            )
            
            # 确保result是AgentState对象
            if isinstance(result, dict):
                from src.core.models import AgentState
                result = AgentState(**result)
            
            # 提取预测结果
            if hasattr(result, 'table_matches') and result.table_matches:
                # 按分数排序并取top-k
                sorted_matches = sorted(result.table_matches, key=lambda x: x.score, reverse=True)
                for match in sorted_matches[:top_k]:
                    predictions[table_name].add(match.target_table)
            
        except Exception as e:
            print(f"\n错误处理查询 {table_name}: {e}")
            continue
        
        end_time = time.time()
        query_time = end_time - start_time
        query_times.append(query_time)
        total_time += query_time
    
    print("\n")
    
    # 计算指标
    metrics = calculate_metrics(predictions, ground_truth)
    
    # 计算时间统计
    avg_time = sum(query_times) / len(query_times) if query_times else 0
    
    return {
        'metrics': metrics,
        'timing': {
            'total_time': total_time,
            'average_time': avg_time,
            'query_count': len(queries)
        },
        'predictions': dict(predictions)
    }


async def main():
    """主评估函数"""
    print("🚀 数据湖多智能体系统评估")
    print("="*60)
    
    # 文件路径
    tables_file = Path("examples/final_subset_tables.json")
    queries_file = Path("examples/final_subset_queries.json")
    ground_truth_file = Path("examples/final_subset_ground_truth.json")
    
    # 检查文件存在
    for file_path in [tables_file, queries_file, ground_truth_file]:
        if not file_path.exists():
            print(f"❌ 错误：找不到文件 {file_path}")
            return
    
    # 加载数据
    print("📊 加载数据...")
    with open(tables_file) as f:
        all_tables_data = json.load(f)
    
    queries = load_queries(queries_file, tables_file)
    ground_truth = load_ground_truth(ground_truth_file)
    
    print(f"✅ 已加载:")
    print(f"   - {len(all_tables_data)} 个表")
    print(f"   - {len(queries)} 个查询")
    print(f"   - {len(ground_truth)} 个ground truth条目")
    
    # 评估基础工作流
    print("\n" + "="*60)
    print("📊 评估基础工作流")
    print("="*60)
    basic_results = await evaluate_workflow(
        queries[:10],  # 使用前10个查询进行测试
        all_tables_data,
        ground_truth,
        use_optimized=False,
        top_k=10
    )
    
    # 评估优化工作流
    print("\n" + "="*60)
    print("🚀 评估优化工作流")
    print("="*60)
    optimized_results = await evaluate_workflow(
        queries[:10],  # 使用相同的查询
        all_tables_data,
        ground_truth,
        use_optimized=True,
        top_k=10
    )
    
    # 输出结果对比
    print("\n" + "="*60)
    print("📊 评估结果对比")
    print("="*60)
    
    print("\n基础工作流:")
    print(f"  Precision: {basic_results['metrics']['precision']:.3f}")
    print(f"  Recall: {basic_results['metrics']['recall']:.3f}")
    print(f"  F1-Score: {basic_results['metrics']['f1']:.3f}")
    print(f"  平均查询时间: {basic_results['timing']['average_time']:.2f}秒")
    
    print("\n优化工作流:")
    print(f"  Precision: {optimized_results['metrics']['precision']:.3f}")
    print(f"  Recall: {optimized_results['metrics']['recall']:.3f}")
    print(f"  F1-Score: {optimized_results['metrics']['f1']:.3f}")
    print(f"  平均查询时间: {optimized_results['timing']['average_time']:.2f}秒")
    
    # 计算提升
    speedup = basic_results['timing']['average_time'] / optimized_results['timing']['average_time']
    print(f"\n性能提升: {speedup:.1f}x")
    
    # 保存详细结果
    results = {
        'basic_workflow': basic_results,
        'optimized_workflow': optimized_results,
        'comparison': {
            'speedup': speedup,
            'precision_diff': optimized_results['metrics']['precision'] - basic_results['metrics']['precision'],
            'recall_diff': optimized_results['metrics']['recall'] - basic_results['metrics']['recall'],
            'f1_diff': optimized_results['metrics']['f1'] - basic_results['metrics']['f1']
        }
    }
    
    with open('evaluation_results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print("\n✅ 详细结果已保存到 evaluation_results.json")


if __name__ == "__main__":
    asyncio.run(main())