#!/usr/bin/env python
"""
三层架构消融实验脚本
测试L1（元数据过滤）、L2（向量搜索）、L3（LLM验证）各层的贡献
"""
import os
import sys
import json
import time
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Tuple
import numpy as np

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent))

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_dataset(task_type: str, dataset_type: str = 'subset') -> tuple:
    """加载数据集"""
    base_dir = Path(f'examples/separated_datasets/{task_type}_{dataset_type}')
    
    with open(base_dir / 'tables.json', 'r') as f:
        tables = json.load(f)
    with open(base_dir / 'queries.json', 'r') as f:
        queries = json.load(f)
    with open(base_dir / 'ground_truth.json', 'r') as f:
        ground_truth = json.load(f)
    
    return tables, queries, ground_truth


def convert_ground_truth_format(ground_truth_list: List[Dict]) -> Dict[str, List[str]]:
    """
    将ground truth转换为字典格式
    """
    query_to_candidates = {}
    
    for item in ground_truth_list:
        query_table = item.get('query_table', '')
        candidate_table = item.get('candidate_table', '')
        
        if query_table and candidate_table:
            # 过滤自匹配
            if query_table != candidate_table:
                if query_table not in query_to_candidates:
                    query_to_candidates[query_table] = []
                query_to_candidates[query_table].append(candidate_table)
    
    return query_to_candidates


def run_layer1_only(tables: List[Dict], queries: List[Dict]) -> Dict[str, List[str]]:
    """仅运行L1层（元数据过滤）"""
    from src.tools.metadata_filter import MetadataFilter
    
    logger.info("🔬 Running L1 only (Metadata Filter)")
    metadata_filter = MetadataFilter()
    predictions = {}
    
    for query in queries:
        query_table_name = query.get('query_table', '')
        
        # 查找查询表 - 适配不同的数据格式
        query_table = None
        for t in tables:
            table_name = t.get('name') or t.get('table_name', '')
            if table_name == query_table_name:
                # 确保表有正确的name字段
                if 'name' not in t:
                    t['name'] = table_name
                query_table = t
                break
        
        if not query_table:
            predictions[query_table_name] = []
            continue
        
        # L1: 元数据过滤
        candidates = metadata_filter.filter_by_column_overlap(
            query_table, tables, top_k=10
        )
        
        # 过滤自匹配
        predictions[query_table_name] = [
            c.get('name') or c.get('table_name', '') for c in candidates 
            if (c.get('name') or c.get('table_name', '')) != query_table_name
        ][:5]
    
    return predictions


def run_layer1_layer2(tables: List[Dict], queries: List[Dict]) -> Dict[str, List[str]]:
    """运行L1+L2层（元数据过滤 + 向量搜索）"""
    from src.tools.metadata_filter import MetadataFilter
    from src.tools.vector_search_tool import VectorSearchTool
    
    logger.info("🔬 Running L1+L2 (Metadata + Vector Search)")
    metadata_filter = MetadataFilter()
    vector_search = VectorSearchTool()
    
    # 确保所有表有name字段
    for t in tables:
        if 'name' not in t and 'table_name' in t:
            t['name'] = t['table_name']
    
    # 初始化向量索引
    vector_search.initialize_from_tables(tables)
    
    predictions = {}
    
    for query in queries:
        query_table_name = query.get('query_table', '')
        
        # 查找查询表 - 适配不同的数据格式
        query_table = None
        for t in tables:
            table_name = t.get('name') or t.get('table_name', '')
            if table_name == query_table_name:
                if 'name' not in t:
                    t['name'] = table_name
                query_table = t
                break
        
        if not query_table:
            predictions[query_table_name] = []
            continue
        
        # L1: 元数据过滤
        l1_candidates = metadata_filter.filter_by_column_overlap(
            query_table, tables, top_k=50
        )
        
        # L2: 向量搜索（在L1结果基础上）
        try:
            l2_results = vector_search.search_similar_tables(
                query_table_name, 
                top_k=10,
                candidate_tables=[c['name'] for c in l1_candidates]
            )
            
            # 过滤自匹配
            predictions[query_table_name] = [
                r.get('table_name') or r.get('name', '') for r in l2_results 
                if (r.get('table_name') or r.get('name', '')) != query_table_name
            ][:5]
        except Exception as e:
            logger.warning(f"Vector search failed for {query_table_name}: {e}")
            # 回退到L1结果
            predictions[query_table_name] = [
                c.get('name') or c.get('table_name', '') for c in l1_candidates 
                if (c.get('name') or c.get('table_name', '')) != query_table_name
            ][:5]
    
    return predictions


def run_full_pipeline(tables: List[Dict], queries: List[Dict], task_type: str) -> Dict[str, List[str]]:
    """运行完整的三层流水线（L1+L2+L3）"""
    from src.core.langgraph_workflow import DataLakeDiscoveryWorkflow
    
    logger.info("🔬 Running Full Pipeline (L1+L2+L3)")
    
    # 确保所有表有name字段
    for t in tables:
        if 'name' not in t and 'table_name' in t:
            t['name'] = t['table_name']
    
    workflow = DataLakeDiscoveryWorkflow()
    predictions = {}
    
    for query in queries:
        query_table_name = query.get('query_table', '')
        
        try:
            # 运行完整工作流
            result = workflow.run(
                query=f"find {task_type}able tables for {query_table_name}",
                tables=tables,
                task_type=task_type,
                query_table_name=query_table_name
            )
            
            if result.get('success'):
                # 获取预测结果并过滤自匹配
                predictions[query_table_name] = [
                    r['table_name'] for r in result.get('results', [])[:5]
                    if r['table_name'] != query_table_name
                ]
            else:
                predictions[query_table_name] = []
                
        except Exception as e:
            logger.error(f"Full pipeline failed for {query_table_name}: {e}")
            predictions[query_table_name] = []
    
    return predictions


def calculate_metrics(predictions: Dict[str, List[str]], 
                     ground_truth: Dict[str, List[str]]) -> Dict[str, float]:
    """计算评估指标"""
    hit_at_1 = 0
    hit_at_3 = 0
    hit_at_5 = 0
    
    total_precision = 0
    total_recall = 0
    total_f1 = 0
    valid_queries = 0
    
    for query_table, pred_tables in predictions.items():
        if query_table not in ground_truth:
            continue
        
        valid_queries += 1
        true_tables = set(ground_truth[query_table])
        
        # Hit@K metrics
        for k in [1, 3, 5]:
            top_k_predictions = set(pred_tables[:k])
            if top_k_predictions & true_tables:
                if k == 1:
                    hit_at_1 += 1
                elif k == 3:
                    hit_at_3 += 1
                elif k == 5:
                    hit_at_5 += 1
        
        # Precision, Recall, F1
        if pred_tables:
            predicted_set = set(pred_tables[:5])
            tp = len(predicted_set & true_tables)
            fp = len(predicted_set - true_tables)
            fn = len(true_tables - predicted_set)
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            total_precision += precision
            total_recall += recall
            total_f1 += f1
    
    if valid_queries > 0:
        return {
            'hit@1': hit_at_1 / valid_queries,
            'hit@3': hit_at_3 / valid_queries,
            'hit@5': hit_at_5 / valid_queries,
            'precision': total_precision / valid_queries,
            'recall': total_recall / valid_queries,
            'f1_score': total_f1 / valid_queries,
            'valid_queries': valid_queries
        }
    else:
        return {
            'hit@1': 0.0, 'hit@3': 0.0, 'hit@5': 0.0,
            'precision': 0.0, 'recall': 0.0, 'f1_score': 0.0,
            'valid_queries': 0
        }


def run_ablation_experiment(task_type: str, max_queries: int = 10):
    """运行消融实验"""
    logger.info(f"\n{'='*80}")
    logger.info(f"Running Ablation Experiment for {task_type.upper()} Task")
    logger.info(f"{'='*80}")
    
    # 加载数据
    tables, queries, ground_truth = load_dataset(task_type, 'subset')
    logger.info(f"Dataset: {len(tables)} tables, {len(queries)} queries")
    
    # 限制查询数量
    queries = queries[:max_queries]
    logger.info(f"Using {len(queries)} queries for experiment")
    
    # 转换ground truth格式
    gt_dict = convert_ground_truth_format(ground_truth)
    
    # 存储结果
    results = {}
    
    # 实验1: L1 only
    logger.info("\n" + "-"*60)
    start_time = time.time()
    predictions_l1 = run_layer1_only(tables, queries)
    time_l1 = time.time() - start_time
    metrics_l1 = calculate_metrics(predictions_l1, gt_dict)
    results['L1_only'] = {
        'metrics': metrics_l1,
        'time': time_l1,
        'avg_time': time_l1 / len(queries)
    }
    logger.info(f"L1 Only - Time: {time_l1:.2f}s, F1: {metrics_l1['f1_score']:.3f}")
    
    # 实验2: L1+L2
    logger.info("\n" + "-"*60)
    start_time = time.time()
    predictions_l12 = run_layer1_layer2(tables, queries)
    time_l12 = time.time() - start_time
    metrics_l12 = calculate_metrics(predictions_l12, gt_dict)
    results['L1+L2'] = {
        'metrics': metrics_l12,
        'time': time_l12,
        'avg_time': time_l12 / len(queries)
    }
    logger.info(f"L1+L2 - Time: {time_l12:.2f}s, F1: {metrics_l12['f1_score']:.3f}")
    
    # 实验3: L1+L2+L3 (Full)
    logger.info("\n" + "-"*60)
    # 确保不跳过LLM
    os.environ['SKIP_LLM'] = 'false'
    start_time = time.time()
    predictions_full = run_full_pipeline(tables, queries, task_type)
    time_full = time.time() - start_time
    metrics_full = calculate_metrics(predictions_full, gt_dict)
    results['L1+L2+L3'] = {
        'metrics': metrics_full,
        'time': time_full,
        'avg_time': time_full / len(queries)
    }
    logger.info(f"L1+L2+L3 - Time: {time_full:.2f}s, F1: {metrics_full['f1_score']:.3f}")
    
    return results


def print_comparison_table(all_results: Dict[str, Dict]):
    """打印对比表格"""
    print("\n" + "="*100)
    print("THREE-LAYER ABLATION EXPERIMENT RESULTS")
    print("="*100)
    
    for task_type, results in all_results.items():
        print(f"\n{task_type.upper()} Task Results:")
        print("-"*80)
        print(f"{'Layer Config':<15} {'Hit@1':<10} {'Hit@3':<10} {'Hit@5':<10} "
              f"{'Precision':<12} {'Recall':<10} {'F1-Score':<10} {'Time(s)':<10}")
        print("-"*80)
        
        for config in ['L1_only', 'L1+L2', 'L1+L2+L3']:
            if config in results:
                m = results[config]['metrics']
                t = results[config]['avg_time']
                print(f"{config:<15} {m['hit@1']:<10.3f} {m['hit@3']:<10.3f} "
                      f"{m['hit@5']:<10.3f} {m['precision']:<12.3f} "
                      f"{m['recall']:<10.3f} {m['f1_score']:<10.3f} {t:<10.2f}")
    
    print("\n" + "="*100)
    print("LAYER CONTRIBUTION ANALYSIS")
    print("="*100)
    
    for task_type, results in all_results.items():
        print(f"\n{task_type.upper()} Task - Incremental Improvements:")
        
        if 'L1_only' in results and 'L1+L2' in results:
            f1_l1 = results['L1_only']['metrics']['f1_score']
            f1_l12 = results['L1+L2']['metrics']['f1_score']
            improvement = (f1_l12 - f1_l1) * 100
            print(f"  L2 Contribution: {improvement:+.1f}% F1 improvement")
        
        if 'L1+L2' in results and 'L1+L2+L3' in results:
            f1_l12 = results['L1+L2']['metrics']['f1_score']
            f1_full = results['L1+L2+L3']['metrics']['f1_score']
            improvement = (f1_full - f1_l12) * 100
            print(f"  L3 Contribution: {improvement:+.1f}% F1 improvement")
        
        if 'L1_only' in results and 'L1+L2+L3' in results:
            f1_l1 = results['L1_only']['metrics']['f1_score']
            f1_full = results['L1+L2+L3']['metrics']['f1_score']
            total_improvement = (f1_full - f1_l1) * 100
            print(f"  Total Improvement: {total_improvement:+.1f}% F1 improvement")


def main():
    """主函数"""
    import argparse
    parser = argparse.ArgumentParser(description='三层架构消融实验')
    parser.add_argument('--task', choices=['join', 'union', 'both'], default='both',
                       help='任务类型')
    parser.add_argument('--max-queries', type=int, default=5,
                       help='最大查询数')
    parser.add_argument('--output', type=str, default=None,
                       help='输出文件路径')
    args = parser.parse_args()
    
    # 运行实验
    tasks = ['join', 'union'] if args.task == 'both' else [args.task]
    all_results = {}
    
    for task in tasks:
        results = run_ablation_experiment(task, args.max_queries)
        all_results[task] = results
    
    # 打印结果表格
    print_comparison_table(all_results)
    
    # 保存结果
    if args.output:
        output_path = Path(args.output)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = Path(f"experiment_results/ablation_{timestamp}.json")
    
    output_path.parent.mkdir(exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    logger.info(f"\n✅ Results saved to: {output_path}")


if __name__ == "__main__":
    main()