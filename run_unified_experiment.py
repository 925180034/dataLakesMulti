#!/usr/bin/env python
"""
统一实验运行器 - 支持WebTable、SANTOS和NLCTables三个数据集
可以使用同一个系统运行所有数据集的实验
支持与three_layer_ablation_optimized.py相同的所有参数
"""

import os
import sys
import json
import time
import argparse
import logging
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from datetime import datetime

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def detect_dataset_type(tables_path: str) -> str:
    """自动检测数据集类型"""
    path_str = str(tables_path).lower()
    
    if 'nlctables' in path_str:
        return 'nlctables'
    elif 'santos' in path_str:
        return 'santos'
    elif 'webtable' in path_str or 'final' in path_str:
        return 'webtable'
    else:
        # 默认根据表名格式判断
        with open(tables_path, 'r') as f:
            tables = json.load(f)
            if tables and isinstance(tables[0], dict):
                first_table_name = tables[0].get('name', '')
                if first_table_name.startswith('dl_table_') or first_table_name.startswith('q_table_'):
                    return 'nlctables'
                elif 'santos' in first_table_name.lower():
                    return 'santos'
        return 'webtable'

def run_nlctables_experiment(layer: str, tables: List[Dict], queries: List[Dict], 
                            task_type: str, max_queries: int = None, 
                            max_workers: int = 4, challenging: bool = True) -> Tuple[List[Dict], float]:
    """运行NLCTables实验 - 使用主系统通过适配器"""
    logger.info(f"🔬 Running NLCTables experiment with layer {layer}")
    logger.info(f"  Task type: {task_type}")
    logger.info(f"  Input queries: {len(queries)}")
    
    # 使用主系统运行NLCTables
    from three_layer_ablation_optimized import run_layer_experiment
    
    # 处理查询数量 - 只在queries长度大于max_queries时限制
    if max_queries is not None and len(queries) > max_queries:
        queries = queries[:max_queries]
        logger.info(f"  Limited to {max_queries} queries")
    
    # TODO: 如果需要挑战性查询，可以在这里处理
    # if challenging:
    #     queries = create_challenging_queries(queries, tables)
    
    # 运行实验
    results_dict, elapsed_time = run_layer_experiment(
        layer=layer,
        tables=tables,
        queries=queries,
        task_type=task_type,
        dataset_type='nlctables',
        max_workers=max_workers
    )
    
    # 转换结果格式从字典到列表
    results = []
    if isinstance(results_dict, dict):
        for query_table, predictions in results_dict.items():
            results.append({
                'query_table': query_table,
                'predictions': predictions[:5] if isinstance(predictions, list) else []
            })
    else:
        results = results_dict
    
    logger.info(f"  Output results: {len(results)} entries")
    if results and len(results) > 0:
        logger.info(f"  First result: {results[0]['query_table']} -> {len(results[0].get('predictions', []))} predictions")
    
    return results, elapsed_time

def run_webtable_santos_experiment(layer: str, tables: List[Dict], queries: List[Dict],
                                  task_type: str, dataset_type: str, max_queries: int = None,
                                  max_workers: int = 4, challenging: bool = True) -> Tuple[List[Dict], float]:
    """运行WebTable/SANTOS实验 - 使用主系统"""
    logger.info(f"🔬 Running {dataset_type.upper()} experiment with layer {layer}")
    
    # 导入主系统
    from three_layer_ablation_optimized import run_layer_experiment
    
    # 处理查询数量
    if max_queries is not None:
        queries = queries[:max_queries]
    
    # TODO: 如果需要挑战性查询，可以在这里处理
    # if challenging:
    #     queries = create_challenging_queries(queries, tables)
    
    # 运行实验
    results_dict, elapsed_time = run_layer_experiment(
        layer=layer,
        tables=tables,
        queries=queries,
        task_type=task_type,
        dataset_type=dataset_type,
        max_workers=max_workers
    )
    
    # 转换结果格式从字典到列表
    results = []
    if isinstance(results_dict, dict):
        for query_table, predictions in results_dict.items():
            results.append({
                'query_table': query_table,
                'predictions': predictions[:5] if isinstance(predictions, list) else []
            })
    else:
        results = results_dict
    
    logger.info(f"  Output results: {len(results)} entries")
    if results and len(results) > 0:
        logger.info(f"  First result: {results[0]['query_table']} -> {len(results[0].get('predictions', []))} predictions")
    
    return results, elapsed_time

def evaluate_results(results: List[Dict], ground_truth, k_values: List[int] = [1, 3, 5]) -> Dict:
    """评估结果 - ground_truth can be Dict or List"""
    from src.utils.evaluation import calculate_hit_at_k, calculate_precision_recall_f1
    
    metrics = {}
    
    # 计算Hit@K
    for k in k_values:
        hit_rate = calculate_hit_at_k(results, ground_truth, k)
        metrics[f'hit@{k}'] = hit_rate
    
    # 计算Precision/Recall/F1
    pr_metrics = calculate_precision_recall_f1(results, ground_truth)
    metrics.update(pr_metrics)
    
    return metrics

def print_results_table(all_results: Dict, all_metrics: Dict):
    """打印结果统计表格"""
    # 分离JOIN和UNION结果
    join_results = {}
    union_results = {}
    
    for exp_key, metrics in all_metrics.items():
        exp_data = all_results[exp_key]
        task = exp_data['task']
        layer = exp_data['layer']
        elapsed_time = exp_data['elapsed_time']
        
        result_data = {
            'layer': layer,
            'hit@1': metrics.get('hit@1', 0.0),
            'hit@3': metrics.get('hit@3', 0.0),
            'hit@5': metrics.get('hit@5', 0.0),
            'precision': metrics.get('precision', 0.0),
            'recall': metrics.get('recall', 0.0),
            'f1': metrics.get('f1', 0.0),
            'time': elapsed_time
        }
        
        if task == 'join':
            join_results[layer] = result_data
        elif task == 'union':
            union_results[layer] = result_data
    
    # 打印JOIN结果表格
    if join_results:
        print("\nJOIN Task Results:")
        print("-" * 116)
        print(f"{'Layer Config':<15} {'Hit@1':<10} {'Hit@3':<10} {'Hit@5':<10} {'Precision':<12} {'Recall':<10} {'F1-Score':<10} {'Time(s)':<10}")
        print("-" * 116)
        
        # 按L1, L1+L2, L1+L2+L3顺序排序
        layer_order = ['L1', 'L1+L2', 'L1+L2+L3']
        for layer in layer_order:
            if layer in join_results:
                data = join_results[layer]
                print(f"{layer:<15} {data['hit@1']:<10.3f} {data['hit@3']:<10.3f} {data['hit@5']:<10.3f} "
                      f"{data['precision']:<12.3f} {data['recall']:<10.3f} {data['f1']:<10.3f} {data['time']:<10.2f}")
    
    # 打印UNION结果表格
    if union_results:
        print("\nUNION Task Results:")
        print("-" * 116)
        print(f"{'Layer Config':<15} {'Hit@1':<10} {'Hit@3':<10} {'Hit@5':<10} {'Precision':<12} {'Recall':<10} {'F1-Score':<10} {'Time(s)':<10}")
        print("-" * 116)
        
        # 按L1, L1+L2, L1+L2+L3顺序排序
        for layer in layer_order:
            if layer in union_results:
                data = union_results[layer]
                print(f"{layer:<15} {data['hit@1']:<10.3f} {data['hit@3']:<10.3f} {data['hit@5']:<10.3f} "
                      f"{data['precision']:<12.3f} {data['recall']:<10.3f} {data['f1']:<10.3f} {data['time']:<10.2f}")

def main():
    parser = argparse.ArgumentParser(description='统一实验运行器')
    parser.add_argument('--dataset', type=str, required=True,
                       help='数据集路径或名称 (webtable/santos/nlctables)')
    parser.add_argument('--task', type=str, choices=['join', 'union', 'both'], default='join',
                       help='任务类型 (both会同时运行join和union)')
    parser.add_argument('--layer', type=str, choices=['L1', 'L1+L2', 'L1+L2+L3', 'all'], 
                       default='L1+L2+L3', help='运行的层级 (all运行所有层级)')
    parser.add_argument('--dataset-type', choices=['subset', 'complete', 'true_subset'], default='subset',
                       help='数据集类型: subset(子集), complete(完整), true_subset(WebTable的真子集)')
    parser.add_argument('--max-queries', type=str, default='10',
                       help='最大查询数 (数字或"all"表示使用全部)')
    parser.add_argument('--workers', type=int, default=4,
                       help='并行进程数')
    parser.add_argument('--challenging', action='store_true', default=True,
                       help='使用挑战性混合查询（默认启用）')
    parser.add_argument('--simple', action='store_true',
                       help='使用简单原始查询（禁用挑战性查询）')
    parser.add_argument('--output', type=str, default=None,
                       help='输出文件路径')
    parser.add_argument('--verbose', action='store_true',
                       help='显示详细输出')
    parser.add_argument('--skip-llm', action='store_true',
                       help='跳过LLM层（仅用于测试）')
    
    args = parser.parse_args()
    
    # 处理max_queries参数
    if args.max_queries.lower() in ['all', '-1', 'none']:
        max_queries = None  # None表示使用所有查询
        if args.verbose:
            print(f"📊 使用整个数据集的所有查询")
    else:
        try:
            max_queries = int(args.max_queries)
            if args.verbose:
                print(f"📊 限制最大查询数为: {max_queries}")
        except ValueError:
            print(f"⚠️ 无效的max-queries值: {args.max_queries}，使用默认值10")
            max_queries = 10
    
    # 处理simple/challenging冲突
    if args.simple:
        args.challenging = False
    
    # 设置环境变量（如果需要跳过LLM）
    if args.skip_llm:
        os.environ['SKIP_LLM'] = 'true'
    else:
        os.environ['SKIP_LLM'] = 'false'
    
    # 确定数据集路径
    if args.dataset in ['webtable', 'santos', 'nlctables']:
        # 使用预定义路径，根据dataset-type选择
        if args.dataset == 'webtable':
            if args.dataset_type == 'subset' or args.dataset_type == 'true_subset':
                tables_path = 'examples/final_subset_tables.json'
            elif args.dataset_type == 'complete':
                tables_path = 'examples/final_complete_tables.json'
        elif args.dataset == 'santos':
            if args.dataset_type == 'subset':
                tables_path = 'examples/santos_subset/tables.json'
            elif args.dataset_type == 'complete':
                tables_path = 'examples/santos_complete/tables.json'  # 如果存在
        elif args.dataset == 'nlctables':
            # NLCTables路径格式：examples/nlctables/{task}_{dataset_type}/
            tables_path = f'examples/nlctables/{args.task}_{args.dataset_type}/tables.json'
        
        dataset_type = args.dataset
    else:
        # 使用提供的路径
        tables_path = args.dataset
        dataset_type = detect_dataset_type(tables_path)
    
    logger.info(f"📊 Dataset type detected: {dataset_type}")
    
    # 初始化ground_truth_path（即使对NLCTables也需要）
    base_dir = Path(tables_path).parent
    ground_truth_path = base_dir / 'ground_truth.json'
    
    # 加载数据 - 对NLCTables使用适配器
    if dataset_type == 'nlctables':
        # 使用适配器加载NLCTables数据
        from nlctables_adapter import NLCTablesAdapter
        adapter = NLCTablesAdapter()
        
        # 使用参数中的dataset_type
        subset_type = args.dataset_type
        
        # 如果task是both，先加载join的数据（后面会根据需要重新加载）
        initial_task = 'join' if args.task == 'both' else args.task
        
        tables, queries, ground_truth_list = adapter.load_nlctables_dataset(initial_task, subset_type)
        ground_truth = ground_truth_list  # 适配器已经转换为列表格式
        logger.info(f"Loaded {len(tables)} tables via NLCTables adapter")
        logger.info(f"Loaded {len(queries)} queries")
    else:
        # 原有的加载逻辑（WebTable/SANTOS）
        with open(tables_path, 'r') as f:
            tables = json.load(f)
        logger.info(f"Loaded {len(tables)} tables")
        
        # 加载查询
        queries_path = base_dir / 'queries.json'
        
        if queries_path.exists():
            with open(queries_path, 'r') as f:
                queries = json.load(f)
            logger.info(f"Loaded {len(queries)} queries")
        else:
            # 生成默认查询
            queries = [{'query_table': t['name'], 'task_type': args.task} for t in tables[:10]]
            logger.warning("No queries file found, using first 10 tables as queries")
    
    # 确定要运行的任务列表
    if args.task == 'both':
        tasks_to_run = ['join', 'union']
    else:
        tasks_to_run = [args.task]
    
    # 确定要运行的层级列表
    if args.layer == 'all':
        layers_to_run = ['L1', 'L1+L2', 'L1+L2+L3']
    else:
        layers_to_run = [args.layer]
    
    # 运行所有组合的实验
    all_results = {}
    for task in tasks_to_run:
        for layer in layers_to_run:
            logger.info(f"\n{'='*60}")
            logger.info(f"🚀 运行实验: 任务={task}, 层级={layer}")
            logger.info(f"{'='*60}")
            
            # 根据数据集类型运行实验
            if dataset_type == 'nlctables':
                # NLCTables需要为不同任务重新加载数据
                if task != initial_task:
                    # 重新加载对应任务的数据
                    from nlctables_adapter import NLCTablesAdapter
                    adapter = NLCTablesAdapter()
                    current_tables, current_queries, current_ground_truth = adapter.load_nlctables_dataset(task, args.dataset_type)
                    logger.info(f"重新加载 {task} 任务数据: {len(current_tables)} 表, {len(current_queries)} 查询")
                else:
                    # 使用初始加载的数据
                    current_tables, current_queries, current_ground_truth = tables, queries, ground_truth
                
                results, elapsed_time = run_nlctables_experiment(
                    layer=layer,
                    tables=current_tables,
                    queries=current_queries,
                    task_type=task,
                    max_queries=max_queries,
                    max_workers=args.workers,
                    challenging=args.challenging
                )
                
                # 使用当前任务的ground truth
                ground_truth = current_ground_truth
            else:
                results, elapsed_time = run_webtable_santos_experiment(
                    layer=layer,
                    tables=tables,
                    queries=queries,
                    task_type=task,
                    dataset_type=dataset_type,
                    max_queries=max_queries,
                    max_workers=args.workers,
                    challenging=args.challenging
                )
            
            # 保存结果和对应的ground truth
            experiment_key = f"{task}_{layer}"
            all_results[experiment_key] = {
                'results': results,
                'elapsed_time': elapsed_time,
                'task': task,
                'layer': layer,
                'ground_truth': ground_truth if dataset_type == 'nlctables' else None  # 保存对应的ground truth
            }
    
    # 评估所有结果
    all_metrics = {}
    for exp_key, exp_data in all_results.items():
        results = exp_data['results']
        
        # 评估结果（如果有ground truth）
        if dataset_type == 'nlctables':
            # 使用每个实验保存的对应ground truth
            exp_ground_truth = exp_data.get('ground_truth')
            if exp_ground_truth:
                metrics = evaluate_results(results, exp_ground_truth)
                all_metrics[exp_key] = metrics
        elif ground_truth_path.exists():
            # 其他数据集使用文件中的ground truth
            with open(ground_truth_path, 'r') as f:
                ground_truth = json.load(f)
            metrics = evaluate_results(results, ground_truth)
            all_metrics[exp_key] = metrics
        
        # 打印评估指标
        if exp_key in all_metrics:
            logger.info(f"\n📈 评估指标 [{exp_key}]:")
            for metric, value in all_metrics[exp_key].items():
                logger.info(f"  {metric}: {value:.3f}")
    
    # 保存结果
    if args.output:
        output_path = args.output
    else:
        # 自动保存到experiment_results文件夹
        experiment_dir = Path('experiment_results')
        experiment_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        layer_str = args.layer.replace('+', '_')
        filename = f'unified_results_{dataset_type}_{args.task}_{layer_str}_{timestamp}.json'
        output_path = experiment_dir / filename
    
    # 构建输出数据
    output_data = {
        'dataset': dataset_type,
        'dataset_type': args.dataset_type,
        'task': args.task,
        'layer': args.layer,
        'workers': args.workers,
        'max_queries': max_queries,
        'challenging': args.challenging,
        'experiments': all_results,
        'metrics': all_metrics if all_metrics else None
    }
    
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    logger.info(f"💾 Results saved to {output_path}")
    
    # 打印摘要
    print("\n" + "="*60)
    print(f"🎯 统一实验完成")
    print(f"   数据集: {dataset_type} ({args.dataset_type})")
    print(f"   任务: {args.task}")
    print(f"   层级: {args.layer}")
    print(f"   并行进程: {args.workers}")
    print(f"   查询数: {max_queries if max_queries else 'all'}")
    print(f"   挑战性查询: {'启用' if args.challenging else '禁用'}")
    
    # 打印每个实验的结果
    for exp_key, exp_data in all_results.items():
        print(f"\n   📊 {exp_key}:")
        print(f"      用时: {exp_data['elapsed_time']:.2f}s")
        if exp_key in all_metrics:
            metrics = all_metrics[exp_key]
            print(f"      Hit@1: {metrics.get('hit@1', 0):.3f}")
            print(f"      Hit@3: {metrics.get('hit@3', 0):.3f}")
            print(f"      F1: {metrics.get('f1', 0):.3f}")
    
    print("="*60)
    
    # 打印统计表格
    if all_metrics:
        print_results_table(all_results, all_metrics)
        print()

if __name__ == "__main__":
    main()