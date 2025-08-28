#!/usr/bin/env python
"""
统一实验运行器 - 支持WebTable、OpenData和NLCTables三个数据集
可以使用同一个系统运行所有数据集的实验
支持与three_layer_ablation_optimized.py相同的所有参数
"""

import os
import sys
import json
import time
import shutil
import pickle
import argparse
import logging
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any
from datetime import datetime

# 从three_layer_ablation_optimized导入需要的函数
from three_layer_ablation_optimized import convert_ground_truth_format

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 可配置的最大预测数量（支持@K计算，K最大为10，设置为20留有余量）
MAX_PREDICTIONS = int(os.environ.get('MAX_PREDICTIONS', '20'))
logger.info(f"📊 MAX_PREDICTIONS set to {MAX_PREDICTIONS} (supports up to @{MAX_PREDICTIONS//2} evaluation)")

# 全局缓存存储
global_unified_cache = {}

def clear_experiment_cache(specific_dataset: str = None):
    """清理实验缓存（但保留嵌入向量缓存）
    
    Args:
        specific_dataset: 如果指定，只清理该数据集的缓存
    """
    cache_root = Path("cache")
    
    if not cache_root.exists():
        logger.info("📦 缓存目录不存在，无需清理")
        return 0
    
    cleared_count = 0
    
    if specific_dataset:
        # 清理特定数据集的临时缓存（但保留嵌入向量）
        patterns = [
            f"ablation_{specific_dataset}_*",
            f"experiment_cache/{specific_dataset}_*",
            f"experiment_{specific_dataset}_*",
            f"unified_{specific_dataset}_*"
        ]
        logger.info(f"🧹 清理 {specific_dataset} 数据集的缓存...")
    else:
        # 清理所有临时缓存（但保留嵌入向量）
        patterns = ["ablation_*", "experiment_*", "unified_*"]
        logger.info("🧹 清理所有实验缓存...")
    
    for pattern in patterns:
        for cache_path in cache_root.glob(pattern):
            if cache_path.is_dir():
                try:
                    shutil.rmtree(cache_path)
                    cleared_count += 1
                    logger.debug(f"  ✅ 删除缓存目录: {cache_path}")
                except Exception as e:
                    logger.warning(f"  ⚠️ 无法删除 {cache_path}: {e}")
    
    # 重要：不删除 cache/{dataset}/ 目录本身，因为嵌入向量在那里！
    # 只删除 ablation_*, experiment_*, unified_* 等临时缓存
    
    if cleared_count > 0:
        logger.info(f"✅ 清理完成，删除了 {cleared_count} 个临时缓存目录")
        logger.info(f"📦 保留了嵌入向量缓存在 cache/{specific_dataset or '*'}/ 目录")
    else:
        logger.info("📦 没有找到需要清理的临时缓存")
    
    return cleared_count


def prepare_unified_cache(tables: List[Dict], dataset_name: str, task_type: str) -> Dict[str, Any]:
    """为整个实验准备统一的缓存（向量索引和嵌入）
    
    Args:
        tables: 表列表
        dataset_name: 数据集名称
        task_type: 任务类型
        
    Returns:
        包含预计算数据的字典
    """
    global global_unified_cache
    
    # 如果已经有缓存，直接返回
    cache_key = f"{dataset_name}_{task_type}_{len(tables)}"
    if cache_key in global_unified_cache:
        logger.info("📦 使用已有的统一缓存")
        return global_unified_cache[cache_key]
    
    logger.info("📊 准备统一的实验缓存...")
    
    # 创建缓存目录
    cache_dir = Path("cache") / dataset_name
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    # 向量索引和嵌入文件
    index_file = cache_dir / f"vector_index_{len(tables)}.pkl"
    embeddings_file = cache_dir / f"table_embeddings_{len(tables)}.pkl"
    
    # 检查是否已存在
    if index_file.exists() and embeddings_file.exists():
        logger.info("  📦 加载现有的向量索引和嵌入...")
        try:
            with open(index_file, 'rb') as f:
                vector_index = pickle.load(f)
            with open(embeddings_file, 'rb') as f:
                table_embeddings = pickle.load(f)
        except Exception as e:
            logger.warning(f"  ⚠️ 加载缓存失败: {e}，重新生成...")
            vector_index = None
            table_embeddings = None
    else:
        vector_index = None
        table_embeddings = None
    
    # 如果没有缓存，生成新的
    if vector_index is None or table_embeddings is None:
        logger.info("  ⚙️ 生成新的向量索引和嵌入...")
        from precompute_embeddings import precompute_all_embeddings
        precompute_all_embeddings(tables, dataset_name)
        
        # 重新加载生成的文件
        with open(index_file, 'rb') as f:
            vector_index = pickle.load(f)
        with open(embeddings_file, 'rb') as f:
            table_embeddings = pickle.load(f)
    
    logger.info(f"  ✅ 统一缓存准备完成")
    logger.info(f"  📊 向量索引大小: {index_file.stat().st_size / 1024:.2f}KB")
    logger.info(f"  📊 表嵌入大小: {embeddings_file.stat().st_size / 1024:.2f}KB")
    
    result = {
        'vector_index': vector_index,
        'table_embeddings': table_embeddings,
        'cache_dir': cache_dir,
        'cache_key': cache_key
    }
    
    # 存储到全局缓存
    global_unified_cache[cache_key] = result
    
    return result

def detect_dataset_type(tables_path: str) -> str:
    """自动检测数据集类型"""
    path_str = str(tables_path).lower()
    
    if 'nlctables' in path_str:
        return 'nlctables'
    elif 'opendata' in path_str:
        return 'opendata'
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
                elif 'opendata' in first_table_name.lower():
                    return 'opendata'
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
                'predictions': predictions[:MAX_PREDICTIONS] if isinstance(predictions, list) else []
            })
    else:
        results = results_dict
    
    logger.info(f"  Output results: {len(results)} entries")
    if results and len(results) > 0:
        logger.info(f"  First result: {results[0]['query_table']} -> {len(results[0].get('predictions', []))} predictions")
    
    return results, elapsed_time

def run_webtable_opendata_experiment(layer: str, tables: List[Dict], queries: List[Dict],
                                  task_type: str, dataset_type: str, max_queries: int = None,
                                  max_workers: int = 4, challenging: bool = True) -> Tuple[List[Dict], float]:
    """运行WebTable/OpenData实验 - 使用主系统"""
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
                'predictions': predictions[:MAX_PREDICTIONS] if isinstance(predictions, list) else []
            })
    else:
        results = results_dict
    
    logger.info(f"  Output results: {len(results)} entries")
    if results and len(results) > 0:
        logger.info(f"  First result: {results[0]['query_table']} -> {len(results[0].get('predictions', []))} predictions")
    
    return results, elapsed_time

def evaluate_results(results: List[Dict], ground_truth, k_values: List[int] = [1, 3, 5]) -> Dict:
    """评估结果 - ground_truth can be Dict or List"""
    from src.utils.evaluation import calculate_hit_at_k, calculate_precision_recall_f1, calculate_precision_recall_at_k
    
    metrics = {}
    
    # 计算Hit@K
    for k in k_values:
        hit_rate = calculate_hit_at_k(results, ground_truth, k)
        metrics[f'hit@{k}'] = hit_rate
    
    # 计算Precision@K和Recall@K for k=1, 5, 10
    for k in [1, 5, 10]:
        pr_at_k = calculate_precision_recall_at_k(results, ground_truth, k)
        metrics[f'precision@{k}'] = pr_at_k['precision']
        metrics[f'recall@{k}'] = pr_at_k['recall']
    
    # 计算全量Precision/Recall/F1（保留，用于兼容性）
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
            'precision@1': metrics.get('precision@1', 0.0),
            'precision@5': metrics.get('precision@5', 0.0),
            'precision@10': metrics.get('precision@10', 0.0),
            'recall@1': metrics.get('recall@1', 0.0),
            'recall@5': metrics.get('recall@5', 0.0),
            'recall@10': metrics.get('recall@10', 0.0),
            'time': elapsed_time
        }
        
        if task == 'join':
            join_results[layer] = result_data
        elif task == 'union':
            union_results[layer] = result_data
    
    # 打印JOIN结果表格
    if join_results:
        print("\nJOIN Task Results:")
        print("-" * 150)
        print(f"{'Layer':<12} {'Hit@1':<8} {'Hit@3':<8} {'Hit@5':<8} "
              f"{'P@1':<8} {'P@5':<8} {'P@10':<8} "
              f"{'R@1':<8} {'R@5':<8} {'R@10':<8} {'Time(s)':<8}")
        print("-" * 150)
        
        # 按L1, L1+L2, L1+L2+L3顺序排序
        layer_order = ['L1', 'L1+L2', 'L1+L2+L3']
        for layer in layer_order:
            if layer in join_results:
                data = join_results[layer]
                print(f"{layer:<12} {data['hit@1']:<8.3f} {data['hit@3']:<8.3f} {data['hit@5']:<8.3f} "
                      f"{data['precision@1']:<8.3f} {data['precision@5']:<8.3f} {data['precision@10']:<8.3f} "
                      f"{data['recall@1']:<8.3f} {data['recall@5']:<8.3f} {data['recall@10']:<8.3f} "
                      f"{data['time']:<8.2f}")
    
    # 打印UNION结果表格
    if union_results:
        print("\nUNION Task Results:")
        print("-" * 150)
        print(f"{'Layer':<12} {'Hit@1':<8} {'Hit@3':<8} {'Hit@5':<8} "
              f"{'P@1':<8} {'P@5':<8} {'P@10':<8} "
              f"{'R@1':<8} {'R@5':<8} {'R@10':<8} {'Time(s)':<8}")
        print("-" * 150)
        
        # 按L1, L1+L2, L1+L2+L3顺序排序
        layer_order = ['L1', 'L1+L2', 'L1+L2+L3']
        for layer in layer_order:
            if layer in union_results:
                data = union_results[layer]
                print(f"{layer:<12} {data['hit@1']:<8.3f} {data['hit@3']:<8.3f} {data['hit@5']:<8.3f} "
                      f"{data['precision@1']:<8.3f} {data['precision@5']:<8.3f} {data['precision@10']:<8.3f} "
                      f"{data['recall@1']:<8.3f} {data['recall@5']:<8.3f} {data['recall@10']:<8.3f} "
                      f"{data['time']:<8.2f}")

def main():
    parser = argparse.ArgumentParser(description='统一实验运行器')
    parser.add_argument('--clear-cache', action='store_true',
                       help='实验前清理缓存（现在默认自动清理）')
    parser.add_argument('--no-clear-cache', action='store_true',
                       help='禁用自动缓存清理（用于同一会话内的连续实验）')
    parser.add_argument('--no-cache', action='store_true',
                       help='强制重新生成所有缓存')
    parser.add_argument('--dataset', type=str, required=True,
                       help='数据集路径或名称 (webtable/opendata/nlctables)')
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
    
    # 总是清理缓存以确保实验结果的可重复性
    # 除非明确指定了--no-clear-cache
    if not hasattr(args, 'no_clear_cache') or not args.no_clear_cache:
        if not args.clear_cache:
            logger.info("🧹 自动清理缓存（使用 --no-clear-cache 禁用）")
        clear_experiment_cache(args.dataset if args.dataset in ['webtable', 'opendata', 'nlctables'] else None)
    
    # 如果需要强制重新生成缓存
    if args.no_cache:
        logger.info("⚠️ 强制重新生成所有缓存")
        os.environ['FORCE_REBUILD_CACHE'] = 'true'
    
    # 确定数据集路径
    if args.dataset in ['webtable', 'opendata', 'nlctables']:
        # 使用预定义路径，根据dataset-type选择
        if args.dataset == 'webtable':
            # WebTable路径格式：examples/webtable/{task}_{dataset_type}/ (与OpenData相同)
            if args.task == 'both':
                # 对于both任务，先使用join数据集，后面会根据任务动态切换
                tables_path = f'examples/webtable/join_{args.dataset_type}/tables.json'
            else:
                tables_path = f'examples/webtable/{args.task}_{args.dataset_type}/tables.json'
        elif args.dataset == 'opendata':
            # OpenData路径格式：examples/opendata/{task}_{dataset_type}/
            if args.task == 'both':
                # 对于both任务，先使用join数据集，后面会根据任务动态切换
                tables_path = f'examples/opendata/join_{args.dataset_type}/tables.json'
            else:
                tables_path = f'examples/opendata/{args.task}_{args.dataset_type}/tables.json'
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
        # 原有的加载逻辑（WebTable/OpenData）
        with open(tables_path, 'r') as f:
            tables = json.load(f)
        logger.info(f"Loaded {len(tables)} tables")
        
        # 对于OpenData和WebTable，确保表有name字段（兼容性）
        if dataset_type in ['opendata', 'webtable']:
            for t in tables:
                if 'name' not in t and 'table_name' in t:
                    t['name'] = t['table_name']
        
        # 加载查询
        queries_path = base_dir / 'queries.json'
        
        if queries_path.exists():
            with open(queries_path, 'r') as f:
                queries = json.load(f)
            logger.info(f"Loaded {len(queries)} queries")
        else:
            # 生成默认查询
            queries = [{'query_table': t.get('name', t.get('table_name')), 'task_type': args.task} for t in tables[:10]]
            logger.warning("No queries file found, using first 10 tables as queries")
        
        # 加载ground truth
        ground_truth_path = base_dir / 'ground_truth.json'
        if ground_truth_path.exists():
            with open(ground_truth_path, 'r') as f:
                ground_truth_list = json.load(f)
            logger.info(f"Loaded {len(ground_truth_list)} ground truth entries")
            # 转换ground truth格式
            ground_truth = convert_ground_truth_format(ground_truth_list, task_type=args.task)
        else:
            ground_truth_list = []
            ground_truth = {}
            logger.warning("No ground truth file found")
    
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
    
    # 为整个实验准备统一缓存（所有层和任务共享）
    if not args.skip_llm and len(layers_to_run) > 1:  # 只有多层时才需要统一缓存
        unified_cache = prepare_unified_cache(tables, dataset_type, tasks_to_run[0])
        # 将缓存信息存储到环境变量供子进程使用
        os.environ['UNIFIED_CACHE_DIR'] = str(unified_cache['cache_dir'])
        logger.info(f"  📦 所有层将共享统一的向量索引和嵌入")
    
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
                # OpenData和WebTable需要为不同任务重新加载数据
                if dataset_type in ['opendata', 'webtable'] and args.task == 'both':
                    # 为OpenData/WebTable的不同任务加载相应的数据
                    task_tables_path = f'examples/{dataset_type}/{task}_{args.dataset_type}/tables.json'
                    task_queries_path = f'examples/{dataset_type}/{task}_{args.dataset_type}/queries.json'
                    task_ground_truth_path = f'examples/{dataset_type}/{task}_{args.dataset_type}/ground_truth.json'
                    
                    with open(task_tables_path, 'r') as f:
                        current_tables = json.load(f)
                    with open(task_queries_path, 'r') as f:
                        current_queries = json.load(f)
                    with open(task_ground_truth_path, 'r') as f:
                        current_ground_truth = json.load(f)
                    
                    logger.info(f"加载{dataset_type.upper()} {task} 任务数据: {len(current_tables)} 表, {len(current_queries)} 查询")
                    
                    # 确保表有name字段
                    for t in current_tables:
                        if 'name' not in t and 'table_name' in t:
                            t['name'] = t['table_name']
                    
                    ground_truth = convert_ground_truth_format(current_ground_truth, task_type=task)
                else:
                    # 使用已加载的数据
                    current_tables = tables
                    current_queries = queries
                    current_ground_truth = ground_truth_list
                    
                results, elapsed_time = run_webtable_opendata_experiment(
                    layer=layer,
                    tables=current_tables,
                    queries=current_queries,
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
                'ground_truth': ground_truth if dataset_type in ['nlctables', 'opendata', 'webtable'] else None  # 保存对应的ground truth
            }
    
    # 评估所有结果
    all_metrics = {}
    for exp_key, exp_data in all_results.items():
        results = exp_data['results']
        
        # 评估结果（如果有ground truth）
        if dataset_type in ['nlctables', 'opendata', 'webtable']:
            # 使用每个实验保存的对应ground truth
            exp_ground_truth = exp_data.get('ground_truth')
            if exp_ground_truth:
                metrics = evaluate_results(results, exp_ground_truth)
                all_metrics[exp_key] = metrics
        elif ground_truth_path.exists():
            # 其他数据集使用文件中的ground truth
            with open(ground_truth_path, 'r') as f:
                ground_truth_raw = json.load(f)
            # 转换ground truth格式（如果是列表格式）
            if isinstance(ground_truth_raw, list):
                # 从exp_key中提取任务类型（exp_key格式为"task_layer"）
                exp_task = exp_key.split('_')[0]  # 提取task部分
                ground_truth = convert_ground_truth_format(ground_truth_raw, task_type=exp_task)
            else:
                ground_truth = ground_truth_raw
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
            print(f"      P@5: {metrics.get('precision@5', 0):.3f}")
            print(f"      R@5: {metrics.get('recall@5', 0):.3f}")
    
    print("="*60)
    
    # 打印统计表格
    if all_metrics:
        print_results_table(all_results, all_metrics)
        print()

if __name__ == "__main__":
    main()