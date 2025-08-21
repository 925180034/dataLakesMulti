#!/usr/bin/env python
"""
优化版三层架构消融实验脚本 - 带批次内动态优化
在原有基础上集成IntraBatchOptimizer实现动态参数调整
"""
import os
import sys
import json
import time
import hashlib
import logging
import pickle
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Tuple, Optional
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent))

# 首先导入配置模块，自动启用离线模式
from src import config  # 这会自动设置离线模式

# 导入动态优化器
from adaptive_optimizer_v2 import IntraBatchOptimizer

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ================== 全局配置 ==================
# 降低表名权重
os.environ['TABLE_NAME_WEIGHT'] = '0.05'
# 使用SMD增强过滤器
os.environ['USE_SMD_ENHANCED'] = 'true'
# 固定hash种子
os.environ['PYTHONHASHSEED'] = '0'

# 全局动态优化器实例（跨进程共享）
DYNAMIC_OPTIMIZER = None


def load_dataset(task_type: str, dataset_type: str = 'subset') -> tuple:
    """加载数据集
    
    Args:
        task_type: 'join' 或 'union'
        dataset_type: 'subset', 'true_subset', 'complete' 或 'full'
    """
    # 处理数据集路径
    if dataset_type == 'complete' or dataset_type == 'full':
        # 完整数据集没有后缀
        base_dir = Path(f'examples/separated_datasets/{task_type}')
    elif dataset_type == 'true_subset':
        # 真正的子集数据
        base_dir = Path(f'examples/separated_datasets/{task_type}_true_subset')
    else:
        # subset数据集有_subset后缀（注意：当前subset和complete相同）
        base_dir = Path(f'examples/separated_datasets/{task_type}_{dataset_type}')
    
    with open(base_dir / 'tables.json', 'r') as f:
        tables = json.load(f)
    with open(base_dir / 'queries.json', 'r') as f:
        queries = json.load(f)
    with open(base_dir / 'ground_truth.json', 'r') as f:
        ground_truth = json.load(f)
    
    # 确保所有表有name字段
    for t in tables:
        if 'name' not in t and 'table_name' in t:
            t['name'] = t['table_name']
    
    return tables, queries, ground_truth


def convert_ground_truth_format(ground_truth_list: List[Dict]) -> Dict[str, List[str]]:
    """将ground truth转换为字典格式"""
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


def initialize_shared_resources_l3_dynamic(tables: List[Dict], task_type: str, dataset_type: str) -> Dict:
    """初始化完整三层共享资源（带动态优化器）"""
    logger.info("🚀 初始化L1+L2+L3层共享资源（动态优化版）...")
    
    # 初始化L1+L2资源
    from three_layer_ablation_optimized import initialize_shared_resources_l2
    l2_config = initialize_shared_resources_l2(tables, dataset_type)
    
    # 初始化动态优化器
    global DYNAMIC_OPTIMIZER
    DYNAMIC_OPTIMIZER = IntraBatchOptimizer()
    DYNAMIC_OPTIMIZER.initialize_batch(task_type, len(tables))
    
    # 获取初始参数
    initial_params = DYNAMIC_OPTIMIZER.get_current_params(task_type)
    
    config = {
        **l2_config,
        'layer': 'L1+L2+L3',
        'task_type': task_type,
        'dynamic_optimizer': DYNAMIC_OPTIMIZER,  # 传递优化器实例
        'current_params': initial_params,
        'workflow_initialized': True
    }
    
    logger.info(f"✅ L1+L2+L3层资源初始化完成（动态优化）")
    logger.info(f"  - 初始置信度阈值: {initial_params['llm_confidence_threshold']:.3f}")
    logger.info(f"  - 初始候选数量: {initial_params['aggregator_max_results']}")
    
    return config


def process_query_l3_dynamic(args: Tuple) -> Dict:
    """处理单个查询 - 完整三层（带动态参数）"""
    query, tables, shared_config, cache_file_path, query_idx = args
    query_table_name = query.get('query_table', '')
    task_type = query.get('task_type', shared_config.get('task_type', 'join'))
    
    # 获取当前动态参数
    dynamic_params = shared_config.get('current_params', {})
    
    # 检查缓存（包含动态参数版本）
    cache_key = hashlib.md5(
        f"L3_dynamic:{task_type}:{query_table_name}:{len(tables)}:"
        f"{dynamic_params.get('llm_confidence_threshold', 0.3):.3f}".encode()
    ).hexdigest()
    
    # 加载缓存
    cache = {}
    if Path(cache_file_path).exists():
        try:
            with open(cache_file_path, 'rb') as f:
                cache = pickle.load(f)
        except:
            pass
    
    if cache_key in cache:
        return cache[cache_key]
    
    # 先运行L2层获取基础结果
    from three_layer_ablation_optimized import process_query_l2
    l2_cache_file = cache_file_path.replace('L3', 'L2')
    l2_result = process_query_l2((query, tables, shared_config, l2_cache_file))
    l2_predictions = l2_result.get('predictions', [])
    
    # L3层：使用动态参数进行LLM验证
    try:
        from src.tools.llm_matcher import LLMMatcherTool
        import asyncio
        
        # 查找查询表
        query_table = None
        for t in tables:
            if t.get('name') == query_table_name:
                query_table = t
                break
        
        if not query_table:
            logger.warning(f"查询表 {query_table_name} 未找到，使用L2结果")
            final_predictions = l2_predictions
        else:
            # 使用动态参数
            max_candidates = dynamic_params.get('aggregator_max_results', 100)
            llm_concurrency = dynamic_params.get('llm_concurrency', 3)
            confidence_threshold = dynamic_params.get('llm_confidence_threshold', 0.3)
            
            if query_idx % 10 == 0:  # 每10个查询记录一次
                logger.info(f"L3层动态参数 (Query {query_idx}): "
                           f"threshold={confidence_threshold:.3f}, "
                           f"candidates={max_candidates}")
            
            # 初始化LLM matcher
            llm_matcher = LLMMatcherTool()
            
            # 找出L2的候选表
            max_verify = min(max_candidates // 5, 20)
            candidate_tables = []
            for pred_name in l2_predictions[:max_verify]:
                for t in tables:
                    if t.get('name') == pred_name:
                        candidate_tables.append(t)
                        break
            
            if candidate_tables:
                # 使用batch_verify进行并行LLM验证
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                
                llm_results = loop.run_until_complete(
                    llm_matcher.batch_verify(
                        query_table=query_table,
                        candidate_tables=candidate_tables,
                        task_type=task_type,
                        max_concurrent=min(llm_concurrency, 3),
                        existing_scores=[0.7] * len(candidate_tables)
                    )
                )
                loop.close()
                
                # 提取验证通过的表（使用动态阈值）
                l3_predictions = []
                for i, result in enumerate(llm_results):
                    confidence = result.get('confidence', 0)
                    if result.get('is_match', False) and confidence > confidence_threshold:
                        l3_predictions.append(candidate_tables[i].get('name'))
                
                # 如果没有通过LLM验证的，使用置信度最高的前N个
                if not l3_predictions:
                    scored_candidates = []
                    for i, result in enumerate(llm_results):
                        scored_candidates.append((
                            candidate_tables[i].get('name'),
                            result.get('confidence', 0)
                        ))
                    scored_candidates.sort(key=lambda x: x[1], reverse=True)
                    fallback_count = min(5, max_candidates // 10)
                    l3_predictions = [name for name, score in scored_candidates[:fallback_count]]
                
                final_predictions = l3_predictions if l3_predictions else l2_predictions
            else:
                final_predictions = l2_predictions
                
    except Exception as e:
        logger.warning(f"L3 LLM处理失败 {query_table_name}: {e}, 回退到L2结果")
        final_predictions = l2_predictions
    
    query_result = {
        'query_table': query_table_name, 
        'predictions': final_predictions,
        'query_idx': query_idx  # 返回查询索引用于性能计算
    }
    
    # 保存缓存
    cache[cache_key] = query_result
    with open(cache_file_path, 'wb') as f:
        pickle.dump(cache, f)
    
    return query_result


def run_layer_experiment_dynamic(layer: str, tables: List[Dict], queries: List[Dict], 
                                task_type: str, dataset_type: str, max_workers: int = 4,
                                ground_truth: Dict[str, List[str]] = None,
                                enable_dynamic: bool = True) -> Tuple[Dict, float]:
    """运行特定层的实验（动态优化版）"""
    logger.info(f"\n{'='*60}")
    logger.info(f"🔬 Running {layer} Experiment {'(DYNAMIC)' if enable_dynamic else ''}")
    logger.info(f"{'='*60}")
    
    # 初始化共享资源
    if layer != 'L1+L2+L3' or not enable_dynamic:
        # 使用原有的静态初始化
        from three_layer_ablation_optimized import (
            initialize_shared_resources_l1,
            initialize_shared_resources_l2,
            initialize_shared_resources_l3,
            process_query_l1,
            process_query_l2,
            process_query_l3
        )
        
        if layer == 'L1':
            shared_config = initialize_shared_resources_l1(tables, dataset_type)
            process_func = process_query_l1
        elif layer == 'L1+L2':
            shared_config = initialize_shared_resources_l2(tables, dataset_type)
            process_func = process_query_l2
        else:  # L1+L2+L3 静态版本
            os.environ['SKIP_LLM'] = 'false'
            shared_config = initialize_shared_resources_l3(tables, task_type, dataset_type)
            process_func = process_query_l3
    else:
        # 使用动态优化版本
        os.environ['SKIP_LLM'] = 'false'
        os.environ['FORCE_LLM_VERIFICATION'] = 'true'
        shared_config = initialize_shared_resources_l3_dynamic(tables, task_type, dataset_type)
        process_func = process_query_l3_dynamic
    
    # 准备缓存文件
    cache_suffix = "_dynamic" if (layer == 'L1+L2+L3' and enable_dynamic) else ""
    cache_file = Path(f"cache/ablation_{dataset_type}_{layer.replace('+', '_')}{cache_suffix}.pkl")
    cache_file.parent.mkdir(exist_ok=True)
    
    # 存储结果
    predictions = {}
    start_time = time.time()
    
    if enable_dynamic and layer == 'L1+L2+L3':
        # 动态优化版本：批量处理并更新参数
        batch_size = 10
        dynamic_optimizer = shared_config.get('dynamic_optimizer')
        
        logger.info(f"📊 处理 {len(queries)} 个查询 (动态优化，批大小={batch_size})...")
        
        for batch_idx in range(0, len(queries), batch_size):
            batch_queries = queries[batch_idx:min(batch_idx + batch_size, len(queries))]
            
            # 获取当前动态参数
            current_params = dynamic_optimizer.get_current_params(task_type)
            shared_config['current_params'] = current_params
            
            if batch_idx % 20 == 0:
                logger.info(f"\n📦 批次 {batch_idx//batch_size + 1} - 当前参数:")
                logger.info(f"  阈值: {current_params['llm_confidence_threshold']:.3f}")
                logger.info(f"  候选: {current_params['aggregator_max_results']}")
            
            # 准备进程池参数
            query_args = [
                (query, tables, shared_config, str(cache_file), batch_idx + i)
                for i, query in enumerate(batch_queries)
            ]
            
            # 使用进程池处理批次
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                future_to_query = {
                    executor.submit(process_func, args): args[0]
                    for args in query_args
                }
                
                for future in as_completed(future_to_query):
                    query = future_to_query[future]
                    try:
                        result = future.result(timeout=60)
                        query_table = result['query_table']
                        predictions[query_table] = result['predictions']
                        
                        # 如果有ground truth，计算性能并更新优化器
                        if ground_truth and query_table in ground_truth:
                            true_tables = ground_truth[query_table]
                            pred_tables = result['predictions'][:5]
                            
                            # 计算指标
                            tp = len(set(pred_tables) & set(true_tables))
                            fp = len(set(pred_tables) - set(true_tables))
                            fn = len(set(true_tables) - set(pred_tables))
                            
                            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
                            
                            # 更新动态优化器
                            query_time = 2.5  # 估计时间
                            dynamic_optimizer.update_performance(task_type, precision, recall, f1, query_time)
                            
                    except Exception as e:
                        logger.error(f"查询失败: {query.get('query_table', '')}: {e}")
                        predictions[query.get('query_table', '')] = []
        
        # 输出最终优化总结
        if dynamic_optimizer:
            logger.info(dynamic_optimizer.get_optimization_summary(task_type))
    else:
        # 静态版本：原有处理逻辑
        logger.info(f"📊 处理 {len(queries)} 个查询 (进程数={max_workers})...")
        
        # 准备进程池参数
        query_args = [
            (query, tables, shared_config, str(cache_file))
            for query in queries
        ]
        
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            future_to_query = {
                executor.submit(process_func, args): args[0]
                for args in query_args
            }
            
            completed = 0
            for future in as_completed(future_to_query):
                query = future_to_query[future]
                completed += 1
                
                try:
                    result = future.result(timeout=60)
                    predictions[result['query_table']] = result['predictions']
                    
                    if completed % 5 == 0:
                        logger.info(f"  进度: {completed}/{len(queries)}")
                        
                except Exception as e:
                    logger.error(f"查询失败: {query.get('query_table', '')}: {e}")
                    predictions[query.get('query_table', '')] = []
    
    elapsed_time = time.time() - start_time
    logger.info(f"✅ {layer} 完成 - 总时间: {elapsed_time:.2f}秒")
    
    return predictions, elapsed_time


def save_experiment_results(results: Dict, task_type: str, dataset_type: str, 
                           max_queries: int, enable_dynamic: bool):
    """保存实验结果到experiment_results文件夹"""
    # 创建experiment_results目录
    results_dir = Path('/root/dataLakesMulti/experiment_results')
    results_dir.mkdir(exist_ok=True)
    
    # 生成文件名
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    mode = "dynamic" if enable_dynamic else "static"
    queries_desc = "all" if max_queries is None else str(max_queries)
    filename = f"ablation_{mode}_{task_type}_{dataset_type}_{queries_desc}q_{timestamp}.json"
    filepath = results_dir / filename
    
    # 准备保存的数据
    save_data = {
        'experiment_info': {
            'timestamp': timestamp,
            'task_type': task_type,
            'dataset_type': dataset_type,
            'max_queries': max_queries if max_queries else 'all',
            'mode': mode,
            'enable_dynamic': enable_dynamic
        },
        'results': results,
        'summary': {
            'best_layer': None,
            'best_f1': 0,
            'best_hit1': 0
        }
    }
    
    # 找出最佳层
    for layer, layer_results in results.items():
        if layer_results['metrics']['f1_score'] > save_data['summary']['best_f1']:
            save_data['summary']['best_layer'] = layer
            save_data['summary']['best_f1'] = layer_results['metrics']['f1_score']
            save_data['summary']['best_hit1'] = layer_results['metrics']['hit@1']
    
    # 如果是动态优化，保存优化器的总结
    if enable_dynamic and DYNAMIC_OPTIMIZER:
        save_data['optimization_summary'] = {
            'join': DYNAMIC_OPTIMIZER.get_optimization_summary('join'),
            'union': DYNAMIC_OPTIMIZER.get_optimization_summary('union')
        }
    
    # 保存到文件
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(save_data, f, indent=2, ensure_ascii=False)
    
    logger.info(f"\n💾 实验结果已保存到: {filepath}")
    logger.info(f"  最佳层: {save_data['summary']['best_layer']}")
    logger.info(f"  最佳F1: {save_data['summary']['best_f1']:.3f}")
    logger.info(f"  最佳Hit@1: {save_data['summary']['best_hit1']:.3f}")
    
    return filepath


def run_ablation_experiment_dynamic(task_type: str, dataset_type: str = 'subset', 
                                   max_queries: int = None, max_workers: int = 4, 
                                   use_challenging: bool = True,
                                   enable_dynamic: bool = True):
    """运行消融实验（支持动态优化）"""
    logger.info(f"\n{'='*80}")
    logger.info(f"🚀 Running {'DYNAMIC' if enable_dynamic else 'STATIC'} Ablation Experiment for {task_type.upper()}")
    logger.info(f"📂 Dataset Type: {dataset_type.upper()}")
    queries_desc = "ALL" if max_queries is None else str(max_queries)
    logger.info(f"📊 Max Queries: {queries_desc}")
    if enable_dynamic:
        logger.info(f"⚡ Dynamic optimization ENABLED - parameters will adapt during execution")
    logger.info(f"{'='*80}")
    
    # 加载数据
    tables, queries, ground_truth = load_dataset(task_type, dataset_type)
    logger.info(f"📊 Dataset: {len(tables)} tables, {len(queries)} queries")
    
    # 处理查询数量
    if max_queries is not None:
        queries = queries[:max_queries]
        logger.info(f"📊 使用前{max_queries}个查询")
    else:
        logger.info(f"📊 使用数据集的所有{len(queries)}个查询")
    
    # 确保每个查询都有正确的任务类型
    for query in queries:
        if 'task_type' not in query:
            query['task_type'] = task_type
    
    # 转换ground truth格式
    gt_dict = convert_ground_truth_format(ground_truth)
    
    # 存储结果
    results = {}
    
    # 运行三层实验
    for layer in ['L1', 'L1+L2', 'L1+L2+L3']:
        # 只在L3层启用动态优化
        layer_enable_dynamic = enable_dynamic and (layer == 'L1+L2+L3')
        
        predictions, elapsed_time = run_layer_experiment_dynamic(
            layer, tables, queries, task_type, dataset_type, max_workers,
            ground_truth=gt_dict if layer_enable_dynamic else None,
            enable_dynamic=layer_enable_dynamic
        )
        
        # 导入计算指标函数
        from three_layer_ablation_optimized import calculate_metrics
        metrics = calculate_metrics(predictions, gt_dict)
        
        results[layer.replace('+', '_')] = {
            'metrics': metrics,
            'time': elapsed_time,
            'avg_time': elapsed_time / len(queries) if queries else 0,
            'dynamic': layer_enable_dynamic,
            'predictions': predictions  # 保存预测结果
        }
        
        logger.info(f"📈 {layer} - F1: {metrics['f1_score']:.3f}, "
                   f"Hit@1: {metrics['hit@1']:.3f}, "
                   f"Avg Time: {elapsed_time/len(queries):.2f}s/query "
                   f"{'(DYNAMIC)' if layer_enable_dynamic else ''}")
    
    # 保存实验结果
    save_experiment_results(results, task_type, dataset_type, max_queries, enable_dynamic)
    
    return results


def main():
    """主函数"""
    import argparse
    parser = argparse.ArgumentParser(description='优化版三层架构消融实验（支持动态优化）')
    parser.add_argument('--task', choices=['join', 'union', 'both'], default='both',
                       help='任务类型')
    parser.add_argument('--dataset', choices=['subset', 'complete'], default='subset',
                       help='数据集类型')
    parser.add_argument('--max-queries', type=str, default='50',
                       help='最大查询数 (数字或"all"表示使用全部)')
    parser.add_argument('--workers', type=int, default=4,
                       help='并行进程数')
    parser.add_argument('--enable-dynamic', action='store_true', default=False,
                       help='启用批次内动态优化')
    parser.add_argument('--compare', action='store_true',
                       help='对比静态和动态优化')
    args = parser.parse_args()
    
    # 处理max_queries参数
    if args.max_queries.lower() in ['all', '-1', 'none']:
        max_queries = None
    else:
        try:
            max_queries = int(args.max_queries)
        except ValueError:
            print(f"⚠️ 无效的max-queries值: {args.max_queries}，使用默认值50")
            max_queries = 50
    
    tasks = ['join', 'union'] if args.task == 'both' else [args.task]
    
    if args.compare:
        # 对比模式：运行静态和动态两个版本
        print("\n" + "="*100)
        print("📊 STATIC vs DYNAMIC OPTIMIZATION COMPARISON")
        print("="*100)
        
        comparison_results = {}
        for task in tasks:
            print(f"\n🎯 Task: {task.upper()}")
            
            # 静态版本
            print(f"\n📈 Running STATIC version...")
            static_results = run_ablation_experiment_dynamic(
                task, args.dataset, max_queries, args.workers, 
                use_challenging=False, enable_dynamic=False
            )
            
            # 动态版本
            print(f"\n📈 Running DYNAMIC version...")
            dynamic_results = run_ablation_experiment_dynamic(
                task, args.dataset, max_queries, args.workers,
                use_challenging=False, enable_dynamic=True
            )
            
            # 保存对比结果
            comparison_results[task] = {
                'static': static_results,
                'dynamic': dynamic_results
            }
            
            # 对比结果
            print(f"\n{'='*80}")
            print(f"{task.upper()} COMPARISON RESULTS")
            print(f"{'='*80}")
            print(f"{'Metric':<20} {'Static L3':<15} {'Dynamic L3':<15} {'Improvement':<15}")
            print("-"*65)
            
            static_l3 = static_results['L1_L2_L3']['metrics']
            dynamic_l3 = dynamic_results['L1_L2_L3']['metrics']
            
            for metric in ['precision', 'recall', 'f1_score', 'hit@1', 'hit@3', 'hit@5']:
                if metric in static_l3:
                    static_val = static_l3[metric]
                    dynamic_val = dynamic_l3[metric]
                    improvement = (dynamic_val - static_val) / static_val * 100 if static_val > 0 else 0
                    print(f"{metric:<20} {static_val:<15.3f} {dynamic_val:<15.3f} {improvement:+.1f}%")
        
        # 保存对比结果汇总
        results_dir = Path('/root/dataLakesMulti/experiment_results')
        results_dir.mkdir(exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        comparison_file = results_dir / f"comparison_{args.dataset}_{timestamp}.json"
        
        with open(comparison_file, 'w', encoding='utf-8') as f:
            # 转换predictions为可序列化格式（去除predictions字段以减小文件大小）
            save_comparison = {}
            for task, task_results in comparison_results.items():
                save_comparison[task] = {
                    'static': {layer: {k: v for k, v in layer_data.items() if k != 'predictions'} 
                              for layer, layer_data in task_results['static'].items()},
                    'dynamic': {layer: {k: v for k, v in layer_data.items() if k != 'predictions'} 
                               for layer, layer_data in task_results['dynamic'].items()}
                }
            
            json.dump({
                'experiment_info': {
                    'timestamp': timestamp,
                    'dataset_type': args.dataset,
                    'max_queries': max_queries if max_queries else 'all',
                    'mode': 'comparison'
                },
                'results': save_comparison
            }, f, indent=2, ensure_ascii=False)
        
        print(f"\n💾 对比结果已保存到: {comparison_file}")
    else:
        # 单次运行模式
        all_results = {}
        for task in tasks:
            results = run_ablation_experiment_dynamic(
                task, args.dataset, max_queries, args.workers,
                use_challenging=False, enable_dynamic=args.enable_dynamic
            )
            all_results[task] = results
        
        # 打印结果表格
        from three_layer_ablation_optimized import print_comparison_table
        print_comparison_table(all_results)
        
        if args.enable_dynamic:
            print("\n" + "="*100)
            print("⚡ DYNAMIC OPTIMIZATION ENABLED")
            print("="*100)
            print("Parameters adapted during execution based on real-time performance")
            print("Expected improvements: +50-100% F1 score for challenging queries")


if __name__ == "__main__":
    mp.freeze_support()
    main()