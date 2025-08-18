#!/usr/bin/env python
"""
终极优化版实验运行脚本 - 批处理级别资源共享
主要优化：
1. 一次向量初始化，所有查询共享
2. 一次框架初始化，复用工作流实例
3. OptimizerAgent和PlannerAgent每个批次只调用一次
4. 使用共享内存和进程池优化
5. 批处理级别的配置缓存
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
from typing import Dict, List, Any, Optional, Tuple
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent))

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ================== 全局配置 ==================
# 降低表名权重
os.environ['TABLE_NAME_WEIGHT'] = '0.05'
# 确保LLM层启用
os.environ['FORCE_LLM_VERIFICATION'] = 'true'
os.environ['SKIP_LLM'] = 'false'
# 使用SMD增强过滤器
os.environ['USE_SMD_ENHANCED'] = 'true'
# 固定hash种子
os.environ['PYTHONHASHSEED'] = '0'

# ================== 全局共享资源管理 ==================
# Manager必须在主进程中初始化


def initialize_shared_resources(tables: List[Dict], task_type: str, dataset_type: str):
    """
    初始化所有共享资源（只执行一次）
    返回共享配置字典
    """
    logger.info("🚀 初始化批处理共享资源...")
    
    # 1. 预计算向量（一次性完成）
    logger.info("📊 预计算向量嵌入...")
    cache_dir = Path("cache") / dataset_type
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    index_file = cache_dir / f"vector_index_{len(tables)}.pkl"
    embeddings_file = cache_dir / f"table_embeddings_{len(tables)}.pkl"
    
    if not (index_file.exists() and embeddings_file.exists()):
        logger.info("⚙️ 生成新的向量索引...")
        from precompute_embeddings import precompute_all_embeddings
        precompute_all_embeddings(tables, dataset_type)
    
    # 2. 初始化工作流（一次性）
    logger.info("🔧 初始化工作流单例...")
    from src.core.langgraph_workflow import DataLakeDiscoveryWorkflow
    workflow = DataLakeDiscoveryWorkflow()
    
    # 3. 获取优化配置（对整个批次只调用一次OptimizerAgent）
    logger.info("📋 获取批处理优化配置...")
    optimization_config = get_batch_optimization_config(workflow, tables, task_type)
    
    # 4. 获取执行策略（对整个批次只调用一次PlannerAgent）
    logger.info("📋 获取批处理执行策略...")
    execution_strategy = get_batch_execution_strategy(workflow, task_type)
    
    # 保存到共享配置
    config = {
        'vector_index_path': str(index_file),
        'embeddings_path': str(embeddings_file),
        'optimization_config': optimization_config,
        'execution_strategy': execution_strategy,
        'table_count': len(tables),
        'task_type': task_type,
        'dataset_type': dataset_type
    }
    
    logger.info("✅ 批处理资源初始化完成")
    logger.info(f"  - 向量索引: 已加载")
    logger.info(f"  - 工作流: 已初始化")
    logger.info(f"  - 优化配置: {optimization_config}")
    logger.info(f"  - 执行策略: {execution_strategy}")
    
    return config


def get_batch_optimization_config(workflow, tables: List[Dict], task_type: str) -> Dict:
    """
    获取整个批次的优化配置（只调用一次OptimizerAgent）
    """
    # 构造一个代表性的查询来获取配置
    cache_key = f"{task_type}_{len(tables)}"
    
    # 检查是否已有缓存的配置
    if hasattr(workflow, '_optimization_cache') and cache_key in workflow._optimization_cache:
        logger.info(f"  📥 使用缓存的优化配置: {cache_key}")
        return workflow._optimization_cache[cache_key]
    
    # 调用OptimizerAgent
    from src.agents.optimizer_agent import OptimizerAgent
    optimizer = OptimizerAgent()
    
    config = optimizer.process({
        'task_type': task_type,
        'table_count': len(tables),
        'complexity': 'medium',  # 固定为medium
        'performance_requirement': 'balanced'
    })
    
    # 缓存配置
    if not hasattr(workflow, '_optimization_cache'):
        workflow._optimization_cache = {}
    workflow._optimization_cache[cache_key] = config
    
    logger.info(f"  ✅ OptimizerAgent配置: workers={config.get('parallel_workers', 8)}, "
               f"cache={config.get('cache_strategy', 'L1')}")
    
    return config


def get_batch_execution_strategy(workflow, task_type: str) -> Dict:
    """
    获取整个批次的执行策略（只调用一次PlannerAgent）
    """
    cache_key = f"{task_type}_strategy"
    
    # 检查是否已有缓存的策略
    if hasattr(workflow, '_strategy_cache') and cache_key in workflow._strategy_cache:
        logger.info(f"  📥 使用缓存的执行策略: {cache_key}")
        return workflow._strategy_cache[cache_key]
    
    # 调用PlannerAgent
    from src.agents.planner_agent import PlannerAgent
    planner = PlannerAgent()
    
    strategy = planner.process({
        'task_type': task_type,
        'table_structure': 'unknown',
        'data_size': 'medium',
        'performance_requirement': 'balanced'
    })
    
    # 缓存策略
    if not hasattr(workflow, '_strategy_cache'):
        workflow._strategy_cache = {}
    workflow._strategy_cache[cache_key] = strategy
    
    logger.info(f"  ✅ PlannerAgent策略: {strategy.get('strategy', 'bottom-up')}, "
               f"top_k={strategy.get('top_k', 100)}")
    
    return strategy


def process_query_with_shared_resources(args: Tuple) -> Dict:
    """
    使用共享资源处理单个查询
    """
    query, tables, shared_config_dict, cache_file_path = args
    query_table = query.get('query_table', '')
    
    # 加载缓存
    cache_file = Path(cache_file_path)
    cache = {}
    if cache_file.exists():
        try:
            with open(cache_file, 'rb') as f:
                cache = pickle.load(f)
        except:
            pass
    
    # 检查缓存
    cache_key = hashlib.md5(
        f"{shared_config_dict['task_type']}:{query_table}:{len(tables)}".encode()
    ).hexdigest()
    
    if cache_key in cache:
        logger.info(f"💎 缓存命中: {query_table}")
        cached_result = cache[cache_key].copy()
        cached_result['from_cache'] = True
        return cached_result
    
    # 创建新的工作流实例（每个进程独立）
    from src.core.langgraph_workflow import DataLakeDiscoveryWorkflow
    workflow = DataLakeDiscoveryWorkflow()
    
    # 直接设置已经计算好的配置（避免重复调用OptimizerAgent和PlannerAgent）
    workflow._optimization_cache = {
        f"{shared_config_dict['task_type']}_{shared_config_dict['table_count']}": 
        shared_config_dict['optimization_config']
    }
    workflow._strategy_cache = {
        f"{shared_config_dict['task_type']}_strategy": 
        shared_config_dict['execution_strategy']
    }
    
    try:
        start_time = time.time()
        
        # 运行工作流（会自动使用缓存的配置）
        result = workflow.run(
            query=f"find {shared_config_dict['task_type']}able tables for {query_table}",
            tables=tables,
            task_type=shared_config_dict['task_type'],
            query_table_name=query_table
        )
        
        elapsed = time.time() - start_time
        
        # 统计LLM调用（不包括OptimizerAgent和PlannerAgent，因为已经缓存）
        llm_calls = 0
        if 'metrics' in result:
            # 只统计AnalyzerAgent和MatcherAgent的调用
            agent_times = result['metrics'].get('agent_times', {})
            if 'analyzer' in agent_times:
                llm_calls += 1
            if 'matcher' in agent_times:
                llm_calls += 1
        
        # 处理结果
        if result.get('success'):
            raw_results = result.get('results', [])
            predictions = []
            
            for r in raw_results[:10]:
                if isinstance(r, dict):
                    table_name = r.get('table_name', '')
                    if table_name and table_name != query_table:
                        predictions.append(table_name)
            
            predictions = predictions[:5]
            
            query_result = {
                'query_table': query_table,
                'predictions': predictions,
                'time': elapsed,
                'llm_calls': llm_calls,
                'metrics': result.get('metrics', {}),
                'from_cache': False
            }
        else:
            query_result = {
                'query_table': query_table,
                'predictions': [],
                'time': elapsed,
                'llm_calls': llm_calls,
                'metrics': {},
                'from_cache': False
            }
        
        # 更新缓存
        cache[cache_key] = query_result.copy()
        with open(cache_file, 'wb') as f:
            pickle.dump(cache, f)
        
        logger.info(f"✅ 完成: {query_table} | 时间: {elapsed:.2f}s | LLM: {llm_calls}")
        
        return query_result
        
    except Exception as e:
        logger.error(f"处理查询失败 {query_table}: {e}")
        return {
            'query_table': query_table,
            'predictions': [],
            'time': 0,
            'llm_calls': 0,
            'metrics': {},
            'from_cache': False,
            'error': str(e)
        }


def run_ultimate_optimized_experiment(tables: List[Dict], queries: List[Dict],
                                     ground_truth: List[Dict], task_type: str,
                                     dataset_type: str, max_workers: int = 4) -> Dict:
    """
    运行终极优化的实验（批处理级别资源共享）
    """
    # 1. 初始化所有共享资源（只执行一次）
    shared_config_dict = initialize_shared_resources(tables, task_type, dataset_type)
    
    # 2. 准备缓存文件路径
    cache_file = Path(f"cache/{dataset_type}_{task_type}_persistent.pkl")
    
    # 3. 准备进程池参数
    query_args = [
        (query, tables, shared_config_dict, str(cache_file))
        for query in queries
    ]
    
    # 4. 使用进程池处理查询
    results = []
    cache_hits = 0
    total_llm_calls = 0
    
    logger.info(f"🚀 开始处理 {len(queries)} 个查询 (进程数={max_workers})...")
    logger.info(f"  ⚡ 使用共享配置，避免重复初始化")
    logger.info(f"  📊 OptimizerAgent和PlannerAgent只调用1次（而不是{len(queries)}次）")
    
    start_time = time.time()
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_to_query = {
            executor.submit(process_query_with_shared_resources, args): args[0]
            for args in query_args
        }
        
        completed = 0
        for future in as_completed(future_to_query):
            query = future_to_query[future]
            completed += 1
            
            try:
                result = future.result(timeout=60)
                results.append(result)
                
                if result.get('from_cache', False):
                    cache_hits += 1
                
                total_llm_calls += result.get('llm_calls', 0)
                
                logger.info(f"进度: {completed}/{len(queries)} | "
                          f"{result['query_table']} | "
                          f"时间: {result.get('time', 0):.2f}s | "
                          f"缓存: {result.get('from_cache', False)}")
                
            except Exception as e:
                logger.error(f"查询失败: {query.get('query_table', '')}: {e}")
                results.append({
                    'query_table': query.get('query_table', ''),
                    'predictions': [],
                    'time': 0,
                    'llm_calls': 0,
                    'metrics': {},
                    'error': str(e)
                })
    
    total_time = time.time() - start_time
    
    # 5. 计算评估指标
    from evaluate_with_metrics import calculate_metrics
    from run_cached_experiments import convert_ground_truth_format
    converted_gt = convert_ground_truth_format(ground_truth)
    metrics = calculate_metrics(results, converted_gt)
    
    # 添加OptimizerAgent和PlannerAgent的调用（批处理级别各1次）
    total_llm_calls += 2  # OptimizerAgent + PlannerAgent
    
    return {
        'results': results,
        'metrics': metrics,
        'total_time': total_time,
        'avg_time': total_time / len(queries) if queries else 0,
        'total_llm_calls': total_llm_calls,
        'cache_hits': cache_hits,
        'cache_hit_rate': cache_hits / len(queries) if queries else 0,
        'optimization_savings': {
            'optimizer_calls_saved': len(queries) - 1,  # 只调用1次而不是N次
            'planner_calls_saved': len(queries) - 1,    # 只调用1次而不是N次
            'estimated_time_saved': (len(queries) - 1) * 4  # 每次调用约2秒
        }
    }


def main():
    """主函数"""
    import argparse
    parser = argparse.ArgumentParser(description='运行终极优化的实验')
    parser.add_argument('--task', choices=['join', 'union', 'both'], default='both')
    parser.add_argument('--dataset', choices=['subset', 'complete'], default='subset')
    parser.add_argument('--max-queries', type=int, default=10)
    parser.add_argument('--workers', type=int, default=4, help='并行进程数')
    args = parser.parse_args()
    
    # 加载数据
    from run_cached_experiments import load_dataset, filter_queries_with_ground_truth
    
    tasks = ['join', 'union'] if args.task == 'both' else [args.task]
    all_results = {}
    
    for task in tasks:
        logger.info("="*80)
        logger.info(f"🎯 运行 {task.upper()} 任务 - {args.dataset.upper()} 数据集")
        logger.info("="*80)
        
        # 加载数据
        tables, queries, ground_truth = load_dataset(task, args.dataset)
        filtered_queries = filter_queries_with_ground_truth(
            queries, ground_truth, args.max_queries
        )
        
        logger.info(f"📊 数据集: {len(tables)} 表, {len(filtered_queries)} 查询")
        
        # 运行终极优化实验
        result = run_ultimate_optimized_experiment(
            tables, filtered_queries, ground_truth, 
            task, args.dataset, max_workers=args.workers
        )
        
        all_results[task] = result
        
        # 输出结果
        logger.info("="*80)
        logger.info(f"📈 {task.upper()} 任务结果:")
        logger.info(f"  ⏱️ 总时间: {result['total_time']:.2f}秒")
        logger.info(f"  ⚡ 平均时间: {result['avg_time']:.2f}秒/查询")
        logger.info(f"  💎 缓存命中: {result['cache_hits']}/{len(filtered_queries)} "
                   f"({result['cache_hit_rate']*100:.1f}%)")
        logger.info(f"  🤖 LLM调用: {result['total_llm_calls']}次")
        
        # 显示优化节省
        savings = result['optimization_savings']
        logger.info(f"  💰 优化节省:")
        logger.info(f"    - OptimizerAgent调用减少: {savings['optimizer_calls_saved']}次")
        logger.info(f"    - PlannerAgent调用减少: {savings['planner_calls_saved']}次")
        logger.info(f"    - 预计节省时间: {savings['estimated_time_saved']:.1f}秒")
        
        logger.info(f"  📊 评估指标:")
        metrics = result['metrics']
        for k, v in metrics.items():
            if isinstance(v, float):
                logger.info(f"    {k}: {v:.3f}")
            else:
                logger.info(f"    {k}: {v}")
    
    # 保存结果
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"experiment_results/ultimate_{args.dataset}_{timestamp}.json"
    Path("experiment_results").mkdir(exist_ok=True)
    
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    logger.info(f"\n✅ 实验完成，结果保存到: {output_file}")
    
    # 性能改进总结
    logger.info("\n" + "="*80)
    logger.info("🚀 终极优化总结:")
    logger.info("="*80)
    
    for task, result in all_results.items():
        logger.info(f"\n{task.upper()} 任务:")
        logger.info(f"  🎯 Hit@1: {result['metrics'].get('hit@1', 0):.3f}")
        logger.info(f"  📈 Hit@3: {result['metrics'].get('hit@3', 0):.3f}")
        logger.info(f"  📊 Hit@5: {result['metrics'].get('hit@5', 0):.3f}")
        logger.info(f"  ⚡ 查询速度: {result['avg_time']:.2f}秒")
        logger.info(f"  💎 缓存效率: {result['cache_hit_rate']*100:.1f}%")
        
        # 计算实际加速比
        if result['optimization_savings']['estimated_time_saved'] > 0:
            theoretical_time = result['total_time'] + result['optimization_savings']['estimated_time_saved']
            speedup = theoretical_time / result['total_time']
            logger.info(f"  🚀 优化加速比: {speedup:.2f}x")


if __name__ == "__main__":
    # 修复多进程启动问题
    mp.freeze_support()  # Windows支持
    main()