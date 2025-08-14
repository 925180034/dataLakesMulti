#!/usr/bin/env python
"""
运行带缓存优化的完整实验
通过缓存OptimizerAgent和PlannerAgent的结果，大幅减少LLM调用
"""
import os
import sys
import json
import time
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any
import subprocess

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent))

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_dataset(task_type: str, dataset_type: str) -> tuple:
    """加载数据集"""
    base_dir = Path(f'examples/separated_datasets/{task_type}_{dataset_type}')
    
    with open(base_dir / 'tables.json', 'r') as f:
        tables = json.load(f)
    with open(base_dir / 'queries.json', 'r') as f:
        queries = json.load(f)
    with open(base_dir / 'ground_truth.json', 'r') as f:
        ground_truth = json.load(f)
    
    return tables, queries, ground_truth


def filter_queries_with_ground_truth(queries: List[Dict], ground_truth: List[Dict], 
                                    max_queries: int = 100) -> List[Dict]:
    """只选择有ground truth的查询"""
    gt_query_tables = set()
    for gt in ground_truth:
        if 'query_table' in gt:
            gt_query_tables.add(gt['query_table'])
    
    filtered = []
    seen = set()
    for q in queries:
        qt = q.get('query_table', '')
        if qt and qt in gt_query_tables and qt not in seen:
            filtered.append(q)
            seen.add(qt)
            if len(filtered) >= max_queries:
                break
    
    return filtered


def precompute_embeddings(tables: List[Dict], dataset_type: str):
    """预计算向量嵌入"""
    cache_dir = Path("cache") / dataset_type
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    index_file = cache_dir / f"vector_index_{len(tables)}.pkl"
    embeddings_file = cache_dir / f"table_embeddings_{len(tables)}.pkl"
    
    if index_file.exists() and embeddings_file.exists():
        logger.info(f"✅ 向量索引已存在: {index_file}")
        os.environ['USE_PERSISTENT_INDEX'] = str(index_file)
        os.environ['USE_PRECOMPUTED_EMBEDDINGS'] = str(embeddings_file)
    else:
        logger.info(f"⚠️ 需要生成向量索引...")
        from precompute_embeddings import precompute_all_embeddings
        precompute_all_embeddings(tables, dataset_type)
        os.environ['USE_PERSISTENT_INDEX'] = str(index_file)
        os.environ['USE_PRECOMPUTED_EMBEDDINGS'] = str(embeddings_file)


def convert_ground_truth_format(ground_truth_list: List[Dict]) -> List[Dict]:
    """
    将ground truth从单个候选表格式转换为列表格式
    并按查询表聚合，同时过滤自匹配
    """
    query_to_candidates = {}
    
    for item in ground_truth_list:
        query_table = item.get('query_table', '')
        candidate_table = item.get('candidate_table', '')
        
        if query_table and candidate_table:
            # 过滤自匹配
            if query_table != candidate_table:
                if query_table not in query_to_candidates:
                    query_to_candidates[query_table] = set()
                query_to_candidates[query_table].add(candidate_table)
    
    # 转换为期望的格式
    converted = []
    for query_table, candidates in query_to_candidates.items():
        if candidates:  # 只保留有候选表的
            converted.append({
                'query_table': query_table,
                'candidate_tables': list(candidates)
            })
    
    return converted


def run_multi_agent_with_cache(tables: List[Dict], queries: List[Dict], 
                               ground_truth: List[Dict], task_type: str) -> Dict:
    """运行多智能体系统（带缓存）"""
    from src.core.langgraph_workflow import DataLakeDiscoveryWorkflow
    
    # 创建工作流实例（保持单例以利用缓存）
    workflow = DataLakeDiscoveryWorkflow()
    
    results = []
    total_time = 0
    total_llm_calls = 0
    cache_hits = 0
    
    logger.info(f"开始处理 {len(queries)} 个查询...")
    
    for i, query in enumerate(queries, 1):
        query_table_name = query.get('query_table', '')
        
        if i == 1:
            logger.info(f"🔥 第一个查询将初始化缓存（{task_type} 任务）...")
        elif i == 2:
            logger.info(f"✨ 从第二个查询开始使用缓存...")
        
        start = time.time()
        
        # 运行工作流
        result = workflow.run(
            query=f"find {task_type}able tables for {query_table_name}",
            tables=tables,
            task_type=task_type,
            query_table_name=query_table_name
        )
        
        elapsed = time.time() - start
        total_time += elapsed
        
        # 收集指标
        if result.get('success'):
            # 获取预测结果并过滤自匹配
            predictions = [r['table_name'] for r in result.get('results', [])[:10]]
            filtered_predictions = [p for p in predictions if p != query_table_name]
            
            results.append({
                'query_table': query_table_name,
                'predictions': filtered_predictions[:5],  # 保留top-5
                'time': elapsed
            })
            
            # 检查是否使用了缓存
            metrics = result.get('metrics', {})
            agent_times = metrics.get('agent_times', {})
            
            # 如果优化器和规划器时间极短，说明使用了缓存
            if agent_times.get('optimizer', 1) < 0.01 and agent_times.get('planner', 1) < 0.01:
                cache_hits += 1
            
            total_llm_calls += metrics.get('llm_calls_made', 0)
        
        # 进度报告
        if i % 10 == 0:
            avg_time = total_time / i
            logger.info(f"进度: {i}/{len(queries)} | 平均时间: {avg_time:.2f}s | 缓存命中: {cache_hits}/{i}")
    
    # 转换ground truth格式
    converted_ground_truth = convert_ground_truth_format(ground_truth)
    logger.info(f"Ground truth转换: {len(ground_truth)} 条 -> {len(converted_ground_truth)} 个查询表")
    
    # 计算评估指标
    from evaluate_with_metrics import calculate_metrics
    metrics = calculate_metrics(results, converted_ground_truth)
    
    return {
        'results': results,
        'metrics': metrics,
        'total_time': total_time,
        'avg_time': total_time / len(queries) if queries else 0,
        'total_llm_calls': total_llm_calls,
        'cache_hits': cache_hits,
        'cache_hit_rate': cache_hits / len(queries) if queries else 0
    }


def main():
    """主函数"""
    import argparse
    parser = argparse.ArgumentParser(description='运行带缓存的完整实验')
    parser.add_argument('--task', choices=['join', 'union', 'both'], default='both',
                       help='任务类型')
    parser.add_argument('--dataset', choices=['subset', 'complete'], default='complete',
                       help='数据集类型')
    parser.add_argument('--max-queries', type=int, default=100,
                       help='最大查询数')
    args = parser.parse_args()
    
    # 设置环境变量
    os.environ['SKIP_LLM'] = 'false'  # 绝不跳过LLM
    
    # 运行实验
    tasks = ['join', 'union'] if args.task == 'both' else [args.task]
    
    all_results = {}
    
    for task in tasks:
        logger.info("="*80)
        logger.info(f"运行 {task.upper()} 任务 - {args.dataset.upper()} 数据集")
        logger.info("="*80)
        
        # 加载数据
        tables, queries, ground_truth = load_dataset(task, args.dataset)
        logger.info(f"数据集: {len(tables)} 个表, {len(queries)} 个查询")
        
        # 过滤查询
        filtered_queries = filter_queries_with_ground_truth(queries, ground_truth, args.max_queries)
        logger.info(f"过滤后: {len(filtered_queries)} 个有ground truth的查询")
        
        # 预计算嵌入
        precompute_embeddings(tables, args.dataset)
        
        # 运行实验
        result = run_multi_agent_with_cache(tables, filtered_queries, ground_truth, task)
        
        # 保存结果
        all_results[task] = result
        
        # 输出总结
        logger.info("="*80)
        logger.info(f"{task.upper()} 任务结果:")
        logger.info(f"  总时间: {result['total_time']:.2f}秒")
        logger.info(f"  平均时间: {result['avg_time']:.2f}秒/查询")
        logger.info(f"  缓存命中: {result['cache_hits']}/{len(filtered_queries)} ({result['cache_hit_rate']*100:.1f}%)")
        logger.info(f"  LLM调用: {result['total_llm_calls']}次")
        logger.info(f"  评估指标:")
        
        metrics = result['metrics']
        for k, v in metrics.items():
            if isinstance(v, float):
                logger.info(f"    {k}: {v:.3f}")
            else:
                logger.info(f"    {k}: {v}")
    
    # 保存完整结果
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"experiment_results/cached_experiment_{args.dataset}_{timestamp}.json"
    Path("experiment_results").mkdir(exist_ok=True)
    
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    logger.info(f"\n✅ 实验完成，结果保存到: {output_file}")
    
    # 显示缓存效率总结
    logger.info("\n" + "="*80)
    logger.info("缓存效率总结:")
    logger.info("="*80)
    
    for task, result in all_results.items():
        if result['cache_hits'] > 0:
            # 估算节省的时间（假设每个优化器+规划器调用需要5秒）
            saved_time = result['cache_hits'] * 5
            logger.info(f"{task.upper()} 任务:")
            logger.info(f"  缓存命中率: {result['cache_hit_rate']*100:.1f}%")
            logger.info(f"  估计节省时间: {saved_time:.1f}秒")
            logger.info(f"  实际加速比: {(result['total_time'] + saved_time) / result['total_time']:.1f}x")


if __name__ == "__main__":
    main()