#!/usr/bin/env python
"""
优化版统一实验脚本 - 真正的共享资源实现
核心改进：
1. 完全按照用户参数决定进程数，不做任何限制
2. 主进程初始化一次，子进程通过pickle共享
3. LLM在主进程批量处理，避免子进程初始化
4. 支持任意数量的worker进程（1-128或更多）
"""

import os
import sys
import json
import time
import pickle
import logging
import argparse
import multiprocessing as mp
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Tuple, Optional, Any
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# 设置环境变量
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['NUMEXPR_MAX_THREADS'] = '16'
os.environ['USE_REAL_EMBEDDINGS'] = 'true'

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 全局变量：子进程资源缓存
_process_cache = None

def load_shared_resources(cache_file: str) -> Dict:
    """子进程加载共享资源（每个进程只加载一次）"""
    global _process_cache
    
    if _process_cache is None:
        logger.debug(f"进程 {os.getpid()} 加载共享资源...")
        with open(cache_file, 'rb') as f:
            _process_cache = pickle.load(f)
    
    return _process_cache

def initialize_resources_in_main(tables: List[Dict], dataset_name: str, 
                                task_type: str, layer: str) -> Dict:
    """
    主进程一次性初始化所有共享资源
    返回序列化的缓存文件路径
    """
    logger.info("="*80)
    logger.info("🚀 主进程初始化共享资源")
    logger.info("="*80)
    
    start_time = time.time()
    
    # 缓存目录
    cache_dir = Path("cache") / dataset_name
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    # 缓存key
    cache_key = f"{dataset_name}_{task_type}_{len(tables)}_{layer}"
    cache_file = cache_dir / f"shared_{cache_key}.pkl"
    
    # 检查是否已有缓存
    if cache_file.exists():
        logger.info(f"📦 发现现有缓存: {cache_file}")
        return {'cache_file': str(cache_file)}
    
    shared_data = {
        'dataset_name': dataset_name,
        'task_type': task_type,
        'table_count': len(tables),
        'layer': layer
    }
    
    # Layer 1: 元数据过滤器
    if 'L1' in layer:
        logger.info("🔍 初始化元数据过滤器...")
        from src.tools.smd_enhanced_metadata_filter import SMDEnhancedMetadataFilter
        metadata_filter = SMDEnhancedMetadataFilter()
        metadata_filter.build_index(tables)
        shared_data['metadata_filter'] = metadata_filter
    
    # Layer 2: 向量嵌入
    if 'L2' in layer:
        logger.info("📊 加载向量嵌入...")
        embeddings_file = cache_dir / f"table_embeddings_{len(tables)}.pkl"
        
        if not embeddings_file.exists():
            logger.info("⚙️ 生成嵌入向量...")
            from precompute_embeddings import precompute_all_embeddings
            precompute_all_embeddings(tables, dataset_name)
        
        with open(embeddings_file, 'rb') as f:
            shared_data['table_embeddings'] = pickle.load(f)
        
        # 尝试加载HNSW索引
        index_file = cache_dir / f"vector_index_{len(tables)}.pkl"
        if index_file.exists():
            with open(index_file, 'rb') as f:
                shared_data['vector_index'] = pickle.load(f)
        else:
            shared_data['vector_index'] = None
    
    # Layer 3: LLM配置（不在子进程中初始化）
    if 'L3' in layer:
        logger.info("🤖 配置LLM参数...")
        shared_data['llm_config'] = {
            'enable': True,
            'confidence_threshold': 0.5,
            'max_candidates': 10
        }
    
    # 序列化所有资源
    logger.info(f"💾 保存共享资源到: {cache_file}")
    with open(cache_file, 'wb') as f:
        pickle.dump(shared_data, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    elapsed = time.time() - start_time
    logger.info(f"✅ 初始化完成，耗时: {elapsed:.2f} 秒")
    
    return {'cache_file': str(cache_file)}

def process_query_worker(args: Tuple) -> Dict:
    """
    子进程工作函数 - 处理单个查询
    根据层级执行不同的处理逻辑
    """
    query, tables, cache_info, layer = args
    
    # 加载共享资源
    resources = load_shared_resources(cache_info['cache_file'])
    
    query_table_name = query.get('query_table', '')
    
    # 找到查询表
    query_table = None
    for t in tables:
        if t.get('name', t.get('table_name')) == query_table_name:
            query_table = t
            break
    
    if not query_table:
        return {
            'query_table': query_table_name,
            'predictions': [],
            'needs_llm': False
        }
    
    predictions = []
    
    # Layer 1: 元数据过滤
    if 'L1' in layer and 'metadata_filter' in resources:
        metadata_filter = resources['metadata_filter']
        candidates = metadata_filter.filter_candidates(
            query_table,
            threshold=0.5,
            max_candidates=100 if 'L2' in layer else 20
        )
        predictions = [(name, score) for name, score in candidates 
                      if name != query_table_name]
    
    # Layer 2: 向量相似度重排序
    if 'L2' in layer and predictions and 'table_embeddings' in resources:
        table_embeddings = resources['table_embeddings']
        
        if query_table_name in table_embeddings:
            query_embedding = np.array(table_embeddings[query_table_name])
            
            reranked = []
            for name, l1_score in predictions:
                if name in table_embeddings:
                    cand_embedding = np.array(table_embeddings[name])
                    # 余弦相似度
                    sim = np.dot(query_embedding, cand_embedding) / (
                        np.linalg.norm(query_embedding) * np.linalg.norm(cand_embedding) + 1e-8
                    )
                    # 组合分数
                    combined_score = 0.3 * l1_score + 0.7 * float(sim)
                    reranked.append((name, combined_score))
            
            reranked.sort(key=lambda x: x[1], reverse=True)
            predictions = reranked[:20]
    
    # 准备结果
    result = {
        'query_table': query_table_name,
        'predictions': [name for name, _ in predictions[:20]],
        'needs_llm': 'L3' in layer  # 标记是否需要LLM验证
    }
    
    # 如果需要L3，返回更多信息供主进程处理
    if result['needs_llm']:
        result['query_table_obj'] = query_table
        result['candidate_scores'] = predictions[:10]  # 保留分数供LLM使用
    
    return result

def batch_llm_verification(results: List[Dict], tables: List[Dict], 
                          task_type: str) -> List[Dict]:
    """
    主进程批量进行LLM验证
    避免在子进程中初始化LLM
    """
    if not any(r.get('needs_llm', False) for r in results):
        return results
    
    logger.info("🤖 开始批量LLM验证...")
    
    # 延迟导入，只在需要时初始化
    from src.tools.llm_matcher import LLMMatcherTool
    
    # 初始化LLM（只在主进程中初始化一次）
    llm_matcher = LLMMatcherTool()
    
    # 创建表名到表对象的映射
    table_dict = {t.get('name', t.get('table_name')): t for t in tables}
    
    # 批量处理需要LLM验证的结果
    for result in results:
        if not result.get('needs_llm', False):
            continue
        
        query_table = result.get('query_table_obj')
        if not query_table:
            continue
        
        candidate_scores = result.get('candidate_scores', [])
        if not candidate_scores:
            continue
        
        # LLM重新评分
        final_predictions = []
        for cand_name, base_score in candidate_scores[:5]:  # 只验证前5个
            if cand_name in table_dict:
                cand_table = table_dict[cand_name]
                try:
                    # 使用LLM验证
                    is_match, confidence = llm_matcher.verify_match(
                        query_table, cand_table, task_type
                    )
                    if is_match and confidence > 0.5:
                        final_predictions.append(cand_name)
                except:
                    # LLM失败时保留原始预测
                    if base_score > 0.6:
                        final_predictions.append(cand_name)
        
        # 如果LLM筛选太严格，保留一些高分原始预测
        if len(final_predictions) < 3:
            for cand_name, score in candidate_scores[:5]:
                if cand_name not in final_predictions and score > 0.7:
                    final_predictions.append(cand_name)
        
        # 更新结果
        result['predictions'] = final_predictions[:20]
        
        # 清理临时数据
        result.pop('query_table_obj', None)
        result.pop('candidate_scores', None)
        result.pop('needs_llm', None)
    
    logger.info("✅ LLM验证完成")
    return results

def run_experiment(dataset_name: str, task_type: str, layer: str,
                   tables: List[Dict], queries: List[Dict],
                   max_queries: Optional[int] = None,
                   workers: int = None) -> Tuple[Dict, float]:
    """
    运行实验主函数
    
    Args:
        workers: 工作进程数，None表示使用CPU核心数
    """
    # 确定工作进程数
    if workers is None:
        workers = mp.cpu_count()
    
    # 限制查询数
    if max_queries and max_queries < len(queries):
        queries = queries[:max_queries]
        logger.info(f"📊 限制查询数到 {max_queries}")
    
    logger.info(f"\n{'='*80}")
    logger.info(f"🚀 运行实验: {dataset_name} - {task_type} - {layer}")
    logger.info(f"  表数量: {len(tables)}")
    logger.info(f"  查询数: {len(queries)}")
    logger.info(f"  CPU核心: {mp.cpu_count()}")
    logger.info(f"  使用进程: {workers}")
    logger.info(f"{'='*80}")
    
    # 主进程初始化共享资源
    cache_info = initialize_resources_in_main(tables, dataset_name, task_type, layer)
    
    # 准备进程池参数
    process_args = [
        (query, tables, cache_info, layer)
        for query in queries
    ]
    
    # 使用进程池处理
    results = []
    start_time = time.time()
    
    logger.info(f"🔄 启动 {workers} 个进程处理 {len(queries)} 个查询...")
    
    with ProcessPoolExecutor(max_workers=workers) as executor:
        # 提交所有任务
        futures = [executor.submit(process_query_worker, args) 
                  for args in process_args]
        
        # 收集结果
        completed = 0
        for future in as_completed(futures):
            try:
                result = future.result(timeout=60)
                results.append(result)
                
                completed += 1
                if completed % 50 == 0 or completed == len(queries):
                    elapsed = time.time() - start_time
                    rate = completed / elapsed if elapsed > 0 else 0
                    remaining = len(queries) - completed
                    eta = remaining / rate if rate > 0 else 0
                    logger.info(f"  进度: {completed}/{len(queries)} "
                              f"({100*completed/len(queries):.1f}%) "
                              f"速率: {rate:.1f} q/s "
                              f"剩余: {eta:.1f}s")
            except Exception as e:
                logger.error(f"处理查询失败: {e}")
                results.append({
                    'query_table': 'error',
                    'predictions': []
                })
    
    # 如果需要L3层，在主进程批量处理LLM验证
    if 'L3' in layer:
        results = batch_llm_verification(results, tables, task_type)
    
    # 转换为字典格式
    final_results = {
        r['query_table']: r['predictions'] 
        for r in results
    }
    
    elapsed_time = time.time() - start_time
    
    logger.info(f"✅ 实验完成!")
    logger.info(f"  总耗时: {elapsed_time:.2f} 秒")
    logger.info(f"  平均速度: {len(queries)/elapsed_time:.2f} 查询/秒")
    logger.info(f"  结果数: {len(final_results)}")
    
    return final_results, elapsed_time

def calculate_metrics(predictions: Dict, ground_truth: Dict) -> Dict:
    """计算评估指标"""
    hit_at_k = {1: 0, 3: 0, 5: 0, 10: 0, 20: 0}
    total = 0
    
    for query_table, pred_list in predictions.items():
        if query_table not in ground_truth:
            continue
        
        expected = set(ground_truth[query_table])
        total += 1
        
        for k in hit_at_k.keys():
            if any(p in expected for p in pred_list[:k]):
                hit_at_k[k] += 1
    
    # 计算百分比
    if total > 0:
        for k in hit_at_k.keys():
            hit_at_k[k] = hit_at_k[k] / total
    
    return {
        f'hit@{k}': v for k, v in hit_at_k.items()
    }

def main():
    parser = argparse.ArgumentParser(description='优化版统一实验脚本')
    
    parser.add_argument('--dataset',
                       choices=['nlctables', 'opendata', 'webtable', 'all'],
                       required=True,
                       help='数据集名称')
    
    parser.add_argument('--task',
                       choices=['join', 'union', 'both'],
                       default='join',
                       help='任务类型')
    
    parser.add_argument('--dataset-type',
                       choices=['subset', 'complete'],
                       default='subset',
                       help='数据集类型')
    
    parser.add_argument('--max-queries',
                       type=int,
                       default=None,
                       help='最大查询数')
    
    parser.add_argument('--workers',
                       type=int,
                       default=None,
                       help='工作进程数（默认使用所有CPU核心）')
    
    parser.add_argument('--layer',
                       choices=['L1', 'L1+L2', 'L1+L2+L3', 'all'],
                       default='all',
                       help='运行层级')
    
    args = parser.parse_args()
    
    # 显示配置
    logger.info("="*80)
    logger.info("⚙️ 实验配置")
    logger.info(f"  数据集: {args.dataset}")
    logger.info(f"  任务: {args.task}")
    logger.info(f"  类型: {args.dataset_type}")
    logger.info(f"  层级: {args.layer}")
    logger.info(f"  查询数: {args.max_queries if args.max_queries else '全部'}")
    logger.info(f"  进程数: {args.workers if args.workers else f'自动 ({mp.cpu_count()})'}")
    logger.info("="*80)
    
    # 处理数据集选择
    if args.dataset == 'all':
        datasets = ['nlctables', 'opendata', 'webtable']
    else:
        datasets = [args.dataset]
    
    # 运行实验
    all_results = {}
    
    for dataset in datasets:
        logger.info(f"\n{'#'*80}")
        logger.info(f"# 数据集: {dataset}")
        logger.info('#'*80)
        
        if args.task == 'both':
            for task in ['join', 'union']:
                logger.info(f"\n📋 任务: {task}")
                
                # 加载数据
                data_dir = Path("examples") / dataset / f"{task}_{args.dataset_type}"
                if not data_dir.exists():
                    logger.warning(f"⚠️ 数据目录不存在: {data_dir}")
                    continue
                
                with open(data_dir / "tables.json", 'r') as f:
                    tables = json.load(f)
                with open(data_dir / "queries.json", 'r') as f:
                    queries = json.load(f)
                
                # 确保表有name字段
                for t in tables:
                    if 'name' not in t and 'table_name' in t:
                        t['name'] = t['table_name']
                
                # 运行实验
                if args.layer == 'all':
                    for layer in ['L1', 'L1+L2', 'L1+L2+L3']:
                        logger.info(f"\n🔸 层级: {layer}")
                        results, elapsed = run_experiment(
                            dataset, task, layer, tables, queries,
                            args.max_queries, args.workers
                        )
                        
                        key = f"{dataset}_{task}_{layer}"
                        all_results[key] = {
                            'results': results,
                            'elapsed': elapsed,
                            'queries': len(results)
                        }
                        
                        # 计算指标
                        gt_file = data_dir / "ground_truth.json"
                        if gt_file.exists():
                            with open(gt_file, 'r') as f:
                                ground_truth = json.load(f)
                            metrics = calculate_metrics(results, ground_truth)
                            all_results[key]['metrics'] = metrics
                            logger.info(f"  指标: {metrics}")
                else:
                    results, elapsed = run_experiment(
                        dataset, task, args.layer, tables, queries,
                        args.max_queries, args.workers
                    )
                    
                    key = f"{dataset}_{task}_{args.layer}"
                    all_results[key] = {
                        'results': results,
                        'elapsed': elapsed,
                        'queries': len(results)
                    }
        else:
            # 单任务
            # 加载数据
            data_dir = Path("examples") / dataset / f"{args.task}_{args.dataset_type}"
            if not data_dir.exists():
                logger.error(f"❌ 数据目录不存在: {data_dir}")
                continue
            
            with open(data_dir / "tables.json", 'r') as f:
                tables = json.load(f)
            with open(data_dir / "queries.json", 'r') as f:
                queries = json.load(f)
            
            # 确保表有name字段
            for t in tables:
                if 'name' not in t and 'table_name' in t:
                    t['name'] = t['table_name']
            
            # 运行实验
            if args.layer == 'all':
                for layer in ['L1', 'L1+L2', 'L1+L2+L3']:
                    logger.info(f"\n🔸 层级: {layer}")
                    results, elapsed = run_experiment(
                        dataset, args.task, layer, tables, queries,
                        args.max_queries, args.workers
                    )
                    
                    key = f"{dataset}_{args.task}_{layer}"
                    all_results[key] = {
                        'results': results,
                        'elapsed': elapsed,
                        'queries': len(results)
                    }
                    
                    # 计算指标
                    gt_file = data_dir / "ground_truth.json"
                    if gt_file.exists():
                        with open(gt_file, 'r') as f:
                            ground_truth = json.load(f)
                        
                        # 转换ground truth格式
                        if isinstance(ground_truth, list):
                            # Union格式
                            gt_dict = {}
                            for item in ground_truth:
                                if 'query_table' in item and 'candidate_tables' in item:
                                    gt_dict[item['query_table']] = item['candidate_tables']
                            ground_truth = gt_dict
                        
                        metrics = calculate_metrics(results, ground_truth)
                        all_results[key]['metrics'] = metrics
                        logger.info(f"  指标: Hit@1={metrics['hit@1']:.3f}, "
                                  f"Hit@5={metrics['hit@5']:.3f}, "
                                  f"Hit@20={metrics['hit@20']:.3f}")
            else:
                results, elapsed = run_experiment(
                    dataset, args.task, args.layer, tables, queries,
                    args.max_queries, args.workers
                )
                
                key = f"{dataset}_{args.task}_{args.layer}"
                all_results[key] = {
                    'results': results,
                    'elapsed': elapsed,
                    'queries': len(results)
                }
    
    # 保存结果
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"experiment_results/shared_{args.dataset}_{args.task}_{args.layer}_{timestamp}.json"
    
    Path("experiment_results").mkdir(exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    logger.info(f"\n💾 结果已保存: {output_file}")
    
    # 打印总结
    logger.info("\n" + "="*80)
    logger.info("📊 实验总结")
    logger.info("="*80)
    
    for key, data in all_results.items():
        logger.info(f"\n{key}:")
        logger.info(f"  查询数: {data['queries']}")
        logger.info(f"  耗时: {data['elapsed']:.2f} 秒")
        logger.info(f"  速度: {data['queries']/data['elapsed']:.2f} q/s")
        if 'metrics' in data:
            logger.info(f"  Hit@1: {data['metrics']['hit@1']:.3f}")
            logger.info(f"  Hit@5: {data['metrics']['hit@5']:.3f}")

if __name__ == "__main__":
    main()