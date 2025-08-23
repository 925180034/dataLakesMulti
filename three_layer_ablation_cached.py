#!/usr/bin/env python
"""
缓存增强版三层架构消融实验脚本
在优化版基础上增加缓存机制，显著加速重复实验
主要优化：
1. 批处理级别资源共享
2. 进程池并行处理
3. 持久化缓存系统（新增）
4. 预计算向量索引
5. 结果缓存和复用（新增）
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

# ================== 缓存管理器 ==================
class CacheManager:
    """统一的缓存管理器"""
    
    def __init__(self, cache_dir: str = "cache/experiment_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.memory_cache = {}  # 内存缓存
        self.stats = {
            'hits': 0,
            'misses': 0,
            'saves': 0
        }
        logger.info(f"✅ 缓存系统初始化: {self.cache_dir}")
    
    def _get_cache_key(self, operation: str, query: Dict, params: Dict = None) -> str:
        """生成缓存键"""
        # 使用操作类型、查询内容和参数生成唯一键
        key_data = {
            'op': operation,
            'query': query.get('query_table', ''),
            'params': params or {}
        }
        key_str = json.dumps(key_data, sort_keys=True)
        return hashlib.md5(key_str.encode()).hexdigest()
    
    def get(self, operation: str, query: Dict, params: Dict = None) -> Optional[Any]:
        """获取缓存结果"""
        cache_key = self._get_cache_key(operation, query, params)
        
        # 先检查内存缓存
        if cache_key in self.memory_cache:
            self.stats['hits'] += 1
            return self.memory_cache[cache_key]
        
        # 检查磁盘缓存
        cache_file = self.cache_dir / f"{operation}_{cache_key}.pkl"
        if cache_file.exists():
            try:
                with open(cache_file, 'rb') as f:
                    result = pickle.load(f)
                    self.memory_cache[cache_key] = result  # 加载到内存
                    self.stats['hits'] += 1
                    return result
            except:
                pass
        
        self.stats['misses'] += 1
        return None
    
    def set(self, operation: str, query: Dict, result: Any, params: Dict = None):
        """保存缓存结果"""
        cache_key = self._get_cache_key(operation, query, params)
        
        # 保存到内存缓存
        self.memory_cache[cache_key] = result
        
        # 保存到磁盘缓存
        cache_file = self.cache_dir / f"{operation}_{cache_key}.pkl"
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(result, f)
            self.stats['saves'] += 1
        except Exception as e:
            logger.warning(f"缓存保存失败: {e}")
    
    def get_stats(self) -> Dict:
        """获取缓存统计"""
        total = self.stats['hits'] + self.stats['misses']
        hit_rate = self.stats['hits'] / total if total > 0 else 0
        return {
            'hits': self.stats['hits'],
            'misses': self.stats['misses'],
            'saves': self.stats['saves'],
            'hit_rate': f"{hit_rate:.1%}",
            'memory_items': len(self.memory_cache)
        }
    
    def clear(self):
        """清空缓存"""
        self.memory_cache.clear()
        for cache_file in self.cache_dir.glob("*.pkl"):
            cache_file.unlink()
        logger.info("缓存已清空")


# 全局缓存管理器
cache_manager = None

def init_cache_manager(cache_dir: str = None):
    """初始化全局缓存管理器"""
    global cache_manager
    if cache_manager is None:
        cache_dir = cache_dir or "cache/experiment_cache"
        cache_manager = CacheManager(cache_dir)
    return cache_manager


def load_dataset(task_type: str, dataset_name: str = 'webtable', dataset_type: str = 'subset') -> tuple:
    """加载数据集
    
    Args:
        task_type: 'join' 或 'union'
        dataset_name: 数据集名称 ('webtable', 'opendata', 或自定义路径)
        dataset_type: 数据集类型 ('subset', 'complete', 'true_subset')
    """
    # 处理数据集路径
    if '/' in dataset_name or dataset_name.startswith('examples'):
        # 直接使用提供的路径
        base_dir = Path(dataset_name)
    else:
        # 构建标准路径
        if dataset_type in ['complete', 'full']:
            suffix = '_complete'
        elif dataset_type == 'true_subset':
            suffix = '_true_subset'
        else:  # subset
            suffix = '_subset'
        
        base_dir = Path(f'examples/{dataset_name}/{task_type}{suffix}')
    
    # 确保路径存在
    if not base_dir.exists():
        logger.error(f"数据集路径不存在: {base_dir}")
        raise FileNotFoundError(f"Dataset path not found: {base_dir}")
    
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


def calculate_metrics(predictions: List[str], ground_truth: List[str], k_values: List[int] = [1, 3, 5]) -> Dict:
    """计算评估指标"""
    if not ground_truth:
        return {
            'precision': 0.0,
            'recall': 0.0,
            'f1_score': 0.0,
            **{f'hit@{k}': 0.0 for k in k_values}
        }
    
    # 计算Hit@K
    hit_at_k = {}
    for k in k_values:
        top_k = predictions[:k] if len(predictions) >= k else predictions
        hit = 1.0 if any(p in ground_truth for p in top_k) else 0.0
        hit_at_k[f'hit@{k}'] = hit
    
    # 计算Precision和Recall
    predictions_set = set(predictions[:5])  # 使用top-5
    ground_truth_set = set(ground_truth)
    
    true_positives = len(predictions_set & ground_truth_set)
    
    precision = true_positives / len(predictions_set) if predictions_set else 0.0
    recall = true_positives / len(ground_truth_set) if ground_truth_set else 0.0
    f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return {
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score,
        **hit_at_k
    }


# ================== Layer 1: 元数据过滤 ==================
def initialize_shared_resources_l1(tables: List[Dict], dataset_type: str) -> Dict:
    """初始化L1层共享资源（预构建SMD索引）"""
    from src.tools.semantic_similarity_simplified import SemanticSearch
    
    metadata_filter = SemanticSearch()
    # 预构建SMD索引以加速查询
    metadata_filter.build_index(tables)
    
    # 序列化以便进程间共享
    smd_index_serialized = pickle.dumps(metadata_filter)
    
    return {
        'smd_index': smd_index_serialized,
        'tables': tables
    }


def process_query_l1(args):
    """处理单个查询 - L1层（使用预构建的SMD索引）"""
    query, smd_index_serialized, tables = args
    
    # 先检查缓存
    global cache_manager
    if cache_manager:
        cached = cache_manager.get('l1', query)
        if cached is not None:
            return cached
    
    # 反序列化SMD索引
    from src.tools.semantic_similarity_simplified import SemanticSearch
    metadata_filter = pickle.loads(smd_index_serialized)
    
    start_time = time.time()
    
    # 使用预构建的索引进行快速查询
    candidates = metadata_filter.search_similar_tables_smd(
        query_table_name=query['query_table'],
        query_columns=[col['name'] for col in query.get('columns', [])],
        top_k=40
    )
    
    predictions = [c['table_name'] for c in candidates if c['table_name'] != query['query_table']]
    
    result = {
        'query_table': query['query_table'],
        'predictions': predictions[:5],
        'time': time.time() - start_time
    }
    
    # 保存到缓存
    if cache_manager:
        cache_manager.set('l1', query, result)
    
    return result


# ================== Layer 2: 向量搜索 ==================
def initialize_shared_resources_l2(tables: List[Dict], task_type: str, dataset_type: str) -> Dict:
    """初始化L1+L2层共享资源"""
    l1_resources = initialize_shared_resources_l1(tables, dataset_type)
    
    # 生成或加载向量索引
    cache_dir = Path(f"cache/ablation_examples/{dataset_type}")
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    vector_index_path = cache_dir / f"vector_index_{len(tables)}.pkl"
    
    if vector_index_path.exists() and not os.environ.get('FORCE_REBUILD'):
        logger.info(f"📂 加载缓存向量索引: {vector_index_path}")
        with open(vector_index_path, 'rb') as f:
            vector_index_serialized = pickle.load(f)
    else:
        logger.info(f"⚙️ 生成新的向量索引...")
        from src.tools.vector_search import VectorSearch
        vector_search = VectorSearch()
        vector_search.index_tables(tables)
        vector_index_serialized = pickle.dumps(vector_search)
        
        with open(vector_index_path, 'wb') as f:
            pickle.dump(vector_index_serialized, f)
        logger.info(f"✅ 创建向量索引: {vector_index_path}")
    
    return {
        **l1_resources,
        'vector_index': vector_index_serialized,
        'task_type': task_type
    }


def process_query_l1_l2(args):
    """处理单个查询 - L1+L2层"""
    query, smd_index_serialized, vector_index_serialized, tables, task_type = args
    
    # 先检查缓存
    global cache_manager
    cache_key_params = {'task_type': task_type}
    if cache_manager:
        cached = cache_manager.get('l1_l2', query, cache_key_params)
        if cached is not None:
            return cached
    
    # 反序列化索引
    from src.tools.semantic_similarity_simplified import SemanticSearch
    from src.tools.vector_search import VectorSearch
    
    metadata_filter = pickle.loads(smd_index_serialized)
    vector_search = pickle.loads(vector_index_serialized)
    
    start_time = time.time()
    
    # L1: 元数据过滤
    l1_candidates = metadata_filter.search_similar_tables_smd(
        query_table_name=query['query_table'],
        query_columns=[col['name'] for col in query.get('columns', [])],
        top_k=40
    )
    
    # 准备候选表进行向量搜索
    candidate_names = [c['table_name'] for c in l1_candidates]
    candidate_tables = [t for t in tables if t['name'] in candidate_names]
    
    # L2: 向量搜索（带任务特定权重）
    if task_type == 'union':
        column_weight, value_weight = 0.5, 0.5
    else:
        column_weight, value_weight = 0.7, 0.3
    
    # 准备查询表
    query_table = {
        'name': query['query_table'],
        'columns': query.get('columns', [])
    }
    
    # 向量搜索
    vector_results = vector_search.search_similar_tables(
        query_table, 
        candidate_tables,
        column_weight=column_weight,
        value_weight=value_weight,
        top_k=5
    )
    
    predictions = [r['table_name'] for r in vector_results if r['table_name'] != query['query_table']]
    
    result = {
        'query_table': query['query_table'],
        'predictions': predictions[:5],
        'time': time.time() - start_time
    }
    
    # 保存到缓存
    if cache_manager:
        cache_manager.set('l1_l2', query, result, cache_key_params)
    
    return result


# ================== Layer 3: LLM验证 ==================  
def initialize_shared_resources_l3(tables: List[Dict], task_type: str, dataset_type: str) -> Dict:
    """初始化L1+L2+L3层共享资源"""
    return initialize_shared_resources_l2(tables, task_type, dataset_type)


def process_query_l1_l2_l3(args):
    """处理单个查询 - L1+L2+L3层（带缓存）"""
    query, smd_index_serialized, vector_index_serialized, tables, task_type = args
    
    # 先检查缓存
    global cache_manager
    cache_key_params = {'task_type': task_type}
    if cache_manager:
        cached = cache_manager.get('l1_l2_l3', query, cache_key_params)
        if cached is not None:
            return cached
    
    # 反序列化索引
    from src.tools.semantic_similarity_simplified import SemanticSearch
    from src.tools.vector_search import VectorSearch
    
    metadata_filter = pickle.loads(smd_index_serialized)
    vector_search = pickle.loads(vector_index_serialized)
    
    start_time = time.time()
    
    # L1: 元数据过滤
    l1_candidates = metadata_filter.search_similar_tables_smd(
        query_table_name=query['query_table'],
        query_columns=[col['name'] for col in query.get('columns', [])],
        top_k=40
    )
    
    # 准备候选表进行向量搜索
    candidate_names = [c['table_name'] for c in l1_candidates]
    candidate_tables = [t for t in tables if t['name'] in candidate_names]
    
    # L2: 向量搜索
    if task_type == 'union':
        column_weight, value_weight = 0.5, 0.5
    else:
        column_weight, value_weight = 0.7, 0.3
    
    query_table = {
        'name': query['query_table'],
        'columns': query.get('columns', [])
    }
    
    vector_results = vector_search.search_similar_tables(
        query_table, 
        candidate_tables,
        column_weight=column_weight,
        value_weight=value_weight,
        top_k=5
    )
    
    # L3: LLM验证（简化版）
    from src.tools.llm_verification_simplified import verify_matches_batch
    
    # 准备LLM验证的候选表
    candidates_for_llm = [
        {
            'name': r['table_name'],
            'columns': next((t['columns'] for t in tables if t['name'] == r['table_name']), [])
        }
        for r in vector_results[:5]
    ]
    
    if candidates_for_llm:
        llm_results = verify_matches_batch(query_table, candidates_for_llm, task_type)
        
        # 基于LLM分数重新排序
        scored_results = []
        for i, candidate in enumerate(candidates_for_llm):
            score = llm_results.get(candidate['name'], 0.0)
            # 结合向量分数和LLM分数
            combined_score = vector_results[i]['score'] * 0.5 + score * 0.5
            scored_results.append({
                'table_name': candidate['name'],
                'score': combined_score
            })
        
        # 按组合分数排序
        scored_results.sort(key=lambda x: x['score'], reverse=True)
        predictions = [r['table_name'] for r in scored_results if r['table_name'] != query['query_table']]
    else:
        predictions = [r['table_name'] for r in vector_results if r['table_name'] != query['query_table']]
    
    result = {
        'query_table': query['query_table'],
        'predictions': predictions[:5],
        'time': time.time() - start_time
    }
    
    # 保存到缓存
    if cache_manager:
        cache_manager.set('l1_l2_l3', query, result, cache_key_params)
    
    return result


def run_experiment(task_type: str, dataset_name: str, dataset_type: str, max_queries: int = None):
    """运行完整的消融实验（带缓存）"""
    logger.info(f"\n{'='*80}")
    logger.info(f"🚀 Running CACHED Ablation Experiment for {task_type.upper()} Task")
    logger.info(f"📂 Dataset: {dataset_name}/{dataset_type}")
    logger.info(f"📊 Max Queries: {max_queries or 'All'}")
    logger.info(f"{'='*80}")
    
    # 初始化缓存管理器
    init_cache_manager(f"cache/experiment_cache/{dataset_name}_{task_type}_{dataset_type}")
    
    # 加载数据
    tables, queries, ground_truth_list = load_dataset(task_type, dataset_name, dataset_type)
    ground_truth = convert_ground_truth_format(ground_truth_list)
    
    logger.info(f"📊 Dataset: {len(tables)} tables, {len(queries)} queries")
    
    # 限制查询数量
    if max_queries:
        queries = queries[:max_queries]
        logger.info(f"📊 使用前{len(queries)}个查询")
    
    # 过滤有效查询
    valid_queries = [q for q in queries if q['query_table'] in ground_truth]
    logger.info(f"📋 Using {len(valid_queries)} queries for {task_type.upper()} task experiment")
    
    # 存储结果
    results = {}
    
    # 设置进程数
    num_processes = min(4, mp.cpu_count())
    
    # ============ L1实验 ============
    logger.info(f"\n{'='*60}")
    logger.info(f"🔬 Running L1 Experiment")
    logger.info(f"{'='*60}")
    
    logger.info("🚀 初始化L1层共享资源...")
    shared_resources_l1 = initialize_shared_resources_l1(tables, dataset_type)
    logger.info("✅ L1层资源初始化完成")
    
    # 准备参数
    l1_args = [
        (q, shared_resources_l1['smd_index'], shared_resources_l1['tables'])
        for q in valid_queries
    ]
    
    # 并行处理
    l1_predictions = []
    l1_times = []
    
    logger.info(f"📊 处理 {len(valid_queries)} 个查询 (进程数={num_processes})...")
    
    with ProcessPoolExecutor(max_workers=num_processes) as executor:
        futures = [executor.submit(process_query_l1, args) for args in l1_args]
        
        for i, future in enumerate(as_completed(futures)):
            result = future.result()
            l1_predictions.append(result)
            l1_times.append(result['time'])
            
            if (i + 1) % 5 == 0:
                logger.info(f"  进度: {i+1}/{len(valid_queries)}")
    
    # 计算L1指标
    l1_metrics = []
    for pred in l1_predictions:
        gt = ground_truth.get(pred['query_table'], [])
        metrics = calculate_metrics(pred['predictions'], gt)
        l1_metrics.append(metrics)
    
    avg_l1_metrics = {
        key: np.mean([m[key] for m in l1_metrics])
        for key in l1_metrics[0].keys()
    }
    
    avg_l1_time = np.mean(l1_times)
    logger.info(f"✅ L1 完成 - 总时间: {sum(l1_times):.2f}秒")
    logger.info(f"📈 L1 - F1: {avg_l1_metrics['f1_score']:.3f}, Hit@1: {avg_l1_metrics['hit@1']:.3f}, Avg Time: {avg_l1_time:.2f}s/query")
    
    results['L1'] = {
        'metrics': avg_l1_metrics,
        'avg_time': avg_l1_time
    }
    
    # ============ L1+L2实验 ============
    logger.info(f"\n{'='*60}")
    logger.info(f"🔬 Running L1+L2 Experiment")
    logger.info(f"{'='*60}")
    
    logger.info("🚀 初始化L1+L2层共享资源...")
    shared_resources_l2 = initialize_shared_resources_l2(tables, task_type, dataset_type)
    logger.info("✅ L1+L2层资源初始化完成")
    
    # 准备参数
    l2_args = [
        (q, shared_resources_l2['smd_index'], shared_resources_l2['vector_index'], 
         shared_resources_l2['tables'], shared_resources_l2['task_type'])
        for q in valid_queries
    ]
    
    # 并行处理
    l2_predictions = []
    l2_times = []
    
    logger.info(f"📊 处理 {len(valid_queries)} 个查询 (进程数={num_processes})...")
    
    with ProcessPoolExecutor(max_workers=num_processes) as executor:
        futures = [executor.submit(process_query_l1_l2, args) for args in l2_args]
        
        for i, future in enumerate(as_completed(futures)):
            result = future.result()
            l2_predictions.append(result)
            l2_times.append(result['time'])
            
            if (i + 1) % 5 == 0:
                logger.info(f"  进度: {i+1}/{len(valid_queries)}")
    
    # 计算L1+L2指标
    l2_metrics = []
    for pred in l2_predictions:
        gt = ground_truth.get(pred['query_table'], [])
        metrics = calculate_metrics(pred['predictions'], gt)
        l2_metrics.append(metrics)
    
    avg_l2_metrics = {
        key: np.mean([m[key] for m in l2_metrics])
        for key in l2_metrics[0].keys()
    }
    
    avg_l2_time = np.mean(l2_times)
    logger.info(f"✅ L1+L2 完成 - 总时间: {sum(l2_times):.2f}秒")
    logger.info(f"📈 L1+L2 - F1: {avg_l2_metrics['f1_score']:.3f}, Hit@1: {avg_l2_metrics['hit@1']:.3f}, Avg Time: {avg_l2_time:.2f}s/query")
    
    results['L1+L2'] = {
        'metrics': avg_l2_metrics,
        'avg_time': avg_l2_time
    }
    
    # ============ L1+L2+L3实验 ============
    if not os.environ.get('SKIP_LLM'):
        logger.info(f"\n{'='*60}")
        logger.info(f"🔬 Running L1+L2+L3 Experiment")
        logger.info(f"{'='*60}")
        
        logger.info("🚀 初始化L1+L2+L3层共享资源...")
        shared_resources_l3 = initialize_shared_resources_l3(tables, task_type, dataset_type)
        logger.info("✅ L1+L2+L3层资源初始化完成")
        
        # 准备参数
        l3_args = [
            (q, shared_resources_l3['smd_index'], shared_resources_l3['vector_index'],
             shared_resources_l3['tables'], shared_resources_l3['task_type'])
            for q in valid_queries
        ]
        
        # 并行处理（减少并发以避免LLM限流）
        l3_predictions = []
        l3_times = []
        
        logger.info(f"📊 处理 {len(valid_queries)} 个查询 (进程数=2)...")
        
        with ProcessPoolExecutor(max_workers=2) as executor:
            futures = [executor.submit(process_query_l1_l2_l3, args) for args in l3_args]
            
            for i, future in enumerate(as_completed(futures)):
                result = future.result()
                l3_predictions.append(result)
                l3_times.append(result['time'])
                
                if (i + 1) % 5 == 0:
                    logger.info(f"  进度: {i+1}/{len(valid_queries)}")
        
        # 计算L1+L2+L3指标
        l3_metrics = []
        for pred in l3_predictions:
            gt = ground_truth.get(pred['query_table'], [])
            metrics = calculate_metrics(pred['predictions'], gt)
            l3_metrics.append(metrics)
        
        avg_l3_metrics = {
            key: np.mean([m[key] for m in l3_metrics])
            for key in l3_metrics[0].keys()
        }
        
        avg_l3_time = np.mean(l3_times)
        logger.info(f"✅ L1+L2+L3 完成 - 总时间: {sum(l3_times):.2f}秒")
        logger.info(f"📈 L1+L2+L3 - F1: {avg_l3_metrics['f1_score']:.3f}, Hit@1: {avg_l3_metrics['hit@1']:.3f}, Avg Time: {avg_l3_time:.2f}s/query")
        
        results['L1+L2+L3'] = {
            'metrics': avg_l3_metrics,
            'avg_time': avg_l3_time
        }
    
    # 打印缓存统计
    logger.info(f"\n{'='*60}")
    logger.info(f"📊 缓存统计")
    logger.info(f"{'='*60}")
    stats = cache_manager.get_stats()
    logger.info(f"缓存命中: {stats['hits']}, 未命中: {stats['misses']}, 命中率: {stats['hit_rate']}")
    logger.info(f"缓存保存: {stats['saves']}, 内存项: {stats['memory_items']}")
    
    return results


def print_results(results: Dict[str, Dict], task_type: str):
    """打印实验结果（格式化表格）"""
    print(f"\n{'='*100}")
    print(f"🚀 CACHED THREE-LAYER ABLATION EXPERIMENT RESULTS")
    print(f"{'='*100}")
    
    print(f"\n{task_type.upper()} Task Results:")
    print("-" * 80)
    print(f"{'Layer Config':<15} {'Hit@1':<10} {'Hit@3':<10} {'Hit@5':<10} {'Precision':<12} {'Recall':<10} {'F1-Score':<10} {'Time(s)':<10}")
    print("-" * 80)
    
    for layer_config, layer_results in results.items():
        metrics = layer_results['metrics']
        print(f"{layer_config:<15} "
              f"{metrics['hit@1']:<10.3f} "
              f"{metrics['hit@3']:<10.3f} "
              f"{metrics['hit@5']:<10.3f} "
              f"{metrics['precision']:<12.3f} "
              f"{metrics['recall']:<10.3f} "
              f"{metrics['f1_score']:<10.3f} "
              f"{layer_results['avg_time']:<10.2f}")
    
    # 分析层贡献
    print(f"\n{'='*100}")
    print(f"📊 LAYER CONTRIBUTION ANALYSIS")
    print(f"{'='*100}")
    
    if 'L1' in results and 'L1+L2' in results:
        l1_f1 = results['L1']['metrics']['f1_score']
        l2_f1 = results['L1+L2']['metrics']['f1_score']
        l2_time = results['L1+L2']['avg_time'] - results['L1']['avg_time']
        
        improvement = ((l2_f1 - l1_f1) / l1_f1 * 100) if l1_f1 > 0 else 0
        print(f"\n{task_type.upper()} Task - Incremental Improvements:")
        print(f"  L2 Contribution: {improvement:+.1f}% F1 improvement, +{l2_time:.2f}s time cost")
        
        if 'L1+L2+L3' in results:
            l3_f1 = results['L1+L2+L3']['metrics']['f1_score']
            l3_time = results['L1+L2+L3']['avg_time'] - results['L1+L2']['avg_time']
            
            l3_improvement = ((l3_f1 - l2_f1) / l2_f1 * 100) if l2_f1 > 0 else 0
            total_improvement = ((l3_f1 - l1_f1) / l1_f1 * 100) if l1_f1 > 0 else 0
            
            print(f"  L3 Contribution: {l3_improvement:+.1f}% F1 improvement, +{l3_time:.2f}s time cost")
            print(f"  Total Improvement: {total_improvement:+.1f}% F1 improvement")
            
            # 成本效益分析
            total_time = results['L1+L2+L3']['avg_time'] - results['L1']['avg_time']
            if total_time > 0:
                cost_benefit = total_improvement / total_time
                print(f"  Cost-Benefit Ratio: {cost_benefit:.2f}% improvement per time unit")
    
    print(f"\n{'='*100}")
    print(f"⚡ OPTIMIZATION SUMMARY")
    print(f"{'='*100}")
    print("Layer Optimizations Applied:")
    print("  🔍 L1 (Metadata): SMD Enhanced filter with reduced table name weight (5%)")
    print("  ⚡ L2 (Vector): Task-specific value similarity (UNION: 50%+50%, JOIN: 70%+30%)")
    print("  🧠 L3 (LLM): Simplified robust verification with enhanced fallback")
    print("\nSystem Optimizations:")
    print("  1. Result caching for repeated experiments")
    print("  2. Batch-level resource sharing (OptimizerAgent & PlannerAgent)")
    print("  3. Process pool parallelization")
    print("  4. Persistent disk and memory caching")
    print("  5. Pre-computed vector indices")
    print(f"{'='*100}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='缓存增强版三层消融实验')
    parser.add_argument('--task', choices=['join', 'union', 'both'], default='join',
                      help='任务类型')
    parser.add_argument('--dataset', type=str, default='webtable',
                      help='数据集名称: webtable, opendata, 或自定义路径')
    parser.add_argument('--dataset-type', choices=['subset', 'complete', 'true_subset'], default='subset',
                      help='数据集类型: subset(子集), complete(完整), true_subset(WebTable的真子集)')
    parser.add_argument('--max-queries', type=int, default=None,
                      help='最大查询数')
    parser.add_argument('--output', type=str, help='输出文件路径')
    parser.add_argument('--simple', action='store_true', help='使用简单查询（不使用挑战性查询）')
    parser.add_argument('--skip-llm', action='store_true', help='跳过L3层LLM验证')
    parser.add_argument('--clear-cache', action='store_true', help='清空缓存后重新运行')
    
    args = parser.parse_args()
    
    if args.skip_llm:
        os.environ['SKIP_LLM'] = 'true'
    
    # 限制查询数
    if args.max_queries:
        logger.info(f"📊 限制最大查询数为: {args.max_queries}")
    
    all_results = {}
    
    if args.task == 'both':
        # 运行两个任务
        for task in ['join', 'union']:
            results = run_experiment(task, args.dataset, args.dataset_type, args.max_queries)
            all_results[task] = results
            print_results(results, task)
    else:
        # 运行单个任务
        results = run_experiment(args.task, args.dataset, args.dataset_type, args.max_queries)
        all_results[args.task] = results
        print_results(results, args.task)
    
    # 保存结果
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(all_results, f, indent=2)
        logger.info(f"📁 结果已保存到: {output_path}")
    
    # 如果设置了清空缓存，在运行结束后清空
    if args.clear_cache and cache_manager:
        cache_manager.clear()


if __name__ == "__main__":
    main()