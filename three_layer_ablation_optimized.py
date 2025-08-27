#!/usr/bin/env python
"""
优化版三层架构消融实验脚本
基于run_ultimate_optimized.py的优化策略
测试L1（元数据过滤）、L2（向量搜索）、L3（LLM验证）各层的贡献
主要优化：
1. 批处理级别资源共享
2. 进程池并行处理
3. 持久化缓存系统
4. 预计算向量索引
5. 全局单例减少初始化
"""
import os
import sys
import json
import time
import hashlib
import logging
import pickle
import yaml
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
# 禁用tokenizers并行以避免fork警告
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

# 可配置的最大预测数量（支持@K计算，K最大为10，设置为20留有余量）
MAX_PREDICTIONS = int(os.environ.get('MAX_PREDICTIONS', '20'))
logger.info(f"📊 MAX_PREDICTIONS set to {MAX_PREDICTIONS} (supports up to @{MAX_PREDICTIONS//2} evaluation)")


def load_task_config(task_type: str, dataset_type: str = None) -> Dict[str, Any]:
    """
    从任务特定的配置文件加载配置
    优先级：
    0. TEMP_CONFIG环境变量指定的临时配置文件（参数优化专用 - 最高优先级）
    1. config_{task}_universal.yml (通用任务配置)
    2. config_{dataset}_{task}.yml (数据集+任务特定配置)
    3. config_{task}_optimized.yml (任务特定优化配置)
    4. config_optimized.yml (通用优化配置)
    5. 默认配置
    
    Args:
        task_type: 'join' 或 'union'
        dataset_type: 'nlctables', 'opendata', 'webtable' (可选，用于日志)
    
    Returns:
        任务特定的配置字典
    """
    import os
    
    # 检查参数优化临时配置文件（最高优先级）
    temp_config_file = os.environ.get('TEMP_CONFIG')
    if temp_config_file and Path(temp_config_file).exists():
        try:
            with open(temp_config_file, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            
            # 提取任务配置
            if 'task_configs' in config and task_type in config['task_configs']:
                task_config = config['task_configs'][task_type]
                
                # 扁平化配置，提取关键参数
                flat_config = {
                    'llm_confidence_threshold': task_config.get('llm_matcher', {}).get('confidence_threshold', 0.05 if task_type == 'join' else 0.03),
                    'aggregator_max_results': task_config.get('aggregator', {}).get('max_results', 20 if task_type == 'join' else 30),
                    'llm_concurrency': 3,
                    'metadata_threshold': task_config.get('metadata_filter', {}).get('column_similarity_threshold', 0.25 if task_type == 'join' else 0.15),
                    'vector_threshold': task_config.get('vector_search', {}).get('similarity_threshold', 0.35 if task_type == 'join' else 0.25),
                    'min_column_overlap': task_config.get('metadata_filter', {}).get('min_column_overlap', 2 if task_type == 'join' else 1),
                    'vector_top_k': task_config.get('vector_search', {}).get('top_k', 60 if task_type == 'join' else 100),
                    'enable_llm': task_config.get('llm_matcher', {}).get('enable_llm', True),
                    # 层组合策略（关键！）
                    'layer_combination': task_config.get('layer_combination', 'weighted_union' if task_type == 'join' else 'union'),
                    # JOIN特定配置
                    'use_column_types': task_config.get('metadata_filter', {}).get('use_column_types', task_type == 'join'),
                    'use_value_overlap': task_config.get('metadata_filter', {}).get('use_value_overlap', task_type == 'join'),
                    'focus_on_join_keys': task_config.get('llm_matcher', {}).get('focus_on_join_keys', task_type == 'join'),
                    # UNION特定配置
                    'allow_subset_matching': task_config.get('metadata_filter', {}).get('allow_subset_matching', task_type == 'union'),
                    'allow_type_coercion': task_config.get('metadata_filter', {}).get('allow_type_coercion', task_type == 'union'),
                    'allow_self_matches': task_config.get('metadata_filter', {}).get('allow_self_match', task_type == 'union'),
                    'focus_on_compatibility': task_config.get('llm_matcher', {}).get('focus_on_compatibility', task_type == 'union'),
                    'check_semantic_similarity': task_config.get('llm_matcher', {}).get('check_semantic_compatibility', task_type == 'union'),
                    'allow_partial_matches': task_config.get('llm_matcher', {}).get('allow_partial_matches', task_type == 'union'),
                    # 其他优化配置
                    'cache_enabled': config.get('optimization_config', {}).get('cache_enabled', True),
                    'parallel_processing': config.get('optimization_config', {}).get('parallel_processing', True),
                    'max_workers': config.get('optimization_config', {}).get('max_workers', 8),
                    # 权重配置 - 从aggregator中提取
                    'metadata_weight': task_config.get('aggregator', {}).get('ranking_weights', {}).get('metadata_score', 0.30 if task_type == 'join' else 0.20),
                    'vector_weight': task_config.get('aggregator', {}).get('ranking_weights', {}).get('vector_score', 0.40 if task_type == 'join' else 0.50),
                    'llm_weight': task_config.get('aggregator', {}).get('ranking_weights', {}).get('llm_score', 0.30),
                    # 值相似性权重
                    'value_similarity_weight': task_config.get('llm_matcher', {}).get('value_similarity_weight', 0.5 if task_type == 'join' else 0.6),
                    # 语义匹配（UNION特有）
                    'semantic_matching': task_config.get('llm_matcher', {}).get('check_semantic_compatibility', task_type == 'union')
                }
                
                logger.info(f"✅ 加载临时优化配置从 {temp_config_file}")
                logger.info(f"   层组合策略={flat_config.get('layer_combination')}")
                logger.info(f"   阈值: meta={flat_config['metadata_threshold']:.3f}, vec={flat_config['vector_threshold']:.3f}, llm={flat_config['llm_confidence_threshold']:.3f}")
                logger.info(f"   权重: L1={flat_config['metadata_weight']:.2f}, L2={flat_config['vector_weight']:.2f}, L3={flat_config['llm_weight']:.2f}")
                return flat_config
                    
        except Exception as e:
            logger.warning(f"加载临时配置失败: {e}，回退到默认配置")
    
    # 首先尝试加载通用任务配置（第二优先级）
    universal_config_path = Path(f'config_{task_type}_universal.yml')
    if universal_config_path.exists():
        try:
            with open(universal_config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            
            # 提取任务配置
            if 'task_configs' in config and task_type in config['task_configs']:
                task_config = config['task_configs'][task_type]
                
                # 扁平化配置，提取关键参数
                flat_config = {
                    'llm_confidence_threshold': task_config.get('llm_matcher', {}).get('confidence_threshold', 0.05 if task_type == 'join' else 0.03),
                    'aggregator_max_results': task_config.get('aggregator', {}).get('max_results', 20 if task_type == 'join' else 30),
                    'llm_concurrency': 3,
                    'metadata_threshold': task_config.get('metadata_filter', {}).get('column_similarity_threshold', 0.25 if task_type == 'join' else 0.15),
                    'vector_threshold': task_config.get('vector_search', {}).get('similarity_threshold', 0.35 if task_type == 'join' else 0.25),
                    'min_column_overlap': task_config.get('metadata_filter', {}).get('min_column_overlap', 2 if task_type == 'join' else 1),
                    'vector_top_k': task_config.get('vector_search', {}).get('top_k', 60 if task_type == 'join' else 100),
                    'enable_llm': task_config.get('llm_matcher', {}).get('enable_llm', True),
                    # 层组合策略（关键！）
                    'layer_combination': task_config.get('layer_combination', 'weighted_union' if task_type == 'join' else 'union'),
                    # JOIN特定配置
                    'use_column_types': task_config.get('metadata_filter', {}).get('use_column_types', task_type == 'join'),
                    'use_value_overlap': task_config.get('metadata_filter', {}).get('use_value_overlap', task_type == 'join'),
                    'focus_on_join_keys': task_config.get('llm_matcher', {}).get('focus_on_join_keys', task_type == 'join'),
                    # UNION特定配置
                    'allow_subset_matching': task_config.get('metadata_filter', {}).get('allow_subset_matching', task_type == 'union'),
                    'allow_type_coercion': task_config.get('metadata_filter', {}).get('allow_type_coercion', task_type == 'union'),
                    'allow_self_matches': task_config.get('metadata_filter', {}).get('allow_self_match', task_type == 'union'),
                    'focus_on_compatibility': task_config.get('llm_matcher', {}).get('focus_on_compatibility', task_type == 'union'),
                    'check_semantic_similarity': task_config.get('llm_matcher', {}).get('check_semantic_compatibility', task_type == 'union'),
                    'allow_partial_matches': task_config.get('llm_matcher', {}).get('allow_partial_matches', task_type == 'union'),
                    # 其他优化配置
                    'cache_enabled': config.get('optimization_config', {}).get('cache_enabled', True),
                    'parallel_processing': config.get('optimization_config', {}).get('parallel_processing', True),
                    'max_workers': config.get('optimization_config', {}).get('max_workers', 8),
                    # 权重配置 - 从aggregator中提取
                    'metadata_weight': task_config.get('aggregator', {}).get('ranking_weights', {}).get('metadata_score', 0.30 if task_type == 'join' else 0.20),
                    'vector_weight': task_config.get('aggregator', {}).get('ranking_weights', {}).get('vector_score', 0.40 if task_type == 'join' else 0.50),
                    'llm_weight': task_config.get('aggregator', {}).get('ranking_weights', {}).get('llm_score', 0.30),
                    # 值相似性权重
                    'value_similarity_weight': task_config.get('llm_matcher', {}).get('value_similarity_weight', 0.5 if task_type == 'join' else 0.6),
                    # 语义匹配（UNION特有）
                    'semantic_matching': task_config.get('llm_matcher', {}).get('check_semantic_compatibility', task_type == 'union')
                }
                
                # 获取自适应调整因子（如果有数据集类型）
                adaptive_factors = task_config.get('optimization_config', {}).get('adaptive_adjustment', {}).get('factors', {})
                if dataset_type and adaptive_factors.get('enable'):
                    # 根据数据集类型应用调整因子
                    adjustment_factor = 1.0
                    if dataset_type == 'nlctables':
                        adjustment_factor = adaptive_factors.get('high_quality_data', 1.2) if task_type == 'join' else adaptive_factors.get('clean_data', 1.1)
                    elif dataset_type == 'webtable':
                        adjustment_factor = adaptive_factors.get('noisy_data', 0.8) if task_type == 'join' else adaptive_factors.get('high_diversity', 0.8)
                    elif dataset_type == 'opendata':
                        adjustment_factor = adaptive_factors.get('medium_quality', 1.0) if task_type == 'join' else adaptive_factors.get('has_self_matches', 0.9)
                    
                    # 应用调整因子到阈值
                    if adjustment_factor != 1.0:
                        flat_config['metadata_threshold'] *= adjustment_factor
                        flat_config['vector_threshold'] *= adjustment_factor
                        flat_config['llm_confidence_threshold'] *= adjustment_factor
                        logger.info(f"  📊 应用{dataset_type}自适应因子: {adjustment_factor:.1f}")
                
                logger.info(f"✅ 加载{task_type.upper()}通用配置从config_{task_type}_universal.yml")
                logger.info(f"   层组合策略={flat_config.get('layer_combination')}")
                logger.info(f"   LLM阈值={flat_config['llm_confidence_threshold']:.3f}, 元数据阈值={flat_config['metadata_threshold']:.2f}")
                logger.info(f"   向量阈值={flat_config['vector_threshold']:.2f}, 最大结果={flat_config['aggregator_max_results']}")
                logger.info(f"   权重分配: 元数据={flat_config['metadata_weight']:.1f}, 向量={flat_config['vector_weight']:.1f}, LLM={flat_config['llm_weight']:.1f}")
                if dataset_type:
                    logger.info(f"   数据集: {dataset_type}")
                return flat_config
                    
        except Exception as e:
            logger.warning(f"加载{task_type}通用配置失败: {e}，尝试其他配置")
    
    # 如果通用配置不存在，尝试数据集特定配置
    if dataset_type:
        dataset_task_config_path = Path(f'config_{dataset_type}_{task_type}.yml')
        if dataset_task_config_path.exists():
            try:
                with open(dataset_task_config_path, 'r', encoding='utf-8') as f:
                    config = yaml.safe_load(f)
                
                # 提取任务配置（保留原有逻辑）
                if 'task_configs' in config and task_type in config['task_configs']:
                    task_config = config['task_configs'][task_type]
                    
                    # 扁平化配置
                    flat_config = {
                        'llm_confidence_threshold': task_config.get('llm_matcher', {}).get('confidence_threshold', 0.10 if task_type == 'join' else 0.05),
                        'aggregator_max_results': task_config.get('aggregator', {}).get('max_results', 20 if task_type == 'join' else 30),
                        'llm_concurrency': 3,
                        'metadata_threshold': task_config.get('metadata_filter', {}).get('column_similarity_threshold', 0.35 if task_type == 'join' else 0.20),
                        'vector_threshold': task_config.get('vector_search', {}).get('similarity_threshold', 0.40 if task_type == 'join' else 0.35),
                        'min_column_overlap': task_config.get('metadata_filter', {}).get('min_column_overlap', 2 if task_type == 'join' else 1),
                        'vector_top_k': task_config.get('vector_search', {}).get('top_k', 50 if task_type == 'join' else 60),
                        'enable_llm': task_config.get('llm_matcher', {}).get('enable_llm', True),
                        'layer_combination': task_config.get('layer_combination', 'intersection'),
                        'use_column_types': task_config.get('metadata_filter', {}).get('use_column_types', task_type == 'join'),
                        'use_value_overlap': task_config.get('metadata_filter', {}).get('use_value_overlap', task_type == 'join'),
                        'focus_on_join_keys': task_config.get('llm_matcher', {}).get('focus_on_join_keys', task_type == 'join'),
                        'allow_subset_matching': task_config.get('metadata_filter', {}).get('allow_subset_matching', task_type == 'union'),
                        'allow_type_coercion': task_config.get('metadata_filter', {}).get('allow_type_coercion', task_type == 'union'),
                        'allow_self_matches': task_config.get('llm_matcher', {}).get('allow_self_matches', task_type == 'union'),
                        'focus_on_compatibility': task_config.get('llm_matcher', {}).get('focus_on_compatibility', task_type == 'union'),
                        'check_semantic_similarity': task_config.get('llm_matcher', {}).get('check_semantic_similarity', task_type == 'union'),
                        'allow_partial_matches': task_config.get('llm_matcher', {}).get('allow_partial_matches', task_type == 'union'),
                        'cache_enabled': config.get('optimization_config', {}).get('cache_enabled', True),
                        'parallel_processing': config.get('optimization_config', {}).get('parallel_processing', True),
                        'max_workers': config.get('optimization_config', {}).get('max_workers', 4),
                        'metadata_weight': task_config.get('aggregator', {}).get('ranking_weights', {}).get('metadata_score', 0.25 if task_type == 'join' else 0.20),
                        'vector_weight': task_config.get('aggregator', {}).get('ranking_weights', {}).get('vector_score', 0.35 if task_type == 'join' else 0.40),
                        'llm_weight': task_config.get('aggregator', {}).get('ranking_weights', {}).get('llm_score', 0.40),
                        'value_similarity_weight': task_config.get('llm_matcher', {}).get('value_similarity_weight', 0.5)
                    }
                    
                    logger.info(f"✅ 加载{dataset_type.upper()} {task_type.upper()}配置从config_{dataset_type}_{task_type}.yml")
                    logger.info(f"   层组合策略={flat_config.get('layer_combination')}")
                    logger.info(f"   LLM阈值={flat_config['llm_confidence_threshold']:.3f}, 元数据阈值={flat_config['metadata_threshold']:.2f}")
                    return flat_config
                    
            except Exception as e:
                logger.warning(f"加载{dataset_type} {task_type}特定配置失败: {e}，尝试任务配置")
    
    # 尝试加载任务特定的配置文件
    task_config_path = Path(f'config_{task_type}_optimized.yml')
    if task_config_path.exists():
        try:
            with open(task_config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            
            # 提取任务配置（保留原有逻辑）
            if 'task_configs' in config and task_type in config['task_configs']:
                task_config = config['task_configs'][task_type]
                
                flat_config = {
                    'llm_confidence_threshold': task_config.get('llm_matcher', {}).get('confidence_threshold', 0.10 if task_type == 'join' else 0.05),
                    'aggregator_max_results': task_config.get('aggregator', {}).get('max_results', 20 if task_type == 'join' else 30),
                    'llm_concurrency': 3,
                    'metadata_threshold': task_config.get('metadata_filter', {}).get('column_similarity_threshold', 0.35 if task_type == 'join' else 0.20),
                    'vector_threshold': task_config.get('vector_search', {}).get('similarity_threshold', 0.40 if task_type == 'join' else 0.35),
                    'min_column_overlap': task_config.get('metadata_filter', {}).get('min_column_overlap', 2 if task_type == 'join' else 1),
                    'vector_top_k': task_config.get('vector_search', {}).get('top_k', 50 if task_type == 'join' else 60),
                    'enable_llm': task_config.get('llm_matcher', {}).get('enable_llm', True),
                    'layer_combination': 'intersection',
                    'use_column_types': task_config.get('metadata_filter', {}).get('use_column_types', task_type == 'join'),
                    'use_value_overlap': task_config.get('metadata_filter', {}).get('use_value_overlap', task_type == 'join'),
                    'focus_on_join_keys': task_config.get('llm_matcher', {}).get('focus_on_join_keys', task_type == 'join'),
                    'allow_subset_matching': task_config.get('metadata_filter', {}).get('allow_subset_matching', task_type == 'union'),
                    'allow_type_coercion': task_config.get('metadata_filter', {}).get('allow_type_coercion', task_type == 'union'),
                    'allow_self_matches': task_config.get('llm_matcher', {}).get('allow_self_matches', task_type == 'union'),
                    'focus_on_compatibility': task_config.get('llm_matcher', {}).get('focus_on_compatibility', task_type == 'union'),
                    'check_semantic_similarity': task_config.get('llm_matcher', {}).get('check_semantic_similarity', task_type == 'union'),
                    'allow_partial_matches': task_config.get('llm_matcher', {}).get('allow_partial_matches', task_type == 'union'),
                    'cache_enabled': config.get('optimization_config', {}).get('cache_enabled', True),
                    'parallel_processing': config.get('optimization_config', {}).get('parallel_processing', True),
                    'max_workers': config.get('optimization_config', {}).get('max_workers', 4),
                    'metadata_weight': task_config.get('aggregator', {}).get('ranking_weights', {}).get('metadata_score', 0.25 if task_type == 'join' else 0.20),
                    'vector_weight': task_config.get('aggregator', {}).get('ranking_weights', {}).get('vector_score', 0.35 if task_type == 'join' else 0.40),
                    'llm_weight': task_config.get('aggregator', {}).get('ranking_weights', {}).get('llm_score', 0.40)
                }
                
                logger.info(f"✅ 加载{task_type.upper()}任务特定配置从config_{task_type}_optimized.yml")
                return flat_config
                
        except Exception as e:
            logger.warning(f"加载{task_type}特定配置失败: {e}，使用默认配置")
    
    # 使用默认配置
    logger.warning(f"所有配置文件不存在，使用默认{task_type}配置")
    if task_type == 'join':
        return {
            'llm_confidence_threshold': 0.10,
            'aggregator_max_results': 500,
            'llm_concurrency': 3,
            'metadata_threshold': 0.40,
            'vector_threshold': 0.45,
            'min_column_overlap': 3,
            'vector_top_k': 100,
            'layer_combination': 'intersection'
        }
    else:  # union
        return {
            'llm_confidence_threshold': 0.30,
            'aggregator_max_results': 200,
            'llm_concurrency': 3,
            'metadata_threshold': 0.25,
            'vector_threshold': 0.25,
            'min_column_overlap': 1,
            'vector_top_k': 150,
            'include_query_variants': True,
            'allow_prefix_match': True,
            'layer_combination': 'intersection'
        }


# ================== 缓存管理器 ==================
class CacheManager:
    """统一的缓存管理器，提供内存和磁盘双层缓存"""
    
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
        key_data = {
            'op': operation,
            'query': query.get('query_table', '') if isinstance(query, dict) else str(query),
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

def init_cache_manager(dataset_name: str = '', task_type: str = '', dataset_type: str = ''):
    """初始化全局缓存管理器"""
    global cache_manager
    if cache_manager is None:
        cache_dir = f"cache/experiment_cache/{dataset_name}_{task_type}_{dataset_type}".strip('_')
        cache_manager = CacheManager(cache_dir)
    return cache_manager


def load_dataset(task_type: str, dataset_type: str = 'subset', dataset_name: str = None) -> tuple:
    """加载数据集（支持多种数据集格式）
    
    Args:
        task_type: 'join' 或 'union'
        dataset_type: 'subset', 'true_subset', 'complete', 'full' 等
        dataset_name: 'nlctables', 'opendata', 'webtable' 或 None
    """
    # 如果dataset_name未指定，尝试从dataset_type解析
    if dataset_name is None:
        if 'nlctables' in dataset_type.lower():
            dataset_name = 'nlctables'
        elif 'opendata' in dataset_type.lower():
            dataset_name = 'opendata'
        elif 'webtable' in dataset_type.lower():
            dataset_name = 'webtable'
    
    # 特殊处理NLCTables adapter（如果存在）
    if dataset_name == 'nlctables':
        try:
            from nlctables_adapter import NLCTablesAdapter
            adapter = NLCTablesAdapter()
            
            # 解析subset类型
            if 'complete' in dataset_type:
                subset_type = 'complete'
            else:
                subset_type = 'subset'
            
            # 使用适配器加载数据
            tables, queries, ground_truth = adapter.load_nlctables_dataset(task_type, subset_type)
            
            logger.info(f"📊 Loaded NLCTables dataset via adapter: {len(tables)} tables, {len(queries)} queries")
            return tables, queries, ground_truth
        except ImportError:
            # 如果适配器不存在，使用标准路径
            pass
    
    # 检查是否是自定义路径
    if '/' in dataset_type or dataset_type.startswith('examples'):
        # 直接使用提供的路径
        base_dir = Path(dataset_type)
    else:
        # 构建标准路径格式
        # 新格式：examples/{dataset_name}/{task}_{dataset_type}/
        if dataset_name:
            # 规范化dataset_type
            if dataset_type in ['complete', 'full']:
                subset_str = 'complete'
            elif dataset_type in ['subset', 'true_subset']:
                subset_str = 'subset'
            else:
                subset_str = dataset_type
            
            base_dir = Path(f'examples/{dataset_name}/{task_type}_{subset_str}')
            
            # 如果新格式不存在，尝试旧格式
            if not base_dir.exists():
                logger.info(f"新格式路径不存在: {base_dir}, 尝试旧格式...")
                # 旧格式：examples/separated_datasets/{task}_{subset}/
                if dataset_type == 'complete' or dataset_type == 'full':
                    base_dir = Path(f'examples/separated_datasets/{task_type}')
                elif dataset_type == 'true_subset':
                    base_dir = Path(f'examples/separated_datasets/{task_type}_true_subset')
                elif dataset_type == 'subset':
                    base_dir = Path(f'examples/separated_datasets/{task_type}_subset')
                else:
                    base_dir = Path(f'examples/separated_datasets/{task_type}_{dataset_type}')
        else:
            # 兼容旧格式
            if dataset_type == 'complete' or dataset_type == 'full':
                base_dir = Path(f'examples/separated_datasets/{task_type}')
            elif dataset_type == 'true_subset':
                base_dir = Path(f'examples/separated_datasets/{task_type}_true_subset')
            elif dataset_type == 'subset':
                base_dir = Path(f'examples/separated_datasets/{task_type}_subset')
            else:
                base_dir = Path(f'examples/separated_datasets/{task_type}_{dataset_type}')
    
    # 检查路径是否存在
    if not base_dir.exists():
        raise FileNotFoundError(f"数据集路径不存在: {base_dir}")
    
    # 加载文件
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
    
    logger.info(f"📊 Loaded dataset from {base_dir}: {len(tables)} tables, {len(queries)} queries")
    return tables, queries, ground_truth


def convert_ground_truth_format(ground_truth_list: List[Dict], task_type: str = None) -> Dict[str, List[str]]:
    """将ground truth转换为字典格式
    
    Args:
        ground_truth_list: ground truth列表
        task_type: 任务类型 ('join' 或 'union')，如果为None则根据内容推断
    """
    query_to_candidates = {}
    
    # 如果没有指定任务类型，尝试推断
    if task_type is None:
        # 检查是否有自匹配的情况来推断是否为UNION任务
        has_self_match = any(
            item.get('query_table', '') == item.get('candidate_table', '')
            for item in ground_truth_list
            if item.get('query_table') and item.get('candidate_table')
        )
        # 如果有自匹配，很可能是UNION任务
        task_type = 'union' if has_self_match else 'join'
    
    for item in ground_truth_list:
        query_table = item.get('query_table', '')
        
        # 处理两种格式：
        # 格式1: {'query_table': 'xxx', 'candidate_table': 'yyy'} (原格式)
        # 格式2: {'query_table': 'xxx', 'ground_truth': ['yyy', 'zzz']} (NLCTables格式)
        
        if 'ground_truth' in item and isinstance(item['ground_truth'], list):
            # NLCTables格式：直接使用ground_truth列表
            if query_table:
                candidates = item['ground_truth']
                # 对于JOIN任务，过滤自匹配；对于UNION任务，保留自匹配
                if task_type == 'union':
                    query_to_candidates[query_table] = candidates
                else:  # join
                    query_to_candidates[query_table] = [c for c in candidates if c != query_table]
        
        elif 'candidate_table' in item:
            # 原始格式：逐个添加候选表
            candidate_table = item.get('candidate_table', '')
            if query_table and candidate_table:
                # 对于JOIN任务，过滤自匹配；对于UNION任务，保留自匹配
                if task_type == 'union' or query_table != candidate_table:
                    if query_table not in query_to_candidates:
                        query_to_candidates[query_table] = []
                    query_to_candidates[query_table].append(candidate_table)
    
    return query_to_candidates


def initialize_shared_resources_l1(tables: List[Dict], dataset_type: str, task_type: str = None) -> Dict:
    """初始化L1层共享资源（支持任务特定配置）"""
    logger.info("🚀 初始化L1层共享资源...")
    
    from src.tools.smd_enhanced_metadata_filter import SMDEnhancedMetadataFilter
    
    # 对于OpenData，确保表有name字段（兼容性）
    if dataset_type == 'opendata':
        for t in tables:
            if 'name' not in t and 'table_name' in t:
                t['name'] = t['table_name']
    
    # 如果提供了任务类型，加载任务特定配置
    task_config = {}
    if task_type:
        task_config = load_task_config(task_type)
        logger.info(f"  📋 L1使用{task_type.upper()}配置: 阈值={task_config.get('metadata_threshold', 0.30):.2f}")
    
    # 初始化元数据过滤器并预构建索引
    metadata_filter = SMDEnhancedMetadataFilter()
    
    # 预构建SMD索引（只构建一次，所有查询共享）
    logger.info(f"📊 预构建SMD索引（{len(tables)}个表）...")
    metadata_filter.build_index(tables)
    
    # 序列化索引以便在进程间共享
    smd_index_serialized = pickle.dumps(metadata_filter)
    logger.info(f"✅ SMD索引构建完成，大小: {len(smd_index_serialized) / 1024:.1f}KB")
    
    config = {
        'layer': 'L1',
        'table_count': len(tables),
        'dataset_type': dataset_type,
        'filter_initialized': True,
        'smd_index': smd_index_serialized,  # 添加序列化的索引
        'task_config': task_config  # 添加任务配置
    }
    
    logger.info("✅ L1层资源初始化完成")
    return config


def initialize_shared_resources_l2(tables: List[Dict], dataset_type: str, task_type: str = None) -> Dict:
    """初始化L1+L2层共享资源（支持任务特定配置）"""
    logger.info("🚀 初始化L1+L2层共享资源...")
    
    # 如果提供了任务类型，加载任务特定配置
    task_config = {}
    if task_type:
        task_config = load_task_config(task_type, dataset_type)  # 传递dataset_type
        logger.info(f"  📋 L2使用{task_type.upper()}配置: 向量阈值={task_config.get('vector_threshold', 0.30):.2f}, top_k={task_config.get('vector_top_k', 100)}")
        logger.info(f"  📋 层组合策略: {task_config.get('layer_combination', 'intersection')}")
    
    # 对于OpenData，确保表有name字段（兼容性）
    if dataset_type == 'opendata':
        for t in tables:
            if 'name' not in t and 'table_name' in t:
                t['name'] = t['table_name']
    
    # 预计算向量索引
    cache_dir = Path("cache") / dataset_type
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    index_file = cache_dir / f"vector_index_{len(tables)}.pkl"
    embeddings_file = cache_dir / f"table_embeddings_{len(tables)}.pkl"
    
    if not (index_file.exists() and embeddings_file.exists()):
        logger.info("⚙️ 生成新的向量索引...")
        from precompute_embeddings import precompute_all_embeddings
        precompute_all_embeddings(tables, dataset_type)
    
    config = {
        'layer': 'L1+L2',
        'table_count': len(tables),
        'dataset_type': dataset_type,
        'vector_index_path': str(index_file),
        'embeddings_path': str(embeddings_file),
        'filter_initialized': True,
        'vector_initialized': True,
        'task_config': task_config,  # 添加任务配置
        'optimization_config': task_config  # 也作为optimization_config传递
    }
    
    logger.info("✅ L1+L2层资源初始化完成")
    return config


def initialize_shared_resources_l3(tables: List[Dict], task_type: str, dataset_type: str) -> Dict:
    """初始化完整三层共享资源（包含任务特定的优化配置）"""
    logger.info("🚀 初始化L1+L2+L3层共享资源...")
    
    # 对于OpenData，确保表有name字段（兼容性）
    if dataset_type == 'opendata':
        for t in tables:
            if 'name' not in t and 'table_name' in t:
                t['name'] = t['table_name']
    
    # 初始化L1+L2资源（传递task_type）
    l2_config = initialize_shared_resources_l2(tables, dataset_type, task_type)
    
    # 从配置文件加载任务特定的配置（传递dataset_type）
    task_config = load_task_config(task_type, dataset_type)
    
    # 创建优化配置字典
    optimization_config = {
        'llm_confidence_threshold': task_config['llm_confidence_threshold'],
        'aggregator_max_results': task_config['aggregator_max_results'],
        'llm_concurrency': task_config.get('llm_concurrency', 3),
        'metadata_threshold': task_config.get('metadata_threshold', 0.30),
        'vector_threshold': task_config.get('vector_threshold', 0.30),
        'min_column_overlap': task_config.get('min_column_overlap', 2),
        'vector_top_k': task_config.get('vector_top_k', 100),
        'enable_llm': task_config.get('enable_llm', True),
        # 层组合策略（关键！）
        'layer_combination': task_config.get('layer_combination', 'intersection'),
        # UNION特定配置
        'include_query_variants': task_config.get('include_query_variants', False),
        'allow_prefix_match': task_config.get('allow_prefix_match', False),
        'semantic_matching': task_config.get('semantic_matching', False),
        'value_similarity_weight': task_config.get('value_similarity_weight', 0.5)
    }
    
    # 初始化工作流（获取优化配置）
    from src.core.langgraph_workflow import DataLakeDiscoveryWorkflow
    workflow = DataLakeDiscoveryWorkflow()
    
    # 如果需要原始OptimizerAgent的其他功能，仍然调用它
    # 但使用任务特定的配置覆盖其默认设置
    from src.agents.optimizer_agent import OptimizerAgent
    from types import SimpleNamespace
    optimizer = OptimizerAgent()
    
    # 创建正确的state格式，OptimizerAgent期望query_task对象
    query_task = SimpleNamespace(task_type=task_type)
    state = {
        'query_task': query_task,
        'all_tables': tables
    }
    
    # 获取原始配置并与任务特定配置合并
    result = optimizer.process(state)
    original_config = result.get('optimization_config', {})
    
    # 如果original_config不是字典，转换为字典
    if hasattr(original_config, '__dict__'):
        original_config = original_config.__dict__
    elif not isinstance(original_config, dict):
        original_config = {}
    
    # 合并配置，任务特定配置优先
    merged_config = {**original_config, **optimization_config}
    
    # 获取批处理执行策略（只调用一次PlannerAgent）
    from src.agents.planner_agent import PlannerAgent
    planner = PlannerAgent()
    execution_strategy = planner.process({
        'task_type': task_type,
        'table_structure': 'unknown',
        'data_size': 'medium',
        'performance_requirement': 'balanced'
    })
    
    config = {
        **l2_config,
        'layer': 'L1+L2+L3',
        'optimization_config': merged_config,
        'execution_strategy': execution_strategy,
        'task_type': task_type,
        'workflow_initialized': True
    }
    
    config_source = f"config_{dataset_type}_{task_type}.yml" if Path(f'config_{dataset_type}_{task_type}.yml').exists() else \
                    f"config_{task_type}_optimized.yml" if Path(f'config_{task_type}_optimized.yml').exists() else \
                    "config_optimized.yml"
    
    logger.info(f"✅ L1+L2+L3层资源初始化完成 - {task_type.upper()}任务优化")
    logger.info(f"  - 配置阈值: {optimization_config['llm_confidence_threshold']:.3f}")
    logger.info(f"  - 层组合策略: {optimization_config['layer_combination']}")
    logger.info(f"  - 最大候选: {optimization_config['aggregator_max_results']}")
    logger.info(f"  - 配置来源: {config_source}")
    
    return config


def process_query_l1(args: Tuple) -> Dict:
    """处理单个查询 - L1层"""
    query, tables, shared_config, cache_file_path = args
    query_table_name = query.get('query_table', '')
    
    # 初始化或获取缓存管理器（子进程需要）
    global cache_manager
    if cache_manager is None and cache_file_path:
        # cache_file_path现在是缓存目录路径
        cache_dir = cache_file_path
        cache_manager = CacheManager(cache_dir)
    
    # 使用缓存管理器
    if cache_manager:
        cached = cache_manager.get('l1', query)
        if cached is not None:
            return cached
    
    # 使用预构建的SMD索引（通过pickle序列化）
    if 'smd_index' in shared_config:
        # 反序列化SMD索引
        import io
        from src.tools.smd_enhanced_metadata_filter import SMDEnhancedMetadataFilter
        metadata_filter = pickle.loads(shared_config['smd_index'])
    else:
        # 降级：构建新索引（不应该发生）
        from src.tools.smd_enhanced_metadata_filter import SMDEnhancedMetadataFilter
        metadata_filter = SMDEnhancedMetadataFilter()
        metadata_filter.build_index(tables)
    
    # 查找查询表 - 兼容不同数据集的字段名
    query_table = None
    for t in tables:
        # 兼容 'name' (NLCTables) 和 'table_name' (OpenData/WebTable)
        table_name = t.get('name') or t.get('table_name')
        if table_name == query_table_name:
            query_table = t
            break
    
    if not query_table:
        logger.warning(f"Query table {query_table_name} not found in tables")
        result = {'query_table': query_table_name, 'predictions': []}
    else:
        # L1: 元数据过滤 - 使用预构建的索引
        # NLCTables需要更低的阈值和更多候选以提高召回率
        # 让阈值自然控制候选数量，不设置max_candidates限制
        logger.debug(f"Filtering candidates for {query_table_name}")
        candidates = metadata_filter.filter_candidates(
            query_table,
            None,  # all_tables - None表示使用预构建的索引
            threshold=0.05,  # NLCTables需要更低的阈值以提高召回率
            max_candidates=1000  # 设置一个很高的上限，实际由阈值控制
        )
        
        logger.debug(f"L1 found {len(candidates)} candidates")
        # 候选格式是[(table_name, score), ...]，提取表名
        predictions = [
            table_name for table_name, score in candidates 
            if table_name != query_table_name
        ][:MAX_PREDICTIONS]
        
        logger.debug(f"L1 final predictions: {len(predictions)}")
        result = {'query_table': query_table_name, 'predictions': predictions}
    
    # 保存到全局缓存
    if cache_manager:
        cache_manager.set('l1', query, result)
    
    return result


def process_query_l2(args: Tuple) -> Dict:
    """处理单个查询 - L1+L2层（优化版：支持UNION/INTERSECTION组合策略）"""
    query, tables, shared_config, cache_file_path = args
    query_table_name = query.get('query_table', '')
    task_type = query.get('task_type', 'join')  # 获取任务类型
    dataset_type = query.get('dataset_type', '')  # 获取数据集类型
    
    # 初始化或获取缓存管理器（子进程需要）
    global cache_manager
    if cache_manager is None and cache_file_path:
        # cache_file_path现在是缓存目录路径
        cache_dir = cache_file_path
        cache_manager = CacheManager(cache_dir)
    
    # 使用缓存管理器
    if cache_manager:
        cached = cache_manager.get('l1_l2', query, {'task_type': task_type})
        if cached is not None:
            return cached
    
    # 获取配置（包括层组合策略）
    task_config = shared_config.get('optimization_config', {})
    layer_combination = task_config.get('layer_combination', 'intersection')
    
    # 运行L1+L2层
    from src.tools.smd_enhanced_metadata_filter import SMDEnhancedMetadataFilter
    from src.tools.vector_search_tool import VectorSearchTool
    from src.tools.value_similarity_tool import ValueSimilarityTool
    
    metadata_filter = SMDEnhancedMetadataFilter()
    vector_search = VectorSearchTool()
    value_similarity = ValueSimilarityTool()
    
    # 查找查询表 - 兼容不同数据集的字段名
    query_table = None
    for t in tables:
        # 兼容 'name' (NLCTables) 和 'table_name' (OpenData/WebTable)
        table_name = t.get('name') or t.get('table_name')
        if table_name == query_table_name:
            query_table = t
            break
    
    if not query_table:
        result = {'query_table': query_table_name, 'predictions': []}
    else:
        # L1: 元数据过滤（扩大候选集）
        # 对于OpenData，确保表有name字段再构建索引
        if any('table_name' in t and 'name' not in t for t in tables):
            for t in tables:
                if 'name' not in t and 'table_name' in t:
                    t['name'] = t['table_name']
        metadata_filter.build_index(tables)
        
        # 使用配置中的阈值（WebTable会更低）
        metadata_threshold = task_config.get('metadata_threshold', 0.05)
        l1_candidates = metadata_filter.filter_candidates(
            query_table,
            None,  # all_tables - None表示使用预构建的索引
            threshold=metadata_threshold,
            max_candidates=1000  # 设置高上限，让阈值自然控制
        )
        
        # L2: 向量搜索
        try:
            # 根据层组合策略决定搜索范围
            if layer_combination == 'union':
                # UNION策略：在所有表中搜索，不限于L1候选
                logger.debug(f"使用UNION策略：L2搜索所有{len(tables)}个表")
                search_tables = tables
            else:
                # INTERSECTION策略：只在L1候选中搜索（原有逻辑）
                candidate_names = [name for name, score in l1_candidates]
                candidate_tables = []
                for t in tables:
                    table_name = t.get('name') or t.get('table_name')
                    if table_name in candidate_names:
                        candidate_tables.append(t)
                search_tables = candidate_tables if candidate_tables else tables
                logger.debug(f"使用INTERSECTION策略：L2搜索{len(search_tables)}个L1候选表")
            
            # 使用向量搜索
            vector_threshold = task_config.get('vector_threshold', 0.1)
            vector_top_k = task_config.get('vector_top_k', 200)
            l2_results = vector_search.search(
                query_table, 
                search_tables,
                top_k=vector_top_k
            )
            
            # 添加任务特定的值相似性重排序（L2增强）
            enhanced_results = []
            for table_name, vec_score in l2_results:
                # 过滤低相似度的候选
                if vec_score < vector_threshold:
                    continue
                if table_name != query_table_name:
                    # 找到候选表对象
                    cand_table = None
                    for t in tables:
                        t_name = t.get('name') or t.get('table_name')
                        if t_name == table_name:
                            cand_table = t
                            break
                    
                    if cand_table:
                        # 从配置中获取权重
                        value_sim_weight = task_config.get('value_similarity_weight', 0.5)
                        
                        # 任务特定的值相似性计算
                        if task_type == 'union':
                            # UNION任务：更关注结构兼容性和数据分布相似性
                            val_sim = value_similarity._calculate_union_value_similarity(
                                query_table, cand_table
                            )
                            # 使用配置的权重（UNION默认更重视值相似性）
                            if task_config.get('semantic_matching', False):
                                combined_score = (1 - value_sim_weight) * vec_score + value_sim_weight * val_sim
                            else:
                                combined_score = 0.4 * vec_score + 0.6 * val_sim
                            
                            # UNION额外检查：列数差异不能太大
                            query_col_count = len(query_table.get('columns', []))
                            cand_col_count = len(cand_table.get('columns', []))
                            col_diff_ratio = abs(query_col_count - cand_col_count) / max(query_col_count, 1)
                            if col_diff_ratio > 0.5:  # 列数差异超过50%，降低分数
                                combined_score *= 0.7
                        else:
                            # JOIN任务：更关注值重叠和关联性
                            val_sim = value_similarity._calculate_join_value_similarity(
                                query_table, cand_table
                            )
                            # 使用配置的权重（JOIN默认更重视向量相似性）
                            combined_score = 0.7 * vec_score + 0.3 * val_sim
                        
                        enhanced_results.append((table_name, combined_score))
            
            # 合并L1和L2结果
            final_scores = {}
            
            if layer_combination == 'union':
                # UNION策略：合并所有结果，使用最高分数
                # 先添加L2的增强结果
                for name, score in enhanced_results:
                    final_scores[name] = score
                
                # 添加L1独有的结果（L2没找到的）
                for name, l1_score in l1_candidates:
                    if name != query_table_name and name not in final_scores:
                        # L1独有的结果使用原始分数的0.6倍
                        final_scores[name] = l1_score * 0.6
            else:
                # INTERSECTION策略：只保留两者都有的（但分数取最高）
                l1_names = {name for name, _ in l1_candidates}
                for name, score in enhanced_results:
                    if name in l1_names:
                        final_scores[name] = score
                
                # 如果intersection结果太少，补充一些高分的L1结果
                if len(final_scores) < 10:
                    for name, l1_score in l1_candidates[:20]:
                        if name != query_table_name and name not in final_scores:
                            final_scores[name] = l1_score * 0.5
            
            # 排序并提取预测
            sorted_results = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)
            predictions = [name for name, score in sorted_results][:MAX_PREDICTIONS]
            
            logger.debug(f"L1+L2组合结果：策略={layer_combination}, L1={len(l1_candidates)}个, L2={len(enhanced_results)}个, 最终={len(predictions)}个")
            
        except Exception as e:
            logger.warning(f"L2处理失败 {query_table_name}: {e}, 回退到L1结果")
            # 回退到L1结果
            predictions = [
                table_name for table_name, score in l1_candidates
                if table_name != query_table_name
            ][:MAX_PREDICTIONS]
        
        result = {'query_table': query_table_name, 'predictions': predictions}
    
    # 保存到全局缓存
    if cache_manager:
        cache_manager.set('l1_l2', query, result, {'task_type': task_type})
    
    return result


def process_query_l3(args: Tuple) -> Dict:
    """处理单个查询 - 完整三层（优化版：任务特定优化和boost factors）"""
    query, tables, shared_config, cache_file_path = args
    query_table_name = query.get('query_table', '')
    task_type = query.get('task_type', shared_config.get('task_type', 'join'))
    
    # 获取动态优化器实例（如果存在）
    dynamic_optimizer = shared_config.get('dynamic_optimizer', None)
    
    # 初始化或获取缓存管理器（子进程需要）
    global cache_manager
    if cache_manager is None and cache_file_path:
        # cache_file_path现在是缓存目录路径
        cache_dir = cache_file_path
        cache_manager = CacheManager(cache_dir)
    
    # 使用缓存管理器
    if cache_manager:
        cached = cache_manager.get('l1_l2_l3', query, {'task_type': task_type})
        if cached is not None:
            return cached
    
    # 先运行L2层获取基础结果
    l2_cache_file = cache_file_path.replace('L3', 'L2')
    l2_result = process_query_l2((query, tables, shared_config, l2_cache_file))
    l2_predictions = l2_result.get('predictions', [])
    
    logger.info(f"L3层接收到L2预测: {len(l2_predictions)} 个候选")
    
    # 如果L2预测太少，直接返回L2结果
    if len(l2_predictions) < 2:
        logger.warning(f"L2预测太少（{len(l2_predictions)}个），跳过L3层LLM验证")
        return {'query_table': query_table_name, 'predictions': l2_predictions}
    
    # L3层：直接使用LLM验证（确保UNION任务正确处理）
    try:
        # 方案1：直接使用LLMMatcherTool进行验证
        from src.tools.llm_matcher import LLMMatcherTool
        import asyncio
        
        # 查找查询表 - 兼容不同数据集的字段名
        query_table = None
        for t in tables:
            # 兼容 'name' (NLCTables) 和 'table_name' (OpenData/WebTable)
            table_name = t.get('name') or t.get('table_name')
            if table_name == query_table_name:
                query_table = t
                break
        
        if not query_table:
            logger.warning(f"查询表 {query_table_name} 未找到，使用L2结果")
            final_predictions = l2_predictions
        else:
            # 从配置中获取L3层参数
            optimizer_config = shared_config.get('optimization_config', {})
            
            # 使用配置文件中的任务特定参数
            max_candidates = optimizer_config.get('aggregator_max_results', 300)
            llm_concurrency = optimizer_config.get('llm_concurrency', 3)
            confidence_threshold = optimizer_config.get('llm_confidence_threshold', 0.20)
            
            logger.info(f"L3层使用{task_type.upper()}任务配置: max_candidates={max_candidates}, "
                       f"concurrency={llm_concurrency}, confidence={confidence_threshold:.2f}")
            
            # 初始化LLM matcher
            llm_matcher = LLMMatcherTool()
            
            # L3改进：限制LLM验证数量，避免被低质量候选淹没
            # 只验证TOP候选，确保LLM看到的都是高质量候选
            # 从配置中获取最大验证数量
            max_verify_config = optimizer_config.get('max_llm_verify', 25)
            max_verify = min(len(l2_predictions), max_verify_config)
            logger.info(f"L3层准备验证 {max_verify} 个L2候选（配置最大值: {max_verify_config}）")
            
            candidate_tables = []
            for pred_name in l2_predictions[:max_verify]:
                for t in tables:
                    # 兼容不同字段名
                    t_name = t.get('name') or t.get('table_name')
                    if t_name == pred_name:
                        candidate_tables.append(t)
                        break
            
            logger.info(f"L3层找到 {len(candidate_tables)} 个候选表进行LLM验证")
            
            if candidate_tables and len(candidate_tables) > 0:
                # 使用batch_verify进行并行LLM验证
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                
                # 关键：确保task_type正确传递，使用OptimizerAgent的参数
                logger.info(f"L3层开始LLM批量验证: {len(candidate_tables)} 个表, 并发数={llm_concurrency}")
                
                llm_results = loop.run_until_complete(
                    llm_matcher.batch_verify(
                        query_table=query_table,
                        candidate_tables=candidate_tables,
                        task_type=task_type,  # 明确传递task_type
                        max_concurrent=llm_concurrency,  # 使用OptimizerAgent优化的并发数
                        existing_scores=[0.7] * len(candidate_tables)  # 假设L2给出的都是高分候选
                    )
                )
                loop.close()
                
                logger.info(f"L3层LLM验证完成: 返回 {len(llm_results)} 个结果")
                
                # L3改进：使用重排序而非过滤
                # 收集所有候选表的相关性分数
                l3_scored = []
                for i, result in enumerate(llm_results):
                    # 优先使用relevance_score（新的重排序字段），如果没有则使用confidence
                    relevance_score = result.get('relevance_score', result.get('confidence', 0))
                    # 兼容不同字段名
                    candidate_name = candidate_tables[i].get('name') or candidate_tables[i].get('table_name')
                    
                    # 应用任务特定的boost factor（如果有优化器）
                    if dynamic_optimizer:
                        boosted_score = dynamic_optimizer.apply_boost_factor(
                            task_type, relevance_score, query_table_name, candidate_name
                        )
                    else:
                        boosted_score = relevance_score
                    
                    # 重排序：收集所有候选，不过滤
                    l3_scored.append((candidate_name, boosted_score))
                
                # 按相关性分数降序排序（重排序的核心）
                l3_scored.sort(key=lambda x: x[1], reverse=True)
                
                # 合并L3验证的结果和剩余的L2结果
                l3_verified_names = {name for name, _ in l3_scored}
                remaining_l2 = [name for name in l2_predictions if name not in l3_verified_names]
                
                # 最终预测：L3重排序的结果 + 剩余的L2结果（确保使用所有L2预测）
                l3_predictions = [name for name, score in l3_scored]
                
                # 确保包含所有L2的预测，不只限于MAX_PREDICTIONS
                for name in remaining_l2:
                    if len(l3_predictions) < MAX_PREDICTIONS:
                        l3_predictions.append(name)
                
                logger.info(f"L3层重排序: 对 {len(l3_scored)} 个候选进行了LLM评分重排序")
                logger.info(f"L3层最终预测数: {len(l3_predictions)} (L3验证: {len(l3_scored)}, L2补充: {len(l3_predictions) - len(l3_scored)})")
                
                if l3_scored:
                    top_scores = [(name, score) for name, score in l3_scored[:5]]
                    logger.info(f"L3层Top5得分: {top_scores}")
                
                final_predictions = l3_predictions[:MAX_PREDICTIONS]  # 确保最终输出不超过MAX_PREDICTIONS
            else:
                logger.warning(f"没有找到L2候选表的详细信息，使用L2结果")
                final_predictions = l2_predictions
                
    except ImportError as e:
        logger.error(f"无法导入LLMMatcherTool: {e}")
        # 方案2：如果直接LLM调用失败，尝试使用workflow
        try:
            from src.core.langgraph_workflow import DataLakeDiscoveryWorkflow
            
            workflow = DataLakeDiscoveryWorkflow()
            
            # 确保不跳过LLM
            import os
            old_skip_llm = os.environ.get('SKIP_LLM', 'false')
            os.environ['SKIP_LLM'] = 'false'
            
            result = workflow.run(
                query=f"find {task_type}able tables for {query_table_name}",
                tables=tables,
                task_type=task_type,
                query_table_name=query_table_name
            )
            
            # 恢复原设置
            os.environ['SKIP_LLM'] = old_skip_llm
            
            if result and result.get('success') and result.get('results'):
                l3_predictions = [
                    r['table_name'] for r in result.get('results', [])[:MAX_PREDICTIONS]
                    if r['table_name'] != query_table_name
                ]
                final_predictions = l3_predictions if l3_predictions else l2_predictions
            else:
                logger.warning(f"L3 工作流返回空结果 {query_table_name}, 使用L2结果")
                final_predictions = l2_predictions
                
        except Exception as e2:
            logger.warning(f"L3 工作流也失败 {query_table_name}: {e2}, 回退到L2结果")
            final_predictions = l2_predictions
            
    except Exception as e:
        logger.warning(f"L3 LLM处理失败 {query_table_name}: {e}, 回退到L2结果")
        final_predictions = l2_predictions
    
    query_result = {'query_table': query_table_name, 'predictions': final_predictions}
    
    # 保存到全局缓存
    if cache_manager:
        cache_manager.set('l1_l2_l3', query, query_result, {'task_type': task_type})
    
    return query_result


def run_layer_experiment(layer: str, tables: List[Dict], queries: List[Dict], 
                         task_type: str, dataset_type: str, max_workers: int = 4) -> Tuple[Dict, float]:
    """运行特定层的实验（优化版）"""
    logger.info(f"\n{'='*60}")
    logger.info(f"🔬 Running {layer} Experiment")
    logger.info(f"{'='*60}")
    
    # 初始化共享资源（传递task_type以使用任务特定配置）
    if layer == 'L1':
        shared_config = initialize_shared_resources_l1(tables, dataset_type, task_type)
        process_func = process_query_l1
    elif layer == 'L1+L2':
        shared_config = initialize_shared_resources_l2(tables, dataset_type, task_type)
        process_func = process_query_l2
    else:  # L1+L2+L3
        # 确保LLM不被跳过，特别重要！
        os.environ['SKIP_LLM'] = 'false'
        os.environ['FORCE_LLM_VERIFICATION'] = 'true'
        # 从配置文件加载任务特定配置（传递dataset_type）
        task_config = load_task_config(task_type, dataset_type)
        if task_config.get('semantic_matching', False):
            os.environ['UNION_OPTIMIZATION'] = 'true'
        shared_config = initialize_shared_resources_l3(tables, task_type, dataset_type)
        process_func = process_query_l3
    
    # 使用缓存管理器的目录（如果存在）
    if cache_manager:
        cache_dir = cache_manager.cache_dir
    else:
        # 降级到默认缓存目录
        cache_dir = Path(f"cache/ablation_{dataset_type}_{layer.replace('+', '_')}")
        cache_dir.mkdir(parents=True, exist_ok=True)
    
    # 准备进程池参数（每个查询传递缓存目录路径和dataset_type）
    query_args = []
    for query in queries:
        # 在query中添加dataset_type以供process函数使用
        query_with_dataset = {**query, 'dataset_type': dataset_type}
        query_args.append((query_with_dataset, tables, shared_config, str(cache_dir)))
    
    # 使用进程池处理
    predictions = {}
    start_time = time.time()
    
    logger.info(f"📊 处理 {len(queries)} 个查询 (进程数={max_workers})...")
    
    if layer == 'L1+L2+L3' and shared_config.get('optimization_config'):
        logger.info(f"  ⚡ 批处理优化: OptimizerAgent和PlannerAgent只调用1次")
    
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
        
        # Precision, Recall, F1 (Use all predictions, not limited)
        if pred_tables:
            predicted_set = set(pred_tables)  # Use all predictions for metrics calculation
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


def create_challenging_queries(tables: List[Dict], queries: List[Dict], ground_truth: List[Dict], max_queries: int = None) -> Tuple[List[Dict], List[Dict]]:
    """创建更具挑战性的查询，降低L1准确率
    
    Args:
        tables: 所有表的列表
        queries: 原始查询列表
        ground_truth: 真实标签
        max_queries: 最大查询数限制
    """
    # 选择具有相似结构但语义不同的表作为挑战性查询
    challenging_queries = []
    challenging_gt = []
    
    # 按列数分组表
    tables_by_col_count = {}
    for table in tables:
        col_count = len(table.get('columns', []))
        if col_count not in tables_by_col_count:
            tables_by_col_count[col_count] = []
        tables_by_col_count[col_count].append(table.get('name'))
    
    # 确定要处理的查询数
    if max_queries is None:
        # 如果max_queries是None（使用all），返回所有原始查询，不创建挑战性查询
        # 这样可以确保使用数据集的所有查询
        logger.info(f"📊 使用所有原始查询（{len(queries)}个），不创建挑战性查询")
        return queries, ground_truth
    else:
        # 如果指定了具体数量，则按原逻辑分配一半原始、一半挑战性
        num_queries = min(len(queries), max_queries)
        num_original = num_queries // 2
        num_challenging = num_queries - num_original
    
    # 为每个原始查询创建一个挑战性版本
    for i, query in enumerate(queries[:num_original]):
        query_table_name = query.get('query_table', '')
        
        # 找到查询表的列数
        query_table = None
        for t in tables:
            if t.get('name') == query_table_name:
                query_table = t
                break
        
        if query_table:
            col_count = len(query_table.get('columns', []))
            
            # 在同列数的表中选择结构相似但语义不同的表作为查询
            similar_tables = tables_by_col_count.get(col_count, [])
            if len(similar_tables) > 1:
                for similar_table_name in similar_tables:
                    if similar_table_name != query_table_name:
                        # 创建挑战性查询
                        challenging_query = {
                            'query_table': similar_table_name,
                            'task_type': query.get('task_type', 'join')
                        }
                        challenging_queries.append(challenging_query)
                        
                        # 查找ground truth（如果存在）
                        gt_matches = []
                        for gt_item in ground_truth:
                            if gt_item.get('query_table') == similar_table_name:
                                gt_matches.append(gt_item)
                        
                        challenging_gt.extend(gt_matches)
                        break
    
    # 混合原始查询和挑战性查询
    mixed_queries = queries[:num_original] + challenging_queries[:num_challenging]
    
    # 对应的ground truth
    original_gt = []
    for query in queries[:num_original]:
        query_table_name = query.get('query_table', '')
        for gt_item in ground_truth:
            if gt_item.get('query_table') == query_table_name:
                original_gt.append(gt_item)
    
    mixed_gt = original_gt + challenging_gt
    
    logger.info(f"📈 Created {len(challenging_queries)} challenging queries")
    logger.info(f"📈 Total mixed queries: {len(mixed_queries)}")
    
    return mixed_queries, mixed_gt


def run_ablation_experiment_optimized(task_type: str, dataset_type: str = 'subset', max_queries: int = None, max_workers: int = 4, use_challenging: bool = True):
    """运行优化的消融实验
    
    Args:
        task_type: 任务类型 ('join' 或 'union')
        dataset_type: 数据集类型 ('subset' 或 'complete')
        max_queries: 最大查询数 (None表示使用所有查询)
        max_workers: 并行进程数
        use_challenging: 是否使用挑战性查询
    """
    logger.info(f"\n{'='*80}")
    logger.info(f"🚀 Running OPTIMIZED Ablation Experiment for {task_type.upper()} Task")
    logger.info(f"📂 Dataset Type: {dataset_type.upper()}")
    queries_desc = "ALL" if max_queries is None else str(max_queries)
    logger.info(f"📊 Max Queries: {queries_desc}")
    if use_challenging:
        logger.info(f"🎯 Using challenging mixed queries to test layer improvements")
    logger.info(f"{'='*80}")
    
    # 初始化缓存管理器
    init_cache_manager(dataset_type, task_type, str(max_queries) if max_queries else 'all')
    
    # 加载数据
    tables, queries, ground_truth = load_dataset(task_type, dataset_type)
    logger.info(f"📊 Dataset: {len(tables)} tables, {len(queries)} queries")
    
    # 创建挑战性查询或使用原始查询
    if use_challenging and max_queries is not None:
        # 只有在指定了具体数量时才创建挑战性查询
        queries, ground_truth = create_challenging_queries(tables, queries, ground_truth, max_queries)
        # 不需要再次截断，create_challenging_queries已经处理了max_queries
    else:
        # 如果max_queries是None，使用所有查询；否则截断
        if max_queries is not None:
            queries = queries[:max_queries]
            logger.info(f"📊 使用前{max_queries}个查询")
        else:
            logger.info(f"📊 使用数据集的所有{len(queries)}个查询")
        # else: 使用所有查询
    
    # 确保每个查询都有正确的任务类型
    for query in queries:
        if 'task_type' not in query:
            query['task_type'] = task_type
    
    logger.info(f"📋 Using {len(queries)} queries for {task_type.upper()} task experiment")
    
    # 转换ground truth格式
    gt_dict = convert_ground_truth_format(ground_truth)
    
    # 存储结果
    results = {}
    
    # 运行三层实验
    for layer in ['L1', 'L1+L2', 'L1+L2+L3']:
        predictions, elapsed_time = run_layer_experiment(
            layer, tables, queries, task_type, dataset_type, max_workers
        )
        
        metrics = calculate_metrics(predictions, gt_dict)
        
        results[layer.replace('+', '_')] = {
            'metrics': metrics,
            'time': elapsed_time,
            'avg_time': elapsed_time / len(queries) if queries else 0
        }
        
        logger.info(f"📈 {layer} - F1: {metrics['f1_score']:.3f}, "
                   f"Hit@1: {metrics['hit@1']:.3f}, "
                   f"Avg Time: {elapsed_time/len(queries):.2f}s/query")
    
    return results


def print_comparison_table(all_results: Dict[str, Dict]):
    """打印对比表格"""
    print("\n" + "="*100)
    print("🚀 OPTIMIZED THREE-LAYER ABLATION EXPERIMENT RESULTS")
    print("="*100)
    
    for task_type, results in all_results.items():
        print(f"\n{task_type.upper()} Task Results:")
        print("-"*80)
        print(f"{'Layer Config':<15} {'Hit@1':<10} {'Hit@3':<10} {'Hit@5':<10} "
              f"{'Precision':<12} {'Recall':<10} {'F1-Score':<10} {'Time(s)':<10}")
        print("-"*80)
        
        for config in ['L1', 'L1_L2', 'L1_L2_L3']:
            if config in results:
                m = results[config]['metrics']
                t = results[config]['avg_time']
                config_display = config.replace('_', '+')
                print(f"{config_display:<15} {m['hit@1']:<10.3f} {m['hit@3']:<10.3f} "
                      f"{m['hit@5']:<10.3f} {m['precision']:<12.3f} "
                      f"{m['recall']:<10.3f} {m['f1_score']:<10.3f} {t:<10.2f}")
    
    print("\n" + "="*100)
    print("📊 LAYER CONTRIBUTION ANALYSIS")
    print("="*100)
    
    for task_type, results in all_results.items():
        print(f"\n{task_type.upper()} Task - Incremental Improvements:")
        
        if 'L1' in results and 'L1_L2' in results:
            f1_l1 = results['L1']['metrics']['f1_score']
            f1_l12 = results['L1_L2']['metrics']['f1_score']
            improvement = (f1_l12 - f1_l1) * 100
            time_increase = results['L1_L2']['avg_time'] - results['L1']['avg_time']
            print(f"  L2 Contribution: {improvement:+.1f}% F1 improvement, {time_increase:+.2f}s time cost")
        
        if 'L1_L2' in results and 'L1_L2_L3' in results:
            f1_l12 = results['L1_L2']['metrics']['f1_score']
            f1_full = results['L1_L2_L3']['metrics']['f1_score']
            improvement = (f1_full - f1_l12) * 100
            time_increase = results['L1_L2_L3']['avg_time'] - results['L1_L2']['avg_time']
            print(f"  L3 Contribution: {improvement:+.1f}% F1 improvement, {time_increase:+.2f}s time cost")
        
        if 'L1' in results and 'L1_L2_L3' in results:
            f1_l1 = results['L1']['metrics']['f1_score']
            f1_full = results['L1_L2_L3']['metrics']['f1_score']
            total_improvement = (f1_full - f1_l1) * 100
            speedup = results['L1']['avg_time'] / results['L1_L2_L3']['avg_time'] if results['L1_L2_L3']['avg_time'] > 0 else 0
            print(f"  Total Improvement: {total_improvement:+.1f}% F1 improvement")
            
            # 计算成本效益比
            if results['L1_L2_L3']['avg_time'] > results['L1']['avg_time']:
                cost_benefit = total_improvement / (results['L1_L2_L3']['avg_time'] / results['L1']['avg_time'])
                print(f"  Cost-Benefit Ratio: {cost_benefit:.2f}% improvement per time unit")


def main():
    """主函数"""
    import argparse
    parser = argparse.ArgumentParser(description='优化版三层架构消融实验')
    parser.add_argument('--task', choices=['join', 'union', 'both'], default='both',
                       help='任务类型 (both会同时运行join和union)')
    parser.add_argument('--dataset', type=str, default='webtable',
                       help='数据集名称: webtable, opendata, 或自定义路径')
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
    args = parser.parse_args()
    
    # 处理max_queries参数
    if args.max_queries.lower() in ['all', '-1', 'none']:
        max_queries = None  # None表示使用所有查询
        print(f"📊 使用整个数据集的所有查询")
    else:
        try:
            max_queries = int(args.max_queries)
            print(f"📊 限制最大查询数为: {max_queries}")
        except ValueError:
            print(f"⚠️ 无效的max-queries值: {args.max_queries}，使用默认值10")
            max_queries = 10
    
    # 决定是否使用挑战性查询
    use_challenging = args.challenging and not args.simple
    
    # 运行实验
    tasks = ['join', 'union'] if args.task == 'both' else [args.task]
    all_results = {}
    
    # 构建数据集路径
    for task in tasks:
        # 处理数据集路径
        if '/' in args.dataset and not args.dataset.startswith('examples/'):
            # 自定义完整路径（非examples开头）
            task_dataset = args.dataset
        elif args.dataset.startswith('examples/') and args.task != 'both':
            # 直接使用提供的路径（单任务模式）
            task_dataset = args.dataset
        elif args.dataset in ['webtable', 'opendata']:
            # 使用标准数据集
            task_dataset = f"examples/{args.dataset}/{task}_{args.dataset_type}"
        elif args.dataset == 'true_subset':
            # WebTable的真子集（向后兼容）
            task_dataset = f"examples/separated_datasets/{task}_true_subset"
        elif 'nlctables' in args.dataset.lower():
            # NLCTables数据集 - 直接传递给load_dataset
            if args.dataset_type == 'complete':
                task_dataset = 'nlctables_complete'
            else:
                task_dataset = 'nlctables_subset'
        else:
            # 其他预定义类型（向后兼容）
            if args.dataset_type == 'complete':
                task_dataset = f"examples/separated_datasets/{task}"
            else:
                task_dataset = f"examples/separated_datasets/{task}_{args.dataset_type}"
            
        results = run_ablation_experiment_optimized(task, task_dataset, max_queries, args.workers, use_challenging)
        all_results[task] = results
    
    # 打印结果表格
    print_comparison_table(all_results)
    
    # 保存结果
    if args.output:
        output_path = Path(args.output)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        dataset_suffix = args.dataset if args.dataset != 'subset' else 'subset'
        output_path = Path(f"experiment_results/ablation_optimized_{dataset_suffix}_{timestamp}.json")
    
    output_path.parent.mkdir(exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    logger.info(f"\n✅ Results saved to: {output_path}")
    
    # 输出缓存统计
    if cache_manager:
        cache_stats = cache_manager.get_stats()
        print("\n" + "="*100)
        print("📊 CACHE STATISTICS")
        print("="*100)
        print(f"  Cache Hits: {cache_stats['hits']}")
        print(f"  Cache Misses: {cache_stats['misses']}")
        print(f"  Cache Saves: {cache_stats['saves']}")
        print(f"  Hit Rate: {cache_stats['hit_rate']}")
        print(f"  Memory Items: {cache_stats['memory_items']}")
    
    # 优化总结
    print("\n" + "="*100)
    print("⚡ OPTIMIZATION SUMMARY")
    print("="*100)
    print("Layer Optimizations Applied:")
    print("  🔍 L1 (Metadata): SMD Enhanced filter with reduced table name weight (5%)")
    print("  ⚡ L2 (Vector): Task-specific value similarity (UNION: 50%+50%, JOIN: 70%+30%)")
    print("  🧠 L3 (LLM): Simplified robust verification with enhanced fallback")
    print("\nSystem Optimizations:")
    print("  1. Batch-level resource sharing (OptimizerAgent & PlannerAgent)")
    print("  2. Process pool parallelization")
    print("  3. Persistent caching system")
    print("  4. Pre-computed vector indices")
    print("  5. Challenging mixed queries for better layer evaluation")
    print("="*100)


if __name__ == "__main__":
    # 修复多进程启动问题
    mp.freeze_support()
    main()