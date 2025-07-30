"""
优化版并行处理管道 - Phase 2性能优化
智能组件路由、快速路径处理、减少协调开销
"""

import asyncio
import logging
import time
from typing import List, Dict, Any, Optional, Tuple, Callable
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
from src.config.settings import settings
from src.tools.performance_profiler import profile_component, get_performance_profiler
from src.core.models import ColumnInfo, TableInfo, MatchResult, TableMatchResult

logger = logging.getLogger(__name__)


@dataclass
class OptimizedPipelineConfig:
    """优化管道配置"""
    # 智能路由配置
    enable_smart_routing: bool = True
    fast_path_threshold: int = 50        # 快速路径阈值
    lsh_activation_threshold: int = 200   # LSH启用阈值
    vectorized_activation_threshold: int = 100  # 向量化启用阈值
    cache_activation_threshold: int = 10  # 缓存启用阈值
    parallel_activation_threshold: int = 20  # 并行启用阈值
    
    # 性能配置
    max_workers: int = 4
    batch_size: int = 100
    cache_ttl: int = 3600
    component_warmup: bool = True        # 组件预热
    lazy_initialization: bool = True     # 延迟初始化
    
    # 快速路径配置
    fast_similarity_threshold: float = 0.5
    enable_fast_path: bool = True


@dataclass
class PipelineDecision:
    """管道决策"""
    use_lsh_prefilter: bool = False
    use_vectorized_compute: bool = False
    use_multi_cache: bool = False
    use_parallel_processing: bool = False
    use_fast_path: bool = False
    estimated_speedup: float = 1.0
    reasoning: str = ""


class SmartComponentRouter:
    """智能组件路由器"""
    
    def __init__(self, config: OptimizedPipelineConfig):
        self.config = config
        self.decision_cache = {}  # 缓存路由决策
        
        # 组件性能基准
        self.component_benchmarks = {
            'lsh_prefilter': {'setup_time': 0.1, 'process_time_per_item': 0.001},
            'vectorized_compute': {'setup_time': 0.05, 'process_time_per_item': 0.0001},
            'multi_cache': {'setup_time': 0.01, 'process_time_per_item': 0.00001},
            'parallel_processing': {'setup_time': 0.2, 'process_time_per_item': 0.0005}
        }
        
        logger.info("智能组件路由器初始化完成")
    
    def make_routing_decision(
        self, 
        data_size: int, 
        operation_type: str,
        complexity_hint: Optional[str] = None
    ) -> PipelineDecision:
        """制定路由决策"""
        # 检查缓存的决策
        cache_key = f"{data_size}_{operation_type}_{complexity_hint}"
        if cache_key in self.decision_cache:
            return self.decision_cache[cache_key]
        
        decision = PipelineDecision()
        reasoning_parts = []
        
        # 快速路径检查
        if (self.config.enable_fast_path and 
            data_size < self.config.fast_path_threshold):
            decision.use_fast_path = True
            decision.estimated_speedup = 2.0  # 快速路径通常有2x加速
            reasoning_parts.append(f"小数据集({data_size}<{self.config.fast_path_threshold})使用快速路径")
        else:
            # 智能组件选择
            decision.use_lsh_prefilter = data_size >= self.config.lsh_activation_threshold
            decision.use_vectorized_compute = data_size >= self.config.vectorized_activation_threshold
            decision.use_multi_cache = data_size >= self.config.cache_activation_threshold
            decision.use_parallel_processing = data_size >= self.config.parallel_activation_threshold
            
            # 估算性能提升
            speedup = 1.0
            
            if decision.use_lsh_prefilter:
                # LSH预过滤通常能减少70-90%的候选
                lsh_speedup = min(3.0, data_size / 100)  # 动态计算
                speedup *= lsh_speedup
                reasoning_parts.append(f"启用LSH预过滤(预计{lsh_speedup:.1f}x加速)")
            
            if decision.use_vectorized_compute:
                vectorized_speedup = min(2.0, data_size / 500)
                speedup *= vectorized_speedup
                reasoning_parts.append(f"启用向量化计算(预计{vectorized_speedup:.1f}x加速)")
            
            if decision.use_multi_cache:
                cache_speedup = 1.2  # 缓存通常有20%提升
                speedup *= cache_speedup
                reasoning_parts.append("启用多级缓存")
            
            if decision.use_parallel_processing:
                parallel_speedup = min(self.config.max_workers * 0.6, 2.5)
                speedup *= parallel_speedup
                reasoning_parts.append(f"启用并行处理(预计{parallel_speedup:.1f}x加速)")
            
            # 考虑组件初始化开销
            total_setup_time = 0
            if decision.use_lsh_prefilter:
                total_setup_time += self.component_benchmarks['lsh_prefilter']['setup_time']
            if decision.use_vectorized_compute:
                total_setup_time += self.component_benchmarks['vectorized_compute']['setup_time']
            if decision.use_multi_cache:
                total_setup_time += self.component_benchmarks['multi_cache']['setup_time']
            if decision.use_parallel_processing:
                total_setup_time += self.component_benchmarks['parallel_processing']['setup_time']
            
            # 如果初始化开销过大，降低预期加速比
            if total_setup_time > 0.5:  # 超过500ms初始化时间
                speedup *= 0.8
                reasoning_parts.append("考虑初始化开销，调整预期加速比")
            
            decision.estimated_speedup = speedup
        
        decision.reasoning = "; ".join(reasoning_parts)
        
        # 缓存决策
        self.decision_cache[cache_key] = decision
        
        return decision


class OptimizedPipeline:
    """优化版搜索管道"""
    
    def __init__(self, config: Optional[OptimizedPipelineConfig] = None):
        self.config = config or OptimizedPipelineConfig()
        self.router = SmartComponentRouter(self.config)
        
        # 延迟初始化的组件
        self._lsh_prefilter = None
        self._vectorized_calculator = None
        self._optimized_calculator = None
        self._multi_cache = None
        
        # 线程池
        self.thread_pool = ThreadPoolExecutor(max_workers=self.config.max_workers)
        
        # 统计信息
        self.stats = {
            'total_queries': 0,
            'fast_path_queries': 0,
            'cache_hits': 0,
            'component_activations': {
                'lsh_prefilter': 0,
                'vectorized_compute': 0,
                'multi_cache': 0,
                'parallel_processing': 0
            },
            'avg_processing_time': 0.0,
            'total_speedup': 0.0
        }
        
        logger.info("优化管道初始化完成")
    
    @property
    def lsh_prefilter(self):
        """延迟加载LSH预过滤器"""
        if self._lsh_prefilter is None and self.config.lazy_initialization:
            from src.tools.lsh_prefilter import get_lsh_prefilter
            self._lsh_prefilter = get_lsh_prefilter()
            logger.debug("LSH预过滤器已延迟加载")
        return self._lsh_prefilter
    
    @property
    def optimized_vectorized_calculator(self):
        """延迟加载优化向量化计算器"""
        if self._optimized_calculator is None and self.config.lazy_initialization:
            from src.tools.optimized_vectorized_calculator import get_optimized_vectorized_calculator
            self._optimized_calculator = get_optimized_vectorized_calculator()
            logger.debug("优化向量化计算器已延迟加载")
        return self._optimized_calculator
    
    @property
    def multi_cache(self):
        """延迟加载多级缓存"""
        if self._multi_cache is None and self.config.lazy_initialization:
            from src.tools.multi_level_cache import get_multi_level_cache
            self._multi_cache = get_multi_level_cache()
            logger.debug("多级缓存已延迟加载")
        return self._multi_cache
    
    @profile_component("optimized_pipeline")
    async def enhanced_column_search(
        self, 
        query_column: ColumnInfo, 
        candidate_columns: List[ColumnInfo],
        k: int = 10
    ) -> List[MatchResult]:
        """优化版列搜索"""
        start_time = time.time()
        self.stats['total_queries'] += 1
        
        try:
            # 智能路由决策
            decision = self.router.make_routing_decision(
                data_size=len(candidate_columns),
                operation_type="column_search"
            )
            
            logger.debug(f"路由决策: {decision.reasoning}")
            
            # 快速路径处理
            if decision.use_fast_path:
                self.stats['fast_path_queries'] += 1
                results = await self._fast_path_column_search(query_column, candidate_columns, k)
            else:
                # 标准管道处理
                results = await self._standard_column_search(query_column, candidate_columns, k, decision)
            
            # 更新统计
            processing_time = time.time() - start_time
            self._update_stats(processing_time, decision.estimated_speedup)
            
            logger.debug(f"优化列搜索完成: {len(results)} 个结果, 用时 {processing_time:.3f}s")
            return results
            
        except Exception as e:
            logger.error(f"优化列搜索失败: {e}")
            return []
    
    async def _fast_path_column_search(
        self,
        query_column: ColumnInfo,
        candidate_columns: List[ColumnInfo],
        k: int
    ) -> List[MatchResult]:
        """快速路径列搜索"""
        results = []
        
        # 简单而快速的相似度计算
        for candidate in candidate_columns[:k*2]:  # 多取一些候选以提高质量
            # 列名相似度
            name_sim = self._fast_string_similarity(
                query_column.column_name, candidate.column_name
            )
            
            # 数据类型匹配
            type_match = 1.0 if query_column.data_type == candidate.data_type else 0.3
            
            # 值重叠度（简化）
            value_sim = 0.0
            if (query_column.sample_values and candidate.sample_values):
                common_values = set(query_column.sample_values) & set(candidate.sample_values)
                total_values = set(query_column.sample_values) | set(candidate.sample_values)
                value_sim = len(common_values) / len(total_values) if total_values else 0
            
            # 综合相似度
            overall_similarity = 0.5 * name_sim + 0.3 * type_match + 0.2 * value_sim
            
            if overall_similarity > self.config.fast_similarity_threshold:
                result = MatchResult(
                    source_column=query_column.full_name,
                    target_column=candidate.full_name,
                    confidence=overall_similarity,
                    reason=f"快速路径匹配: {overall_similarity:.3f}",
                    match_type="fast_path"
                )
                results.append(result)
        
        # 排序并返回前k个
        results.sort(key=lambda x: x.confidence, reverse=True)
        return results[:k]
    
    async def _standard_column_search(
        self,
        query_column: ColumnInfo,
        candidate_columns: List[ColumnInfo],
        k: int,
        decision: PipelineDecision
    ) -> List[MatchResult]:
        """标准管道列搜索"""
        # 检查缓存
        if decision.use_multi_cache:
            cache_key = self.multi_cache.generate_cache_key(
                "opt_column_search", query_column.full_name, len(candidate_columns), k
            )
            cached_result = self.multi_cache.get(cache_key)
            if cached_result:
                self.stats['cache_hits'] += 1
                self.stats['component_activations']['multi_cache'] += 1
                return cached_result
        
        # LSH预过滤
        filtered_candidates = candidate_columns
        if decision.use_lsh_prefilter and self.lsh_prefilter:
            self.stats['component_activations']['lsh_prefilter'] += 1
            query_data = {
                'column_name': query_column.column_name,
                'data_type': query_column.data_type,
                'sample_values': query_column.sample_values or []
            }
            
            candidate_ids = self.lsh_prefilter.prefilter_columns(query_data, max_candidates=k * 5)
            filtered_candidates = [
                c for c in candidate_columns if c.full_name in candidate_ids
            ]
        
        # 向量化相似度计算
        if decision.use_vectorized_compute and len(filtered_candidates) > 10:
            self.stats['component_activations']['vectorized_compute'] += 1
            results = await self._optimized_vectorized_matching(
                query_column, filtered_candidates, k
            )
        else:
            # 传统匹配
            results = await self._traditional_column_matching(
                query_column, filtered_candidates, k
            )
        
        # 缓存结果
        if decision.use_multi_cache and self.multi_cache:
            self.multi_cache.put(cache_key, results, ttl=self.config.cache_ttl)
        
        return results
    
    async def _optimized_vectorized_matching(
        self,
        query_column: ColumnInfo,
        candidates: List[ColumnInfo],
        k: int
    ) -> List[MatchResult]:
        """优化版向量化匹配"""
        try:
            from src.tools.embedding import get_embedding_generator
            embedding_gen = get_embedding_generator()
            
            # 生成向量
            query_text = f"{query_column.column_name} {query_column.data_type}"
            query_embedding = await embedding_gen.generate_text_embedding(query_text)
            
            candidate_embeddings = []
            for candidate in candidates:
                candidate_text = f"{candidate.column_name} {candidate.data_type}"
                candidate_embedding = await embedding_gen.generate_text_embedding(candidate_text)
                candidate_embeddings.append(candidate_embedding)
            
            # 使用优化计算器
            query_array = np.array([query_embedding])
            candidate_array = np.array(candidate_embeddings)
            
            similarity_matrix = self.optimized_vectorized_calculator.optimized_cosine_similarity(
                query_array, candidate_array
            )
            
            # 构建结果
            results = []
            similarities = similarity_matrix[0]
            sorted_indices = np.argsort(similarities)[::-1][:k]
            
            for idx in sorted_indices:
                if similarities[idx] > 0.5:  # 阈值过滤
                    candidate = candidates[idx]
                    result = MatchResult(
                        source_column=query_column.full_name,
                        target_column=candidate.full_name,
                        confidence=float(similarities[idx]),
                        reason=f"优化向量化匹配: {similarities[idx]:.3f}",
                        match_type="optimized_vectorized"
                    )
                    results.append(result)
            
            return results
            
        except Exception as e:
            logger.error(f"优化向量化匹配失败: {e}")
            return []
    
    async def _traditional_column_matching(
        self,
        query_column: ColumnInfo,
        candidates: List[ColumnInfo],
        k: int
    ) -> List[MatchResult]:
        """传统列匹配"""
        results = []
        
        for candidate in candidates[:k]:
            name_similarity = self._fast_string_similarity(
                query_column.column_name, candidate.column_name
            )
            type_match = 1.0 if query_column.data_type == candidate.data_type else 0.3
            overall_similarity = 0.7 * name_similarity + 0.3 * type_match
            
            if overall_similarity > 0.3:
                result = MatchResult(
                    source_column=query_column.full_name,
                    target_column=candidate.full_name,
                    confidence=overall_similarity,
                    reason=f"传统匹配: {overall_similarity:.3f}",
                    match_type="traditional"
                )
                results.append(result)
        
        results.sort(key=lambda x: x.confidence, reverse=True)
        return results[:k]
    
    def _fast_string_similarity(self, str1: str, str2: str) -> float:
        """快速字符串相似度计算"""
        if not str1 or not str2:
            return 0.0
        
        str1, str2 = str1.lower(), str2.lower()
        
        if str1 == str2:
            return 1.0
        
        # 简化的Jaccard相似度
        set1, set2 = set(str1), set(str2)
        intersection = len(set1 & set2)
        union = len(set1 | set2)
        
        return intersection / union if union > 0 else 0.0
    
    def _update_stats(self, processing_time: float, estimated_speedup: float):
        """更新统计信息"""
        # 更新平均处理时间
        total_time = self.stats['avg_processing_time'] * (self.stats['total_queries'] - 1) + processing_time
        self.stats['avg_processing_time'] = total_time / self.stats['total_queries']
        
        # 累计加速比
        self.stats['total_speedup'] += estimated_speedup
    
    def get_optimization_report(self) -> Dict[str, Any]:
        """获取优化报告"""
        # 计算平均加速比
        avg_speedup = self.stats['total_speedup'] / max(1, self.stats['total_queries'])
        
        # 计算组件使用率
        total_activations = sum(self.stats['component_activations'].values())
        component_usage = {}
        for component, count in self.stats['component_activations'].items():
            component_usage[component] = count / max(1, self.stats['total_queries'])
        
        return {
            'performance_summary': {
                'total_queries': self.stats['total_queries'],
                'avg_processing_time': self.stats['avg_processing_time'],
                'avg_estimated_speedup': avg_speedup,
                'fast_path_usage_rate': self.stats['fast_path_queries'] / max(1, self.stats['total_queries']),
                'cache_hit_rate': self.stats['cache_hits'] / max(1, self.stats['total_queries'])
            },
            'component_usage': component_usage,
            'routing_decisions': len(self.router.decision_cache),
            'config': {
                'enable_smart_routing': self.config.enable_smart_routing,
                'fast_path_threshold': self.config.fast_path_threshold,
                'lazy_initialization': self.config.lazy_initialization
            }
        }
    
    def cleanup(self):
        """清理资源"""
        if self.thread_pool:
            self.thread_pool.shutdown(wait=True)
        
        if self._optimized_calculator:
            self._optimized_calculator.cleanup()
        
        # 清理缓存
        self.router.decision_cache.clear()
        
        logger.info("优化管道资源清理完成")


def create_optimized_pipeline(config: Optional[OptimizedPipelineConfig] = None) -> OptimizedPipeline:
    """创建优化管道"""
    return OptimizedPipeline(config)


# 全局优化管道实例
_global_optimized_pipeline = None

def get_optimized_pipeline() -> OptimizedPipeline:
    """获取全局优化管道"""
    global _global_optimized_pipeline
    if _global_optimized_pipeline is None:
        _global_optimized_pipeline = create_optimized_pipeline()
    return _global_optimized_pipeline