"""
并行处理管道 - Phase 2架构升级核心组件
集成LSH预过滤、向量化计算、多级缓存的高性能处理管道
"""

import asyncio
import logging
import time
from typing import List, Dict, Any, Optional, Tuple, Callable
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import numpy as np
from src.config.settings import settings
from src.tools.lsh_prefilter import get_lsh_prefilter, LSHConfig
from src.tools.vectorized_optimizer import get_vectorized_calculator, get_hybrid_similarity_calculator
from src.tools.multi_level_cache import get_multi_level_cache
from src.core.models import ColumnInfo, TableInfo, MatchResult, TableMatchResult

logger = logging.getLogger(__name__)


@dataclass
class PipelineConfig:
    """并行管道配置"""
    enable_lsh_prefilter: bool = True
    enable_vectorized_compute: bool = True
    enable_multi_cache: bool = True
    enable_parallel_processing: bool = True
    max_workers: int = 4
    batch_size: int = 100
    cache_ttl: int = 3600
    prefilter_threshold: float = 0.5


@dataclass
class PipelineStats:
    """管道统计信息"""
    total_queries: int = 0
    cache_hits: int = 0
    lsh_prefilter_reductions: int = 0
    vectorized_computations: int = 0
    parallel_tasks: int = 0
    total_processing_time: float = 0.0
    avg_query_time: float = 0.0


class Phase2SearchPipeline:
    """Phase 2增强搜索管道"""
    
    def __init__(self, config: Optional[PipelineConfig] = None):
        self.config = config or PipelineConfig()
        
        # 初始化组件
        self.lsh_prefilter = get_lsh_prefilter() if self.config.enable_lsh_prefilter else None
        self.vectorized_calculator = get_vectorized_calculator() if self.config.enable_vectorized_compute else None
        self.hybrid_calculator = get_hybrid_similarity_calculator() if self.config.enable_vectorized_compute else None
        self.multi_cache = get_multi_level_cache() if self.config.enable_multi_cache else None
        
        # 线程池
        self.thread_pool = ThreadPoolExecutor(max_workers=self.config.max_workers)
        
        # 统计信息
        self.stats = PipelineStats()
        
        logger.info("Phase 2搜索管道初始化完成")
    
    async def enhanced_column_search(
        self, 
        query_column: ColumnInfo, 
        candidate_columns: List[ColumnInfo],
        k: int = 10
    ) -> List[MatchResult]:
        """增强列搜索 - 完整的Phase 2管道"""
        start_time = time.time()
        self.stats.total_queries += 1
        
        try:
            # 第1步: 检查缓存
            cache_key = None
            if self.multi_cache:
                cache_key = self.multi_cache.generate_cache_key(
                    "column_search",
                    query_column.full_name,
                    len(candidate_columns),
                    k
                )
                cached_result = self.multi_cache.get(cache_key)
                if cached_result:
                    self.stats.cache_hits += 1
                    logger.debug(f"缓存命中: 列搜索 {query_column.full_name}")
                    return cached_result
            
            # 第2步: LSH预过滤
            filtered_candidates = candidate_columns
            if self.lsh_prefilter and len(candidate_columns) > 100:
                query_data = {
                    'column_name': query_column.column_name,
                    'data_type': query_column.data_type,
                    'sample_values': query_column.sample_values or []
                }
                
                candidate_ids = self.lsh_prefilter.prefilter_columns(
                    query_data, max_candidates=k * 5
                )
                
                # 匹配候选列
                filtered_candidates = []
                for candidate in candidate_columns:
                    if candidate.full_name in candidate_ids:
                        filtered_candidates.append(candidate)
                
                reduction_ratio = 1 - (len(filtered_candidates) / len(candidate_columns))
                if reduction_ratio > 0.1:  # 至少10%的减少才记录
                    self.stats.lsh_prefilter_reductions += 1
                    logger.debug(f"LSH预过滤: {len(candidate_columns)} -> {len(filtered_candidates)} "
                               f"(减少 {reduction_ratio:.1%})")
            
            # 第3步: 向量化相似度计算
            if self.vectorized_calculator and len(filtered_candidates) > 10:
                results = await self._vectorized_column_matching(
                    query_column, filtered_candidates, k
                )
                self.stats.vectorized_computations += 1
            else:
                # 降级到传统计算
                results = await self._traditional_column_matching(
                    query_column, filtered_candidates, k
                )
            
            # 第4步: 缓存结果
            if self.multi_cache and cache_key:
                self.multi_cache.put(cache_key, results, ttl=self.config.cache_ttl)
            
            # 更新统计
            processing_time = time.time() - start_time
            self.stats.total_processing_time += processing_time
            self.stats.avg_query_time = self.stats.total_processing_time / self.stats.total_queries
            
            logger.debug(f"增强列搜索完成: {len(results)} 个结果, 用时 {processing_time:.3f}s")
            return results
            
        except Exception as e:
            logger.error(f"增强列搜索失败: {e}")
            return []
    
    async def enhanced_table_search(
        self,
        query_table: TableInfo,
        candidate_tables: List[TableInfo],
        k: int = 10
    ) -> List[TableMatchResult]:
        """增强表搜索 - 完整的Phase 2管道"""
        start_time = time.time()
        self.stats.total_queries += 1
        
        try:
            # 第1步: 检查缓存
            cache_key = None
            if self.multi_cache:
                cache_key = self.multi_cache.generate_cache_key(
                    "table_search",
                    query_table.table_name,
                    len(candidate_tables),
                    k
                )
                cached_result = self.multi_cache.get(cache_key)
                if cached_result:
                    self.stats.cache_hits += 1
                    logger.debug(f"缓存命中: 表搜索 {query_table.table_name}")
                    return cached_result
            
            # 第2步: LSH预过滤
            filtered_candidates = candidate_tables
            if self.lsh_prefilter and len(candidate_tables) > 50:
                query_data = {
                    'table_name': query_table.table_name,
                    'columns': [{'column_name': col.column_name, 'data_type': col.data_type} 
                               for col in query_table.columns]
                }
                
                candidate_names = self.lsh_prefilter.prefilter_tables(
                    query_data, max_candidates=k * 3
                )
                
                # 匹配候选表
                filtered_candidates = []
                for candidate in candidate_tables:
                    if candidate.table_name in candidate_names:
                        filtered_candidates.append(candidate)
                
                reduction_ratio = 1 - (len(filtered_candidates) / len(candidate_tables))
                if reduction_ratio > 0.1:
                    self.stats.lsh_prefilter_reductions += 1
                    logger.debug(f"LSH预过滤: {len(candidate_tables)} -> {len(filtered_candidates)} "
                               f"(减少 {reduction_ratio:.1%})")
            
            # 第3步: 并行表匹配
            if self.config.enable_parallel_processing and len(filtered_candidates) > 5:
                results = await self._parallel_table_matching(
                    query_table, filtered_candidates, k
                )
                self.stats.parallel_tasks += 1
            else:
                # 串行处理
                results = await self._sequential_table_matching(
                    query_table, filtered_candidates, k
                )
            
            # 第4步: 缓存结果
            if self.multi_cache and cache_key:
                self.multi_cache.put(cache_key, results, ttl=self.config.cache_ttl)
            
            # 更新统计
            processing_time = time.time() - start_time
            self.stats.total_processing_time += processing_time
            self.stats.avg_query_time = self.stats.total_processing_time / self.stats.total_queries
            
            logger.debug(f"增强表搜索完成: {len(results)} 个结果, 用时 {processing_time:.3f}s")
            return results
            
        except Exception as e:
            logger.error(f"增强表搜索失败: {e}")
            return []
    
    async def _vectorized_column_matching(
        self,
        query_column: ColumnInfo,
        candidates: List[ColumnInfo],
        k: int
    ) -> List[MatchResult]:
        """向量化列匹配"""
        try:
            from src.tools.embedding import get_embedding_generator
            embedding_gen = get_embedding_generator()
            
            # 生成查询向量
            query_text = f"{query_column.column_name} {query_column.data_type}"
            query_embedding = await embedding_gen.generate_text_embedding(query_text)
            
            # 生成候选向量
            candidate_embeddings = []
            for candidate in candidates:
                candidate_text = f"{candidate.column_name} {candidate.data_type}"
                candidate_embedding = await embedding_gen.generate_text_embedding(candidate_text)
                candidate_embeddings.append(candidate_embedding)
            
            # 向量化相似度计算
            query_array = np.array([query_embedding])
            candidate_array = np.array(candidate_embeddings)
            
            similarity_matrix = self.vectorized_calculator.batch_cosine_similarity(
                query_array, candidate_array
            )
            
            # 构建结果
            results = []
            similarities = similarity_matrix[0]  # 取第一行
            
            # 排序并取前k个
            sorted_indices = np.argsort(similarities)[::-1][:k]
            
            for idx in sorted_indices:
                if similarities[idx] > self.config.prefilter_threshold:
                    candidate = candidates[idx]
                    result = MatchResult(
                        source_column=query_column.full_name,
                        target_column=candidate.full_name,
                        confidence=float(similarities[idx]),
                        reason=f"向量化相似度计算: {similarities[idx]:.3f}",
                        match_type="vectorized"
                    )
                    results.append(result)
            
            return results
            
        except Exception as e:
            logger.error(f"向量化列匹配失败: {e}")
            return []
    
    async def _traditional_column_matching(
        self,
        query_column: ColumnInfo,
        candidates: List[ColumnInfo],
        k: int
    ) -> List[MatchResult]:
        """传统列匹配（降级处理）"""
        results = []
        
        for candidate in candidates[:k]:
            # 简单的名称相似度
            name_similarity = self._simple_string_similarity(
                query_column.column_name, candidate.column_name
            )
            
            # 数据类型匹配
            type_match = 1.0 if query_column.data_type == candidate.data_type else 0.3
            
            # 综合相似度
            overall_similarity = 0.7 * name_similarity + 0.3 * type_match
            
            if overall_similarity > self.config.prefilter_threshold:
                result = MatchResult(
                    source_column=query_column.full_name,
                    target_column=candidate.full_name,
                    confidence=overall_similarity,
                    reason=f"传统相似度计算: {overall_similarity:.3f}",
                    match_type="traditional"
                )
                results.append(result)
        
        # 按相似度排序
        results.sort(key=lambda x: x.confidence, reverse=True)
        return results[:k]
    
    async def _parallel_table_matching(
        self,
        query_table: TableInfo,
        candidates: List[TableInfo],
        k: int
    ) -> List[TableMatchResult]:
        """并行表匹配"""
        try:
            # 分批处理
            batch_size = max(1, len(candidates) // self.config.max_workers)
            batches = [candidates[i:i + batch_size] for i in range(0, len(candidates), batch_size)]
            
            # 提交并行任务
            loop = asyncio.get_event_loop()
            futures = []
            
            for batch in batches:
                future = loop.run_in_executor(
                    self.thread_pool,
                    self._match_table_batch,
                    query_table,
                    batch
                )
                futures.append(future)
            
            # 收集结果
            all_results = []
            for future in asyncio.as_completed(futures):
                batch_results = await future
                all_results.extend(batch_results)
            
            # 排序并返回前k个
            all_results.sort(key=lambda x: x.score, reverse=True)
            return all_results[:k]
            
        except Exception as e:
            logger.error(f"并行表匹配失败: {e}")
            return []
    
    def _match_table_batch(
        self,
        query_table: TableInfo,
        candidate_batch: List[TableInfo]
    ) -> List[TableMatchResult]:
        """匹配表批次（在线程池中执行）"""
        results = []
        
        for candidate in candidate_batch:
            try:
                # 列级别匹配
                column_matches = []
                for query_col in query_table.columns:
                    best_match = None
                    best_score = 0
                    
                    for candidate_col in candidate.columns:
                        # 简单相似度计算
                        name_sim = self._simple_string_similarity(
                            query_col.column_name, candidate_col.column_name
                        )
                        type_match = 1.0 if query_col.data_type == candidate_col.data_type else 0.3
                        score = 0.7 * name_sim + 0.3 * type_match
                        
                        if score > best_score and score > 0.5:
                            best_score = score
                            best_match = MatchResult(
                                source_column=query_col.full_name,
                                target_column=candidate_col.full_name,
                                confidence=score,
                                reason=f"表匹配中的列匹配: {score:.3f}",
                                match_type="table_matching"
                            )
                    
                    if best_match:
                        column_matches.append(best_match)
                
                # 计算表级别分数
                if column_matches:
                    avg_confidence = sum(m.confidence for m in column_matches) / len(column_matches)
                    coverage = len(column_matches) / len(query_table.columns)
                    table_score = avg_confidence * coverage * 100
                    
                    result = TableMatchResult(
                        source_table=query_table.table_name,
                        target_table=candidate.table_name,
                        score=table_score,
                        matched_columns=column_matches,
                        evidence={
                            'matching_method': 'parallel_pipeline',
                            'column_matches': len(column_matches),
                            'avg_confidence': avg_confidence,
                            'coverage': coverage
                        }
                    )
                    results.append(result)
                    
            except Exception as e:
                logger.error(f"批次表匹配失败 {candidate.table_name}: {e}")
        
        return results
    
    async def _sequential_table_matching(
        self,
        query_table: TableInfo,
        candidates: List[TableInfo],
        k: int
    ) -> List[TableMatchResult]:
        """串行表匹配"""
        return self._match_table_batch(query_table, candidates[:k])
    
    def _simple_string_similarity(self, str1: str, str2: str) -> float:
        """简单字符串相似度计算"""
        if not str1 or not str2:
            return 0.0
        
        str1 = str1.lower()
        str2 = str2.lower()
        
        if str1 == str2:
            return 1.0
        
        # Jaccard相似度
        set1 = set(str1)
        set2 = set(str2)
        intersection = len(set1 & set2)
        union = len(set1 | set2)
        
        return intersection / union if union > 0 else 0.0
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """获取性能统计"""
        stats = {
            'pipeline_stats': {
                'total_queries': self.stats.total_queries,
                'cache_hits': self.stats.cache_hits,
                'cache_hit_rate': self.stats.cache_hits / max(1, self.stats.total_queries),
                'lsh_prefilter_reductions': self.stats.lsh_prefilter_reductions,
                'vectorized_computations': self.stats.vectorized_computations,
                'parallel_tasks': self.stats.parallel_tasks,
                'avg_query_time': self.stats.avg_query_time
            },
            'config': {
                'enable_lsh_prefilter': self.config.enable_lsh_prefilter,
                'enable_vectorized_compute': self.config.enable_vectorized_compute,
                'enable_multi_cache': self.config.enable_multi_cache,
                'enable_parallel_processing': self.config.enable_parallel_processing,
                'max_workers': self.config.max_workers
            }
        }
        
        # 添加组件统计
        if self.lsh_prefilter:
            stats['lsh_prefilter'] = self.lsh_prefilter.get_performance_stats()
        
        if self.vectorized_calculator:
            stats['vectorized_calculator'] = self.vectorized_calculator.get_performance_stats()
        
        if self.multi_cache:
            stats['multi_cache'] = self.multi_cache.get_comprehensive_stats()
        
        return stats
    
    def cleanup(self):
        """清理资源"""
        if self.thread_pool:
            self.thread_pool.shutdown(wait=True)
        
        if self.multi_cache:
            self.multi_cache.cleanup_expired()
        
        logger.info("Phase 2搜索管道资源清理完成")


def create_phase2_pipeline(config: Optional[PipelineConfig] = None) -> Phase2SearchPipeline:
    """创建Phase 2搜索管道"""
    return Phase2SearchPipeline(config)


# 全局管道实例
_global_pipeline = None

def get_phase2_pipeline() -> Phase2SearchPipeline:
    """获取全局Phase 2搜索管道"""
    global _global_pipeline
    if _global_pipeline is None:
        _global_pipeline = create_phase2_pipeline()
    return _global_pipeline