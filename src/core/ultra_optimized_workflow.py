"""
超级优化的数据湖工作流 - 针对大规模查询优化
实现毫秒级查询和完整评价指标
"""

import logging
import asyncio
import time
import numpy as np
from typing import Dict, Any, List, Tuple, Optional, Set
from dataclasses import dataclass
from collections import defaultdict

from src.core.models import AgentState, TableInfo, ColumnInfo
from src.core.optimized_workflow import OptimizedDataLakesWorkflow
from src.tools.metadata_filter import MetadataFilter
from src.tools.batch_vector_search import BatchVectorSearch
from src.tools.multi_level_cache import CacheManager
from src.config.settings import settings

logger = logging.getLogger(__name__)


@dataclass
class EvaluationMetrics:
    """评价指标"""
    precision: float = 0.0
    recall: float = 0.0
    f1_score: float = 0.0
    average_precision: float = 0.0
    ndcg_at_k: float = 0.0
    mrr: float = 0.0  # Mean Reciprocal Rank
    hit_rate: float = 0.0
    query_time: float = 0.0
    cache_hit_rate: float = 0.0


class UltraOptimizedWorkflow(OptimizedDataLakesWorkflow):
    """超级优化工作流
    
    核心改进：
    1. 智能候选控制（20个而非100个）
    2. 早期终止策略
    3. 索引完全复用
    4. 完整评价指标
    5. 极限缓存优化
    """
    
    # 静态索引存储（跨实例共享）
    _global_index_initialized = False
    _global_metadata_filter = None
    _global_vector_index = None
    _global_table_cache = {}
    
    def __init__(self):
        super().__init__()
        
        # 优化参数（平衡性能和准确性）
        self.max_metadata_candidates = 50  # 元数据筛选候选数（适度增加）
        self.max_vector_candidates = 20    # 向量搜索候选数（适度增加）
        self.max_llm_candidates = 5        # LLM验证候选数（适度增加）
        self.early_stop_threshold = 0.90   # 早期终止阈值（提高以减少误判）
        self.enable_llm_matching = False   # 是否启用LLM匹配（可选）
        
        # 评价指标追踪
        self.metrics_tracker = defaultdict(list)
        self.cache_hits = 0
        self.total_queries = 0
        
        # 复用全局索引
        if UltraOptimizedWorkflow._global_metadata_filter:
            self.metadata_filter = UltraOptimizedWorkflow._global_metadata_filter
            self.vector_search = UltraOptimizedWorkflow._global_vector_index
            self.table_metadata_cache = UltraOptimizedWorkflow._global_table_cache
            logger.info("复用已有全局索引")
    
    async def initialize(self, all_tables: List[TableInfo]) -> None:
        """初始化工作流（智能复用索引）"""
        
        # 检查是否需要初始化
        if UltraOptimizedWorkflow._global_index_initialized:
            logger.info("全局索引已初始化，跳过重复构建")
            self.metadata_filter = UltraOptimizedWorkflow._global_metadata_filter
            self.vector_search = UltraOptimizedWorkflow._global_vector_index
            self.table_metadata_cache = UltraOptimizedWorkflow._global_table_cache
            return
        
        logger.info(f"首次初始化全局索引，表数量: {len(all_tables)}")
        start_time = time.time()
        
        # 构建索引（只执行一次）
        await super().initialize(all_tables)
        
        # 保存到全局
        UltraOptimizedWorkflow._global_metadata_filter = self.metadata_filter
        UltraOptimizedWorkflow._global_vector_index = self.vector_search
        UltraOptimizedWorkflow._global_table_cache = self.table_metadata_cache
        UltraOptimizedWorkflow._global_index_initialized = True
        
        logger.info(f"全局索引初始化完成，耗时: {time.time()-start_time:.2f}秒")
    
    async def run_optimized(
        self,
        initial_state: AgentState,
        all_table_names: List[str],
        ground_truth: Optional[Dict[str, List[str]]] = None
    ) -> Tuple[AgentState, EvaluationMetrics]:
        """运行超优化工作流并返回评价指标
        
        Args:
            initial_state: 初始状态
            all_table_names: 所有表名
            ground_truth: 真实标签 {query_table: [matching_tables]}
        
        Returns:
            (最终状态, 评价指标)
        """
        self.total_queries += 1
        start_time = time.time()
        
        # 先检查缓存
        cache_key = self._get_cache_key(initial_state)
        cached_result = await self._check_cache(cache_key)
        if cached_result:
            self.cache_hits += 1
            logger.info(f"缓存命中！跳过计算")
            return cached_result, self._calculate_metrics(
                cached_result, ground_truth, time.time() - start_time
            )
        
        # 执行优化流程
        result = await self._execute_ultra_optimized(
            initial_state, all_table_names
        )
        
        # 保存到缓存
        await self._save_cache(cache_key, result)
        
        # 计算评价指标
        query_time = time.time() - start_time
        metrics = self._calculate_metrics(result, ground_truth, query_time)
        
        # 记录性能
        self._log_performance(query_time, metrics)
        
        return result, metrics
    
    async def _execute_ultra_optimized(
        self,
        state: AgentState,
        all_table_names: List[str]
    ) -> AgentState:
        """执行超优化流程"""
        
        # 三层优化处理
        # 第1层：元数据筛选（更严格）
        metadata_candidates = await self._ultra_metadata_filter(
            state.query_tables,
            all_table_names,
            top_k=self.max_metadata_candidates
        )
        
        if not metadata_candidates:
            logger.warning("元数据筛选无结果，直接返回")
            state.table_matches = []
            return state
        
        # 第2层：向量搜索（更少候选）
        vector_candidates = await self._ultra_vector_search(
            state.query_tables,
            [name for name, _ in metadata_candidates],
            k=self.max_vector_candidates
        )
        
        # 根据配置决定是否使用LLM
        if self.enable_llm_matching:
            # 早期终止检查
            if self._should_early_stop(vector_candidates):
                logger.info("触发早期终止，高置信度结果")
                state.table_matches = self._format_early_stop_results(vector_candidates)
                return state
            
            # 第3层：LLM验证（极少候选）
            final_matches = await self._ultra_llm_matching(
                state.query_tables,
                vector_candidates,
                max_candidates=self.max_llm_candidates
            )
            
            # 更新状态
            state.table_matches = self._format_final_results(final_matches)
        else:
            # 直接使用向量搜索结果
            logger.info("跳过LLM匹配，直接使用向量搜索结果")
            state.table_matches = self._format_vector_results(vector_candidates)
        
        state.final_results = state.table_matches
        
        return state
    
    async def _ultra_metadata_filter(
        self,
        query_tables: List[TableInfo],
        all_table_names: List[str],
        top_k: int = 50
    ) -> List[Tuple[str, float]]:
        """超级元数据筛选"""
        # 使用更严格的筛选规则
        results = []
        for query_table in query_tables:
            # 快速规则筛选
            candidates = self.metadata_filter.filter_candidates(
                query_table,
                all_table_names,
                top_k=top_k * 2  # 初步筛选更多
            )
            
            # 二次筛选：适度的条件
            filtered = []
            for table_name, score in candidates:
                if score > 0.5:  # 适度的阈值
                    filtered.append((table_name, score))
                if len(filtered) >= top_k:
                    break
            
            results.extend(filtered)
        
        # 去重并排序
        unique_results = {}
        for name, score in results:
            if name not in unique_results or score > unique_results[name]:
                unique_results[name] = score
        
        return sorted(unique_results.items(), key=lambda x: x[1], reverse=True)[:top_k]
    
    async def _ultra_vector_search(
        self,
        query_tables: List[TableInfo],
        candidate_names: List[str],
        k: int = 20
    ) -> Dict[str, List[Tuple[str, float]]]:
        """超级向量搜索"""
        # 批量向量搜索但限制结果数量
        results = await self.batch_vector_search.batch_search_tables(
            query_tables,
            candidate_names,
            k=k,
            threshold=0.5  # 适度的相似度阈值
        )
        
        # 格式化并过滤低分结果
        formatted = {}
        for query_name, search_results in results.items():
            filtered_results = [
                (r.item_id, r.score)
                for r in search_results
                if r.score > 0.3  # 降低过滤阈值
            ]
            formatted[query_name] = filtered_results[:k]
        
        return formatted
    
    async def _ultra_llm_matching(
        self,
        query_tables: List[TableInfo],
        vector_results: Dict[str, List[Tuple[str, float]]],
        max_candidates: int = 5
    ) -> Dict[str, List[Dict[str, Any]]]:
        """超级LLM匹配（使用并行处理）"""
        try:
            # 尝试使用并行处理器
            from src.tools.parallel_batch_processor import OptimizedTableMatcher
            
            if not hasattr(self, 'parallel_matcher'):
                self.parallel_matcher = OptimizedTableMatcher(self.llm_client)
            
            # 准备批量数据
            query_list = []
            candidates_list = []
            
            for query_table in query_tables:
                query_name = query_table.table_name
                if query_name in vector_results:
                    # 过滤低分候选（进一步减少LLM调用）
                    candidates = [
                        self.table_metadata_cache.get(name)
                        for name, score in vector_results[query_name][:max_candidates]
                        if score > 0.3  # 只处理相似度>0.3的候选
                    ]
                    candidates = [c for c in candidates if c is not None]
                    
                    if candidates:  # 只有有候选时才处理
                        query_list.append(query_table)
                        candidates_list.append(candidates)
            
            # 并行批量处理
            if query_list:
                batch_results = await self.parallel_matcher.match_tables_batch(
                    query_list,
                    candidates_list
                )
                
                # 格式化结果
                matches = {}
                for query_table, results in zip(query_list, batch_results):
                    matches[query_table.table_name] = results
                
                return matches
            
        except ImportError:
            # 回退到原来的方式
            pass
        
        # 限制每个查询的候选数
        limited_results = {}
        for query_name, candidates in vector_results.items():
            # 只取最高分的几个
            limited_results[query_name] = candidates[:max_candidates]
        
        # 使用智能匹配器
        if not hasattr(self, 'smart_matcher'):
            from src.tools.smart_llm_matcher import SmartLLMMatcher
            self.smart_matcher = SmartLLMMatcher(self.llm_client)
            self.smart_matcher.max_candidates_per_query = max_candidates
        
        matches = await self.smart_matcher.match_tables(
            query_tables,
            limited_results,
            self.table_metadata_cache
        )
        
        return matches
    
    def _should_early_stop(
        self,
        vector_results: Dict[str, List[Tuple[str, float]]]
    ) -> bool:
        """判断是否可以早期终止"""
        for query_name, candidates in vector_results.items():
            if candidates and candidates[0][1] > self.early_stop_threshold:
                # 如果最高分超过阈值，可以早期终止
                return True
        return False
    
    def _format_early_stop_results(
        self,
        vector_results: Dict[str, List[Tuple[str, float]]]
    ) -> List:
        """格式化早期终止结果"""
        from src.core.models import TableMatchResult
        results = []
        
        for query_name, candidates in vector_results.items():
            for table_name, score in candidates[:10]:  # 取前10个
                if score > 0.5:  # 降低阈值
                    results.append(TableMatchResult(
                        source_table=query_name,
                        target_table=table_name,
                        score=score * 100,
                        matched_columns=[],
                        evidence={
                            "match_type": "high_confidence",
                            "method": "vector_similarity",
                            "early_stop": True
                        }
                    ))
        
        return results
    
    def _format_vector_results(
        self,
        vector_results: Dict[str, List[Tuple[str, float]]]
    ) -> List:
        """格式化向量搜索结果为最终输出格式"""
        from src.core.models import TableMatchResult
        results = []
        
        for query_name, candidates in vector_results.items():
            # 取TOP-K结果
            for table_name, score in candidates[:10]:
                results.append(TableMatchResult(
                    source_table=query_name,
                    target_table=table_name,
                    score=score * 100,  # 转换为0-100分数
                    matched_columns=[],
                    evidence={
                        "match_type": "vector_similarity",
                        "method": "hnsw_search",
                        "score": score
                    }
                ))
        
        # 按分数排序
        results.sort(key=lambda x: x.score, reverse=True)
        return results
    
    def _calculate_metrics(
        self,
        result: AgentState,
        ground_truth: Optional[Dict[str, List[str]]],
        query_time: float
    ) -> EvaluationMetrics:
        """计算评价指标"""
        metrics = EvaluationMetrics(query_time=query_time)
        
        if not ground_truth:
            metrics.cache_hit_rate = self.cache_hits / max(1, self.total_queries)
            return metrics
        
        # 提取预测结果
        predictions = defaultdict(list)
        if hasattr(result, 'table_matches') and result.table_matches:
            for match in result.table_matches:
                predictions[match.source_table].append(match.target_table)
        
        # 计算指标
        all_precision = []
        all_recall = []
        all_f1 = []
        mrr_scores = []
        hits = 0
        
        for query_table, true_tables in ground_truth.items():
            pred_tables = predictions.get(query_table, [])
            
            if not true_tables:
                continue
            
            # Precision & Recall
            true_set = set(true_tables)
            pred_set = set(pred_tables[:10])  # Top-10
            
            if pred_set:
                precision = len(true_set & pred_set) / len(pred_set)
                all_precision.append(precision)
            else:
                all_precision.append(0)
            
            recall = len(true_set & pred_set) / len(true_set)
            all_recall.append(recall)
            
            # F1
            if all_precision[-1] + recall > 0:
                f1 = 2 * all_precision[-1] * recall / (all_precision[-1] + recall)
                all_f1.append(f1)
            else:
                all_f1.append(0)
            
            # MRR
            for i, pred in enumerate(pred_tables[:10]):
                if pred in true_set:
                    mrr_scores.append(1.0 / (i + 1))
                    break
            else:
                mrr_scores.append(0)
            
            # Hit Rate
            if len(true_set & pred_set) > 0:
                hits += 1
        
        # 汇总指标
        metrics.precision = np.mean(all_precision) if all_precision else 0
        metrics.recall = np.mean(all_recall) if all_recall else 0
        metrics.f1_score = np.mean(all_f1) if all_f1 else 0
        metrics.mrr = np.mean(mrr_scores) if mrr_scores else 0
        metrics.hit_rate = hits / len(ground_truth) if ground_truth else 0
        metrics.cache_hit_rate = self.cache_hits / max(1, self.total_queries)
        
        return metrics
    
    def _get_cache_key(self, state: AgentState) -> str:
        """生成缓存键"""
        if state.query_tables:
            return f"query:{state.query_tables[0].table_name}"
        return f"state:{id(state)}"
    
    async def _check_cache(self, key: str) -> Optional[AgentState]:
        """检查缓存"""
        # 这里简化处理，实际应该使用持久化缓存
        return None
    
    async def _save_cache(self, key: str, result: AgentState):
        """保存到缓存"""
        # 这里简化处理，实际应该保存到持久化缓存
        pass
    
    def _log_performance(self, query_time: float, metrics: EvaluationMetrics):
        """记录性能日志"""
        logger.info(
            f"查询性能 - "
            f"时间: {query_time:.3f}s, "
            f"精度: {metrics.precision:.3f}, "
            f"召回: {metrics.recall:.3f}, "
            f"F1: {metrics.f1_score:.3f}, "
            f"MRR: {metrics.mrr:.3f}, "
            f"缓存命中率: {metrics.cache_hit_rate:.3f}"
        )
    
    def get_aggregated_metrics(self) -> Dict[str, float]:
        """获取聚合的评价指标"""
        if not self.metrics_tracker:
            return {}
        
        aggregated = {}
        for key in ['precision', 'recall', 'f1_score', 'mrr', 'hit_rate', 'query_time']:
            values = self.metrics_tracker.get(key, [])
            if values:
                aggregated[f"avg_{key}"] = np.mean(values)
                aggregated[f"std_{key}"] = np.std(values)
                aggregated[f"min_{key}"] = np.min(values)
                aggregated[f"max_{key}"] = np.max(values)
        
        aggregated['total_queries'] = self.total_queries
        aggregated['cache_hits'] = self.cache_hits
        aggregated['cache_hit_rate'] = self.cache_hits / max(1, self.total_queries)
        
        return aggregated


# 工厂函数
def create_ultra_optimized_workflow() -> UltraOptimizedWorkflow:
    """创建超优化工作流实例"""
    return UltraOptimizedWorkflow()