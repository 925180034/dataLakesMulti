"""
增强表匹配智能体 - 集成匈牙利算法的精确匹配
基于Phase 1架构升级计划实现
"""

from typing import List, Dict, Any, Tuple, Optional
import logging
import asyncio
from src.agents.base import BaseAgent
from src.core.models import (
    AgentState, TableInfo, ColumnInfo, MatchResult, TableMatchResult
)
from src.tools.embedding import get_embedding_generator
from src.tools.hungarian_matcher import create_hungarian_matcher
from src.tools.hybrid_similarity import HybridSimilarityCalculator
from src.config.settings import settings

logger = logging.getLogger(__name__)


class EnhancedTableMatchingAgent(BaseAgent):
    """增强表匹配智能体 - 集成匈牙利算法精确匹配
    
    技术特点：
    - 使用匈牙利算法进行精确的二分图匹配
    - 集成混合相似度计算引擎
    - 支持批量匹配和阈值过滤
    - 提供详细的匹配解释和置信度
    """
    
    def __init__(self):
        super().__init__("EnhancedTableMatchingAgent")
        self.embedding_gen = get_embedding_generator()
        
        # 初始化匈牙利匹配器
        threshold = settings.hungarian_matcher.similarity_threshold
        self.hungarian_matcher = create_hungarian_matcher(threshold=threshold)
        
        # 初始化混合相似度计算器
        self.hybrid_calculator = HybridSimilarityCalculator()
        
        # 缓存表信息
        self.table_info_cache: Dict[str, TableInfo] = {}
        
        # 性能统计
        self.stats = {
            "total_matches": 0,
            "hungarian_matches": 0,
            "avg_matching_time": 0.0
        }
    
    async def process(self, state: AgentState) -> AgentState:
        """处理增强表匹配任务"""
        state = self._ensure_agent_state(state)
        self.log_progress(state, "开始增强表匹配处理（匈牙利算法）")
        
        if not state.query_tables or not state.table_candidates:
            self.log_error(state, "缺少查询表或候选表信息")
            return state
        
        try:
            import time
            start_time = time.time()
            
            # 执行精确匹配
            matching_results = await self._perform_enhanced_matching(
                state.query_tables, state.table_candidates
            )
            
            # 按匹配分数排序
            matching_results.sort(key=lambda x: x.score, reverse=True)
            
            # 保存结果
            state.table_matches = matching_results
            state.final_results = matching_results[:settings.thresholds.top_k_results]
            
            # 更新统计信息
            elapsed_time = time.time() - start_time
            self.stats["total_matches"] += len(matching_results)
            self.stats["hungarian_matches"] += sum(
                1 for r in matching_results if "hungarian" in r.evidence.get("matching_method", "")
            )
            self.stats["avg_matching_time"] = elapsed_time
            
            self.log_progress(
                state, 
                f"增强表匹配完成，找到 {len(matching_results)} 个匹配，"
                f"用时 {elapsed_time:.2f}s"
            )
            
            state.current_step = "finalization"
            
        except Exception as e:
            self.log_error(state, f"增强表匹配处理失败: {e}")
        
        return state
    
    async def _perform_enhanced_matching(
        self, 
        query_tables: List[TableInfo], 
        candidate_table_names: List[str]
    ) -> List[TableMatchResult]:
        """执行增强匹配"""
        all_results = []
        
        for query_table in query_tables:
            # 准备查询表数据
            query_columns = query_table.columns
            query_embeddings = await self._get_column_embeddings(query_columns)
            
            if not query_embeddings:
                logger.warning(f"无法获取查询表 {query_table.table_name} 的列向量")
                continue
            
            # 准备候选表数据
            candidate_tables_data = []
            for table_name in candidate_table_names:
                candidate_table = await self._get_table_info(table_name)
                if candidate_table and candidate_table.columns:
                    candidate_embeddings = await self._get_column_embeddings(candidate_table.columns)
                    if candidate_embeddings:
                        candidate_tables_data.append((candidate_table.columns, candidate_embeddings))
            
            if not candidate_tables_data:
                logger.warning("没有有效的候选表数据")
                continue
            
            # 使用匈牙利算法进行批量匹配
            hungarian_results = self.hungarian_matcher.batch_match_tables(
                query_table=(query_columns, query_embeddings),
                candidate_tables=candidate_tables_data,
                threshold=settings.hungarian_matcher.similarity_threshold,
                top_k=settings.thresholds.top_k_results * 2  # 获取更多候选以供后续筛选
            )
            
            # 转换为标准格式并添加增强信息
            for result in hungarian_results:
                if result.get("match_count", 0) > 0:
                    enhanced_result = await self._create_enhanced_result(
                        query_table, result, query_columns
                    )
                    if enhanced_result:
                        all_results.append(enhanced_result)
        
        return all_results
    
    async def _create_enhanced_result(
        self, 
        query_table: TableInfo, 
        hungarian_result: Dict[str, Any],
        query_columns: List[ColumnInfo]
    ) -> Optional[TableMatchResult]:
        """创建增强的匹配结果"""
        try:
            target_table_name = hungarian_result.get("candidate_table_name", "unknown")
            
            # 创建匹配列结果
            matched_columns = []
            for match_detail in hungarian_result.get("detailed_matches", []):
                # 构建MatchResult对象
                match_result = MatchResult(
                    source_column=match_detail["table1_column"]["full_name"],
                    target_column=match_detail["table2_column"]["full_name"],
                    confidence=match_detail["similarity"],
                    reason=f"匈牙利算法精确匹配 (相似度: {match_detail['similarity']:.3f})",
                    match_type="hungarian_optimal"
                )
                matched_columns.append(match_result)
            
            # 计算增强评分
            scores = hungarian_result.get("scores", {})
            enhanced_score = self._calculate_enhanced_score(scores)
            
            # 创建详细证据
            evidence = {
                "matching_method": "hungarian_algorithm",
                "total_score": hungarian_result.get("total_score", 0),
                "match_count": hungarian_result.get("match_count", 0),
                "table1_columns": hungarian_result.get("table1_columns", 0),
                "table2_columns": hungarian_result.get("table2_columns", 0),
                "match_ratio": hungarian_result.get("match_ratio", 0),
                "average_similarity": hungarian_result.get("average_similarity", 0),
                "scores": scores,
                "threshold_used": hungarian_result.get("threshold_used", 0),
                "explanation": self.hungarian_matcher.explain_matching(hungarian_result)
            }
            
            return TableMatchResult(
                source_table=query_table.table_name,
                target_table=target_table_name,
                score=enhanced_score,
                matched_columns=matched_columns,
                evidence=evidence
            )
            
        except Exception as e:
            logger.error(f"创建增强结果失败: {e}")
            return None
    
    def _calculate_enhanced_score(self, scores: Dict[str, float]) -> float:
        """计算增强评分"""
        if not scores:
            return 0.0
        
        # 使用配置的权重
        weights = settings.hungarian_matcher.scoring_weights
        
        # 计算加权分数
        weighted_score = 0.0
        total_weight = 0.0
        
        for score_type, weight in weights.items():
            if score_type in scores:
                weighted_score += scores[score_type] * weight
                total_weight += weight
        
        # 标准化到0-100分制
        if total_weight > 0:
            normalized_score = (weighted_score / total_weight) * 100
            return min(normalized_score, 100.0)
        
        return 0.0
    
    async def _get_column_embeddings(self, columns: List[ColumnInfo]) -> List[List[float]]:
        """获取列的向量表示"""
        embeddings = []
        
        for column in columns:
            try:
                # 使用混合相似度计算器获取列的特征表示
                column_text = self._format_column_for_embedding(column)
                embedding = await self.embedding_gen.generate_text_embedding(column_text)
                embeddings.append(embedding)
                
            except Exception as e:
                logger.error(f"获取列向量失败 {column.full_name}: {e}")
                # 使用零向量作为后备
                embeddings.append([0.0] * 384)  # 默认维度
        
        return embeddings
    
    def _format_column_for_embedding(self, column: ColumnInfo) -> str:
        """格式化列信息用于向量化"""
        parts = [column.column_name]
        
        if column.data_type:
            parts.append(f"类型:{column.data_type}")
        
        if column.sample_values:
            sample_str = " ".join(str(v) for v in column.sample_values[:3] if v is not None)
            if sample_str:
                parts.append(f"样本:{sample_str}")
        
        return " ".join(parts)
    
    async def _get_table_info(self, table_name: str) -> Optional[TableInfo]:
        """获取表信息"""
        if table_name in self.table_info_cache:
            return self.table_info_cache[table_name]
        
        try:
            # 从向量搜索引擎获取表信息
            from src.tools.vector_search import get_vector_search_engine
            vector_search_engine = get_vector_search_engine()
            
            # 检查当前使用的搜索引擎类型
            if hasattr(vector_search_engine, 'table_metadata'):
                # FAISS搜索引擎
                for table_info in vector_search_engine.table_metadata.values():
                    if hasattr(table_info, 'table_name') and table_info.table_name == table_name:
                        self.table_info_cache[table_name] = table_info
                        return table_info
            elif hasattr(vector_search_engine, 'table_metadata'):
                # HNSW搜索引擎
                for table_info in vector_search_engine.table_metadata.values():
                    if table_info.get('table_name') == table_name:
                        # 构建TableInfo对象
                        columns = []
                        for i, col_name in enumerate(table_info.get('column_names', [])):
                            column = ColumnInfo(
                                table_name=table_name,
                                column_name=col_name,
                                data_type=table_info.get('data_types', [None])[i] if i < len(table_info.get('data_types', [])) else None,
                                sample_values=[]
                            )
                            columns.append(column)
                        
                        table_obj = TableInfo(
                            table_name=table_name,
                            columns=columns,
                            row_count=table_info.get('row_count', 0)
                        )
                        
                        self.table_info_cache[table_name] = table_obj
                        return table_obj
            
            logger.warning(f"表信息未找到: {table_name}")
            return None
            
        except Exception as e:
            logger.error(f"获取表信息失败 {table_name}: {e}")
            return None
    
    async def precise_matching(self, query_table: TableInfo, candidate_tables: List[TableInfo]) -> List[Dict[str, Any]]:
        """精确匹配接口（按照架构升级计划的接口设计）"""
        try:
            # 第一层: 快速预筛选
            candidates = await self._prefilter_candidates(candidate_tables)
            
            # 准备数据
            query_columns = query_table.columns
            query_embeddings = await self._get_column_embeddings(query_columns)
            
            candidate_tables_data = []
            for candidate in candidates:
                candidate_embeddings = await self._get_column_embeddings(candidate.columns)
                if candidate_embeddings:
                    candidate_tables_data.append((candidate.columns, candidate_embeddings))
            
            # 第二层: 匈牙利算法精确匹配
            return self.hungarian_matcher.batch_match_tables(
                query_table=(query_columns, query_embeddings),
                candidate_tables=candidate_tables_data,
                k=10
            )
            
        except Exception as e:
            logger.error(f"精确匹配失败: {e}")
            return []
    
    async def _prefilter_candidates(self, candidate_tables: List[TableInfo]) -> List[TableInfo]:
        """快速预筛选候选表"""
        # 简单的预筛选逻辑：过滤掉列数过少或过多的表
        max_table_size = settings.hungarian_matcher.max_table_size
        
        filtered = []
        for table in candidate_tables:
            if 1 <= len(table.columns) <= max_table_size:
                filtered.append(table)
        
        return filtered
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """获取性能统计"""
        return {
            "stats": self.stats.copy(),
            "hungarian_config": {
                "enabled": settings.hungarian_matcher.enabled,
                "threshold": settings.hungarian_matcher.similarity_threshold,
                "batch_size": settings.hungarian_matcher.batch_size,
                "max_table_size": settings.hungarian_matcher.max_table_size
            }
        }