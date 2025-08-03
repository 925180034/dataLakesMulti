"""
优化的数据湖工作流 - 集成三层加速架构
提供10,000+表规模下的高性能数据发现
"""

import logging
import asyncio
import time
from typing import Dict, Any, List, Tuple, Optional
from langgraph.graph import StateGraph, END

from src.core.models import AgentState, TableInfo, ColumnInfo
from src.core.workflow import DataLakesWorkflow
from src.tools.metadata_filter import MetadataFilter
from src.tools.batch_vector_search import BatchVectorSearch
from src.tools.smart_llm_matcher import SmartLLMMatcher
from src.tools.multi_level_cache import CacheManager
from src.tools.vector_search import get_vector_search_engine
from src.config.settings import settings

logger = logging.getLogger(__name__)


class OptimizedDataLakesWorkflow(DataLakesWorkflow):
    """优化的数据湖工作流
    
    核心优化：
    1. 三层加速架构
    2. 批量处理
    3. 多级缓存
    4. 智能LLM调用
    """
    
    def __init__(self):
        super().__init__()
        
        # 初始化基础组件
        from src.tools.vector_search import get_vector_search_engine
        from src.config.settings import settings
        from src.utils.llm_client import create_llm_client
        
        self.vector_search = get_vector_search_engine()
        self.llm_client = create_llm_client()
        
        # 初始化优化组件
        self.metadata_filter = MetadataFilter()
        self.batch_vector_search = BatchVectorSearch(self.vector_search)
        self.smart_matcher = SmartLLMMatcher(self.llm_client)
        
        # 初始化缓存
        cache_config = getattr(settings.cache, 'multi_level_cache', {})
        self.cache_manager = CacheManager(cache_config)
        
        # 预加载的表元数据
        self.table_metadata_cache = {}
        
        # 性能统计
        self.performance_stats = {
            "metadata_filter_time": 0,
            "vector_search_time": 0,
            "llm_match_time": 0,
            "total_time": 0,
            "tables_processed": 0,
            "llm_calls": 0
        }
    
    async def initialize(self, all_tables: List[TableInfo]) -> None:
        """初始化工作流（构建索引等）
        
        Args:
            all_tables: 数据湖中的所有表
        """
        logger.info(f"开始初始化优化工作流，表数量: {len(all_tables)}")
        
        # 1. 构建元数据索引
        self.metadata_filter.build_index(all_tables)
        
        # 2. 缓存表元数据
        for table in all_tables:
            self.table_metadata_cache[table.table_name] = table
        
        # 3. 构建HNSW向量索引
        logger.info("开始构建HNSW向量索引...")
        from src.tools.embedding import get_embedding_generator
        embedding_gen = get_embedding_generator()
        
        # 批量生成表的向量
        table_texts = [self._table_to_text(table) for table in all_tables]
        
        # 批量生成向量
        embeddings = []
        if hasattr(embedding_gen, 'model') and embedding_gen.model is not None:
            # 确保模型已初始化
            if not embedding_gen._model_initialized:
                embedding_gen._initialize_model()
            
            # 使用SentenceTransformer的批量编码
            if embedding_gen.model is not None:
                batch_embeddings = embedding_gen.model.encode(table_texts, convert_to_numpy=True)
                embeddings = [emb.tolist() for emb in batch_embeddings]
            else:
                # 备用方案：逐个生成
                for text in table_texts:
                    emb = await embedding_gen.generate_text_embedding(text)
                    embeddings.append(emb)
        else:
            # 备用方案：逐个生成
            for text in table_texts:
                emb = await embedding_gen.generate_text_embedding(text)
                embeddings.append(emb)
        
        # 添加到HNSW索引
        try:
            for table, embedding in zip(all_tables, embeddings):
                await self.vector_search.add_table_vector(table, embedding)
            logger.info(f"HNSW索引构建完成，已添加 {len(all_tables)} 个表")
            
            # 如果是HNSW引擎，显示元数据统计
            if hasattr(self.vector_search, 'table_metadata'):
                logger.info(f"HNSW表元数据数量: {len(self.vector_search.table_metadata)}")
        except Exception as e:
            logger.error(f"向量索引构建失败: {e}")
            logger.warning("将继续使用元数据筛选，但无向量搜索功能")
        
        # 4. 预热常用缓存
        await self._warm_up_cache(all_tables[:100])  # 预热前100个表
        
        logger.info("优化工作流初始化完成")
    
    async def run_optimized(
        self,
        initial_state: AgentState,
        all_table_names: List[str]
    ) -> AgentState:
        """运行优化的工作流
        
        Args:
            initial_state: 初始状态
            all_table_names: 数据湖中所有表名
        """
        start_time = time.time()
        
        try:
            # 验证输入
            self._validate_initial_state(initial_state)
            
            # 如果没有设置策略，自动决定策略
            if initial_state.strategy is None:
                await self._decide_strategy(initial_state)
            
            # 根据策略选择处理路径 (修复：使用正确的枚举值)
            from src.core.models import TaskStrategy
            if initial_state.strategy == TaskStrategy.BOTTOM_UP:
                result = await self._run_bottom_up_optimized(initial_state, all_table_names)
            else:  # TaskStrategy.TOP_DOWN
                result = await self._run_top_down_optimized(initial_state, all_table_names)
            
            # 更新性能统计
            self.performance_stats["total_time"] = time.time() - start_time
            self.performance_stats["tables_processed"] = len(all_table_names)
            
            # 记录性能日志
            self._log_performance_stats()
            
            return result
            
        except Exception as e:
            import traceback
            logger.error(f"优化工作流执行失败: {e}")
            logger.error(f"错误详情: {traceback.format_exc()}")
            initial_state.add_error(f"优化工作流执行失败: {e}")
            return initial_state
    
    async def _run_top_down_optimized(
        self,
        state: AgentState,
        all_table_names: List[str]
    ) -> AgentState:
        """优化的Top-Down策略执行 - 并行处理版本"""
        logger.info("执行优化的Top-Down策略（并行版本）")
        
        # 并行执行第一层和预处理
        metadata_task = self._metadata_filtering(state.query_tables, all_table_names)
        
        # 等待元数据筛选完成（这是必须的，因为后续步骤依赖结果）
        metadata_start = time.time()
        filtered_candidates = await metadata_task
        self.performance_stats["metadata_filter_time"] = time.time() - metadata_start
        
        logger.info(f"元数据筛选完成: {len(all_table_names)} -> {len(filtered_candidates)}")
        
        # 并行执行第二层和第三层的准备工作
        vector_start = time.time()
        
        # 批量向量搜索（异步）
        vector_task = self._batch_vector_search(
            state.query_tables,
            [name for name, _ in filtered_candidates[:200]]  # 限制候选数量以提高速度
        )
        
        # 等待向量搜索完成
        vector_results = await vector_task
        self.performance_stats["vector_search_time"] = time.time() - vector_start
        
        # 智能LLM匹配（使用批量处理）
        llm_start = time.time()
        final_matches = await self._smart_llm_matching(
            state.query_tables,
            vector_results
        )
        self.performance_stats["llm_match_time"] = time.time() - llm_start
        
        # 更新状态 (修复：同时设置table_matches和final_results)
        formatted_results = self._format_final_results(final_matches)
        state.final_results = formatted_results
        state.table_matches = formatted_results  # 确保兼容性
        state.current_step = "completed"
        
        logger.info(f"策略完成，找到 {len(formatted_results)} 个匹配")
        return state
    
    async def _run_bottom_up_optimized(
        self,
        state: AgentState,
        all_table_names: List[str]
    ) -> AgentState:
        """优化的Bottom-Up策略执行"""
        logger.info("执行优化的Bottom-Up策略")
        
        # 并行搜索所有查询列
        column_results = await self.batch_vector_search.parallel_search_columns(
            [self._column_to_dict(col) for col in state.query_columns]
        )
        
        # 聚合到表级别
        table_scores = self._aggregate_column_matches(column_results)
        
        # 获取Top候选表
        top_tables = sorted(table_scores.items(), key=lambda x: x[1], reverse=True)[:100]
        
        # 使用智能匹配器验证
        candidate_dict = {
            "query": [(name, score) for name, score in top_tables]
        }
        
        final_matches = await self.smart_matcher.match_tables(
            [self._create_pseudo_query_table(state.query_columns)],
            candidate_dict,
            self.table_metadata_cache
        )
        
        # 更新状态 (修复：同时设置table_matches和final_results)
        formatted_results = self._format_final_results(final_matches)
        state.final_results = formatted_results
        state.table_matches = formatted_results  # 确保兼容性
        state.current_step = "completed"
        
        logger.info(f"策略完成，找到 {len(formatted_results)} 个匹配")
        return state
    
    async def _metadata_filtering(
        self,
        query_tables: List[TableInfo],
        all_table_names: List[str]
    ) -> List[Tuple[str, float]]:
        """第一层：元数据预筛选"""
        all_filtered = []
        
        for query_table in query_tables:
            # 使用缓存
            cache_key = f"metadata:{query_table.table_name}"
            
            # Create an async wrapper for the synchronous filter_candidates method
            async def compute_metadata():
                return self.metadata_filter.filter_candidates(
                    query_table,
                    all_table_names,
                    top_k=1000
                )
            
            filtered = await self.cache_manager.get_or_compute(
                namespace="metadata_filter",
                key=cache_key,
                compute_func=compute_metadata,
                ttl=3600
            )
            
            all_filtered.extend(filtered)
        
        # 去重并返回
        unique_tables = {}
        for table_name, score in all_filtered:
            if table_name not in unique_tables or score > unique_tables[table_name]:
                unique_tables[table_name] = score
        
        return list(unique_tables.items())
    
    async def _batch_vector_search(
        self,
        query_tables: List[TableInfo],
        candidate_table_names: List[str]
    ) -> Dict[str, List[Tuple[str, float]]]:
        """第二层：批量向量搜索"""
        # 使用批量搜索优化器
        results = await self.batch_vector_search.batch_search_tables(
            query_tables,
            candidate_table_names,
            k=100,
            threshold=0.6
        )
        
        # 转换结果格式
        formatted_results = {}
        for query_name, search_results in results.items():
            formatted_results[query_name] = [
                (result.item_id, result.score)
                for result in search_results
            ]
        
        return formatted_results
    
    async def _smart_llm_matching(
        self,
        query_tables: List[TableInfo],
        vector_results: Dict[str, List[Tuple[str, float]]]
    ) -> Dict[str, List[Dict[str, Any]]]:
        """第三层：智能LLM匹配"""
        # 使用智能匹配器
        matches = await self.smart_matcher.match_tables(
            query_tables,
            vector_results,
            self.table_metadata_cache
        )
        
        # 更新LLM调用统计
        self.performance_stats["llm_calls"] = len(
            [m for matches_list in matches.values() 
             for m in matches_list if m.get("method") == "llm"]
        )
        
        return matches
    
    def _aggregate_column_matches(
        self,
        column_results: Dict[str, List[Dict[str, Any]]]
    ) -> Dict[str, float]:
        """聚合列匹配结果到表级别"""
        table_scores = {}
        
        for table_name, col_matches in column_results.items():
            for match_info in col_matches:
                for match in match_info.get("matches", []):
                    matched_table = match.item_id.split(".")[0]  # 提取表名
                    if matched_table not in table_scores:
                        table_scores[matched_table] = 0
                    table_scores[matched_table] += match.score
        
        # 归一化分数
        if table_scores:
            max_score = max(table_scores.values())
            for table in table_scores:
                table_scores[table] /= max_score
        
        return table_scores
    
    def _format_final_results(
        self,
        matches: Dict[str, List[Dict[str, Any]]]
    ) -> List['TableMatchResult']:
        """格式化最终结果"""
        from src.core.models import TableMatchResult, MatchResult
        
        all_results = []
        
        for query_table, table_matches in matches.items():
            for match in table_matches:
                result = TableMatchResult(
                    source_table=query_table,
                    target_table=match["table"],
                    score=match["score"] * 100,  # 转换为0-100分数
                    matched_columns=[],  # 可以在后续添加列匹配信息
                    evidence={
                        "match_type": match.get("match_type", "unknown"),
                        "reason": match.get("reason", ""),
                        "method": match.get("method", "unknown")
                    }
                )
                all_results.append(result)
        
        return all_results
    
    def _create_pseudo_query_table(self, columns: List[ColumnInfo]) -> TableInfo:
        """从列创建伪查询表"""
        if not columns:
            return TableInfo(table_name="query", columns=[])
        
        # 假设所有列来自同一个表
        table_name = columns[0].table_name if columns[0].table_name else "query"
        
        return TableInfo(
            table_name=table_name,
            columns=columns
        )
    
    def _column_to_dict(self, column: ColumnInfo) -> Dict[str, Any]:
        """将ColumnInfo转换为字典"""
        return {
            "table_name": column.table_name,
            "column_name": column.column_name,
            "data_type": column.data_type,
            "sample_values": column.sample_values
        }
    
    def _table_to_text(self, table: TableInfo) -> str:
        """将表信息转换为文本用于向量化"""
        # 构建表的文本表示
        parts = [f"Table: {table.table_name}"]
        
        # 添加列信息
        for col in table.columns:
            col_text = f"Column {col.column_name} ({col.data_type})"
            if col.sample_values:
                col_text += f" samples: {', '.join(str(v) for v in col.sample_values[:3])}"
            parts.append(col_text)
        
        # 添加表描述（如果有）
        if table.description:
            parts.append(f"Description: {table.description}")
        
        return " | ".join(parts)
    
    async def _decide_strategy(self, state: AgentState) -> None:
        """决定处理策略（与基础工作流保持一致）"""
        from src.core.models import TaskStrategy
        
        # 基于输入数据类型的策略决策
        has_tables = len(state.query_tables) > 0
        has_columns = len(state.query_columns) > 0
        
        if has_columns and not has_tables:
            # 只有列数据，使用BOTTOM_UP
            state.strategy = TaskStrategy.BOTTOM_UP
            logger.info("检测到列数据，选择Bottom-Up策略")
        elif has_tables and not has_columns:
            # 只有表数据，使用TOP_DOWN
            state.strategy = TaskStrategy.TOP_DOWN
            logger.info("检测到表数据，选择Top-Down策略")
        else:
            # 默认使用TOP_DOWN策略（对于表匹配更高效）
            state.strategy = TaskStrategy.TOP_DOWN
            logger.info("使用默认Top-Down策略")
    
    async def _warm_up_cache(self, tables: List[TableInfo]) -> None:
        """预热缓存"""
        logger.info(f"开始预热缓存，表数量: {len(tables)}")
        
        # 预热元数据筛选缓存
        keys = [f"metadata:{table.table_name}" for table in tables]
        
        async def generate_metadata_result(key: str) -> Any:
            table_name = key.split(":")[-1]
            table = next((t for t in tables if t.table_name == table_name), None)
            if table:
                # filter_candidates returns a plain list, no need to await
                result = self.metadata_filter.filter_candidates(
                    table,
                    [t.table_name for t in tables],
                    top_k=100
                )
                return result
            return None
        
        await self.cache_manager.cache.warm_up(
            "metadata_filter",
            keys[:50],  # 预热前50个
            generate_metadata_result
        )
        
        logger.info("缓存预热完成")
    
    def _log_performance_stats(self) -> None:
        """记录性能统计"""
        stats = self.performance_stats
        logger.info(
            f"性能统计 - "
            f"总时间: {stats['total_time']:.2f}s, "
            f"元数据筛选: {stats['metadata_filter_time']:.2f}s, "
            f"向量搜索: {stats['vector_search_time']:.2f}s, "
            f"LLM匹配: {stats['llm_match_time']:.2f}s, "
            f"处理表数: {stats['tables_processed']}, "
            f"LLM调用: {stats['llm_calls']}"
        )
    
    def get_performance_report(self) -> Dict[str, Any]:
        """获取性能报告"""
        return {
            "performance_stats": self.performance_stats,
            "cache_stats": self.cache_manager.cache.get_stats(),
            "optimization_summary": {
                "metadata_reduction": f"{self.performance_stats.get('metadata_reduction', 0):.1%}",
                "vector_reduction": f"{self.performance_stats.get('vector_reduction', 0):.1%}",
                "llm_reduction": f"{self.performance_stats.get('llm_reduction', 0):.1%}",
                "total_speedup": f"{self.performance_stats.get('speedup', 1):.1f}x"
            }
        }


# 工厂函数
def create_optimized_workflow() -> OptimizedDataLakesWorkflow:
    """创建优化的工作流实例"""
    return OptimizedDataLakesWorkflow()