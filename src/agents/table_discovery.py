from typing import List, Dict, Any
import logging
import asyncio
from src.agents.base import BaseAgent
from src.core.models import AgentState, TableInfo, VectorSearchResult
from src.tools.vector_search import get_vector_search_engine
from src.tools.embedding import get_embedding_generator
from src.config.prompts import format_prompt
from src.config.settings import settings

logger = logging.getLogger(__name__)


class TableDiscoveryAgent(BaseAgent):
    """表发现智能体 - 给定查询表，返回数据湖中语义相似的表"""
    
    def __init__(self):
        super().__init__("TableDiscoveryAgent")
        self.vector_search = get_vector_search_engine()
        self.embedding_gen = get_embedding_generator()
        # 移除立即索引加载 - 改为延迟加载以避免重复加载问题
    
    async def process(self, state: AgentState) -> AgentState:
        """处理表发现任务"""
        # 确保状态对象正确
        state = self._ensure_agent_state(state)
        self.log_progress(state, "开始表发现处理")
        
        # 如果没有查询表但有用户查询，尝试基于查询文本进行发现
        if not state.query_tables and state.user_query:
            self.log_progress(state, "基于用户查询进行表发现")
            candidates = await self.discover_by_keywords([state.user_query])
            state.table_candidates = candidates
            self.log_progress(state, f"基于关键词发现了 {len(candidates)} 个候选表")
            state.current_step = "table_matching"
            return state
            
        if not state.query_tables:
            self.log_error(state, "没有查询表信息")
            return state
        
        try:
            # 并行处理所有查询表
            tasks = []
            for query_table in state.query_tables:
                task = self._discover_similar_tables(query_table)
                tasks.append(task)
            
            all_candidates = []
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    self.log_error(state, f"查询表 {state.query_tables[i].table_name} 处理失败: {result}")
                else:
                    all_candidates.extend(result)
            
            # 去重并排序候选表
            unique_candidates = self._deduplicate_candidates(all_candidates)
            sorted_candidates = sorted(unique_candidates, key=lambda x: x.score, reverse=True)
            
            # 限制候选表数量
            max_candidates = settings.thresholds.max_candidates
            final_candidates = sorted_candidates[:max_candidates]
            
            # 保存候选表名到状态
            state.table_candidates = [candidate.item_id for candidate in final_candidates]
            
            self.log_progress(state, f"表发现完成，找到 {len(state.table_candidates)} 个候选表")
            
            # 更新处理步骤
            state.current_step = "table_matching"
            
        except Exception as e:
            self.log_error(state, f"表发现处理失败: {e}")
        
        return state
    
    async def _discover_similar_tables(self, query_table: TableInfo) -> List[VectorSearchResult]:
        """为单个查询表发现相似表"""
        try:
            # 确保索引已加载
            if not await self._ensure_index_loaded():
                logger.error("向量索引加载失败，无法进行表发现")
                return []
            
            logger.info(f"处理查询表: {query_table.table_name}")
            
            # 生成查询表的嵌入向量
            query_embedding = await self.embedding_gen.generate_table_embedding(query_table)
            
            # 搜索相似表
            similar_tables = await self.vector_search.search_similar_tables(
                query_embedding=query_embedding,
                k=settings.thresholds.max_candidates,
                threshold=settings.thresholds.semantic_similarity_threshold
            )
            
            # 使用LLM进行语义筛选（可选）
            if len(similar_tables) > 10:
                similar_tables = await self._llm_filter_tables(query_table, similar_tables)
            
            logger.debug(f"表 {query_table.table_name} 找到 {len(similar_tables)} 个相似表")
            return similar_tables
            
        except Exception as e:
            logger.error(f"表 {query_table.table_name} 发现失败: {e}")
            return []
    
    async def _llm_filter_tables(
        self, 
        query_table: TableInfo, 
        candidate_tables: List[VectorSearchResult]
    ) -> List[VectorSearchResult]:
        """使用LLM过滤和排序候选表"""
        try:
            # 准备候选表信息
            candidates_info = self._format_candidates_for_llm(candidate_tables[:15])  # 只评估前15个
            
            # 构建查询表的样本数据
            sample_data = self._extract_sample_data(query_table)
            
            prompt = format_prompt(
                "table_discovery",
                table_name=query_table.table_name,
                columns=[col.column_name for col in query_table.columns],
                sample_data=sample_data,
                candidates=candidates_info
            )
            
            # 调用LLM
            response = await self.call_llm(prompt)
            
            # 解析LLM响应，提取推荐的表名
            recommended_tables = self._parse_llm_recommendations(response)
            
            # 根据LLM推荐重新排序
            filtered_results = []
            for table_name in recommended_tables:
                for candidate in candidate_tables:
                    if candidate.item_id == table_name:
                        # 提升LLM推荐表的分数
                        candidate.score = min(candidate.score * 1.2, 1.0)
                        filtered_results.append(candidate)
                        break
            
            # 添加未被LLM推荐但分数较高的表
            for candidate in candidate_tables:
                if candidate.item_id not in recommended_tables and candidate.score >= 0.8:
                    filtered_results.append(candidate)
            
            # 去重并排序
            seen = set()
            unique_results = []
            for candidate in filtered_results:
                if candidate.item_id not in seen:
                    seen.add(candidate.item_id)
                    unique_results.append(candidate)
            
            unique_results.sort(key=lambda x: x.score, reverse=True)
            
            logger.debug(f"LLM筛选后保留 {len(unique_results)} 个候选表")
            return unique_results[:10]  # 返回前10个
            
        except Exception as e:
            logger.error(f"LLM表筛选失败: {e}")
            return candidate_tables[:10]  # 返回原始结果的前10个
    
    def _format_candidates_for_llm(self, candidates: List[VectorSearchResult]) -> str:
        """格式化候选表信息供LLM使用"""
        formatted_parts = []
        
        for i, candidate in enumerate(candidates, 1):
            metadata = candidate.metadata
            table_name = metadata.get("table_name", candidate.item_id)
            columns = metadata.get("columns", [])
            row_count = metadata.get("row_count", "未知")
            
            formatted_parts.append(
                f"{i}. 表名: {table_name}\n"
                f"   列数: {len(columns)}, 行数: {row_count}\n"
                f"   相似度: {candidate.score:.3f}\n"
                f"   列名: {', '.join(columns[:8])}{'...' if len(columns) > 8 else ''}"
            )
        
        return "\n\n".join(formatted_parts)
    
    def _extract_sample_data(self, table_info: TableInfo) -> str:
        """提取表的样本数据描述"""
        sample_parts = []
        
        for col in table_info.columns[:5]:  # 只使用前5列
            if col.sample_values:
                samples = [str(v) for v in col.sample_values[:3] if v is not None]
                if samples:
                    sample_parts.append(f"{col.column_name}: {', '.join(samples)}")
        
        return "; ".join(sample_parts) if sample_parts else "无样本数据"
    
    def _parse_llm_recommendations(self, llm_response: str) -> List[str]:
        """解析LLM推荐的表名"""
        try:
            # 简单的解析逻辑，查找表名模式
            import re
            
            # 查找表名模式（假设表名不包含空格）
            table_patterns = [
                r'表名?[:：]\s*([a-zA-Z_][a-zA-Z0-9_]*)',  # 表名: table_name
                r'([a-zA-Z_][a-zA-Z0-9_]*(?:\.csv)?)',     # 直接的表名
            ]
            
            recommended_tables = []
            for pattern in table_patterns:
                matches = re.findall(pattern, llm_response, re.IGNORECASE)
                for match in matches:
                    table_name = match.strip()
                    if table_name and table_name not in recommended_tables:
                        recommended_tables.append(table_name)
            
            # 限制推荐数量
            return recommended_tables[:10]
            
        except Exception as e:
            logger.error(f"解析LLM推荐失败: {e}")
            return []
    
    def _deduplicate_candidates(self, candidates: List[VectorSearchResult]) -> List[VectorSearchResult]:
        """去重候选表结果"""
        seen = {}
        unique_candidates = []
        
        for candidate in candidates:
            table_id = candidate.item_id
            
            if table_id not in seen or candidate.score > seen[table_id].score:
                seen[table_id] = candidate
        
        return list(seen.values())
    
    async def initialize_table_index(self, tables_data: List[TableInfo]) -> None:
        """初始化表级别的搜索索引"""
        try:
            logger.info("开始初始化表搜索索引")
            
            # 先尝试加载现有索引
            index_path = settings.vector_db.db_path
            await self.vector_search.load_index(index_path)
            
            # 批量处理表
            batch_size = settings.performance.batch_size
            for i in range(0, len(tables_data), batch_size):
                batch_tables = tables_data[i:i + batch_size]
                tasks = [self._add_table_to_index(table_info) for table_info in batch_tables]
                
                await asyncio.gather(*tasks, return_exceptions=True)
                logger.info(f"已处理 {min(i + batch_size, len(tables_data))}/{len(tables_data)} 表")
            
            # 保存索引
            await self.vector_search.save_index(index_path)
            
            logger.info(f"表索引初始化完成，处理了 {len(tables_data)} 个表")
            
        except Exception as e:
            logger.error(f"表索引初始化失败: {e}")
            raise
    
    async def _add_table_to_index(self, table_info: TableInfo) -> None:
        """将表添加到索引"""
        try:
            # 生成表的嵌入向量
            embedding = await self.embedding_gen.generate_table_embedding(table_info)
            
            # 添加到向量索引
            await self.vector_search.add_table_vector(table_info, embedding)
            
        except Exception as e:
            logger.error(f"添加表 {table_info.table_name} 到索引失败: {e}")
    
    def get_discovery_statistics(self, state: AgentState) -> Dict[str, Any]:
        """获取发现统计信息"""
        return {
            "query_tables_count": len(state.query_tables),
            "discovered_candidates_count": len(state.table_candidates),
            "candidates": state.table_candidates[:10] if state.table_candidates else []
        }
    
    async def discover_by_keywords(self, keywords: List[str], k: int = 10) -> List[str]:
        """根据关键词发现相关表（辅助功能）"""
        try:
            # 确保索引已加载
            if not await self._ensure_index_loaded():
                logger.error("向量索引加载失败，无法进行关键词发现")
                return []
            
            # 将关键词组合成查询文本
            query_text = " ".join(keywords)
            
            # 生成查询嵌入
            query_embedding = await self.embedding_gen.generate_text_embedding(query_text)
            
            # 搜索相似表
            results = await self.vector_search.search_similar_tables(
                query_embedding=query_embedding,
                k=k,
                threshold=0.5  # 较低的阈值用于关键词搜索
            )
            
            return [result.item_id for result in results]
            
        except Exception as e:
            logger.error(f"关键词表发现失败: {e}")
            return []
    
    async def _ensure_index_loaded(self):
        """简化的索引加载确保方法 - 异步版本"""
        try:
            # 简单检查：尝试获取索引统计信息
            if hasattr(self.vector_search, 'get_collection_stats'):
                stats = self.vector_search.get_collection_stats()
                # 兼容不同的向量搜索引擎
                tables_count = 0
                if 'tables' in stats:
                    tables_count = stats.get('tables', {}).get('count', 0)
                elif 'hnsw_index' in stats:
                    tables_count = stats.get('hnsw_index', {}).get('total_elements', 0)
                
                if tables_count > 0:
                    logger.debug(f"向量搜索索引已加载，包含 {tables_count} 个元素")
                    return True
            
            # 如果索引未加载，尝试加载
            from src.config.settings import settings
            import os
            logger.info("开始加载向量搜索索引...")
            
            # 根据向量搜索引擎类型确定索引文件路径
            if hasattr(self.vector_search, '__class__') and 'HNSW' in self.vector_search.__class__.__name__:
                # HNSW需要完整的文件路径
                index_path = os.path.join(settings.vector_db.db_path, 'hnsw_index.bin')
            else:
                # 其他类型使用目录路径
                index_path = settings.vector_db.db_path
            
            await self.vector_search.load_index(index_path)
            
            # 验证加载结果
            if hasattr(self.vector_search, 'get_collection_stats'):
                stats = self.vector_search.get_collection_stats()
                # 兼容不同的向量搜索引擎
                tables_count = 0
                if 'tables' in stats:
                    tables_count = stats.get('tables', {}).get('count', 0)
                elif 'hnsw_index' in stats:
                    tables_count = stats.get('hnsw_index', {}).get('total_elements', 0)
                logger.info(f"索引加载完成，包含 {tables_count} 个元素")
                return tables_count > 0
            return False
            
        except Exception as e:
            logger.error(f"索引加载失败: {e}")
            return False