from typing import List, Dict, Any
import logging
import asyncio
from src.agents.base import BaseAgent
from src.core.models import (
    AgentState, ColumnInfo, MatchResult, 
    VectorSearchResult, ValueSearchResult
)
from src.tools.vector_search import vector_search_engine
from src.tools.value_search import value_search_engine
from src.tools.embedding import get_embedding_generator
from src.tools.hybrid_similarity import hybrid_similarity_engine
from src.config.prompts import format_prompt
from src.config.settings import settings

logger = logging.getLogger(__name__)


class ColumnDiscoveryAgent(BaseAgent):
    """列发现智能体 - 负责给定查询列，在数据湖中找到匹配的列"""
    
    def __init__(self):
        super().__init__("ColumnDiscoveryAgent")
        self.vector_search = vector_search_engine
        self.value_search = value_search_engine
        self.embedding_gen = get_embedding_generator()
        self.hybrid_similarity = hybrid_similarity_engine
    
    async def process(self, state: AgentState) -> AgentState:
        """处理列发现任务"""
        # 确保状态对象正确
        state = self._ensure_agent_state(state)
        self.log_progress(state, "开始列发现处理")
        
        if not state.query_columns:
            self.log_error(state, "没有查询列信息")
            return state
        
        # 并行处理所有查询列
        tasks = []
        for query_column in state.query_columns:
            task = self._discover_matches_for_column(query_column)
            tasks.append(task)
        
        try:
            # 等待所有列的匹配结果
            all_column_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # 处理结果
            for i, result in enumerate(all_column_results):
                if isinstance(result, Exception):
                    self.log_error(state, f"列 {state.query_columns[i].full_name} 处理失败: {result}")
                else:
                    # 添加匹配结果到状态
                    state.column_matches.extend(result)
            
            self.log_progress(state, f"列发现完成，找到 {len(state.column_matches)} 个匹配")
            
            # 更新处理步骤
            state.current_step = "table_aggregation"
            
        except Exception as e:
            self.log_error(state, f"列发现处理失败: {e}")
        
        return state
    
    async def _discover_matches_for_column(self, query_column: ColumnInfo) -> List[MatchResult]:
        """为单个查询列发现匹配"""
        matches = []
        
        try:
            logger.info(f"处理查询列: {query_column.full_name}")
            
            # 并行执行语义搜索和值重叠搜索
            semantic_task = self._semantic_search(query_column)
            value_task = self._value_overlap_search(query_column)
            
            semantic_results, value_results = await asyncio.gather(
                semantic_task, value_task, return_exceptions=True
            )
            
            # 处理语义搜索结果
            if not isinstance(semantic_results, Exception):
                for result in semantic_results:
                    match = MatchResult(
                        source_column=query_column.full_name,
                        target_column=result.item_id,
                        confidence=result.score,
                        reason=f"语义相似度: {result.score:.3f}",
                        match_type="semantic"
                    )
                    matches.append(match)
            
            # 处理值重叠搜索结果
            if not isinstance(value_results, Exception):
                for result in value_results:
                    # 检查是否已有语义匹配结果
                    existing_match = None
                    for match in matches:
                        if match.target_column == result.item_id:
                            existing_match = match
                            break
                    
                    if existing_match:
                        # 合并结果，提高置信度
                        combined_confidence = 0.6 * existing_match.confidence + 0.4 * result.score
                        existing_match.confidence = min(combined_confidence, 1.0)
                        existing_match.reason += f" + 值重叠: {result.overlap_ratio:.3f}"
                        existing_match.match_type = "semantic+value"
                    else:
                        # 新的值重叠匹配
                        match = MatchResult(
                            source_column=query_column.full_name,
                            target_column=result.item_id,
                            confidence=result.score,
                            reason=f"值重叠比例: {result.overlap_ratio:.3f}, 匹配值: {len(result.matched_values)}个",
                            match_type="value_overlap"
                        )
                        matches.append(match)
            
            # 使用混合相似度计算增强匹配结果
            if matches:
                matches = await self._enhance_matches_with_hybrid_similarity(query_column, matches)
            
            # 使用LLM进行最终的匹配评估和筛选
            if matches:
                matches = await self._llm_evaluate_matches(query_column, matches)
            
            # 按置信度排序并过滤
            matches.sort(key=lambda x: x.confidence, reverse=True)
            threshold = settings.thresholds.column_match_confidence_threshold
            matches = [m for m in matches if m.confidence >= threshold]
            
            # 限制返回结果数量
            max_results = settings.thresholds.top_k_results
            matches = matches[:max_results]
            
            logger.debug(f"列 {query_column.full_name} 找到 {len(matches)} 个匹配")
            
        except Exception as e:
            logger.error(f"列 {query_column.full_name} 匹配失败: {e}")
        
        return matches
    
    async def _enhance_matches_with_hybrid_similarity(
        self, 
        query_column: ColumnInfo, 
        candidate_matches: List[MatchResult]
    ) -> List[MatchResult]:
        """使用混合相似度计算增强匹配结果"""
        try:
            enhanced_matches = []
            
            # 确定场景类型
            scenario = "SLD" if query_column.sample_values else "SMD"
            
            for match in candidate_matches:
                # 解析目标列信息（从metadata中获取或重新构建）
                target_column = self._create_target_column_info(match.target_column)
                
                if target_column:
                    # 使用混合相似度计算精确相似度
                    similarity_result = self.hybrid_similarity.calculate_column_similarity(
                        query_column, target_column, scenario
                    )
                    
                    # 创建增强的匹配结果
                    enhanced_confidence = similarity_result['combined_similarity']
                    
                    # 合并原有置信度和新计算的相似度
                    final_confidence = 0.3 * match.confidence + 0.7 * enhanced_confidence
                    
                    enhanced_reason = (
                        f"混合相似度 ({scenario}): {enhanced_confidence:.3f} "
                        f"[名称: {similarity_result['name_similarity']:.3f}, "
                        f"结构: {similarity_result['structural_similarity']:.3f}, "
                        f"语义: {similarity_result['semantic_similarity']:.3f}] | "
                        f"原始: {match.reason}"
                    )
                    
                    enhanced_match = MatchResult(
                        source_column=match.source_column,
                        target_column=match.target_column,
                        confidence=min(final_confidence, 1.0),
                        reason=enhanced_reason,
                        match_type=f"hybrid_{scenario.lower()}"
                    )
                    
                    enhanced_matches.append(enhanced_match)
                    
                    logger.debug(f"增强匹配 {match.target_column}: "
                               f"{match.confidence:.3f} -> {final_confidence:.3f}")
                else:
                    # 如果无法获取目标列信息，保留原匹配
                    enhanced_matches.append(match)
            
            logger.info(f"混合相似度增强完成，处理了 {len(enhanced_matches)} 个匹配")
            return enhanced_matches
            
        except Exception as e:
            logger.error(f"混合相似度增强失败: {e}")
            return candidate_matches
    
    def _create_target_column_info(self, target_column_full_name: str) -> ColumnInfo:
        """根据目标列全名创建ColumnInfo对象"""
        try:
            # 解析表名和列名
            if '.' in target_column_full_name:
                table_name, column_name = target_column_full_name.split('.', 1)
            else:
                table_name = "unknown"
                column_name = target_column_full_name
            
            # 尝试从向量搜索引擎的元数据中获取详细信息
            target_column_info = None
            
            # 遍历已索引的列信息查找匹配
            for col_id, col_info in self.vector_search.column_metadata.items():
                if col_info.full_name == target_column_full_name:
                    target_column_info = col_info
                    break
            
            # 如果找不到完整信息，创建基础信息
            if not target_column_info:
                target_column_info = ColumnInfo(
                    table_name=table_name,
                    column_name=column_name,
                    data_type=None,
                    sample_values=[]
                )
                logger.debug(f"为 {target_column_full_name} 创建了基础列信息")
            
            return target_column_info
            
        except Exception as e:
            logger.error(f"创建目标列信息失败 {target_column_full_name}: {e}")
            return None
    
    async def _semantic_search(self, query_column: ColumnInfo) -> List[VectorSearchResult]:
        """执行语义搜索"""
        try:
            # 生成查询列的嵌入向量
            query_embedding = await self.embedding_gen.generate_column_embedding(query_column)
            
            # 搜索相似列
            results = await self.vector_search.search_similar_columns(
                query_embedding=query_embedding,
                k=settings.thresholds.max_candidates,
                threshold=settings.thresholds.semantic_similarity_threshold
            )
            
            logger.debug(f"语义搜索返回 {len(results)} 个结果")
            return results
            
        except Exception as e:
            logger.error(f"语义搜索失败: {e}")
            return []
    
    async def _value_overlap_search(self, query_column: ColumnInfo) -> List[ValueSearchResult]:
        """执行值重叠搜索"""
        try:
            if not query_column.sample_values:
                logger.debug(f"列 {query_column.full_name} 没有样本值，跳过值重叠搜索")
                return []
            
            # 搜索值重叠的列
            results = await self.value_search.search_by_values(
                query_values=query_column.sample_values,
                min_overlap_ratio=settings.thresholds.value_overlap_threshold,
                k=settings.thresholds.max_candidates
            )
            
            logger.debug(f"值重叠搜索返回 {len(results)} 个结果")
            return results
            
        except Exception as e:
            logger.error(f"值重叠搜索失败: {e}")
            return []
    
    async def _llm_evaluate_matches(
        self, 
        query_column: ColumnInfo, 
        candidate_matches: List[MatchResult]
    ) -> List[MatchResult]:
        """使用LLM评估和筛选匹配结果"""
        try:
            if len(candidate_matches) <= 5:
                # 候选较少时直接返回
                return candidate_matches
            
            # 构建候选信息
            candidates_info = []
            for match in candidate_matches[:10]:  # 只评估前10个候选
                candidates_info.append({
                    "column": match.target_column,
                    "confidence": match.confidence,
                    "reason": match.reason,
                    "match_type": match.match_type
                })
            
            # 构建评估prompt
            prompt = format_prompt(
                "column_discovery",
                table_name=query_column.table_name,
                column_name=query_column.column_name,
                data_type=query_column.data_type or "未知",
                sample_values=str(query_column.sample_values[:5]),
                candidates=self._format_candidates_for_llm(candidates_info)
            )
            
            # 调用LLM进行评估
            response = await self.call_llm_json(prompt)
            
            # 解析LLM响应
            if "matches" in response:
                llm_matches = response["matches"]
                
                # 更新匹配结果
                updated_matches = []
                for llm_match in llm_matches:
                    target_column = llm_match.get("target_column", "")
                    llm_confidence = llm_match.get("confidence", 0)
                    llm_reason = llm_match.get("reason", "")
                    
                    # 找到对应的原始匹配
                    for original_match in candidate_matches:
                        if original_match.target_column == target_column:
                            # 合并LLM评估结果
                            combined_confidence = 0.7 * original_match.confidence + 0.3 * llm_confidence
                            updated_match = MatchResult(
                                source_column=original_match.source_column,
                                target_column=original_match.target_column,
                                confidence=min(combined_confidence, 1.0),
                                reason=f"{original_match.reason} | LLM评估: {llm_reason}",
                                match_type=original_match.match_type
                            )
                            updated_matches.append(updated_match)
                            break
                
                if updated_matches:
                    logger.debug(f"LLM评估更新了 {len(updated_matches)} 个匹配")
                    return updated_matches
            
            # 如果LLM评估失败，返回原始结果
            logger.warning("LLM评估失败，使用原始匹配结果")
            return candidate_matches
            
        except Exception as e:
            logger.error(f"LLM匹配评估失败: {e}")
            return candidate_matches
    
    def _format_candidates_for_llm(self, candidates_info: List[Dict]) -> str:
        """格式化候选信息供LLM使用"""
        formatted = []
        for i, candidate in enumerate(candidates_info, 1):
            formatted.append(
                f"{i}. {candidate['column']} "
                f"(置信度: {candidate['confidence']:.3f}, "
                f"类型: {candidate['match_type']}, "
                f"原因: {candidate['reason']})"
            )
        return "\n".join(formatted)
    
    async def initialize_indices(self, columns_data: List[ColumnInfo]) -> None:
        """初始化搜索索引（用于系统启动时）"""
        try:
            logger.info("开始初始化列搜索索引")
            
            # 并行生成嵌入向量和添加值索引
            tasks = []
            for column_info in columns_data:
                task = self._add_column_to_indices(column_info)
                tasks.append(task)
            
            # 批量处理
            batch_size = settings.performance.batch_size
            for i in range(0, len(tasks), batch_size):
                batch_tasks = tasks[i:i + batch_size]
                await asyncio.gather(*batch_tasks, return_exceptions=True)
                
                logger.info(f"已处理 {min(i + batch_size, len(tasks))}/{len(tasks)} 列")
            
            logger.info(f"索引初始化完成，处理了 {len(columns_data)} 列")
            
        except Exception as e:
            logger.error(f"索引初始化失败: {e}")
            raise
    
    async def _add_column_to_indices(self, column_info: ColumnInfo) -> None:
        """将列添加到搜索索引"""
        try:
            # 生成并添加向量
            embedding = await self.embedding_gen.generate_column_embedding(column_info)
            await self.vector_search.add_column_vector(column_info, embedding)
            
            # 添加值索引
            if column_info.sample_values:
                await self.value_search.add_column_values(column_info)
            
        except Exception as e:
            logger.error(f"添加列 {column_info.full_name} 到索引失败: {e}")
    
    async def save_indices(self, index_path: str) -> None:
        """保存搜索索引"""
        try:
            await self.vector_search.save_index(index_path)
            await self.value_search.save_index(index_path)
            logger.info(f"搜索索引已保存到: {index_path}")
        except Exception as e:
            logger.error(f"保存搜索索引失败: {e}")
            raise
    
    async def load_indices(self, index_path: str) -> None:
        """加载搜索索引"""
        try:
            await self.vector_search.load_index(index_path)
            await self.value_search.load_index(index_path)
            logger.info(f"搜索索引已从 {index_path} 加载")
        except Exception as e:
            logger.error(f"加载搜索索引失败: {e}")
            raise