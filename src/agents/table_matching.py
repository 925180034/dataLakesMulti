from typing import List, Dict, Any, Tuple, Optional
import logging
import asyncio
from itertools import product
from src.agents.base import BaseAgent
from src.core.models import (
    AgentState, TableInfo, ColumnInfo, MatchResult, TableMatchResult
)
from src.tools.embedding import get_embedding_generator
from src.config.prompts import format_prompt
from src.config.settings import settings

logger = logging.getLogger(__name__)


class TableMatchingAgent(BaseAgent):
    """表匹配智能体 - 详细比较两个表，找出所有匹配的列对"""
    
    def __init__(self):
        super().__init__("TableMatchingAgent")
        self.embedding_gen = get_embedding_generator()
        # 缓存表信息以避免重复加载
        self.table_info_cache: Dict[str, TableInfo] = {}
    
    async def process(self, state: AgentState) -> AgentState:
        """处理表匹配任务"""
        # 确保状态对象正确
        state = self._ensure_agent_state(state)
        self.log_progress(state, "开始表对表匹配处理")
        
        if not state.query_tables or not state.table_candidates:
            self.log_error(state, "缺少查询表或候选表信息")
            return state
        
        try:
            # 为每个查询表与每个候选表进行匹配
            matching_tasks = []
            
            for query_table in state.query_tables:
                for candidate_table_name in state.table_candidates:
                    # 获取候选表信息
                    candidate_table = await self._get_table_info(candidate_table_name)
                    if candidate_table:
                        task = self._match_table_pair(query_table, candidate_table)
                        matching_tasks.append(task)
            
            # 并行执行所有匹配任务
            matching_results = await asyncio.gather(*matching_tasks, return_exceptions=True)
            
            # 处理匹配结果
            valid_matches = []
            for result in matching_results:
                if isinstance(result, Exception):
                    self.log_error(state, f"表匹配失败: {result}")
                elif result and result.score > 0:
                    valid_matches.append(result)
            
            # 按分数排序
            valid_matches.sort(key=lambda x: x.score, reverse=True)
            
            # 保存结果
            state.table_matches = valid_matches
            state.final_results = valid_matches[:settings.thresholds.top_k_results]
            
            self.log_progress(state, f"表匹配完成，找到 {len(valid_matches)} 个有效匹配")
            
            # 更新处理步骤
            state.current_step = "finalization"
            
        except Exception as e:
            self.log_error(state, f"表匹配处理失败: {e}")
        
        return state
    
    async def _match_table_pair(
        self, 
        source_table: TableInfo, 
        target_table: TableInfo
    ) -> Optional[TableMatchResult]:
        """匹配一对表"""
        try:
            logger.info(f"匹配表对: {source_table.table_name} -> {target_table.table_name}")
            
            # 获取所有列对的匹配结果
            column_matches = await self._match_all_column_pairs(source_table, target_table)
            
            if not column_matches:
                logger.debug(f"表对 {source_table.table_name} -> {target_table.table_name} 没有列匹配")
                return None
            
            # 计算表级别的整体相似度
            overall_similarity = self._calculate_table_similarity(
                source_table, target_table, column_matches
            )
            
            # 过滤低置信度的列匹配
            threshold = settings.thresholds.column_match_confidence_threshold
            filtered_matches = [m for m in column_matches if m.confidence >= threshold]
            
            if not filtered_matches:
                logger.debug(f"表对 {source_table.table_name} -> {target_table.table_name} 没有高置信度匹配")
                return None
            
            # 构建表匹配结果
            result = TableMatchResult(
                source_table=source_table.table_name,
                target_table=target_table.table_name,
                score=overall_similarity * 100,  # 转换为百分制
                matched_columns=filtered_matches,
                evidence={
                    "total_column_pairs": len(source_table.columns) * len(target_table.columns),
                    "matched_pairs": len(filtered_matches),
                    "avg_match_confidence": sum(m.confidence for m in filtered_matches) / len(filtered_matches),
                    "source_columns": len(source_table.columns),
                    "target_columns": len(target_table.columns),
                    "similarity_score": overall_similarity
                }
            )
            
            logger.debug(f"表对匹配完成: {source_table.table_name} -> {target_table.table_name}, "
                        f"评分: {result.score:.1f}, 匹配列: {len(filtered_matches)}")
            
            return result
            
        except Exception as e:
            logger.error(f"匹配表对失败 {source_table.table_name} -> {target_table.table_name}: {e}")
            return None
    
    async def _match_all_column_pairs(
        self, 
        source_table: TableInfo, 
        target_table: TableInfo
    ) -> List[MatchResult]:
        """匹配两个表的所有列对"""
        try:
            # 限制列数以避免组合爆炸
            max_columns = 20
            source_columns = source_table.columns[:max_columns]
            target_columns = target_table.columns[:max_columns]
            
            # 生成所有列对
            column_pairs = list(product(source_columns, target_columns))
            
            # 并行计算所有列对的相似度
            tasks = []
            for source_col, target_col in column_pairs:
                task = self._calculate_column_similarity(source_col, target_col)
                tasks.append(task)
            
            # 批量处理以控制并发数
            batch_size = settings.performance.batch_size
            all_similarities = []
            
            for i in range(0, len(tasks), batch_size):
                batch_tasks = tasks[i:i + batch_size]
                batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
                
                for result in batch_results:
                    if not isinstance(result, Exception) and result:
                        all_similarities.append(result)
            
            # 使用LLM进行最终评估（如果列对较多）
            if len(all_similarities) > 10:
                all_similarities = await self._llm_evaluate_column_matches(
                    source_table, target_table, all_similarities
                )
            
            # 过滤并排序结果
            valid_matches = [match for match in all_similarities if match.confidence > 0.3]
            valid_matches.sort(key=lambda x: x.confidence, reverse=True)
            
            # 避免重复匹配（一个列只能匹配到最佳候选）
            final_matches = self._deduplicate_matches(valid_matches)
            
            return final_matches
            
        except Exception as e:
            logger.error(f"匹配所有列对失败: {e}")
            return []
    
    async def _calculate_column_similarity(
        self, 
        source_col: ColumnInfo, 
        target_col: ColumnInfo
    ) -> Optional[MatchResult]:
        """计算两个列的相似度"""
        try:
            # 基于多个维度计算相似度
            similarities = {}
            
            # 1. 列名语义相似度
            name_similarity = await self._calculate_name_similarity(
                source_col.column_name, target_col.column_name
            )
            similarities['name'] = name_similarity
            
            # 2. 数据类型相似度
            type_similarity = self._calculate_type_similarity(
                source_col.data_type, target_col.data_type
            )
            similarities['type'] = type_similarity
            
            # 3. 值分布相似度
            if source_col.sample_values and target_col.sample_values:
                value_similarity = self._calculate_value_similarity(
                    source_col.sample_values, target_col.sample_values
                )
                similarities['value'] = value_similarity
            else:
                similarities['value'] = 0.0
            
            # 4. 统计特征相似度
            stats_similarity = self._calculate_stats_similarity(source_col, target_col)
            similarities['stats'] = stats_similarity
            
            # 综合评分（可配置权重）
            weights = {
                'name': 0.4,
                'type': 0.2,
                'value': 0.3,
                'stats': 0.1
            }
            
            overall_confidence = sum(
                similarities[key] * weights[key] 
                for key in similarities
            )
            
            # 生成匹配理由
            reason = self._generate_match_reason(similarities)
            
            # 确定匹配类型
            match_type = "comprehensive"
            if similarities['name'] > 0.8 and similarities['value'] > 0.5:
                match_type = "semantic+value"
            elif similarities['name'] > 0.7:
                match_type = "semantic"
            elif similarities['value'] > 0.6:
                match_type = "value_overlap"
            
            if overall_confidence < 0.1:  # 太低的相似度直接过滤
                return None
            
            return MatchResult(
                source_column=source_col.full_name,
                target_column=target_col.full_name,
                confidence=overall_confidence,
                reason=reason,
                match_type=match_type
            )
            
        except Exception as e:
            logger.error(f"计算列相似度失败 {source_col.full_name} -> {target_col.full_name}: {e}")
            return None
    
    async def _calculate_name_similarity(self, name1: str, name2: str) -> float:
        """计算列名语义相似度"""
        try:
            if not name1 or not name2:
                return 0.0
            
            # 简单的字符串相似度
            if name1.lower() == name2.lower():
                return 1.0
            
            # 使用嵌入向量计算语义相似度
            embedding1 = await self.embedding_gen.generate_text_embedding(name1)
            embedding2 = await self.embedding_gen.generate_text_embedding(name2)
            
            # 计算余弦相似度
            import numpy as np
            vec1 = np.array(embedding1)
            vec2 = np.array(embedding2)
            
            similarity = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
            return max(0.0, float(similarity))
            
        except Exception as e:
            logger.error(f"计算列名相似度失败: {e}")
            return 0.0
    
    def _calculate_type_similarity(self, type1: Optional[str], type2: Optional[str]) -> float:
        """计算数据类型相似度"""
        if not type1 or not type2:
            return 0.0
        
        type1_clean = type1.lower().strip()
        type2_clean = type2.lower().strip()
        
        if type1_clean == type2_clean:
            return 1.0
        
        # 类型映射表
        type_groups = {
            'integer': ['int', 'integer', 'bigint', 'smallint', 'tinyint'],
            'float': ['float', 'double', 'decimal', 'numeric', 'real'],
            'string': ['varchar', 'char', 'text', 'string', 'nvarchar'],
            'datetime': ['datetime', 'timestamp', 'date', 'time'],
            'boolean': ['bool', 'boolean', 'bit']
        }
        
        # 查找类型所属组
        group1 = group2 = None
        for group, types in type_groups.items():
            if any(t in type1_clean for t in types):
                group1 = group
            if any(t in type2_clean for t in types):
                group2 = group
        
        if group1 and group2:
            if group1 == group2:
                return 0.8  # 同组但不完全相同
            elif (group1 in ['integer', 'float'] and group2 in ['integer', 'float']):
                return 0.6  # 数值类型之间
            else:
                return 0.2  # 不同组
        
        return 0.1  # 无法分类的类型
    
    def _calculate_value_similarity(self, values1: List[Any], values2: List[Any]) -> float:
        """计算值分布相似度"""
        try:
            if not values1 or not values2:
                return 0.0
            
            # 标准化值
            normalized_values1 = set(str(v).lower().strip() for v in values1 if v is not None)
            normalized_values2 = set(str(v).lower().strip() for v in values2 if v is not None)
            
            if not normalized_values1 or not normalized_values2:
                return 0.0
            
            # 计算Jaccard相似度
            intersection = len(normalized_values1 & normalized_values2)
            union = len(normalized_values1 | normalized_values2)
            
            if union == 0:
                return 0.0
            
            jaccard_similarity = intersection / union
            
            # 考虑值的分布特征
            len_similarity = min(len(values1), len(values2)) / max(len(values1), len(values2))
            
            # 综合相似度
            return 0.8 * jaccard_similarity + 0.2 * len_similarity
            
        except Exception as e:
            logger.error(f"计算值相似度失败: {e}")
            return 0.0
    
    def _calculate_stats_similarity(self, col1: ColumnInfo, col2: ColumnInfo) -> float:
        """计算统计特征相似度"""
        try:
            similarity_scores = []
            
            # 唯一值数量相似度
            if col1.unique_count is not None and col2.unique_count is not None:
                if max(col1.unique_count, col2.unique_count) > 0:
                    unique_sim = min(col1.unique_count, col2.unique_count) / max(col1.unique_count, col2.unique_count)
                    similarity_scores.append(unique_sim)
            
            # 空值比例相似度
            if col1.null_count is not None and col2.null_count is not None:
                # 计算空值比例（需要知道总行数，这里简化处理）
                null_sim = 1.0 - abs(col1.null_count - col2.null_count) / max(col1.null_count + col2.null_count, 1)
                similarity_scores.append(null_sim)
            
            return sum(similarity_scores) / len(similarity_scores) if similarity_scores else 0.0
            
        except Exception as e:
            logger.error(f"计算统计相似度失败: {e}")
            return 0.0
    
    def _generate_match_reason(self, similarities: Dict[str, float]) -> str:
        """生成匹配理由"""
        reasons = []
        
        if similarities['name'] > 0.8:
            reasons.append("列名高度相似")
        elif similarities['name'] > 0.6:
            reasons.append("列名相似")
        
        if similarities['type'] > 0.8:
            reasons.append("数据类型匹配")
        elif similarities['type'] > 0.5:
            reasons.append("数据类型兼容")
        
        if similarities['value'] > 0.6:
            reasons.append("值分布相似")
        elif similarities['value'] > 0.3:
            reasons.append("部分值重叠")
        
        if similarities['stats'] > 0.7:
            reasons.append("统计特征相似")
        
        return "，".join(reasons) if reasons else "弱相似性"
    
    def _calculate_table_similarity(
        self, 
        source_table: TableInfo, 
        target_table: TableInfo, 
        column_matches: List[MatchResult]
    ) -> float:
        """计算表级别的整体相似度"""
        if not column_matches:
            return 0.0
        
        # 匹配比例
        source_coverage = len(set(m.source_column for m in column_matches)) / len(source_table.columns)
        target_coverage = len(set(m.target_column for m in column_matches)) / len(target_table.columns)
        
        # 平均匹配置信度
        avg_confidence = sum(m.confidence for m in column_matches) / len(column_matches)
        
        # 结构相似度（列数比例）
        column_ratio = min(len(source_table.columns), len(target_table.columns)) / max(len(source_table.columns), len(target_table.columns))
        
        # 综合相似度
        overall_similarity = (
            0.4 * avg_confidence +
            0.3 * max(source_coverage, target_coverage) +
            0.2 * min(source_coverage, target_coverage) +
            0.1 * column_ratio
        )
        
        return min(overall_similarity, 1.0)
    
    async def _llm_evaluate_column_matches(
        self, 
        source_table: TableInfo, 
        target_table: TableInfo, 
        candidate_matches: List[MatchResult]
    ) -> List[MatchResult]:
        """使用LLM评估列匹配结果"""
        try:
            if len(candidate_matches) <= 5:
                return candidate_matches
            
            # 只评估前10个最佳候选
            top_candidates = sorted(candidate_matches, key=lambda x: x.confidence, reverse=True)[:10]
            
            # 构建评估prompt
            source_columns_info = self._format_columns_for_llm(source_table.columns)
            target_columns_info = self._format_columns_for_llm(target_table.columns)
            
            prompt = format_prompt(
                "table_matching",
                source_table=source_table.table_name,
                target_table=target_table.table_name,
                source_columns=source_columns_info,
                target_columns=target_columns_info
            )
            
            # 调用LLM
            response = await self.call_llm_json(prompt)
            
            # 解析结果并更新匹配
            if "column_matches" in response:
                llm_matches = response["column_matches"]
                updated_matches = []
                
                for llm_match in llm_matches:
                    source_col = llm_match.get("source_column", "")
                    target_col = llm_match.get("target_column", "")
                    llm_confidence = llm_match.get("confidence", 0)
                    
                    # 找到对应的原始匹配
                    for original_match in top_candidates:
                        if (source_col in original_match.source_column and 
                            target_col in original_match.target_column):
                            
                            # 合并评分
                            combined_confidence = 0.7 * original_match.confidence + 0.3 * llm_confidence
                            original_match.confidence = min(combined_confidence, 1.0)
                            original_match.reason += f" | LLM确认"
                            updated_matches.append(original_match)
                            break
                
                if updated_matches:
                    return updated_matches
            
            return top_candidates
            
        except Exception as e:
            logger.error(f"LLM列匹配评估失败: {e}")
            return candidate_matches
    
    def _format_columns_for_llm(self, columns: List[ColumnInfo]) -> str:
        """格式化列信息供LLM使用"""
        formatted = []
        for col in columns:
            col_info = f"- {col.column_name}"
            if col.data_type:
                col_info += f" ({col.data_type})"
            if col.sample_values:
                samples = [str(v) for v in col.sample_values[:3] if v is not None]
                if samples:
                    col_info += f" 样本: {', '.join(samples)}"
            formatted.append(col_info)
        
        return "\n".join(formatted)
    
    def _deduplicate_matches(self, matches: List[MatchResult]) -> List[MatchResult]:
        """去重匹配结果（每个源列只保留最佳匹配）"""
        source_to_best_match = {}
        
        for match in matches:
            source_col = match.source_column
            if (source_col not in source_to_best_match or 
                match.confidence > source_to_best_match[source_col].confidence):
                source_to_best_match[source_col] = match
        
        return list(source_to_best_match.values())
    
    async def _get_table_info(self, table_name: str) -> Optional[TableInfo]:
        """获取表信息（从缓存或向量搜索元数据）"""
        if table_name in self.table_info_cache:
            return self.table_info_cache[table_name]
        
        try:
            # 从向量搜索引擎的元数据中获取表信息
            from src.tools.vector_search import vector_search_engine
            
            # 查找表元数据
            for table_info in vector_search_engine.table_metadata.values():
                if table_info.table_name == table_name:
                    # 缓存表信息
                    self.table_info_cache[table_name] = table_info
                    return table_info
            
            logger.warning(f"表信息未找到: {table_name}")
            return None
            
        except Exception as e:
            logger.error(f"获取表信息失败 {table_name}: {e}")
            return None
    
    def cache_table_info(self, table_info: TableInfo):
        """缓存表信息"""
        self.table_info_cache[table_info.table_name] = table_info