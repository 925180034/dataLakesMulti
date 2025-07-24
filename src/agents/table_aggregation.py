from typing import List, Dict, Any, Set
import logging
from collections import defaultdict
from src.agents.base import BaseAgent
from src.core.models import AgentState, MatchResult, TableMatchResult
from src.config.prompts import format_prompt
from src.config.settings import settings

logger = logging.getLogger(__name__)


class TableAggregationAgent(BaseAgent):
    """表聚合智能体 - 基于列匹配结果，评估每个目标表的相关性"""
    
    def __init__(self):
        super().__init__("TableAggregationAgent")
    
    async def process(self, state: AgentState) -> AgentState:
        """处理表聚合任务"""
        # 确保状态对象正确
        state = self._ensure_agent_state(state)
        self.log_progress(state, "开始表聚合处理")
        
        if not state.column_matches:
            self.log_error(state, "没有列匹配结果进行聚合")
            state.current_step = "finalization"
            return state
        
        try:
            # 1. 按目标表分组列匹配结果
            table_groups = self._group_matches_by_table(state.column_matches)
            self.log_progress(state, f"发现 {len(table_groups)} 个候选表")
            
            # 2. 为每个表计算评分
            table_scores = []
            for table_name, matches in table_groups.items():
                score_result = await self._calculate_table_score(table_name, matches, state)
                table_scores.append(score_result)
            
            # 3. 排序并筛选结果
            table_scores.sort(key=lambda x: x.score, reverse=True)
            
            # 4. 使用LLM进行最终评估（可选）
            if len(table_scores) > 3:
                table_scores = await self._llm_evaluate_tables(table_scores, state)
            
            # 5. 保存结果到状态
            state.table_matches = table_scores
            state.final_results = table_scores[:settings.thresholds.top_k_results]
            
            self.log_progress(state, f"表聚合完成，排序后的表数量: {len(state.final_results)}")
            
            # 更新处理步骤
            state.current_step = "finalization"
            
        except Exception as e:
            self.log_error(state, f"表聚合处理失败: {e}")
        
        return state
    
    def _group_matches_by_table(self, column_matches: List[MatchResult]) -> Dict[str, List[MatchResult]]:
        """按目标表名对列匹配结果进行分组"""
        table_groups = defaultdict(list)
        
        for match in column_matches:
            # 从 target_column 中提取表名 (格式: table_name.column_name)
            if '.' in match.target_column:
                table_name = match.target_column.split('.')[0]
            else:
                table_name = "unknown_table"
            
            table_groups[table_name].append(match)
        
        logger.debug(f"按表分组完成: {len(table_groups)} 个表")
        for table_name, matches in table_groups.items():
            logger.debug(f"  {table_name}: {len(matches)} 个匹配列")
        
        return dict(table_groups)
    
    async def _calculate_table_score(
        self, 
        table_name: str, 
        matches: List[MatchResult], 
        state: AgentState
    ) -> TableMatchResult:
        """计算单个表的综合评分"""
        try:
            # 基础统计
            match_count = len(matches)
            avg_confidence = sum(match.confidence for match in matches) / match_count
            
            # 匹配类型分析
            match_types = defaultdict(int)
            for match in matches:
                match_types[match.match_type] += 1
            
            # 关键列检测（ID、主键等）
            key_column_bonus = self._calculate_key_column_bonus(matches)
            
            # 查询列覆盖率
            query_columns = set(col.full_name for col in state.query_columns)
            matched_query_columns = set(match.source_column for match in matches)
            coverage_ratio = len(matched_query_columns) / len(query_columns) if query_columns else 0
            
            # 综合评分计算
            base_score = (
                settings.thresholds.column_count_weight * min(match_count / len(query_columns), 1.0) +
                settings.thresholds.confidence_weight * avg_confidence +
                0.2 * coverage_ratio
            ) * 100
            
            # 添加奖励
            final_score = base_score + key_column_bonus
            final_score = min(final_score, 100.0)  # 限制在100分以内
            
            # 构建证据信息
            evidence = {
                "match_count": match_count,
                "avg_confidence": avg_confidence,
                "coverage_ratio": coverage_ratio,
                "match_types": dict(match_types),
                "key_column_bonus": key_column_bonus,
                "matched_columns": [match.target_column.split('.')[-1] for match in matches]
            }
            
            # 生成推荐理由
            reason = self._generate_recommendation_reason(matches, evidence)
            
            result = TableMatchResult(
                source_table=", ".join([col.table_name for col in state.query_columns]),
                target_table=table_name,
                score=final_score,
                matched_columns=matches,
                evidence=evidence
            )
            
            logger.debug(f"表 {table_name} 评分: {final_score:.1f} (匹配列: {match_count})")
            return result
            
        except Exception as e:
            logger.error(f"计算表 {table_name} 评分失败: {e}")
            # 返回最低分数的结果
            return TableMatchResult(
                source_table="",
                target_table=table_name,
                score=0.0,
                matched_columns=matches,
                evidence={"error": str(e)}
            )
    
    def _calculate_key_column_bonus(self, matches: List[MatchResult]) -> float:
        """计算关键列匹配的奖励分数"""
        key_column_patterns = [
            'id', 'key', 'uuid', 'guid', 'primary', 'pk',
            'user_id', 'customer_id', 'order_id', 'product_id'
        ]
        
        bonus = 0.0
        for match in matches:
            column_name = match.target_column.split('.')[-1].lower()
            
            # 检查是否为关键列
            for pattern in key_column_patterns:
                if pattern in column_name:
                    bonus += settings.thresholds.key_column_bonus
                    logger.debug(f"关键列匹配奖励: {match.target_column}")
                    break
        
        return min(bonus, 20.0)  # 限制奖励上限
    
    def _generate_recommendation_reason(
        self, 
        matches: List[MatchResult], 
        evidence: Dict[str, Any]
    ) -> str:
        """生成推荐理由"""
        reason_parts = []
        
        # 匹配列数量
        match_count = evidence["match_count"]
        if match_count == 1:
            reason_parts.append("找到1个匹配列")
        else:
            reason_parts.append(f"找到{match_count}个匹配列")
        
        # 平均置信度
        avg_conf = evidence["avg_confidence"]
        if avg_conf >= 0.8:
            reason_parts.append("高置信度匹配")
        elif avg_conf >= 0.6:
            reason_parts.append("中等置信度匹配")
        else:
            reason_parts.append("低置信度匹配")
        
        # 匹配类型
        match_types = evidence["match_types"]
        if "semantic+value" in match_types:
            reason_parts.append("包含语义和值双重匹配")
        elif "semantic" in match_types and "value_overlap" in match_types:
            reason_parts.append("包含语义和值重叠匹配")
        elif "semantic" in match_types:
            reason_parts.append("主要为语义匹配")
        elif "value_overlap" in match_types:
            reason_parts.append("主要为值重叠匹配")
        
        # 关键列
        if evidence["key_column_bonus"] > 0:
            reason_parts.append("包含关键列匹配")
        
        # 覆盖率
        coverage = evidence["coverage_ratio"]
        if coverage >= 0.8:
            reason_parts.append("高查询覆盖率")
        elif coverage >= 0.5:
            reason_parts.append("中等查询覆盖率")
        
        return "，".join(reason_parts)
    
    async def _llm_evaluate_tables(
        self, 
        table_scores: List[TableMatchResult],
        state: AgentState
    ) -> List[TableMatchResult]:
        """使用LLM对表评分结果进行最终评估"""
        try:
            if len(table_scores) <= 3:
                return table_scores
            
            # 准备LLM输入
            column_matches_summary = self._format_matches_for_llm(state.column_matches)
            
            prompt = format_prompt(
                "table_aggregation",
                column_matches=column_matches_summary
            )
            
            # 调用LLM
            response = await self.call_llm_json(prompt)
            
            if "ranked_tables" in response:
                llm_rankings = response["ranked_tables"]
                
                # 根据LLM排序更新分数
                updated_scores = []
                for llm_result in llm_rankings:
                    table_name = llm_result.get("table_name", "")
                    llm_score = llm_result.get("score", 0)
                    llm_reason = llm_result.get("reason", "")
                    
                    # 找到对应的原始结果
                    for original_result in table_scores:
                        if original_result.target_table == table_name:
                            # 合并LLM评估
                            combined_score = 0.7 * original_result.score + 0.3 * llm_score
                            
                            # 更新证据
                            original_result.evidence["llm_score"] = llm_score
                            original_result.evidence["llm_reason"] = llm_reason
                            original_result.evidence["combined_score"] = combined_score
                            original_result.score = combined_score
                            
                            updated_scores.append(original_result)
                            break
                
                if updated_scores:
                    self.log_progress(state, f"LLM评估更新了 {len(updated_scores)} 个表的评分")
                    return updated_scores
            
            self.log_progress(state, "LLM评估失败，使用原始评分")
            return table_scores
            
        except Exception as e:
            logger.error(f"LLM表评估失败: {e}")
            return table_scores
    
    def _format_matches_for_llm(self, column_matches: List[MatchResult]) -> str:
        """格式化列匹配结果供LLM使用"""
        # 按表分组
        table_groups = self._group_matches_by_table(column_matches)
        
        formatted_parts = []
        for table_name, matches in table_groups.items():
            match_details = []
            for match in matches:
                source_col = match.source_column.split('.')[-1]
                target_col = match.target_column.split('.')[-1]
                match_details.append(
                    f"  {source_col} -> {target_col} "
                    f"(置信度: {match.confidence:.3f}, 类型: {match.match_type})"
                )
            
            formatted_parts.append(
                f"表 {table_name}:\n" + "\n".join(match_details)
            )
        
        return "\n\n".join(formatted_parts)
    
    def get_top_tables(self, state: AgentState, k: int = 5) -> List[TableMatchResult]:
        """获取评分最高的k个表"""
        if not state.table_matches:
            return []
        
        sorted_tables = sorted(state.table_matches, key=lambda x: x.score, reverse=True)
        return sorted_tables[:k]
    
    def get_table_statistics(self, state: AgentState) -> Dict[str, Any]:
        """获取表聚合的统计信息"""
        if not state.table_matches:
            return {}
        
        scores = [result.score for result in state.table_matches]
        match_counts = [len(result.matched_columns) for result in state.table_matches]
        
        return {
            "total_tables": len(state.table_matches),
            "avg_score": sum(scores) / len(scores),
            "max_score": max(scores),
            "min_score": min(scores),
            "avg_match_count": sum(match_counts) / len(match_counts),
            "top_3_tables": [
                {"table": result.target_table, "score": result.score}
                for result in state.table_matches[:3]
            ]
        }