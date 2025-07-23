from typing import Dict, Any
import logging
from src.agents.base import BaseAgent
from src.core.models import AgentState, TaskStrategy
from src.config.prompts import format_prompt

logger = logging.getLogger(__name__)


class PlannerAgent(BaseAgent):
    """规划器智能体 - 负责任务理解、策略选择和流程协调"""
    
    def __init__(self):
        super().__init__("PlannerAgent")
    
    async def process(self, state: AgentState) -> AgentState:
        """处理规划任务"""
        self.log_progress(state, "开始任务规划")
        
        # 如果还没有选择策略，进行策略决策
        if state.strategy is None:
            await self._decide_strategy(state)
        
        # 根据当前步骤决定下一步行动
        if state.current_step == "planning":
            await self._plan_execution(state)
        elif state.current_step == "finalization":
            await self._generate_final_report(state)
        
        return state
    
    async def _decide_strategy(self, state: AgentState) -> None:
        """决定处理策略"""
        self.log_progress(state, "分析用户需求，选择处理策略")
        
        try:
            # 构建策略决策Prompt
            prompt = format_prompt(
                "planner_strategy_decision",
                user_query=state.user_query,
                table_count=len(state.query_tables),
                column_count=len(state.query_columns)
            )
            
            # 调用LLM进行策略选择
            response = await self.call_llm(prompt)
            
            # 解析策略选择
            if "1" in response or "Bottom-Up" in response or "自下而上" in response:
                state.strategy = TaskStrategy.BOTTOM_UP
                self.log_progress(state, "选择Bottom-Up策略: 先匹配列，再聚合表")
            elif "2" in response or "Top-Down" in response or "自上而下" in response:
                state.strategy = TaskStrategy.TOP_DOWN
                self.log_progress(state, "选择Top-Down策略: 先发现表，再匹配列")
            else:
                # 默认策略
                state.strategy = TaskStrategy.BOTTOM_UP
                self.log_progress(state, "使用默认Bottom-Up策略")
                
        except Exception as e:
            self.log_error(state, f"策略决策失败: {e}")
            # 使用默认策略
            state.strategy = TaskStrategy.BOTTOM_UP
    
    async def _plan_execution(self, state: AgentState) -> None:
        """规划执行步骤"""
        if state.strategy == TaskStrategy.BOTTOM_UP:
            self.log_progress(state, "规划Bottom-Up执行流程")
            state.current_step = "column_discovery"
        elif state.strategy == TaskStrategy.TOP_DOWN:
            self.log_progress(state, "规划Top-Down执行流程")
            state.current_step = "table_discovery"
        else:
            self.log_error(state, f"未知策略: {state.strategy}")
            state.current_step = "error"
    
    async def _generate_final_report(self, state: AgentState) -> None:
        """生成最终报告"""
        self.log_progress(state, "生成最终报告")
        
        try:
            # 整理结果数据
            steps = " -> ".join([
                step for step in [
                    "策略选择",
                    "列发现" if state.strategy == TaskStrategy.BOTTOM_UP else "表发现",
                    "表聚合" if state.strategy == TaskStrategy.BOTTOM_UP else "表匹配",
                    "结果整合"
                ]
            ])
            
            results_summary = self._summarize_results(state)
            
            # 生成报告Prompt
            prompt = format_prompt(
                "planner_final_report",
                strategy=state.strategy.value,
                steps=steps,
                results=results_summary
            )
            
            # 调用LLM生成报告
            final_report = await self.call_llm(prompt)
            state.final_report = final_report
            
            self.log_progress(state, "最终报告生成完成")
            
        except Exception as e:
            self.log_error(state, f"生成最终报告失败: {e}")
            state.final_report = self._create_fallback_report(state)
    
    def _summarize_results(self, state: AgentState) -> str:
        """总结匹配结果"""
        if not state.final_results:
            return "未找到匹配的表"
        
        summary_parts = []
        for result in state.final_results[:5]:  # 只显示前5个结果
            summary_parts.append(
                f"- {result.target_table} (评分: {result.score:.1f}, "
                f"匹配列数: {result.match_count})"
            )
        
        return "\n".join(summary_parts)
    
    def _create_fallback_report(self, state: AgentState) -> str:
        """创建备用报告"""
        report_parts = [
            f"# 数据发现报告",
            f"",
            f"**查询内容**: {state.user_query}",
            f"**处理策略**: {state.strategy.value if state.strategy else '未知'}",
            f"**处理状态**: {'完成' if not state.error_messages else '部分完成'}",
            f"",
        ]
        
        if state.final_results:
            report_parts.extend([
                f"**发现的相关表** ({len(state.final_results)}个):",
                ""
            ])
            for result in state.final_results[:10]:
                report_parts.append(
                    f"- **{result.target_table}** (评分: {result.score:.1f})"
                )
                if result.matched_columns:
                    report_parts.append(f"  - 匹配列: {', '.join([m.target_column for m in result.matched_columns[:3]])}")
        else:
            report_parts.append("**结果**: 未找到匹配的表")
        
        if state.error_messages:
            report_parts.extend([
                "",
                "**处理过程中的问题**:",
                ""
            ])
            for error in state.error_messages[-3:]:  # 只显示最后3个错误
                report_parts.append(f"- {error}")
        
        return "\n".join(report_parts)
    
    def should_continue_bottom_up(self, state: AgentState) -> str:
        """判断Bottom-Up流程是否应该继续"""
        if state.current_step == "column_discovery":
            return "table_aggregation"
        elif state.current_step == "table_aggregation":
            return "finalization"
        else:
            return "end"
    
    def should_continue_top_down(self, state: AgentState) -> str:
        """判断Top-Down流程是否应该继续"""
        if state.current_step == "table_discovery":
            return "table_matching"
        elif state.current_step == "table_matching":
            return "finalization"
        else:
            return "end"