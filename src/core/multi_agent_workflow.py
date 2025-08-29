"""
Multi-Agent Workflow Orchestration for Data Lake Discovery System
多智能体工作流编排 - 协调6个Agent使用三层加速工具
"""

import logging
import time
from typing import Dict, List
from langgraph.graph import StateGraph, END

from src.core.state import WorkflowState
from src.agents.optimizer_agent import OptimizerAgent
from src.agents.planner_agent import PlannerAgent
from src.agents.analyzer_agent import AnalyzerAgent
from src.agents.searcher_agent import SearcherAgent
from src.agents.matcher_agent import MatcherAgent
from src.agents.aggregator_agent import AggregatorAgent

logger = logging.getLogger(__name__)


class MultiAgentWorkflow:
    """
    Multi-Agent Workflow for Data Lake Discovery
    
    协调6个智能体：
    1. OptimizerAgent - 系统优化配置（执行一次）
    2. PlannerAgent - 策略规划决策（执行一次）
    3. AnalyzerAgent - 数据结构分析（每个查询执行）
    4. SearcherAgent - 候选搜索（Layer 1+2）
    5. MatcherAgent - 精确匹配（Layer 3）
    6. AggregatorAgent - 结果聚合排序
    """
    
    def __init__(self):
        # 初始化所有Agent
        self.optimizer = OptimizerAgent()
        self.planner = PlannerAgent()
        self.analyzer = AnalyzerAgent()
        self.searcher = SearcherAgent()
        self.matcher = MatcherAgent()
        self.aggregator = AggregatorAgent()
        
        # Build workflow
        self.workflow = self._build_workflow()
        self.app = self.workflow.compile()
        
        # Cache for batch processing optimization
        self.optimization_config = None
        self.planning_strategy = None
        
        logger.info("Multi-Agent Workflow initialized with 6 agents")
    
    def _build_workflow(self) -> StateGraph:
        """Build the multi-agent workflow graph"""
        workflow = StateGraph(WorkflowState)
        
        # Add all agent nodes
        workflow.add_node("optimizer", self._optimizer_node)
        workflow.add_node("planner", self._planner_node)
        workflow.add_node("analyzer", self._analyzer_node)
        workflow.add_node("searcher", self._searcher_node)
        workflow.add_node("matcher", self._matcher_node)
        workflow.add_node("aggregator", self._aggregator_node)
        
        # Set entry point
        workflow.set_entry_point("optimizer")
        
        # Define workflow edges (sequential flow as per architecture)
        workflow.add_edge("optimizer", "planner")
        workflow.add_edge("planner", "analyzer")
        workflow.add_edge("analyzer", "searcher")
        workflow.add_edge("searcher", "matcher")
        workflow.add_edge("matcher", "aggregator")
        workflow.add_edge("aggregator", END)
        
        return workflow
    
    async def _optimizer_node(self, state: WorkflowState) -> WorkflowState:
        """OptimizerAgent node - 系统优化配置"""
        try:
            # Check if already optimized for this batch
            if self.optimization_config and not state.get('force_reoptimize'):
                logger.info("Using cached optimization config for batch")
                state['optimization_config'] = self.optimization_config
                return state
            
            logger.info("OptimizerAgent: Determining system optimization configuration")
            # Agents use synchronous process method
            state = self.optimizer.process(state)
            
            # Cache the optimization config for batch processing
            self.optimization_config = state.get('optimization_config')
            
            return state
        except Exception as e:
            logger.error(f"OptimizerAgent failed: {e}")
            # Use default optimization if agent fails
            state['optimization_config'] = {
                'parallel_workers': 8,
                'llm_concurrency': 3,
                'cache_strategy': 'L2',
                'batch_size': 10
            }
            return state
    
    async def _planner_node(self, state: WorkflowState) -> WorkflowState:
        """PlannerAgent node - 策略规划决策"""
        try:
            # Check if already planned for this batch
            if self.planning_strategy and not state.get('force_replan'):
                logger.info("Using cached planning strategy for batch")
                state['strategy'] = self.planning_strategy
                return state
            
            logger.info("PlannerAgent: Determining execution strategy")
            # Agents use synchronous process method
            state = self.planner.process(state)
            
            # Cache the planning strategy for batch processing
            self.planning_strategy = state.get('strategy')
            
            return state
        except Exception as e:
            logger.error(f"PlannerAgent failed: {e}")
            # Use default strategy if agent fails
            state['strategy'] = {
                'name': 'bottom-up',
                'use_metadata': True,
                'use_vector': True,
                'top_k': 50
            }
            return state
    
    async def _analyzer_node(self, state: WorkflowState) -> WorkflowState:
        """AnalyzerAgent node - 数据结构分析"""
        try:
            logger.info("AnalyzerAgent: Analyzing query table structure")
            # Agents use synchronous process method
            state = self.analyzer.process(state)
            return state
        except Exception as e:
            logger.error(f"AnalyzerAgent failed: {e}")
            state['analysis'] = {
                'table_type': 'unknown',
                'key_columns': [],
                'complexity': 'medium'
            }
            return state
    
    async def _searcher_node(self, state: WorkflowState) -> WorkflowState:
        """SearcherAgent node - Layer 1+2 候选搜索"""
        try:
            logger.info("SearcherAgent: Executing Layer 1+2 candidate search")
            # Agents use synchronous process method
            state = self.searcher.process(state)
            
            candidates = state.get('candidates', [])
            logger.info(f"SearcherAgent found {len(candidates)} candidates")
            
            return state
        except Exception as e:
            logger.error(f"SearcherAgent failed: {e}")
            state['candidates'] = []
            return state
    
    async def _matcher_node(self, state: WorkflowState) -> WorkflowState:
        """MatcherAgent node - Layer 3 精确匹配"""
        try:
            # Check if we have candidates to verify
            candidates = state.get('candidates', [])
            if not candidates:
                logger.info("No candidates to verify, skipping MatcherAgent")
                state['matches'] = []
                return state
            
            logger.info("MatcherAgent: Executing Layer 3 LLM verification")
            # Agents use synchronous process method
            state = self.matcher.process(state)
            
            matches = state.get('matches', [])
            logger.info(f"MatcherAgent verified {len(matches)} matches")
            
            return state
        except Exception as e:
            logger.error(f"MatcherAgent failed: {e}")
            # Convert candidates to matches without verification
            state['matches'] = []
            for c in state.get('candidates', [])[:5]:
                state['matches'].append({
                    'table_name': c.table_name,
                    'score': c.score * 0.5,  # Lower confidence without LLM
                    'source': 'searcher_only'
                })
            return state
    
    async def _aggregator_node(self, state: WorkflowState) -> WorkflowState:
        """AggregatorAgent node - 结果聚合排序"""
        try:
            logger.info("AggregatorAgent: Aggregating and ranking final results")
            # Agents use synchronous process method
            state = self.aggregator.process(state)
            
            final_results = state.get('final_results', [])
            logger.info(f"AggregatorAgent produced {len(final_results)} final results")
            
            return state
        except Exception as e:
            logger.error(f"AggregatorAgent failed: {e}")
            # Use matches as final results if aggregator fails
            state['final_results'] = state.get('matches', [])
            return state
    
    async def run(self, initial_state: WorkflowState) -> WorkflowState:
        """
        Run the multi-agent workflow
        
        Args:
            initial_state: Initial workflow state
            
        Returns:
            Final workflow state with results
        """
        try:
            start_time = time.time()
            logger.info("Starting Multi-Agent Workflow execution")
            
            # Validate initial state
            if not initial_state.get('query_table'):
                raise ValueError("query_table is required in initial state")
            
            if not initial_state.get('all_tables'):
                raise ValueError("all_tables is required in initial state")
            
            # Execute workflow
            result = await self.app.ainvoke(initial_state)
            
            # Log performance
            elapsed_time = time.time() - start_time
            logger.info(f"Multi-Agent Workflow completed in {elapsed_time:.2f} seconds")
            
            # Add timing to result
            result['execution_time'] = elapsed_time
            
            return result
            
        except Exception as e:
            logger.error(f"Multi-Agent Workflow failed: {e}")
            initial_state['error'] = str(e)
            return initial_state
    
    async def run_batch(self, queries: List[Dict], all_tables: List[Dict]) -> List[WorkflowState]:
        """
        Run workflow for multiple queries (batch processing)
        OptimizerAgent and PlannerAgent execute once for the batch
        
        Args:
            queries: List of query configurations
            all_tables: All tables in the data lake
            
        Returns:
            List of workflow results
        """
        results = []
        
        # Reset cached configs for new batch
        self.optimization_config = None
        self.planning_strategy = None
        
        for i, query in enumerate(queries):
            logger.info(f"Processing query {i+1}/{len(queries)}")
            
            # Prepare initial state
            # Create proper query_task object with required attributes
            query_task = {
                'table_name': query.get('query_table', {}).get('table_name', 'unknown'),
                'task_type': query.get('task_type', 'join')  # 修复: 使用 task_type 而不是 task
            }
            
            initial_state = WorkflowState(
                query_table=query.get('query_table'),
                query_task=query_task,  # Pass proper task object
                all_tables=all_tables,
                force_reoptimize=False,  # Use cached optimization
                force_replan=False  # Use cached planning
            )
            
            # Run workflow
            result = await self.run(initial_state)
            results.append(result)
        
        return results


def create_multi_agent_workflow() -> MultiAgentWorkflow:
    """Factory function to create multi-agent workflow"""
    return MultiAgentWorkflow()

# Compatibility aliases for old code
def create_workflow() -> MultiAgentWorkflow:
    """Factory function for compatibility with old code"""
    return create_multi_agent_workflow()

# Alias for compatibility
DataLakeDiscoveryWorkflow = MultiAgentWorkflow