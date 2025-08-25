"""
Main LangGraph workflow for multi-agent data lake discovery system
"""
import logging
import time
from typing import Dict, Any, List, Optional
from langgraph.graph import StateGraph, END
from langgraph.graph.state import CompiledStateGraph

from src.core.state import WorkflowState, QueryTask, PerformanceMetrics
from src.agents.optimizer_agent import OptimizerAgent
from src.agents.planner_agent import PlannerAgent
from src.agents.analyzer_agent import AnalyzerAgent
from src.agents.searcher_agent import SearcherAgent
from src.agents.matcher_agent import MatcherAgent
from src.agents.aggregator_agent import AggregatorAgent

logger = logging.getLogger(__name__)


class DataLakeDiscoveryWorkflow:
    """
    LangGraph-based workflow for multi-agent data lake discovery
    """
    
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Initialize agents
        self.optimizer = OptimizerAgent()
        self.planner = PlannerAgent()
        self.analyzer = AnalyzerAgent()
        self.searcher = SearcherAgent()
        self.matcher = MatcherAgent()
        self.aggregator = AggregatorAgent()
        
        # 缓存机制 - 避免重复调用
        self.optimization_cache = {}  # {task_type: optimization_config}
        self.strategy_cache = {}      # {task_type: strategy}
        self.analysis_cache = {}      # {table_name: analysis_result}
        
        # Build workflow
        self.workflow = self._build_workflow()
        
    def _build_workflow(self) -> CompiledStateGraph:
        """Build the LangGraph workflow"""
        
        # Create state graph
        graph = StateGraph(WorkflowState)
        
        # Add nodes (agents) with caching wrappers
        graph.add_node("optimizer", self._cached_optimizer_process)
        graph.add_node("planner", self._cached_planner_process)
        graph.add_node("analyzer", self._cached_analyzer_process)
        graph.add_node("searcher", self.searcher.process)
        graph.add_node("matcher", self.matcher.process)
        graph.add_node("aggregator", self.aggregator.process)
        
        # Define edges (workflow)
        graph.set_entry_point("optimizer")
        
        # Sequential flow
        graph.add_edge("optimizer", "planner")
        graph.add_edge("planner", "analyzer")
        graph.add_edge("analyzer", "searcher")
        
        # Conditional edge from searcher
        graph.add_conditional_edges(
            "searcher",
            self._should_use_matcher,
            {
                True: "matcher",
                False: "aggregator"
            }
        )
        
        # Matcher always goes to aggregator
        graph.add_edge("matcher", "aggregator")
        
        # Aggregator is the end
        graph.add_edge("aggregator", END)
        
        # Compile the graph
        return graph.compile()
    
    def _cached_optimizer_process(self, state: WorkflowState) -> WorkflowState:
        """Optimizer with caching"""
        # 获取缓存键
        query_task = state.get('query_task')
        all_tables = state.get('all_tables', [])
        if query_task:
            cache_key = f"{query_task.task_type}_{len(all_tables)}"
        else:
            cache_key = None
        
        # 检查缓存
        if cache_key and cache_key in self.optimization_cache:
            self.logger.info(f"✅ Using cached optimization config for {cache_key}")
            state['optimization_config'] = self.optimization_cache[cache_key]
            state['should_use_llm'] = state['optimization_config'].use_llm_verification
            # 更新metrics - 记录跳过了优化器
            if 'metrics' in state:
                state['metrics'].agent_times['optimizer'] = 0.001  # 缓存命中，几乎无耗时
            return state
        
        # 调用原始处理
        self.logger.info(f"🔄 Computing new optimization config for {cache_key}")
        state = self.optimizer.process(state)
        
        # 保存到缓存
        if cache_key and 'optimization_config' in state:
            self.optimization_cache[cache_key] = state['optimization_config']
            self.logger.info(f"💾 Cached optimization config for {cache_key}")
        
        return state
    
    def _cached_planner_process(self, state: WorkflowState) -> WorkflowState:
        """Planner with caching"""
        # 获取缓存键
        query_task = state.get('query_task')
        all_tables = state.get('all_tables', [])
        if query_task:
            cache_key = f"{query_task.task_type}_{len(all_tables)}"
        else:
            cache_key = None
        
        # 检查缓存
        if cache_key and cache_key in self.strategy_cache:
            self.logger.info(f"✅ Using cached strategy for {cache_key}")
            state['strategy'] = self.strategy_cache[cache_key]
            # 更新metrics
            if 'metrics' in state:
                state['metrics'].agent_times['planner'] = 0.001
            return state
        
        # 调用原始处理
        self.logger.info(f"🔄 Computing new strategy for {cache_key}")
        state = self.planner.process(state)
        
        # 保存到缓存
        if cache_key and 'strategy' in state:
            self.strategy_cache[cache_key] = state['strategy']
            self.logger.info(f"💾 Cached strategy for {cache_key}")
        
        return state
    
    def _cached_analyzer_process(self, state: WorkflowState) -> WorkflowState:
        """Analyzer with caching for table analysis"""
        query_table = state.get('query_table')
        
        if query_table:
            table_name = query_table.get('table_name', '')
            
            # 检查缓存
            if table_name and table_name in self.analysis_cache:
                self.logger.info(f"✅ Using cached analysis for table {table_name}")
                state['table_analysis'] = self.analysis_cache[table_name]
                # 更新metrics
                if 'metrics' in state:
                    state['metrics'].agent_times['analyzer'] = 0.001
                return state
            
            # 调用原始处理
            self.logger.info(f"🔄 Analyzing table {table_name}")
            state = self.analyzer.process(state)
            
            # 保存到缓存
            if table_name and state and 'table_analysis' in state:
                self.analysis_cache[table_name] = state['table_analysis']
                self.logger.info(f"💾 Cached analysis for table {table_name}")
        else:
            # 没有查询表，直接调用原始处理
            state = self.analyzer.process(state)
        
        return state
    
    def _should_use_matcher(self, state: WorkflowState) -> bool:
        """Determine if we should use the matcher agent"""
        # Check if matcher should be skipped
        if state.get('skip_matcher', False):
            self.logger.info("Skipping matcher agent (flag set)")
            return False
            
        # Check if we have candidates to match
        candidates = state.get('candidates', [])
        if not candidates:
            self.logger.info("No candidates for matcher")
            return False
            
        # Check strategy
        strategy = state.get('strategy')
        if strategy and not strategy.use_llm:
            self.logger.info("LLM disabled in strategy")
            return False
            
        self.logger.info("Using matcher agent for LLM verification")
        return True
    
    def run(self, query: str, tables: List[Dict[str, Any]], 
            task_type: str = 'join',
            query_table_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Run the workflow
        
        Args:
            query: User query or table name
            tables: List of all tables
            task_type: 'join' or 'union'
            query_table_name: Specific table to find matches for
            
        Returns:
            Workflow results
        """
        start_time = time.time()
        self.logger.info(f"Starting workflow for query: {query}, task: {task_type}")
        self.logger.debug(f"Query table name: {query_table_name}, Tables count: {len(tables)}")
        
        # 检查缓存 - 对于相同任务类型，复用优化配置和策略
        cache_key = f"{task_type}_{len(tables)}"
        self.logger.info(f"Cache key: {cache_key}, Existing caches: {list(self.optimization_cache.keys())}")
        
        # Find query table
        query_table = None
        if query_table_name:
            # Find specific table
            for table in tables:
                if table.get('table_name') == query_table_name:
                    query_table = table
                    break
            if not query_table:
                # Log first few table names for debugging
                sample_names = [t.get('table_name', 'unnamed') for t in tables[:5]]
                self.logger.debug(f"Sample table names: {sample_names}")
                self.logger.debug(f"Looking for: {query_table_name}")
        else:
            # Use first table as query table for testing
            if tables:
                query_table = tables[0]
        
        if not query_table:
            self.logger.error(f"No query table found. Query table name: {query_table_name}, Tables count: {len(tables)}")
            return {
                'success': False,
                'error': f'No query table found: {query_table_name}',
                'results': []
            }
        
        # Initialize state
        initial_state: WorkflowState = {
            'query_task': QueryTask(
                query=query,
                task_type=task_type,
                table_name=query_table.get('table_name', '')
            ),
            'query_table': query_table,
            'all_tables': tables,
            'metrics': PerformanceMetrics(
                total_time=0,
                agent_times={},
                candidates_generated=0,
                llm_calls_made=0
            ),
            'errors': []
        }
        
        try:
            # Run workflow
            self.logger.info("Executing LangGraph workflow...")
            result_state = self.workflow.invoke(initial_state)
            
            # Extract results
            final_results = result_state.get('final_results', [])
            metrics = result_state.get('metrics')
            errors = result_state.get('errors', [])
            
            # Calculate total time
            total_time = time.time() - start_time
            if metrics:
                metrics.total_time = total_time
            
            # Format results
            formatted_results = []
            for match in final_results:
                formatted_results.append({
                    'table_name': match.matched_table,
                    'score': match.score,
                    'confidence': match.confidence,
                    'match_type': match.match_type,
                    'agent': match.agent_used,
                    'evidence': match.evidence
                })
            
            self.logger.info(f"Workflow completed in {total_time:.2f}s")
            self.logger.info(f"Found {len(formatted_results)} matches")
            
            return {
                'success': True,
                'query_table': query_table.get('table_name'),
                'task_type': task_type,
                'results': formatted_results,
                'metrics': {
                    'total_time': total_time,
                    'candidates_generated': metrics.candidates_generated if metrics else 0,
                    'llm_calls_made': metrics.llm_calls_made if metrics else 0,
                    'agent_times': metrics.agent_times if metrics else {}
                },
                'errors': errors
            }
            
        except Exception as e:
            self.logger.error(f"Workflow failed: {e}", exc_info=True)
            return {
                'success': False,
                'error': str(e),
                'results': [],
                'total_time': time.time() - start_time
            }

    def run_nlctables(self, nlc_query: Dict[str, Any], tables: List[Dict[str, Any]], 
                     task_type: str = 'join') -> Dict[str, Any]:
        """
        Run the workflow for NLCTables queries
        
        Args:
            nlc_query: NLCTables query with query_text and features
            tables: List of all tables
            task_type: 'join' or 'union'
            
        Returns:
            Workflow results
        """
        start_time = time.time()
        query_id = nlc_query.get('query_id', '')
        query_text = nlc_query.get('query_text', '')
        features = nlc_query.get('features', {})
        seed_table = nlc_query.get('seed_table', '')
        
        self.logger.info(f"Starting NLCTables workflow for query: {query_id}")
        self.logger.debug(f"Query text: {query_text}")
        self.logger.debug(f"Features: {features}")
        
        # Find seed table if it exists
        query_table = None
        if seed_table:
            for table in tables:
                if table.get('table_name') == seed_table or table.get('name') == seed_table:
                    query_table = table
                    break
        
        # If no seed table found, create a pseudo table from features
        if not query_table and features:
            column_mentions = features.get('column_mentions', [])
            keywords = features.get('keywords', [])
            
            # Create pseudo table for NLCTables
            query_table = {
                'table_name': f'nlc_query_{query_id}',
                'name': f'nlc_query_{query_id}',
                'columns': [{'name': col, 'column_name': col} for col in column_mentions],
                'description': query_text,
                'keywords': keywords,
                'is_nlc_query': True
            }
            self.logger.info(f"Created pseudo table for NLCTables query: {query_table['name']}")
        
        # Initialize state with NLCTables information
        initial_state: WorkflowState = {
            'query_task': QueryTask(
                query=query_text,
                task_type=task_type,
                table_name=query_table.get('table_name', '') if query_table else '',
                query_text=query_text,  # NLCTables specific
                features=features,  # NLCTables specific
                query_id=query_id  # NLCTables specific
            ),
            'query_table': query_table,
            'all_tables': tables,
            'is_nlctables': True,  # Flag to indicate NLCTables query
            'nl_features': features,  # Natural language features
            'query_text': query_text,  # Natural language query
            'metrics': PerformanceMetrics(
                total_time=0,
                agent_times={},
                candidates_generated=0,
                llm_calls_made=0
            ),
            'errors': []
        }
        
        try:
            # Run workflow
            self.logger.info("Executing NLCTables workflow...")
            result_state = self.workflow.invoke(initial_state)
            
            # Extract results
            final_results = result_state.get('final_results', [])
            metrics = result_state.get('metrics')
            errors = result_state.get('errors', [])
            
            # Calculate total time
            total_time = time.time() - start_time
            if metrics:
                metrics.total_time = total_time
            
            # Format results
            formatted_results = []
            for match in final_results:
                formatted_results.append({
                    'table_name': match.matched_table,
                    'score': match.score,
                    'confidence': match.confidence,
                    'match_type': match.match_type,
                    'agent': match.agent_used,
                    'evidence': match.evidence
                })
            
            self.logger.info(f"NLCTables workflow completed in {total_time:.2f}s")
            self.logger.info(f"Found {len(formatted_results)} matches")
            
            return {
                'success': True,
                'query_id': query_id,
                'task_type': task_type,
                'results': formatted_results,
                'metrics': {
                    'total_time': total_time,
                    'candidates_generated': metrics.candidates_generated if metrics else 0,
                    'llm_calls_made': metrics.llm_calls_made if metrics else 0,
                    'agent_times': metrics.agent_times if metrics else {}
                },
                'errors': errors
            }
            
        except Exception as e:
            self.logger.error(f"NLCTables workflow failed: {e}", exc_info=True)
            return {
                'success': False,
                'error': str(e),
                'results': [],
                'total_time': time.time() - start_time
            }
    
    def run_batch(self, queries: List[Dict[str, Any]], 
                  tables: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Run workflow for multiple queries
        
        Args:
            queries: List of query specifications
            tables: List of all tables
            
        Returns:
            List of results
        """
        results = []
        
        for query_spec in queries:
            query = query_spec.get('query', '')
            task_type = query_spec.get('task_type', 'join')
            table_name = query_spec.get('table_name')
            
            result = self.run(
                query=query,
                tables=tables,
                task_type=task_type,
                query_table_name=table_name
            )
            
            results.append(result)
        
        return results


def create_workflow() -> DataLakeDiscoveryWorkflow:
    """Create and return workflow instance"""
    return DataLakeDiscoveryWorkflow()