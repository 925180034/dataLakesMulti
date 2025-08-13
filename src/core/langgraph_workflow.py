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
        
        # Build workflow
        self.workflow = self._build_workflow()
        
    def _build_workflow(self) -> CompiledStateGraph:
        """Build the LangGraph workflow"""
        
        # Create state graph
        graph = StateGraph(WorkflowState)
        
        # Add nodes (agents)
        graph.add_node("optimizer", self.optimizer.process)
        graph.add_node("planner", self.planner.process)
        graph.add_node("analyzer", self.analyzer.process)
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
        
        # Find query table
        query_table = None
        if query_table_name:
            # Find specific table
            for table in tables:
                if table.get('table_name') == query_table_name:
                    query_table = table
                    break
        else:
            # Use first table as query table for testing
            if tables:
                query_table = tables[0]
        
        if not query_table:
            self.logger.error("No query table found")
            return {
                'success': False,
                'error': 'No query table found',
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