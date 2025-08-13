"""
PlannerAgent - Strategy selection and execution planning
"""
from src.agents.base_agent import BaseAgent
from src.core.state import WorkflowState, ExecutionStrategy


class PlannerAgent(BaseAgent):
    """
    Planner Agent responsible for determining the execution strategy
    """
    
    def __init__(self):
        super().__init__(
            name="PlannerAgent",
            description="Selects optimal execution strategy based on task type and table characteristics"
        )
        
    def process(self, state: WorkflowState) -> WorkflowState:
        """
        Determine execution strategy based on task type
        
        Args:
            state: Current workflow state
            
        Returns:
            Updated state with execution strategy
        """
        self.logger.info("Planning execution strategy")
        
        # Get query task and optimization config
        query_task = state.get('query_task')
        optimization_config = state.get('optimization_config')
        
        # Initialize strategy
        strategy = ExecutionStrategy(name="hybrid")
        
        # Determine strategy based on task type
        if query_task and query_task.task_type == 'join':
            # JOIN task - use bottom-up approach (column matching first)
            strategy.name = "bottom-up"
            strategy.use_metadata = True
            strategy.use_vector = optimization_config.use_vector_search if optimization_config else True
            strategy.use_llm = optimization_config.use_llm_verification if optimization_config else True
            strategy.top_k = 50  # Get more candidates for JOIN
            strategy.confidence_threshold = 0.7  # Higher threshold for JOIN
            
            self.logger.info("Selected BOTTOM-UP strategy for JOIN task")
            self.logger.info("  - Focus on column-level matching")
            self.logger.info("  - Higher confidence threshold (0.7)")
            
        elif query_task and query_task.task_type == 'union':
            # UNION task - use top-down approach (table similarity first)
            strategy.name = "top-down"
            strategy.use_metadata = True
            strategy.use_vector = optimization_config.use_vector_search if optimization_config else True
            strategy.use_llm = optimization_config.use_llm_verification if optimization_config else True
            strategy.top_k = 30  # Fewer candidates needed for UNION
            strategy.confidence_threshold = 0.6  # Lower threshold for UNION
            
            self.logger.info("Selected TOP-DOWN strategy for UNION task")
            self.logger.info("  - Focus on table-level similarity")
            self.logger.info("  - Lower confidence threshold (0.6)")
            
        else:
            # Unknown or mixed task - use hybrid approach
            strategy.name = "hybrid"
            strategy.use_metadata = True
            strategy.use_vector = True
            strategy.use_llm = True
            strategy.top_k = 40
            strategy.confidence_threshold = 0.65
            
            self.logger.info("Selected HYBRID strategy for general task")
            self.logger.info("  - Balanced approach")
            self.logger.info("  - Medium confidence threshold (0.65)")
        
        # Adjust based on optimization config
        if optimization_config:
            if optimization_config.batch_size < 10:
                # Small batch size, reduce candidates
                strategy.top_k = min(strategy.top_k, 30)
            
            if not optimization_config.use_vector_search:
                strategy.use_vector = False
                self.logger.info("  - Vector search disabled by optimizer")
        
        # Log final strategy
        self.logger.info(f"Final execution strategy:")
        self.logger.info(f"  - Strategy: {strategy.name}")
        self.logger.info(f"  - Use metadata: {strategy.use_metadata}")
        self.logger.info(f"  - Use vector: {strategy.use_vector}")
        self.logger.info(f"  - Use LLM: {strategy.use_llm}")
        self.logger.info(f"  - Top-K: {strategy.top_k}")
        self.logger.info(f"  - Confidence threshold: {strategy.confidence_threshold}")
        
        # Update state
        state['strategy'] = strategy
        
        # Set skip_matcher flag if we shouldn't use LLM
        if not strategy.use_llm:
            state['skip_matcher'] = True
            self.logger.info("Will skip MatcherAgent (LLM disabled)")
        
        return state
    
    def validate_input(self, state: WorkflowState) -> bool:
        """
        Validate required inputs
        """
        if 'query_task' not in state or state['query_task'] is None:
            self.logger.error("Missing query_task in state")
            return False
        
        return True