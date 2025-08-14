"""
PlannerAgent - Strategy selection and execution planning with LLM intelligence
"""
import asyncio
import json
from src.agents.base_agent import BaseAgent
from src.core.state import WorkflowState, ExecutionStrategy
from src.config.prompts import get_agent_prompt, format_user_prompt


class PlannerAgent(BaseAgent):
    """
    Planner Agent responsible for determining the execution strategy using LLM
    """
    
    def __init__(self):
        super().__init__(
            name="PlannerAgent",
            description="Selects optimal execution strategy based on task type and table characteristics",
            use_llm=True  # Enable LLM for intelligent planning
        )
        
        # Use centralized prompt from config
        self.system_prompt = get_agent_prompt("PlannerAgent", "system_prompt")
        
    def process(self, state: WorkflowState) -> WorkflowState:
        """
        Determine execution strategy using LLM intelligence
        
        Args:
            state: Current workflow state
            
        Returns:
            Updated state with execution strategy
        """
        self.logger.info("Planning execution strategy with LLM")
        
        # Get query task and optimization config
        query_task = state.get('query_task')
        optimization_config = state.get('optimization_config')
        
        # Initialize strategy
        strategy = ExecutionStrategy(name="hybrid")
        
        # First try LLM-based planning
        llm_strategy = self._get_llm_strategy(query_task, optimization_config)
        
        if llm_strategy and 'strategy_name' in llm_strategy:
            # Apply LLM recommendations
            strategy.name = llm_strategy.get('strategy_name', 'hybrid')
            strategy.top_k = llm_strategy.get('top_k', 40)
            strategy.confidence_threshold = llm_strategy.get('confidence_threshold', 0.65)
            strategy.use_metadata = True
            strategy.use_vector = optimization_config.use_vector_search if optimization_config else True
            strategy.use_llm = optimization_config.use_llm_verification if optimization_config else True
            
            self.logger.info(f"LLM Strategy Selected: {strategy.name}")
            self.logger.info(f"LLM Reasoning: {llm_strategy.get('reasoning', 'No reasoning provided')}")
            
            # Log optimization tips if provided
            tips = llm_strategy.get('optimization_tips', [])
            if tips:
                for tip in tips:
                    self.logger.info(f"  Optimization Tip: {tip}")
        else:
            # Fallback to rule-based strategy if LLM fails
            self.logger.info("Falling back to rule-based strategy selection")
            
            if query_task and query_task.task_type == 'join':
                # JOIN task - use bottom-up approach
                strategy.name = "bottom-up"
                strategy.use_metadata = True
                strategy.use_vector = optimization_config.use_vector_search if optimization_config else True
                strategy.use_llm = optimization_config.use_llm_verification if optimization_config else True
                strategy.top_k = 50
                strategy.confidence_threshold = 0.7
                
                self.logger.info("Selected BOTTOM-UP strategy for JOIN task (rule-based)")
                
            elif query_task and query_task.task_type == 'union':
                # UNION task - use top-down approach
                strategy.name = "top-down"
                strategy.use_metadata = True
                strategy.use_vector = optimization_config.use_vector_search if optimization_config else True
                strategy.use_llm = optimization_config.use_llm_verification if optimization_config else True
                strategy.top_k = 30
                strategy.confidence_threshold = 0.6
                
                self.logger.info("Selected TOP-DOWN strategy for UNION task (rule-based)")
                
            else:
                # Unknown or mixed task - use hybrid approach
                strategy.name = "hybrid"
                strategy.use_metadata = True
                strategy.use_vector = True
                strategy.use_llm = True
                strategy.top_k = 40
                strategy.confidence_threshold = 0.65
                
                self.logger.info("Selected HYBRID strategy for general task (rule-based)")
        
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
        
        return state
    
    def _get_llm_strategy(self, query_task, optimization_config) -> dict:
        """Use LLM to determine optimal strategy"""
        if not query_task:
            return {}
        
        # Build prompt with context
        data_size = len(query_task.all_tables) if hasattr(query_task, 'all_tables') else 'Unknown'
        
        # Use centralized user prompt template
        prompt = format_user_prompt(
            "PlannerAgent",
            task_type=query_task.task_type,
            query_table=getattr(query_task, 'table_name', 'Unknown'),
            table_structure='Unknown',  # Could extract if needed
            data_size=data_size,
            performance_req='balanced'
        )
        
        try:
            # Handle async call in sync context
            loop = None
            try:
                loop = asyncio.get_running_loop()
                # We're in an async context, but called from sync
                # Create a new event loop in thread
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(asyncio.run, self.call_llm_json(prompt, self.system_prompt))
                    response = future.result(timeout=10)
            except RuntimeError:
                # No running loop, we can create one
                response = asyncio.run(self.call_llm_json(prompt, self.system_prompt))
            
            return response
        except Exception as e:
            self.logger.warning(f"LLM strategy planning failed: {e}, using rule-based fallback")
            return {}
    
    def validate_input(self, state: WorkflowState) -> bool:
        """
        Validate required inputs
        """
        if 'query_task' not in state or state['query_task'] is None:
            self.logger.error("Missing query_task in state")
            return False
        
        return True