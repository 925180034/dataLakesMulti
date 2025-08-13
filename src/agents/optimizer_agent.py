"""
OptimizerAgent - System optimization and resource allocation
"""
from typing import Dict, Any
from src.agents.base_agent import BaseAgent
from src.core.state import WorkflowState, OptimizationConfig


class OptimizerAgent(BaseAgent):
    """
    Optimizer Agent responsible for system configuration and resource allocation
    """
    
    def __init__(self):
        super().__init__(
            name="OptimizerAgent",
            description="Optimizes system configuration based on query complexity and data size"
        )
        
    def process(self, state: WorkflowState) -> WorkflowState:
        """
        Determine optimal system configuration
        
        Args:
            state: Current workflow state
            
        Returns:
            Updated state with optimization_config
        """
        self.logger.info("Analyzing query complexity and determining optimal configuration")
        
        # Extract information for optimization decisions
        query_task = state.get('query_task')
        all_tables = state.get('all_tables', [])
        
        # Calculate complexity factors
        data_size = len(all_tables)
        task_type = query_task.task_type if query_task else 'join'
        
        # Initialize optimization config
        config = OptimizationConfig()
        
        # Determine parallel workers based on data size
        if data_size < 100:
            config.parallel_workers = 4
        elif data_size < 500:
            config.parallel_workers = 8
        elif data_size < 1000:
            config.parallel_workers = 12
        else:
            config.parallel_workers = 16
        
        # Determine LLM concurrency based on task type and data size (reduced to avoid rate limiting)
        if task_type == 'join':
            # JOIN tasks need more precise matching (reduced from 20 to 5)
            config.llm_concurrency = min(5, max(3, data_size // 100))
        else:
            # UNION tasks can be more aggressive (reduced from 20 to 3)
            config.llm_concurrency = min(3, max(2, data_size // 200))
        
        # Determine cache strategy
        if data_size > 1000:
            config.cache_level = "L3"  # Use all cache levels for large datasets
        elif data_size > 500:
            config.cache_level = "L2"  # Use memory and Redis cache
        else:
            config.cache_level = "L1"  # Use only memory cache
        
        # Determine which tools to use
        config.use_vector_search = data_size > 50  # Use vector search for larger datasets
        config.use_llm_verification = True  # Always use LLM for accuracy
        
        # Set batch size for processing
        config.batch_size = min(20, max(5, data_size // 100))
        
        # Log optimization decisions
        self.logger.info(f"Optimization config determined:")
        self.logger.info(f"  - Parallel workers: {config.parallel_workers}")
        self.logger.info(f"  - LLM concurrency: {config.llm_concurrency}")
        self.logger.info(f"  - Cache level: {config.cache_level}")
        self.logger.info(f"  - Use vector search: {config.use_vector_search}")
        self.logger.info(f"  - Batch size: {config.batch_size}")
        
        # Update state
        state['optimization_config'] = config
        
        # Set control flow flags
        state['should_use_llm'] = config.use_llm_verification
        
        return state
    
    def validate_input(self, state: WorkflowState) -> bool:
        """
        Validate required inputs
        """
        if 'query_task' not in state or state['query_task'] is None:
            self.logger.error("Missing query_task in state")
            return False
        
        if 'all_tables' not in state:
            self.logger.warning("Missing all_tables in state, using empty list")
            state['all_tables'] = []
        
        return True