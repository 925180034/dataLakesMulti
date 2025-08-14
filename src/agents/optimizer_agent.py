"""
OptimizerAgent - System optimization and resource allocation
"""
import asyncio
import json
from typing import Dict, Any
from src.agents.base_agent import BaseAgent
from src.core.state import WorkflowState, OptimizationConfig
from src.config.prompts import get_agent_prompt, format_user_prompt


class OptimizerAgent(BaseAgent):
    """
    Optimizer Agent responsible for system configuration and resource allocation
    """
    
    def __init__(self):
        super().__init__(
            name="OptimizerAgent",
            description="Optimizes system configuration based on query complexity and data size",
            use_llm=True  # Enable LLM for intelligent optimization
        )
        
        # Use centralized prompt from config
        self.system_prompt = get_agent_prompt("OptimizerAgent", "system_prompt")
        
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
        
        # Try LLM-based optimization first
        llm_config = self._get_llm_optimization(query_task, data_size)
        
        if llm_config and 'parallel_workers' in llm_config:
            # Apply LLM recommendations
            config.parallel_workers = llm_config.get('parallel_workers', 8)
            config.llm_concurrency = llm_config.get('llm_concurrency', 3)
            config.batch_size = llm_config.get('batch_size', 10)
            
            # Parse cache strategy
            cache_strategy = llm_config.get('cache_strategy', 'L2')
            config.cache_level = cache_strategy
            
            self.logger.info(f"LLM Optimization Applied:")
            self.logger.info(f"  Reasoning: {llm_config.get('reasoning', 'No reasoning provided')}")
            self.logger.info(f"  Estimated Time: {llm_config.get('estimated_time', 'Unknown')}")
            self.logger.info(f"  Resource Usage: {llm_config.get('resource_usage', 'Unknown')}")
        else:
            # Fallback to rule-based optimization
            self.logger.info("Using rule-based optimization (LLM unavailable or failed)")
            
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
    
    def _get_llm_optimization(self, query_task, data_size: int) -> dict:
        """Use LLM to determine optimal configuration"""
        if not query_task:
            return {}
        
        # Determine complexity
        if data_size < 100:
            complexity = "simple"
        elif data_size < 500:
            complexity = "medium"
        elif data_size < 1000:
            complexity = "complex"
        else:
            complexity = "very complex"
        
        # Use centralized user prompt template
        prompt = format_user_prompt(
            "OptimizerAgent",
            task_type=query_task.task_type,
            data_size=data_size,
            complexity=complexity,
            memory_gb=16,  # Default assumption
            rate_limit=100  # Default API rate limit
        )
        
        try:
            # Handle async call in sync context
            try:
                loop = asyncio.get_running_loop()
                # We're in an async context, but called from sync
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(asyncio.run, self.call_llm_json(prompt, self.system_prompt))
                    response = future.result(timeout=10)
            except RuntimeError:
                # No running loop, we can create one
                response = asyncio.run(self.call_llm_json(prompt, self.system_prompt))
            
            return response
        except Exception as e:
            self.logger.warning(f"LLM optimization failed: {e}, using rule-based fallback")
            return {}
    
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