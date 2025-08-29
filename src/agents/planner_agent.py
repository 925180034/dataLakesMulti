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
        
    def identify_query_type(self, query_task) -> str:
        """
        Identify whether this is a NLCTables query or traditional WebTables query
        
        Args:
            query_task: The query task object
            
        Returns:
            'nlctables', 'webtables', or 'unknown'
        """
        if not query_task:
            return 'unknown'
            
        # Check for NLCTables indicators
        # 1. Check if query_id starts with 'nlc_'
        if hasattr(query_task, 'query_id') and str(query_task.query_id).startswith('nlc_'):
            self.logger.info("Identified NLCTables query by query_id prefix")
            return 'nlctables'
            
        # 2. Check if we have query_text and features fields with actual values (NLCTables specific)
        if hasattr(query_task, 'query_text') and hasattr(query_task, 'features'):
            if query_task.query_text is not None and query_task.features is not None:
                self.logger.info("Identified NLCTables query by query_text and features")
                return 'nlctables'
            
        # 3. Check if the query dict contains NLCTables fields with actual values
        if hasattr(query_task, '__dict__'):
            task_dict = query_task.__dict__
            if 'query_text' in task_dict and 'features' in task_dict:
                if task_dict['query_text'] is not None and task_dict['features'] is not None:
                    self.logger.info("Identified NLCTables query by dict fields")
                    return 'nlctables'
                
        # Default to WebTables for traditional queries
        if hasattr(query_task, 'table_name') or hasattr(query_task, 'query_table'):
            return 'webtables'
            
        return 'unknown'
    
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
        
        # Identify query type
        query_type = self.identify_query_type(query_task)
        state['query_type'] = query_type  # Store for other agents to use
        self.logger.info(f"Query type identified: {query_type}")
        
        # Initialize strategy
        strategy = ExecutionStrategy(name="hybrid")
        
        # For NLCTables, use specialized strategy
        if query_type == 'nlctables':
            strategy.name = "semantic_search"
            strategy.use_metadata = True  # Still use for keyword matching
            strategy.use_vector = True    # Essential for semantic search
            strategy.use_llm = True       # Essential for NL condition verification
            strategy.top_k = 150          # Increase candidates for semantic search
            strategy.confidence_threshold = 0.5  # Lower threshold for NL matching
            strategy.search_mode = 'semantic'  # New field for search mode
            
            self.logger.info("Selected SEMANTIC_SEARCH strategy for NLCTables query")
            self.logger.info("  - Using natural language understanding")
            self.logger.info("  - Prioritizing semantic similarity over structure")
            
            # Store NLCTables specific info in state
            if hasattr(query_task, 'features'):
                state['nl_features'] = query_task.features
            if hasattr(query_task, 'query_text'):
                state['query_text'] = query_task.query_text
                
        else:
            # First try LLM-based planning for traditional queries
            llm_strategy = self._get_llm_strategy(query_task, optimization_config)
            
            if llm_strategy and 'strategy_name' in llm_strategy:
                # Apply LLM recommendations
                strategy.name = llm_strategy.get('strategy_name', 'hybrid')
                # 强制使用100个候选，忽略LLM建议的值
                strategy.top_k = 100  # 固定为100，不使用LLM的建议
                strategy.confidence_threshold = llm_strategy.get('confidence_threshold', 0.65)
                strategy.use_metadata = True
                strategy.use_vector = optimization_config.get('use_vector_search', True) if optimization_config else True
                strategy.use_llm = optimization_config.get('use_llm_verification', True) if optimization_config else True
                
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
            
            # Enhanced strategy selection with hybrid mode support
            enable_hybrid = optimization_config.get('enable_hybrid_search', False) if optimization_config else False
            
            if query_task and query_task.get('task_type') == 'join':
                # JOIN task - use bottom-up approach
                if enable_hybrid:
                    # Hybrid mode: combine structure and semantic search
                    strategy.name = "hybrid_enhanced"
                    strategy.use_metadata = True
                    strategy.use_vector = True  # Always use vector in hybrid mode
                    strategy.use_llm = True     # Always use LLM in hybrid mode
                    strategy.top_k = 120        # More candidates for hybrid
                    strategy.confidence_threshold = 0.65
                    strategy.search_mode = 'hybrid'  # Both structural and semantic
                    
                    self.logger.info("Selected HYBRID_ENHANCED strategy for JOIN task")
                    self.logger.info("  - Combining structural and semantic matching")
                else:
                    strategy.name = "bottom-up"
                    strategy.use_metadata = True
                    strategy.use_vector = optimization_config.get('use_vector_search', True) if optimization_config else True
                    strategy.use_llm = optimization_config.get('use_llm_verification', True) if optimization_config else True
                    strategy.top_k = 100  # 增加候选数量
                    strategy.confidence_threshold = 0.7
                    strategy.search_mode = 'structural'  # Traditional structural matching
                    
                    self.logger.info("Selected BOTTOM-UP strategy for JOIN task (rule-based)")
                
            elif query_task and query_task.get('task_type') == 'union':
                # UNION task - use top-down approach
                strategy.name = "top-down"
                strategy.use_metadata = True
                strategy.use_vector = optimization_config.get('use_vector_search', True) if optimization_config else True
                strategy.use_llm = optimization_config.get('use_llm_verification', True) if optimization_config else True
                strategy.top_k = 100  # 增加候选数量
                strategy.confidence_threshold = 0.6
                
                self.logger.info("Selected TOP-DOWN strategy for UNION task (rule-based)")
                
            else:
                # Unknown or mixed task - use hybrid approach
                strategy.name = "hybrid"
                strategy.use_metadata = True
                strategy.use_vector = True
                strategy.use_llm = True
                strategy.top_k = 100  # 增加候选数量
                strategy.confidence_threshold = 0.65
                
                self.logger.info("Selected HYBRID strategy for general task (rule-based)")
        
        # Adjust based on optimization config
        if optimization_config:
            # 始终保持100个候选，不根据batch_size调整
            strategy.top_k = 100  # 固定为100，不调整
            
            if optimization_config and not optimization_config.get('use_vector_search', True):
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
            task_type=query_task.get('task_type', 'unknown'),
            query_table=query_task.get('table_name', 'Unknown'),
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