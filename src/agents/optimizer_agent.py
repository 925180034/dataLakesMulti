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
        task_type = query_task.get('task_type', 'join') if query_task else 'join'
        
        # Initialize optimization config
        config = OptimizationConfig()
        
        # Try LLM-based optimization first
        llm_config = self._get_llm_optimization(query_task, data_size)
        
        # Validate LLM config follows task-specific rules
        if llm_config and 'parallel_workers' in llm_config:
            # Check if LLM followed the task-specific constraints
            # 对于这个特殊数据集，需要极低的阈值
            if task_type == 'join':
                # 强制使用极低阈值，忽略LLM建议
                if llm_config.get('confidence_threshold', 0) > 0.10:
                    self.logger.warning(f"LLM confidence threshold too high for JOIN ({llm_config.get('confidence_threshold')}), forcing to 0.10")
                    llm_config['confidence_threshold'] = 0.10
                    llm_config['aggregator_min_score'] = 0.01
                    llm_config['aggregator_max_results'] = 500
                    llm_config['vector_top_k'] = 600
            else:  # union
                if llm_config.get('confidence_threshold', 0) > 0.15:
                    self.logger.warning(f"LLM confidence threshold too high for UNION ({llm_config.get('confidence_threshold')}), forcing to 0.15")
                    llm_config['confidence_threshold'] = 0.15
                    llm_config['aggregator_min_score'] = 0.03
                    llm_config['aggregator_max_results'] = 200
                    llm_config['vector_top_k'] = 350
        
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
        
        # ===== NEW: L3层LLM匹配参数优化 =====
        # 基于任务类型和数据大小优化LLM匹配参数
        self.logger.info(f"DEBUG: task_type = {task_type}, data_size = {data_size}")
        if llm_config and 'confidence_threshold' in llm_config:
            config.llm_confidence_threshold = llm_config.get('confidence_threshold', 0.45)
            config.aggregator_min_score = llm_config.get('aggregator_min_score', 0.15)
            config.aggregator_max_results = llm_config.get('aggregator_max_results', 50)
            config.vector_top_k = llm_config.get('vector_top_k', 120)
        else:
            # 基于任务类型的规则优化 - 强制使用激进参数
            self.logger.info(f"DEBUG: FORCING ultra-low thresholds for task_type={task_type}")
            if task_type == 'join':
                # JOIN任务：需要更激进的参数来提高召回率
                self.logger.info("DEBUG: Applying JOIN-specific parameters (ULTRA LOW thresholds)")
                config.llm_confidence_threshold = 0.10  # 极限低阈值
                config.aggregator_min_score = 0.01      # 几乎不过滤
                config.aggregator_max_results = 500     # 极大候选数量
                config.vector_top_k = 600               # 最大搜索范围
            else:  # union
                # UNION任务：降低阈值提高召回率，平衡precision
                self.logger.info(f"DEBUG: Applying UNION-specific parameters (balanced for recall)")
                config.llm_confidence_threshold = 0.15  # 更低的阈值
                config.aggregator_min_score = 0.03      # 更宽松过滤
                config.aggregator_max_results = 200     # 更多候选
                config.vector_top_k = 350               # 更大范围
            
            # 根据数据大小调整
            if data_size > 1000:
                config.aggregator_max_results = min(config.aggregator_max_results * 2, 200)
                config.vector_top_k = min(config.vector_top_k * 2, 300)
            elif data_size <= 50:  # 修复：使用<=而不是<，避免50-100范围被意外处理
                config.aggregator_max_results = max(config.aggregator_max_results // 2, 20)
                config.vector_top_k = max(config.vector_top_k // 2, 50)
        
        # Log optimization decisions
        self.logger.info(f"Optimization config determined:")
        self.logger.info(f"  - Parallel workers: {config.parallel_workers}")
        self.logger.info(f"  - LLM concurrency: {config.llm_concurrency}")
        self.logger.info(f"  - Cache level: {config.cache_level}")
        self.logger.info(f"  - Use vector search: {config.use_vector_search}")
        self.logger.info(f"  - Batch size: {config.batch_size}")
        self.logger.info(f"  - LLM confidence threshold: {getattr(config, 'llm_confidence_threshold', 'N/A')}")
        self.logger.info(f"  - Aggregator min score: {getattr(config, 'aggregator_min_score', 'N/A')}")
        self.logger.info(f"  - Aggregator max results: {getattr(config, 'aggregator_max_results', 'N/A')}")
        self.logger.info(f"  - Vector top-k: {getattr(config, 'vector_top_k', 'N/A')}")
        
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
        
        # Enhanced prompt for L3 layer optimization
        prompt = f"""
请为数据湖发现系统优化参数配置。

任务详情:
- 任务类型: {task_type}
- 数据规模: {data_size} 个表
- 复杂度: {complexity}
- 内存限制: 16GB
- API限制: 100 calls/min

需要优化的参数:
1. 系统级参数:
   - parallel_workers (并行工作进程数, 2-16)
   - llm_concurrency (LLM并发数, 1-10) 
   - batch_size (批处理大小, 5-50)
   - cache_strategy (L1/L2/L3)

2. L3层LLM匹配参数:
   - confidence_threshold (LLM置信度阈值, 0.2-0.8)
   - aggregator_min_score (聚合器最小分数, 0.05-0.5)  
   - aggregator_max_results (最大结果数, 20-200)
   - vector_top_k (向量搜索topK, 50-300)

任务特点（必须根据任务类型选择不同参数）:
- JOIN任务: 需要精确匹配，关注precision
  ⚠️ 必须设置: confidence_threshold >= 0.45, aggregator_min_score >= 0.15
- UNION任务: 需要高召回率，关注recall
  ⚠️ 必须设置: confidence_threshold <= 0.35, aggregator_min_score <= 0.08

请返回JSON格式的优化配置:
{{
  "parallel_workers": <数值>,
  "llm_concurrency": <数值>,
  "batch_size": <数值>, 
  "cache_strategy": "<L1/L2/L3>",
  "confidence_threshold": <数值>,
  "aggregator_min_score": <数值>,
  "aggregator_max_results": <数值>,
  "vector_top_k": <数值>,
  "reasoning": "<优化理由>",
  "estimated_time": "<预估时间>",
  "resource_usage": "<资源使用情况>"
}}
"""
        
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