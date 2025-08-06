"""
自适应工作流 - 根据任务类型动态调整参数
JOIN任务使用精确匹配，UNION任务使用宽松匹配
"""

import logging
from src.core.models import TaskStrategy
from src.core.ultra_optimized_workflow import UltraOptimizedWorkflow
from src.core.relaxed_workflow import RelaxedOptimizedWorkflow

logger = logging.getLogger(__name__)


class AdaptiveWorkflow:
    """自适应工作流 - 根据任务类型选择最优策略"""
    
    def __init__(self, task_type: str = None):
        """
        Args:
            task_type: "join" 或 "union"，如果为None则从state中判断
        """
        self.task_type = task_type
        
        if task_type == "join":
            # JOIN任务：使用原版精确匹配
            logger.info("Using UltraOptimizedWorkflow for JOIN task (precise matching)")
            self.workflow = UltraOptimizedWorkflow()
            # JOIN特定优化
            self.workflow.max_metadata_candidates = 60  # 适度增加
            self.workflow.max_vector_candidates = 12    # 保持较少以提速
            self.workflow.max_llm_candidates = 3        # 快速验证
            self.workflow.early_stop_threshold = 0.88   # 较容易触发
            
        elif task_type == "union":
            # UNION任务：使用放松版提高召回率
            logger.info("Using RelaxedOptimizedWorkflow for UNION task (high recall)")
            self.workflow = RelaxedOptimizedWorkflow()
            # UNION已经在RelaxedOptimizedWorkflow中优化好了
            
        else:
            # 默认使用平衡版本
            logger.info("Using balanced UltraOptimizedWorkflow (auto-detect mode)")
            self.workflow = UltraOptimizedWorkflow()
            self.workflow.max_metadata_candidates = 70
            self.workflow.max_vector_candidates = 15
            self.workflow.max_llm_candidates = 4
        
        # 共同设置
        self.enable_llm_matching = False
        self.max_metadata_candidates = self.workflow.max_metadata_candidates
        self.max_vector_candidates = self.workflow.max_vector_candidates
        self.max_llm_candidates = self.workflow.max_llm_candidates
    
    async def initialize(self, all_tables):
        """初始化工作流"""
        await self.workflow.initialize(all_tables)
    
    async def run_optimized(self, state, all_table_names, ground_truth=None):
        """运行优化流程，根据state中的策略动态调整"""
        
        # 如果没有预设task_type，从state中检测
        if self.task_type is None and hasattr(state, 'strategy'):
            if state.strategy == TaskStrategy.BOTTOM_UP:
                # BOTTOM_UP通常用于JOIN
                if not isinstance(self.workflow, UltraOptimizedWorkflow):
                    logger.info("Detected JOIN task, switching to precise matching")
                    self.workflow = UltraOptimizedWorkflow()
                    self.workflow.max_metadata_candidates = 60
                    self.workflow.max_vector_candidates = 12
                    await self.workflow.initialize(all_table_names)
                    
            elif state.strategy == TaskStrategy.TOP_DOWN:
                # TOP_DOWN通常用于UNION
                if not isinstance(self.workflow, RelaxedOptimizedWorkflow):
                    logger.info("Detected UNION task, switching to high recall matching")
                    self.workflow = RelaxedOptimizedWorkflow()
                    await self.workflow.initialize(all_table_names)
        
        # 设置LLM匹配
        self.workflow.enable_llm_matching = self.enable_llm_matching
        
        # 运行工作流
        return await self.workflow.run_optimized(state, all_table_names, ground_truth)


def create_adaptive_workflow(task_type: str = None):
    """创建自适应工作流实例
    
    Args:
        task_type: "join" 或 "union"，如果为None则自动检测
    """
    return AdaptiveWorkflow(task_type)