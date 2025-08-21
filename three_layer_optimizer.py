"""
Three-Layer Complete Dynamic Optimizer
完整三层动态优化器 - 同时优化L1、L2、L3层参数
"""
import logging
import time
import numpy as np
from typing import Dict, List, Optional
from dataclasses import dataclass, field
from collections import deque

logger = logging.getLogger(__name__)


@dataclass
class ThreeLayerOptimizationState:
    """三层优化状态（覆盖所有层）"""
    # 性能历史
    precision_history: deque = field(default_factory=lambda: deque(maxlen=10))
    recall_history: deque = field(default_factory=lambda: deque(maxlen=10))
    f1_history: deque = field(default_factory=lambda: deque(maxlen=10))
    time_history: deque = field(default_factory=lambda: deque(maxlen=10))
    
    # L1层参数（元数据过滤）
    metadata_min_column_overlap: float = 0.2
    metadata_name_similarity: float = 0.3
    metadata_max_candidates: int = 500
    
    # L2层参数（向量搜索）
    vector_top_k: int = 500
    vector_similarity_threshold: float = 0.4
    vector_batch_size: int = 100
    
    # L3层参数（LLM验证）
    llm_confidence_threshold: float = 0.15
    llm_max_candidates: int = 100
    llm_batch_size: int = 20
    llm_rule_high_threshold: float = 0.7
    llm_rule_medium_threshold: float = 0.5
    llm_rule_low_threshold: float = 0.3
    
    # 聚合器参数
    aggregator_max_results: int = 100
    llm_concurrency: int = 3
    batch_size: int = 10
    
    # 优化控制
    adjustment_count: int = 0
    last_adjustment: float = time.time()
    improvement_trend: float = 0.0
    queries_processed: int = 0


class ThreeLayerOptimizer:
    """完整三层动态优化器"""
    
    def __init__(self):
        self.states = {
            'join': ThreeLayerOptimizationState(),
            'union': ThreeLayerOptimizationState()
        }
        self.adjustment_interval = 10  # 每10个查询调整一次
        self.query_count = 0
        
        logger.info("🎯 ThreeLayerOptimizer initialized for all three layers")
        
    def initialize_batch(self, task_type: str, data_size: int):
        """初始化批次参数（针对任务类型和所有三层）"""
        state = self.states[task_type]
        
        if task_type == 'union':
            # UNION任务：优化召回率
            logger.info("📊 UNION任务初始化 - 优化召回率")
            
            # L1层：放宽元数据过滤
            state.metadata_min_column_overlap = 0.1
            state.metadata_max_candidates = 800
            
            # L2层：增大向量搜索范围
            state.vector_top_k = 800
            state.vector_similarity_threshold = 0.3
            
            # L3层：极低的LLM阈值以提高召回率
            state.llm_confidence_threshold = 0.01  # 极低！
            state.llm_max_candidates = 150
            state.llm_rule_high_threshold = 0.5
            state.llm_rule_medium_threshold = 0.3
            state.llm_rule_low_threshold = 0.1
            state.aggregator_max_results = 150
            
        else:  # join
            # JOIN任务：平衡精确率和召回率
            logger.info("📊 JOIN任务初始化 - 平衡精确率和召回率")
            
            # L1层：标准参数
            state.metadata_min_column_overlap = 0.2
            state.metadata_max_candidates = 500
            
            # L2层：标准参数
            state.vector_top_k = 500
            state.vector_similarity_threshold = 0.4
            
            # L3层：平衡的阈值
            state.llm_confidence_threshold = 0.10
            state.llm_max_candidates = 100
            state.llm_rule_high_threshold = 0.7
            state.llm_rule_medium_threshold = 0.5
            state.llm_rule_low_threshold = 0.3
            state.aggregator_max_results = 100
            
        # 重置统计
        state.queries_processed = 0
        state.adjustment_count = 0
        self.query_count = 0
    
    def get_current_params(self, task_type: str) -> Dict:
        """获取当前所有三层的动态参数"""
        state = self.states[task_type]
        
        return {
            # L1层参数
            'metadata_min_column_overlap': state.metadata_min_column_overlap,
            'metadata_name_similarity': state.metadata_name_similarity,
            'metadata_max_candidates': state.metadata_max_candidates,
            
            # L2层参数
            'vector_top_k': state.vector_top_k,
            'vector_similarity_threshold': state.vector_similarity_threshold,
            'vector_batch_size': state.vector_batch_size,
            
            # L3层参数
            'llm_confidence_threshold': state.llm_confidence_threshold,
            'llm_max_candidates': state.llm_max_candidates,
            'llm_batch_size': state.llm_batch_size,
            'rule_high_threshold': state.llm_rule_high_threshold,
            'rule_medium_threshold': state.llm_rule_medium_threshold,
            'rule_low_threshold': state.llm_rule_low_threshold,
            
            # 聚合器参数
            'aggregator_max_results': state.aggregator_max_results,
            'llm_concurrency': state.llm_concurrency,
            'batch_size': state.batch_size,
            
            # 任务类型
            'task_type': task_type,
            'optimize_for_recall': task_type == 'union'
        }
    
    def update_performance(self, task_type: str, precision: float, recall: float, 
                          f1: float, query_time: float):
        """更新性能指标并决定是否调整参数"""
        state = self.states[task_type]
        self.query_count += 1
        state.queries_processed += 1
        
        # 记录性能
        state.precision_history.append(precision)
        state.recall_history.append(recall)
        state.f1_history.append(f1)
        state.time_history.append(query_time)
        
        # 检查是否需要调整
        if self.query_count % self.adjustment_interval == 0 and len(state.f1_history) >= 5:
            self._adjust_parameters(task_type)
    
    def _adjust_parameters(self, task_type: str):
        """调整所有三层的参数"""
        state = self.states[task_type]
        
        # 计算平均性能
        avg_precision = np.mean(state.precision_history)
        avg_recall = np.mean(state.recall_history)
        avg_f1 = np.mean(state.f1_history)
        
        logger.info(f"\n🔄 动态调整参数 (Query {self.query_count})")
        logger.info(f"  平均性能: P={avg_precision:.3f}, R={avg_recall:.3f}, F1={avg_f1:.3f}")
        
        # 计算性能差距
        precision_gap = 0.5 - avg_precision  # 目标精确率50%
        recall_gap = 0.4 - avg_recall        # 目标召回率40%
        
        if task_type == 'union':
            # UNION: 主要关注召回率
            if recall_gap > 0.15:  # 召回率太低
                logger.info("  📈 UNION召回率太低，放宽所有层的阈值")
                
                # L1层：放宽元数据过滤
                state.metadata_min_column_overlap = max(0.05, state.metadata_min_column_overlap - 0.05)
                state.metadata_max_candidates = min(1000, state.metadata_max_candidates + 100)
                
                # L2层：增加向量搜索结果
                state.vector_top_k = min(1000, state.vector_top_k + 150)
                state.vector_similarity_threshold = max(0.2, state.vector_similarity_threshold - 0.05)
                
                # L3层：大幅降低LLM阈值
                state.llm_confidence_threshold = max(0.001, state.llm_confidence_threshold - 0.02)
                state.llm_max_candidates = min(200, state.llm_max_candidates + 20)
                state.llm_rule_high_threshold = max(0.3, state.llm_rule_high_threshold - 0.1)
                state.llm_rule_medium_threshold = max(0.2, state.llm_rule_medium_threshold - 0.1)
                state.llm_rule_low_threshold = max(0.05, state.llm_rule_low_threshold - 0.05)
                
            elif avg_precision < 0.2:  # 精确率太低
                logger.info("  ⚠️ UNION精确率太低，稍微收紧阈值")
                
                # 微调参数
                state.llm_confidence_threshold = min(0.1, state.llm_confidence_threshold + 0.01)
                state.vector_similarity_threshold = min(0.5, state.vector_similarity_threshold + 0.02)
                
        else:  # join
            # JOIN: 平衡精确率和召回率
            if avg_f1 < 0.3:  # F1太低
                if recall_gap > precision_gap:
                    logger.info("  📈 JOIN召回率不足，放宽阈值")
                    
                    # L1层：放宽
                    state.metadata_min_column_overlap = max(0.1, state.metadata_min_column_overlap - 0.02)
                    state.metadata_max_candidates = min(700, state.metadata_max_candidates + 50)
                    
                    # L2层：增加候选
                    state.vector_top_k = min(700, state.vector_top_k + 100)
                    
                    # L3层：降低阈值
                    state.llm_confidence_threshold = max(0.05, state.llm_confidence_threshold - 0.02)
                    state.llm_max_candidates = min(150, state.llm_max_candidates + 10)
                    
                else:
                    logger.info("  📉 JOIN精确率不足，收紧阈值")
                    
                    # L1层：收紧
                    state.metadata_min_column_overlap = min(0.4, state.metadata_min_column_overlap + 0.02)
                    
                    # L2层：减少候选
                    state.vector_top_k = max(300, state.vector_top_k - 50)
                    
                    # L3层：提高阈值
                    state.llm_confidence_threshold = min(0.3, state.llm_confidence_threshold + 0.02)
        
        state.adjustment_count += 1
        state.last_adjustment = time.time()
        
        # 记录调整后的参数
        logger.info(f"  📊 调整后参数:")
        logger.info(f"    L1: overlap={state.metadata_min_column_overlap:.2f}, candidates={state.metadata_max_candidates}")
        logger.info(f"    L2: top_k={state.vector_top_k}, threshold={state.vector_similarity_threshold:.2f}")
        logger.info(f"    L3: confidence={state.llm_confidence_threshold:.3f}, candidates={state.llm_max_candidates}")
    
    def get_optimization_summary(self, task_type: str) -> str:
        """获取优化总结"""
        state = self.states[task_type]
        
        if len(state.f1_history) == 0:
            return "No optimization data available"
        
        summary = f"\n📊 动态优化总结 ({task_type.upper()}):\n"
        summary += f"  调整次数: {state.adjustment_count}\n"
        summary += f"  平均性能: P={np.mean(state.precision_history):.3f}, "
        summary += f"R={np.mean(state.recall_history):.3f}, "
        summary += f"F1={np.mean(state.f1_history):.3f}\n"
        summary += f"  最终参数:\n"
        summary += f"    L1: overlap={state.metadata_min_column_overlap:.2f}, candidates={state.metadata_max_candidates}\n"
        summary += f"    L2: top_k={state.vector_top_k}, threshold={state.vector_similarity_threshold:.2f}\n"
        summary += f"    L3: confidence={state.llm_confidence_threshold:.3f}, candidates={state.llm_max_candidates}\n"
        
        return summary