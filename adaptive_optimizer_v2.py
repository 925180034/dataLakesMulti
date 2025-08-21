"""
Enhanced Adaptive Optimizer with Intra-Batch Dynamic Optimization
批次内动态优化器 - 根据实时反馈动态调整参数
"""
import logging
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
import numpy as np
from collections import deque

logger = logging.getLogger(__name__)


@dataclass
class DynamicOptimizationState:
    """动态优化状态，跟踪实时性能"""
    # 性能指标历史（使用滑动窗口）
    precision_history: deque = field(default_factory=lambda: deque(maxlen=10))
    recall_history: deque = field(default_factory=lambda: deque(maxlen=10))
    f1_history: deque = field(default_factory=lambda: deque(maxlen=10))
    query_time_history: deque = field(default_factory=lambda: deque(maxlen=10))
    
    # 当前参数
    llm_confidence_threshold: float = 0.3
    aggregator_min_score: float = 0.1
    aggregator_max_results: int = 100
    vector_top_k: int = 200
    
    # 参数调整历史
    threshold_adjustments: List[float] = field(default_factory=list)
    
    # 统计信息
    queries_processed: int = 0
    last_adjustment_query: int = 0
    total_adjustments: int = 0


class IntraBatchOptimizer:
    """批次内动态优化器"""
    
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # 为JOIN和UNION维护独立的优化状态
        self.states = {
            'join': DynamicOptimizationState(),
            'union': DynamicOptimizationState()
        }
        
        # 优化策略参数
        self.adjustment_interval = 5  # 每5个查询评估一次
        self.min_samples = 3  # 最少需要3个样本才开始调整
        
        # 目标性能指标
        self.targets = {
            'join': {'precision': 0.3, 'recall': 0.4, 'f1': 0.35},
            'union': {'precision': 0.5, 'recall': 0.35, 'f1': 0.4}
        }
        
        # 调整步长
        self.adjustment_steps = {
            'small': 0.02,   # 微调
            'medium': 0.05,  # 中等调整
            'large': 0.10    # 大幅调整
        }
        
    def initialize_batch(self, task_type: str, data_size: int):
        """初始化批次，设置初始参数"""
        state = self.states[task_type]
        
        # 根据任务类型设置初始参数
        if task_type == 'join':
            # JOIN: 激进初始参数
            state.llm_confidence_threshold = 0.15
            state.aggregator_min_score = 0.02
            state.aggregator_max_results = 400
            state.vector_top_k = 500
        else:  # union
            # UNION: 平衡初始参数
            state.llm_confidence_threshold = 0.20
            state.aggregator_min_score = 0.05
            state.aggregator_max_results = 200
            state.vector_top_k = 300
            
        # 重置统计
        state.queries_processed = 0
        state.last_adjustment_query = 0
        state.total_adjustments = 0
        
        self.logger.info(f"Initialized {task_type.upper()} batch with dynamic optimization")
        self.logger.info(f"  Initial threshold: {state.llm_confidence_threshold}")
        self.logger.info(f"  Initial candidates: {state.aggregator_max_results}")
        
    def should_adjust(self, task_type: str) -> bool:
        """判断是否需要调整参数"""
        state = self.states[task_type]
        
        # 检查是否有足够的样本
        if len(state.precision_history) < self.min_samples:
            return False
            
        # 检查是否到了调整间隔
        queries_since_last = state.queries_processed - state.last_adjustment_query
        if queries_since_last < self.adjustment_interval:
            return False
            
        return True
        
    def calculate_adjustment(self, task_type: str) -> Dict[str, float]:
        """计算参数调整量"""
        state = self.states[task_type]
        targets = self.targets[task_type]
        
        # 计算最近的平均性能
        avg_precision = np.mean(state.precision_history)
        avg_recall = np.mean(state.recall_history)
        avg_f1 = np.mean(state.f1_history)
        
        adjustments = {}
        
        # 分析性能差距
        precision_gap = targets['precision'] - avg_precision
        recall_gap = targets['recall'] - avg_recall
        f1_gap = targets['f1'] - avg_f1
        
        self.logger.info(f"Performance gaps for {task_type}:")
        self.logger.info(f"  Precision: {avg_precision:.3f} (gap: {precision_gap:+.3f})")
        self.logger.info(f"  Recall: {avg_recall:.3f} (gap: {recall_gap:+.3f})")
        self.logger.info(f"  F1: {avg_f1:.3f} (gap: {f1_gap:+.3f})")
        
        # 决定调整策略
        if recall_gap > 0.15:
            # 召回率太低，需要大幅降低阈值
            adjustments['threshold_delta'] = -self.adjustment_steps['large']
            adjustments['candidates_delta'] = 100
            adjustments['top_k_delta'] = 150
            self.logger.info("  Strategy: AGGRESSIVE - Boost recall significantly")
            
        elif recall_gap > 0.05:
            # 召回率偏低，中等调整
            adjustments['threshold_delta'] = -self.adjustment_steps['medium']
            adjustments['candidates_delta'] = 50
            adjustments['top_k_delta'] = 75
            self.logger.info("  Strategy: MODERATE - Improve recall")
            
        elif precision_gap > 0.15 and recall_gap < -0.05:
            # 精度太低但召回率过高，需要提高阈值
            adjustments['threshold_delta'] = self.adjustment_steps['medium']
            adjustments['candidates_delta'] = -50
            adjustments['top_k_delta'] = -75
            self.logger.info("  Strategy: TIGHTEN - Improve precision")
            
        elif abs(f1_gap) > 0.02:
            # F1需要微调
            if f1_gap > 0:
                # F1偏低，稍微降低阈值
                adjustments['threshold_delta'] = -self.adjustment_steps['small']
                adjustments['candidates_delta'] = 25
                adjustments['top_k_delta'] = 30
            else:
                # F1偏高（少见），稍微提高阈值
                adjustments['threshold_delta'] = self.adjustment_steps['small']
                adjustments['candidates_delta'] = -10
                adjustments['top_k_delta'] = -15
            self.logger.info("  Strategy: FINE-TUNE - Optimize F1")
        else:
            # 性能接近目标，不调整
            self.logger.info("  Strategy: MAINTAIN - Performance near target")
            
        return adjustments
        
    def apply_adjustments(self, task_type: str, adjustments: Dict[str, float]):
        """应用参数调整"""
        if not adjustments:
            return
            
        state = self.states[task_type]
        
        # 应用阈值调整
        if 'threshold_delta' in adjustments:
            old_threshold = state.llm_confidence_threshold
            state.llm_confidence_threshold += adjustments['threshold_delta']
            # 限制范围
            state.llm_confidence_threshold = max(0.05, min(0.5, state.llm_confidence_threshold))
            
            # 同步调整最小分数
            state.aggregator_min_score = state.llm_confidence_threshold * 0.2
            
            self.logger.info(f"  Adjusted threshold: {old_threshold:.3f} → {state.llm_confidence_threshold:.3f}")
            
        # 应用候选数调整
        if 'candidates_delta' in adjustments:
            old_candidates = state.aggregator_max_results
            state.aggregator_max_results += int(adjustments['candidates_delta'])
            # 限制范围
            state.aggregator_max_results = max(50, min(800, state.aggregator_max_results))
            
            self.logger.info(f"  Adjusted candidates: {old_candidates} → {state.aggregator_max_results}")
            
        # 应用TopK调整
        if 'top_k_delta' in adjustments:
            old_top_k = state.vector_top_k
            state.vector_top_k += int(adjustments['top_k_delta'])
            # 限制范围
            state.vector_top_k = max(100, min(1000, state.vector_top_k))
            
            self.logger.info(f"  Adjusted top_k: {old_top_k} → {state.vector_top_k}")
            
        # 更新统计
        state.last_adjustment_query = state.queries_processed
        state.total_adjustments += 1
        state.threshold_adjustments.append(state.llm_confidence_threshold)
        
    def update_performance(self, task_type: str, precision: float, recall: float, f1: float, query_time: float):
        """更新性能指标"""
        state = self.states[task_type]
        
        # 添加到历史
        state.precision_history.append(precision)
        state.recall_history.append(recall)
        state.f1_history.append(f1)
        state.query_time_history.append(query_time)
        state.queries_processed += 1
        
        # 检查是否需要调整
        if self.should_adjust(task_type):
            self.logger.info(f"\n🔧 Dynamic adjustment triggered for {task_type} at query {state.queries_processed}")
            adjustments = self.calculate_adjustment(task_type)
            self.apply_adjustments(task_type, adjustments)
            
    def get_current_params(self, task_type: str) -> Dict[str, any]:
        """获取当前参数"""
        state = self.states[task_type]
        return {
            'llm_confidence_threshold': state.llm_confidence_threshold,
            'aggregator_min_score': state.aggregator_min_score,
            'aggregator_max_results': state.aggregator_max_results,
            'vector_top_k': state.vector_top_k,
            'queries_processed': state.queries_processed,
            'total_adjustments': state.total_adjustments
        }
        
    def get_optimization_summary(self, task_type: str) -> str:
        """获取优化总结"""
        state = self.states[task_type]
        
        if len(state.precision_history) == 0:
            return f"No data yet for {task_type}"
            
        summary = f"""
╔══════════════════════════════════════════════════════════════╗
║  {task_type.upper()} Dynamic Optimization Summary              ║
╠══════════════════════════════════════════════════════════════╣
║  Queries Processed: {state.queries_processed:>5}                              ║
║  Total Adjustments: {state.total_adjustments:>5}                              ║
║                                                               ║
║  Current Parameters:                                          ║
║    • Confidence Threshold: {state.llm_confidence_threshold:>6.3f}                     ║
║    • Min Score: {state.aggregator_min_score:>6.3f}                               ║
║    • Max Results: {state.aggregator_max_results:>4}                              ║
║    • Vector TopK: {state.vector_top_k:>4}                              ║
║                                                               ║
║  Recent Performance (last {len(state.precision_history)} queries):                      ║
║    • Avg Precision: {np.mean(state.precision_history):>6.3f}                       ║
║    • Avg Recall: {np.mean(state.recall_history):>6.3f}                          ║
║    • Avg F1: {np.mean(state.f1_history):>6.3f}                              ║
║    • Avg Query Time: {np.mean(state.query_time_history):>5.2f}s                     ║
╚══════════════════════════════════════════════════════════════╝
"""
        return summary


# 使用示例
def demo_intra_batch_optimization():
    """演示批次内动态优化"""
    optimizer = IntraBatchOptimizer()
    
    # 初始化JOIN批次
    optimizer.initialize_batch('join', 100)
    
    # 模拟查询处理
    for i in range(20):
        # 模拟性能（随着优化逐渐改善）
        base_precision = 0.15 + i * 0.01
        base_recall = 0.20 + i * 0.015
        precision = min(0.4, base_precision + np.random.uniform(-0.05, 0.05))
        recall = min(0.5, base_recall + np.random.uniform(-0.05, 0.05))
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        query_time = 2.5 + np.random.uniform(-0.5, 0.5)
        
        # 更新性能
        optimizer.update_performance('join', precision, recall, f1, query_time)
        
        # 获取当前参数
        params = optimizer.get_current_params('join')
        
        if i % 5 == 4:  # 每5个查询输出一次
            print(f"\nQuery {i+1} - Current params:")
            print(f"  Threshold: {params['llm_confidence_threshold']:.3f}")
            print(f"  Candidates: {params['aggregator_max_results']}")
            print(f"  Adjustments: {params['total_adjustments']}")
    
    # 输出总结
    print(optimizer.get_optimization_summary('join'))


if __name__ == "__main__":
    demo_intra_batch_optimization()