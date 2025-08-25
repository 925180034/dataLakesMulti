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
        
        # ⭐ 更新：基于实际测试结果的现实目标
        # JOIN最高F1只能到0.12，UNION最高F1只能到0.31
        self.targets = {
            'join': {'precision': 0.20, 'recall': 0.25, 'f1': 0.20},   # 更现实的JOIN目标
            'union': {'precision': 0.35, 'recall': 0.30, 'f1': 0.32}   # 更现实的UNION目标
        }
        
        # 调整步长
        self.adjustment_steps = {
            'small': 0.02,   # 微调
            'medium': 0.05,  # 中等调整
            'large': 0.10    # 大幅调整
        }
        
        # ⭐ 新增：任务特定的优化特性
        self.task_features = {
            'join': {
                'foreign_key_detection': True,
                'relationship_analysis': True,
                'ignore_table_name': True,
                'boost_factors': {
                    'foreign_key_match': 1.5,
                    'semantic_relationship': 1.3,
                    'llm_high_confidence': 1.4
                }
            },
            'union': {
                'prefix_matching': True,
                'pattern_recognition': True,
                'same_source_detection': True,
                'boost_factors': {
                    'same_prefix': 2.0,
                    'same_pattern': 1.6,
                    'exact_name_match': 1.8
                }
            }
        }
        
    def initialize_batch(self, task_type: str, data_size: int):
        """初始化批次，设置初始参数"""
        state = self.states[task_type]
        
        # ⭐ 更新：基于实验结果优化 - L3不应该过滤而应该重排序
        if task_type == 'join':
            # JOIN: 关系推理优化 - 极低阈值，让LLM重排序而非过滤
            state.llm_confidence_threshold = 0.10  # 降低到0.10，避免过度过滤
            state.aggregator_min_score = 0.01      # 极低阈值，最大化召回
            state.aggregator_max_results = 500     # 增加候选数量
            state.vector_top_k = 600               # 增加向量搜索候选
        else:  # union
            # UNION: 模式匹配优化 - 适中阈值平衡精度和召回
            state.llm_confidence_threshold = 0.15  # 降低到0.15，提高召回率
            state.aggregator_min_score = 0.03      # 降低阈值
            state.aggregator_max_results = 200     # 增加候选数量
            state.vector_top_k = 350                # 增加向量搜索候选
            
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
            'total_adjustments': state.total_adjustments,
            'task_features': self.task_features.get(task_type, {})  # 包含任务特定特性
        }
    
    def apply_boost_factor(self, task_type: str, score: float, table1: str, table2: str) -> float:
        """应用任务特定的boost factor
        
        Args:
            task_type: 'join' 或 'union'
            score: 原始分数
            table1: 查询表名
            table2: 候选表名
            
        Returns:
            调整后的分数
        """
        boost_factors = self.task_features[task_type].get('boost_factors', {})
        
        if task_type == 'join':
            # JOIN特定boost
            if self._has_foreign_key_pattern(table1, table2):
                score *= boost_factors.get('foreign_key_match', 1.5)
                
        elif task_type == 'union':
            # UNION特定boost
            if self._has_same_prefix(table1, table2):
                score *= boost_factors.get('same_prefix', 2.0)
            elif self._has_same_pattern(table1, table2):
                score *= boost_factors.get('same_pattern', 1.6)
        
        return score
    
    def _has_foreign_key_pattern(self, table1: str, table2: str) -> bool:
        """检查是否有潜在的外键关系"""
        fk_patterns = ['_id', '_key', '_code', '_no']
        t1_lower = table1.lower()
        t2_lower = table2.lower()
        
        for pattern in fk_patterns:
            if pattern in t1_lower or pattern in t2_lower:
                base1 = t1_lower.replace(pattern, '')
                base2 = t2_lower.replace(pattern, '')
                if base1 in t2_lower or base2 in t1_lower:
                    return True
        return False
    
    def _has_same_prefix(self, table1: str, table2: str) -> bool:
        """检查是否有相同前缀"""
        def get_prefix(name):
            if '__' in name:
                return name.split('__')[0]
            elif '_' in name:
                parts = name.split('_')
                if len(parts) > 1:
                    return parts[0]
            return name[:min(10, len(name))]
        
        return get_prefix(table1) == get_prefix(table2)
    
    def _has_same_pattern(self, table1: str, table2: str) -> bool:
        """检查是否有相同模式"""
        import re
        
        # 提取模式（将数字替换为占位符）
        pattern1 = re.sub(r'\d+', '#', table1)
        pattern2 = re.sub(r'\d+', '#', table2)
        
        return pattern1 == pattern2
        
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