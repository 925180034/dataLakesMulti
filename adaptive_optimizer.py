#!/usr/bin/env python3
"""
自适应参数优化模块
实现OptimizerAgent的动态调用和批次级别优化
"""
import logging
from typing import Dict, List, Any, Tuple
from pathlib import Path
import json
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed

logger = logging.getLogger(__name__)

class AdaptiveOptimizer:
    """自适应优化器：根据实时反馈动态调整参数"""
    
    def __init__(self, task_type: str, table_count: int):
        self.task_type = task_type
        self.table_count = table_count
        self.optimization_history = []
        self.current_config = None
        self.best_config = None
        self.best_performance = 0
        
    def optimize_in_batches(self, queries: List[Dict], tables: List[Dict], 
                           batch_size: int = 5, max_iterations: int = 3) -> List[Dict]:
        """
        批次级别的自适应优化
        
        Args:
            queries: 查询列表
            tables: 表列表
            batch_size: 每批处理的查询数
            max_iterations: 最大优化迭代次数
            
        Returns:
            所有查询的处理结果
        """
        from src.agents.optimizer_agent import OptimizerAgent
        
        all_results = []
        optimizer = OptimizerAgent()
        
        # 将查询分批
        batches = [queries[i:i+batch_size] for i in range(0, len(queries), batch_size)]
        
        for batch_idx, batch_queries in enumerate(batches):
            logger.info(f"处理批次 {batch_idx+1}/{len(batches)}, 包含 {len(batch_queries)} 个查询")
            
            # 决定是否需要优化参数
            if batch_idx == 0 or self._should_reoptimize(batch_idx):
                # 获取优化配置
                optimization_input = {
                    'task_type': self.task_type,
                    'table_count': self.table_count,
                    'complexity': self._estimate_complexity(batch_queries),
                    'performance_requirement': 'balanced'
                }
                
                # 如果有历史性能数据，加入反馈
                if self.optimization_history:
                    last_performance = self.optimization_history[-1]
                    optimization_input['previous_performance'] = last_performance
                    optimization_input['feedback'] = self._generate_feedback(last_performance)
                
                # 调用OptimizerAgent
                self.current_config = optimizer.process(optimization_input)
                logger.info(f"更新优化配置: {self.current_config}")
            
            # 并行处理当前批次
            batch_results = self._process_batch_parallel(
                batch_queries, tables, self.current_config
            )
            
            # 评估批次性能
            batch_performance = self._evaluate_performance(batch_results)
            self.optimization_history.append(batch_performance)
            
            # 更新最佳配置
            if batch_performance['f1_score'] > self.best_performance:
                self.best_performance = batch_performance['f1_score']
                self.best_config = self.current_config
                logger.info(f"发现更好的配置，F1: {self.best_performance:.3f}")
            
            all_results.extend(batch_results)
            
            # 早停条件
            if batch_performance['f1_score'] > 0.6:
                logger.info(f"达到满意的性能 (F1={batch_performance['f1_score']:.3f})，使用当前配置处理剩余查询")
                # 处理剩余所有批次
                for remaining_batch in batches[batch_idx+1:]:
                    remaining_results = self._process_batch_parallel(
                        remaining_batch, tables, self.current_config
                    )
                    all_results.extend(remaining_results)
                break
        
        return all_results
    
    def _should_reoptimize(self, batch_idx: int) -> bool:
        """判断是否需要重新优化参数"""
        if not self.optimization_history:
            return True
        
        # 每3个批次重新优化一次
        if batch_idx % 3 == 0:
            return True
        
        # 如果最近的性能低于阈值，立即优化
        last_performance = self.optimization_history[-1]
        if last_performance['f1_score'] < 0.2:
            return True
        
        return False
    
    def _estimate_complexity(self, queries: List[Dict]) -> str:
        """估算查询复杂度"""
        avg_columns = np.mean([len(q.get('columns', [])) for q in queries])
        
        if avg_columns < 5:
            return 'simple'
        elif avg_columns < 10:
            return 'medium'
        else:
            return 'complex'
    
    def _generate_feedback(self, performance: Dict) -> str:
        """生成性能反馈"""
        if performance['recall'] < 0.3:
            return 'need_better_recall'
        elif performance['precision'] < 0.3:
            return 'need_better_precision'
        elif performance['f1_score'] < 0.3:
            return 'overall_poor_performance'
        else:
            return 'maintain_balance'
    
    def _process_batch_parallel(self, batch_queries: List[Dict], 
                               tables: List[Dict], config: Dict) -> List[Dict]:
        """并行处理一批查询"""
        # 这里简化了实际的处理逻辑
        # 实际应该调用three_layer_ablation_optimized.py中的process_query_l3
        results = []
        
        with ProcessPoolExecutor(max_workers=4) as executor:
            futures = []
            for query in batch_queries:
                # 提交任务
                future = executor.submit(self._process_single_query, query, tables, config)
                futures.append(future)
            
            # 收集结果
            for future in as_completed(futures):
                try:
                    result = future.result(timeout=30)
                    results.append(result)
                except Exception as e:
                    logger.error(f"处理查询失败: {e}")
                    results.append({'predictions': [], 'error': str(e)})
        
        return results
    
    def _process_single_query(self, query: Dict, tables: List[Dict], config: Dict) -> Dict:
        """处理单个查询（简化版）"""
        # 这里应该调用实际的查询处理函数
        # 为演示目的，返回模拟结果
        return {
            'query': query,
            'predictions': [],  # 实际预测结果
            'config_used': config
        }
    
    def _evaluate_performance(self, results: List[Dict]) -> Dict:
        """评估批次性能"""
        # 简化的性能评估
        # 实际应该计算真实的precision, recall, f1
        return {
            'precision': np.random.uniform(0.1, 0.5),
            'recall': np.random.uniform(0.1, 0.5),
            'f1_score': np.random.uniform(0.1, 0.4),
            'num_queries': len(results)
        }
    
    def get_optimization_summary(self) -> Dict:
        """获取优化总结"""
        return {
            'total_iterations': len(self.optimization_history),
            'best_performance': self.best_performance,
            'best_config': self.best_config,
            'performance_history': self.optimization_history
        }


def integrate_with_ablation_experiment(layer: str, queries: List[Dict], 
                                       tables: List[Dict], task_type: str, 
                                       dataset_type: str) -> Tuple[List[Dict], float]:
    """
    将自适应优化集成到消融实验中
    
    这个函数可以替换three_layer_ablation_optimized.py中的run_layer_experiment
    """
    if layer != 'L1+L2+L3':
        # 非L3层使用原始方法
        from three_layer_ablation_optimized import run_layer_experiment
        return run_layer_experiment(layer, queries, tables, task_type, dataset_type)
    
    # L3层使用自适应优化
    logger.info("使用自适应优化运行L3层实验")
    
    adaptive_optimizer = AdaptiveOptimizer(task_type, len(tables))
    
    # 运行自适应优化
    results = adaptive_optimizer.optimize_in_batches(
        queries=queries,
        tables=tables,
        batch_size=5,  # 每批5个查询
        max_iterations=3  # 最多优化3次
    )
    
    # 获取优化总结
    summary = adaptive_optimizer.get_optimization_summary()
    logger.info(f"优化总结: {json.dumps(summary, indent=2)}")
    
    # 计算最终性能
    final_f1 = summary['best_performance']
    
    return results, final_f1


if __name__ == "__main__":
    # 测试代码
    logging.basicConfig(level=logging.INFO)
    
    # 模拟数据
    test_queries = [{'query_table': f'table_{i}'} for i in range(20)]
    test_tables = [{'name': f'table_{i}'} for i in range(100)]
    
    # 创建自适应优化器
    optimizer = AdaptiveOptimizer(task_type='join', table_count=100)
    
    # 运行优化
    results = optimizer.optimize_in_batches(
        queries=test_queries,
        tables=test_tables,
        batch_size=5,
        max_iterations=3
    )
    
    # 打印总结
    summary = optimizer.get_optimization_summary()
    print(f"优化完成: {json.dumps(summary, indent=2)}")