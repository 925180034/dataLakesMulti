#!/usr/bin/env python
"""
评估指标计算模块
计算Precision, Recall, F1-Score和Hit@K等指标
"""
from typing import List, Dict, Any
import logging

logger = logging.getLogger(__name__)


def calculate_metrics(results: List[Dict], ground_truth: List[Dict]) -> Dict[str, float]:
    """
    计算评估指标
    
    Args:
        results: 预测结果列表，每个元素包含 'query_table' 和 'predictions'
        ground_truth: 真实标签列表，每个元素包含 'query_table' 和 'candidate_tables'
    
    Returns:
        包含各种评估指标的字典
    """
    # 构建ground truth字典
    gt_dict = {}
    for gt in ground_truth:
        query_table = gt.get('query_table', '')
        candidates = gt.get('candidate_tables', [])
        if query_table:
            gt_dict[query_table] = set(candidates)
    
    # 计算指标
    hit_at_1 = 0
    hit_at_3 = 0
    hit_at_5 = 0
    hit_at_10 = 0
    
    total_precision = 0
    total_recall = 0
    total_f1 = 0
    valid_queries = 0
    
    for result in results:
        query_table = result.get('query_table', '')
        predictions = result.get('predictions', [])
        
        if query_table not in gt_dict:
            continue
        
        valid_queries += 1
        true_tables = gt_dict[query_table]
        
        # Hit@K metrics
        for k, hit_counter in [(1, 'hit_at_1'), (3, 'hit_at_3'), (5, 'hit_at_5'), (10, 'hit_at_10')]:
            top_k_predictions = set(predictions[:k])
            if top_k_predictions & true_tables:  # 有交集
                if hit_counter == 'hit_at_1':
                    hit_at_1 += 1
                elif hit_counter == 'hit_at_3':
                    hit_at_3 += 1
                elif hit_counter == 'hit_at_5':
                    hit_at_5 += 1
                elif hit_counter == 'hit_at_10':
                    hit_at_10 += 1
        
        # Precision, Recall, F1
        if predictions:
            predicted_set = set(predictions[:5])  # 使用top-5预测
            
            # True Positives
            tp = len(predicted_set & true_tables)
            
            # False Positives
            fp = len(predicted_set - true_tables)
            
            # False Negatives
            fn = len(true_tables - predicted_set)
            
            # Calculate metrics
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            total_precision += precision
            total_recall += recall
            total_f1 += f1
    
    # 计算平均值
    if valid_queries > 0:
        metrics = {
            'hit@1': hit_at_1 / valid_queries,
            'hit@3': hit_at_3 / valid_queries,
            'hit@5': hit_at_5 / valid_queries,
            'hit@10': hit_at_10 / valid_queries,
            'precision': total_precision / valid_queries,
            'recall': total_recall / valid_queries,
            'f1_score': total_f1 / valid_queries,
            'valid_queries': valid_queries,
            'total_queries': len(results)
        }
    else:
        metrics = {
            'hit@1': 0.0,
            'hit@3': 0.0,
            'hit@5': 0.0,
            'hit@10': 0.0,
            'precision': 0.0,
            'recall': 0.0,
            'f1_score': 0.0,
            'valid_queries': 0,
            'total_queries': len(results)
        }
    
    return metrics


def evaluate_batch_results(predictions: Dict[str, List[str]], 
                          ground_truth: Dict[str, List[str]]) -> Dict[str, float]:
    """
    批量评估结果（简化版本）
    
    Args:
        predictions: {query_table: [predicted_tables]}
        ground_truth: {query_table: [true_tables]}
    
    Returns:
        评估指标字典
    """
    results = []
    gt_list = []
    
    for query_table, pred_tables in predictions.items():
        results.append({
            'query_table': query_table,
            'predictions': pred_tables
        })
        
        if query_table in ground_truth:
            gt_list.append({
                'query_table': query_table,
                'candidate_tables': ground_truth[query_table]
            })
    
    return calculate_metrics(results, gt_list)


def print_metrics(metrics: Dict[str, float], title: str = "Evaluation Metrics"):
    """
    打印评估指标
    
    Args:
        metrics: 指标字典
        title: 标题
    """
    print("\n" + "="*60)
    print(f" {title}")
    print("="*60)
    
    print(f"Valid Queries: {metrics.get('valid_queries', 0)}/{metrics.get('total_queries', 0)}")
    print("-"*40)
    
    print("Hit@K Metrics:")
    print(f"  Hit@1:  {metrics.get('hit@1', 0):.3f}")
    print(f"  Hit@3:  {metrics.get('hit@3', 0):.3f}")
    print(f"  Hit@5:  {metrics.get('hit@5', 0):.3f}")
    print(f"  Hit@10: {metrics.get('hit@10', 0):.3f}")
    
    print("\nClassification Metrics:")
    print(f"  Precision: {metrics.get('precision', 0):.3f}")
    print(f"  Recall:    {metrics.get('recall', 0):.3f}")
    print(f"  F1-Score:  {metrics.get('f1_score', 0):.3f}")
    print("="*60)


if __name__ == "__main__":
    # 测试代码
    test_results = [
        {
            'query_table': 'table1',
            'predictions': ['table2', 'table3', 'table4']
        },
        {
            'query_table': 'table2',
            'predictions': ['table1', 'table5', 'table6']
        }
    ]
    
    test_ground_truth = [
        {
            'query_table': 'table1',
            'candidate_tables': ['table2', 'table5']
        },
        {
            'query_table': 'table2',
            'candidate_tables': ['table1', 'table3']
        }
    ]
    
    metrics = calculate_metrics(test_results, test_ground_truth)
    print_metrics(metrics, "Test Metrics")