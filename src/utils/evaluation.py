"""
Evaluation utilities for data lake discovery system
"""

from typing import List, Dict, Any


def calculate_hit_at_k(results: List[Dict], ground_truth: Dict, k: int) -> float:
    """
    Calculate Hit@K metric
    
    Args:
        results: List of prediction results
        ground_truth: Ground truth data
        k: Top-k value
    
    Returns:
        Hit@K score
    """
    if not results:
        return 0.0
    
    hits = 0
    total = 0
    
    for result in results:
        query_table = result.get('query_table', '')
        predictions = result.get('predictions', [])[:k]
        
        # Get ground truth for this query
        if isinstance(ground_truth, dict):
            # Handle both dict and list formats
            gt = None
            # Check if it's NLCTables format (query_id as key)
            for gt_item in ground_truth.values():
                if isinstance(gt_item, list) and len(gt_item) > 0:
                    if isinstance(gt_item[0], dict) and 'query_table' in gt_item[0]:
                        # List of dicts format
                        for item in gt_item:
                            if item.get('query_table') == query_table:
                                gt = item.get('ground_truth', [])
                                break
                    break
            
            # Also check direct query_table key
            if gt is None and query_table in ground_truth:
                gt = ground_truth[query_table]
        elif isinstance(ground_truth, list):
            # List format
            gt = None
            for gt_item in ground_truth:
                if gt_item.get('query_table') == query_table:
                    gt = gt_item.get('ground_truth', [])
                    break
        else:
            gt = []
        
        if gt is None:
            continue
            
        # Check if any prediction is in ground truth
        if gt:
            total += 1
            for pred in predictions:
                pred_name = pred if isinstance(pred, str) else pred.get('table_name', '')
                if pred_name in gt:
                    hits += 1
                    break
    
    return hits / total if total > 0 else 0.0


def calculate_precision_recall_f1(results: List[Dict], ground_truth: Dict) -> Dict[str, float]:
    """
    Calculate Precision, Recall, and F1 scores
    
    Args:
        results: List of prediction results
        ground_truth: Ground truth data
    
    Returns:
        Dictionary with precision, recall, and f1 scores
    """
    if not results:
        return {'precision': 0.0, 'recall': 0.0, 'f1': 0.0}
    
    total_true_positives = 0
    total_false_positives = 0
    total_false_negatives = 0
    
    for result in results:
        query_table = result.get('query_table', '')
        predictions = result.get('predictions', [])
        
        # Get ground truth for this query
        if isinstance(ground_truth, dict):
            gt = None
            # Check if it's NLCTables format
            for gt_item in ground_truth.values():
                if isinstance(gt_item, list) and len(gt_item) > 0:
                    if isinstance(gt_item[0], dict) and 'query_table' in gt_item[0]:
                        for item in gt_item:
                            if item.get('query_table') == query_table:
                                gt = item.get('ground_truth', [])
                                break
                    break
            
            if gt is None and query_table in ground_truth:
                gt = ground_truth[query_table]
        elif isinstance(ground_truth, list):
            gt = None
            for gt_item in ground_truth:
                if gt_item.get('query_table') == query_table:
                    gt = gt_item.get('ground_truth', [])
                    break
        else:
            gt = []
        
        if gt is None:
            gt = []
        
        # Convert predictions to table names
        pred_names = []
        for pred in predictions:
            if isinstance(pred, str):
                pred_names.append(pred)
            elif isinstance(pred, dict):
                pred_names.append(pred.get('table_name', ''))
        
        # Calculate TP, FP, FN
        pred_set = set(pred_names)
        gt_set = set(gt)
        
        true_positives = len(pred_set & gt_set)
        false_positives = len(pred_set - gt_set)
        false_negatives = len(gt_set - pred_set)
        
        total_true_positives += true_positives
        total_false_positives += false_positives
        total_false_negatives += false_negatives
    
    # Calculate metrics
    precision = total_true_positives / (total_true_positives + total_false_positives) if (total_true_positives + total_false_positives) > 0 else 0.0
    recall = total_true_positives / (total_true_positives + total_false_negatives) if (total_true_positives + total_false_negatives) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1
    }