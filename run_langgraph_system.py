#!/usr/bin/env python
"""
Main entry point for LangGraph-based multi-agent data lake discovery system
"""
import os
import sys
import json
import time
import logging
import asyncio
from pathlib import Path
from typing import List, Dict, Any

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.core.langgraph_workflow import create_workflow
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_test_data(dataset_type: str = 'subset') -> tuple:
    """Load test dataset"""
    if dataset_type == 'subset':
        tables_file = 'examples/final_subset_tables.json'
        queries_file = 'examples/final_subset_queries.json'
        ground_truth_file = 'examples/final_subset_ground_truth.json'
    else:
        tables_file = 'examples/final_complete_tables.json'
        queries_file = 'examples/final_complete_queries.json'
        ground_truth_file = 'examples/final_complete_ground_truth.json'
    
    # Load tables
    with open(tables_file, 'r') as f:
        tables = json.load(f)
    
    # Load queries
    with open(queries_file, 'r') as f:
        queries = json.load(f)
    
    # Load ground truth
    with open(ground_truth_file, 'r') as f:
        ground_truth = json.load(f)
    
    logger.info(f"Loaded {len(tables)} tables, {len(queries)} queries")
    return tables, queries, ground_truth


def run_single_query(workflow, query_info: Dict, tables: List[Dict]) -> Dict:
    """Run workflow for a single query"""
    query_table = query_info.get('query_table', query_info.get('table', ''))
    task_type = query_info.get('query_type', query_info.get('type', 'join'))
    
    result = workflow.run(
        query=f"Find tables that can {task_type} with {query_table}",
        tables=tables,
        task_type=task_type,
        query_table_name=query_table
    )
    
    return result


def evaluate_results(results: List[Dict], ground_truth: Dict) -> Dict:
    """Evaluate results against ground truth"""
    metrics = {
        'total_queries': len(results),
        'successful_queries': 0,
        'failed_queries': 0,
        'average_time': 0,
        'total_time': 0,
        'precision_sum': 0,
        'recall_sum': 0,
        'f1_sum': 0
    }
    
    total_time = 0
    
    for result in results:
        if result.get('success'):
            metrics['successful_queries'] += 1
            
            # Calculate precision/recall if ground truth available
            query_table = result.get('query_table')
            if query_table and query_table in ground_truth:
                expected = set(ground_truth[query_table])
                predicted = set([r['table_name'] for r in result.get('results', [])])
                
                if predicted:
                    precision = len(expected & predicted) / len(predicted)
                    metrics['precision_sum'] += precision
                
                if expected:
                    recall = len(expected & predicted) / len(expected)
                    metrics['recall_sum'] += recall
                
                if predicted or expected:
                    if precision + recall > 0:
                        f1 = 2 * precision * recall / (precision + recall)
                        metrics['f1_sum'] += f1
        else:
            metrics['failed_queries'] += 1
        
        total_time += result.get('metrics', {}).get('total_time', 0)
    
    metrics['total_time'] = total_time
    metrics['average_time'] = total_time / len(results) if results else 0
    
    if metrics['successful_queries'] > 0:
        metrics['avg_precision'] = metrics['precision_sum'] / metrics['successful_queries']
        metrics['avg_recall'] = metrics['recall_sum'] / metrics['successful_queries']
        metrics['avg_f1'] = metrics['f1_sum'] / metrics['successful_queries']
    else:
        metrics['avg_precision'] = 0
        metrics['avg_recall'] = 0
        metrics['avg_f1'] = 0
    
    return metrics


def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Run LangGraph multi-agent system')
    parser.add_argument('--dataset', choices=['subset', 'complete'], default='subset',
                       help='Dataset to use')
    parser.add_argument('--max-queries', type=int, default=5,
                       help='Maximum number of queries to process')
    parser.add_argument('--task', choices=['join', 'union', 'both'], default='both',
                       help='Task type to evaluate')
    parser.add_argument('--output', type=str, help='Output file for results')
    
    args = parser.parse_args()
    
    # Load data
    logger.info(f"Loading {args.dataset} dataset...")
    tables, queries, ground_truth = load_test_data(args.dataset)
    
    # Filter queries by task type
    if args.task != 'both':
        queries = [q for q in queries if q.get('query_type', q.get('type')) == args.task]
    
    # Limit queries
    queries = queries[:args.max_queries]
    
    # Create workflow
    logger.info("Creating LangGraph workflow...")
    workflow = create_workflow()
    
    # Run queries
    logger.info(f"Processing {len(queries)} queries...")
    results = []
    
    for i, query_info in enumerate(queries):
        logger.info(f"\n{'='*60}")
        logger.info(f"Query {i+1}/{len(queries)}: {query_info.get('query_table')} ({query_info.get('type')})")
        logger.info(f"{'='*60}")
        
        result = run_single_query(workflow, query_info, tables)
        results.append(result)
        
        # Log results
        if result.get('success'):
            logger.info(f"✅ Success! Found {len(result.get('results', []))} matches")
            logger.info(f"⏱️  Time: {result.get('metrics', {}).get('total_time', 0):.2f}s")
            
            # Show top 3 matches
            for j, match in enumerate(result.get('results', [])[:3]):
                logger.info(f"  {j+1}. {match['table_name']} (score: {match['score']:.3f})")
        else:
            logger.error(f"❌ Failed: {result.get('error')}")
    
    # Evaluate results
    logger.info(f"\n{'='*60}")
    logger.info("EVALUATION RESULTS")
    logger.info(f"{'='*60}")
    
    metrics = evaluate_results(results, ground_truth)
    
    logger.info(f"Total Queries: {metrics['total_queries']}")
    logger.info(f"Successful: {metrics['successful_queries']}")
    logger.info(f"Failed: {metrics['failed_queries']}")
    if metrics['total_queries'] > 0:
        logger.info(f"Success Rate: {metrics['successful_queries']/metrics['total_queries']*100:.1f}%")
    else:
        logger.info(f"Success Rate: N/A (no queries processed)")
    logger.info(f"Average Time: {metrics['average_time']:.2f}s")
    logger.info(f"Total Time: {metrics['total_time']:.2f}s")
    
    if metrics['successful_queries'] > 0:
        logger.info(f"Average Precision: {metrics['avg_precision']:.3f}")
        logger.info(f"Average Recall: {metrics['avg_recall']:.3f}")
        logger.info(f"Average F1: {metrics['avg_f1']:.3f}")
    
    # Save results if requested
    if args.output:
        output_data = {
            'dataset': args.dataset,
            'task': args.task,
            'queries': queries,
            'results': results,
            'metrics': metrics,
            'timestamp': time.time()
        }
        
        with open(args.output, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        logger.info(f"\n💾 Results saved to {args.output}")
    
    logger.info("\n✨ LangGraph workflow execution complete!")


if __name__ == "__main__":
    main()