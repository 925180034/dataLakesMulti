#!/usr/bin/env python
"""
Test script to verify MAX_PREDICTIONS configuration works correctly
Tests that the system can return enough predictions for proper @10 calculation
"""

import os
import json
import sys
from pathlib import Path

# Set different MAX_PREDICTIONS values
TEST_VALUES = [5, 10, 15, 20]

def test_predictions_count(max_predictions):
    """Test with a specific MAX_PREDICTIONS value"""
    print(f"\n{'='*60}")
    print(f"Testing with MAX_PREDICTIONS={max_predictions}")
    print('='*60)
    
    # Set environment variable
    os.environ['MAX_PREDICTIONS'] = str(max_predictions)
    os.environ['SKIP_LLM'] = 'true'  # Skip LLM for faster testing
    
    # Import after setting env var to ensure it takes effect
    if 'three_layer_ablation_optimized' in sys.modules:
        del sys.modules['three_layer_ablation_optimized']
    
    from three_layer_ablation_optimized import (
        process_query_l1, process_query_l2,
        initialize_shared_resources_l1, initialize_shared_resources_l2,
        MAX_PREDICTIONS
    )
    
    print(f"Module MAX_PREDICTIONS value: {MAX_PREDICTIONS}")
    
    # Load test dataset
    tables_path = Path('examples/webtable/join_subset/tables.json')
    queries_path = Path('examples/webtable/join_subset/queries.json')
    
    with open(tables_path, 'r') as f:
        tables = json.load(f)
    with open(queries_path, 'r') as f:
        queries = json.load(f)
    
    # Test first query
    query = queries[0]
    query_table = query.get('query_table')
    
    # Initialize and run L1
    print(f"\nTesting L1 layer...")
    shared_config_l1 = initialize_shared_resources_l1(tables, 'webtable')
    result_l1 = process_query_l1((query, tables, shared_config_l1, 'cache/test'))
    l1_count = len(result_l1['predictions'])
    print(f"  L1 returned {l1_count} predictions (expected up to {max_predictions})")
    
    # Initialize and run L2
    print(f"\nTesting L2 layer...")
    shared_config_l2 = initialize_shared_resources_l2(tables, 'webtable')
    query['task_type'] = 'join'  # Add task type for L2
    result_l2 = process_query_l2((query, tables, shared_config_l2, 'cache/test'))
    l2_count = len(result_l2['predictions'])
    print(f"  L2 returned {l2_count} predictions (expected up to {max_predictions})")
    
    # Check if we have enough predictions for @10
    can_calculate_at_10 = l1_count >= 10 and l2_count >= 10
    
    print(f"\n✅ Can calculate @10 metrics: {can_calculate_at_10}")
    if not can_calculate_at_10 and max_predictions >= 10:
        print(f"⚠️  WARNING: MAX_PREDICTIONS={max_predictions} but got <10 predictions!")
    
    return l1_count, l2_count, can_calculate_at_10

def main():
    print("Testing MAX_PREDICTIONS configuration")
    print("=" * 70)
    
    results = []
    for max_pred in TEST_VALUES:
        l1_count, l2_count, can_at_10 = test_predictions_count(max_pred)
        results.append({
            'max_predictions': max_pred,
            'l1_count': l1_count,
            'l2_count': l2_count,
            'can_calculate_at_10': can_at_10
        })
    
    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print('='*70)
    print(f"{'MAX_PRED':<10} {'L1':<10} {'L2':<10} {'Can @10':<10}")
    print('-'*40)
    for r in results:
        at_10 = '✅' if r['can_calculate_at_10'] else '❌'
        print(f"{r['max_predictions']:<10} {r['l1_count']:<10} {r['l2_count']:<10} {at_10:<10}")
    
    # Final check
    print(f"\n{'='*70}")
    if all(r['can_calculate_at_10'] for r in results if r['max_predictions'] >= 10):
        print("✅ SUCCESS: System properly supports @10 calculation when MAX_PREDICTIONS >= 10")
    else:
        print("❌ FAILURE: System does not properly support @10 calculation")
        sys.exit(1)

if __name__ == "__main__":
    main()