#!/usr/bin/env python
"""
Simple test script for the new LangGraph multi-agent system
"""
import json
from src.core.langgraph_workflow import create_workflow

def test_langgraph():
    """Test the LangGraph workflow with sample data"""
    
    # Load sample data
    with open('examples/final_subset_tables.json', 'r') as f:
        tables = json.load(f)
    
    # Create workflow
    print("Creating LangGraph workflow...")
    workflow = create_workflow()
    
    # Test with a single query
    query_table = tables[0]  # Use first table as query
    print(f"\nTesting with query table: {query_table.get('table_name')}")
    
    result = workflow.run(
        query=f"Find tables that can join with {query_table.get('table_name')}",
        tables=tables[:10],  # Use only 10 tables for quick test
        task_type='join',
        query_table_name=query_table.get('table_name')
    )
    
    # Display results
    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)
    
    if result.get('success'):
        print(f"✅ Success!")
        print(f"Query Table: {result.get('query_table')}")
        print(f"Task Type: {result.get('task_type')}")
        print(f"Time: {result.get('metrics', {}).get('total_time', 0):.2f}s")
        print(f"Matches Found: {len(result.get('results', []))}")
        
        if result.get('results'):
            print("\nTop Matches:")
            for i, match in enumerate(result.get('results', [])[:3]):
                print(f"  {i+1}. {match['table_name']} (score: {match['score']:.3f})")
    else:
        print(f"❌ Failed: {result.get('error')}")
    
    # Show agent execution times
    if result.get('metrics', {}).get('agent_times'):
        print("\nAgent Execution Times:")
        for agent, time_taken in result.get('metrics', {}).get('agent_times', {}).items():
            print(f"  - {agent}: {time_taken:.3f}s")

if __name__ == "__main__":
    test_langgraph()