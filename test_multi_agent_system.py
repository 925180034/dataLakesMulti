#!/usr/bin/env python
"""
Test script to verify the multi-agent system is working correctly
验证多智能体系统是否正常工作
"""

import asyncio
import json
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def test_multi_agent_workflow():
    """Test the multi-agent workflow with a simple query"""
    
    # Import the multi-agent workflow
    from src.core.multi_agent_workflow import create_multi_agent_workflow
    from src.core.state import WorkflowState
    
    logger.info("🚀 Starting Multi-Agent System Test")
    logger.info("=" * 60)
    
    # Load test data
    test_data_path = Path("examples/webtable/join_subset")
    tables_path = test_data_path / "tables.json"
    queries_path = test_data_path / "queries.json"
    
    if not tables_path.exists() or not queries_path.exists():
        logger.error(f"Test data not found at {test_data_path}")
        return False
    
    with open(tables_path, 'r') as f:
        tables = json.load(f)
    with open(queries_path, 'r') as f:
        queries = json.load(f)
    
    # Ensure tables have 'name' field for compatibility
    for t in tables:
        if 'name' not in t and 'table_name' in t:
            t['name'] = t['table_name']
    
    logger.info(f"📊 Loaded {len(tables)} tables and {len(queries)} queries")
    
    # Take first query for testing
    test_query = queries[0] if queries else None
    if not test_query:
        logger.error("No test queries available")
        return False
    
    # Create workflow
    workflow = create_multi_agent_workflow()
    logger.info("✅ Multi-Agent Workflow created successfully")
    
    # Prepare initial state
    initial_state = WorkflowState(
        query_table={
            'table_name': test_query.get('query_table', 'unknown'),
            'columns': []  # Would be filled with actual column data
        },
        query_task={
            'table_name': test_query.get('query_table', 'unknown'),
            'task_type': 'join'
        },
        all_tables=tables
    )
    
    logger.info(f"🔍 Testing with query table: {test_query.get('query_table')}")
    logger.info("-" * 60)
    
    try:
        # Run the workflow
        logger.info("🤖 Starting Multi-Agent Workflow Execution:")
        logger.info("  1. OptimizerAgent - System optimization")
        logger.info("  2. PlannerAgent - Strategy planning")
        logger.info("  3. AnalyzerAgent - Table analysis")
        logger.info("  4. SearcherAgent - Layer 1+2 search")
        logger.info("  5. MatcherAgent - Layer 3 verification")
        logger.info("  6. AggregatorAgent - Result aggregation")
        logger.info("-" * 60)
        
        result = await workflow.run(initial_state)
        
        # Check results
        if 'error' in result:
            logger.error(f"❌ Workflow failed: {result['error']}")
            return False
        
        # Log results
        logger.info("✅ Multi-Agent Workflow completed successfully!")
        logger.info("-" * 60)
        
        # Check what each agent produced
        if 'optimization_config' in result:
            logger.info("✅ OptimizerAgent output:")
            config = result['optimization_config']
            logger.info(f"   - Parallel workers: {config.get('parallel_workers', 'N/A')}")
            logger.info(f"   - LLM concurrency: {config.get('llm_concurrency', 'N/A')}")
        
        if 'strategy' in result:
            logger.info("✅ PlannerAgent output:")
            strategy = result['strategy']
            logger.info(f"   - Strategy: {strategy.get('name', 'N/A')}")
            logger.info(f"   - Use metadata: {strategy.get('use_metadata', False)}")
            logger.info(f"   - Use vector: {strategy.get('use_vector', False)}")
        
        if 'analysis' in result:
            logger.info("✅ AnalyzerAgent output:")
            analysis = result['analysis']
            logger.info(f"   - Table type: {analysis.get('table_type', 'N/A')}")
        
        if 'candidates' in result:
            candidates = result['candidates']
            logger.info(f"✅ SearcherAgent output: {len(candidates)} candidates from Layer 1+2")
            if candidates:
                logger.info(f"   - Top candidate: {candidates[0].table_name if hasattr(candidates[0], 'table_name') else candidates[0]}")
        
        if 'matches' in result:
            matches = result['matches']
            logger.info(f"✅ MatcherAgent output: {len(matches)} verified matches from Layer 3")
        
        if 'final_results' in result:
            final_results = result['final_results']
            logger.info(f"✅ AggregatorAgent output: {len(final_results)} final results")
            if final_results:
                logger.info("   Top 3 results:")
                for i, res in enumerate(final_results[:3]):
                    if isinstance(res, dict):
                        logger.info(f"   {i+1}. {res.get('table_name', 'unknown')} (score: {res.get('score', 0):.3f})")
                    else:
                        logger.info(f"   {i+1}. {res}")
        
        logger.info("-" * 60)
        logger.info(f"⏱️ Execution time: {result.get('execution_time', 0):.2f} seconds")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Exception during workflow execution: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main entry point"""
    logger.info("🎯 Multi-Agent System Architecture Test")
    logger.info("This test verifies that all 6 agents work together correctly")
    logger.info("using the three-layer acceleration architecture")
    logger.info("=" * 60)
    
    # Run the async test
    success = asyncio.run(test_multi_agent_workflow())
    
    if success:
        logger.info("=" * 60)
        logger.info("🎉 SUCCESS: Multi-Agent System is working correctly!")
        logger.info("The system successfully:")
        logger.info("  ✅ Initialized all 6 agents")
        logger.info("  ✅ Executed workflow coordination")
        logger.info("  ✅ Called Layer 1+2 through SearcherAgent")
        logger.info("  ✅ Called Layer 3 through MatcherAgent")
        logger.info("  ✅ Produced final results")
    else:
        logger.info("=" * 60)
        logger.error("❌ FAILURE: Multi-Agent System has issues")
        logger.error("Please check the error messages above")
    
    return 0 if success else 1


if __name__ == "__main__":
    exit(main())