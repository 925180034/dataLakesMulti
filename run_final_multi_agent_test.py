#\!/usr/bin/env python
"""
运行完整的多智能体系统测试 - 包括JOIN和UNION任务
"""

import asyncio
import json
import time
import os
import sys
from typing import List, Dict, Any
import argparse
import logging

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from run_multi_agent_llm_enabled import FullyFixedMultiAgentSystem, QueryTask, MatchResult

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def create_mixed_query_tasks(tables_data: List[Dict], num_join: int = 50, num_union: int = 50) -> List[QueryTask]:
    """创建混合的JOIN和UNION查询任务"""
    query_tasks = []
    
    # 从表中随机选择作为查询表
    import random
    random.seed(42)  # 固定随机种子以便复现
    
    selected_tables = random.sample(tables_data, min(num_join + num_union, len(tables_data)))
    
    # 创建JOIN任务
    for i in range(num_join):
        if i < len(selected_tables):
            query_table = selected_tables[i]['table_name']
            query_tasks.append(QueryTask(
                query_id=f"join_{i}",
                query_table=query_table,
                task_type='join',
                ground_truth=[]  # 实际系统中会有真实的ground truth
            ))
    
    # 创建UNION任务
    for i in range(num_union):
        if num_join + i < len(selected_tables):
            query_table = selected_tables[num_join + i]['table_name']
            query_tasks.append(QueryTask(
                query_id=f"union_{i}",
                query_table=query_table,
                task_type='union',
                ground_truth=[]
            ))
    
    return query_tasks


async def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='完整多智能体系统测试')
    parser.add_argument('--dataset', choices=['subset', 'complete'], 
                       default='complete', help='使用的数据集')
    parser.add_argument('--join-queries', type=int, default=50, 
                       help='JOIN查询数量')
    parser.add_argument('--union-queries', type=int, default=50,
                       help='UNION查询数量')
    parser.add_argument('--workers', type=int, default=4,
                       help='并行工作线程数')
    parser.add_argument('--max-candidates', type=int, default=10,
                       help='每个查询的最大候选数量')
    
    args = parser.parse_args()
    
    total_queries = args.join_queries + args.union_queries
    
    print("\n" + "="*70)
    print("🚀 COMPLETE MULTI-AGENT SYSTEM TEST")
    print("="*70)
    print(f"📊 Dataset: {args.dataset.upper()}")
    print(f"🔧 Query Distribution:")
    print(f"   - JOIN queries: {args.join_queries}")
    print(f"   - UNION queries: {args.union_queries}")
    print(f"   - Total queries: {total_queries}")
    print(f"⚡ Parallel workers: {args.workers}")
    print(f"🎯 Max candidates per query: {args.max_candidates}")
    print()
    
    # 选择数据集
    if args.dataset == 'subset':
        tables_file = 'examples/final_subset_tables.json'
        print("📂 Using subset dataset (100 tables)")
    else:
        tables_file = 'examples/final_complete_tables.json'
        print("📂 Using complete dataset (1,534 tables)")
    
    # 初始化系统
    print("\n🔄 Initializing Multi-Agent System...")
    system = FullyFixedMultiAgentSystem()
    
    # 加载数据
    print("📥 Loading dataset...")
    start_load = time.time()
    await system.load_data(tables_file)
    load_time = time.time() - start_load
    print(f"✅ Data loaded in {load_time:.2f}s")
    
    # 加载表数据用于创建查询
    with open(tables_file, 'r') as f:
        tables_data = json.load(f)
    
    # 创建混合查询任务
    print("\n📋 Creating query tasks...")
    query_tasks = create_mixed_query_tasks(
        tables_data, 
        num_join=args.join_queries,
        num_union=args.union_queries
    )
    
    join_tasks = [t for t in query_tasks if t.task_type == 'join']
    union_tasks = [t for t in query_tasks if t.task_type == 'union']
    
    print(f"   - Created {len(join_tasks)} JOIN tasks")
    print(f"   - Created {len(union_tasks)} UNION tasks")
    
    # 运行实验
    print("\n🏃 Running multi-agent processing...")
    print("   Processing queries in parallel...")
    
    start_time = time.time()
    
    # 并行处理所有查询
    results = await system.process_queries_parallel(query_tasks, max_workers=args.workers)
    
    total_time = time.time() - start_time
    
    # 统计结果
    join_results = []
    union_results = []
    all_predictions = {}
    
    for query_task, matches in results:
        if matches:
            predictions = [m.matched_table for m in matches[:args.max_candidates]]
            all_predictions[query_task.query_id] = predictions
            
            if query_task.task_type == 'join':
                join_results.append((query_task, matches))
            else:
                union_results.append((query_task, matches))
    
    # 计算性能指标
    successful_queries = system.successful_queries
    success_rate = (successful_queries / len(query_tasks) * 100) if query_tasks else 0
    
    avg_response_time = 0
    if system.query_times:
        avg_response_time = sum(system.query_times) / len(system.query_times)
    
    throughput = len(query_tasks) / total_time if total_time > 0 else 0
    
    # 显示结果
    print("\n" + "="*70)
    print("📊 EXPERIMENT RESULTS")
    print("="*70)
    
    print("\n⏱️  Performance Metrics:")
    print(f"   Total Time: {total_time:.2f}s")
    print(f"   Data Loading: {load_time:.2f}s")
    print(f"   Query Processing: {total_time - load_time:.2f}s")
    print(f"   Average Response Time: {avg_response_time:.3f}s")
    print(f"   Throughput: {throughput:.2f} QPS")
    
    print(f"\n📈 Success Statistics:")
    print(f"   Total Queries: {len(query_tasks)}")
    print(f"   Successful: {successful_queries}")
    print(f"   Success Rate: {success_rate:.1f}%")
    
    print(f"\n🔍 Task Distribution:")
    print(f"   JOIN Results: {len(join_results)}/{len(join_tasks)}")
    print(f"   UNION Results: {len(union_results)}/{len(union_tasks)}")
    
    # 计算每种任务类型的平均匹配数
    if join_results:
        avg_join_matches = sum(len(matches) for _, matches in join_results) / len(join_results)
        print(f"   Avg JOIN matches: {avg_join_matches:.1f}")
    
    if union_results:
        avg_union_matches = sum(len(matches) for _, matches in union_results) / len(union_results)
        print(f"   Avg UNION matches: {avg_union_matches:.1f}")
    
    # 保存结果
    timestamp = int(time.time())
    result_dir = f"experiment_results/complete_{args.dataset}"
    os.makedirs(result_dir, exist_ok=True)
    
    result_file = f"{result_dir}/experiment_{args.join_queries}j_{args.union_queries}u_{timestamp}.json"
    
    experiment_data = {
        'config': {
            'dataset': args.dataset,
            'join_queries': args.join_queries,
            'union_queries': args.union_queries,
            'total_queries': total_queries,
            'workers': args.workers,
            'max_candidates': args.max_candidates,
            'timestamp': timestamp
        },
        'performance': {
            'total_time': total_time,
            'load_time': load_time,
            'processing_time': total_time - load_time,
            'avg_response_time': avg_response_time,
            'throughput': throughput,
            'success_rate': success_rate
        },
        'results': {
            'total_queries': len(query_tasks),
            'successful_queries': successful_queries,
            'join_results_count': len(join_results),
            'union_results_count': len(union_results),
            'predictions': all_predictions
        }
    }
    
    with open(result_file, 'w') as f:
        json.dump(experiment_data, f, indent=2)
    
    print(f"\n💾 Results saved to: {result_file}")
    
    # 显示示例结果
    print("\n📝 Sample Results:")
    
    if join_results and len(join_results) > 0:
        print("\n  JOIN Example:")
        task, matches = join_results[0]
        print(f"    Query: {task.query_table}")
        print(f"    Found {len(matches)} matches:")
        for i, match in enumerate(matches[:3]):
            print(f"      {i+1}. {match.matched_table} (score: {match.score:.3f})")
    
    if union_results and len(union_results) > 0:
        print("\n  UNION Example:")
        task, matches = union_results[0]
        print(f"    Query: {task.query_table}")
        print(f"    Found {len(matches)} matches:")
        for i, match in enumerate(matches[:3]):
            print(f"      {i+1}. {match.matched_table} (score: {match.score:.3f})")
    
    print("\n✅ EXPERIMENT COMPLETED\!")
    print("="*70)


if __name__ == "__main__":
    asyncio.run(main())
