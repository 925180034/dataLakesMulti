#!/usr/bin/env python
"""
测试真实的多智能体数据湖发现系统
Test Real Multi-Agent Data Lake Discovery System
"""

import asyncio
import json
import time
import logging
from pathlib import Path
import sys
import os

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.core.real_multi_agent_system import (
    MultiAgentOrchestrator, 
    QueryTask,
    SystemMetrics
)

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def load_ground_truth(file_path: str, task_type: str = None):
    """加载ground truth数据"""
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    # 如果指定了任务类型，只返回该类型的数据
    if task_type:
        if task_type == 'join':
            return data.get('join_ground_truth', [])
        elif task_type == 'union':
            return data.get('union_ground_truth', [])
    
    # 返回所有数据
    all_queries = []
    
    # 处理不同格式的ground truth
    if isinstance(data, dict):
        # 新格式：分别存储join和union
        if 'join_ground_truth' in data:
            for item in data['join_ground_truth']:
                all_queries.append({
                    'query_table': item['table'],
                    'ground_truth': item.get('ground_truth', item.get('expected', [])),
                    'type': 'join'
                })
        if 'union_ground_truth' in data:
            for item in data['union_ground_truth']:
                all_queries.append({
                    'query_table': item['table'],
                    'ground_truth': item.get('ground_truth', item.get('expected', [])),
                    'type': 'union'
                })
    elif isinstance(data, list):
        # 旧格式：列表形式
        for item in data:
            query_type = item.get('type', 'join')
            all_queries.append({
                'query_table': item.get('query_table', item.get('table')),
                'ground_truth': item.get('ground_truth', item.get('expected', [])),
                'type': query_type
            })
    
    return all_queries

async def test_multi_agent_system(
    dataset: str = 'subset',
    max_queries: int = 100,
    parallel_workers: int = 4
):
    """测试多智能体系统"""
    
    print("\n" + "="*70)
    print("🚀 REAL MULTI-AGENT DATA LAKE DISCOVERY SYSTEM TEST")
    print("="*70)
    
    # 确定数据文件路径
    if dataset == 'subset':
        tables_file = 'examples/final_subset_tables.json'
        ground_truth_file = 'examples/final_subset_ground_truth.json'
        print(f"📊 Using SUBSET dataset (100 tables)")
    else:
        tables_file = 'examples/final_complete_tables.json'
        ground_truth_file = 'examples/final_complete_ground_truth.json'
        print(f"📊 Using COMPLETE dataset (1534 tables)")
    
    # 检查文件是否存在
    if not Path(tables_file).exists():
        logger.error(f"Tables file not found: {tables_file}")
        return
    if not Path(ground_truth_file).exists():
        logger.error(f"Ground truth file not found: {ground_truth_file}")
        return
    
    print(f"📁 Loading data from: {tables_file}")
    print(f"📁 Ground truth from: {ground_truth_file}")
    print(f"🔧 Max queries: {max_queries}")
    print(f"⚡ Parallel workers: {parallel_workers}")
    print()
    
    # 创建协调器
    print("🤖 Initializing Multi-Agent Orchestrator...")
    orchestrator = MultiAgentOrchestrator('config_optimized.yml')
    
    # 加载数据
    print("📥 Loading dataset...")
    await orchestrator.load_data(tables_file)
    
    # 加载ground truth
    print("📥 Loading ground truth...")
    ground_truth_data = await load_ground_truth(ground_truth_file)
    
    if not ground_truth_data:
        logger.error("No ground truth data found")
        return
    
    print(f"✅ Loaded {len(ground_truth_data)} ground truth queries")
    
    # 创建查询任务
    query_tasks = []
    
    # 分别统计JOIN和UNION
    join_count = 0
    union_count = 0
    
    for i, gt_item in enumerate(ground_truth_data[:max_queries]):
        task_type = gt_item.get('type', 'join')
        
        if task_type == 'join':
            query_id = f"join_{join_count}"
            join_count += 1
        else:
            query_id = f"union_{union_count}"
            union_count += 1
        
        task = QueryTask(
            query_id=query_id,
            query_table=gt_item['query_table'],
            task_type=task_type,
            ground_truth=gt_item['ground_truth']
        )
        query_tasks.append(task)
    
    print(f"\n📋 Created {len(query_tasks)} query tasks:")
    print(f"   - JOIN queries: {join_count}")
    print(f"   - UNION queries: {union_count}")
    
    # 运行测试
    print(f"\n🏃 Running multi-agent processing with {parallel_workers} parallel workers...")
    print("   This may take a few minutes...\n")
    
    start_time = time.time()
    
    # 批量处理
    results = await orchestrator.process_batch(query_tasks, parallel_workers=parallel_workers)
    
    total_time = time.time() - start_time
    
    # 获取系统指标
    metrics = orchestrator.get_metrics()
    agent_stats = orchestrator.get_agent_stats()
    
    # 输出结果
    print("\n" + "="*70)
    print("📊 EVALUATION RESULTS")
    print("="*70)
    
    print(f"\n⏱️  Performance Metrics:")
    print(f"   Total Time: {total_time:.2f} seconds")
    print(f"   Queries Processed: {metrics['total_queries']}")
    print(f"   Successful: {metrics['successful_queries']}")
    print(f"   Failed: {metrics['failed_queries']}")
    print(f"   Success Rate: {metrics['successful_queries']/max(metrics['total_queries'], 1)*100:.1f}%")
    print(f"   Avg Response Time: {metrics['avg_response_time']:.3f}s")
    print(f"   Throughput: {metrics['total_queries']/max(total_time, 0.001):.2f} QPS")
    
    print(f"\n🎯 Accuracy Metrics:")
    print(f"   Precision: {metrics['precision']:.3f}")
    print(f"   Recall: {metrics['recall']:.3f}")
    print(f"   F1-Score: {metrics['f1_score']:.3f}")
    print(f"   MRR: {metrics['mrr']:.3f}")
    
    if metrics['hit_at_k']:
        print(f"\n📈 Hit@K Metrics:")
        for k in sorted(metrics['hit_at_k'].keys()):
            print(f"   Hit@{k}: {metrics['hit_at_k'][k]:.3f}")
    
    print(f"\n🤖 Agent Performance:")
    for agent_name, stats in agent_stats.items():
        if stats['processed'] > 0:
            success_rate = stats['success'] / stats['processed'] * 100
            print(f"   {agent_name}:")
            print(f"      Processed: {stats['processed']}")
            print(f"      Success Rate: {success_rate:.1f}%")
            print(f"      Avg Time: {stats['avg_time']:.3f}s")
    
    # 分析JOIN和UNION分别的性能
    join_results = {k: v for k, v in results.items() if 'join' in k}
    union_results = {k: v for k, v in results.items() if 'union' in k}
    
    if join_results:
        print(f"\n🔗 JOIN Task Performance:")
        join_success = sum(1 for v in join_results.values() if v)
        print(f"   Total: {len(join_results)}")
        print(f"   Success: {join_success}")
        print(f"   Success Rate: {join_success/len(join_results)*100:.1f}%")
    
    if union_results:
        print(f"\n🔀 UNION Task Performance:")
        union_success = sum(1 for v in union_results.values() if v)
        print(f"   Total: {len(union_results)}")
        print(f"   Success: {union_success}")
        print(f"   Success Rate: {union_success/len(union_results)*100:.1f}%")
    
    # 保存详细结果
    timestamp = int(time.time())
    output_dir = Path('experiment_results/multi_agent')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_file = output_dir / f"test_{dataset}_{max_queries}queries_{timestamp}.json"
    
    # 准备保存的数据
    save_data = {
        'config': {
            'dataset': dataset,
            'max_queries': max_queries,
            'parallel_workers': parallel_workers,
            'total_time': total_time
        },
        'metrics': metrics,
        'agent_stats': agent_stats,
        'task_breakdown': {
            'join': {
                'total': len(join_results),
                'success': sum(1 for v in join_results.values() if v)
            },
            'union': {
                'total': len(union_results),
                'success': sum(1 for v in union_results.values() if v)
            }
        },
        'sample_results': {}
    }
    
    # 添加一些样本结果
    for query_id in list(results.keys())[:5]:
        if results[query_id]:
            save_data['sample_results'][query_id] = [
                {
                    'matched_table': r.matched_table,
                    'score': r.score,
                    'agent_used': r.agent_used
                }
                for r in results[query_id][:3]
            ]
    
    # 保存到文件
    with open(output_file, 'w') as f:
        json.dump(save_data, f, indent=2, default=str)
    
    print(f"\n💾 Detailed results saved to: {output_file}")
    print("\n" + "="*70)
    print("✅ TEST COMPLETED SUCCESSFULLY!")
    print("="*70 + "\n")
    
    return metrics

async def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Test Real Multi-Agent System')
    parser.add_argument('--dataset', choices=['subset', 'complete'], 
                       default='subset', help='Dataset to use')
    parser.add_argument('--queries', type=int, default=100, 
                       help='Maximum number of queries to test')
    parser.add_argument('--workers', type=int, default=4,
                       help='Number of parallel workers')
    
    args = parser.parse_args()
    
    # 运行测试
    await test_multi_agent_system(
        dataset=args.dataset,
        max_queries=args.queries,
        parallel_workers=args.workers
    )

if __name__ == "__main__":
    asyncio.run(main())