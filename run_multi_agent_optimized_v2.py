#!/usr/bin/env python
"""
优化版多智能体系统 - 3大优化策略
1. 减少LLM候选数量（20→10）
2. 提高并发限制（10→20）
3. 添加缓存机制
"""

import asyncio
import json
import time
import hashlib
from typing import List, Dict, Any
import os
import sys

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from run_multi_agent_llm_enabled import MultiAgentSystem, MatchResult

class OptimizedMultiAgentSystem(MultiAgentSystem):
    """优化版多智能体系统"""
    
    def __init__(self, config_path: str = "config_optimized.yml"):
        super().__init__(config_path)
        # LLM响应缓存
        self.llm_cache = {}
        self.cache_hits = 0
        self.cache_misses = 0
        
    async def _matcher_agent(self, query_table: Dict, candidates: List[tuple], 
                           task_type: str, strategy: Dict) -> List[MatchResult]:
        """优化的匹配Agent - 减少候选数量，增加并发"""
        matches = []
        
        # 快速通过高分候选
        for table_name, score in candidates[:5]:  # 前5个直接通过
            if score > 0.95:
                matches.append(MatchResult(
                    query_table=query_table['table_name'],
                    matched_table=table_name,
                    score=score,
                    match_type=task_type,
                    agent_used='Matcher_HighScore'
                ))
        
        # 优化1: 减少LLM候选数量（20→10）
        if strategy.get('use_llm', True) and hasattr(self, 'llm_matcher'):
            llm_candidates = [
                (name, score) for name, score in candidates[len(matches):15]  # 只取前10个
                if 0.1 <= score <= 0.95
            ]
            
            if llm_candidates:
                logger.info(f"Using LLM to verify {len(llm_candidates)} candidates (optimized)")
                
                # 收集所有LLM任务
                all_llm_tasks = []
                all_task_info = []
                
                for table_name, base_score in llm_candidates:
                    candidate_table = self.table_cache.get(table_name)
                    if not candidate_table:
                        continue
                    
                    # 准备LLM输入
                    query_schema = {
                        'table_name': query_table['table_name'],
                        'columns': [
                            {
                                'name': col.get('column_name', col.get('name', '')),
                                'type': col.get('data_type', col.get('type', ''))
                            }
                            for col in query_table['columns'][:5]  # 减少列数
                        ]
                    }
                    
                    candidate_schema = {
                        'table_name': table_name,
                        'columns': [
                            {
                                'name': col.get('column_name', col.get('name', '')),
                                'type': col.get('data_type', col.get('type', ''))
                            }
                            for col in candidate_table['columns'][:5]  # 减少列数
                        ]
                    }
                    
                    all_llm_tasks.append(
                        self._call_llm_matcher_cached(query_schema, candidate_schema, task_type)
                    )
                    all_task_info.append((table_name, base_score))
                
                # 优化2: 增加并发限制（10→20）
                if all_llm_tasks:
                    logger.info(f"Executing {len(all_llm_tasks)} LLM calls with higher concurrency...")
                    
                    # 增加超时时间
                    timeout_tasks = []
                    for task in all_llm_tasks:
                        timeout_tasks.append(
                            asyncio.wait_for(task, timeout=20.0)  # 减少超时时间
                        )
                    
                    start_llm = time.time()
                    try:
                        # 优化3: 提高并发数
                        max_concurrent = 20  # 增加并发数
                        llm_results = []
                        
                        if len(timeout_tasks) > max_concurrent:
                            # 如果超过最大并发，分批
                            for i in range(0, len(timeout_tasks), max_concurrent):
                                batch_tasks = timeout_tasks[i:i+max_concurrent]
                                batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
                                llm_results.extend(batch_results)
                        else:
                            # 一次性执行所有
                            llm_results = await asyncio.gather(*timeout_tasks, return_exceptions=True)
                            
                    except Exception as e:
                        logger.error(f"LLM calls failed: {e}")
                        llm_results = [e] * len(timeout_tasks)
                    
                    llm_time = time.time() - start_llm
                    logger.info(f"LLM calls completed in {llm_time:.2f}s")
                    logger.info(f"Cache stats: {self.cache_hits} hits, {self.cache_misses} misses")
                    
                    # 处理结果
                    for j, llm_result in enumerate(llm_results):
                        if j >= len(all_task_info):
                            break
                        table_name, base_score = all_task_info[j]
                        
                        if isinstance(llm_result, Exception):
                            # 超时或错误，使用基础分数
                            if base_score > 0.4:  # 降低阈值
                                matches.append(MatchResult(
                                    query_table=query_table['table_name'],
                                    matched_table=table_name,
                                    score=base_score * 0.8,
                                    match_type=task_type,
                                    agent_used='Matcher_Fallback'
                                ))
                        elif llm_result and llm_result.get('is_match', False):
                            llm_score = llm_result.get('confidence', 0.5)
                            final_score = base_score * 0.4 + llm_score * 0.6
                            
                            matches.append(MatchResult(
                                query_table=query_table['table_name'],
                                matched_table=table_name,
                                score=final_score,
                                match_type=task_type,
                                agent_used='Matcher_LLM',
                                evidence=llm_result
                            ))
        
        # 规则匹配作为后备
        if len(matches) < 5:  # 确保至少有5个结果
            for table_name, score in candidates[:10]:
                if score > 0.3 and not any(m.matched_table == table_name for m in matches):
                    matches.append(MatchResult(
                        query_table=query_table['table_name'],
                        matched_table=table_name,
                        score=score * 0.9,
                        match_type=task_type,
                        agent_used='Matcher_Rule'
                    ))
        
        return matches[:10]  # 只返回前10个
    
    async def _call_llm_matcher_cached(self, query_schema: Dict, candidate_schema: Dict, task_type: str) -> Dict:
        """带缓存的LLM调用"""
        # 生成缓存键
        cache_key = hashlib.md5(
            f"{query_schema['table_name']}_{candidate_schema['table_name']}_{task_type}".encode()
        ).hexdigest()
        
        # 检查缓存
        if cache_key in self.llm_cache:
            self.cache_hits += 1
            return self.llm_cache[cache_key]
        
        self.cache_misses += 1
        
        # 调用原始方法
        result = await self._call_llm_matcher(query_schema, candidate_schema, task_type)
        
        # 存入缓存
        self.llm_cache[cache_key] = result
        
        return result


async def main():
    """主函数"""
    import argparse
    import logging
    
    # 配置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    global logger
    logger = logging.getLogger(__name__)
    
    parser = argparse.ArgumentParser(description='优化版多智能体系统')
    parser.add_argument('--dataset', choices=['subset', 'complete'], default='subset')
    parser.add_argument('--queries', type=int, default=5, help='查询数量')
    parser.add_argument('--workers', type=int, default=2, help='并行工作线程数')
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("🚀 OPTIMIZED MULTI-AGENT SYSTEM V2")
    print("=" * 70)
    print(f"📊 Dataset: {args.dataset.upper()}")
    print(f"🔧 Max queries: {args.queries}")
    print(f"⚡ Parallel workers: {args.workers}")
    print(f"🎯 Optimizations:")
    print(f"   - Reduced candidates: 20 → 10")
    print(f"   - Increased concurrency: 10 → 20")
    print(f"   - Added LLM caching")
    print(f"   - Reduced column count: 10 → 5")
    print()
    
    # 初始化系统
    system = OptimizedMultiAgentSystem()
    
    # 加载数据
    if args.dataset == 'subset':
        tables_path = "examples/final_subset_tables.json"
        ground_truth_path = "examples/ground_truth_subset.json"
    else:
        tables_path = "examples/final_complete_tables.json"
        ground_truth_path = "examples/ground_truth_complete.json"
    
    print("📥 Loading dataset...")
    await system.load_data(tables_path)
    
    # 加载ground truth
    print("📥 Loading ground truth...")
    with open(ground_truth_path, 'r') as f:
        ground_truth = json.load(f)
    
    # 创建查询任务
    from src.core.models import QueryTask
    query_tasks = []
    
    for query_id, gt in list(ground_truth.items())[:args.queries]:
        task_type = gt.get('task_type', 'join')
        query_tasks.append(QueryTask(
            query_id=query_id,
            query_table=gt['query_table'],
            task_type=task_type,
            ground_truth=gt.get('ground_truth', [])
        ))
    
    print(f"📋 Created {len(query_tasks)} query tasks")
    
    # 运行实验
    print("\n🏃 Running optimized multi-agent processing...")
    start_time = time.time()
    
    # 并行处理查询
    results = await system.process_queries_parallel(query_tasks, max_workers=args.workers)
    
    total_time = time.time() - start_time
    
    # 计算指标
    from src.evaluation.metrics import calculate_metrics
    
    all_predictions = {}
    for query_task, matches in results:
        if matches:
            all_predictions[query_task.query_id] = [m.matched_table for m in matches[:10]]
        else:
            all_predictions[query_task.query_id] = []
    
    metrics = calculate_metrics(all_predictions, ground_truth)
    
    # 显示结果
    print("\n" + "=" * 70)
    print("📊 OPTIMIZED RESULTS")
    print("=" * 70)
    
    print("\n⏱️  Performance:")
    print(f"   Total Time: {total_time:.2f}s")
    print(f"   Queries: {len(query_tasks)}")
    print(f"   Success: {system.successful_queries}")
    print(f"   Success Rate: {system.successful_queries/len(query_tasks)*100:.1f}%")
    
    if system.query_times:
        avg_time = sum(system.query_times) / len(system.query_times)
        print(f"   Avg Response Time: {avg_time:.3f}s")
        print(f"   Throughput: {len(query_tasks)/total_time:.2f} QPS")
    
    print(f"\n💾 Cache Performance:")
    total_cache_ops = system.cache_hits + system.cache_misses
    if total_cache_ops > 0:
        print(f"   Cache Hits: {system.cache_hits}")
        print(f"   Cache Misses: {system.cache_misses}")
        print(f"   Hit Rate: {system.cache_hits/total_cache_ops*100:.1f}%")
    
    print(f"\n🎯 Accuracy:")
    print(f"   Precision: {metrics.get('precision', 0):.3f}")
    print(f"   Recall: {metrics.get('recall', 0):.3f}")
    print(f"   F1-Score: {metrics.get('f1', 0):.3f}")
    print(f"   MRR: {metrics.get('mrr', 0):.3f}")
    
    print(f"\n📈 Hit@K:")
    for k in [1, 3, 5, 10]:
        print(f"   Hit@{k}: {metrics.get(f'hit@{k}', 0):.3f}")
    
    # 保存结果
    timestamp = int(time.time())
    result_file = f"experiment_results/optimized_v2/experiment_{args.dataset}_{args.queries}q_{timestamp}.json"
    os.makedirs(os.path.dirname(result_file), exist_ok=True)
    
    with open(result_file, 'w') as f:
        json.dump({
            'config': {
                'dataset': args.dataset,
                'queries': len(query_tasks),
                'workers': args.workers,
                'optimizations': [
                    'reduced_candidates',
                    'increased_concurrency',
                    'llm_caching',
                    'reduced_columns'
                ]
            },
            'performance': {
                'total_time': total_time,
                'avg_response_time': avg_time if system.query_times else 0,
                'throughput': len(query_tasks)/total_time,
                'cache_hits': system.cache_hits,
                'cache_misses': system.cache_misses
            },
            'metrics': metrics,
            'predictions': all_predictions
        }, f, indent=2)
    
    print(f"\n💾 Results saved to: {result_file}")
    print("\n✅ OPTIMIZED EXPERIMENT COMPLETED!")
    print("=" * 70)

if __name__ == "__main__":
    asyncio.run(main())