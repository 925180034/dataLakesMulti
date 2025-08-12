#!/usr/bin/env python
"""
完整的多智能体数据湖发现系统测试
Complete Multi-Agent Data Lake Discovery System Test
"""

import json
import time
import numpy as np
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass, field, asdict
from pathlib import Path
import logging
import os
import sys

# 设置Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ===================== 数据结构 =====================

@dataclass
class QueryTask:
    """查询任务"""
    query_id: str
    query_table: str
    task_type: str  # 'join' or 'union'
    ground_truth: List[str]

@dataclass
class MatchResult:
    """匹配结果"""
    query_table: str
    matched_table: str
    score: float
    match_type: str
    method: str = ""

@dataclass
class EvaluationMetrics:
    """评价指标"""
    precision: float = 0.0
    recall: float = 0.0
    f1_score: float = 0.0
    hit_at_1: float = 0.0
    hit_at_3: float = 0.0
    hit_at_5: float = 0.0
    hit_at_10: float = 0.0
    mrr: float = 0.0
    success_rate: float = 0.0
    avg_time: float = 0.0

# ===================== 多Agent系统（同步版本） =====================

class SimpleMultiAgentSystem:
    """简化的多智能体系统（同步版本）"""
    
    def __init__(self):
        """初始化系统"""
        logger.info("Initializing Simple Multi-Agent System...")
        
        # 初始化工具
        self._init_tools()
        
        # 数据缓存
        self.tables = {}
        self.embeddings = {}
        self.metadata_index = {}
        
        # 统计信息
        self.query_count = 0
        self.success_count = 0
        self.query_times = []
        
        logger.info("System initialized")
        
    def _init_tools(self):
        """初始化工具层"""
        try:
            # 导入工具
            from src.core.ultra_optimized_workflow import UltraOptimizedWorkflow
            from src.tools.metadata_filter import MetadataFilter
            from src.tools.hnsw_search import HNSWVectorSearch
            from src.tools.smart_llm_matcher import SmartLLMMatcher
            from src.utils.llm_client import GeminiClient
            
            # 初始化工作流
            self.workflow = UltraOptimizedWorkflow()
            
            # 初始化工具
            self.metadata_filter = MetadataFilter()
            self.vector_search = HNSWVectorSearch(dimension=384)
            
            # 初始化LLM客户端
            llm_config = {
                "model_name": "gemini-1.5-flash",
                "temperature": 0.1,
                "max_tokens": 2000
            }
            self.llm_client = GeminiClient(llm_config)
            self.llm_matcher = SmartLLMMatcher(self.llm_client)
            
            logger.info("Tools initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize tools: {e}")
            raise
            
    def load_data(self, tables_path: str):
        """加载数据集"""
        logger.info(f"Loading data from {tables_path}")
        
        with open(tables_path, 'r') as f:
            tables_data = json.load(f)
        
        # 存储表信息
        for table in tables_data:
            self.tables[table['table_name']] = table
            
            # 构建元数据索引
            col_count = len(table['columns'])
            if col_count not in self.metadata_index:
                self.metadata_index[col_count] = []
            self.metadata_index[col_count].append(table['table_name'])
        
        logger.info(f"Loaded {len(self.tables)} tables")
        
        # 初始化工作流
        if hasattr(self.workflow, 'initialize'):
            self.workflow.initialize(tables_data)
            logger.info("Workflow initialized with table data")
            
    def process_query(self, query_task: QueryTask) -> List[MatchResult]:
        """处理单个查询（使用现有的工作流）"""
        start_time = time.time()
        self.query_count += 1
        
        try:
            query_table = self.tables.get(query_task.query_table)
            if not query_table:
                logger.error(f"Query table {query_task.query_table} not found")
                return []
            
            # 使用优化的工作流
            if hasattr(self.workflow, 'discover_tables'):
                results = self.workflow.discover_tables(
                    query_table=query_table,
                    task_type=query_task.task_type,
                    top_k=10
                )
                
                # 转换结果格式
                matches = []
                for result in results:
                    if isinstance(result, dict):
                        matches.append(MatchResult(
                            query_table=query_task.query_table,
                            matched_table=result.get('table_name', ''),
                            score=result.get('score', 0.0),
                            match_type=query_task.task_type,
                            method=result.get('method', 'workflow')
                        ))
                    elif isinstance(result, tuple):
                        table_name, score = result
                        matches.append(MatchResult(
                            query_table=query_task.query_table,
                            matched_table=table_name,
                            score=score,
                            match_type=query_task.task_type,
                            method='workflow'
                        ))
                
                if matches:
                    self.success_count += 1
                    
            else:
                # 简单的基于规则的匹配（备用）
                matches = self._simple_matching(query_table, query_task.task_type)
            
            query_time = time.time() - start_time
            self.query_times.append(query_time)
            
            logger.info(f"Processed {query_task.query_id} in {query_time:.3f}s, found {len(matches)} matches")
            
            return matches[:10]  # 返回Top-10
            
        except Exception as e:
            logger.error(f"Error processing query {query_task.query_id}: {e}")
            return []
            
    def _simple_matching(self, query_table: Dict, task_type: str) -> List[MatchResult]:
        """简单的基于规则的匹配（备用方案）"""
        matches = []
        query_col_count = len(query_table['columns'])
        
        # 基于列数的简单匹配
        for table_name, table in self.tables.items():
            if table_name == query_table['table_name']:
                continue
                
            col_count = len(table['columns'])
            
            # 列数相似
            if abs(col_count - query_col_count) <= 2:
                score = 1.0 - abs(col_count - query_col_count) / 10.0
                
                # 检查列名重叠
                query_col_names = {col['name'].lower() for col in query_table['columns']}
                table_col_names = {col['name'].lower() for col in table['columns']}
                overlap = len(query_col_names & table_col_names)
                
                if overlap > 0:
                    score += overlap / max(len(query_col_names), len(table_col_names))
                    
                if score > 0.3:
                    matches.append(MatchResult(
                        query_table=query_table['table_name'],
                        matched_table=table_name,
                        score=min(score, 1.0),
                        match_type=task_type,
                        method='simple_rule'
                    ))
        
        # 排序并返回
        matches.sort(key=lambda x: x.score, reverse=True)
        return matches
        
    def process_batch(self, query_tasks: List[QueryTask]) -> Dict[str, List[MatchResult]]:
        """批量处理查询"""
        logger.info(f"Processing batch of {len(query_tasks)} queries")
        
        results = {}
        for query_task in query_tasks:
            results[query_task.query_id] = self.process_query(query_task)
            
        return results
        
    def calculate_metrics(self, results: Dict[str, List[MatchResult]], 
                         query_tasks: List[QueryTask]) -> EvaluationMetrics:
        """计算评价指标"""
        metrics = EvaluationMetrics()
        
        total_precision = 0.0
        total_recall = 0.0
        hit_counts = {1: 0, 3: 0, 5: 0, 10: 0}
        mrr_sum = 0.0
        valid_queries = 0
        
        for query_task in query_tasks:
            query_results = results.get(query_task.query_id, [])
            
            if query_results:
                valid_queries += 1
                predicted = [r.matched_table for r in query_results]
                ground_truth = query_task.ground_truth
                
                if ground_truth:
                    # Precision和Recall
                    true_positives = len(set(predicted[:10]) & set(ground_truth))
                    precision = true_positives / min(10, len(predicted)) if predicted else 0
                    recall = true_positives / len(ground_truth)
                    
                    total_precision += precision
                    total_recall += recall
                    
                    # Hit@K
                    for k in [1, 3, 5, 10]:
                        if len(predicted) >= k:
                            if any(p in ground_truth for p in predicted[:k]):
                                hit_counts[k] += 1
                    
                    # MRR
                    for i, p in enumerate(predicted):
                        if p in ground_truth:
                            mrr_sum += 1.0 / (i + 1)
                            break
        
        # 计算平均值
        if valid_queries > 0:
            metrics.precision = total_precision / valid_queries
            metrics.recall = total_recall / valid_queries
            
            if metrics.precision + metrics.recall > 0:
                metrics.f1_score = 2 * metrics.precision * metrics.recall / (metrics.precision + metrics.recall)
            
            metrics.hit_at_1 = hit_counts[1] / valid_queries
            metrics.hit_at_3 = hit_counts[3] / valid_queries
            metrics.hit_at_5 = hit_counts[5] / valid_queries
            metrics.hit_at_10 = hit_counts[10] / valid_queries
            metrics.mrr = mrr_sum / valid_queries
        
        # 其他指标
        metrics.success_rate = self.success_count / max(self.query_count, 1)
        if self.query_times:
            metrics.avg_time = np.mean(self.query_times)
            
        return metrics

# ===================== 主函数 =====================

def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Test Complete Multi-Agent System')
    parser.add_argument('--dataset', choices=['subset', 'complete'], 
                       default='subset', help='Dataset to use')
    parser.add_argument('--queries', type=int, default=100, 
                       help='Number of queries to test')
    parser.add_argument('--task', choices=['join', 'union', 'both'], 
                       default='both', help='Task type to test')
    
    args = parser.parse_args()
    
    print("\n" + "="*70)
    print("🚀 COMPLETE MULTI-AGENT DATA LAKE DISCOVERY TEST")
    print("="*70)
    
    # 选择数据集
    if args.dataset == 'subset':
        tables_file = 'examples/final_subset_tables.json'
        ground_truth_file = 'examples/final_subset_ground_truth.json'
        print(f"📊 Dataset: SUBSET (100 tables)")
    else:
        tables_file = 'examples/final_complete_tables.json'
        ground_truth_file = 'examples/final_complete_ground_truth.json'
        print(f"📊 Dataset: COMPLETE (1534 tables)")
    
    print(f"🔧 Queries: {args.queries}")
    print(f"📋 Task: {args.task}")
    print()
    
    # 创建系统
    system = SimpleMultiAgentSystem()
    
    # 加载数据
    print("📥 Loading dataset...")
    system.load_data(tables_file)
    
    # 加载ground truth
    print("📥 Loading ground truth...")
    with open(ground_truth_file, 'r') as f:
        ground_truth_data = json.load(f)
    
    # 创建查询任务
    query_tasks = []
    
    if isinstance(ground_truth_data, dict):
        # 新格式：分别存储join和union
        if args.task in ['join', 'both'] and 'join_ground_truth' in ground_truth_data:
            for i, item in enumerate(ground_truth_data['join_ground_truth'][:args.queries//2 if args.task == 'both' else args.queries]):
                query_tasks.append(QueryTask(
                    query_id=f"join_{i}",
                    query_table=item['table'],
                    task_type='join',
                    ground_truth=item.get('ground_truth', item.get('expected', []))
                ))
                
        if args.task in ['union', 'both'] and 'union_ground_truth' in ground_truth_data:
            for i, item in enumerate(ground_truth_data['union_ground_truth'][:args.queries//2 if args.task == 'both' else args.queries]):
                query_tasks.append(QueryTask(
                    query_id=f"union_{i}",
                    query_table=item['table'],
                    task_type='union',
                    ground_truth=item.get('ground_truth', item.get('expected', []))
                ))
    
    if not query_tasks:
        print("❌ No query tasks created. Check ground truth format.")
        return
    
    print(f"📋 Created {len(query_tasks)} query tasks")
    
    # 分别统计
    join_count = sum(1 for t in query_tasks if t.task_type == 'join')
    union_count = sum(1 for t in query_tasks if t.task_type == 'union')
    if join_count > 0:
        print(f"   - JOIN: {join_count}")
    if union_count > 0:
        print(f"   - UNION: {union_count}")
    print()
    
    # 运行测试
    print("🏃 Processing queries...")
    start_time = time.time()
    
    results = system.process_batch(query_tasks)
    
    total_time = time.time() - start_time
    
    # 计算指标
    metrics = system.calculate_metrics(results, query_tasks)
    
    # 输出结果
    print("\n" + "="*70)
    print("📊 EVALUATION RESULTS")
    print("="*70)
    
    print(f"\n⏱️  Performance:")
    print(f"   Total Time: {total_time:.2f}s")
    print(f"   Queries Processed: {system.query_count}")
    print(f"   Successful: {system.success_count}")
    print(f"   Success Rate: {metrics.success_rate*100:.1f}%")
    print(f"   Avg Time/Query: {metrics.avg_time:.3f}s")
    print(f"   Throughput: {system.query_count/max(total_time, 0.001):.2f} QPS")
    
    print(f"\n🎯 Accuracy Metrics:")
    print(f"   Precision: {metrics.precision:.3f}")
    print(f"   Recall: {metrics.recall:.3f}")
    print(f"   F1-Score: {metrics.f1_score:.3f}")
    print(f"   MRR: {metrics.mrr:.3f}")
    
    print(f"\n📈 Hit@K Metrics:")
    print(f"   Hit@1: {metrics.hit_at_1:.3f}")
    print(f"   Hit@3: {metrics.hit_at_3:.3f}")
    print(f"   Hit@5: {metrics.hit_at_5:.3f}")
    print(f"   Hit@10: {metrics.hit_at_10:.3f}")
    
    # 分别统计JOIN和UNION性能
    if join_count > 0 and union_count > 0:
        print(f"\n📊 Task Breakdown:")
        
        # JOIN性能
        join_results = {k: v for k, v in results.items() if 'join' in k}
        join_success = sum(1 for v in join_results.values() if v)
        print(f"   JOIN: {join_success}/{join_count} ({join_success/max(join_count,1)*100:.1f}%)")
        
        # UNION性能
        union_results = {k: v for k, v in results.items() if 'union' in k}
        union_success = sum(1 for v in union_results.values() if v)
        print(f"   UNION: {union_success}/{union_count} ({union_success/max(union_count,1)*100:.1f}%)")
    
    # 保存结果
    timestamp = int(time.time())
    output_dir = Path('experiment_results/multi_agent')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_file = output_dir / f"complete_test_{args.dataset}_{args.queries}q_{timestamp}.json"
    
    save_data = {
        'config': {
            'dataset': args.dataset,
            'queries': args.queries,
            'task': args.task,
            'total_time': total_time
        },
        'metrics': asdict(metrics),
        'statistics': {
            'total_queries': system.query_count,
            'successful_queries': system.success_count,
            'join_queries': join_count,
            'union_queries': union_count
        }
    }
    
    with open(output_file, 'w') as f:
        json.dump(save_data, f, indent=2, default=str)
    
    print(f"\n💾 Results saved to: {output_file}")
    print("\n✅ TEST COMPLETED SUCCESSFULLY!")
    print("="*70 + "\n")

if __name__ == "__main__":
    main()