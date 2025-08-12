#!/usr/bin/env python
"""
修复版多智能体数据湖发现系统
Fixed Multi-Agent Data Lake Discovery System
"""

import json
import time
import numpy as np
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass, field, asdict
from pathlib import Path
import logging
import asyncio
import sys
import os

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
    timestamp: float = field(default_factory=time.time)

@dataclass
class MatchResult:
    """匹配结果"""
    query_table: str
    matched_table: str
    score: float
    match_type: str
    agent_used: str = ""
    evidence: Dict[str, Any] = field(default_factory=dict)

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
    avg_response_time: float = 0.0
    throughput: float = 0.0

# ===================== 多Agent系统（修复版） =====================

class FixedMultiAgentSystem:
    """修复版多智能体系统 - 正确处理异步调用"""
    
    def __init__(self, config_path: str = 'config_optimized.yml'):
        """初始化系统"""
        logger.info("Initializing Fixed Multi-Agent System...")
        
        # 加载配置
        self.config = self._load_config(config_path)
        
        # 初始化工具层
        self._init_tools()
        
        # 初始化Agent
        self._init_agents()
        
        # 数据缓存
        self.table_cache = {}
        self.embedding_cache = {}
        self.metadata_index = {}
        
        # 系统指标
        self.total_queries = 0
        self.successful_queries = 0
        self.query_times = []
        
        logger.info("Multi-Agent System initialized successfully")
        
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """加载配置"""
        import yaml
        if Path(config_path).exists():
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        return {}
        
    def _init_tools(self):
        """初始化工具层"""
        # Layer 1: 元数据过滤器
        from src.tools.metadata_filter import MetadataFilter
        self.metadata_filter = MetadataFilter()
        
        # Layer 2: 向量搜索
        from src.tools.vector_search import get_vector_search_engine
        self.vector_search = get_vector_search_engine()
        
        # Layer 3: LLM匹配器
        from src.tools.smart_llm_matcher import SmartLLMMatcher
        from src.utils.llm_client import GeminiClient
        
        llm_config = {
            "model_name": self.config.get('llm', {}).get('model_name', 'gemini-1.5-flash'),
            "temperature": self.config.get('llm', {}).get('temperature', 0.1),
            "max_tokens": self.config.get('llm', {}).get('max_tokens', 2000)
        }
        self.llm_client = GeminiClient(llm_config)
        self.llm_matcher = SmartLLMMatcher(self.llm_client)
        
        # 嵌入生成器
        from src.tools.embedding import get_embedding_generator
        self.embedding_gen = get_embedding_generator()
        
        logger.info("Tools initialized: MetadataFilter, VectorSearch, LLMMatcher")
        
    def _init_agents(self):
        """初始化Agent（简化版）"""
        self.agents = {
            'planner': self._planner_agent,
            'analyzer': self._analyzer_agent,
            'searcher': self._searcher_agent,
            'matcher': self._matcher_agent,
            'aggregator': self._aggregator_agent
        }
        logger.info(f"Initialized {len(self.agents)} agents")
        
    async def load_data(self, tables_path: str):
        """异步加载数据集"""
        logger.info(f"Loading data from {tables_path}")
        
        with open(tables_path, 'r') as f:
            tables_data = json.load(f)
        
        # 存储表信息
        for table in tables_data:
            self.table_cache[table['table_name']] = table
            
            # 构建元数据索引
            col_count = len(table['columns'])
            if col_count not in self.metadata_index:
                self.metadata_index[col_count] = []
            self.metadata_index[col_count].append(table['table_name'])
        
        logger.info(f"Loaded {len(self.table_cache)} tables")
        
        # 预计算嵌入（并行）
        await self._precompute_embeddings()
        
    async def _precompute_embeddings(self):
        """预计算嵌入向量（异步版本）"""
        logger.info("Precomputing embeddings...")
        
        batch_size = 20  # 减小批大小避免超时
        tables = list(self.table_cache.values())
        
        for i in range(0, len(tables), batch_size):
            batch = tables[i:i+batch_size]
            
            # 并行生成嵌入
            tasks = []
            for table in batch:
                if table['table_name'] not in self.embedding_cache:
                    tasks.append(self._generate_embedding_async(table))
            
            if tasks:
                # 等待所有任务完成
                results = await asyncio.gather(*tasks, return_exceptions=True)
                
                # 处理结果
                for table, result in zip(batch, results):
                    if isinstance(result, Exception):
                        logger.error(f"Failed to generate embedding for {table['table_name']}: {result}")
                    elif result is not None:
                        self.embedding_cache[table['table_name']] = result
        
        # 构建向量索引
        if self.embedding_cache:
            embeddings_list = []
            table_names = []
            
            for name, embedding in self.embedding_cache.items():
                if embedding is not None:
                    embeddings_list.append(embedding)
                    table_names.append(name)
            
            if embeddings_list:
                embeddings_array = np.array(embeddings_list)
                if hasattr(self.vector_search, 'build_index'):
                    self.vector_search.build_index(embeddings_array, table_names)
                    
        logger.info(f"Computed {len(self.embedding_cache)} embeddings")
        
    async def _generate_embedding_async(self, table: Dict[str, Any]):
        """异步生成表嵌入"""
        try:
            # 检查嵌入生成器的方法
            method = self.embedding_gen.generate_table_embedding
            
            # 如果是异步方法，使用await
            import inspect
            if inspect.iscoroutinefunction(method):
                return await method(table)
            else:
                # 如果是同步方法，在线程池中运行
                loop = asyncio.get_event_loop()
                return await loop.run_in_executor(None, method, table)
        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            return None
            
    async def process_query(self, query_task: QueryTask) -> List[MatchResult]:
        """处理单个查询（异步版本）"""
        start_time = time.time()
        self.total_queries += 1
        
        try:
            query_table = self.table_cache.get(query_task.query_table)
            if not query_table:
                logger.error(f"Query table {query_task.query_table} not found")
                return []
            
            # 1. 规划策略
            strategy = await self._planner_agent(query_task)
            
            # 2. 分析表结构
            analysis = await self._analyzer_agent(query_table, query_task.task_type)
            
            # 3. 搜索候选
            candidates = await self._searcher_agent(query_table, strategy, analysis)
            
            # 4. 精确匹配
            matches = await self._matcher_agent(query_table, candidates, query_task.task_type)
            
            # 5. 聚合结果
            final_results = await self._aggregator_agent(matches)
            
            # 记录成功
            self.successful_queries += 1
            query_time = time.time() - start_time
            self.query_times.append(query_time)
            
            logger.info(f"Processed {query_task.query_id} in {query_time:.2f}s, found {len(final_results)} matches")
            
            return final_results
            
        except Exception as e:
            logger.error(f"Error processing query {query_task.query_id}: {e}")
            return []
            
    async def _planner_agent(self, query_task: QueryTask) -> Dict[str, Any]:
        """规划Agent - 制定查询策略"""
        if query_task.task_type == 'join':
            return {
                'name': 'join_strategy',
                'use_metadata': True,
                'use_vector': True,
                'use_llm': False,  # 暂时禁用LLM以加快速度
                'top_k': 100
            }
        elif query_task.task_type == 'union':
            return {
                'name': 'union_strategy',
                'use_metadata': True,
                'use_vector': True,
                'use_llm': False,
                'top_k': 100
            }
        else:
            return {
                'name': 'general_strategy',
                'use_metadata': True,
                'use_vector': True,
                'use_llm': False,
                'top_k': 50
            }
            
    async def _analyzer_agent(self, table: Dict[str, Any], task_type: str) -> Dict[str, Any]:
        """分析Agent - 理解表结构"""
        analysis = {
            'column_count': len(table['columns']),
            'column_types': {},
            'key_columns': [],
            'table_type': 'unknown'
        }
        
        # 分析列类型
        for col in table['columns']:
            col_type = col.get('data_type', col.get('type', 'unknown'))
            analysis['column_types'][col_type] = analysis['column_types'].get(col_type, 0) + 1
            
            # 识别关键列
            col_name = col.get('column_name', col.get('name', '')).lower()
            if any(key in col_name for key in ['_id', '_key', '_code']):
                analysis['key_columns'].append(col_name)
        
        # 识别表类型
        table_name = table['table_name'].lower()
        if 'dim_' in table_name or '_dim' in table_name:
            analysis['table_type'] = 'dimension'
        elif 'fact_' in table_name or '_fact' in table_name:
            analysis['table_type'] = 'fact'
            
        return analysis
        
    async def _searcher_agent(self, query_table: Dict[str, Any], 
                            strategy: Dict[str, Any], 
                            analysis: Dict[str, Any]) -> List[Tuple[str, float]]:
        """搜索Agent - 查找候选表"""
        candidates = []
        
        # Layer 1: 元数据筛选
        if strategy.get('use_metadata', True):
            try:
                # 根据列数查找相似表
                col_count = analysis['column_count']
                similar_tables = []
                
                for count in range(col_count - 2, col_count + 3):
                    if count in self.metadata_index:
                        similar_tables.extend(self.metadata_index[count])
                
                # 计算元数据相似度
                for table_name in similar_tables[:1000]:
                    if table_name != query_table['table_name']:
                        candidates.append((table_name, 0.5))  # 简化分数
                        
                logger.debug(f"Metadata filter found {len(candidates)} candidates")
                
            except Exception as e:
                logger.error(f"Metadata search failed: {e}")
        
        # Layer 2: 向量搜索
        if strategy.get('use_vector', True) and query_table['table_name'] in self.embedding_cache:
            try:
                query_embedding = self.embedding_cache[query_table['table_name']]
                
                if query_embedding is not None and hasattr(self.vector_search, 'search'):
                    # 执行向量搜索
                    vector_results = self.vector_search.search(
                        query_embedding, 
                        top_k=strategy.get('top_k', 100)
                    )
                    
                    for result in vector_results:
                        if isinstance(result, dict):
                            table_name = result.get('table_name', result.get('name'))
                            score = result.get('score', 0.0)
                        else:
                            table_name, score = result
                            
                        if table_name != query_table['table_name']:
                            candidates.append((table_name, score))
                            
                    logger.debug(f"Vector search found {len(vector_results)} candidates")
                    
            except Exception as e:
                logger.error(f"Vector search failed: {e}")
        
        # 去重和排序
        candidate_dict = {}
        for table_name, score in candidates:
            if table_name in candidate_dict:
                candidate_dict[table_name] = max(candidate_dict[table_name], score)
            else:
                candidate_dict[table_name] = score
        
        sorted_candidates = sorted(
            candidate_dict.items(), 
            key=lambda x: x[1], 
            reverse=True
        )
        
        return sorted_candidates[:strategy.get('top_k', 100)]
        
    async def _matcher_agent(self, query_table: Dict[str, Any],
                           candidates: List[Tuple[str, float]],
                           task_type: str) -> List[MatchResult]:
        """匹配Agent - 精确验证"""
        matches = []
        
        # 批量处理候选
        batch_size = 10
        for i in range(0, min(len(candidates), 30), batch_size):  # 最多处理30个
            batch = candidates[i:i+batch_size]
            
            for table_name, score in batch:
                # 高分直接通过
                if score > 0.9:
                    matches.append(MatchResult(
                        query_table=query_table['table_name'],
                        matched_table=table_name,
                        score=score,
                        match_type=task_type,
                        agent_used='Matcher_Rule'
                    ))
                # 中等分数降低权重
                elif score > 0.3:
                    matches.append(MatchResult(
                        query_table=query_table['table_name'],
                        matched_table=table_name,
                        score=score * 0.9,
                        match_type=task_type,
                        agent_used='Matcher_Rule'
                    ))
        
        return matches
        
    async def _aggregator_agent(self, matches: List[MatchResult]) -> List[MatchResult]:
        """聚合Agent - 整合结果"""
        # 按分数排序
        sorted_matches = sorted(matches, key=lambda x: x.score, reverse=True)
        
        # 添加排名
        for i, match in enumerate(sorted_matches[:10]):
            match.evidence['rank'] = i + 1
            
        return sorted_matches[:10]
        
    async def process_batch(self, query_tasks: List[QueryTask], 
                          parallel_workers: int = 4) -> Dict[str, List[MatchResult]]:
        """批量处理查询（异步并行）"""
        logger.info(f"Processing {len(query_tasks)} queries with {parallel_workers} workers")
        
        results = {}
        
        # 创建任务
        tasks = []
        for query_task in query_tasks:
            tasks.append(self.process_query(query_task))
        
        # 并行执行（限制并发数）
        for i in range(0, len(tasks), parallel_workers):
            batch_tasks = tasks[i:i+parallel_workers]
            batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
            
            for j, query_task in enumerate(query_tasks[i:i+parallel_workers]):
                if isinstance(batch_results[j], Exception):
                    logger.error(f"Failed to process {query_task.query_id}: {batch_results[j]}")
                    results[query_task.query_id] = []
                else:
                    results[query_task.query_id] = batch_results[j]
        
        return results
        
    def calculate_metrics(self, results: Dict[str, List[MatchResult]], 
                         query_tasks: List[QueryTask]) -> EvaluationMetrics:
        """计算评价指标"""
        metrics = EvaluationMetrics()
        
        precision_sum = 0.0
        recall_sum = 0.0
        hit_at_k = {1: 0, 3: 0, 5: 0, 10: 0}
        mrr_sum = 0.0
        valid_queries = 0
        
        for query_task in query_tasks:
            query_results = results.get(query_task.query_id, [])
            if not query_results:
                continue
                
            valid_queries += 1
            predicted = [r.matched_table for r in query_results]
            ground_truth = query_task.ground_truth
            
            # 计算Precision和Recall
            if predicted and ground_truth:
                true_positives = len(set(predicted[:10]) & set(ground_truth))
                precision = true_positives / min(10, len(predicted)) if predicted else 0
                recall = true_positives / len(ground_truth) if ground_truth else 0
                
                precision_sum += precision
                recall_sum += recall
                
                # Hit@K
                for k in [1, 3, 5, 10]:
                    if len(predicted) >= k:
                        if any(p in ground_truth for p in predicted[:k]):
                            hit_at_k[k] += 1
                
                # MRR
                for i, p in enumerate(predicted):
                    if p in ground_truth:
                        mrr_sum += 1.0 / (i + 1)
                        break
        
        if valid_queries > 0:
            metrics.precision = precision_sum / valid_queries
            metrics.recall = recall_sum / valid_queries
            
            if metrics.precision + metrics.recall > 0:
                metrics.f1_score = 2 * metrics.precision * metrics.recall / (metrics.precision + metrics.recall)
            
            metrics.hit_at_1 = hit_at_k[1] / valid_queries
            metrics.hit_at_3 = hit_at_k[3] / valid_queries
            metrics.hit_at_5 = hit_at_k[5] / valid_queries
            metrics.hit_at_10 = hit_at_k[10] / valid_queries
            metrics.mrr = mrr_sum / valid_queries
            
            if self.query_times:
                metrics.avg_response_time = np.mean(self.query_times)
                total_time = sum(self.query_times)
                metrics.throughput = len(self.query_times) / total_time if total_time > 0 else 0
        
        return metrics

# ===================== 主函数 =====================

async def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Fixed Multi-Agent Data Lake Discovery Experiment')
    parser.add_argument('--dataset', choices=['subset', 'complete'], 
                       default='subset', help='Dataset to use')
    parser.add_argument('--queries', type=int, default=100, 
                       help='Number of queries to test')
    parser.add_argument('--workers', type=int, default=4,
                       help='Number of parallel workers')
    
    args = parser.parse_args()
    
    print("\n" + "="*70)
    print("🚀 FIXED MULTI-AGENT DATA LAKE DISCOVERY SYSTEM")
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
    
    print(f"🔧 Max queries: {args.queries}")
    print(f"⚡ Parallel workers: {args.workers}")
    print()
    
    # 创建系统
    system = FixedMultiAgentSystem('config_optimized.yml')
    
    # 加载数据（异步）
    print("📥 Loading dataset...")
    await system.load_data(tables_file)
    
    # 加载ground truth
    print("📥 Loading ground truth...")
    with open(ground_truth_file, 'r') as f:
        ground_truth_data = json.load(f)
    
    # 创建查询任务
    query_tasks = []
    
    if isinstance(ground_truth_data, dict):
        # 新格式
        for task_type in ['join', 'union']:
            key = f'{task_type}_ground_truth'
            if key in ground_truth_data:
                for i, item in enumerate(ground_truth_data[key][:args.queries//2]):
                    query_tasks.append(QueryTask(
                        query_id=f"{task_type}_{i}",
                        query_table=item['table'],
                        task_type=task_type,
                        ground_truth=item.get('ground_truth', item.get('expected', []))
                    ))
    else:
        # 旧格式
        for i, item in enumerate(ground_truth_data[:args.queries]):
            task_type = item.get('query_type', 'join')
            query_table = item.get('query_table')
            
            if query_table:
                # 收集所有相同query_table的ground truth
                gt_tables = []
                for gt_item in ground_truth_data:
                    if gt_item.get('query_table') == query_table:
                        candidate = gt_item.get('candidate_table')
                        if candidate and candidate not in gt_tables:
                            gt_tables.append(candidate)
                
                if f"{task_type}_{query_table}" not in [t.query_id for t in query_tasks]:
                    query_tasks.append(QueryTask(
                        query_id=f"{task_type}_{len(query_tasks)}",
                        query_table=query_table,
                        task_type=task_type,
                        ground_truth=gt_tables[:10]  # 限制ground truth数量
                    ))
            
            if len(query_tasks) >= args.queries:
                break
    
    if not query_tasks:
        print("❌ No valid query tasks created")
        return
        
    print(f"📋 Created {len(query_tasks)} query tasks")
    
    # JOIN和UNION分别统计
    join_count = sum(1 for t in query_tasks if t.task_type == 'join')
    union_count = sum(1 for t in query_tasks if t.task_type == 'union')
    print(f"   - JOIN: {join_count}")
    print(f"   - UNION: {union_count}")
    print()
    
    # 运行实验（异步）
    print("🏃 Running multi-agent processing...")
    start_time = time.time()
    
    results = await system.process_batch(query_tasks, args.workers)
    
    total_time = time.time() - start_time
    
    # 计算指标
    metrics = system.calculate_metrics(results, query_tasks)
    
    # 输出结果
    print("\n" + "="*70)
    print("📊 EXPERIMENT RESULTS")
    print("="*70)
    
    print(f"\n⏱️  Performance:")
    print(f"   Total Time: {total_time:.2f}s")
    print(f"   Queries: {system.total_queries}")
    print(f"   Success: {system.successful_queries}")
    print(f"   Success Rate: {system.successful_queries/max(system.total_queries,1)*100:.1f}%")
    print(f"   Avg Response Time: {metrics.avg_response_time:.3f}s")
    print(f"   Throughput: {metrics.throughput:.2f} QPS")
    
    print(f"\n🎯 Accuracy:")
    print(f"   Precision: {metrics.precision:.3f}")
    print(f"   Recall: {metrics.recall:.3f}")
    print(f"   F1-Score: {metrics.f1_score:.3f}")
    print(f"   MRR: {metrics.mrr:.3f}")
    
    print(f"\n📈 Hit@K:")
    print(f"   Hit@1: {metrics.hit_at_1:.3f}")
    print(f"   Hit@3: {metrics.hit_at_3:.3f}")
    print(f"   Hit@5: {metrics.hit_at_5:.3f}")
    print(f"   Hit@10: {metrics.hit_at_10:.3f}")
    
    # 保存结果
    timestamp = int(time.time())
    output_dir = Path('experiment_results/multi_agent_fixed')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_file = output_dir / f"experiment_{args.dataset}_{args.queries}q_{timestamp}.json"
    
    save_data = {
        'config': {
            'dataset': args.dataset,
            'queries': args.queries,
            'workers': args.workers,
            'total_time': total_time
        },
        'metrics': asdict(metrics),
        'statistics': {
            'total_queries': system.total_queries,
            'successful_queries': system.successful_queries,
            'join_queries': join_count,
            'union_queries': union_count
        }
    }
    
    with open(output_file, 'w') as f:
        json.dump(save_data, f, indent=2, default=str)
    
    print(f"\n💾 Results saved to: {output_file}")
    print("\n✅ EXPERIMENT COMPLETED!")
    print("="*70 + "\n")

if __name__ == "__main__":
    asyncio.run(main())