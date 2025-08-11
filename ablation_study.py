#!/usr/bin/env python3
"""
Three-Layer Architecture Ablation Study
三层架构消融实验 - 验证每层的贡献和优化参数
"""

import json
import time
import asyncio
import argparse
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from tabulate import tabulate
import os

# Set environment for consistent testing
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['HF_DATASETS_OFFLINE'] = '1'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
# 确保LLM功能启用
os.environ['SKIP_LLM'] = 'false'


class AblationStudy:
    """三层架构消融实验主类"""
    
    def __init__(self, debug_mode=False, dataset_size="subset"):
        """初始化实验配置
        
        Args:
            debug_mode: 是否启用调试输出
            dataset_size: 数据集大小 ('subset' 或 'complete')
        """
        self.results_dir = Path("experiment_results/ablation")
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.debug_mode = debug_mode
        self.dataset_size = dataset_size
        
        # 导入最佳参数配置
        from optimal_parameters import JOIN_PARAMS, UNION_PARAMS
        
        # 根据数据集选择参数
        if dataset_size == "complete":
            # 使用优化后的参数
            self.configs = {
                "L1": {
                    "name": "L1_only",
                    "metadata_filter": True,
                    "vector_search": False,
                    "llm_matching": False,
                    "max_metadata_candidates": JOIN_PARAMS["L1"]["max_metadata_candidates"],
                },
                "L2": {
                    "name": "L2_only", 
                    "metadata_filter": False,
                    "vector_search": True,
                    "llm_matching": False,
                    "max_vector_candidates": 200,  # 单独L2需要更多候选
                    "vector_threshold": 0.4,
                },
                "L3": {
                    "name": "L3_only",
                    "metadata_filter": False,
                    "vector_search": False,
                    "llm_matching": True,
                    "max_llm_candidates": 30,  # 直接LLM
                },
                "L1+L2": {
                    "name": "L1_L2",
                    "metadata_filter": True,
                    "vector_search": True,
                    "llm_matching": False,
                    **JOIN_PARAMS["L1+L2"],  # 使用最佳参数
                },
                "L1+L3": {
                    "name": "L1_L3",
                    "metadata_filter": True,
                    "vector_search": False,
                    "llm_matching": True,
                    "max_metadata_candidates": 1000,
                    "max_llm_candidates": 30,
                },
                "L2+L3": {
                    "name": "L2_L3",
                    "metadata_filter": False,
                    "vector_search": True,
                    "llm_matching": True,
                    "max_vector_candidates": 150,
                    "max_llm_candidates": 20,
                    "vector_threshold": 0.5,
                },
                "L1+L2+L3": {
                    "name": "full_pipeline",
                    "metadata_filter": True,
                    "vector_search": True,
                    "llm_matching": True,
                    **JOIN_PARAMS["L1+L2+L3"],  # 使用最佳参数
                }
            }
        else:
            # Subset数据集使用简单参数
            self.configs = {
                "L1": {
                    "name": "L1_only",
                    "metadata_filter": True,
                    "vector_search": False,
                    "llm_matching": False,
                    "max_metadata_candidates": 200,
                },
                "L2": {
                    "name": "L2_only", 
                    "metadata_filter": False,
                    "vector_search": True,
                    "llm_matching": False,
                    "max_vector_candidates": 50,
                    "vector_threshold": 0.5,
                },
                "L3": {
                    "name": "L3_only",
                    "metadata_filter": False,
                    "vector_search": False,
                    "llm_matching": True,
                    "max_llm_candidates": 10,
                },
                "L1+L2": {
                    "name": "L1_L2",
                    "metadata_filter": True,
                    "vector_search": True,
                    "llm_matching": False,
                    "max_metadata_candidates": 100,
                    "max_vector_candidates": 30,
                    "vector_threshold": 0.5,
                },
                "L1+L3": {
                    "name": "L1_L3",
                    "metadata_filter": True,
                    "vector_search": False,
                    "llm_matching": True,
                    "max_metadata_candidates": 50,
                    "max_llm_candidates": 10,
                },
                "L2+L3": {
                    "name": "L2_L3",
                    "metadata_filter": False,
                    "vector_search": True,
                    "llm_matching": True,
                    "max_vector_candidates": 30,
                    "max_llm_candidates": 5,
                    "vector_threshold": 0.6,
                },
                "L1+L2+L3": {
                    "name": "full_pipeline",
                    "metadata_filter": True,
                    "vector_search": True,
                    "llm_matching": True,
                    "max_metadata_candidates": 100,
                    "max_vector_candidates": 30,
                    "max_llm_candidates": 5,
                    "early_stop_threshold": 0.90,
                }
            }
        
        # 性能统计
        self.perf_stats = {}
        
    async def run_single_configuration(
        self,
        config_name: str,
        task_type: str,
        dataset_size: str,
        max_queries: int = 10
    ) -> Dict[str, Any]:
        """运行单个配置的实验"""
        
        print(f"\n{'='*80}")
        print(f"🧪 Running Ablation: {config_name}")
        print(f"📊 Task: {task_type.upper()} | Dataset: {dataset_size}")
        print(f"🕐 Start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*80}\n")
        
        config = self.configs[config_name]
        
        # 加载数据
        tables, queries, ground_truth = await self.load_data(task_type, dataset_size)
        print(f"✅ Loaded {len(tables)} tables, {len(queries)} queries")
        
        # 限制查询数量
        if max_queries:
            queries = queries[:max_queries]
            print(f"🎯 Limited to {max_queries} queries")
        
        # 创建workflow
        workflow = await self.create_ablation_workflow(config)
        
        # 初始化（构建索引）
        init_start = time.time()
        await workflow.initialize(tables)
        init_time = time.time() - init_start
        print(f"✅ Workflow initialized in {init_time:.2f}s")
        
        # 运行查询并收集指标
        results = []
        layer_times = {"L1": [], "L2": [], "L3": [], "total": []}
        successful_queries = 0
        empty_predictions = 0  # 统计空预测
        
        print(f"\n🔍 Running {len(queries)} queries...")
        print("-" * 80)
        
        if self.debug_mode:
            print("\n🐛 调试模式已启用 - 显示详细匹配信息")
        
        for i, query in enumerate(queries, 1):
            try:
                # 运行单个查询
                result, timing = await self.run_single_query(
                    workflow, query, config, ground_truth
                )
                
                if result:
                    results.append(result)
                    successful_queries += 1
                    
                    # 检查是否有空预测
                    if not result.get('predictions'):
                        empty_predictions += 1
                    
                    # 收集时间统计
                    for layer in ["L1", "L2", "L3", "total"]:
                        if layer in timing:
                            layer_times[layer].append(timing[layer])
                
                # 显示进度
                if i % max(1, len(queries) // 10) == 0 or i == len(queries):
                    success_rate = (successful_queries / i * 100) if i > 0 else 0
                    print(f"Progress: {i}/{len(queries)} | Success: {successful_queries}/{i} ({success_rate:.1f}%)")
                    if empty_predictions > 0:
                        print(f"         ⚠️  空预测: {empty_predictions}/{successful_queries}")
                    
            except Exception as e:
                print(f"❌ Query {i} failed: {str(e)[:50]}")
        
        # 计算聚合指标
        metrics = self.calculate_metrics(results, ground_truth, config_name)
        
        # 添加性能指标
        metrics["performance"] = {
            "init_time": init_time,
            "avg_times": {
                layer: np.mean(times) if times else 0
                for layer, times in layer_times.items()
            },
            "total_queries": len(queries),
            "successful_queries": successful_queries,
            "success_rate": successful_queries / len(queries) if queries else 0,
        }
        
        metrics["config"] = config
        metrics["task_type"] = task_type
        metrics["dataset_size"] = dataset_size
        
        # 打印结果摘要
        self.print_results_summary(metrics, config_name)
        
        # 保存结果
        self.save_results(metrics, config_name, task_type, dataset_size)
        
        return metrics
    
    async def load_data(
        self,
        task_type: str,
        dataset_size: str
    ) -> Tuple[List, List, Dict]:
        """加载实验数据"""
        from src.core.models import TableInfo, ColumnInfo
        
        dataset_dir = Path(f"examples/separated_datasets/{task_type}_{dataset_size}")
        
        # 加载表数据
        with open(dataset_dir / "tables.json") as f:
            tables_data = json.load(f)
        
        # 优先使用过滤后的查询（如果存在）
        queries_filtered_path = dataset_dir / "queries_filtered.json"
        if queries_filtered_path.exists():
            print(f"📌 使用过滤后的查询: {queries_filtered_path}")
            with open(queries_filtered_path) as f:
                queries_data = json.load(f)
        else:
            with open(dataset_dir / "queries.json") as f:
                queries_data = json.load(f)
        
        # 优先使用转换后的ground truth（如果存在）
        gt_transformed_path = dataset_dir / "ground_truth_transformed.json"
        if gt_transformed_path.exists():
            print(f"📌 使用转换后的ground truth: {gt_transformed_path}")
            with open(gt_transformed_path) as f:
                ground_truth_data = json.load(f)
            # 转换后的格式已经是字典形式
            ground_truth = ground_truth_data
        else:
            with open(dataset_dir / "ground_truth.json") as f:
                ground_truth_data = json.load(f)
        
        # 转换为TableInfo对象
        tables = []
        for table_data in tables_data:
            table = TableInfo(
                table_name=table_data['table_name'],
                columns=[
                    ColumnInfo(
                        table_name=table_data['table_name'],
                        column_name=col.get('column_name', col.get('name', '')),
                        data_type=col.get('data_type', col.get('type', 'unknown')),
                        sample_values=col.get('sample_values', [])[:5]
                    )
                    for col in table_data.get('columns', [])[:20]
                ]
            )
            tables.append(table)
        
        # 如果ground truth还不是字典格式，进行转换
        if isinstance(ground_truth_data, list):
            ground_truth = {}
            for gt in ground_truth_data:
                if task_type == "join":
                    key = f"{gt['query_table']}:{gt.get('query_column', '')}"
                else:
                    key = gt['query_table']
                
                if key not in ground_truth:
                    ground_truth[key] = []
                ground_truth[key].append(gt['candidate_table'])
        
        return tables, queries_data, ground_truth
    
    async def create_ablation_workflow(self, config: Dict):
        """创建消融实验的workflow"""
        from src.core.ultra_optimized_workflow import UltraOptimizedWorkflow
        
        # 使用UltraOptimizedWorkflow，它有更好的层控制
        workflow = UltraOptimizedWorkflow()
        
        # 根据配置调整参数
        # L1: 元数据筛选
        if config.get("metadata_filter"):
            workflow.enable_metadata_filter = True
            workflow.max_metadata_candidates = config.get("max_metadata_candidates", 100)
        else:
            workflow.enable_metadata_filter = False
            workflow.max_metadata_candidates = 10000  # 不筛选，返回所有
        
        # L2: 向量搜索
        if config.get("vector_search"):
            workflow.enable_vector_search = True
            workflow.max_vector_candidates = config.get("max_vector_candidates", 50)
            workflow.vector_threshold = config.get("vector_threshold", 0.6)
        else:
            workflow.enable_vector_search = False
            workflow.max_vector_candidates = 0
        
        # L3: LLM匹配
        if config.get("llm_matching"):
            workflow.enable_llm_matching = True
            workflow.max_llm_candidates = config.get("max_llm_candidates", 3)
        else:
            workflow.enable_llm_matching = False
            workflow.max_llm_candidates = 0
        
        # 设置早停阈值
        workflow.early_stop_threshold = config.get("early_stop_threshold", 0.9)
        
        return workflow
    
    async def run_single_query(
        self,
        workflow,
        query: Dict,
        config: Dict,
        ground_truth: Dict
    ) -> Tuple[Dict, Dict]:
        """运行单个查询并收集指标"""
        from src.core.models import AgentState, TaskStrategy
        
        # 准备查询状态
        state = AgentState()
        query_table_name = query.get('query_table', '')
        query_column = query.get('query_column', '')
        
        # 调试输出
        if self.debug_mode:
            print(f"\n  📝 查询: {query_table_name}:{query_column if query_column else 'ALL'}")
        
        # 确保table_metadata_cache存在并正确
        if not hasattr(workflow, 'table_metadata_cache'):
            print(f"Warning: workflow没有table_metadata_cache属性")
            return None, {}
        
        if not isinstance(workflow.table_metadata_cache, dict):
            print(f"Warning: table_metadata_cache不是dict，是{type(workflow.table_metadata_cache)}")
            # 尝试从全局缓存获取
            if hasattr(workflow, '_global_table_cache'):
                workflow.table_metadata_cache = workflow._global_table_cache
            else:
                return None, {}
        
        # 设置查询表
        query_table = workflow.table_metadata_cache.get(query_table_name)
        
        if not query_table:
            if self.debug_mode:
                print(f"  ❌ 查询表 {query_table_name} 不在缓存中")
            return None, {}
        
        state.query_tables = [query_table]
        state.strategy = TaskStrategy.TOP_DOWN
        
        # 设置user_query（必需字段）
        if query_column:
            state.user_query = f"Find tables with joinable column {query_column} for {query_table_name}"
        else:
            state.user_query = f"Find tables similar to {query_table_name}"
        
        # 记录各层时间
        timing = {}
        total_start = time.time()
        
        # 执行workflow（带超时）
        try:
            # 运行优化的workflow，返回(state, metrics)
            result = await asyncio.wait_for(
                workflow.run_optimized(
                    state,
                    list(workflow.table_metadata_cache.keys()),
                    ground_truth  # 传入ground truth用于评估
                ),
                timeout=10.0
            )
            
            # 解包结果
            if isinstance(result, tuple) and len(result) == 2:
                result_state, eval_metrics = result
            else:
                result_state = result
                eval_metrics = None
            
            timing["total"] = time.time() - total_start
            
            # 提取各层时间（如果有）
            if hasattr(workflow, 'performance_stats'):
                timing["L1"] = workflow.performance_stats.get("metadata_filter_time", 0)
                timing["L2"] = workflow.performance_stats.get("vector_search_time", 0)
                timing["L3"] = workflow.performance_stats.get("llm_match_time", 0)
            
            # 收集预测结果
            predictions = []
            
            # 尝试多种方式获取预测结果
            if hasattr(result_state, 'table_matches') and result_state.table_matches:
                for match in result_state.table_matches[:20]:  # 增加到20个
                    if hasattr(match, 'target_table'):
                        predictions.append(match.target_table)
                    elif isinstance(match, dict):
                        # 尝试多个字段名
                        for field in ['target_table', 'table_name', 'name', 'table']:
                            if field in match:
                                predictions.append(match[field])
                                break
                    elif isinstance(match, str):
                        predictions.append(match)
                    elif isinstance(match, tuple) and len(match) >= 1:
                        predictions.append(match[0])  # 第一个元素通常是表名
            
            # 如果还是没有预测，尝试从final_results获取
            if not predictions and hasattr(result_state, 'final_results'):
                if isinstance(result_state.final_results, list):
                    for item in result_state.final_results[:20]:
                        if isinstance(item, str):
                            predictions.append(item)
                        elif isinstance(item, dict):
                            for field in ['target_table', 'table_name', 'name']:
                                if field in item:
                                    predictions.append(item[field])
                                    break
                        elif isinstance(item, tuple) and len(item) >= 1:
                            predictions.append(item[0])
            
            # 获取ground truth
            gt_key = f"{query_table_name}:{query_column}" if query_column else query_table_name
            true_matches = ground_truth.get(gt_key, [])
            
            # 调试输出
            if self.debug_mode:
                print(f"  🎯 Ground Truth: {true_matches[:5]}..." if len(true_matches) > 5 else f"  🎯 Ground Truth: {true_matches}")
                print(f"  🔮 预测数量: {len(predictions)}")
                if predictions:
                    print(f"  📊 前5个预测: {predictions[:5]}")
                    # 检查命中情况
                    hits = [p for p in predictions if p in true_matches]
                    if hits:
                        print(f"  ✅ 命中: {hits[:3]}")
                    else:
                        print(f"  ❌ 无命中")
                else:
                    print(f"  ⚠️  预测为空！检查workflow返回格式")
            
            return {
                "query": query_table_name,
                "predictions": predictions,
                "ground_truth": true_matches,
                "timing": timing
            }, timing
            
        except asyncio.TimeoutError:
            timing["total"] = 10.0
            return None, timing
        except Exception as e:
            print(f"Query error: {e}")
            return None, {}
    
    def calculate_metrics(
        self,
        results: List[Dict],
        ground_truth: Dict,
        config_name: str
    ) -> Dict:
        """计算评估指标 - 包含Precision、Recall、F1"""
        if not results:
            return {
                "accuracy": {},
                "precision": {},
                "recall": {},
                "f1": {},
                "config_name": config_name
            }
        
        # 计算Hit@K指标
        hit_at_k = {1: [], 3: [], 5: [], 10: []}
        precision_at_k = {1: [], 3: [], 5: [], 10: []}
        recall_at_k = {1: [], 3: [], 5: [], 10: []}
        f1_at_k = {1: [], 3: [], 5: [], 10: []}
        
        for result in results:
            predictions = result.get("predictions", [])
            true_matches = result.get("ground_truth", [])
            
            if not true_matches:
                continue
            
            for k in [1, 3, 5, 10]:
                pred_at_k = set(predictions[:k])
                true_set = set(true_matches)
                
                # Hit@K
                hit = 1.0 if len(pred_at_k & true_set) > 0 else 0.0
                hit_at_k[k].append(hit)
                
                # Precision@K - 在预测集合中有多少是正确的
                if pred_at_k:
                    precision = len(pred_at_k & true_set) / len(pred_at_k)
                else:
                    precision = 0.0
                precision_at_k[k].append(precision)
                
                # Recall@K - 真实匹配中有多少被找到
                recall = len(pred_at_k & true_set) / len(true_set) if true_set else 0.0
                recall_at_k[k].append(recall)
                
                # F1@K - Precision和Recall的调和平均
                if precision + recall > 0:
                    f1 = 2 * precision * recall / (precision + recall)
                else:
                    f1 = 0.0
                f1_at_k[k].append(f1)
        
        # 计算平均值
        metrics = {
            "accuracy": {
                f"hit@{k}": np.mean(hit_at_k[k]) * 100 if hit_at_k[k] else 0
                for k in [1, 3, 5, 10]
            },
            "precision": {
                f"p@{k}": np.mean(precision_at_k[k]) * 100 if precision_at_k[k] else 0
                for k in [1, 3, 5, 10]
            },
            "recall": {
                f"r@{k}": np.mean(recall_at_k[k]) * 100 if recall_at_k[k] else 0
                for k in [1, 3, 5, 10]
            },
            "f1": {
                f"f1@{k}": np.mean(f1_at_k[k]) * 100 if f1_at_k[k] else 0
                for k in [1, 3, 5, 10]
            },
            "evaluated_queries": len(results),
            "config_name": config_name
        }
        
        return metrics
    
    def print_results_summary(self, metrics: Dict, config_name: str):
        """打印结果摘要 - 包含完整评价指标"""
        print(f"\n{'='*80}")
        print(f"📊 Ablation Results: {config_name}")
        print(f"{'='*80}")
        
        # 主要指标表格 (K=5)
        main_data = [
            ["Metric", "@1", "@5", "@10"],
            ["Hit Rate", 
             f"{metrics['accuracy'].get('hit@1', 0):.1f}%",
             f"{metrics['accuracy'].get('hit@5', 0):.1f}%",
             f"{metrics['accuracy'].get('hit@10', 0):.1f}%"],
            ["Precision",
             f"{metrics['precision'].get('p@1', 0):.1f}%",
             f"{metrics['precision'].get('p@5', 0):.1f}%",
             f"{metrics['precision'].get('p@10', 0):.1f}%"],
            ["Recall",
             f"{metrics['recall'].get('r@1', 0):.1f}%",
             f"{metrics['recall'].get('r@5', 0):.1f}%",
             f"{metrics['recall'].get('r@10', 0):.1f}%"],
            ["F1-Score",
             f"{metrics['f1'].get('f1@1', 0):.1f}%",
             f"{metrics['f1'].get('f1@5', 0):.1f}%",
             f"{metrics['f1'].get('f1@10', 0):.1f}%"],
        ]
        print("\n🎯 Evaluation Metrics:")
        print(tabulate(main_data, headers="firstrow", tablefmt="grid"))
        
        # 性能表格
        perf = metrics.get("performance", {})
        avg_times = perf.get("avg_times", {})
        
        total_time = avg_times.get('total', 0.001) if avg_times.get('total', 0) > 0 else 0.001
        perf_data = [
            ["Layer", "Avg Time (s)", "Percentage"],
            ["L1 (Metadata)", f"{avg_times.get('L1', 0):.3f}", 
             f"{avg_times.get('L1', 0)/total_time*100:.1f}%"],
            ["L2 (Vector)", f"{avg_times.get('L2', 0):.3f}",
             f"{avg_times.get('L2', 0)/total_time*100:.1f}%"],
            ["L3 (LLM)", f"{avg_times.get('L3', 0):.3f}",
             f"{avg_times.get('L3', 0)/total_time*100:.1f}%"],
            ["Total", f"{avg_times.get('total', 0):.3f}", "100%"],
        ]
        print("\n⚡ Performance Breakdown:")
        print(tabulate(perf_data, headers="firstrow", tablefmt="grid"))
        
        # 成功率
        print(f"\n✅ Success Rate: {perf.get('success_rate', 0)*100:.1f}%")
        qps = 1.0/avg_times.get('total', 1) if avg_times.get('total', 0) > 0 else 0
        print(f"📈 QPS: {qps:.2f}" if qps > 0 else "📈 QPS: N/A")
    
    def save_results(
        self,
        metrics: Dict,
        config_name: str,
        task_type: str,
        dataset_size: str
    ):
        """保存实验结果"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"ablation_{config_name}_{task_type}_{dataset_size}_{timestamp}.json"
        filepath = self.results_dir / filename
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)
        
        print(f"\n💾 Results saved to: {filepath}")
    
    async def run_all_configurations(
        self,
        task_type: str,
        dataset_size: str,
        max_queries: int = 10
    ):
        """运行所有配置的实验"""
        all_results = {}
        
        for config_name in self.configs.keys():
            print(f"\n{'#'*80}")
            print(f"# Configuration {list(self.configs.keys()).index(config_name)+1}/{len(self.configs)}: {config_name}")
            print(f"{'#'*80}")
            
            try:
                result = await self.run_single_configuration(
                    config_name, task_type, dataset_size, max_queries
                )
                all_results[config_name] = result
            except Exception as e:
                print(f"❌ Failed to run {config_name}: {e}")
                all_results[config_name] = {"error": str(e)}
        
        # 生成比较报告
        self.generate_comparison_report(all_results, task_type, dataset_size)
        
        return all_results
    
    def generate_comparison_report(
        self,
        all_results: Dict,
        task_type: str,
        dataset_size: str
    ):
        """生成比较报告"""
        print(f"\n{'='*80}")
        print(f"📊 ABLATION STUDY COMPARISON REPORT")
        print(f"Task: {task_type.upper()} | Dataset: {dataset_size}")
        print(f"{'='*80}")
        
        # 准备比较数据
        comparison_data = []
        for config_name, result in all_results.items():
            if "error" in result:
                continue
            
            acc = result.get("accuracy", {})
            prec = result.get("precision", {})
            rec = result.get("recall", {})
            f1 = result.get("f1", {})
            perf = result.get("performance", {})
            avg_times = perf.get("avg_times", {})
            
            comparison_data.append([
                config_name,
                f"{acc.get('hit@1', 0):.1f}%",
                f"{acc.get('hit@5', 0):.1f}%",
                f"{prec.get('p@5', 0):.1f}%",
                f"{rec.get('r@5', 0):.1f}%",
                f"{f1.get('f1@5', 0):.1f}%",
                f"{avg_times.get('total', 0):.3f}s",
                f"{1.0/avg_times.get('total', 1):.1f}" if avg_times.get('total') else "N/A"
            ])
        
        # 打印比较表
        headers = ["Config", "Hit@1", "Hit@5", "Precision@5", "Recall@5", "F1@5", "Latency", "QPS"]
        print("\n📈 Performance Comparison:")
        print(tabulate(comparison_data, headers=headers, tablefmt="grid"))
        
        # 层贡献分析 - 显示逐层提升
        if "L1" in all_results and "L1+L2" in all_results and "L1+L2+L3" in all_results:
            # 获取各层指标
            l1_metrics = all_results["L1"]
            l12_metrics = all_results["L1+L2"]
            l123_metrics = all_results["L1+L2+L3"]
            
            print("\n🔬 Layer-by-Layer Contribution Analysis:")
            print("\n📊 Progressive Performance Improvement:")
            
            # 创建逐层提升表格
            layer_data = [
                ["Metric", "L1 Only", "L1+L2", "L1+L2+L3", "Total Gain"],
                ["Hit@5",
                 f"{l1_metrics.get('accuracy', {}).get('hit@5', 0):.1f}%",
                 f"{l12_metrics.get('accuracy', {}).get('hit@5', 0):.1f}%",
                 f"{l123_metrics.get('accuracy', {}).get('hit@5', 0):.1f}%",
                 f"+{l123_metrics.get('accuracy', {}).get('hit@5', 0) - l1_metrics.get('accuracy', {}).get('hit@5', 0):.1f}%"],
                ["Precision@5",
                 f"{l1_metrics.get('precision', {}).get('p@5', 0):.1f}%",
                 f"{l12_metrics.get('precision', {}).get('p@5', 0):.1f}%",
                 f"{l123_metrics.get('precision', {}).get('p@5', 0):.1f}%",
                 f"+{l123_metrics.get('precision', {}).get('p@5', 0) - l1_metrics.get('precision', {}).get('p@5', 0):.1f}%"],
                ["Recall@5",
                 f"{l1_metrics.get('recall', {}).get('r@5', 0):.1f}%",
                 f"{l12_metrics.get('recall', {}).get('r@5', 0):.1f}%",
                 f"{l123_metrics.get('recall', {}).get('r@5', 0):.1f}%",
                 f"+{l123_metrics.get('recall', {}).get('r@5', 0) - l1_metrics.get('recall', {}).get('r@5', 0):.1f}%"],
                ["F1@5",
                 f"{l1_metrics.get('f1', {}).get('f1@5', 0):.1f}%",
                 f"{l12_metrics.get('f1', {}).get('f1@5', 0):.1f}%",
                 f"{l123_metrics.get('f1', {}).get('f1@5', 0):.1f}%",
                 f"+{l123_metrics.get('f1', {}).get('f1@5', 0) - l1_metrics.get('f1', {}).get('f1@5', 0):.1f}%"]
            ]
            print(tabulate(layer_data, headers="firstrow", tablefmt="grid"))
            
            # 计算并显示增量贡献
            l1_f1 = l1_metrics.get('f1', {}).get('f1@5', 0)
            l12_f1 = l12_metrics.get('f1', {}).get('f1@5', 0)
            l123_f1 = l123_metrics.get('f1', {}).get('f1@5', 0)
            
            print("\n📈 Incremental Contribution (F1@5):")
            print(f"  L1 Baseline: {l1_f1:.1f}%")
            print(f"  +L2 Contribution: {l12_f1 - l1_f1:+.1f}% → Total: {l12_f1:.1f}%")
            print(f"  +L3 Contribution: {l123_f1 - l12_f1:+.1f}% → Total: {l123_f1:.1f}%")
        
        # 保存比较报告
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = self.results_dir / f"comparison_{task_type}_{dataset_size}_{timestamp}.json"
        
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)
        
        print(f"\n💾 Comparison report saved to: {report_file}")


async def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="Three-Layer Ablation Study")
    parser.add_argument("--config", type=str, default="progressive",
                        choices=["L1", "L2", "L3", "L1+L2", "L1+L3", "L2+L3", "L1+L2+L3", "all", "progressive"],
                        help="Configuration to test (progressive = L1, L1+L2, L1+L2+L3)")
    parser.add_argument("--task", type=str, default="join",
                        choices=["join", "union", "both"],
                        help="Task type")
    parser.add_argument("--dataset", type=str, default="subset",
                        choices=["subset", "complete"],
                        help="Dataset size")
    parser.add_argument("--max-queries", type=int, default=10,
                        help="Maximum queries to test")
    parser.add_argument("--debug", action="store_true",
                        help="Enable debug mode for detailed output")
    
    args = parser.parse_args()
    
    # 创建实验实例，传递数据集大小
    study = AblationStudy(debug_mode=args.debug, dataset_size=args.dataset)
    
    # 运行实验
    if args.config == "progressive":
        # 专注于递进实验：L1 → L1+L2 → L1+L2+L3
        print("\n" + "#"*80)
        print("# 🚀 Progressive Layer Ablation Study")
        print("# Testing: L1 → L1+L2 → L1+L2+L3")
        print("#"*80 + "\n")
        
        progressive_results = {}
        for config in ["L1", "L1+L2", "L1+L2+L3"]:
            print(f"\n{'='*80}")
            print(f"Testing Configuration: {config}")
            print(f"{'='*80}")
            
            if args.task == "both":
                for task in ["join", "union"]:
                    result = await study.run_single_configuration(
                        config, task, args.dataset, args.max_queries
                    )
                    progressive_results[f"{config}_{task}"] = result
            else:
                result = await study.run_single_configuration(
                    config, args.task, args.dataset, args.max_queries
                )
                progressive_results[config] = result
        
        # 生成递进报告
        study.generate_comparison_report(progressive_results, args.task, args.dataset)
        
    elif args.config == "all":
        # 运行所有配置
        if args.task == "both":
            # 运行两个任务
            for task in ["join", "union"]:
                print(f"\n{'#'*80}")
                print(f"# Task: {task.upper()}")
                print(f"{'#'*80}")
                await study.run_all_configurations(task, args.dataset, args.max_queries)
        else:
            await study.run_all_configurations(args.task, args.dataset, args.max_queries)
    else:
        # 运行单个配置
        if args.task == "both":
            for task in ["join", "union"]:
                await study.run_single_configuration(
                    args.config, task, args.dataset, args.max_queries
                )
        else:
            await study.run_single_configuration(
                args.config, args.task, args.dataset, args.max_queries
            )
    
    print("\n✅ Ablation study completed!")


if __name__ == "__main__":
    asyncio.run(main())