#!/usr/bin/env python3
"""
Parameter Optimization Experiment for Three-Layer Architecture
参数优化实验 - 为JOIN和UNION任务找到最佳三层参数配置
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
os.environ['SKIP_LLM'] = 'false'


class ParameterOptimizer:
    """三层架构参数优化器"""
    
    def __init__(self):
        """初始化参数优化器"""
        self.results_dir = Path("experiment_results/optimization")
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # 参数搜索空间 - 根据complete数据集特点调整
        # Complete数据集: 1534表, 平均2.1个正确答案
        self.parameter_grid = {
            "join": {
                "L1": {
                    # 元数据过滤候选数量 - 需要保留足够多的候选
                    "max_metadata_candidates": [200, 300, 500, 700, 1000],
                },
                "L1+L2": {
                    # 两层组合参数
                    "max_metadata_candidates": [300, 500, 700],
                    "max_vector_candidates": [50, 100, 150],
                    "vector_threshold": [0.4, 0.5, 0.6],
                },
                "L1+L2+L3": {
                    # 三层完整流水线
                    "max_metadata_candidates": [500, 700],
                    "max_vector_candidates": [100, 150],
                    "max_llm_candidates": [10, 15, 20],
                    "vector_threshold": [0.5],
                    "early_stop_threshold": [0.90, 0.95],
                }
            },
            "union": {
                "L1": {
                    # Union任务可能需要不同的参数
                    "max_metadata_candidates": [150, 250, 400, 600],
                },
                "L1+L2": {
                    "max_metadata_candidates": [250, 400, 600],
                    "max_vector_candidates": [40, 80, 120],
                    "vector_threshold": [0.45, 0.55, 0.65],
                },
                "L1+L2+L3": {
                    "max_metadata_candidates": [400, 600],
                    "max_vector_candidates": [80, 120],
                    "max_llm_candidates": [8, 12, 15],
                    "vector_threshold": [0.55],
                    "early_stop_threshold": [0.85, 0.90],
                }
            }
        }
        
        # 快速测试用的精简参数网格
        self.quick_parameter_grid = {
            "join": {
                "L1": {
                    "max_metadata_candidates": [300, 500, 700],
                },
                "L1+L2": {
                    "max_metadata_candidates": [500],
                    "max_vector_candidates": [100],
                    "vector_threshold": [0.5],
                },
                "L1+L2+L3": {
                    "max_metadata_candidates": [500],
                    "max_vector_candidates": [100],
                    "max_llm_candidates": [15],
                    "vector_threshold": [0.5],
                    "early_stop_threshold": [0.90],
                }
            },
            "union": {
                "L1": {
                    "max_metadata_candidates": [250, 400],
                },
                "L1+L2": {
                    "max_metadata_candidates": [400],
                    "max_vector_candidates": [80],
                    "vector_threshold": [0.55],
                },
                "L1+L2+L3": {
                    "max_metadata_candidates": [400],
                    "max_vector_candidates": [80],
                    "max_llm_candidates": [12],
                    "vector_threshold": [0.55],
                    "early_stop_threshold": [0.90],
                }
            }
        }
        
        # 性能统计
        self.best_params = {}
        
    def generate_parameter_combinations(self, params_dict: Dict) -> List[Dict]:
        """生成参数组合"""
        import itertools
        
        keys = list(params_dict.keys())
        values = list(params_dict.values())
        
        combinations = []
        for combo in itertools.product(*values):
            combinations.append(dict(zip(keys, combo)))
        
        return combinations
    
    async def evaluate_parameters(
        self,
        config_name: str,
        params: Dict,
        task_type: str,
        max_queries: int = 5
    ) -> Dict:
        """评估单组参数"""
        from src.core.ultra_optimized_workflow import UltraOptimizedWorkflow
        from src.core.models import AgentState, TaskStrategy, TableInfo, ColumnInfo
        
        print(f"\n  🔬 Testing params: {params}")
        
        # 加载数据
        dataset_dir = Path(f"examples/separated_datasets/{task_type}_complete")
        
        # 加载表数据
        with open(dataset_dir / "tables.json") as f:
            tables_data = json.load(f)
        
        # 加载查询（优先使用过滤后的）
        queries_path = dataset_dir / "queries_filtered.json"
        if not queries_path.exists():
            queries_path = dataset_dir / "queries.json"
        with open(queries_path) as f:
            queries_data = json.load(f)[:max_queries]
        
        # 加载ground truth（优先使用转换后的）
        gt_path = dataset_dir / "ground_truth_transformed.json"
        if not gt_path.exists():
            gt_path = dataset_dir / "ground_truth.json"
        with open(gt_path) as f:
            ground_truth = json.load(f)
        
        # 转换表数据
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
        
        # 创建并配置workflow
        workflow = UltraOptimizedWorkflow()
        
        # 根据配置启用/禁用层
        if config_name == "L1":
            workflow.enable_metadata_filter = True
            workflow.enable_vector_search = False
            workflow.enable_llm_matching = False
            workflow.max_metadata_candidates = params.get("max_metadata_candidates", 300)
            
        elif config_name == "L1+L2":
            workflow.enable_metadata_filter = True
            workflow.enable_vector_search = True
            workflow.enable_llm_matching = False
            workflow.max_metadata_candidates = params.get("max_metadata_candidates", 500)
            workflow.max_vector_candidates = params.get("max_vector_candidates", 100)
            workflow.vector_threshold = params.get("vector_threshold", 0.5)
            
        elif config_name == "L1+L2+L3":
            workflow.enable_metadata_filter = True
            workflow.enable_vector_search = True
            workflow.enable_llm_matching = True
            workflow.max_metadata_candidates = params.get("max_metadata_candidates", 500)
            workflow.max_vector_candidates = params.get("max_vector_candidates", 100)
            workflow.max_llm_candidates = params.get("max_llm_candidates", 15)
            workflow.vector_threshold = params.get("vector_threshold", 0.5)
            workflow.early_stop_threshold = params.get("early_stop_threshold", 0.90)
        
        # 初始化workflow
        await workflow.initialize(tables)
        
        # 运行查询并收集指标
        hit_at_k = {1: [], 3: [], 5: [], 10: []}
        precision_at_k = {5: []}
        recall_at_k = {5: []}
        latencies = []
        
        for query in queries_data:
            try:
                # 准备查询
                state = AgentState()
                query_table_name = query.get('query_table', '')
                query_column = query.get('query_column', '')
                
                query_table = workflow.table_metadata_cache.get(query_table_name)
                if not query_table:
                    continue
                
                state.query_tables = [query_table]
                state.strategy = TaskStrategy.TOP_DOWN
                
                if query_column:
                    state.user_query = f"Find tables with joinable column {query_column} for {query_table_name}"
                else:
                    state.user_query = f"Find tables similar to {query_table_name}"
                
                # 执行查询
                start_time = time.time()
                result = await asyncio.wait_for(
                    workflow.run_optimized(
                        state,
                        list(workflow.table_metadata_cache.keys()),
                        ground_truth
                    ),
                    timeout=10.0
                )
                latency = time.time() - start_time
                latencies.append(latency)
                
                # 解包结果
                if isinstance(result, tuple):
                    result_state, _ = result
                else:
                    result_state = result
                
                # 提取预测
                predictions = []
                if hasattr(result_state, 'table_matches') and result_state.table_matches:
                    for match in result_state.table_matches[:20]:
                        if hasattr(match, 'target_table'):
                            predictions.append(match.target_table)
                        elif isinstance(match, tuple):
                            predictions.append(match[0])
                        elif isinstance(match, str):
                            predictions.append(match)
                
                # 获取ground truth
                gt_key = f"{query_table_name}:{query_column}" if query_column else query_table_name
                true_matches = ground_truth.get(gt_key, [])
                
                if not true_matches:
                    continue
                
                # 计算指标
                for k in [1, 3, 5, 10]:
                    pred_at_k = set(predictions[:k])
                    true_set = set(true_matches)
                    hit = 1.0 if len(pred_at_k & true_set) > 0 else 0.0
                    hit_at_k[k].append(hit)
                
                # Precision@5 and Recall@5
                pred_at_5 = set(predictions[:5])
                if pred_at_5:
                    precision = len(pred_at_5 & true_set) / len(pred_at_5)
                    precision_at_k[5].append(precision)
                
                recall = len(pred_at_5 & true_set) / len(true_set) if true_set else 0.0
                recall_at_k[5].append(recall)
                
            except Exception as e:
                continue
        
        # 计算平均指标
        metrics = {
            "params": params,
            "hit@1": np.mean(hit_at_k[1]) * 100 if hit_at_k[1] else 0,
            "hit@5": np.mean(hit_at_k[5]) * 100 if hit_at_k[5] else 0,
            "hit@10": np.mean(hit_at_k[10]) * 100 if hit_at_k[10] else 0,
            "precision@5": np.mean(precision_at_k[5]) * 100 if precision_at_k[5] else 0,
            "recall@5": np.mean(recall_at_k[5]) * 100 if recall_at_k[5] else 0,
            "avg_latency": np.mean(latencies) if latencies else 0,
            "evaluated_queries": len(hit_at_k[5])
        }
        
        # 计算F1分数
        if metrics["precision@5"] + metrics["recall@5"] > 0:
            metrics["f1@5"] = 2 * metrics["precision@5"] * metrics["recall@5"] / (metrics["precision@5"] + metrics["recall@5"])
        else:
            metrics["f1@5"] = 0
        
        print(f"    ✅ F1@5: {metrics['f1@5']:.1f}% | Hit@5: {metrics['hit@5']:.1f}% | Latency: {metrics['avg_latency']:.2f}s")
        
        return metrics
    
    async def optimize_layer_parameters(
        self,
        config_name: str,
        task_type: str,
        max_queries: int = 5,
        use_quick_grid: bool = True
    ) -> Dict:
        """优化单层配置的参数"""
        print(f"\n{'='*80}")
        print(f"🔧 Optimizing {config_name} for {task_type.upper()}")
        print(f"{'='*80}")
        
        # 选择参数网格
        if use_quick_grid:
            param_grid = self.quick_parameter_grid[task_type][config_name]
        else:
            param_grid = self.parameter_grid[task_type][config_name]
        
        # 生成所有参数组合
        param_combinations = self.generate_parameter_combinations(param_grid)
        print(f"📊 Testing {len(param_combinations)} parameter combinations")
        
        # 评估每组参数
        results = []
        best_f1 = 0
        best_params = None
        
        for i, params in enumerate(param_combinations, 1):
            print(f"\n  [{i}/{len(param_combinations)}] Evaluating combination...")
            metrics = await self.evaluate_parameters(config_name, params, task_type, max_queries)
            results.append(metrics)
            
            # 跟踪最佳参数
            if metrics["f1@5"] > best_f1:
                best_f1 = metrics["f1@5"]
                best_params = params
                print(f"    🏆 New best F1@5: {best_f1:.1f}%")
        
        # 排序结果
        results.sort(key=lambda x: x["f1@5"], reverse=True)
        
        # 打印最佳结果
        print(f"\n🏆 Best parameters for {config_name}:")
        print(f"   Parameters: {best_params}")
        print(f"   F1@5: {best_f1:.1f}%")
        
        return {
            "config": config_name,
            "task": task_type,
            "best_params": best_params,
            "best_f1": best_f1,
            "all_results": results[:5]  # 保存前5个最佳结果
        }
    
    async def run_progressive_optimization(
        self,
        task_type: str,
        max_queries: int = 5,
        use_quick_grid: bool = True
    ) -> Dict:
        """运行渐进式参数优化"""
        print(f"\n{'#'*80}")
        print(f"# 🚀 Progressive Parameter Optimization for {task_type.upper()}")
        print(f"# Dataset: Complete (1534 tables)")
        print(f"# Queries: {max_queries} samples")
        print(f"{'#'*80}")
        
        optimization_results = {}
        
        # 优化每层
        for config in ["L1", "L1+L2", "L1+L2+L3"]:
            result = await self.optimize_layer_parameters(
                config, task_type, max_queries, use_quick_grid
            )
            optimization_results[config] = result
            
            # 保存中间结果
            self.save_optimization_results(optimization_results, task_type)
        
        # 生成优化报告
        self.generate_optimization_report(optimization_results, task_type)
        
        return optimization_results
    
    def save_optimization_results(self, results: Dict, task_type: str):
        """保存优化结果"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"optimization_{task_type}_{timestamp}.json"
        filepath = self.results_dir / filename
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print(f"\n💾 Results saved to: {filepath}")
    
    def generate_optimization_report(self, results: Dict, task_type: str):
        """生成优化报告"""
        print(f"\n{'='*80}")
        print(f"📊 PARAMETER OPTIMIZATION REPORT - {task_type.upper()}")
        print(f"{'='*80}")
        
        # 创建最佳参数表
        best_params_data = []
        for config_name, result in results.items():
            if result.get("best_params"):
                row = [config_name]
                # 添加参数
                for param, value in result["best_params"].items():
                    row.append(f"{param.split('_')[-1]}={value}")
                # 添加性能
                row.append(f"{result['best_f1']:.1f}%")
                best_params_data.append(row)
        
        print("\n🏆 Optimal Parameters per Layer:")
        headers = ["Config", "Param1", "Param2", "Param3", "F1@5"]
        print(tabulate(best_params_data, headers=headers, tablefmt="grid"))
        
        # 显示渐进提升
        if all(k in results for k in ["L1", "L1+L2", "L1+L2+L3"]):
            l1_f1 = results["L1"]["best_f1"]
            l12_f1 = results["L1+L2"]["best_f1"]
            l123_f1 = results["L1+L2+L3"]["best_f1"]
            
            print("\n📈 Progressive Performance Gain:")
            print(f"  L1 Baseline: {l1_f1:.1f}%")
            print(f"  L1+L2: {l12_f1:.1f}% (Δ = {l12_f1 - l1_f1:+.1f}%)")
            print(f"  L1+L2+L3: {l123_f1:.1f}% (Δ = {l123_f1 - l12_f1:+.1f}%)")
            print(f"  Total Improvement: {l123_f1 - l1_f1:+.1f}%")
        
        # 生成配置文件建议
        print("\n💡 Recommended Configuration:")
        print("```python")
        print("# Add to src/core/ultra_optimized_workflow.py")
        print(f"if task_type == '{task_type}':")
        
        if "L1+L2+L3" in results:
            params = results["L1+L2+L3"]["best_params"]
            print(f"    self.max_metadata_candidates = {params.get('max_metadata_candidates', 500)}")
            print(f"    self.max_vector_candidates = {params.get('max_vector_candidates', 100)}")
            print(f"    self.max_llm_candidates = {params.get('max_llm_candidates', 15)}")
            print(f"    self.vector_threshold = {params.get('vector_threshold', 0.5)}")
            print(f"    self.early_stop_threshold = {params.get('early_stop_threshold', 0.90)}")
        print("```")
    
    async def run_full_optimization(self, max_queries: int = 5):
        """运行完整优化流程"""
        print("\n" + "="*80)
        print("🎯 FULL PARAMETER OPTIMIZATION EXPERIMENT")
        print("="*80)
        
        all_results = {}
        
        # 优化JOIN任务
        print("\n📌 Task 1/2: JOIN")
        join_results = await self.run_progressive_optimization("join", max_queries, True)
        all_results["join"] = join_results
        
        # 优化UNION任务
        print("\n📌 Task 2/2: UNION")
        union_results = await self.run_progressive_optimization("union", max_queries, True)
        all_results["union"] = union_results
        
        # 生成综合报告
        self.generate_final_report(all_results)
        
        return all_results
    
    def generate_final_report(self, all_results: Dict):
        """生成最终综合报告"""
        print("\n" + "="*80)
        print("📊 FINAL OPTIMIZATION SUMMARY")
        print("="*80)
        
        # 创建对比表
        comparison_data = []
        for task in ["join", "union"]:
            if task in all_results:
                task_results = all_results[task]
                for config in ["L1", "L1+L2", "L1+L2+L3"]:
                    if config in task_results:
                        comparison_data.append([
                            task.upper(),
                            config,
                            f"{task_results[config]['best_f1']:.1f}%",
                            str(task_results[config]['best_params'])[:50] + "..."
                        ])
        
        print("\n🏆 Best Parameters Summary:")
        headers = ["Task", "Config", "F1@5", "Parameters"]
        print(tabulate(comparison_data, headers=headers, tablefmt="grid"))
        
        # 保存最终报告
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = self.results_dir / f"final_optimization_report_{timestamp}.json"
        
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)
        
        print(f"\n💾 Final report saved to: {report_file}")
        
        # 生成可执行的配置代码
        print("\n📝 Implementation Ready Configuration:")
        print("="*60)
        
        for task in ["join", "union"]:
            if task in all_results and "L1+L2+L3" in all_results[task]:
                params = all_results[task]["L1+L2+L3"]["best_params"]
                print(f"\n# {task.upper()} Task Optimal Parameters:")
                print(f"elif self.task_type == '{task}':")
                print(f"    self.max_metadata_candidates = {params.get('max_metadata_candidates', 500)}")
                print(f"    self.max_vector_candidates = {params.get('max_vector_candidates', 100)}")
                print(f"    self.max_llm_candidates = {params.get('max_llm_candidates', 15)}")
                print(f"    self.vector_threshold = {params.get('vector_threshold', 0.5)}")
                print(f"    self.early_stop_threshold = {params.get('early_stop_threshold', 0.90)}")


async def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="Parameter Optimization for Three-Layer Architecture")
    parser.add_argument("--task", type=str, default="both",
                        choices=["join", "union", "both"],
                        help="Task type to optimize")
    parser.add_argument("--max-queries", type=int, default=5,
                        help="Number of queries for evaluation (5 for quick test, 20 for thorough)")
    parser.add_argument("--quick", action="store_true",
                        help="Use quick parameter grid for faster testing")
    
    args = parser.parse_args()
    
    # 创建优化器
    optimizer = ParameterOptimizer()
    
    # 运行优化
    if args.task == "both":
        await optimizer.run_full_optimization(args.max_queries)
    else:
        await optimizer.run_progressive_optimization(
            args.task, 
            args.max_queries, 
            use_quick_grid=args.quick
        )
    
    print("\n✅ Parameter optimization completed!")


if __name__ == "__main__":
    asyncio.run(main())