#!/usr/bin/env python3
"""
Ultra Fast Evaluation with Enhanced Metrics
包含 Hit@1, Hit@3, Hit@5, P@1, P@3, P@5, R@1, R@3, R@5 等详细指标
"""

import json
import time
import asyncio
import numpy as np
from tabulate import tabulate
from typing import List, Dict, Any, Tuple
from collections import defaultdict
import os
import sys
from pathlib import Path
from datetime import datetime

# 强制设置环境变量
os.environ['SKIP_LLM'] = 'false'
os.environ['LLM_TIMEOUT'] = '5'
os.environ['LLM_MAX_RETRIES'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['HF_DATASETS_OFFLINE'] = '1'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['HF_HUB_OFFLINE'] = '1'


def calculate_metrics_at_k(predictions: List[str], ground_truth: List[str], k: int) -> Dict[str, float]:
    """计算@K的各种指标
    
    Returns:
        hit: 前K个预测中是否有命中（0或1）
        precision: 前K个预测中正确的比例
        recall: 召回了多少ground truth
    """
    if not ground_truth:
        return {'hit': 0.0, 'precision': 0.0, 'recall': 0.0}
    
    pred_at_k = set(predictions[:k])
    truth_set = set(ground_truth)
    
    # Hit@K: 前K个中有任何命中就是1
    hit = 1.0 if len(pred_at_k & truth_set) > 0 else 0.0
    
    # Precision@K: 前K个中正确的比例
    correct = len(pred_at_k & truth_set)
    precision = correct / min(k, len(predictions)) if predictions else 0.0
    
    # Recall@K: 召回了多少ground truth
    recall = correct / len(truth_set) if truth_set else 0.0
    
    return {
        'hit': hit,
        'precision': precision,
        'recall': recall,
        'correct': correct
    }


async def evaluate_task_with_metrics(
    task_type: str,
    dataset_size: str = "subset",
    max_queries: int = None,
    skip_llm: bool = False,
    verbose: bool = False
):
    """评估JOIN或UNION任务，包含详细的@K指标"""
    
    print("="*80)
    print(f"🚀 Ultra Fast Evaluation with Enhanced Metrics")
    print(f"📊 Task: {task_type.upper()} | Dataset: {dataset_size}")
    print(f"🎯 Metrics: Hit@1/3/5, Precision@1/3/5, Recall@1/3/5")
    print(f"🕐 Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)
    
    # 1. 加载数据
    print("\n📂 Loading dataset...")
    
    dataset_dir = Path(f"examples/separated_datasets/{task_type}_{dataset_size}")
    if not dataset_dir.exists():
        print(f"❌ Error: Dataset not found: {dataset_dir}")
        return None
    
    with open(dataset_dir / "tables.json") as f:
        tables_data = json.load(f)
    
    with open(dataset_dir / "queries.json") as f:
        queries_data = json.load(f)
    
    with open(dataset_dir / "ground_truth.json") as f:
        ground_truth_data = json.load(f)
    
    print(f"✅ Loaded {len(tables_data)} tables")
    print(f"✅ Loaded {len(queries_data)} queries")
    print(f"✅ Loaded {len(ground_truth_data)} ground truth entries")
    
    # 限制查询数量
    if max_queries:
        queries_data = queries_data[:max_queries]
        print(f"🎯 Limited to {max_queries} queries")
    
    # 2. 准备数据
    print("\n⚙️ Preparing data...")
    from src.core.models import TableInfo, ColumnInfo, AgentState, TaskStrategy
    
    # 转换表数据
    table_infos = []
    table_dict = {}
    for table_data in tables_data:
        table_name = table_data['table_name']
        table_info = TableInfo(
            table_name=table_name,
            columns=[
                ColumnInfo(
                    table_name=table_name,
                    column_name=col.get('column_name', col.get('name', '')),
                    data_type=col.get('data_type', col.get('type', 'unknown')),
                    sample_values=col.get('sample_values', [])[:5]
                )
                for col in table_data.get('columns', [])[:10]
            ]
        )
        table_infos.append(table_info)
        table_dict[table_name] = table_info
    
    # 准备ground truth字典
    ground_truth_dict = {}
    for gt in ground_truth_data:
        query_table = gt.get('query_table', '')
        candidate_table = gt.get('candidate_table', '')
        
        if task_type == "join":
            query_column = gt.get('query_column', '')
            key = f"{query_table}:{query_column}"
        else:
            key = query_table
        
        if key not in ground_truth_dict:
            ground_truth_dict[key] = []
        if candidate_table not in ground_truth_dict[key]:
            ground_truth_dict[key].append(candidate_table)
    
    print(f"✅ Prepared {len(ground_truth_dict)} ground truth keys")
    
    # 3. 初始化工作流
    print(f"\n🔧 Initializing Optimized Workflow for {task_type.upper()} task...")
    from src.core.ultra_optimized_workflow import UltraOptimizedWorkflow
    
    workflow = UltraOptimizedWorkflow()
    
    # 根据任务类型调整参数
    if task_type == "join":
        # JOIN：保持精确但增加候选
        workflow.max_metadata_candidates = 70  # 从50增加到70
        workflow.max_vector_candidates = 15    # 从20减少到15（平衡速度）
        workflow.max_llm_candidates = 3 if not skip_llm else 0
        workflow.early_stop_threshold = 0.88   # 略微降低（从0.90）
        print("  Using JOIN-optimized parameters (balanced precision/recall)")
    else:
        # UNION：大幅增加候选提高召回率
        workflow.max_metadata_candidates = 100  # 大幅增加
        workflow.max_vector_candidates = 25     # 增加
        workflow.max_llm_candidates = 3 if not skip_llm else 0
        workflow.early_stop_threshold = 0.95   # 提高阈值减少早停
        print("  Using UNION-optimized parameters (high recall)")
    
    workflow.enable_llm_matching = not skip_llm
    
    print(f"  - Metadata candidates: {workflow.max_metadata_candidates}")
    print(f"  - Vector candidates: {workflow.max_vector_candidates}")
    print(f"  - LLM candidates: {workflow.max_llm_candidates}")
    print(f"  - LLM enabled: {workflow.enable_llm_matching}")
    
    init_start = time.time()
    await workflow.initialize(table_infos)
    init_time = time.time() - init_start
    print(f"✅ Workflow initialized in {init_time:.2f}s")
    
    # 4. 执行评估
    print(f"\n🔍 Evaluating {len(queries_data)} queries...")
    print("-"*80)
    
    # 收集所有指标
    all_results = []
    query_times = []
    successful_queries = 0
    queries_with_gt = 0
    failed_queries = []
    
    # @K指标收集器
    metrics_at_1 = []
    metrics_at_3 = []
    metrics_at_5 = []
    metrics_at_10 = []
    
    show_progress_every = max(1, len(queries_data) // 20)
    
    for i, query in enumerate(queries_data, 1):
        try:
            # 准备查询
            if task_type == "join":
                query_table_name = query.get('query_table', '')
                query_column = query.get('query_column', '')
                gt_key = f"{query_table_name}:{query_column}"
            else:
                query_table_name = query.get('query_table', '')
                gt_key = query_table_name
            
            # 获取查询表
            query_table = table_dict.get(query_table_name)
            if not query_table:
                failed_queries.append(i)
                continue
            
            # 准备状态
            state = AgentState()
            state.query_tables = [query_table]
            state.user_query = f"Find tables similar to {query_table_name}"
            state.strategy = TaskStrategy.BOTTOM_UP if task_type == "join" else TaskStrategy.TOP_DOWN
            
            if task_type == "join" and query_column:
                state.user_query = f"Find tables with joinable column {query_column} for {query_table_name}"
            
            # 准备ground truth
            current_gt = ground_truth_dict.get(gt_key, [])
            workflow_gt = {gt_key: current_gt} if current_gt else None
            
            if current_gt:
                queries_with_gt += 1
            
            # 执行查询（带超时）
            start_time = time.time()
            try:
                result_state, metrics = await asyncio.wait_for(
                    workflow.run_optimized(
                        state,
                        [t.table_name for t in table_infos],
                        workflow_gt
                    ),
                    timeout=10.0
                )
                
                query_time = time.time() - start_time
                query_times.append(query_time)
                successful_queries += 1
                
                # 收集预测结果
                predictions = []
                if hasattr(result_state, 'table_matches') and result_state.table_matches:
                    for match in result_state.table_matches[:10]:
                        predictions.append(match.target_table)
                
                # 计算@K指标
                if current_gt:
                    m1 = calculate_metrics_at_k(predictions, current_gt, 1)
                    m3 = calculate_metrics_at_k(predictions, current_gt, 3)
                    m5 = calculate_metrics_at_k(predictions, current_gt, 5)
                    m10 = calculate_metrics_at_k(predictions, current_gt, 10)
                    
                    metrics_at_1.append(m1)
                    metrics_at_3.append(m3)
                    metrics_at_5.append(m5)
                    metrics_at_10.append(m10)
                
                # 保存结果
                all_results.append({
                    'query_id': i,
                    'query_table': query_table_name,
                    'query_column': query_column if task_type == "join" else None,
                    'predictions': predictions[:10],
                    'ground_truth': current_gt,
                    'query_time': query_time,
                    'metrics': {
                        '@1': m1 if current_gt else None,
                        '@3': m3 if current_gt else None,
                        '@5': m5 if current_gt else None,
                        '@10': m10 if current_gt else None
                    }
                })
                
            except asyncio.TimeoutError:
                query_times.append(10.0)
                failed_queries.append(i)
                if verbose:
                    print(f"  ⚠️ Query {i} timeout (10s)")
            
        except Exception as e:
            failed_queries.append(i)
            if verbose:
                print(f"  ❌ Query {i} failed: {str(e)[:50]}")
        
        # 显示进度
        if i % show_progress_every == 0 or i == len(queries_data):
            valid_times = [t for t in query_times if t > 0]
            if valid_times:
                avg_time = np.mean(valid_times)
                success_rate = successful_queries / i * 100
                qps = 1.0 / avg_time if avg_time > 0 else 0
                
                # 实时计算Hit@K
                if metrics_at_1:
                    hit_1 = np.mean([m['hit'] for m in metrics_at_1]) * 100
                    hit_3 = np.mean([m['hit'] for m in metrics_at_3]) * 100
                    hit_5 = np.mean([m['hit'] for m in metrics_at_5]) * 100
                    
                    print(f"Progress: {i}/{len(queries_data)} | "
                          f"QPS: {qps:.1f} | "
                          f"Success: {success_rate:.1f}% | "
                          f"Hit@1/3/5: {hit_1:.1f}/{hit_3:.1f}/{hit_5:.1f}%")
                else:
                    print(f"Progress: {i}/{len(queries_data)} | "
                          f"QPS: {qps:.1f} | "
                          f"Success: {success_rate:.1f}%")
    
    # 5. 计算最终结果
    print("\n" + "="*80)
    print(f"📊 {task_type.upper()} EVALUATION RESULTS WITH ENHANCED METRICS")
    print("="*80)
    
    # 性能统计
    valid_times = [t for t in query_times if t > 0]
    total_time = sum(valid_times) if valid_times else 0
    
    if valid_times:
        performance_data = [
            ["Task Type", task_type.upper()],
            ["Dataset", f"{dataset_size} ({len(table_infos)} tables)"],
            ["Total Queries", len(queries_data)],
            ["Successful", successful_queries],
            ["Failed", len(failed_queries)],
            ["Success Rate", f"{successful_queries/len(queries_data)*100:.1f}%"],
            ["Queries with GT", queries_with_gt],
            ["", ""],
            ["Avg Query Time", f"{np.mean(valid_times):.3f}s"],
            ["Median Time", f"{np.median(valid_times):.3f}s"],
            ["Min Time", f"{np.min(valid_times):.3f}s"],
            ["Max Time", f"{np.max(valid_times):.3f}s"],
            ["Total Time", f"{total_time:.1f}s"],
            ["QPS", f"{len(valid_times)/total_time:.2f}" if total_time > 0 else "0"],
        ]
        
        print("\n⚡ Performance Metrics:")
        print(tabulate(performance_data, headers=["Metric", "Value"], tablefmt="grid"))
    
    # 详细@K指标
    if metrics_at_1:
        # 计算各个K值的平均指标
        hit_at_1 = np.mean([m['hit'] for m in metrics_at_1]) * 100
        hit_at_3 = np.mean([m['hit'] for m in metrics_at_3]) * 100
        hit_at_5 = np.mean([m['hit'] for m in metrics_at_5]) * 100
        hit_at_10 = np.mean([m['hit'] for m in metrics_at_10]) * 100
        
        precision_at_1 = np.mean([m['precision'] for m in metrics_at_1])
        precision_at_3 = np.mean([m['precision'] for m in metrics_at_3])
        precision_at_5 = np.mean([m['precision'] for m in metrics_at_5])
        precision_at_10 = np.mean([m['precision'] for m in metrics_at_10])
        
        recall_at_1 = np.mean([m['recall'] for m in metrics_at_1])
        recall_at_3 = np.mean([m['recall'] for m in metrics_at_3])
        recall_at_5 = np.mean([m['recall'] for m in metrics_at_5])
        recall_at_10 = np.mean([m['recall'] for m in metrics_at_10])
        
        # Hit Rate表格（最重要的指标）
        hit_data = [
            ["Hit@1", f"{hit_at_1:.2f}%", "第1个预测命中率"],
            ["Hit@3", f"{hit_at_3:.2f}%", "前3个有命中率"],
            ["Hit@5", f"{hit_at_5:.2f}%", "前5个有命中率"],
            ["Hit@10", f"{hit_at_10:.2f}%", "前10个有命中率"],
        ]
        
        print("\n🎯 Hit Rate (命中率) - 主要指标:")
        print(tabulate(hit_data, headers=["Metric", "Value", "Description"], tablefmt="grid"))
        
        # 详细指标表格
        detailed_data = [
            ["@K", "Precision", "Recall", "Hit Rate"],
            ["@1", f"{precision_at_1:.3f}", f"{recall_at_1:.3f}", f"{hit_at_1:.1f}%"],
            ["@3", f"{precision_at_3:.3f}", f"{recall_at_3:.3f}", f"{hit_at_3:.1f}%"],
            ["@5", f"{precision_at_5:.3f}", f"{recall_at_5:.3f}", f"{hit_at_5:.1f}%"],
            ["@10", f"{precision_at_10:.3f}", f"{recall_at_10:.3f}", f"{hit_at_10:.1f}%"],
        ]
        
        print("\n📈 Detailed Metrics:")
        print(tabulate(detailed_data, headers="firstrow", tablefmt="grid"))
        
        # 评估标准
        print("\n🏆 Quality Assessment:")
        if hit_at_1 >= 50:
            print(f"  ✅ EXCELLENT! Hit@1 = {hit_at_1:.1f}% ≥ 50%")
        elif hit_at_1 >= 30:
            print(f"  👍 GOOD! Hit@1 = {hit_at_1:.1f}% ≥ 30%")
        elif hit_at_1 >= 10:
            print(f"  ⚠️ MODERATE. Hit@1 = {hit_at_1:.1f}% ≥ 10%")
        else:
            print(f"  ❌ NEEDS IMPROVEMENT. Hit@1 = {hit_at_1:.1f}% < 10%")
        
        if hit_at_5 >= 70:
            print(f"  ✅ EXCELLENT! Hit@5 = {hit_at_5:.1f}% ≥ 70%")
        elif hit_at_5 >= 50:
            print(f"  👍 GOOD! Hit@5 = {hit_at_5:.1f}% ≥ 50%")
        else:
            print(f"  ⚠️ NEEDS IMPROVEMENT. Hit@5 = {hit_at_5:.1f}% < 50%")
    else:
        print("\n⚠️ No queries with ground truth - cannot calculate @K metrics")
        hit_at_1 = hit_at_3 = hit_at_5 = hit_at_10 = 0
        precision_at_1 = precision_at_3 = precision_at_5 = precision_at_10 = 0
        recall_at_1 = recall_at_3 = recall_at_5 = recall_at_10 = 0
    
    # 6. 保存结果
    output_dir = Path("experiment_results")
    output_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"{task_type}_{dataset_size}_{timestamp}_with_metrics.json"
    
    results = {
        "task_type": task_type.upper(),
        "dataset_size": dataset_size,
        "timestamp": datetime.now().isoformat(),
        "config": {
            "total_tables": len(table_infos),
            "total_queries": len(queries_data),
            "max_queries": max_queries,
            "skip_llm": skip_llm,
            "successful_queries": successful_queries,
            "queries_with_gt": queries_with_gt,
            "init_time": init_time
        },
        "performance": {
            "avg_time": np.mean(valid_times) if valid_times else 0,
            "median_time": np.median(valid_times) if valid_times else 0,
            "min_time": np.min(valid_times) if valid_times else 0,
            "max_time": np.max(valid_times) if valid_times else 0,
            "total_time": total_time,
            "qps": len(valid_times)/total_time if total_time > 0 else 0,
            "success_rate": successful_queries / len(queries_data) if queries_data else 0
        },
        "metrics": {
            "hit_at_1": hit_at_1,
            "hit_at_3": hit_at_3,
            "hit_at_5": hit_at_5,
            "hit_at_10": hit_at_10,
            "precision_at_1": precision_at_1,
            "precision_at_3": precision_at_3,
            "precision_at_5": precision_at_5,
            "precision_at_10": precision_at_10,
            "recall_at_1": recall_at_1,
            "recall_at_3": recall_at_3,
            "recall_at_5": recall_at_5,
            "recall_at_10": recall_at_10,
            "evaluated_queries": len(metrics_at_1)
        },
        "detailed_results": all_results[:100] if verbose else []
    }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\n💾 Results saved to: {output_file}")
    print("="*80)
    
    return results


async def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Ultra Fast Evaluation with Enhanced Metrics (@1/3/5)")
    parser.add_argument("--task", choices=["join", "union", "both"], default="both",
                        help="Task type to evaluate")
    parser.add_argument("--dataset", choices=["subset", "complete"], default="complete",
                        help="Dataset size")
    parser.add_argument("--max-queries", type=int, default=None,
                        help="Maximum number of queries to evaluate")
    parser.add_argument("--skip-llm", action="store_true",
                        help="Skip LLM layer for faster evaluation")
    parser.add_argument("--verbose", action="store_true",
                        help="Verbose output with detailed results")
    
    args = parser.parse_args()
    
    # 确定要运行的任务
    if args.task == "both":
        tasks = ["join", "union"]
    else:
        tasks = [args.task]
    
    # 运行评估
    all_results = []
    for task in tasks:
        print(f"\n{'='*80}")
        print(f"🚀 Starting {task.upper()} evaluation with enhanced metrics...")
        print(f"{'='*80}")
        
        try:
            result = await evaluate_task_with_metrics(
                task_type=task,
                dataset_size=args.dataset,
                max_queries=args.max_queries,
                skip_llm=args.skip_llm,
                verbose=args.verbose
            )
            
            if result:
                all_results.append((task, result))
                
        except Exception as e:
            print(f"❌ Error in {task} evaluation: {e}")
            import traceback
            traceback.print_exc()
    
    # 打印总结
    if len(all_results) > 1:
        print("\n" + "="*80)
        print("📊 OVERALL SUMMARY - All Metrics @1/3/5")
        print("="*80)
        
        summary_data = []
        for task, result in all_results:
            m = result['metrics']
            p = result['performance']
            summary_data.append([
                task.upper(),
                f"{m['hit_at_1']:.1f}%",
                f"{m['hit_at_3']:.1f}%",
                f"{m['hit_at_5']:.1f}%",
                f"{m['precision_at_5']:.3f}",
                f"{m['recall_at_5']:.3f}",
                f"{p['qps']:.1f}",
                f"{p['success_rate']*100:.1f}%"
            ])
        
        print(tabulate(summary_data, 
                      headers=["Task", "Hit@1", "Hit@3", "Hit@5", "P@5", "R@5", "QPS", "Success"],
                      tablefmt="grid"))
        
        # 最终评价
        print("\n🎯 Final Assessment:")
        for task, result in all_results:
            m = result['metrics']
            print(f"  {task.upper()}: ", end="")
            if m['hit_at_1'] >= 30 and m['hit_at_5'] >= 50:
                print("✅ Good quality")
            elif m['hit_at_1'] >= 10 and m['hit_at_5'] >= 30:
                print("⚠️ Moderate quality")
            else:
                print("❌ Needs improvement")
    
    print("\n✅ Evaluation completed with enhanced metrics!")


if __name__ == "__main__":
    asyncio.run(main())