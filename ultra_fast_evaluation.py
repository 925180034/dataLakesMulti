#!/usr/bin/env python3
"""
Ultra Fast Evaluation - 专门用于JOIN/UNION分离测试
基于fixed版本，针对完整数据集优化
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


async def evaluate_task(
    task_type: str,
    dataset_size: str = "subset",
    max_queries: int = None,
    skip_llm: bool = False,
    verbose: bool = False
):
    """评估JOIN或UNION任务
    
    Args:
        task_type: "join" 或 "union"
        dataset_size: "subset" 或 "complete"
        max_queries: 最大查询数（None表示全部）
        skip_llm: 是否跳过LLM层
        verbose: 详细输出
    """
    
    print("="*80)
    print(f"🚀 Ultra Fast Evaluation - {task_type.upper()} Task")
    print(f"📊 Dataset: {dataset_size} | Skip LLM: {skip_llm}")
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
                    sample_values=col.get('sample_values', [])[:5]  # 限制样本数
                )
                for col in table_data.get('columns', [])[:10]  # 限制列数
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
    print("\n🔧 Initializing Ultra Optimized Workflow...")
    from src.core.ultra_optimized_workflow import UltraOptimizedWorkflow
    
    workflow = UltraOptimizedWorkflow()
    
    # 配置优化参数
    workflow.max_metadata_candidates = 30 if dataset_size == "complete" else 20
    workflow.max_vector_candidates = 10 if dataset_size == "complete" else 5
    workflow.max_llm_candidates = 3 if not skip_llm else 0
    workflow.early_stop_threshold = 0.75
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
    
    all_results = []
    all_metrics = []
    query_times = []
    successful_queries = 0
    queries_with_gt = 0
    failed_queries = []
    
    # 批量处理配置
    batch_size = 50
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
            current_gt = {gt_key: ground_truth_dict.get(gt_key, [])} if gt_key in ground_truth_dict else None
            if current_gt and current_gt[gt_key]:
                queries_with_gt += 1
            
            # 执行查询（带超时）
            start_time = time.time()
            try:
                result_state, metrics = await asyncio.wait_for(
                    workflow.run_optimized(
                        state,
                        [t.table_name for t in table_infos],
                        current_gt
                    ),
                    timeout=10.0
                )
                
                query_time = time.time() - start_time
                query_times.append(query_time)
                successful_queries += 1
                
                # 收集结果
                predictions = []
                if hasattr(result_state, 'table_matches') and result_state.table_matches:
                    for match in result_state.table_matches[:10]:
                        predictions.append(match.target_table)
                
                # 计算metrics
                if current_gt and current_gt[gt_key]:
                    pred_set = set(predictions[:10])
                    truth_set = set(current_gt[gt_key])
                    intersection = pred_set & truth_set
                    
                    tp = len(intersection)
                    fp = len(pred_set - truth_set)
                    fn = len(truth_set - pred_set)
                    
                    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
                    
                    all_metrics.append({
                        'precision': precision,
                        'recall': recall,
                        'f1': f1
                    })
                
                # 保存结果
                all_results.append({
                    'query_id': i,
                    'query_table': query_table_name,
                    'query_column': query_column if task_type == "join" else None,
                    'predictions': predictions[:10],
                    'ground_truth': current_gt[gt_key] if current_gt else [],
                    'query_time': query_time
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
                print(f"Progress: {i}/{len(queries_data)} | "
                      f"Avg: {avg_time:.3f}s | "
                      f"QPS: {qps:.1f} | "
                      f"Success: {success_rate:.1f}%")
    
    # 5. 计算最终结果
    print("\n" + "="*80)
    print(f"📊 {task_type.upper()} EVALUATION RESULTS")
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
    
    # 质量指标
    if all_metrics:
        avg_precision = np.mean([m['precision'] for m in all_metrics])
        avg_recall = np.mean([m['recall'] for m in all_metrics])
        avg_f1 = np.mean([m['f1'] for m in all_metrics])
        
        quality_data = [
            ["Precision@10", f"{avg_precision:.3f}"],
            ["Recall@10", f"{avg_recall:.3f}"],
            ["F1-Score", f"{avg_f1:.3f}"],
            ["Evaluated Queries", len(all_metrics)],
        ]
        
        print("\n🎯 Quality Metrics:")
        print(tabulate(quality_data, headers=["Metric", "Value"], tablefmt="grid"))
    else:
        avg_precision = avg_recall = avg_f1 = 0.0
        print("\n⚠️ No quality metrics available (no ground truth matches)")
    
    # 性能评估
    if valid_times:
        avg_time = np.mean(valid_times)
        print("\n🏆 Performance Assessment:")
        if avg_time <= 1:
            print(f"  ✅ EXCELLENT! Avg {avg_time:.3f}s ≤ 1s")
        elif avg_time <= 3:
            print(f"  ✅ GOOD! Avg {avg_time:.3f}s ≤ 3s")
        elif avg_time <= 8:
            print(f"  ⚠️ ACCEPTABLE. Avg {avg_time:.3f}s ≤ 8s")
        else:
            print(f"  ❌ NEEDS OPTIMIZATION. Avg {avg_time:.3f}s > 8s")
    
    # 6. 保存结果
    output_dir = Path("experiment_results")
    output_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"{task_type}_{dataset_size}_{timestamp}_ultra_fast.json"
    
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
            "precision": avg_precision,
            "recall": avg_recall,
            "f1": avg_f1,
            "evaluated_queries": len(all_metrics)
        },
        "detailed_results": all_results[:100] if verbose else []  # 保存前100个结果
    }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\n💾 Results saved to: {output_file}")
    print("="*80)
    
    return results


async def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Ultra Fast Evaluation for JOIN/UNION tasks")
    parser.add_argument("--task", choices=["join", "union", "both"], default="both",
                        help="Task type to evaluate")
    parser.add_argument("--dataset", choices=["subset", "complete"], default="complete",
                        help="Dataset size")
    parser.add_argument("--max-queries", type=int, default=None,
                        help="Maximum number of queries to evaluate")
    parser.add_argument("--skip-llm", action="store_true",
                        help="Skip LLM layer for faster evaluation")
    parser.add_argument("--verbose", action="store_true",
                        help="Verbose output")
    
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
        print(f"🚀 Starting {task.upper()} evaluation...")
        print(f"{'='*80}")
        
        try:
            result = await evaluate_task(
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
        print("📊 OVERALL SUMMARY")
        print("="*80)
        
        summary_data = []
        for task, result in all_results:
            m = result['metrics']
            p = result['performance']
            summary_data.append([
                task.upper(),
                f"{m['precision']:.3f}",
                f"{m['recall']:.3f}",
                f"{m['f1']:.3f}",
                f"{p['avg_time']:.3f}s",
                f"{p['qps']:.2f}",
                f"{p['success_rate']*100:.1f}%"
            ])
        
        print(tabulate(summary_data, 
                      headers=["Task", "Precision", "Recall", "F1", "Avg Time", "QPS", "Success"],
                      tablefmt="grid"))
    
    print("\n✅ Evaluation completed!")


if __name__ == "__main__":
    asyncio.run(main())