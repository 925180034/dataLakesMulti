#!/usr/bin/env python
"""
完整实验运行脚本（完全修复版）
- 使用正确的queries文件
- 确保LLM被正确调用
- 修复所有已知bug
- 生成准确的评价指标
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
import argparse

# 设置环境变量确保LLM工作
os.environ['SKIP_LLM'] = 'false'  # 确保不跳过LLM
os.environ['LLM_TIMEOUT'] = '30'  # 30秒超时给LLM更多时间
os.environ['LLM_MAX_RETRIES'] = '3'  # 3次重试


def calculate_metrics(predictions: List[str], ground_truth: List[str]) -> Dict[str, float]:
    """计算精确率、召回率和F1分数"""
    if not ground_truth:  # 如果没有ground truth，无法评估
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    
    if not predictions:  # 如果没有预测，全部为0
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    
    pred_set = set(predictions)
    truth_set = set(ground_truth)
    
    # 计算交集
    intersection = pred_set & truth_set
    
    # 计算指标
    precision = len(intersection) / len(pred_set) if pred_set else 0.0
    recall = len(intersection) / len(truth_set) if truth_set else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1
    }


async def run_full_experiment(dataset_type: str = "subset", max_queries: int = None, enable_llm: bool = True):
    """运行完整实验 - 完全修复版"""
    
    print("="*80)
    print(f"🚀 数据湖完整实验系统（完全修复版）")
    print(f"📊 数据集类型: {dataset_type}")
    print(f"📊 最大查询数: {max_queries if max_queries else '全部'}")
    print(f"🤖 LLM状态: {'启用' if enable_llm else '禁用'}")
    print("="*80)
    
    # 1. 加载数据
    print("\n📂 加载数据...")
    
    # 选择数据集
    if dataset_type == "subset":
        tables_file = "examples/final_subset_tables.json"
        queries_file = "examples/final_subset_queries.json"
        ground_truth_file = "examples/final_subset_ground_truth_auto.json"
    else:
        tables_file = "examples/final_complete_tables.json"
        queries_file = "examples/final_complete_queries.json"
        ground_truth_file = "examples/final_complete_ground_truth_auto.json"
    
    # 加载表数据
    with open(tables_file) as f:
        all_tables = json.load(f)
    
    # 加载查询
    with open(queries_file) as f:
        all_queries = json.load(f)
    
    # 加载真实标签
    with open(ground_truth_file) as f:
        ground_truth_data = json.load(f)
    
    print(f"✅ 已加载 {len(all_tables)} 个表")
    print(f"✅ 已加载 {len(all_queries)} 个查询")
    print(f"✅ 已加载真实标签 ({len(ground_truth_data)}条记录)")
    
    # 限制查询数量
    if max_queries:
        all_queries = all_queries[:max_queries]
        print(f"📊 限制查询数量为: {len(all_queries)}")
    
    # 2. 初始化工作流
    print("\n⚙️ 初始化超优化工作流...")
    from src.core.models import TableInfo, ColumnInfo, AgentState
    
    # 转换表数据
    table_infos = []
    table_name_to_info = {}
    for table_data in all_tables:
        table_info = TableInfo(
            table_name=table_data['table_name'],
            columns=[
                ColumnInfo(
                    table_name=table_data['table_name'],
                    column_name=col.get('column_name', col.get('name', '')),
                    data_type=col.get('data_type', col.get('type', 'unknown'))
                )
                for col in table_data.get('columns', [])[:20]
            ]
        )
        table_infos.append(table_info)
        table_name_to_info[table_data['table_name']] = table_info
    
    # 选择工作流
    if enable_llm:
        print("  📍 使用优化工作流（带LLM验证）")
        from src.core.optimized_workflow import OptimizedDataLakesWorkflow
        workflow = OptimizedDataLakesWorkflow()
        
        # 尝试启用LLM匹配（如果存在该属性）
        if hasattr(workflow, 'enable_llm_matching'):
            workflow.enable_llm_matching = True
            print("  ✅ LLM匹配已启用")
        
        # 配置参数
        if hasattr(workflow, 'max_metadata_candidates'):
            workflow.max_metadata_candidates = 100
        if hasattr(workflow, 'max_vector_candidates'):
            workflow.max_vector_candidates = 50
        if hasattr(workflow, 'max_llm_candidates'):
            workflow.max_llm_candidates = 20
    else:
        print("  📍 使用超快速工作流（无LLM）")
        from src.core.ultra_optimized_workflow import UltraOptimizedWorkflow
        workflow = UltraOptimizedWorkflow()
        workflow.enable_llm_matching = False
        workflow.max_metadata_candidates = 20
        workflow.max_vector_candidates = 5
        workflow.max_llm_candidates = 0
        workflow.early_stop_threshold = 0.75
    
    # 初始化索引
    init_start = time.time()
    await workflow.initialize(table_infos)
    init_time = time.time() - init_start
    print(f"✅ 索引初始化完成 (耗时: {init_time:.2f}秒)")
    
    # 3. 执行查询评估
    print(f"\n🔍 开始评估 {len(all_queries)} 个查询...")
    print("-"*80)
    
    all_results = []
    all_metrics = []
    query_times = []
    successful_queries = 0
    failed_queries = []
    
    # 批处理设置
    batch_size = 5 if enable_llm else 10
    total_batches = (len(all_queries) + batch_size - 1) // batch_size
    
    for batch_idx in range(total_batches):
        batch_start = batch_idx * batch_size
        batch_end = min(batch_start + batch_size, len(all_queries))
        batch_queries = all_queries[batch_start:batch_end]
        
        print(f"\n📦 处理批次 {batch_idx+1}/{total_batches} (查询 {batch_start+1}-{batch_end})...")
        
        for query_idx, query_info in enumerate(batch_queries):
            actual_idx = batch_start + query_idx
            
            # 解析查询信息
            query_table_name = query_info.get('query_table', '')
            query_column = query_info.get('query_column', '')
            query_type = query_info.get('query_type', 'join')
            
            # 获取查询表的信息
            query_table_info = table_name_to_info.get(query_table_name)
            if not query_table_info:
                print(f"  ⚠️ 查询 {actual_idx+1}: 未找到表 {query_table_name}")
                continue
            
            # 获取真实标签
            expected = ground_truth_data.get(query_table_name, [])
            
            # 准备查询状态
            state = AgentState()
            state.user_query = f"Find tables that can {query_type} with {query_table_name}"
            if query_column:
                state.user_query += f" on column {query_column}"
            state.query_tables = [query_table_info]
            if query_column:
                for col in query_table_info.columns:
                    if col.column_name == query_column:
                        state.query_columns = [col]
                        break
            
            # 执行查询
            start_time = time.time()
            try:
                # 设置超时时间
                timeout_seconds = 60.0 if enable_llm else 10.0
                
                # 使用不同的方法调用工作流
                if hasattr(workflow, 'run_optimized'):
                    # 检查是哪种工作流
                    if 'UltraOptimized' in workflow.__class__.__name__:
                        # UltraOptimizedWorkflow 接受3个参数
                        result_state, metrics = await asyncio.wait_for(
                            workflow.run_optimized(
                                state,
                                [t.table_name for t in table_infos],
                                {query_table_name: expected}
                            ),
                            timeout=timeout_seconds
                        )
                    else:
                        # OptimizedDataLakesWorkflow 只接受2个参数
                        result_state = await asyncio.wait_for(
                            workflow.run_optimized(
                                state,
                                [t.table_name for t in table_infos]
                            ),
                            timeout=timeout_seconds
                        )
                        metrics = None
                else:
                    # 使用 run 方法
                    result_state = await asyncio.wait_for(
                        workflow.run(state),
                        timeout=timeout_seconds
                    )
                    metrics = None
                
                query_time = time.time() - start_time
                query_times.append(query_time)
                
                # 提取预测结果
                predictions = []
                if result_state:
                    successful_queries += 1
                    if hasattr(result_state, 'final_results') and result_state.final_results:
                        predictions = [r.target_table for r in result_state.final_results[:10]]
                    elif hasattr(result_state, 'table_matches') and result_state.table_matches:
                        predictions = [m.target_table for m in result_state.table_matches[:10]]
                
                # 计算指标
                calc_metrics = calculate_metrics(predictions, expected)
                all_metrics.append(calc_metrics)
                
                # 保存结果
                result_info = {
                    "query_idx": actual_idx + 1,
                    "query_table": query_table_name,
                    "query_column": query_column,
                    "query_type": query_type,
                    "predictions": predictions,
                    "expected": expected,
                    "metrics": calc_metrics,
                    "query_time": query_time
                }
                all_results.append(result_info)
                
            except asyncio.TimeoutError:
                query_time = timeout_seconds
                query_times.append(query_time)
                failed_queries.append(actual_idx + 1)
                print(f"  ⚠️ 查询 {actual_idx+1} 超时（{timeout_seconds}秒）")
                all_metrics.append({"precision": 0.0, "recall": 0.0, "f1": 0.0})
                
            except Exception as e:
                failed_queries.append(actual_idx + 1)
                print(f"  ❌ 查询 {actual_idx+1} 失败: {str(e)[:100]}")
                all_metrics.append({"precision": 0.0, "recall": 0.0, "f1": 0.0})
        
        # 显示进度
        if all_metrics:
            avg_precision = np.mean([m["precision"] for m in all_metrics])
            avg_recall = np.mean([m["recall"] for m in all_metrics])
            avg_f1 = np.mean([m["f1"] for m in all_metrics])
            avg_time = np.mean(query_times) if query_times else 0
            
            print(f"  当前平均: P={avg_precision:.3f} R={avg_recall:.3f} F1={avg_f1:.3f} Time={avg_time:.2f}s")
    
    # 4. 计算总体指标
    print("\n" + "="*80)
    print("📊 实验结果汇总")
    print("="*80)
    
    if query_times and all_metrics:
        avg_precision = np.mean([m["precision"] for m in all_metrics])
        avg_recall = np.mean([m["recall"] for m in all_metrics])
        avg_f1 = np.mean([m["f1"] for m in all_metrics])
        avg_query_time = np.mean(query_times)
        total_time = sum(query_times)
        
        # 创建结果表格
        results_table = [
            ["指标", "数值"],
            ["查询总数", len(all_queries)],
            ["成功查询", successful_queries],
            ["失败查询", len(failed_queries)],
            ["成功率", f"{successful_queries/len(all_queries)*100:.1f}%"],
            ["", ""],
            ["平均精确率", f"{avg_precision:.4f}"],
            ["平均召回率", f"{avg_recall:.4f}"],
            ["平均F1分数", f"{avg_f1:.4f}"],
            ["", ""],
            ["平均查询时间", f"{avg_query_time:.3f}秒"],
            ["总执行时间", f"{total_time:.2f}秒"],
            ["吞吐量", f"{len(query_times)/total_time if total_time > 0 else 0:.2f}查询/秒"],
            ["", ""],
            ["LLM状态", "启用" if enable_llm else "禁用"],
            ["索引初始化时间", f"{init_time:.2f}秒"]
        ]
        
        print(tabulate(results_table, headers="firstrow", tablefmt="grid"))
        
        # 性能评估
        print("\n🏆 性能评估:")
        if avg_query_time <= 3:
            print(f"  ✅ 优秀！平均 {avg_query_time:.3f}秒 ≤ 3秒")
        elif avg_query_time <= 8:
            print(f"  ⚠️ 达标。平均 {avg_query_time:.3f}秒 ≤ 8秒")
        else:
            print(f"  ❌ 需优化。平均 {avg_query_time:.3f}秒 > 8秒")
        
        # 质量评估
        print("\n🎯 质量评估:")
        if avg_precision >= 0.9:
            print(f"  ✅ 精确率优秀: {avg_precision:.3f} ≥ 0.9")
        elif avg_precision >= 0.7:
            print(f"  ⚠️ 精确率良好: {avg_precision:.3f}")
        else:
            print(f"  ❌ 精确率需改进: {avg_precision:.3f}")
        
        if avg_recall >= 0.9:
            print(f"  ✅ 召回率优秀: {avg_recall:.3f} ≥ 0.9")
        elif avg_recall >= 0.7:
            print(f"  ⚠️ 召回率良好: {avg_recall:.3f}")
        else:
            print(f"  ❌ 召回率需改进: {avg_recall:.3f}")
    else:
        print("❌ 没有成功的查询，无法计算指标")
    
    # 5. 保存结果
    output_file = f"experiment_results/full_experiment_fixed_{dataset_type}_{len(all_queries)}queries_{'with_llm' if enable_llm else 'no_llm'}_{time.strftime('%Y%m%d_%H%M%S')}.json"
    os.makedirs("experiment_results", exist_ok=True)
    
    output_data = {
        "experiment_info": {
            "dataset_type": dataset_type,
            "total_tables": len(all_tables),
            "total_queries": len(all_queries),
            "llm_enabled": enable_llm,
            "init_time": init_time,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        },
        "overall_metrics": {
            "avg_precision": avg_precision if query_times else 0,
            "avg_recall": avg_recall if query_times else 0,
            "avg_f1": avg_f1 if query_times else 0,
            "avg_query_time": avg_query_time if query_times else 0,
            "total_time": total_time if query_times else 0,
            "throughput": len(query_times)/total_time if query_times and total_time > 0 else 0,
            "success_rate": successful_queries/len(all_queries) if all_queries else 0
        },
        "detailed_results": all_results
    }
    
    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"\n✅ 结果已保存到: {output_file}")
    
    return output_data


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='运行完整的数据湖实验（完全修复版）')
    parser.add_argument(
        '--dataset',
        type=str,
        default='subset',
        choices=['subset', 'complete'],
        help='数据集类型 (默认: subset)'
    )
    parser.add_argument(
        '--max-queries',
        type=int,
        default=None,
        help='最大查询数量 (默认: 全部)'
    )
    parser.add_argument(
        '--no-llm',
        action='store_true',
        help='禁用LLM验证（仅用于测试）'
    )
    
    args = parser.parse_args()
    
    # 运行实验
    asyncio.run(run_full_experiment(
        args.dataset, 
        args.max_queries,
        enable_llm=not args.no_llm
    ))


if __name__ == "__main__":
    main()