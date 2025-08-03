#!/usr/bin/env python
"""
修复版：超快速数据湖评估系统
解决502查询的问题
"""

import json
import time
import asyncio
import numpy as np
from tabulate import tabulate
from typing import List, Dict, Any, Tuple
from collections import defaultdict
import os

# 强制设置短超时和少重试
os.environ['SKIP_LLM'] = 'false'
os.environ['LLM_TIMEOUT'] = '5'  # 5秒超时
os.environ['LLM_MAX_RETRIES'] = '1'  # 最多1次重试


async def evaluate_queries_fixed(
    num_queries: int,
    dataset_type: str = "subset"
):
    """修复的评估函数"""
    
    print("="*80)
    print(f"🚀 超快速数据湖评估系统（修复版）")
    print(f"📊 测试配置: {num_queries}个查询, {dataset_type}数据集")
    print("="*80)
    
    # 1. 加载数据
    print("\n📂 加载数据...")
    
    # 根据查询数量选择合适的数据集
    if dataset_type == "subset" and num_queries > 100:
        print(f"⚠️ Subset只有100个表，但请求{num_queries}个查询")
        print(f"🔄 自动切换到Complete数据集")
        dataset_type = "complete"
    
    # 加载对应数据集
    if dataset_type == "subset":
        tables_file = "examples/final_subset_tables.json"
        ground_truth_file = "examples/final_subset_ground_truth_auto.json"
    else:
        tables_file = "examples/final_complete_tables.json"
        ground_truth_file = "examples/final_complete_ground_truth_auto.json"
    
    with open(tables_file) as f:
        all_tables = json.load(f)
    
    with open(ground_truth_file) as f:
        ground_truth = json.load(f)
    
    print(f"✅ 已加载 {len(all_tables)} 个表")
    print(f"✅ 已加载真实标签 ({len(ground_truth)}条记录)")
    
    # 2. 初始化优化工作流（带超时控制）
    print("\n⚙️ 初始化超优化工作流...")
    from src.core.models import TableInfo, ColumnInfo, AgentState
    
    # 转换表数据
    table_infos = []
    for table_data in all_tables:
        table_info = TableInfo(
            table_name=table_data['table_name'],
            columns=[
                ColumnInfo(
                    table_name=table_data['table_name'],
                    column_name=col.get('column_name', col.get('name', '')),
                    data_type=col.get('data_type', col.get('type', 'unknown'))
                )
                for col in table_data.get('columns', [])[:10]  # 限制列数
            ]
        )
        table_infos.append(table_info)
    
    # 创建优化工作流（带快速失败配置）
    from src.core.ultra_optimized_workflow import UltraOptimizedWorkflow
    workflow = UltraOptimizedWorkflow()
    
    # 配置更激进的优化参数
    workflow.max_metadata_candidates = 20  # 进一步减少
    workflow.max_vector_candidates = 5     # 进一步减少
    workflow.max_llm_candidates = 2        # 只验证前2个
    workflow.early_stop_threshold = 0.75   # 更早停止
    
    init_start = time.time()
    await workflow.initialize(table_infos)
    init_time = time.time() - init_start
    print(f"✅ 索引初始化完成 (耗时: {init_time:.2f}秒)")
    
    # 3. 准备查询（正确处理数量）
    actual_queries = min(num_queries, len(all_tables))
    if num_queries > len(all_tables):
        print(f"\n⚠️ 请求{num_queries}个查询，但只有{len(all_tables)}个表")
        print(f"📊 将循环查询以达到{num_queries}次")
        
        # 循环使用表以达到请求的查询数
        query_tables = []
        for i in range(num_queries):
            query_tables.append(table_infos[i % len(table_infos)])
    else:
        query_tables = table_infos[:num_queries]
    
    print(f"\n🔍 开始评估 {len(query_tables)} 个查询...")
    print("-"*80)
    
    # 4. 执行批量查询（带超时保护）
    all_metrics = []
    query_times = []
    successful_queries = 0
    total_matches = 0
    failed_queries = []
    
    # 进度显示
    progress_interval = max(1, len(query_tables) // 10)
    
    for i, query_table in enumerate(query_tables, 1):
        # 准备查询状态
        state = AgentState()
        state.query_tables = [query_table]
        state.query_columns = query_table.columns
        
        # 获取该查询的真实标签
        query_gt = {query_table.table_name: ground_truth.get(query_table.table_name, [])}
        
        # 执行查询（带超时保护）
        start_time = time.time()
        try:
            # 设置单个查询的最大超时时间（10秒）
            result_state, metrics = await asyncio.wait_for(
                workflow.run_optimized(
                    state,
                    [t.table_name for t in table_infos],
                    query_gt if ground_truth else None
                ),
                timeout=10.0
            )
            
            query_time = time.time() - start_time
            query_times.append(query_time)
            
            # 统计结果
            if result_state:
                successful_queries += 1
                
                # 统计匹配数量
                if hasattr(result_state, 'final_results') and result_state.final_results:
                    total_matches += len(result_state.final_results)
                elif hasattr(result_state, 'matches') and result_state.matches:
                    total_matches += len(result_state.matches)
                elif hasattr(result_state, 'results') and result_state.results:
                    total_matches += len(result_state.results)
            
            # 收集评价指标
            if metrics:
                all_metrics.append(metrics)
                # 打印第一个查询的metrics结构，用于调试
                if i == 1:
                    print(f"  📊 Metrics示例: precision={getattr(metrics, 'precision', 'N/A')}, recall={getattr(metrics, 'recall', 'N/A')}")
                
        except asyncio.TimeoutError:
            query_time = 10.0  # 超时记为10秒
            query_times.append(query_time)
            failed_queries.append(i)
            print(f"  ⚠️ 查询 {i} 超时（10秒）")
            
        except Exception as e:
            query_time = 0.0
            query_times.append(query_time)
            failed_queries.append(i)
            print(f"  ❌ 查询 {i} 失败: {str(e)[:50]}")
        
        # 显示进度
        if i % progress_interval == 0 or i == len(query_tables):
            valid_times = [t for t in query_times if t > 0]
            if valid_times:
                avg_time = np.mean(valid_times)
                success_rate = successful_queries / i * 100
                print(f"进度: {i}/{len(query_tables)} | "
                      f"平均时间: {avg_time:.3f}s | "
                      f"成功率: {success_rate:.1f}%")
    
    # 5. 计算统计
    print("\n" + "="*80)
    print("📊 评估结果汇总")
    print("="*80)
    
    # 性能统计
    valid_times = [t for t in query_times if t > 0]
    if valid_times:
        performance_data = [
            ["查询总数", len(query_tables)],
            ["成功查询", successful_queries],
            ["失败查询", len(failed_queries)],
            ["成功率", f"{successful_queries/len(query_tables)*100:.1f}%"],
            ["总匹配数", total_matches],
            ["平均每查询匹配数", f"{total_matches/successful_queries:.1f}" if successful_queries > 0 else "0"],
            ["", ""],
            ["平均查询时间", f"{np.mean(valid_times):.3f}秒"],
            ["中位数时间", f"{np.median(valid_times):.3f}秒"],
            ["标准差", f"{np.std(valid_times):.3f}秒"],
            ["最小时间", f"{np.min(valid_times):.3f}秒"],
            ["最大时间", f"{np.max(valid_times):.3f}秒"],
        ]
        
        print("\n⚡ 性能指标:")
        print(tabulate(performance_data, headers=["指标", "值"], tablefmt="grid"))
    
    # 质量指标
    if all_metrics:
        # 处理不同类型的metrics对象
        precisions = []
        recalls = []
        f1_scores = []
        mrrs = []
        hit_rates = []
        
        for m in all_metrics:
            if hasattr(m, 'precision'):
                precisions.append(getattr(m, 'precision', 0))
            elif isinstance(m, dict) and 'precision' in m:
                precisions.append(m['precision'])
                
            if hasattr(m, 'recall'):
                recalls.append(getattr(m, 'recall', 0))
            elif isinstance(m, dict) and 'recall' in m:
                recalls.append(m['recall'])
                
            if hasattr(m, 'f1_score'):
                f1_scores.append(getattr(m, 'f1_score', 0))
            elif isinstance(m, dict) and 'f1_score' in m:
                f1_scores.append(m['f1_score'])
                
            if hasattr(m, 'mrr'):
                mrrs.append(getattr(m, 'mrr', 0))
            elif isinstance(m, dict) and 'mrr' in m:
                mrrs.append(m['mrr'])
                
            if hasattr(m, 'hit_rate'):
                hit_rates.append(getattr(m, 'hit_rate', 0))
            elif isinstance(m, dict) and 'hit_rate' in m:
                hit_rates.append(m['hit_rate'])
        
        # 计算平均值（只有有值时才计算）
        avg_precision = np.mean(precisions) if precisions else 0.0
        avg_recall = np.mean(recalls) if recalls else 0.0
        avg_f1 = np.mean(f1_scores) if f1_scores else 0.0
        avg_mrr = np.mean(mrrs) if mrrs else 0.0
        avg_hit_rate = np.mean(hit_rates) if hit_rates else 0.0
        
        quality_data = [
            ["Precision@10", f"{avg_precision:.3f}"],
            ["Recall@10", f"{avg_recall:.3f}"],
            ["F1-Score", f"{avg_f1:.3f}"],
            ["MRR", f"{avg_mrr:.3f}"],
            ["Hit Rate", f"{avg_hit_rate:.3f}"],
            ["有效评价数", f"{len(precisions)}/{len(all_metrics)}"],
        ]
        
        print("\n🎯 质量指标:")
        print(tabulate(quality_data, headers=["指标", "值"], tablefmt="grid"))
        
        # 如果没有有效的评价指标，说明原因
        if not precisions:
            print("\n⚠️ 注意：未能计算准确的质量指标")
            print("  可能原因：")
            print("  1. Ground truth数据不完整")
            print("  2. 查询结果为空")
            print("  3. 评价指标计算模块未正确返回数据")
    
    # 性能评估
    if valid_times:
        avg_time = np.mean(valid_times)
        print("\n🏆 性能评估:")
        if avg_time <= 1:
            print(f"  ✅ 卓越！平均 {avg_time:.3f}秒 ≤ 1秒 (毫秒级)")
        elif avg_time <= 3:
            print(f"  ✅ 优秀！平均 {avg_time:.3f}秒 ≤ 3秒")
        elif avg_time <= 8:
            print(f"  ⚠️ 达标。平均 {avg_time:.3f}秒 ≤ 8秒")
        else:
            print(f"  ❌ 需优化。平均 {avg_time:.3f}秒 > 8秒")
    
    # 保存结果
    results = {
        "config": {
            "num_queries": len(query_tables),
            "dataset_type": dataset_type,
            "init_time": init_time
        },
        "performance": {
            "avg_time": np.mean(valid_times) if valid_times else 0,
            "median_time": np.median(valid_times) if valid_times else 0,
            "std_time": np.std(valid_times) if valid_times else 0,
            "min_time": np.min(valid_times) if valid_times else 0,
            "max_time": np.max(valid_times) if valid_times else 0,
            "success_rate": successful_queries / len(query_tables) if query_tables else 0,
            "total_matches": total_matches,
            "failed_queries": failed_queries[:10]  # 只保存前10个失败的
        },
        "query_times": query_times[:500]  # 最多保存500个时间
    }
    
    output_file = f"ultra_evaluation_{dataset_type}_{len(query_tables)}_fixed.json"
    output_path = f"experiment_results/final/{output_file}"
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n💾 详细结果已保存到 {output_path}")
    print("="*80)


if __name__ == "__main__":
    import sys
    
    # 解析命令行参数
    num_queries = int(sys.argv[1]) if len(sys.argv) > 1 else 50
    dataset_type = sys.argv[2] if len(sys.argv) > 2 else "subset"
    
    # 运行评估
    asyncio.run(evaluate_queries_fixed(num_queries, dataset_type))