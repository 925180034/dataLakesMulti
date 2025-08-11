#!/usr/bin/env python3
"""
集成测试：多智能体系统 + 三层加速架构
验证真正的多Agent协同工作流
"""

import asyncio
import json
import time
from pathlib import Path
from typing import List, Dict, Any

# 多智能体系统
from src.core.enhanced_multi_agent_system import EnhancedMultiAgentOrchestrator
from src.core.models import TableInfo, ColumnInfo

# 评价指标
from src.core.ultra_optimized_workflow import EvaluationMetrics
import numpy as np


async def test_multi_agent_with_acceleration():
    """测试多Agent系统与三层加速的集成"""
    
    print("="*80)
    print("🚀 多智能体系统 + 三层加速架构 集成测试")
    print("="*80)
    
    # 1. 加载数据
    dataset_path = Path("examples/separated_datasets/join_subset")
    
    # 加载表
    with open(dataset_path / "tables.json") as f:
        tables_data = json.load(f)[:50]  # 使用50个表测试
    
    # 加载查询
    queries_path = dataset_path / "queries_filtered.json"
    if not queries_path.exists():
        queries_path = dataset_path / "queries.json"
    with open(queries_path) as f:
        queries_data = json.load(f)[:5]  # 测试5个查询
    
    # 加载ground truth
    gt_path = dataset_path / "ground_truth_transformed.json"
    if not gt_path.exists():
        gt_path = dataset_path / "ground_truth.json"
    with open(gt_path) as f:
        ground_truth = json.load(f)
    
    print(f"\n📊 数据集信息:")
    print(f"  - 表数量: {len(tables_data)}")
    print(f"  - 查询数量: {len(queries_data)}")
    print(f"  - Ground Truth条目: {len(ground_truth)}")
    
    # 2. 转换数据格式
    tables = []
    for td in tables_data:
        table = TableInfo(
            table_name=td['table_name'],
            columns=[
                ColumnInfo(
                    table_name=td['table_name'],
                    column_name=col.get('column_name', col.get('name', '')),
                    data_type=col.get('data_type', 'unknown'),
                    sample_values=col.get('sample_values', [])[:3]
                )
                for col in td.get('columns', [])[:15]
            ]
        )
        tables.append(table)
    
    # 3. 创建多Agent系统协调器
    print("\n🤖 初始化多智能体系统...")
    orchestrator = EnhancedMultiAgentOrchestrator()
    
    # 初始化系统（包含三层加速工具）
    start_init = time.time()
    await orchestrator.initialize(tables)
    init_time = time.time() - start_init
    print(f"✅ 系统初始化完成，耗时: {init_time:.2f}秒")
    
    # 显示系统状态
    status = orchestrator.get_detailed_status()
    print(f"\n📋 系统状态:")
    print(f"  - Agents数量: {status['system']['num_agents']}")
    print(f"  - Agent列表: {status['system']['agents']}")
    print(f"  - 三层加速状态:")
    print(f"    • Layer 1 (Metadata Filter): {'✅ 启用' if status['acceleration']['layer1_enabled'] else '❌ 禁用'}")
    print(f"    • Layer 2 (Vector Search): {'✅ 启用' if status['acceleration']['layer2_enabled'] else '❌ 禁用'}")
    print(f"    • Layer 3 (LLM Matcher): {'✅ 启用' if status['acceleration']['layer3_enabled'] else '❌ 禁用'}")
    
    # 4. 测试查询处理
    print(f"\n🔬 开始测试查询处理...")
    print("-"*60)
    
    # 评价指标收集
    all_metrics = []
    hit_at_k = {1: 0, 3: 0, 5: 0, 10: 0}
    
    for i, query in enumerate(queries_data, 1):
        query_table_name = query.get('query_table', '')
        query_column = query.get('query_column', '')
        
        # 查找查询表
        query_table = next((t for t in tables if t.table_name == query_table_name), None)
        if not query_table:
            print(f"\n❌ 查询{i}: 找不到表 {query_table_name}")
            continue
        
        print(f"\n📍 查询{i}/{len(queries_data)}: {query_table_name}")
        
        # 构建查询
        query_text = f"Find tables that can be joined with {query_table_name}"
        
        # 执行多Agent协同处理
        start_time = time.time()
        try:
            results = await orchestrator.process_query_with_collaboration(
                query_text,
                query_table,
                strategy="auto"
            )
            query_time = time.time() - start_time
            
            # 提取预测结果
            predictions = []
            for result in results:
                if isinstance(result, dict):
                    table_name = result.get('table', result.get('table_name'))
                    if table_name:
                        predictions.append(table_name)
                else:
                    # Handle other result formats
                    predictions.append(str(result))
            
            # 获取ground truth
            gt_key = f"{query_table_name}:{query_column}" if query_column else query_table_name
            true_matches = ground_truth.get(gt_key, [])
            
            # 计算Hit@K
            for k in [1, 3, 5, 10]:
                if any(p in true_matches for p in predictions[:k]):
                    hit_at_k[k] += 1
            
            # 计算评价指标
            precision = len(set(predictions[:10]) & set(true_matches)) / min(10, len(predictions)) if predictions else 0
            recall = len(set(predictions[:10]) & set(true_matches)) / len(true_matches) if true_matches else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            
            # 找到第一个正确答案的位置（MRR）
            mrr = 0
            for idx, pred in enumerate(predictions[:10], 1):
                if pred in true_matches:
                    mrr = 1.0 / idx
                    break
            
            metric = EvaluationMetrics(
                precision=precision,
                recall=recall,
                f1_score=f1,
                mrr=mrr,
                query_time=query_time
            )
            all_metrics.append(metric)
            
            # 显示结果
            print(f"  ⏱️ 查询时间: {query_time:.2f}秒")
            print(f"  📊 返回结果数: {len(results)}")
            print(f"  🎯 Ground Truth: {true_matches[:3]}{'...' if len(true_matches) > 3 else ''}")
            print(f"  🔮 预测Top-5: {predictions[:5]}")
            print(f"  📈 指标: P={precision:.2f}, R={recall:.2f}, F1={f1:.2f}, MRR={mrr:.2f}")
            
            # 检查正确答案位置
            for gt in true_matches[:3]:
                if gt in predictions:
                    idx = predictions.index(gt)
                    print(f"    ✅ '{gt}' 在第{idx+1}位")
                else:
                    print(f"    ❌ '{gt}' 不在结果中")
            
        except Exception as e:
            print(f"  ❌ 处理失败: {e}")
            continue
    
    # 5. 汇总评价指标
    print("\n" + "="*60)
    print("📊 评价指标汇总")
    print("="*60)
    
    if all_metrics:
        avg_precision = np.mean([m.precision for m in all_metrics])
        avg_recall = np.mean([m.recall for m in all_metrics])
        avg_f1 = np.mean([m.f1_score for m in all_metrics])
        avg_mrr = np.mean([m.mrr for m in all_metrics])
        avg_time = np.mean([m.query_time for m in all_metrics])
        
        print(f"\n🎯 平均指标:")
        print(f"  - Precision: {avg_precision:.3f}")
        print(f"  - Recall: {avg_recall:.3f}")
        print(f"  - F1-Score: {avg_f1:.3f}")
        print(f"  - MRR: {avg_mrr:.3f}")
        print(f"  - 查询时间: {avg_time:.2f}秒")
        
        print(f"\n📈 Hit@K:")
        for k in [1, 3, 5, 10]:
            hit_rate = hit_at_k[k] / len(queries_data) * 100
            print(f"  - Hit@{k}: {hit_at_k[k]}/{len(queries_data)} ({hit_rate:.1f}%)")
    
    # 6. 显示Agent协作细节
    final_status = orchestrator.get_detailed_status()
    print(f"\n🤝 Agent协作统计:")
    print(f"  - 处理查询数: {final_status['performance']['queries_processed']}")
    print(f"  - 平均处理时间: {final_status['performance']['avg_time']:.2f}秒")
    print(f"  - 消息总数: {final_status['system']['message_count']}")
    
    # 显示优化器报告
    if 'optimizer_report' in final_status and final_status['optimizer_report']:
        opt_report = final_status['optimizer_report']
        if 'recommendations' in opt_report:
            print(f"\n💡 优化建议:")
            for rec in opt_report['recommendations']:
                print(f"  - {rec}")
    
    print("\n" + "="*80)
    print("✅ 多智能体系统集成测试完成！")
    print("="*80)
    
    # 7. 对比分析
    print("\n" + "="*80)
    print("📊 架构对比分析")
    print("="*80)
    print("\n✨ 多智能体系统优势:")
    print("  1. 每个Agent独立决策，可根据任务复杂度选择使用LLM或规则")
    print("  2. Agent间协同工作，共享信息和优化策略")
    print("  3. 三层加速作为工具供Agent调用，而非固定流程")
    print("  4. OptimizerAgent动态调整系统性能")
    print("  5. 支持并行处理和智能缓存")
    
    print("\n🔄 与纯三层加速的区别:")
    print("  • 三层加速: Layer1 → Layer2 → Layer3 (固定流程)")
    print("  • 多Agent+三层: Agents协同决策 + 选择性使用加速层")
    print("    - PlannerAgent: 可用LLM分析复杂查询")
    print("    - AnalyzerAgent: 可用LLM深度分析schema")
    print("    - SearcherAgent: 灵活选择Layer1/Layer2/混合")
    print("    - MatcherAgent: 智能选择是否调用Layer3")
    print("    - AggregatorAgent: 可用LLM重排序")
    print("    - OptimizerAgent: 动态优化系统配置")
    
    return all_metrics


if __name__ == "__main__":
    # 运行测试
    asyncio.run(test_multi_agent_with_acceleration())