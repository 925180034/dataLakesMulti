#!/usr/bin/env python
"""
测试优化后的系统性能
验证是否达到3-8秒的查询目标
"""

import asyncio
import json
import time
import logging
from pathlib import Path

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 禁用某些详细日志
logging.getLogger("sentence_transformers").setLevel(logging.WARNING)


async def test_optimized_performance():
    """测试优化后的性能"""
    from src.core.workflow import discover_data
    from src.utils.data_parser import parse_tables_data
    
    print("\n" + "="*60)
    print("🚀 数据湖多智能体系统 - 优化性能测试")
    print("="*60)
    
    # 加载数据
    print("\n📊 加载测试数据...")
    with open('examples/final_subset_tables.json') as f:
        all_tables_data = json.load(f)
    
    print(f"✅ 已加载 {len(all_tables_data)} 个表")
    
    # 准备测试查询
    test_queries = all_tables_data[:5]  # 测试前5个表
    
    print(f"\n📝 测试配置:")
    print(f"  - 查询数量: {len(test_queries)}")
    print(f"  - 数据集大小: {len(all_tables_data)} 表")
    print(f"  - 优化功能: 批量API调用, 并行处理, 多级缓存")
    print(f"  - 目标性能: 3-8秒/查询")
    
    # 第一次运行（包含初始化）
    print("\n" + "-"*60)
    print("🔄 第一次运行（包含初始化开销）")
    print("-"*60)
    
    first_run_times = []
    for i, query_table in enumerate(test_queries, 1):
        print(f"\n查询 {i}/{len(test_queries)}: {query_table['table_name']}")
        
        start_time = time.time()
        
        try:
            result = await discover_data(
                user_query=f"Find joinable tables for {query_table['table_name']}",
                query_tables=[query_table],
                all_tables_data=all_tables_data if i == 1 else None,  # 只第一次初始化
                use_optimized=True
            )
            
            query_time = time.time() - start_time
            first_run_times.append(query_time)
            
            # 检查结果
            if hasattr(result, 'table_matches') and result.table_matches:
                match_count = len(result.table_matches)
                print(f"  ✅ 找到 {match_count} 个匹配")
                print(f"  ⏱️ 查询时间: {query_time:.2f}秒")
                
                # 显示前3个匹配
                for j, match in enumerate(result.table_matches[:3], 1):
                    print(f"     {j}. {match.target_table} (分数: {match.score:.2f})")
            else:
                print(f"  ❌ 未找到匹配")
                print(f"  ⏱️ 查询时间: {query_time:.2f}秒")
                
        except Exception as e:
            logger.error(f"查询失败: {e}")
            first_run_times.append(30.0)  # 默认超时时间
    
    # 第二次运行（使用缓存）
    print("\n" + "-"*60)
    print("🔄 第二次运行（使用缓存）")
    print("-"*60)
    
    second_run_times = []
    for i, query_table in enumerate(test_queries, 1):
        print(f"\n查询 {i}/{len(test_queries)}: {query_table['table_name']}")
        
        start_time = time.time()
        
        try:
            result = await discover_data(
                user_query=f"Find joinable tables for {query_table['table_name']}",
                query_tables=[query_table],
                use_optimized=True  # 不再初始化，使用缓存
            )
            
            query_time = time.time() - start_time
            second_run_times.append(query_time)
            
            # 检查结果
            if hasattr(result, 'table_matches') and result.table_matches:
                match_count = len(result.table_matches)
                print(f"  ✅ 找到 {match_count} 个匹配")
                print(f"  ⏱️ 查询时间: {query_time:.2f}秒（缓存加速）")
            else:
                print(f"  ❌ 未找到匹配")
                print(f"  ⏱️ 查询时间: {query_time:.2f}秒")
                
        except Exception as e:
            logger.error(f"查询失败: {e}")
            second_run_times.append(30.0)
    
    # 性能分析
    print("\n" + "="*60)
    print("📊 性能分析报告")
    print("="*60)
    
    if first_run_times:
        avg_first = sum(first_run_times) / len(first_run_times)
        min_first = min(first_run_times)
        max_first = max(first_run_times)
        
        print("\n第一次运行（含初始化）:")
        print(f"  - 平均时间: {avg_first:.2f}秒")
        print(f"  - 最快时间: {min_first:.2f}秒")
        print(f"  - 最慢时间: {max_first:.2f}秒")
    
    if second_run_times:
        avg_second = sum(second_run_times) / len(second_run_times)
        min_second = min(second_run_times)
        max_second = max(second_run_times)
        
        print("\n第二次运行（使用缓存）:")
        print(f"  - 平均时间: {avg_second:.2f}秒")
        print(f"  - 最快时间: {min_second:.2f}秒")
        print(f"  - 最慢时间: {max_second:.2f}秒")
        
        if first_run_times:
            speedup = avg_first / avg_second
            print(f"\n缓存加速比: {speedup:.2f}x")
    
    # 性能评估
    print("\n" + "-"*60)
    print("🎯 性能目标评估")
    print("-"*60)
    
    target_met = False
    if second_run_times:
        # 使用缓存后的性能作为评估标准
        if avg_second <= 8:
            print(f"✅ 性能达标！平均查询时间 {avg_second:.2f}秒 ≤ 8秒目标")
            target_met = True
        else:
            print(f"⚠️ 性能未达标。平均查询时间 {avg_second:.2f}秒 > 8秒目标")
            print("\n建议优化方向:")
            print("  1. 进一步减少LLM调用次数")
            print("  2. 优化向量搜索算法")
            print("  3. 实现更激进的缓存策略")
            print("  4. 使用更快的LLM模型")
    
    # 保存测试结果
    results = {
        "test_config": {
            "query_count": len(test_queries),
            "dataset_size": len(all_tables_data),
            "optimizations": ["batch_api", "parallel_processing", "multi_level_cache"]
        },
        "first_run": {
            "times": first_run_times,
            "avg": avg_first if first_run_times else 0,
            "min": min_first if first_run_times else 0,
            "max": max_first if first_run_times else 0
        },
        "second_run": {
            "times": second_run_times,
            "avg": avg_second if second_run_times else 0,
            "min": min_second if second_run_times else 0,
            "max": max_second if second_run_times else 0
        },
        "performance": {
            "cache_speedup": speedup if first_run_times and second_run_times else 1,
            "target_met": target_met,
            "target_range": "3-8 seconds"
        }
    }
    
    with open('optimized_performance_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\n✅ 测试结果已保存到 optimized_performance_results.json")
    
    return target_met


if __name__ == "__main__":
    target_met = asyncio.run(test_optimized_performance())
    
    # 退出码：0表示达标，1表示未达标
    exit(0 if target_met else 1)