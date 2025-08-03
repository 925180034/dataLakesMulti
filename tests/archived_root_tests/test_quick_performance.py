#!/usr/bin/env python
"""
快速性能测试 - 测试单个查询性能
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

# 减少某些组件的日志噪音
logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
logging.getLogger("src.agents").setLevel(logging.WARNING)
logging.getLogger("src.tools").setLevel(logging.WARNING)


async def test_single_query():
    """测试单个查询的性能"""
    from src.core.workflow import discover_data
    from src.utils.data_parser import parse_tables_data
    
    print("\n" + "="*60)
    print("⚡ 数据湖多智能体系统 - 快速性能测试")
    print("="*60)
    
    # 加载数据
    print("\n📊 加载测试数据...")
    with open('examples/final_subset_tables.json') as f:
        all_tables_data = json.load(f)
    
    print(f"✅ 已加载 {len(all_tables_data)} 个表")
    
    # 选择一个测试查询
    query_table = all_tables_data[0]  # 使用第一个表作为查询
    
    print(f"\n📝 测试查询表: {query_table['table_name']}")
    print(f"  - 列数: {len(query_table['columns'])}")
    print(f"  - 目标: 3-8秒内返回结果")
    
    # 第一次运行（包含初始化）
    print("\n" + "-"*60)
    print("🔄 第一次运行（包含初始化开销）")
    print("-"*60)
    
    start_time = time.time()
    
    try:
        result = await discover_data(
            user_query=f"Find joinable tables for {query_table['table_name']}",
            query_tables=[query_table],
            all_tables_data=all_tables_data,
            use_optimized=True
        )
        
        first_time = time.time() - start_time
        
        # 检查结果
        if hasattr(result, 'table_matches') and result.table_matches:
            match_count = len(result.table_matches)
            print(f"\n✅ 找到 {match_count} 个匹配")
            print(f"⏱️ 查询时间: {first_time:.2f}秒")
            
            # 显示前3个匹配
            print("\n匹配结果（前3个）:")
            for i, match in enumerate(result.table_matches[:3], 1):
                print(f"  {i}. {match.target_table} (分数: {match.score:.2f})")
        else:
            print(f"\n❌ 未找到匹配")
            print(f"⏱️ 查询时间: {first_time:.2f}秒")
            
    except Exception as e:
        logger.error(f"查询失败: {e}")
        first_time = 30.0  # 默认超时时间
    
    # 第二次运行（使用缓存）
    print("\n" + "-"*60)
    print("🔄 第二次运行（使用缓存）")
    print("-"*60)
    
    start_time = time.time()
    
    try:
        result = await discover_data(
            user_query=f"Find joinable tables for {query_table['table_name']}",
            query_tables=[query_table],
            use_optimized=True  # 不再初始化，使用缓存
        )
        
        second_time = time.time() - start_time
        
        # 检查结果
        if hasattr(result, 'table_matches') and result.table_matches:
            match_count = len(result.table_matches)
            print(f"\n✅ 找到 {match_count} 个匹配")
            print(f"⏱️ 查询时间: {second_time:.2f}秒（缓存加速）")
        else:
            print(f"\n❌ 未找到匹配")
            print(f"⏱️ 查询时间: {second_time:.2f}秒")
            
    except Exception as e:
        logger.error(f"查询失败: {e}")
        second_time = 30.0
    
    # 性能评估
    print("\n" + "="*60)
    print("🎯 性能评估")
    print("="*60)
    
    print(f"\n第一次运行: {first_time:.2f}秒（含初始化）")
    print(f"第二次运行: {second_time:.2f}秒（使用缓存）")
    
    if first_time < second_time:
        speedup = first_time / second_time if second_time > 0 else 1
        print(f"缓存加速比: {speedup:.2f}x")
    
    # 判断是否达标
    target_met = False
    if second_time <= 8:
        print(f"\n✅ 性能达标！查询时间 {second_time:.2f}秒 ≤ 8秒目标")
        target_met = True
    elif second_time <= 15:
        print(f"\n⚠️ 接近目标。查询时间 {second_time:.2f}秒，略高于8秒目标")
        print("\n优化建议:")
        print("  1. 进一步减少LLM调用")
        print("  2. 增加批处理大小")
        print("  3. 优化向量搜索")
    else:
        print(f"\n❌ 性能未达标。查询时间 {second_time:.2f}秒 > 8秒目标")
        print("\n需要深度优化:")
        print("  1. 重新设计工作流减少串行步骤")
        print("  2. 实现更激进的缓存策略")
        print("  3. 考虑使用更快的LLM模型")
        print("  4. 优化数据结构和算法")
    
    # 保存结果
    results = {
        "query_table": query_table['table_name'],
        "first_run": first_time,
        "second_run": second_time,
        "target_met": target_met,
        "target_range": "3-8 seconds"
    }
    
    with open('quick_performance_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\n✅ 测试结果已保存到 quick_performance_results.json")
    
    return target_met


if __name__ == "__main__":
    target_met = asyncio.run(test_single_query())
    
    # 退出码：0表示达标，1表示未达标
    exit(0 if target_met else 1)