#!/usr/bin/env python
"""
快速测试脚本 - 简单的查询测试
"""

import asyncio
import json
import time
from src.core.workflow import discover_data

async def main():
    print("="*60)
    print("数据湖多智能体系统 - 快速测试")
    print("="*60)
    
    # 1. 加载数据
    print("\n1. 加载数据...")
    with open('examples/final_subset_tables.json') as f:
        all_tables = json.load(f)
    print(f"   ✅ 已加载 {len(all_tables)} 个表")
    
    # 2. 选择查询表
    query_table = all_tables[0]  # 使用第一个表
    print(f"\n2. 查询表: {query_table['table_name']}")
    print(f"   列数: {len(query_table['columns'])}")
    
    # 3. 执行查询
    print(f"\n3. 执行查询...")
    start_time = time.time()
    
    result = await discover_data(
        user_query=f"Find joinable tables for {query_table['table_name']}",
        query_tables=[query_table],
        all_tables_data=all_tables,
        use_optimized=True  # 使用优化版本
    )
    
    query_time = time.time() - start_time
    
    # 4. 显示结果
    print(f"\n4. 查询结果:")
    if hasattr(result, 'table_matches') and result.table_matches:
        print(f"   ✅ 找到 {len(result.table_matches)} 个匹配")
        print(f"   ⏱️ 查询时间: {query_time:.2f}秒")
        print(f"\n   前5个匹配:")
        for i, match in enumerate(result.table_matches[:5], 1):
            print(f"   {i}. {match.target_table} (分数: {match.score:.2f})")
    else:
        print(f"   ❌ 未找到匹配")
        print(f"   ⏱️ 查询时间: {query_time:.2f}秒")
    
    # 5. 性能评估
    print(f"\n5. 性能评估:")
    if query_time <= 8:
        print(f"   ✅ 性能达标！{query_time:.2f}秒 ≤ 8秒目标")
    else:
        print(f"   ⚠️ 性能略高于目标: {query_time:.2f}秒")
    
    # 6. 保存结果
    output = {
        "query_table": query_table['table_name'],
        "matches_found": len(result.table_matches) if hasattr(result, 'table_matches') else 0,
        "query_time": query_time,
        "top_matches": [
            {"table": m.target_table, "score": m.score}
            for m in (result.table_matches[:5] if hasattr(result, 'table_matches') and result.table_matches else [])
        ]
    }
    
    with open('quick_test_result.json', 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\n✅ 结果已保存到 quick_test_result.json")
    print("="*60)

if __name__ == "__main__":
    asyncio.run(main())