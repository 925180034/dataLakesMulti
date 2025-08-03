#!/usr/bin/env python
"""
快速性能测试脚本 - 测试优化后的系统性能
"""

import asyncio
import json
import time
import logging
from pathlib import Path

# 设置日志级别为WARNING，减少输出
logging.basicConfig(level=logging.WARNING)

async def test_performance():
    """测试系统性能"""
    from src.core.workflow import discover_data
    
    # 加载数据
    print("📊 数据湖多智能体系统 - 性能测试")
    print("="*60)
    
    with open('examples/final_subset_tables.json') as f:
        all_tables = json.load(f)
    
    print(f"数据集规模: {len(all_tables)} 表")
    print("测试查询数: 3")
    print()
    
    # 测试查询
    test_tables = all_tables[:3]
    query_times = []
    results_summary = []
    
    for i, query_table in enumerate(test_tables, 1):
        print(f"查询 {i}/{len(test_tables)}: {query_table['table_name']}")
        
        start_time = time.time()
        
        try:
            # 使用基础工作流（避免初始化开销）
            result = await discover_data(
                user_query=f"Find joinable tables for {query_table['table_name']}",
                query_tables=[query_table],
                use_optimized=False  # 使用基础工作流
            )
            
            query_time = time.time() - start_time
            query_times.append(query_time)
            
            # 统计结果
            if hasattr(result, 'table_matches') and result.table_matches:
                match_count = len(result.table_matches)
                top_match = result.table_matches[0] if match_count > 0 else None
                
                results_summary.append({
                    'query': query_table['table_name'],
                    'matches': match_count,
                    'time': query_time,
                    'top_match': top_match.target_table if top_match else None
                })
                
                print(f"  ✅ 找到 {match_count} 个匹配 (耗时: {query_time:.2f}秒)")
                if top_match:
                    print(f"     最佳匹配: {top_match.target_table} (分数: {top_match.score:.2f})")
            else:
                print(f"  ❌ 未找到匹配 (耗时: {query_time:.2f}秒)")
                results_summary.append({
                    'query': query_table['table_name'],
                    'matches': 0,
                    'time': query_time,
                    'top_match': None
                })
                
        except Exception as e:
            print(f"  ❌ 查询失败: {e}")
            query_times.append(30.0)  # 默认超时时间
    
    # 性能统计
    print()
    print("📈 性能统计")
    print("="*60)
    
    if query_times:
        avg_time = sum(query_times) / len(query_times)
        min_time = min(query_times)
        max_time = max(query_times)
        
        print(f"平均查询时间: {avg_time:.2f}秒")
        print(f"最快查询时间: {min_time:.2f}秒")
        print(f"最慢查询时间: {max_time:.2f}秒")
        
        # 判断是否达标
        print()
        if avg_time <= 8:
            print(f"✅ 性能达标！目标: 3-8秒, 实际: {avg_time:.2f}秒")
        else:
            print(f"⚠️ 性能未达标。目标: 3-8秒, 实际: {avg_time:.2f}秒")
            
            # 分析瓶颈
            print()
            print("🔍 性能瓶颈分析:")
            print("  - 可能原因1: LLM调用耗时过长")
            print("  - 可能原因2: 向量搜索效率低")
            print("  - 可能原因3: 缺少有效缓存")
            print("  - 建议: 启用并行处理和批量优化")
    
    # 保存结果
    with open('performance_test_results.json', 'w') as f:
        json.dump({
            'summary': {
                'total_queries': len(test_tables),
                'avg_time': avg_time if query_times else 0,
                'min_time': min_time if query_times else 0,
                'max_time': max_time if query_times else 0,
                'performance_target_met': avg_time <= 8 if query_times else False
            },
            'details': results_summary
        }, f, indent=2)
    
    print()
    print("✅ 结果已保存到 performance_test_results.json")


if __name__ == "__main__":
    asyncio.run(test_performance())