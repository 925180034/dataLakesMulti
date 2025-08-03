#!/usr/bin/env python
"""
批量评估脚本 - 测试多个查询
模拟真实的评估场景
"""

import asyncio
import json
import time
import logging
from typing import List, Dict, Any

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 减少噪音
logging.getLogger("src").setLevel(logging.WARNING)

async def batch_evaluate(num_queries: int = 10):
    """
    批量评估多个查询
    
    Args:
        num_queries: 要测试的查询数量
    """
    from src.core.workflow import discover_data
    
    print("="*70)
    print(f"📊 数据湖多智能体系统 - 批量评估 ({num_queries} 个查询)")
    print("="*70)
    
    # 1. 加载数据
    print("\n📂 加载数据...")
    with open('examples/final_subset_tables.json') as f:
        all_tables = json.load(f)
    print(f"✅ 已加载 {len(all_tables)} 个表")
    
    # 2. 准备查询（选择前num_queries个表作为查询）
    query_tables = all_tables[:num_queries]
    print(f"\n🔍 准备 {num_queries} 个查询:")
    for i, table in enumerate(query_tables[:5], 1):  # 只显示前5个
        print(f"  {i}. {table['table_name']}")
    if num_queries > 5:
        print(f"  ... 还有 {num_queries-5} 个查询")
    
    # 3. 执行批量查询
    print(f"\n⚡ 开始批量查询...")
    print("-"*70)
    
    results = []
    total_matches = 0
    query_times = []
    llm_calls_total = 0
    
    # 第一个查询包含初始化
    start_total = time.time()
    
    for i, query_table in enumerate(query_tables, 1):
        print(f"\n查询 {i}/{num_queries}: {query_table['table_name']}")
        
        start_time = time.time()
        
        try:
            # 执行查询（只第一次初始化all_tables_data）
            result = await discover_data(
                user_query=f"Find joinable tables for {query_table['table_name']}",
                query_tables=[query_table],
                all_tables_data=all_tables if i == 1 else None,  # 只第一次初始化
                use_optimized=True
            )
            
            query_time = time.time() - start_time
            query_times.append(query_time)
            
            # 收集结果
            if hasattr(result, 'table_matches') and result.table_matches:
                match_count = len(result.table_matches)
                total_matches += match_count
                print(f"  ✅ 找到 {match_count} 个匹配 (耗时: {query_time:.2f}秒)")
                
                # 显示前3个匹配
                for j, match in enumerate(result.table_matches[:3], 1):
                    print(f"     {j}. {match.target_table} (分数: {match.score:.2f})")
                    
                results.append({
                    "query": query_table['table_name'],
                    "matches": match_count,
                    "time": query_time,
                    "top_match": result.table_matches[0].target_table if result.table_matches else None
                })
            else:
                print(f"  ❌ 未找到匹配 (耗时: {query_time:.2f}秒)")
                results.append({
                    "query": query_table['table_name'],
                    "matches": 0,
                    "time": query_time,
                    "top_match": None
                })
                
        except Exception as e:
            logger.error(f"查询失败: {e}")
            query_times.append(30.0)  # 默认超时
            results.append({
                "query": query_table['table_name'],
                "matches": 0,
                "time": 30.0,
                "error": str(e)
            })
    
    total_time = time.time() - start_total
    
    # 4. 统计分析
    print("\n" + "="*70)
    print("📊 批量评估结果")
    print("="*70)
    
    avg_time = sum(query_times) / len(query_times) if query_times else 0
    min_time = min(query_times) if query_times else 0
    max_time = max(query_times) if query_times else 0
    
    # 分离初始化时间
    avg_time_no_init = sum(query_times[1:]) / len(query_times[1:]) if len(query_times) > 1 else avg_time
    
    print(f"\n📈 性能统计:")
    print(f"  • 查询总数: {num_queries}")
    print(f"  • 总耗时: {total_time:.2f}秒")
    print(f"  • 平均每查询: {avg_time:.2f}秒")
    print(f"  • 平均每查询(不含初始化): {avg_time_no_init:.2f}秒")
    print(f"  • 最快查询: {min_time:.2f}秒")
    print(f"  • 最慢查询: {max_time:.2f}秒")
    
    print(f"\n🎯 匹配统计:")
    print(f"  • 总匹配数: {total_matches}")
    print(f"  • 平均每查询匹配数: {total_matches/num_queries:.1f}")
    print(f"  • 成功查询率: {len([r for r in results if r['matches'] > 0])/num_queries*100:.1f}%")
    
    print(f"\n⚡ 性能评估:")
    if avg_time_no_init <= 3:
        print(f"  ✅ 优秀！平均 {avg_time_no_init:.2f}秒 ≤ 3秒")
    elif avg_time_no_init <= 8:
        print(f"  ✅ 达标！平均 {avg_time_no_init:.2f}秒 ≤ 8秒目标")
    else:
        print(f"  ⚠️ 未达标。平均 {avg_time_no_init:.2f}秒 > 8秒目标")
    
    # 5. 保存详细结果
    output = {
        "summary": {
            "total_queries": num_queries,
            "total_time": total_time,
            "avg_time": avg_time,
            "avg_time_no_init": avg_time_no_init,
            "min_time": min_time,
            "max_time": max_time,
            "total_matches": total_matches,
            "success_rate": len([r for r in results if r['matches'] > 0])/num_queries
        },
        "results": results
    }
    
    with open('batch_evaluation_results.json', 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\n💾 详细结果已保存到 batch_evaluation_results.json")
    print("="*70)
    
    return output


async def main():
    """主函数"""
    import sys
    
    # 从命令行获取查询数量
    num_queries = 10  # 默认10个
    if len(sys.argv) > 1:
        try:
            num_queries = int(sys.argv[1])
        except:
            print("用法: python batch_evaluation.py [查询数量]")
            print("示例: python batch_evaluation.py 20")
            return
    
    # 确保不超过可用表数量
    with open('examples/final_subset_tables.json') as f:
        all_tables = json.load(f)
    num_queries = min(num_queries, len(all_tables))
    
    # 运行评估
    await batch_evaluate(num_queries)


if __name__ == "__main__":
    asyncio.run(main())