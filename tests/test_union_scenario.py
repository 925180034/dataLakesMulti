#!/usr/bin/env python3
"""
Union场景测试脚本
"""

import json
import subprocess
import sys
from pathlib import Path

# 添加项目根目录到Python路径
sys.path.insert(0, str(Path(__file__).parent.absolute()))


def create_union_test_data():
    """创建Union场景测试数据"""
    print("=== 创建Union测试数据 ===")
    
    # 使用Join场景的表数据作为Union测试的候选表
    with open("examples/webtable_join_tables.json", 'r', encoding='utf-8') as f:
        tables_data = json.load(f)
    
    # 选择几个表作为Union测试
    union_tables = tables_data[:5]  # 使用前5个表
    
    # 保存Union测试表
    with open("examples/webtable_union_tables.json", 'w', encoding='utf-8') as f:
        json.dump(union_tables, f, ensure_ascii=False, indent=2)
    
    print(f"创建了 {len(union_tables)} 个Union测试表")
    
    # 显示表信息
    for i, table in enumerate(union_tables):
        print(f"  {i+1}. {table['table_name']} ({len(table['columns'])} 列)")


def test_union_scenario():
    """测试Union场景"""
    print("\n=== Union场景测试 ===")
    
    try:
        # 创建测试数据
        create_union_test_data()
        
        print("\n🔍 运行Union发现测试...")
        
        # 使用CLI测试Union场景
        result = subprocess.run([
            "python", "run_cli.py", "discover",
            "-q", "find tables that can be merged together for union operations",
            "-t", "examples/webtable_union_tables.json",
            "-f", "json"
        ], capture_output=True, text=True, timeout=60)
        
        if result.returncode == 0:
            print("✅ Union场景测试成功")
            
            # 解析结果
            try:
                import re
                json_match = re.search(r'\{.*\}', result.stdout, re.DOTALL)
                if json_match:
                    response_data = json.loads(json_match.group())
                    print(f"策略: {response_data.get('strategy', 'unknown')}")
                    print(f"结果数量: {response_data.get('results_count', 0)}")
                    
                    results = response_data.get('results', [])
                    if results:
                        print("匹配结果:")
                        for i, result in enumerate(results[:3]):
                            table_name = result.get('table_name', 'unknown')
                            confidence = result.get('confidence', 0)
                            print(f"  {i+1}. {table_name} (置信度: {confidence:.3f})")
                    else:
                        print("⚠️ 没有找到匹配结果")
                        
                    # 显示最终报告
                    final_report = response_data.get('final_report', '')
                    if final_report:
                        print(f"\n📋 最终报告:\n{final_report}")
                        
                else:
                    print("⚠️ 无法解析JSON结果")
                    print(f"输出:\n{result.stdout}")
                    
            except Exception as e:
                print(f"⚠️ 解析结果失败: {e}")
                print(f"原始输出:\n{result.stdout}")
                
        else:
            print(f"❌ Union场景测试失败")
            print(f"错误: {result.stderr}")
            print(f"输出: {result.stdout}")
            
    except subprocess.TimeoutExpired:
        print("❌ Union测试超时")
    except Exception as e:
        print(f"❌ Union测试异常: {e}")


def analyze_union_ground_truth():
    """分析Union场景的真实匹配数据"""
    print("\n=== Union真实匹配分析 ===")
    
    try:
        with open("examples/webtable_union_ground_truth.json", 'r', encoding='utf-8') as f:
            union_gt = json.load(f)
        
        print(f"Union真实匹配数量: {len(union_gt)}")
        
        # 统计查询表频率
        query_table_count = {}
        for gt in union_gt:
            query_table = gt.get('query_table', '')
            query_table_count[query_table] = query_table_count.get(query_table, 0) + 1
        
        # 显示最频繁的查询表
        sorted_tables = sorted(query_table_count.items(), key=lambda x: x[1], reverse=True)
        print("\n最频繁的查询表 (前5个):")
        for i, (table, count) in enumerate(sorted_tables[:5]):
            print(f"  {i+1}. {table}: {count} 个匹配")
        
        # 显示样本匹配
        print(f"\nUnion匹配样本 (前5个):")
        for i, gt in enumerate(union_gt[:5]):
            query_table = gt.get('query_table', '')
            candidate_table = gt.get('candidate_table', '')
            print(f"  {i+1}. {query_table} -> {candidate_table}")
            
    except Exception as e:
        print(f"❌ Union真实匹配分析失败: {e}")


if __name__ == "__main__":
    analyze_union_ground_truth()
    test_union_scenario()