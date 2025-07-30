#!/usr/bin/env python3
"""
简单的WebTable数据集测试
"""

import json
import sys
from pathlib import Path

# 添加项目根目录到Python路径
sys.path.insert(0, str(Path(__file__).parent.absolute()))


def test_simple_discovery():
    """简单测试数据发现功能"""
    print("=== WebTable简单测试 ===")
    
    # 使用CLI进行测试
    import subprocess
    
    try:
        # 使用真实的WebTable数据集测试Join场景
        print("🔍 测试Join场景...")
        result = subprocess.run([
            "python", "run_cli.py", "discover",
            "-q", "find tables with similar column structures",
            "-t", "examples/webtable_join_tables.json",
            "-f", "json"
        ], capture_output=True, text=True, timeout=30)
        
        if result.returncode == 0:
            print("✅ Join场景测试成功")
            
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
                        print("前3个匹配结果:")
                        for i, result in enumerate(results[:3]):
                            print(f"  {i+1}. {result.get('table_name', 'unknown')}")
                else:
                    print("⚠️ 无法解析JSON结果")
                    print(f"输出:\n{result.stdout}")
            except Exception as e:
                print(f"⚠️ 解析结果失败: {e}")
                print(f"原始输出:\n{result.stdout}")
        else:
            print(f"❌ Join场景测试失败: {result.stderr}")
            print(f"输出: {result.stdout}")
        
        print("\n" + "="*50)
        
        # 测试列匹配场景
        print("🔍 测试列匹配场景...")
        result2 = subprocess.run([
            "python", "run_cli.py", "discover", 
            "-q", "find columns that can be joined together",
            "-c", "examples/webtable_join_columns.json",
            "-f", "json"
        ], capture_output=True, text=True, timeout=30)
        
        if result2.returncode == 0:
            print("✅ 列匹配场景测试成功")
            
            # 解析结果
            try:
                json_match = re.search(r'\{.*\}', result2.stdout, re.DOTALL)
                if json_match:
                    response_data = json.loads(json_match.group())
                    print(f"策略: {response_data.get('strategy', 'unknown')}")
                    print(f"结果数量: {response_data.get('results_count', 0)}")
                else:
                    print("⚠️ 无法解析JSON结果") 
                    print(f"输出:\n{result2.stdout}")
            except Exception as e:
                print(f"⚠️ 解析结果失败: {e}")
                print(f"原始输出:\n{result2.stdout}")
        else:
            print(f"❌ 列匹配场景测试失败: {result2.stderr}")
            print(f"输出: {result2.stdout}")
        
    except subprocess.TimeoutExpired:
        print("❌ 测试超时")
    except Exception as e:
        print(f"❌ 测试异常: {e}")


def analyze_dataset():
    """分析数据集基本信息"""
    print("\n=== 数据集分析 ===")
    
    try:
        # 分析表数据集
        with open("examples/webtable_join_tables.json", 'r', encoding='utf-8') as f:
            tables = json.load(f)
        
        print(f"表数量: {len(tables)}")
        
        # 统计列信息
        total_columns = 0
        data_types = {}
        
        for table in tables:
            columns = table.get('columns', [])
            total_columns += len(columns)
            
            for col in columns:
                dtype = col.get('data_type', 'unknown')
                data_types[dtype] = data_types.get(dtype, 0) + 1
        
        print(f"总列数: {total_columns}")
        print(f"数据类型分布: {data_types}")
        
        # 显示几个表的样本
        print(f"\n表样本 (前3个):")
        for i, table in enumerate(tables[:3]):
            print(f"  {i+1}. {table['table_name']} ({len(table['columns'])} 列)")
            for col in table['columns'][:2]:
                print(f"     - {col['column_name']} ({col['data_type']})")
        
        # 分析查询数据
        with open("examples/webtable_join_queries.json", 'r', encoding='utf-8') as f:
            queries = json.load(f)
        
        print(f"\n查询数量: {len(queries)}")
        print(f"查询样本 (前3个):")
        for i, query in enumerate(queries[:3]):
            print(f"  {i+1}. {query['query_table']} -> {query['query_column']}")
        
        # 分析真实匹配数据
        with open("examples/webtable_join_ground_truth.json", 'r', encoding='utf-8') as f:
            ground_truth = json.load(f)
        
        print(f"\n真实匹配数量: {len(ground_truth)}")
        print(f"匹配样本 (前3个):")
        for i, gt in enumerate(ground_truth[:3]):
            print(f"  {i+1}. {gt['query_table']}.{gt['query_column']} -> {gt['candidate_table']}.{gt['candidate_column']}")
        
    except Exception as e:
        print(f"❌ 数据集分析失败: {e}")


if __name__ == "__main__":
    analyze_dataset()
    test_simple_discovery()