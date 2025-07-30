#!/usr/bin/env python3
"""
WebTable数据集测试脚本
"""

import json
import asyncio
import sys
from pathlib import Path

# 添加项目根目录到Python路径
sys.path.insert(0, str(Path(__file__).parent.absolute()))

from src.core.workflow import discover_data
from src.core.models import AgentState, TaskStrategy, ColumnInfo, TableInfo


def load_sample_data():
    """加载样本数据"""
    
    # 加载表数据
    with open("examples/webtable_join_tables.json", 'r', encoding='utf-8') as f:
        tables_data = json.load(f)
    
    # 加载查询数据
    with open("examples/webtable_join_queries.json", 'r', encoding='utf-8') as f:
        queries_data = json.load(f)
    
    # 加载真实匹配数据
    with open("examples/webtable_join_ground_truth.json", 'r', encoding='utf-8') as f:
        ground_truth_data = json.load(f)
    
    return tables_data, queries_data, ground_truth_data


async def test_join_scenario():
    """测试Join场景"""
    print("=== WebTable Join场景测试 ===")
    
    # 加载数据
    tables_data, queries_data, ground_truth_data = load_sample_data()
    
    print(f"加载了 {len(tables_data)} 个表")
    print(f"加载了 {len(queries_data)} 个查询")
    print(f"加载了 {len(ground_truth_data)} 个真实匹配")
    
    # 转换表数据格式
    tables = []
    for table_data in tables_data[:10]:  # 使用前10个表进行测试
        table_info = TableInfo(**table_data)
        tables.append(table_info)
    
    # 找到一个存在于我们数据集中的查询
    available_table_names = [table.table_name for table in tables]
    print(f"可用表: {available_table_names[:5]}...")
    
    # 找到匹配的查询
    test_query = None
    for query in queries_data:
        query_table_name = query["query_table"].replace('.csv', '')
        if query_table_name in available_table_names:
            test_query = query
            break
    
    if not test_query:
        print("❌ 未找到可用的测试查询")
        return
    
    print(f"\\n测试查询: {test_query}")
    
    # 构建状态
    try:
        # 构建查询列
        query_table_name = test_query["query_table"].replace('.csv', '')
        query_column_name = test_query["query_column"]
        
        # 寻找查询表
        query_table = None
        for table in tables:
            if table.table_name == query_table_name:
                query_table = table
                break
        
        if not query_table:
            print(f"❌ 未找到查询表: {query_table_name}")
            return
        
        # 寻找查询列
        query_column = query_table.get_column(query_column_name)
        if not query_column:
            print(f"❌ 未找到查询列: {query_column_name}")
            return
        
        print(f"✅ 找到查询列: {query_column.full_name}")
        print(f"   数据类型: {query_column.data_type}")
        print(f"   样本值: {query_column.sample_values[:3]}")
        
        # 执行发现
        print("\\n🔍 开始数据发现...")
        
        # discover_data期望字符串参数，不是AgentState对象
        user_query = f"find columns similar to {query_column.full_name}"
        
        # 转换为discover_data期望的格式
        query_columns_data = []
        query_columns_data.append({
            "table_name": query_column.table_name,
            "column_name": query_column.column_name,
            "data_type": query_column.data_type,
            "sample_values": query_column.sample_values
        })
        
        candidate_tables_data = []
        for table in tables:
            table_dict = {
                "table_name": table.table_name,
                "columns": []
            }
            for col in table.columns:
                table_dict["columns"].append({
                    "table_name": col.table_name,
                    "column_name": col.column_name,
                    "data_type": col.data_type,
                    "sample_values": col.sample_values
                })
            candidate_tables_data.append(table_dict)
        
        result_state = await discover_data(
            user_query=user_query,
            query_tables=None,
            query_columns=query_columns_data,
            candidate_tables=candidate_tables_data
        )
        
        # 显示结果
        print(f"\\n📊 发现结果:")
        print(f"策略: {result_state.strategy}")
        print(f"匹配数量: {len(result_state.column_matches)}")
        
        for i, match in enumerate(result_state.column_matches[:3]):
            print(f"  {i+1}. {match.target_column} (置信度: {match.confidence:.3f})")
            print(f"     原因: {match.reason}")
        
        # 与真实匹配对比
        actual_matches = []
        for gt in ground_truth_data:
            if (gt["query_table"] == test_query["query_table"] and 
                gt["query_column"] == test_query["query_column"]):
                actual_matches.append(f"{gt['candidate_table'].replace('.csv', '')}.{gt['candidate_column']}")
        
        print(f"\\n🎯 真实匹配 ({len(actual_matches)} 个):")
        for match in actual_matches[:3]:
            print(f"  - {match}")
        
        # 计算准确率
        found_matches = [match.target_column for match in result_state.column_matches]
        correct_matches = set(found_matches) & set(actual_matches)
        
        if found_matches:
            precision = len(correct_matches) / len(found_matches)
            print(f"\\n📈 准确率: {precision:.3f} ({len(correct_matches)}/{len(found_matches)})")
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()


async def main():
    """主函数"""
    await test_join_scenario()


if __name__ == "__main__":
    asyncio.run(main())