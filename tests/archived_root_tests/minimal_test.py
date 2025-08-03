#!/usr/bin/env python
"""
最小化测试 - 诊断性能瓶颈
"""

import asyncio
import json
import time
from src.core.models import AgentState, TableInfo, TaskStrategy


async def minimal_test():
    """最小化测试"""
    print("🔍 最小化性能诊断")
    print("="*60)
    
    # 1. 测试数据加载
    start = time.time()
    with open('examples/final_subset_tables.json') as f:
        all_tables = json.load(f)
    print(f"✅ 数据加载: {time.time()-start:.2f}秒")
    
    # 2. 测试表信息转换
    start = time.time()
    query_table = TableInfo(**all_tables[0])
    print(f"✅ 表信息转换: {time.time()-start:.2f}秒")
    
    # 3. 测试状态创建
    start = time.time()
    state = AgentState(
        user_query="Find joinable tables",
        query_tables=[query_table],
        query_columns=[],
        strategy=TaskStrategy.TOP_DOWN
    )
    print(f"✅ 状态创建: {time.time()-start:.2f}秒")
    
    # 4. 测试向量搜索（不使用LLM）
    try:
        from src.tools.vector_search import get_vector_search_engine
        start = time.time()
        search = get_vector_search_engine()
        # 只初始化，不执行搜索
        print(f"✅ 向量搜索初始化: {time.time()-start:.2f}秒")
    except Exception as e:
        print(f"❌ 向量搜索初始化失败: {e}")
    
    # 5. 测试最简单的匹配（不使用工作流）
    start = time.time()
    # 简单的表名相似度匹配
    matches = []
    query_name = query_table.table_name.lower()
    for table in all_tables[:10]:  # 只测试前10个
        table_name = table['table_name'].lower()
        # 简单的字符串相似度
        if any(part in table_name for part in query_name.split('_')):
            matches.append(table_name)
    print(f"✅ 简单匹配: {time.time()-start:.2f}秒, 找到 {len(matches)} 个匹配")
    
    # 6. 测试单个LLM调用
    try:
        from src.utils.llm_client import create_llm_client
        llm = create_llm_client()
        
        start = time.time()
        # 使用正确的调用方法
        response = await llm.generate("Say 'test' and nothing else")
        print(f"✅ LLM调用: {time.time()-start:.2f}秒")
    except Exception as e:
        print(f"❌ LLM调用失败: {e}")
    
    print()
    print("📊 诊断结果:")
    print("  - 如果LLM调用 > 5秒: API响应慢是瓶颈")
    print("  - 如果HNSW初始化 > 2秒: 向量索引是瓶颈")
    print("  - 如果都很快: 工作流协调是瓶颈")


if __name__ == "__main__":
    asyncio.run(minimal_test())