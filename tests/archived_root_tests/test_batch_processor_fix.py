#!/usr/bin/env python
"""
测试修复后的批量处理器
验证批量API调用是否正常工作
"""

import asyncio
import json
import logging
from src.tools.batch_llm_processor import BatchLLMProcessor, TableMatchingPromptBuilder
from src.utils.llm_client import create_llm_client

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_batch_processor():
    """测试批量处理器"""
    print("\n" + "="*60)
    print("🧪 测试批量LLM处理器")
    print("="*60)
    
    # 创建LLM客户端
    llm_client = create_llm_client()
    
    # 创建批量处理器
    batch_processor = BatchLLMProcessor(
        llm_client=llm_client,
        max_batch_size=3,
        max_concurrent=2
    )
    
    # 准备测试数据
    test_items = [
        {
            "query_table": {
                "name": "users",
                "columns": [
                    {"name": "user_id", "type": "int"},
                    {"name": "username", "type": "string"},
                    {"name": "email", "type": "string"}
                ]
            },
            "candidate_table": {
                "name": "orders",
                "columns": [
                    {"name": "order_id", "type": "int"},
                    {"name": "user_id", "type": "int"},
                    {"name": "total", "type": "float"}
                ]
            }
        },
        {
            "query_table": {
                "name": "users",
                "columns": [
                    {"name": "user_id", "type": "int"},
                    {"name": "username", "type": "string"},
                    {"name": "email", "type": "string"}
                ]
            },
            "candidate_table": {
                "name": "products",
                "columns": [
                    {"name": "product_id", "type": "int"},
                    {"name": "name", "type": "string"},
                    {"name": "price", "type": "float"}
                ]
            }
        },
        {
            "query_table": {
                "name": "users",
                "columns": [
                    {"name": "user_id", "type": "int"},
                    {"name": "username", "type": "string"},
                    {"name": "email", "type": "string"}
                ]
            },
            "candidate_table": {
                "name": "user_profiles",
                "columns": [
                    {"name": "user_id", "type": "int"},
                    {"name": "bio", "type": "string"},
                    {"name": "avatar", "type": "string"}
                ]
            }
        }
    ]
    
    print(f"\n📊 测试数据: {len(test_items)} 个表对")
    for i, item in enumerate(test_items, 1):
        print(f"  {i}. {item['query_table']['name']} <-> {item['candidate_table']['name']}")
    
    print("\n🔄 执行批量处理...")
    
    try:
        # 调用批量处理器
        results = await batch_processor.batch_process(
            items=test_items,
            prompt_builder=TableMatchingPromptBuilder.build_batch_prompt,
            response_parser=TableMatchingPromptBuilder.parse_batch_response,
            use_cache=False  # 测试时不使用缓存
        )
        
        print("\n✅ 批量处理成功!")
        print(f"返回 {len(results)} 个结果\n")
        
        # 显示结果
        for i, (item, result) in enumerate(zip(test_items, results), 1):
            print(f"结果 {i}: {item['candidate_table']['name']}")
            print(f"  - 匹配: {result.get('match', 'N/A')}")
            print(f"  - 置信度: {result.get('score', 'N/A')}")
            print(f"  - 原因: {result.get('reason', 'N/A')}")
            print(f"  - 方法: {result.get('method', 'N/A')}")
            print()
        
        # 显示统计信息
        stats = batch_processor.get_statistics()
        print("-"*40)
        print("📊 性能统计:")
        print(f"  - 总调用数: {stats['total_calls']}")
        print(f"  - 总耗时: {stats['total_time']:.2f}秒")
        print(f"  - 平均每调用: {stats['avg_time_per_call']:.2f}秒")
        print(f"  - 批次数: {stats['batch_count']}")
        print(f"  - 缓存命中率: {stats['cache_hit_rate']:.1%}")
        
        return True
        
    except Exception as e:
        print(f"\n❌ 批量处理失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = asyncio.run(test_batch_processor())
    exit(0 if success else 1)