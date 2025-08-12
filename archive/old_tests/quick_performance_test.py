#\!/usr/bin/env python
"""
快速性能测试 - 分析为什么LLM调用慢
"""

import asyncio
import time
import os
from src.utils.llm_client_proxy import GeminiClientWithProxy

async def test_llm_performance():
    """测试LLM性能"""
    
    print("=" * 60)
    print("LLM性能测试")
    print("=" * 60)
    
    config = {
        "model_name": "gemini-1.5-flash",
        "temperature": 0.1,
        "max_tokens": 100
    }
    
    client = GeminiClientWithProxy(config)
    
    # 1. 测试单个调用
    print("\n1. 单个LLM调用:")
    start = time.time()
    result = await client.generate("Return JSON: {\"test\": true}")
    single_time = time.time() - start
    print(f"   耗时: {single_time:.2f}秒")
    
    # 2. 测试10个并行调用
    print("\n2. 10个并行调用:")
    queries = [f"Is table{i} valid? Return JSON" for i in range(10)]
    start = time.time()
    tasks = [client.generate(q) for q in queries]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    parallel_10_time = time.time() - start
    success = sum(1 for r in results if not isinstance(r, Exception))
    print(f"   耗时: {parallel_10_time:.2f}秒")
    print(f"   成功: {success}/10")
    print(f"   平均: {parallel_10_time/10:.2f}秒/调用")
    
    # 3. 测试20个并行调用
    print("\n3. 20个并行调用:")
    queries = [f"Is table{i} valid? Return JSON" for i in range(20)]
    start = time.time()
    tasks = [client.generate(q) for q in queries]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    parallel_20_time = time.time() - start
    success = sum(1 for r in results if not isinstance(r, Exception))
    print(f"   耗时: {parallel_20_time:.2f}秒")
    print(f"   成功: {success}/20")
    print(f"   平均: {parallel_20_time/20:.2f}秒/调用")
    
    # 4. 分析结果
    print("\n📊 分析:")
    if parallel_10_time < single_time * 2:
        print("   ✅ 并行有效！API支持高并发")
    else:
        print("   ⚠️ 可能有API限流或网络瓶颈")
    
    speedup_10 = (single_time * 10) / parallel_10_time
    speedup_20 = (single_time * 20) / parallel_20_time
    
    print(f"   10并发加速比: {speedup_10:.1f}x")
    print(f"   20并发加速比: {speedup_20:.1f}x")
    
    # 5. 建议
    print("\n💡 优化建议:")
    if parallel_20_time > 30:
        print("   1. 减少候选数量是关键（20→10或更少）")
        print("   2. 使用更短的prompt")
        print("   3. 考虑使用更快的模型(如gemini-1.5-flash)")
        print("   4. 实现结果缓存")
    else:
        print("   性能良好，无需进一步优化")

if __name__ == "__main__":
    asyncio.run(test_llm_performance())
