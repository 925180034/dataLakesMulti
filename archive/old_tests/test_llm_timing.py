#\!/usr/bin/env python
"""
测试LLM调用时间 - 验证并行执行
"""

import asyncio
import time
import os
from src.utils.llm_client_proxy import GeminiClientWithProxy

async def test_serial_vs_parallel():
    """比较串行和并行LLM调用"""
    
    config = {
        "model_name": "gemini-1.5-flash",
        "temperature": 0.1,
        "max_tokens": 100
    }
    
    client = GeminiClientWithProxy(config)
    
    # 准备5个测试查询
    queries = [
        f"Is table{i} joinable with table{i+1}? Return JSON: {{\"is_match\": true/false}}"
        for i in range(5)
    ]
    
    print("=" * 60)
    print("LLM调用性能测试")
    print("=" * 60)
    
    # 1. 串行调用
    print("\n1. 串行调用 (5个查询)...")
    start = time.time()
    serial_results = []
    for query in queries:
        try:
            result = await client.generate(query)
            serial_results.append(result)
        except Exception as e:
            serial_results.append(f"Error: {e}")
    serial_time = time.time() - start
    print(f"   串行耗时: {serial_time:.2f}s")
    print(f"   平均每个: {serial_time/5:.2f}s")
    
    # 2. 并行调用
    print("\n2. 并行调用 (5个查询)...")
    start = time.time()
    tasks = [client.generate(query) for query in queries]
    parallel_results = await asyncio.gather(*tasks, return_exceptions=True)
    parallel_time = time.time() - start
    print(f"   并行耗时: {parallel_time:.2f}s")
    print(f"   平均每个: {parallel_time/5:.2f}s")
    
    # 3. 计算提升
    speedup = serial_time / parallel_time if parallel_time > 0 else 0
    print(f"\n📊 性能提升: {speedup:.1f}x")
    print(f"   节省时间: {serial_time - parallel_time:.2f}s")
    
    # 4. 批量测试（20个候选，分批处理）
    print("\n3. 批量处理测试 (20个候选，批大小=5)...")
    candidates = [f"table{i}" for i in range(20)]
    start = time.time()
    
    batch_size = 5
    all_results = []
    for i in range(0, len(candidates), batch_size):
        batch = candidates[i:i+batch_size]
        batch_queries = [
            f"Is {table} valid? Return JSON: {{\"valid\": true/false}}"
            for table in batch
        ]
        batch_tasks = [client.generate(q) for q in batch_queries]
        batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
        all_results.extend(batch_results)
    
    batch_time = time.time() - start
    print(f"   批量处理耗时: {batch_time:.2f}s")
    print(f"   处理速度: {20/batch_time:.2f} queries/s")
    
    # 5. 与超时保护
    print("\n4. 超时保护测试...")
    timeout_query = "Complex query that might timeout..."
    
    try:
        start = time.time()
        result = await asyncio.wait_for(
            client.generate(timeout_query),
            timeout=5.0
        )
        print(f"   ✅ 成功响应: {(time.time()-start):.2f}s")
    except asyncio.TimeoutError:
        print(f"   ⏱️ 超时保护生效 (5秒)")
    except Exception as e:
        print(f"   ❌ 其他错误: {e}")

if __name__ == "__main__":
    print("测试代理设置...")
    print(f"HTTP_PROXY: {os.getenv('http_proxy', 'Not set')}")
    print(f"HTTPS_PROXY: {os.getenv('https_proxy', 'Not set')}")
    
    asyncio.run(test_serial_vs_parallel())
EOF < /dev/null
