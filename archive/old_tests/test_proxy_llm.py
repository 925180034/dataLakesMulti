#!/usr/bin/env python
"""
测试代理版LLM客户端
"""

import asyncio
import time
from src.utils.llm_client_proxy import GeminiClientWithProxy

async def test_proxy_client():
    """测试代理版客户端"""
    print("Testing Gemini client with proxy support...")
    
    config = {
        "model_name": "gemini-1.5-flash",
        "temperature": 0.1,
        "max_tokens": 100
    }
    
    try:
        client = GeminiClientWithProxy(config)
        
        # 测试简单查询
        prompt = """
        Determine if table1 and table2 can be joined.
        Return JSON: {"is_match": true/false, "confidence": 0.0-1.0}
        """
        
        print("Sending request through proxy...")
        start = time.time()
        result = await client.generate(prompt)
        elapsed = time.time() - start
        
        print(f"✅ Success in {elapsed:.2f}s")
        print(f"Response: {result[:200]}")
        
        # 测试并行调用
        print("\nTesting 3 parallel calls...")
        tasks = [
            client.generate(f"Is table{i} valid? Return JSON: {{\"valid\": true/false}}")
            for i in range(3)
        ]
        
        start = time.time()
        results = await asyncio.gather(*tasks, return_exceptions=True)
        elapsed = time.time() - start
        
        success = sum(1 for r in results if not isinstance(r, Exception))
        print(f"✅ {success}/3 succeeded in {elapsed:.2f}s")
        
        for i, r in enumerate(results):
            if isinstance(r, Exception):
                print(f"  Call {i} failed: {r}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_proxy_client())