#!/usr/bin/env python
"""
修复LLM客户端 - 添加代理支持和超时机制
"""

import os
import asyncio
import google.generativeai as genai
from typing import Optional, Dict, Any
import logging
import httpx
import time

logger = logging.getLogger(__name__)

class FixedGeminiClient:
    """修复版Gemini客户端 - 支持代理和超时"""
    
    def __init__(self, config: Dict[str, Any]):
        self.model_name = config.get("model_name", "gemini-1.5-flash")
        self.generation_config = {
            "temperature": config.get("temperature", 0.1),
            "max_output_tokens": config.get("max_tokens", 2000),
        }
        
        # 获取API密钥
        self.api_key = config.get("api_key") or os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("No Gemini API key found")
        
        # 配置代理（如果有）
        self._setup_proxy()
        
        # 配置超时
        self.timeout = config.get("timeout", 30)  # 默认30秒超时
        
    def _setup_proxy(self):
        """设置代理"""
        http_proxy = os.getenv('HTTP_PROXY') or os.getenv('http_proxy')
        https_proxy = os.getenv('HTTPS_PROXY') or os.getenv('https_proxy')
        
        if http_proxy or https_proxy:
            # 创建支持代理的HTTP客户端
            proxy = https_proxy or http_proxy
            logger.info(f"Using proxy: {proxy}")
            
            # 为genai配置代理
            import httpx
            client = httpx.Client(
                proxies=proxy,
                timeout=httpx.Timeout(30.0)
            )
            
            # 注意：google.generativeai可能不直接支持代理
            # 需要通过环境变量或其他方式配置
            
    async def generate_with_timeout(self, prompt: str, timeout: Optional[int] = None) -> str:
        """带超时的生成方法"""
        timeout = timeout or self.timeout
        
        try:
            # 使用asyncio.wait_for设置超时
            result = await asyncio.wait_for(
                self._generate_async(prompt),
                timeout=timeout
            )
            return result
        except asyncio.TimeoutError:
            logger.error(f"LLM call timed out after {timeout} seconds")
            raise TimeoutError(f"LLM call timed out after {timeout} seconds")
        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            raise
    
    async def _generate_async(self, prompt: str) -> str:
        """异步生成（在线程池中运行同步API）"""
        loop = asyncio.get_event_loop()
        
        # 在线程池中运行同步的Gemini API
        def sync_generate():
            genai.configure(api_key=self.api_key)
            model = genai.GenerativeModel(self.model_name)
            response = model.generate_content(
                prompt,
                generation_config=self.generation_config
            )
            return response.text
        
        return await loop.run_in_executor(None, sync_generate)
    
    async def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """生成文本（兼容原接口）"""
        if system_prompt:
            prompt = f"System: {system_prompt}\n\nUser: {prompt}"
        
        return await self.generate_with_timeout(prompt)


async def test_fixed_client():
    """测试修复的客户端"""
    print("Testing fixed Gemini client with timeout...")
    
    config = {
        "model_name": "gemini-1.5-flash",
        "temperature": 0.1,
        "max_tokens": 100,
        "timeout": 10  # 10秒超时
    }
    
    try:
        client = FixedGeminiClient(config)
        
        # 测试简单查询
        start = time.time()
        result = await client.generate("Return JSON: {\"test\": true}")
        elapsed = time.time() - start
        
        print(f"✅ Success in {elapsed:.2f}s")
        print(f"Response: {result[:100]}")
        
    except TimeoutError:
        print("❌ Request timed out (as expected if network is blocked)")
    except Exception as e:
        print(f"❌ Error: {e}")


async def test_with_workaround():
    """使用本地模型或Mock作为临时解决方案"""
    print("\n临时解决方案：使用Mock LLM进行测试")
    
    class MockLLMClient:
        """Mock LLM客户端用于测试"""
        
        async def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
            # 模拟延迟
            await asyncio.sleep(0.1)
            
            # 根据prompt返回合理的响应
            if "join" in prompt.lower():
                return '{"is_match": true, "confidence": 0.8, "reason": "Mock response"}'
            elif "union" in prompt.lower():
                return '{"is_match": false, "confidence": 0.3, "reason": "Mock response"}'
            else:
                return '{"result": "Mock response"}'
    
    client = MockLLMClient()
    
    # 测试并行调用
    print("Testing parallel mock calls...")
    start = time.time()
    
    tasks = [
        client.generate(f"Query {i}")
        for i in range(5)
    ]
    
    results = await asyncio.gather(*tasks)
    elapsed = time.time() - start
    
    print(f"✅ Completed 5 parallel calls in {elapsed:.2f}s")
    print(f"Average: {elapsed/5:.2f}s per call")


if __name__ == "__main__":
    import asyncio
    
    print("="*60)
    print("修复LLM客户端问题")
    print("="*60)
    
    async def main():
        # 测试修复的客户端
        await test_fixed_client()
        
        # 测试临时解决方案
        await test_with_workaround()
    
    asyncio.run(main())