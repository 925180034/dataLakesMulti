"""
Gemini客户端代理支持补丁
"""

import os
import json
import logging
import aiohttp
import asyncio
from typing import Dict, Any, Optional
from src.config.settings import settings

logger = logging.getLogger(__name__)


class GeminiClientWithProxy:
    """支持代理的Gemini客户端"""
    
    def __init__(self, config: Dict[str, Any]):
        self.model_name = config.get("model_name", "gemini-1.5-flash") 
        self.api_key = config.get("api_key") or os.getenv("GEMINI_API_KEY")
        
        if not self.api_key:
            raise ValueError("No Gemini API key found")
            
        # 配置代理
        self.proxies = {
            'http': os.getenv('http_proxy', 'http://127.0.0.1:7890'),
            'https': os.getenv('https_proxy', 'http://127.0.0.1:7890')
        }
        
        self.generation_config = {
            "temperature": config.get("temperature", 0.1),
            "maxOutputTokens": config.get("max_tokens", 2000),
        }
        
        # API端点
        self.api_url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model_name}:generateContent"
        
        logger.info(f"Initialized Gemini client with proxy: {self.proxies['https']}")
    
    async def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """通过REST API调用Gemini（支持代理） - 真正的异步版本"""
        
        # 记录API调用开始
        logger.info("🚀 [GEMINI API CALL START] =====================================")
        logger.info(f"   Model: {self.model_name}")
        logger.info(f"   Prompt length: {len(prompt)} chars")
        if system_prompt:
            logger.info(f"   System prompt length: {len(system_prompt)} chars")
        
        if system_prompt:
            prompt = f"{system_prompt}\n\n{prompt}"
        
        # 构建请求URL（包含API key作为参数）
        url_with_key = f"{self.api_url}?key={self.api_key}"
        
        # 构建请求体
        data = {
            "contents": [
                {
                    "parts": [
                        {"text": prompt}
                    ]
                }
            ],
            "generationConfig": self.generation_config
        }
        
        import time
        start_time = time.time()
        
        try:
            # 使用aiohttp进行真正的异步请求
            async with aiohttp.ClientSession() as session:
                logger.info(f"   Sending request to Gemini API...")
                async with session.post(
                    url_with_key,
                    json=data,
                    proxy=self.proxies.get('https'),  # aiohttp使用单个代理URL
                    timeout=aiohttp.ClientTimeout(total=30)
                ) as response:
                    
                    elapsed_time = time.time() - start_time
                    
                    if response.status == 200:
                        result = await response.json()
                        # 提取生成的文本
                        if "candidates" in result and result["candidates"]:
                            content = result["candidates"][0]["content"]
                            if "parts" in content and content["parts"]:
                                response_text = content["parts"][0]["text"]
                                logger.info(f"✅ [GEMINI API SUCCESS] Response received in {elapsed_time:.2f}s")
                                logger.info(f"   Response length: {len(response_text)} chars")
                                logger.info("===========================================================")
                                return response_text
                        
                        logger.error(f"❌ [GEMINI API ERROR] Unexpected response format: {result}")
                        logger.info("===========================================================")
                        return ""
                    else:
                        text = await response.text()
                        logger.error(f"❌ [GEMINI API ERROR] Request failed: {response.status} - {text}")
                        logger.info("===========================================================")
                        raise Exception(f"API request failed: {response.status}")
                        
        except asyncio.TimeoutError:
            elapsed_time = time.time() - start_time
            logger.error(f"⏱️ [GEMINI API TIMEOUT] Request timed out after {elapsed_time:.2f}s")
            logger.info("===========================================================")
            raise TimeoutError("Request timed out")
        except Exception as e:
            elapsed_time = time.time() - start_time
            logger.error(f"❌ [GEMINI API ERROR] Request failed after {elapsed_time:.2f}s: {e}")
            logger.info("===========================================================")
            raise
    
    async def generate_json(self, prompt: str, system_prompt: Optional[str] = None) -> Dict[str, Any]:
        """生成JSON响应"""
        logger.info("📊 [GEMINI JSON REQUEST] Starting JSON generation...")
        json_prompt = f"{prompt}\n\nPlease ensure the response is valid JSON format."
        response = await self.generate(json_prompt, system_prompt)
        
        try:
            # 尝试解析JSON
            if '{' in response and '}' in response:
                json_str = response[response.find('{'):response.rfind('}')+1]
                result = json.loads(json_str)
                logger.info(f"✅ [GEMINI JSON SUCCESS] Successfully parsed JSON with {len(result)} keys")
                return result
            logger.warning("⚠️ [GEMINI JSON WARNING] No JSON found in response, returning empty dict")
            return {}
        except json.JSONDecodeError as e:
            logger.error(f"❌ [GEMINI JSON ERROR] Failed to parse JSON: {e}")
            return {}


def get_llm_client():
    """Get LLM client instance"""
    # Create default config
    config = {
        'api_key': os.getenv('GEMINI_API_KEY'),
        'model': 'gemini-1.5-flash',
        'proxy': os.getenv('HTTP_PROXY', 'http://127.0.0.1:7890')
    }
    return GeminiClientWithProxy(config)