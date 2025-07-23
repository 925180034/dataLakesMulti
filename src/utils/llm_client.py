from typing import Dict, Any, Optional
from abc import ABC, abstractmethod
import json
import logging
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain.schema import HumanMessage, SystemMessage
from src.config.settings import settings

logger = logging.getLogger(__name__)


class LLMClient(ABC):
    """LLM客户端抽象基类"""
    
    @abstractmethod
    async def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """生成文本"""
        pass
    
    @abstractmethod
    async def generate_json(self, prompt: str, system_prompt: Optional[str] = None) -> Dict[str, Any]:
        """生成JSON格式响应"""
        pass


class OpenAIClient(LLMClient):
    """OpenAI客户端"""
    
    def __init__(self, config: Dict[str, Any]):
        self.client = ChatOpenAI(
            model=config.get("model_name", "gpt-3.5-turbo"),
            openai_api_key=config.get("api_key"),
            temperature=config.get("temperature", 0.1),
            max_tokens=config.get("max_tokens", 2000),
            timeout=config.get("timeout", 30)
        )
    
    async def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """生成文本响应"""
        try:
            messages = []
            if system_prompt:
                messages.append(SystemMessage(content=system_prompt))
            messages.append(HumanMessage(content=prompt))
            
            response = await self.client.ainvoke(messages)
            return response.content
            
        except Exception as e:
            logger.error(f"OpenAI生成失败: {e}")
            raise
    
    async def generate_json(self, prompt: str, system_prompt: Optional[str] = None) -> Dict[str, Any]:
        """生成JSON格式响应"""
        json_prompt = f"{prompt}\n\n请确保响应是有效的JSON格式。"
        response = await self.generate(json_prompt, system_prompt)
        
        try:
            # 尝试解析JSON
            return json.loads(response)
        except json.JSONDecodeError:
            # 如果解析失败，尝试提取JSON部分
            import re
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
            else:
                raise ValueError(f"无法解析JSON响应: {response}")


class AnthropicClient(LLMClient):
    """Anthropic客户端"""
    
    def __init__(self, config: Dict[str, Any]):
        self.client = ChatAnthropic(
            model=config.get("model_name", "claude-3-sonnet-20240229"),
            anthropic_api_key=config.get("api_key"),
            temperature=config.get("temperature", 0.1),
            max_tokens=config.get("max_tokens", 2000),
            timeout=config.get("timeout", 30)
        )
    
    async def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """生成文本响应"""
        try:
            messages = []
            if system_prompt:
                messages.append(SystemMessage(content=system_prompt))
            messages.append(HumanMessage(content=prompt))
            
            response = await self.client.ainvoke(messages)
            return response.content
            
        except Exception as e:
            logger.error(f"Anthropic生成失败: {e}")
            raise
    
    async def generate_json(self, prompt: str, system_prompt: Optional[str] = None) -> Dict[str, Any]:
        """生成JSON格式响应"""
        json_prompt = f"{prompt}\n\n请确保响应是有效的JSON格式。"
        response = await self.generate(json_prompt, system_prompt)
        
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            import re
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
            else:
                raise ValueError(f"无法解析JSON响应: {response}")


def create_llm_client() -> LLMClient:
    """创建LLM客户端"""
    config = {
        "model_name": settings.llm.model_name,
        "api_key": settings.llm.api_key,
        "temperature": settings.llm.temperature,
        "max_tokens": settings.llm.max_tokens,
        "timeout": settings.llm.timeout
    }
    
    if settings.llm.provider == "openai":
        return OpenAIClient(config)
    elif settings.llm.provider == "anthropic":
        return AnthropicClient(config)
    else:
        raise ValueError(f"不支持的LLM提供商: {settings.llm.provider}")


# 全局LLM客户端实例
llm_client = create_llm_client()