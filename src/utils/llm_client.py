from typing import Dict, Any, Optional
from abc import ABC, abstractmethod
import json
import logging
import os  # 添加os模块导入
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain.schema import HumanMessage, SystemMessage
import google.generativeai as genai
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
        # 构建ChatOpenAI参数
        openai_params = {
            "model": config.get("model_name", "gpt-3.5-turbo"),
            "openai_api_key": config.get("api_key"),
            "temperature": config.get("temperature", 0.1),
            "max_tokens": config.get("max_tokens", 2000),
            "timeout": config.get("timeout", int(os.getenv('LLM_TIMEOUT', '30')))
        }
        
        # 如果有自定义base_url，添加到参数中
        if config.get("base_url"):
            openai_params["base_url"] = config.get("base_url")
        
        self.client = ChatOpenAI(**openai_params)
    
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
            timeout=config.get("timeout", int(os.getenv('LLM_TIMEOUT', '30')))
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


class GeminiClient(LLMClient):
    """Google Gemini客户端 - 支持多API密钥轮换"""
    
    def __init__(self, config: Dict[str, Any]):
        # 模型配置
        self.model_name = config.get("model_name", "gemini-1.5-flash")
        
        # 配置参数
        self.generation_config = {
            "temperature": config.get("temperature", 0.1),
            "max_output_tokens": config.get("max_tokens", 2000),
        }
        
        # 初始化API密钥管理器
        self._initialize_api_keys(config)
    
    def _initialize_api_keys(self, config: Dict[str, Any]):
        """初始化API密钥管理"""
        from src.utils.api_key_manager import get_api_key_manager, initialize_api_key_manager
        
        # 获取API密钥列表
        api_keys = []
        
        # 支持多种配置方式
        if config.get("api_keys"):
            # 如果配置中直接提供了密钥列表
            api_keys = config["api_keys"]
        elif config.get("api_key"):
            # 如果只有单个密钥，也添加到列表中
            api_keys = [config["api_key"]]
        
        # 从环境变量获取额外的密钥
        import os
        for i in range(1, 10):  # 支持最多10个密钥
            key_name = f"GEMINI_API_KEY_{i}" if i > 1 else "GEMINI_API_KEY"
            key_value = os.getenv(key_name)
            if key_value and key_value not in api_keys:
                api_keys.append(key_value)
        
        if not api_keys:
            raise ValueError("未找到有效的Gemini API密钥")
        
        # 初始化或获取现有的API密钥管理器
        try:
            self.api_key_manager = get_api_key_manager()
            if self.api_key_manager is None:
                self.api_key_manager = initialize_api_key_manager(api_keys, "gemini")
        except Exception as e:
            logger.warning(f"API密钥管理器初始化警告: {e}")
            # 如果管理器初始化失败，使用第一个密钥作为后备
            self.fallback_key = api_keys[0]
            self.api_key_manager = None
        
        logger.info(f"Gemini客户端初始化完成，共 {len(api_keys)} 个API密钥")
    
    def _get_current_api_key(self) -> str:
        """获取当前可用的API密钥"""
        if self.api_key_manager:
            key = self.api_key_manager.get_current_key()
            if key:
                return key
        
        # 如果管理器不可用，使用后备密钥
        if hasattr(self, 'fallback_key'):
            return self.fallback_key
        
        raise RuntimeError("没有可用的API密钥")
    
    async def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """生成文本响应 - 支持自动重试和密钥切换"""
        # 从环境变量读取配置（支持动态调整）
        max_retries = int(os.getenv('LLM_MAX_RETRIES', '3'))
        
        for attempt in range(max_retries):
            try:
                # 获取当前API密钥
                api_key = self._get_current_api_key()
                
                # 配置当前密钥
                genai.configure(api_key=api_key)
                model = genai.GenerativeModel(self.model_name)
                
                # 构建完整的提示
                full_prompt = prompt
                if system_prompt:
                    full_prompt = f"System: {system_prompt}\n\nUser: {prompt}"
                
                # 调用Gemini API
                response = model.generate_content(
                    full_prompt,
                    generation_config=self.generation_config
                )
                
                # 成功时标记密钥可用
                if self.api_key_manager:
                    self.api_key_manager.mark_key_success(api_key)
                
                return response.text
                
            except Exception as e:
                error_message = str(e)
                logger.warning(f"Gemini生成失败 (尝试 {attempt + 1}/{max_retries}): {error_message}")
                
                # 获取当前使用的密钥
                current_key = self._get_current_api_key() if attempt == 0 else api_key
                
                # 判断是否是需要切换密钥的错误
                if self._should_switch_key(error_message):
                    if self.api_key_manager:
                        self.api_key_manager.mark_key_error(current_key, error_message)
                        logger.info(f"由于错误切换API密钥: {error_message[:100]}")
    
                        # 如果还有重试机会，继续下一次尝试
                        if attempt < max_retries - 1:
                            continue
                
                # 如果是最后一次尝试或不需要切换密钥，抛出异常
                if attempt == max_retries - 1:
                    logger.error(f"Gemini生成最终失败: {error_message}")
                    raise
    
    def _should_switch_key(self, error_message: str) -> bool:
        """判断是否应该切换API密钥"""
        switch_keywords = [
            "quota", "exceeded", "429", "rate limit", 
            "too many requests", "billing", "limit"
        ]
        error_lower = error_message.lower()
        return any(keyword in error_lower for keyword in switch_keywords)
    
    async def generate_json(self, prompt: str, system_prompt: Optional[str] = None) -> Dict[str, Any]:
        """生成JSON格式响应 - 支持自动重试和密钥切换"""
        json_prompt = f"{prompt}\n\n请确保响应是有效的JSON格式，不要包含任何注释或额外文本。"
        response = await self.generate(json_prompt, system_prompt)
        
        try:
            return json.loads(response)
        except json.JSONDecodeError as e:
            import re
            # 尝试修复常见的JSON错误
            fixed_response = response
            
            # 1. 移除可能的markdown代码块标记
            fixed_response = re.sub(r'```json\s*', '', fixed_response)
            fixed_response = re.sub(r'```\s*$', '', fixed_response)
            
            # 2. 修复缺失的逗号（在 } 或 ] 后面跟着 " 或 { 或 [ 的情况）
            fixed_response = re.sub(r'([}\]])\s*\n\s*(["{[])', r'\1,\n\2', fixed_response)
            
            # 3. 移除行尾的逗号（在 } 或 ] 前面）
            fixed_response = re.sub(r',\s*([}\]])', r'\1', fixed_response)
            
            # 4. 尝试处理被截断的JSON
            # 如果响应以不完整的方式结束，尝试补全
            if not fixed_response.rstrip().endswith('}'):
                # 计算需要补全的括号
                open_braces = fixed_response.count('{')
                close_braces = fixed_response.count('}')
                open_brackets = fixed_response.count('[')
                close_brackets = fixed_response.count(']')
                
                # 移除最后一个不完整的元素
                # 找到最后一个完整的逗号或开括号位置
                for i in range(len(fixed_response) - 1, -1, -1):
                    if fixed_response[i] in [',', '{', '[']:
                        fixed_response = fixed_response[:i+1]
                        if fixed_response[i] == ',':
                            fixed_response = fixed_response[:i]
                        break
                
                # 补全缺失的括号
                while close_brackets < open_brackets:
                    fixed_response += ']'
                    close_brackets += 1
                    
                while close_braces < open_braces:
                    fixed_response += '}'
                    close_braces += 1
            
            # 5. 尝试提取JSON对象
            json_match = re.search(r'\{.*\}', fixed_response, re.DOTALL)
            if json_match:
                try:
                    return json.loads(json_match.group())
                except json.JSONDecodeError:
                    # 如果还是失败，尝试提取能解析的部分
                    try:
                        # 找到第一个有效的JSON对象
                        text = json_match.group()
                        # 逐步截断尝试
                        for end_pos in range(len(text), 0, -1):
                            try:
                                partial = text[:end_pos]
                                # 补全括号
                                open_b = partial.count('{')
                                close_b = partial.count('}')
                                partial += '}' * (open_b - close_b)
                                result = json.loads(partial)
                                logger.warning(f"成功解析部分JSON响应")
                                return result
                            except:
                                continue
                    except:
                        pass
            
            # 6. 如果还是失败，尝试返回一个默认的空结果
            logger.warning(f"无法解析JSON响应，返回默认值。原始响应: {response[:500]}")
            return {"error": "JSON解析失败", "raw_response": response[:500]}
    
    def get_api_status(self) -> Dict[str, Any]:
        """获取API密钥状态"""
        if self.api_key_manager:
            return self.api_key_manager.get_status_report()
        else:
            return {"provider": "gemini", "status": "fallback_mode", "total_keys": 1}


def create_llm_client() -> LLMClient:
    """创建LLM客户端"""
    config = {
        "model_name": settings.llm.model_name,
        "api_key": settings.llm.api_key,
        "base_url": getattr(settings.llm, 'base_url', None),
        "temperature": settings.llm.temperature,
        "max_tokens": settings.llm.max_tokens,
        "timeout": settings.llm.timeout
    }
    
    if settings.llm.provider == "openai":
        return OpenAIClient(config)
    elif settings.llm.provider == "anthropic":
        return AnthropicClient(config)
    elif settings.llm.provider == "gemini":
        return GeminiClient(config)
    else:
        raise ValueError(f"不支持的LLM提供商: {settings.llm.provider}")


# 全局LLM客户端实例
llm_client = create_llm_client()