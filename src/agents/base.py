from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import logging
from src.core.models import AgentState
from src.utils.llm_client import llm_client

logger = logging.getLogger(__name__)


class BaseAgent(ABC):
    """智能体基类"""
    
    def __init__(self, name: str):
        self.name = name
        self.llm_client = llm_client
    
    @abstractmethod
    async def process(self, state: AgentState) -> AgentState:
        """处理状态并返回更新后的状态"""
        pass
    
    async def call_llm(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """调用LLM生成文本"""
        try:
            response = await self.llm_client.generate(prompt, system_prompt)
            return response
        except Exception as e:
            logger.error(f"{self.name} LLM调用失败: {e}")
            raise
    
    async def call_llm_json(self, prompt: str, system_prompt: Optional[str] = None) -> Dict[str, Any]:
        """调用LLM生成JSON响应"""
        try:
            response = await self.llm_client.generate_json(prompt, system_prompt)
            return response
        except Exception as e:
            logger.error(f"{self.name} LLM JSON调用失败: {e}")
            raise
    
    def log_progress(self, state: AgentState, message: str):
        """记录处理进度"""
        log_msg = f"[{self.name}] {message}"
        state.add_log(log_msg)
        logger.info(log_msg)
    
    def log_error(self, state: AgentState, error: str):
        """记录错误信息"""
        error_msg = f"[{self.name}] Error: {error}"
        state.add_error(error_msg)
        logger.error(error_msg)