import time
import logging
from typing import Dict, List, Optional
from dataclasses import dataclass
from threading import Lock
import json
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class ApiKeyStatus:
    """API密钥状态"""
    key: str
    is_active: bool = True
    last_error_time: float = 0
    error_count: int = 0
    cooldown_until: float = 0
    total_requests: int = 0
    
    def is_available(self) -> bool:
        """检查密钥是否可用"""
        current_time = time.time()
        # 如果在冷却期内，不可用
        if current_time < self.cooldown_until:
            return False
        return self.is_active
    
    def mark_error(self, error_message: str = ""):
        """标记密钥出错"""
        current_time = time.time()
        self.last_error_time = current_time
        self.error_count += 1
        
        # 检查是否是配额错误
        if "quota" in error_message.lower() or "exceeded" in error_message.lower():
            # 配额错误，设置较长的冷却时间（1小时）
            self.cooldown_until = current_time + 3600
            logger.warning(f"API密钥配额用完，冷却1小时: {self.key[:10]}...")
        elif "429" in error_message:
            # 速率限制，设置短期冷却（10分钟）
            self.cooldown_until = current_time + 600
            logger.warning(f"API密钥达到速率限制，冷却10分钟: {self.key[:10]}...")
        else:
            # 其他错误，设置短期冷却（5分钟）
            self.cooldown_until = current_time + 300
            logger.warning(f"API密钥出错，冷却5分钟: {self.key[:10]}...")
    
    def mark_success(self):
        """标记密钥成功使用"""
        self.total_requests += 1
        # 成功使用后重置错误计数
        if self.error_count > 0:
            self.error_count = max(0, self.error_count - 1)


class ApiKeyManager:
    """API密钥轮换管理器"""
    
    def __init__(self, api_keys: List[str], provider: str = "gemini"):
        self.provider = provider
        self.api_keys: Dict[str, ApiKeyStatus] = {}
        self.current_key_index = 0
        self.lock = Lock()
        
        # 初始化API密钥状态
        for key in api_keys:
            if key and key.strip():
                self.api_keys[key.strip()] = ApiKeyStatus(key=key.strip())
        
        if not self.api_keys:
            raise ValueError("至少需要提供一个有效的API密钥")
        
        logger.info(f"初始化API密钥管理器，共 {len(self.api_keys)} 个密钥")
    
    def get_current_key(self) -> Optional[str]:
        """获取当前可用的API密钥"""
        with self.lock:
            # 首先尝试获取当前索引的密钥
            keys_list = list(self.api_keys.keys())
            
            # 从当前索引开始查找可用密钥
            for i in range(len(keys_list)):
                index = (self.current_key_index + i) % len(keys_list)
                key = keys_list[index]
                key_status = self.api_keys[key]
                
                if key_status.is_available():
                    self.current_key_index = index
                    return key
            
            # 如果没有可用密钥，返回None
            logger.error("所有API密钥都不可用")
            return None
    
    def mark_key_error(self, key: str, error_message: str = ""):
        """标记密钥出错并自动切换到下一个"""
        with self.lock:
            if key in self.api_keys:
                self.api_keys[key].mark_error(error_message)
                logger.info(f"标记密钥出错: {key[:10]}..., 错误: {error_message[:100]}")
                
                # 自动切换到下一个密钥
                self._switch_to_next_key()
    
    def mark_key_success(self, key: str):
        """标记密钥成功使用"""
        with self.lock:
            if key in self.api_keys:
                self.api_keys[key].mark_success()
    
    def _switch_to_next_key(self):
        """切换到下一个可用密钥"""
        keys_list = list(self.api_keys.keys())
        original_index = self.current_key_index
        
        # 查找下一个可用密钥
        for i in range(1, len(keys_list)):
            next_index = (self.current_key_index + i) % len(keys_list)
            next_key = keys_list[next_index]
            
            if self.api_keys[next_key].is_available():
                self.current_key_index = next_index
                logger.info(f"切换到下一个API密钥: {next_key[:10]}...")
                return
        
        logger.warning("没有找到可用的API密钥")
    
    def get_status_report(self) -> Dict:
        """获取所有密钥的状态报告"""
        with self.lock:
            current_time = time.time()
            report = {
                "provider": self.provider,
                "total_keys": len(self.api_keys),
                "active_keys": 0,
                "keys_status": []
            }
            
            for key, status in self.api_keys.items():
                key_info = {
                    "key_prefix": key[:10] + "...",
                    "is_available": status.is_available(),
                    "error_count": status.error_count,
                    "total_requests": status.total_requests,
                    "cooldown_remaining": max(0, status.cooldown_until - current_time)
                }
                
                if status.is_available():
                    report["active_keys"] += 1
                
                report["keys_status"].append(key_info)
            
            return report
    
    def save_status(self, file_path: str):
        """保存密钥状态到文件"""
        try:
            status_data = {}
            current_time = time.time()
            
            for key, status in self.api_keys.items():
                status_data[key] = {
                    "error_count": status.error_count,
                    "total_requests": status.total_requests,
                    "cooldown_remaining": max(0, status.cooldown_until - current_time),
                    "last_error_time": status.last_error_time
                }
            
            with open(file_path, 'w') as f:
                json.dump(status_data, f, indent=2)
                
        except Exception as e:
            logger.error(f"保存API密钥状态失败: {e}")
    
    def load_status(self, file_path: str):
        """从文件加载密钥状态"""
        try:
            if not Path(file_path).exists():
                return
                
            with open(file_path, 'r') as f:
                status_data = json.load(f)
            
            current_time = time.time()
            
            for key, data in status_data.items():
                if key in self.api_keys:
                    status = self.api_keys[key]
                    status.error_count = data.get("error_count", 0)
                    status.total_requests = data.get("total_requests", 0)
                    status.last_error_time = data.get("last_error_time", 0)
                    
                    # 恢复冷却时间
                    cooldown_remaining = data.get("cooldown_remaining", 0)
                    if cooldown_remaining > 0:
                        status.cooldown_until = current_time + cooldown_remaining
                        
            logger.info(f"已加载API密钥状态: {file_path}")
            
        except Exception as e:
            logger.error(f"加载API密钥状态失败: {e}")


# 全局API密钥管理器实例
_api_key_manager: Optional[ApiKeyManager] = None


def get_api_key_manager() -> Optional[ApiKeyManager]:
    """获取全局API密钥管理器"""
    return _api_key_manager


def initialize_api_key_manager(api_keys: List[str], provider: str = "gemini") -> ApiKeyManager:
    """初始化全局API密钥管理器"""
    global _api_key_manager
    _api_key_manager = ApiKeyManager(api_keys, provider)
    return _api_key_manager