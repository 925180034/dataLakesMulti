"""
系统级离线模式配置
自动设置环境变量，确保所有模型加载都使用本地缓存
"""
import os
import logging

logger = logging.getLogger(__name__)

def setup_offline_mode():
    """
    配置离线模式环境变量
    防止连接HuggingFace Hub
    """
    # 设置离线模式
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    
    # 设置缓存路径
    os.environ['HF_HOME'] = '/root/.cache/huggingface'
    os.environ['HUGGINGFACE_HUB_CACHE'] = '/root/.cache/huggingface/hub'
    # TRANSFORMERS_CACHE已弃用，使用HF_HOME代替
    
    # 禁用telemetry
    os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'
    
    # 禁用进度条（可选）
    os.environ['HF_HUB_DISABLE_PROGRESS_BARS'] = '1'
    
    # 设置超时为0，避免网络请求
    os.environ['HF_HUB_TIMEOUT'] = '0'
    
    logger.debug("离线模式已启用 - 不会连接HuggingFace Hub")

# 在模块导入时自动执行
setup_offline_mode()