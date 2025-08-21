# Config module

# 自动启用离线模式，防止连接HuggingFace Hub
from .offline_mode import setup_offline_mode

# 确保离线模式在导入时生效
setup_offline_mode()