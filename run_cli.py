#!/usr/bin/env python3
"""
CLI启动脚本 - 解决模块导入问题
"""
import sys
import os
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.absolute()
sys.path.insert(0, str(project_root))

# 设置工作目录
os.chdir(project_root)

if __name__ == '__main__':
    # 导入并运行CLI
    from src.cli import cli
    cli()