# 项目状态
更新时间: 2025-08-14 22:34:58

## ✅ 保留的核心文件

### 主要脚本
- `run_cli.py` - CLI主入口
- `run_cached_experiments.py` - 实验运行脚本
- `run_langgraph_system.py` - LangGraph系统
- `evaluate_with_metrics.py` - 评估模块
- `create_complete_quality_dataset.py` - 数据集创建工具
- `validate_new_datasets.py` - 数据集验证工具
- `precompute_embeddings.py` - 嵌入预计算

### 核心代码
- `src/` - 源代码目录
- `tests/` - 测试目录

### 数据集
- `examples/separated_datasets/` - 高质量数据集
  - `join/` - JOIN任务数据
  - `union/` - UNION任务数据

### 文档
- `README.md` - 项目说明
- `CLAUDE.md` - Claude指导文档
- `docs/` - 详细文档目录

### 配置
- `config.yml` - 主配置文件
- `.env` - 环境变量
- `requirements.txt` - 依赖包

## ❌ 已清理

- Python缓存文件
- 冗余测试脚本
- 重复文档
- 旧实验结果（保留最新3个）
- 临时文件
- 备份目录
