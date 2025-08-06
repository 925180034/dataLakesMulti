# 🚀 数据湖多智能体系统

基于大语言模型（LLM）的智能数据湖模式匹配与数据发现系统，通过多智能体协作实现高精度的表格匹配。

## 📋 系统概述

本系统是一个专为大规模数据湖设计的智能表格匹配系统，采用先进的三层加速架构和多智能体协作机制，能够在海量表格中快速准确地发现数据关联。

### 🎯 核心场景

- **Join场景**：寻找具有相似列结构的表，支持数据连接操作
  - 示例：找到所有包含 user_id, email 的表进行关联分析
- **Union场景**：基于数据内容发现语义相关的表，支持数据合并操作
  - 示例：找到所有销售相关的表进行统一分析

### 🏗️ 系统架构

#### 三层加速架构
1. **元数据层**：快速过滤候选表（毫秒级）
   - 基于列名、数据类型的初步筛选
   - 减少 90% 以上的候选表

2. **向量搜索层**：语义相似度匹配（百毫秒级）
   - HNSW 高性能向量索引
   - 支持批量并行搜索

3. **LLM验证层**：精确匹配验证（秒级）
   - 智能批处理减少 API 调用
   - 并行处理提升效率

#### 多智能体系统
- **PlannerAgent**：理解用户意图，制定搜索策略
- **ColumnDiscoveryAgent**：发现相似列（Bottom-Up策略）
- **TableAggregationAgent**：聚合列匹配结果为表评分
- **TableDiscoveryAgent**：发现相似表（Top-Down策略）
- **TableMatchingAgent**：精确表对表匹配验证

### ✨ 技术亮点

- **智能路由**：自动识别查询类型，选择最优处理路径
- **并行处理**：支持 10 并发请求，大幅提升吞吐量
- **多级缓存**：L1内存 + L2 Redis + L3磁盘，缓存命中率 >95%
- **性能优化**：批处理 + 并行化，实现 857x 加速
- **灵活扩展**：支持 Gemini、OpenAI、Anthropic 等多种 LLM

## 🚀 快速开始

### 环境要求
- Python 3.10+
- 8GB+ 内存（推荐 16GB）
- 5GB+ 存储空间
- 网络连接（用于 LLM API 调用）

### 安装步骤

1. **克隆项目**
```bash
git clone https://github.com/925180034/dataLakesMulti.git
cd dataLakesMulti
```

2. **创建环境**
```bash
# 使用 conda（推荐）
conda create -n datalakes python=3.10 -y
conda activate datalakes

# 或使用 venv
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate  # Windows

# 安装依赖
pip install -r requirements.txt
```

3. **配置API密钥**
```bash
cp .env.example .env
# 编辑 .env 文件，添加您的API密钥（至少选择一个）：
# - GEMINI_API_KEY（推荐，免费额度充足）
# - OPENAI_API_KEY
# - ANTHROPIC_API_KEY
```

4. **验证安装**
```bash
# 检查配置
python run_cli.py config

# 运行简单测试
python run_full_experiment_fixed.py --dataset subset --max-queries 3
```

## 🧪 实验与评估

### 1. 完整实验脚本 (`run_full_experiment_fixed.py`)

最新、功能最完整的实验脚本，支持多种配置：

```bash
# 基础用法
python run_full_experiment_fixed.py [选项]

# 可用选项：
--dataset [subset|complete]   # 数据集类型
  - subset: 100表，502查询（快速测试）
  - complete: 1,534表，7,358查询（完整评估）
--max-queries [数量]          # 限制查询数量
--no-llm                      # 禁用LLM验证（仅向量搜索）

# 实验示例：
# 1. 快速验证（5分钟）
python run_full_experiment_fixed.py --dataset subset --max-queries 3

# 2. 标准测试（10分钟）
python run_full_experiment_fixed.py --dataset subset --max-queries 50

# 3. 深度评估（30分钟）
python run_full_experiment_fixed.py --dataset subset --max-queries 200

# 4. 完整评估（2小时）
python run_full_experiment_fixed.py --dataset subset

# 5. 性能测试（无LLM）
python run_full_experiment_fixed.py --dataset subset --max-queries 50 --no-llm
```

### 2. 快速评估脚本 (`ultra_fast_evaluation_fixed.py`)

用于快速性能验证：

```bash
# 用法：python ultra_fast_evaluation_fixed.py [查询数] [数据集]
python ultra_fast_evaluation_fixed.py 3 subset      # 3个查询
python ultra_fast_evaluation_fixed.py 50 subset     # 50个查询
python ultra_fast_evaluation_fixed.py 502 subset    # 全部查询
```

### 3. CLI交互式工具 (`run_cli.py`)

用于单次查询和交互式探索：

```bash
# 发现查询
python run_cli.py discover -q "find joinable tables" -t examples/final_subset_tables.json -f json

# 特定表查询
python run_cli.py discover -q "find tables similar to csvData6444295__5" -t examples/final_subset_tables.json -f markdown

# 使用优化配置
python run_cli.py discover -q "your query" -t examples/final_subset_tables.json --config config_optimized.yml

# 查看系统配置
python run_cli.py config
```

### 📊 评价指标说明

- **Precision（精确率）**：返回的表中正确的比例
  - 公式：正确返回数 / 总返回数
- **Recall（召回率）**：找到所有正确表的比例
  - 公式：正确返回数 / 应返回数
- **F1-Score**：精确率和召回率的调和平均
  - 公式：2 × (Precision × Recall) / (Precision + Recall)
- **查询延迟**：单个查询的处理时间
- **吞吐量**：每秒处理的查询数

所有实验结果自动保存在 `experiment_results/` 目录，包含：
- 详细的预测结果
- 性能指标统计
- 错误分析报告
- 系统配置快照

## 📁 项目结构

```
dataLakesMulti/
├── src/                    # 源代码
│   ├── agents/            # 智能体实现
│   ├── core/              # 核心工作流
│   ├── tools/             # 搜索工具
│   └── config/            # 配置管理
├── docs/                  # 📚 完整文档中心
│   ├── README.md          # 文档索引
│   ├── SYSTEM_ARCHITECTURE_AND_PLAN.md  # 系统架构
│   ├── SYSTEM_ARCHITECTURE_DIAGRAMS.md  # 架构图表
│   ├── QUICK_START.md     # 快速开始
│   └── ...               # 更多文档
├── examples/              # 示例数据
├── tests/                 # 测试用例（已整理）
├── experiment_results/    # 实验结果
├── run_full_experiment_fixed.py  # 完整实验脚本
├── ultra_fast_evaluation_fixed.py  # 快速评估脚本
└── run_cli.py            # CLI接口
```

## 📖 文档

完整文档请访问 [docs/](docs/) 目录：
- [系统架构与实施计划](docs/SYSTEM_ARCHITECTURE_AND_PLAN.md)
- [快速开始指南](docs/QUICK_START.md)
- [项目设计文档](docs/Project-Design-Document.md)
- [更多文档...](docs/README.md)

## ⚙️ 配置与优化

### 配置文件说明

系统提供两种配置：
- `config.yml`：默认配置，平衡性能和准确性
- `config_optimized.yml`：优化配置，追求极致性能

```bash
# 使用优化配置
cp config_optimized.yml config.yml
```

### 关键配置项

```yaml
# LLM 配置
llm:
  provider: "gemini"         # 可选：gemini, openai, anthropic
  model: "gemini-2.0-flash-exp"
  temperature: 0.1           # 降低随机性，提高一致性
  max_tokens: 500           # 限制输出长度
  timeout: 10               # API 超时时间

# 性能优化
performance:
  batch_size: 20            # 批处理大小
  max_concurrent_requests: 10  # 并发请求数
  enable_cache: true        # 启用缓存
  cache_ttl: 3600          # 缓存有效期（秒）

# 向量搜索
vector_search:
  model: "sentence-transformers/all-MiniLM-L6-v2"
  ef_search: 200           # HNSW 搜索参数
  top_k: 20               # 返回候选数
```

### 性能调优建议

1. **低延迟优先**：
   - 减少 `top_k` 到 10
   - 降低 `batch_size` 到 10
   - 使用更快的 LLM（如 gemini-flash）

2. **高准确率优先**：
   - 增加 `top_k` 到 50
   - 使用更强的 LLM（如 GPT-4）
   - 启用 `enable_llm_matching: true`

3. **成本优化**：
   - 启用所有缓存选项
   - 使用免费的 Gemini API
   - 设置合理的 `max_queries` 限制

## 📈 性能指标

### 最新测试结果（2025年8月）

#### 系统性能
- **查询成功率**：100%
- **平均响应时间**：
  - 带 LLM：1.08秒 ✅
  - 仅向量：0.07秒（缓存命中）
- **吞吐量**：~10 QPS（并发处理）
- **缓存命中率**：>95%

#### 匹配准确率
- **当前性能**：
  - Precision：30%
  - Recall：75%
  - F1-Score：43%

- **优化前对比**：
  - 查询时间：60秒 → 1.08秒（55x提升）
  - LLM调用：20次 → 1-2次（10x减少）
  - 成功率：85% → 100%

### 🎯 性能目标与进展

| 指标 | 目标 | 当前 | 状态 |
|------|------|------|------|
| 响应时间 | 3-8秒 | 1.08秒 | ✅ 达成 |
| 精确率 | >90% | 30% | 🔄 优化中 |
| 召回率 | >90% | 75% | 🔄 优化中 |
| 系统规模 | 10,000表 | 1,534表 | 📋 待扩展 |
| 并发支持 | 10 QPS | 10 QPS | ✅ 达成 |

### 📊 性能优化历程

1. **第一阶段**：基础实现
   - 简单向量搜索 + 串行 LLM 调用
   - 性能：60秒/查询

2. **第二阶段**：三层架构
   - 元数据过滤 + HNSW 索引 + 批量 LLM
   - 性能：5-10秒/查询

3. **第三阶段**：极致优化
   - 并行处理 + 多级缓存 + 智能批处理
   - 性能：1.08秒/查询

4. **下一阶段**：准确率提升
   - 改进向量模型
   - 优化匹配算法
   - 增强 LLM 提示

## 🤝 贡献

欢迎贡献代码和文档！请查看 [贡献指南](docs/README.md#-贡献指南)。

## 📄 许可证

本项目采用 MIT 许可证。

## 📞 联系方式

如有问题，请查看：
- [故障排除指南](docs/QUICK_START.md#故障排除)
- [常见问题解答](docs/SYSTEM_ARCHITECTURE_AND_PLAN.md#-故障排除)

## 🛠️ 高级用法

### 批量处理

```python
# 批量查询处理示例
from src.core.ultra_optimized_workflow import UltraOptimizedWorkflow

workflow = UltraOptimizedWorkflow()
queries = ["find sales tables", "find user tables", "find product tables"]
results = workflow.batch_process(queries)
```

### API 服务

```bash
# 启动 API 服务
python -m src.cli serve

# API 端点
# POST /api/v1/discover
# GET /api/v1/health
# GET /docs (Swagger UI)
```

### 自定义配置

```python
# 程序化配置
from src.config.settings import Settings

settings = Settings(
    llm_provider="openai",
    llm_model="gpt-4",
    vector_db_type="chromadb",
    enable_cache=True
)
```

## 🔍 故障排除

### 常见问题

1. **LLM API 连接失败**
   - 检查 API 密钥是否正确
   - 确认网络连接正常
   - 查看 `.env` 文件配置

2. **内存不足**
   - 减少 `batch_size` 配置
   - 使用 `--max-queries` 限制查询数
   - 清理缓存：`rm -rf cache/`

3. **查询超时**
   - 增加 `timeout` 配置
   - 使用更快的 LLM 模型
   - 启用缓存加速

4. **准确率低**
   - 确保启用 LLM 验证
   - 调整 `similarity_threshold`
   - 使用更强的向量模型

### 日志调试

```bash
# 启用详细日志
export LOG_LEVEL=DEBUG
python run_full_experiment_fixed.py --dataset subset --max-queries 3

# 查看日志文件
tail -f logs/datalakes.log
```

## 🤝 贡献指南

我们欢迎各种形式的贡献：

1. **报告问题**：在 [Issues](https://github.com/925180034/dataLakesMulti/issues) 提交
2. **改进文档**：完善使用说明和示例
3. **优化代码**：提升性能和准确率
4. **添加功能**：扩展新的匹配算法

### 开发流程

```bash
# 1. Fork 项目
# 2. 创建特性分支
git checkout -b feature/your-feature

# 3. 提交更改
git commit -m "feat: add new feature"

# 4. 推送分支
git push origin feature/your-feature

# 5. 创建 Pull Request
```

## 📚 相关资源

- [系统架构文档](docs/SYSTEM_ARCHITECTURE_AND_PLAN.md)
- [架构图表](docs/SYSTEM_ARCHITECTURE_DIAGRAMS.md)
- [API 文档](docs/API_REFERENCE.md)
- [性能优化指南](docs/PERFORMANCE_GUIDE.md)

---

**项目版本**: v2.1  
**更新日期**: 2025年8月5日  
**最近更新**: 完善实验指令、更新系统设计说明、添加高级用法