# 🚀 数据湖多智能体系统

基于大语言模型（LLM）的智能数据湖模式匹配与数据发现系统，通过多智能体协作实现高精度的表格匹配。

## 📋 系统概述

本系统专注于数据湖环境中的两个核心场景：
- **Join场景**：寻找具有相似列结构的表，支持数据连接操作
- **Union场景**：基于数据内容发现语义相关的表，支持数据合并操作

### 🎯 核心特性

- **智能路由**：自动识别用户意图，选择最优处理策略
- **多智能体协作**：5个专业智能体协同工作，各司其职
- **高性能搜索**：HNSW向量索引，支持毫秒级相似度搜索
- **完整评价体系**：Precision、Recall、F1-Score自动计算
- **灵活配置**：支持多种LLM API（Gemini、OpenAI、Anthropic）

## 🚀 快速开始

### 环境要求
- Python 3.10+
- 8GB+ 内存
- 5GB+ 存储空间

### 安装步骤

1. **克隆项目**
```bash
git clone <repository_url>
cd dataLakesMulti
```

2. **创建环境**
```bash
conda create -n datalakes python=3.10 -y
conda activate datalakes
pip install -r requirements.txt
```

3. **配置API密钥**
```bash
cp .env.example .env
# 编辑 .env 文件，添加您的API密钥
```

4. **运行测试**
```bash
# 快速测试（3个查询）
python unified_experiment.py 3 subset 30

# 标准评估（50个查询）
python unified_experiment.py 50 subset 30
```

## 📊 实验与评估

### 统一实验脚本
系统提供统一的实验脚本，自动计算所有评价指标：

```bash
python unified_experiment.py [查询数] [数据集] [超时秒数]

# 示例：
python unified_experiment.py 50 subset 30    # 子集测试
python unified_experiment.py 100 complete 60  # 完整测试
```

### 评价指标
- **Precision（精确率）**：返回结果的准确性
- **Recall（召回率）**：找到所有正确答案的能力  
- **F1-Score**：精确率和召回率的综合指标
- **成功率**：查询成功完成的比例
- **响应时间**：平均查询处理时间

所有结果自动保存在 `experiment_results/` 目录。

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
│   ├── QUICK_START.md     # 快速开始
│   └── ...               # 更多文档
├── examples/              # 示例数据
├── tests/                 # 测试用例
├── experiment_results/    # 实验结果
└── unified_experiment.py  # 统一实验脚本
```

## 📖 文档

完整文档请访问 [docs/](docs/) 目录：
- [系统架构与实施计划](docs/SYSTEM_ARCHITECTURE_AND_PLAN.md)
- [快速开始指南](docs/QUICK_START.md)
- [项目设计文档](docs/Project-Design-Document.md)
- [更多文档...](docs/README.md)

## 🔧 配置优化

使用优化配置获得更好性能：
```bash
cp config_optimized.yml config.yml
```

主要优化项：
- LLM超时：30秒 → 10秒
- 最大Token：1000 → 500
- 请求并发：20 → 1
- 缓存策略：启用多级缓存

## 📈 性能指标

当前系统性能（基于100表子集测试）：
- **查询成功率**：> 95%
- **平均响应时间**：15-20秒
- **平均精确率**：~85%
- **平均召回率**：~78%
- **平均F1分数**：~81%

## 🤝 贡献

欢迎贡献代码和文档！请查看 [贡献指南](docs/README.md#-贡献指南)。

## 📄 许可证

本项目采用 MIT 许可证。

## 📞 联系方式

如有问题，请查看：
- [故障排除指南](docs/QUICK_START.md#故障排除)
- [常见问题解答](docs/SYSTEM_ARCHITECTURE_AND_PLAN.md#-故障排除)

---

**项目版本**: v2.0  
**更新日期**: 2024年7月30日