# 🌊 数据湖多智能体发现系统
**Data Lake Multi-Agent Discovery System with Three-Layer Acceleration**

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Production%20Ready-success)](https://github.com/)

## 🎯 系统简介

数据湖多智能体发现系统是一个高性能的**数据湖发现（Data Lake Discovery）**解决方案，通过**6个专门智能体协同工作**，结合**三层加速架构**，实现大规模数据湖中的智能数据发现、表关联分析和相似数据集匹配。

### ✨ 核心特性

- **🤝 多智能体协同**: OptimizerAgent、PlannerAgent、AnalyzerAgent等6个专门Agent智能分工
- **⚡ 三层加速架构**: L1元数据筛选→L2向量搜索→L3 LLM验证，性能优化
- **🔍 智能数据发现**: JOIN表发现、UNION表匹配、语义相似度分析
- **📈 高性能优化**: 支持10,000+表规模，查询响应2-8秒
- **🧪 质量保证**: 高质量数据集，充足ground truth，可靠评估指标

## 🏗️ 系统架构

```
用户查询
    ↓
┌─────────────────────────────┐
│   多智能体协同系统          │
├─────────────────────────────┤
│  • OptimizerAgent - 系统优化│
│  • PlannerAgent - 策略规划  │
│  • AnalyzerAgent - 数据分析 │
│  • SearcherAgent - 候选搜索 │
│  • MatcherAgent - 精确匹配  │
│  • AggregatorAgent - 结果聚合│
├─────────────────────────────┤
│   三层加速工具（Agent调用）  │
│  • Layer1: 元数据筛选 <10ms │
│  • Layer2: 向量搜索 10-50ms │
│  • Layer3: LLM验证 1-3s     │
└─────────────────────────────┘
    ↓
匹配结果 + 评估指标
```

## 🚀 快速开始

### 环境要求
- Python 3.10+
- 16GB+ RAM
- CUDA GPU（可选，加速向量计算）

### 安装步骤

1. **克隆项目**
```bash
git clone https://github.com/your-repo/dataLakesMulti.git
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
# 编辑 .env 添加 GEMINI_API_KEY 或其他LLM API密钥
```

4. **运行系统测试**
```bash
# 基础功能测试
python run_cli.py config

# 三层消融实验（推荐）
python three_layer_ablation_optimized.py --task join --dataset subset --max-queries 5

# 高质量数据集评估
python evaluate_with_metrics.py --task both --dataset subset --max-queries 10
```

## 📊 性能指标

| 指标 | JOIN任务 | UNION任务 | 说明 |
|-----|----------|-----------|------|
| **查询延迟** | 2-5秒 | 3-8秒 | 包含LLM验证 |
| **数据质量** | 平均4.8 GT/查询 | 平均11.6 GT/查询 | Ground Truth覆盖 |
| **准确率** | 优化中 | >85% | 基于高质量数据集 |
| **数据规模** | 50-1042查询 | 50-3222查询 | Subset/Complete版本 |
| **系统架构** | 6 Agents | 3层加速 | 智能协同工作 |

## 🗂️ 项目结构（已优化）

```
dataLakesMulti/
├── src/                                # 源代码
│   ├── agents/                         # 6个智能体实现
│   ├── core/                          # 核心工作流
│   ├── tools/                         # 三层加速工具
│   └── config/                        # 配置管理
├── examples/separated_datasets/        # 高质量数据集
│   ├── join_subset/                   # JOIN子集（50查询）
│   ├── union_subset/                  # UNION子集（50查询）
│   ├── join/                          # JOIN完整（1042查询）
│   └── union/                         # UNION完整（3222查询）
├── experiment_results/                # 最新实验结果
├── docs/                              # 完整文档
├── tests/                             # 测试用例
├── three_layer_ablation_optimized.py  # 核心实验脚本
├── extract_high_quality_datasets.py   # 数据集提取
├── evaluate_with_metrics.py          # 评估系统
└── run_cli.py                         # 命令行接口
```

## 📚 核心文档

- **[CLAUDE.md](CLAUDE.md)** - 系统开发指南和架构要求
- **[项目文档](docs/)** - 完整技术文档集合
- **[数据集质量报告](examples/separated_datasets/)** - 高质量数据集说明
- **[实验结果](experiment_results/)** - 最新消融实验结果

## 🔧 核心功能

### 数据集质量优化
- **JOIN数据集**: 1042个查询，4976个GT配对，平均4.8个GT/查询
- **UNION数据集**: 3222个查询，37470个GT配对，平均11.6个GT/查询
- **质量保证**: 只包含有充足ground truth的查询，确保评估有效性

### 三层消融实验
```bash
# 运行完整三层消融实验
python three_layer_ablation_optimized.py --task join --dataset subset --max-queries 10

# 评估指标包括：Hit@1/3/5, Precision, Recall, F1-Score
```

### 多智能体系统
- **OptimizerAgent**: 系统配置优化
- **PlannerAgent**: 查询策略规划  
- **AnalyzerAgent**: 数据结构分析
- **SearcherAgent**: 候选表搜索
- **MatcherAgent**: 精确匹配验证
- **AggregatorAgent**: 结果聚合排序

## 🔄 最近更新

### v2.1 (2025-08-18)
- ✅ **项目清理**: 移除15+冗余文件，保持结构整洁
- ✅ **数据集优化**: 重新提取高质量数据集，确保充足GT
- ✅ **核心功能**: 保留6个核心脚本，专注三层消融系统
- ✅ **文档更新**: 更新README，反映当前项目状态

### 系统状态
- **架构**: 多智能体系统 ✅ 完成
- **数据质量**: 高质量数据集 ✅ 优化完成
- **实验系统**: 三层消融实验 ✅ 正常运行
- **性能优化**: 查询响应时间 🔄 持续优化中

---

**最新版本**: v2.1 | **更新时间**: 2025-08-18 | **状态**: 生产就绪