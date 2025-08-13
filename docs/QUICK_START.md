# 快速开始指南

## 📋 运行步骤

### 第一步：环境准备

1. **检查Python版本**（需要Python 3.10+）:
```bash
python --version
```

2. **创建并激活虚拟环境**:
```bash
# 方法1: 使用conda（推荐）
conda create -n data_lakes_multi python=3.10 -y
conda activate data_lakes_multi

# 方法2: 使用venv
python -m venv data_lakes_multi
source data_lakes_multi/bin/activate  # Linux/Mac
# 或 data_lakes_multi\Scripts\activate  # Windows
```

3. **安装依赖**:
```bash
pip install -r requirements.txt
```

### 第二步：API密钥配置

1. **复制环境配置文件**:
```bash
cp .env.example .env
```

2. **编辑.env文件，添加API密钥**（至少选择一个）:
```bash
# 推荐使用Gemini API（免费且稳定）
GEMINI_API_KEY=your_gemini_api_key_here

# 或使用OpenAI API
OPENAI_API_KEY=your_openai_api_key_here

# 或使用Anthropic API
ANTHROPIC_API_KEY=your_anthropic_api_key_here
```

**获取API密钥的方法**:
- **Gemini API**: 访问 https://ai.google.dev/ 注册并获取免费API密钥
- **OpenAI API**: 访问 https://platform.openai.com/ 
- **Anthropic API**: 访问 https://console.anthropic.com/

### 第三步：验证安装

**测试配置**:
```bash
python run_cli.py config
```

**测试API连接**:
```bash
python -c "
import asyncio
import sys
sys.path.append('.')
from src.utils.llm_client import llm_client

async def test():
    response = await llm_client.generate('Hello!', 'You are helpful.')
    print(f'✅ API连接成功: {response}')

asyncio.run(test())
"
```

## 🚀 开始使用

系统现在可以正常运行了！

### 基本命令

```bash
# 使用LangGraph多智能体系统（推荐）
python run_langgraph_system.py --help          # 查看帮助
python run_langgraph_system.py --dataset subset --max-queries 5 --task join

# 查看配置（如需要）
python -c "import yaml; print(yaml.safe_load(open('config.yml')))"
```

### 第四步：运行示例

**1. 查找相似表结构（用于JOIN操作）**:
```bash
# 使用subset数据集（100表，推荐入门）
python run_langgraph_system.py --dataset subset --max-queries 5 --task join

# 或使用complete数据集（1534表）
python run_langgraph_system.py --dataset complete --max-queries 3 --task join
```

**2. 查找语义相关表（用于UNION操作）**:
```bash
# UNION任务测试
python run_langgraph_system.py --dataset subset --max-queries 5 --task union

# 同时测试JOIN和UNION
python run_langgraph_system.py --dataset subset --max-queries 5 --task both
```

**3. 保存结果到文件**:
```bash
# 保存详细结果
python run_langgraph_system.py --dataset subset --max-queries 10 --task join --output results.json

# 查看保存的结果
python -c "import json; print(json.dumps(json.load(open('results.json')), indent=2))"
```

**4. 系统测试与验证**:
```bash
# 快速系统测试
python test_langgraph.py

# 详细性能测试（如果存在）
python -m pytest tests/ -v
```

### 第五步：理解输出结果

系统输出包含以下信息：
- **匹配表列表**: 按相关性排序的候选表
- **匹配分数**: 每个表的匹配置信度 (0-1)
- **匹配类型**: JOIN 或 UNION
- **详细证据**: 元数据、向量、LLM三层的分数详情

### 常见使用场景

**场景1：数据库表JOIN分析**
```bash
# 寻找可JOIN的表（基于列匹配）
python run_langgraph_system.py --dataset subset --max-queries 3 --task join
```

**场景2：数据集UNION分析** 
```bash
# 寻找可UNION的表（基于表语义）
python run_langgraph_system.py --dataset subset --max-queries 3 --task union
```

**场景3：综合数据发现**
```bash
# 同时进行JOIN和UNION分析
python run_langgraph_system.py --dataset subset --max-queries 5 --task both --output comprehensive_results.json
```

## ✅ 已解决的问题

### 1. 网络依赖问题
- **问题**: SentenceTransformer在导入时尝试下载模型
- **解决**: 实现了延迟加载和离线模式
- **效果**: 系统现在可以在没有网络连接时正常启动

### 2. 模块导入问题
- **问题**: Python无法找到src模块
- **解决**: 提供了多种启动方式
- **推荐**: 使用`python run_cli.py`

## 🎯 当前功能状态

### ✅ 完全可用
- **配置系统**: `python run_cli.py config`
- **Gemini API**: 文本生成和JSON输出正常
- **CLI界面**: 所有命令行功能
- **嵌入向量**: 离线模式虚拟向量生成
- **基础测试**: 所有pytest测试通过

### ⚠️ 部分可用
- **数据发现**: 基础功能可用，但工作流中有小bug
- **向量搜索**: 使用虚拟向量，精度较低
- **API服务**: 可以启动，但未全面测试

### 🔧 需要改进
- **网络环境下的完整功能**: 需要能访问HuggingFace下载模型
- **工作流错误处理**: 一些边缘情况需要优化

## 🧪 测试你的安装

### 快速测试
```bash
# 测试配置
python run_cli.py config

# 测试Gemini API
python -c "
import asyncio
import sys
sys.path.append('.')
from src.utils.llm_client import llm_client

async def test():
    response = await llm_client.generate('Hello!', 'You are helpful.')
    print(f'✅ Gemini API: {response}')

asyncio.run(test())
"

# 测试嵌入向量
python -c "
import asyncio
import sys
sys.path.append('.')
from src.tools.embedding import get_embedding_generator

async def test():
    emb_gen = get_embedding_generator()
    embedding = await emb_gen.generate_text_embedding('test')
    print(f'✅ 嵌入向量: 维度={len(embedding)}')

asyncio.run(test())
"
```

### 数据发现测试
```bash
# 测试数据发现（推荐使用真实数据集）
python run_cli.py discover -q "find tables with user columns" -t examples/real_test_tables.json -f json

# 或使用基础示例数据（可能有小错误，但能看到Gemini工作）
python run_cli.py discover -q "find tables with user columns" -t examples/sample_tables.json -f json
```

## 📝 使用说明

### 数据文件说明

系统提供了两套测试数据：

**基础示例数据**（适合快速测试）:
- `examples/sample_tables.json` - 简单的表结构示例
- `examples/sample_columns.json` - 基础列数据示例

**真实数据集样本**（从实际数据湖采样，更接近真实使用场景）:
- `examples/real_test_tables.json` - 从真实数据集采样的表结构
- `examples/real_test_columns.json` - 从真实数据集采样的列数据

### 基本工作流
1. **查看配置**: 确认Gemini API配置正确
2. **准备数据**: 使用examples/目录下的示例文件（推荐使用real_test_数据获得更好效果）
3. **执行发现**: 使用discover命令进行数据发现
4. **查看结果**: 支持json、markdown、table三种输出格式

### 示例命令

**使用基础示例数据**（快速测试）:
```bash
# 查找相似的表
python run_cli.py discover -q "find similar tables for data joining" \
  -t examples/sample_tables.json -f markdown

# 查找匹配的列
python run_cli.py discover -q "find columns that can be joined" \
  -c examples/sample_columns.json -f json
```

**使用真实数据集样本**（推荐，更接近实际使用场景）:
```bash
# 查找相似的表结构（真实数据）
python run_cli.py discover -q "find similar tables for data joining" \
  -t examples/real_test_tables.json -f markdown

# 查找匹配的列（真实数据）
python run_cli.py discover -q "find columns that can be joined" \
  -c examples/real_test_columns.json -f json

# 复杂查询示例（真实数据）
python run_cli.py discover -q "find tables containing user information for customer analytics" \
  -t examples/real_test_tables.json -f table
```

**API服务**:
```bash
# 启动API服务
python run_cli.py serve
```

## 🎉 总结

系统现在**可以正常运行**！主要功能：
- ✅ Gemini API完全工作
- ✅ CLI命令正常
- ✅ 离线模式支持
- ✅ 基础数据处理

虽然在完整的数据发现工作流中还有一些小bug，但所有核心组件都能正常工作。你可以开始使用和测试系统了！