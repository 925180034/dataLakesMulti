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
# 方法1: 使用启动脚本（推荐）
python run_cli.py config          # 查看配置
python run_cli.py --help          # 查看帮助
python run_cli.py discover -q "find similar tables" -t examples/sample_tables.json

# 方法2: 使用shell脚本
./datalakes config
./datalakes --help
./datalakes discover -q "find similar tables" -t examples/sample_tables.json

# 方法3: 设置环境变量
export PYTHONPATH=.
python -m src.cli config
```

### 第四步：运行示例

**1. 查找相似表结构（用于Join操作）**:
```bash
# 使用真实数据集（推荐）
python run_cli.py discover -q "find tables with similar column structures for joining" \
  -t examples/real_test_tables.json -f markdown

# 或使用基础示例
python run_cli.py discover -q "find tables with similar column structures for joining" \
  -t examples/sample_tables.json -f markdown
```

**2. 查找语义相关表（用于Union操作）**:
```bash
# 使用真实数据集（推荐）
python run_cli.py discover -q "find semantically related tables for union operations" \
  -t examples/real_test_tables.json -f json

# 或使用基础示例
python run_cli.py discover -q "find semantically related tables for union operations" \
  -t examples/sample_tables.json -f json
```

**3. 查找匹配的列**:
```bash
# 使用真实数据集（推荐）
python run_cli.py discover -q "find columns that can be joined together" \
  -c examples/real_test_columns.json -f table

# 或使用基础示例
python run_cli.py discover -q "find columns that can be joined together" \
  -c examples/sample_columns.json -f table
```

**4. 启动API服务**:
```bash
python run_cli.py serve
# 访问 http://localhost:8000/docs 查看API文档
```

**5. 索引数据（可选，用于更快搜索）**:
```bash
# 索引真实数据集（推荐）
python run_cli.py index-tables examples/real_test_tables.json
python run_cli.py index-columns examples/real_test_columns.json

# 或索引基础示例数据
python run_cli.py index-tables examples/sample_tables.json
python run_cli.py index-columns examples/sample_columns.json
```

### 第五步：输出格式选择

系统支持多种输出格式：

- **markdown** (`-f markdown`): 结构化的Markdown格式，适合文档
- **json** (`-f json`): JSON格式，适合程序处理
- **table** (`-f table`): 表格格式，适合终端查看

### 常见使用场景

**场景1：数据库表Join分析**
```bash
# 找到可以进行Join操作的表（使用真实数据集）
python run_cli.py discover -q "which tables can be joined based on common columns" \
  -t examples/real_test_tables.json -f markdown
```

**场景2：数据整合分析**
```bash
# 找到相似的数据结构用于合并（使用真实数据集）
python run_cli.py discover -q "find tables with similar schemas for data integration" \
  -t examples/real_test_tables.json -f json
```

**场景3：列级别匹配**
```bash
# 分析列之间的匹配关系（使用真实数据集）
python run_cli.py discover -q "find matching columns across different tables" \
  -c examples/real_test_columns.json -f table
```

**场景4：复杂业务查询**
```bash
# 客户数据分析（使用真实数据集）
python run_cli.py discover -q "find all tables containing customer or user data for analytics" \
  -t examples/real_test_tables.json -f markdown

# 销售数据发现（使用真实数据集）
python run_cli.py discover -q "identify tables with sales, revenue or transaction information" \
  -t examples/real_test_tables.json -f json
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