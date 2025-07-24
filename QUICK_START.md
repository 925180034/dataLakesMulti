# 快速开始指南

## 🚀 立即运行

由于网络依赖问题已修复，你现在可以立即使用系统！

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
# 测试数据发现（可能有小错误，但能看到Gemini工作）
python run_cli.py discover -q "find tables with user columns" -t examples/sample_tables.json -f json
```

## 📝 使用说明

### 基本工作流
1. **查看配置**: 确认Gemini API配置正确
2. **准备数据**: 使用examples/目录下的示例文件
3. **执行发现**: 使用discover命令进行数据发现
4. **查看结果**: 支持json、markdown、table三种输出格式

### 示例命令
```bash
# 查找相似的表
python run_cli.py discover -q "find similar tables for data joining" \
  -t examples/sample_tables.json -f markdown

# 查找匹配的列
python run_cli.py discover -q "find columns that can be joined" \
  -c examples/sample_columns.json -f json

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