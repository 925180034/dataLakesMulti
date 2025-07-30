# 🔑 API密钥配置完整指南

## 推荐选择：Google Gemini（免费）

### 1. 注册Gemini API

1. **访问官网**：https://aistudio.google.com/
2. **登录Google账号**（如果没有需要先注册）
3. **点击"Get API Key"**
4. **创建新项目**或选择现有项目
5. **获取API密钥**（格式类似：`AIzaSyXXXXXXXXXXXXXXXXXXXXXXXXXXXX`）

### 2. 配置环境变量

**Linux/Mac系统**：
```bash
# 临时设置（当前会话有效）
export GEMINI_API_KEY="你的API密钥"

# 永久设置（推荐）
echo 'export GEMINI_API_KEY="你的API密钥"' >> ~/.bashrc
source ~/.bashrc

# 或者添加到 ~/.zshrc (如果你使用zsh)
echo 'export GEMINI_API_KEY="你的API密钥"' >> ~/.zshrc
source ~/.zshrc
```

**Windows系统**：
```cmd
# 命令行设置
set GEMINI_API_KEY=你的API密钥

# 永久设置（推荐）
setx GEMINI_API_KEY "你的API密钥"
```

### 3. 使用.env文件（推荐方式）

在项目根目录创建 `.env` 文件：

```bash
# 复制示例文件
cp .env.example .env

# 编辑.env文件
nano .env
```

在 `.env` 文件中添加：
```env
# Google Gemini API (推荐)
GEMINI_API_KEY=你的API密钥

# 其他可选配置
DEBUG=false
CACHE_ENABLED=true
```

## 备选方案

### OpenAI API配置

1. **注册**: https://platform.openai.com/
2. **获取密钥**: API Keys 页面
3. **配置**:
```bash
export OPENAI_API_KEY="sk-proj-xxxxxxxxxxxxxxxx"
# 或在.env文件中：
OPENAI_API_KEY=sk-proj-xxxxxxxxxxxxxxxx
```

### Anthropic Claude API配置

1. **注册**: https://console.anthropic.com/
2. **获取密钥**: API Keys 页面  
3. **配置**:
```bash
export ANTHROPIC_API_KEY="sk-ant-xxxxxxxxxxxxxxxx"
# 或在.env文件中：
ANTHROPIC_API_KEY=sk-ant-xxxxxxxxxxxxxxxx
```

## 验证配置

### 检查环境变量
```bash
# 检查Gemini密钥
echo $GEMINI_API_KEY

# 检查所有相关环境变量
env | grep -E "(GEMINI|OPENAI|ANTHROPIC)_API_KEY"
```

### 测试API连接
```bash
# 使用系统配置检查
python run_cli.py config

# 或运行简单测试
python -c "
import os
if os.getenv('GEMINI_API_KEY'):
    print('✅ GEMINI_API_KEY 已配置')
else:
    print('❌ GEMINI_API_KEY 未配置')
"
```

## 🔒 安全最佳实践

### 1. 保护API密钥
- ❌ **不要**将API密钥提交到Git仓库
- ❌ **不要**在代码中硬编码API密钥
- ✅ **使用**环境变量或.env文件
- ✅ **添加**.env到.gitignore文件

### 2. 权限管理
- 只授予必要的API权限
- 定期轮换API密钥
- 监控API使用情况

### 3. 费用控制
- 设置API使用限额
- 监控每日/每月使用量
- 选择合适的定价计划

## 🚨 常见问题解决

### 问题1: "API密钥无效"
```bash
# 检查密钥格式
echo $GEMINI_API_KEY | wc -c  # Gemini密钥通常39字符

# 检查特殊字符
echo $GEMINI_API_KEY | cat -v  # 查看是否有隐藏字符
```

### 问题2: "环境变量未生效"
```bash
# 重新加载环境变量
source ~/.bashrc  # 或 ~/.zshrc

# 检查当前shell
echo $SHELL

# 重启终端或创新会话
```

### 问题3: ".env文件不生效"
```bash
# 检查文件位置（必须在项目根目录）
ls -la .env

# 检查文件内容
cat .env

# 确保没有多余的空格或引号
```

## 🧪 完整验证流程

配置完成后，运行以下命令验证：

```bash
# 1. 检查系统配置
python run_cli.py config

# 2. 测试API连接
python -c "
from src.config.settings import settings
print(f'LLM Provider: {settings.llm.provider}')
print(f'Model: {settings.llm.model_name}')
print(f'API Key配置: {\"已配置\" if settings.llm.api_key else \"未配置\"}')
"

# 3. 运行简单测试
python run_cli.py discover -q "test query" -t examples/sample_tables.json

# 4. 运行完整测试
python tests/test_webtable_phase2.py
```

如果以上命令都成功运行，说明配置完成！🎉