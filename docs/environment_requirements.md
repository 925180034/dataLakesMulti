# 环境配置需求文档

## Conda环境创建

```bash
# 创建conda环境
conda create -n data_lakes_multi python=3.10 -y

# 激活环境
conda activate data_lakes_multi
```

## 基础依赖安装

### 核心依赖
```bash
pip install langgraph==0.0.55
pip install langchain==0.1.16
pip install langchain-openai==0.1.6
pip install langchain-anthropic==0.1.11
```

### 向量搜索和索引
```bash
pip install faiss-cpu==1.7.4
pip install chromadb==0.4.22
pip install sentence-transformers==2.7.0
```

### 数据处理
```bash
pip install pandas==2.2.2
pip install numpy==1.24.3
pip install scikit-learn==1.4.2
```

### 配置管理
```bash
pip install pydantic==2.7.1
pip install python-dotenv==1.0.1
pip install pyyaml==6.0.1
```

### 开发工具
```bash
pip install jupyter==1.0.0
pip install pytest==8.2.0
pip install black==24.4.2
pip install isort==5.13.2
```

## 配置文件说明

### 主配置文件 `config.yml`
系统使用YAML文件进行配置，主要配置项包括：
- 项目基础信息
- LLM提供商和模型设置
- 向量数据库配置
- 阈值参数
- 缓存和日志设置
- 性能和安全配置

### 环境变量配置 `.env`
创建 `.env` 文件存放敏感信息：
```bash
# LLM API密钥（必须）
OPENAI_API_KEY=your_openai_api_key_here
# 或者使用Anthropic
ANTHROPIC_API_KEY=your_anthropic_api_key_here

# 可选的路径覆盖
VECTOR_DB_PATH=./data/vector_db
INDEX_DB_PATH=./data/index_db
CACHE_DIR=./cache

# 可选的开关配置
DEBUG=false
CACHE_ENABLED=true
```

### 本地配置覆盖
可以创建 `config.local.yml` 文件覆盖默认配置（该文件已被.gitignore忽略）

## 验证安装

运行以下命令验证环境配置：
```bash
python -c "import langgraph; import langchain; import faiss; import pandas; print('环境配置成功！')"
```