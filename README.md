# 数据湖模式匹配与数据发现系统

基于大语言模型API的多智能体数据湖模式匹配与数据发现系统。

## 项目结构

```
dataLakesMulti/
├── src/
│   ├── agents/          # 智能体实现
│   ├── core/           # 核心数据结构和状态管理
│   ├── tools/          # 搜索和匹配工具
│   ├── config/         # 配置管理
│   └── utils/          # 工具函数
├── data/
│   ├── vector_db/      # 向量数据库
│   ├── index_db/       # 倒排索引
│   └── datasets/       # 数据集
├── cache/              # 缓存目录
├── logs/               # 日志目录
└── tests/              # 测试文件
```

## 安装和配置

### 1. 环境安装
请参考 `environment_requirements.md` 文件配置开发环境。

### 2. 配置系统
- **主配置文件**: `config.yml` - 包含所有系统配置
- **环境变量**: `.env` - 存放API密钥等敏感信息
- **本地配置**: `config.local.yml` - 本地开发时的配置覆盖（可选）

### 3. 快速开始
```bash
# 1. 创建conda环境
conda create -n data_lakes_multi python=3.10 -y
conda activate data_lakes_multi

# 2. 安装依赖
pip install -r requirements.txt

# 3. 配置API密钥
cp .env.example .env
# 编辑.env文件，填入你的API密钥

# 4. 运行测试
python -m pytest tests/
```

## 核心功能

- **表头匹配（Schema Matching）**：寻找具有相似列结构的表
- **数据实例匹配（Data Instance Matching）**：基于数据内容发现语义相关的表
- **智能路由**：根据用户意图自动选择最优处理策略
- **多智能体协作**：专业化智能体协同工作