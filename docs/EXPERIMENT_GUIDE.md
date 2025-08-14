# 🚀 优化版实验系统使用指南

## 系统特性

✅ **核心优化**
- 批量LLM调用（10个一批）
- 向量索引持久化（HNSW索引）
- 多智能体框架单例化（初始化一次）
- 查询去重自动处理

## 快速开始

### 1. 快速测试（5个查询，无LLM）
```bash
python run_experiments.py quick --task join --dataset subset
```

### 2. 消融实验（完整三层对比）
```bash
python run_experiments.py ablation --task join --dataset subset --max-queries 20
```

### 3. 完整实验（使用最佳配置）
```bash
python run_experiments.py full --task join --dataset subset --max-queries 20 --use-best-config
```

### 4. 消融实验（优化版）
```bash
python run_optimized_ablation.py --task join --dataset subset --max-queries 20
```

## 实验类型说明

| 实验类型 | 用途 | 运行时间 | LLM调用 |
|---------|------|----------|---------|
| `quick` | 快速验证系统 | 1-2分钟 | 否 |
| `ablation` | 三层性能对比 | 10-15分钟 | 是（L3层） |
| `full` | 完整系统测试 | 15-20分钟 | 是 |

## 主要脚本

### 核心脚本（保留）
- `run_experiments.py` - 统一实验入口
- `run_optimized_ablation.py` - 优化版消融实验
- `enhanced_ablation_experiment.py` - 原始消融实验
- `run_langgraph_system.py` - 主系统运行

### 配置文件
- `config_multi_agent_best.yml` - 最佳参数配置
- `config.yml` - 当前系统配置

## 向量索引持久化

系统会自动管理向量索引：

1. **首次运行**：构建索引并保存到 `cache/vector_index.pkl`
2. **后续运行**：直接加载已有索引，节省时间
3. **更新索引**：删除缓存文件重新构建

```bash
# 清理缓存（需要重建索引时）
rm -rf cache/vector_index.pkl cache/table_embeddings.pkl
```

## 批量LLM优化

系统自动批量处理LLM请求：
- 批量大小：10个查询
- 重试策略：最多5次，延迟2秒
- API限流：自动处理503错误

## 期望性能指标

| 配置 | Precision | Recall | F1 Score | Hit@1 | 时间/查询 |
|------|-----------|--------|----------|-------|-----------|
| L1_only | 0.2-0.3 | 0.4-0.5 | 0.3-0.4 | 0.1-0.2 | <1秒 |
| L1+L2 | 0.4-0.5 | 0.7-0.8 | 0.5-0.6 | 0.3-0.4 | 2-3秒 |
| L1+L2+L3 | 0.6-0.7 | 0.8-0.9 | 0.7-0.8 | 0.5-0.6 | 3-5秒 |

## 常见问题

### API过载（503错误）
```bash
# 增加延迟和重试
export LLM_RETRY_DELAY=5
export LLM_MAX_RETRIES=10
```

### 内存不足
```bash
# 使用更少的查询
python run_experiments.py ablation --max-queries 10
```

### 清理项目
```bash
# 删除所有缓存和临时文件
rm -rf cache/* experiment_results/* __pycache__/
```

## 实验结果

结果自动保存到 `experiment_results/ablation/` 目录：
- 文件名格式：`optimized_ablation_YYYYMMDD_HHMMSS.json`
- 包含详细指标和运行时间

## 开发说明

### 系统架构
```
统一入口 (run_experiments.py)
    ├── 消融实验 (run_optimized_ablation.py)
    │   ├── 批量LLM处理器
    │   ├── 持久化向量索引
    │   └── 单例Workflow
    └── 完整实验 (run_langgraph_system.py)
        └── 多智能体系统
```

### 关键优化
1. **初始化一次**：Workflow和索引只初始化一次
2. **批量处理**：LLM请求批量发送
3. **缓存复用**：向量和嵌入缓存持久化
4. **内存安全**：禁用并行处理避免段错误

---

**注意**：确保 `.env` 文件包含有效的API密钥（GEMINI_API_KEY）