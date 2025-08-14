# 统一实验框架文档

## 概述

统一实验框架确保消融实验和参数优化实验使用完全相同的系统策略，保证实验结果的可比性和可重复性。

## 核心特性

### 1. 统一的系统初始化
- **预计算嵌入**: 所有表的嵌入向量在实验开始时预计算，避免重复计算
- **缓存机制**: 嵌入向量保存到`cache/unified_embeddings.pkl`
- **环境变量**: 通过`USE_PRECOMPUTED_EMBEDDINGS`环境变量传递给系统

### 2. 真实的多智能体系统
- **完整Agent链**: OptimizerAgent → PlannerAgent → AnalyzerAgent → SearcherAgent → MatcherAgent → AggregatorAgent
- **LLM调用**: 始终启用LLM（`SKIP_LLM=false`）
- **实时推理**: 每个Agent都使用真实的LLM进行决策

### 3. 真实的数据输入
- **数据集**: 使用`examples/separated_datasets/`中的真实数据
- **Ground Truth**: 包含84个验证对，确保评价准确性
- **自动过滤**: 自动过滤没有ground truth的查询

### 4. 统一的评价指标

| 指标 | 描述 | 计算方式 |
|------|------|----------|
| Precision@10 | 前10个结果的精确率 | TP/(TP+FP) |
| Recall | 召回率 | TP/(TP+FN) |
| F1 Score | 精确率和召回率的调和平均 | 2×P×R/(P+R) |
| Hit@K | 前K个结果包含正确答案的比例 | 1 if any correct in top-K else 0 |
| MRR | 平均倒数排名 | 1/rank of first correct |

## 实验类型

### 消融实验（Ablation Study）
测试不同层级组合的效果：
- **L1_only**: 仅使用元数据过滤
- **L2_only**: 仅使用向量搜索
- **L1+L2**: 元数据过滤 + 向量搜索
- **L1+L2+L3**: 完整三层架构
- **L3_only**: 仅使用LLM验证

### 参数优化实验（Parameter Optimization）
测试不同参数配置的效果：
- **baseline**: 基准配置（平衡各项参数）
- **optimized**: 优化配置（基于实验调优）
- **aggressive**: 激进配置（低阈值高召回）

## 使用方法

### 基本命令
```bash
# 运行消融实验
python unified_experiment.py --experiment ablation --max-queries 10

# 运行参数优化实验
python unified_experiment.py --experiment optimization --max-queries 10

# 同时运行两种实验
python unified_experiment.py --experiment both --max-queries 10
```

### 高级选项
```bash
python unified_experiment.py \
  --experiment both \          # 实验类型: ablation/optimization/both
  --task join \               # 任务类型: join/union
  --dataset subset \          # 数据集: subset(100表)/complete(1534表)
  --max-queries 10            # 最大查询数量
```

## 关键配置参数

### 向量搜索参数
- `similarity_threshold`: 相似度阈值（0.25-0.6）
- `top_k`: 返回候选数量（20-100）

### 元数据过滤参数
- `column_similarity_threshold`: 列名相似度阈值（0.3-0.5）
- `min_column_overlap`: 最小列重叠数（1-3）

### LLM匹配参数
- `confidence_threshold`: 置信度阈值（0.5-0.7）
- `batch_size`: 批处理大小（5-15）

### 评分权重
- `metadata`: 元数据权重（0.15-0.45）
- `vector`: 向量搜索权重（0.25-0.50）
- `llm`: LLM验证权重（0.25-0.55）

## 输出结果

### 结果文件位置
```
experiment_results/unified/
├── ablation_YYYYMMDD_HHMMSS.json      # 消融实验结果
└── optimization_YYYYMMDD_HHMMSS.json  # 参数优化结果
```

### 结果格式
```json
{
  "config_name": {
    "config": {...},           // 配置参数
    "avg_metrics": {           // 平均指标
      "precision": 0.xxx,
      "recall": 0.xxx,
      "f1": 0.xxx,
      "hit_at_1": 0.xxx,
      "hit_at_3": 0.xxx,
      "hit_at_5": 0.xxx,
      "mrr": 0.xxx,
      "avg_time": x.xx
    },
    "detailed_results": [...]  // 每个查询的详细结果
  }
}
```

## 验证实验一致性

运行验证脚本确认系统配置：
```bash
python verify_unified_experiment.py
```

验证项目：
- ✅ 系统配置（预计算嵌入、LLM状态）
- ✅ 数据完整性（表、查询、ground truth）
- ✅ 多智能体系统（6个Agent）
- ✅ 评价指标一致性
- ✅ 实验类型支持

## 性能优化

### 预计算嵌入
- 实验开始时一次性计算所有嵌入
- 避免重复计算，提升3-5倍速度

### 批处理
- LLM调用批处理
- 向量搜索批处理

### 缓存机制
- Workflow缓存（消融实验）
- 配置缓存（参数优化）

## 常见问题

### Q: 为什么要使用统一框架？
A: 确保不同实验使用相同的系统策略，结果可比较。

### Q: 如何确保使用真实LLM？
A: 框架自动设置`SKIP_LLM=false`，强制启用LLM调用。

### Q: 预计算嵌入是否影响准确性？
A: 不影响。预计算使用相同的嵌入模型，只是提前计算。

### Q: 如何添加新的参数配置？
A: 在`run_parameter_optimization`方法中添加新的配置字典。

### Q: 实验结果如何对比？
A: 框架自动生成对比报告，显示所有配置的性能指标。

## 总结

统一实验框架提供了一个标准化、可重复、高性能的实验环境，确保：
- **一致性**: 所有实验使用相同的系统和数据
- **真实性**: 使用真实的多智能体系统和LLM
- **可比性**: 统一的评价指标和输出格式
- **高效性**: 预计算和缓存优化性能