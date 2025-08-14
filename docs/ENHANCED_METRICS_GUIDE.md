# 增强版实验框架使用指南

## 📊 新增评价指标

### Precision@K 和 Recall@K
现在系统支持更细粒度的评价指标：

| 指标 | 描述 | 计算方式 |
|------|------|----------|
| **Precision@5** | 前5个结果的精确率 | TP@5 / 5 |
| **Precision@10** | 前10个结果的精确率 | TP@10 / 10 |
| **Recall@5** | 前5个结果的召回率 | TP@5 / Total_GT |
| **Recall@10** | 前10个结果的召回率 | TP@10 / Total_GT |
| **F1@5** | Precision@5和Recall@5的调和平均 | 2×P@5×R@5/(P@5+R@5) |
| **F1@10** | Precision@10和Recall@10的调和平均 | 2×P@10×R@10/(P@10+R@10) |

### 其他高级指标
- **MRR** (Mean Reciprocal Rank): 第一个正确答案的倒数排名
- **NDCG@5/10** (Normalized Discounted Cumulative Gain): 考虑排序质量的指标
- **Hit@1/3/5/10**: 前K个结果包含正确答案的比例

## 🚀 快速开始

### 1. 验证流程（推荐）

使用验证脚本进行渐进式测试：

```bash
# 运行完整验证流程（推荐）
python run_experiment_validation.py --run

# 只运行快速测试
python run_experiment_validation.py --quick

# 查看结果
python run_experiment_validation.py --results
```

验证流程包含三个步骤：
1. **快速验证** - subset数据集，5个查询（1-2分钟）
2. **Subset完整实验** - 20个查询（5-10分钟）
3. **Complete实验** - 1534表，10个查询（10-20分钟）

### 2. 直接运行实验

#### Subset数据集（100表）

```bash
# 快速测试（5个查询）
python unified_experiment_with_metrics.py \
  --experiment both \
  --dataset subset \
  --max-queries 5 \
  --quick

# 完整测试（20个查询）
python unified_experiment_with_metrics.py \
  --experiment both \
  --dataset subset \
  --max-queries 20
```

#### Complete数据集（1534表）

```bash
# 标准测试（10个查询）
python unified_experiment_with_metrics.py \
  --experiment both \
  --dataset complete \
  --max-queries 10

# 完整测试（50个查询，需要更长时间）
python unified_experiment_with_metrics.py \
  --experiment both \
  --dataset complete \
  --max-queries 50
```

## 📈 数据集对比

| 特性 | Subset | Complete |
|------|--------|----------|
| 表数量 | 100 | 1534 |
| 查询数量 | 402 | 更多 |
| Ground Truth | 84 | 更多 |
| 首次运行时间 | 5-10分钟 | 20-30分钟 |
| 嵌入缓存文件 | ~40KB | ~600KB |
| 推荐查询数 | 20 | 10-20 |

## 🔧 性能优化

### 嵌入缓存
系统会自动缓存预计算的嵌入向量：
- Subset: `cache/embeddings_subset.pkl`
- Complete: `cache/embeddings_complete.pkl`

首次运行会生成缓存，后续运行会直接加载。

### 大数据集优化
对于complete数据集，系统自动启用：
- 批处理（batch_size=20）
- 增加并行度（max_workers=5）
- 内存清理（定期gc.collect()）

## 📊 结果解读

### 实验结果位置
```
experiment_results/unified_enhanced/
├── ablation_subset_YYYYMMDD_HHMMSS.json
├── optimization_subset_YYYYMMDD_HHMMSS.json
├── ablation_complete_YYYYMMDD_HHMMSS.json
└── optimization_complete_YYYYMMDD_HHMMSS.json
```

### 关键指标解读

#### 消融实验结果
- **L1_only**: 仅元数据过滤的效果
- **L2_only**: 仅向量搜索的效果
- **L1+L2**: 元数据+向量的组合效果
- **L1+L2+L3**: 完整三层架构的效果

期望结果：L1+L2+L3 > L1+L2 > L2_only > L1_only

#### 参数优化结果
- **baseline**: 基准配置
- **optimized**: 优化后的配置
- **aggressive**: 激进配置（高召回）

期望结果：optimized在F1@10上表现最好

### 典型性能指标

#### Subset数据集预期
- Precision@10: 0.4-0.6
- Recall@10: 0.6-0.8
- F1@10: 0.5-0.7
- Hit@1: 0.3-0.5
- MRR: 0.4-0.6

#### Complete数据集预期
- Precision@10: 0.3-0.5（更难）
- Recall@10: 0.5-0.7
- F1@10: 0.4-0.6
- Hit@1: 0.2-0.4
- MRR: 0.3-0.5

## 🛠️ 故障排除

### 常见问题

#### 1. LLM超时
```
错误: Request timeout
解决: 增加--max-queries参数或使用--quick模式
```

#### 2. 内存不足
```
错误: MemoryError
解决: 减少查询数量或使用subset数据集
```

#### 3. 嵌入生成失败
```
错误: Failed to generate embeddings
解决: 删除cache目录重新生成
```

### 调试建议

1. **先验证后实验**：始终从快速验证开始
2. **逐步增加规模**：subset → complete，少查询 → 多查询
3. **监控资源**：使用htop观察内存和CPU使用
4. **检查日志**：查看详细错误信息

## 📝 实验建议

### 最佳实践工作流

1. **初次运行**
   ```bash
   # 快速验证系统工作
   python run_experiment_validation.py --quick
   ```

2. **Subset实验**
   ```bash
   # 运行subset完整实验
   python unified_experiment_with_metrics.py \
     --experiment both \
     --dataset subset \
     --max-queries 20
   ```

3. **Complete实验**
   ```bash
   # 确认subset结果良好后运行
   python unified_experiment_with_metrics.py \
     --experiment both \
     --dataset complete \
     --max-queries 10
   ```

4. **结果分析**
   ```bash
   # 查看所有结果
   python run_experiment_validation.py --results
   ```

### 参数调优建议

基于实验结果调整参数：
- 如果Precision低：提高confidence_threshold
- 如果Recall低：降低similarity_threshold
- 如果Hit@1低：调整scoring weights

## 总结

增强版实验框架提供了：
- ✅ 更细粒度的评价指标（P@5/10, R@5/10）
- ✅ 支持大数据集（1534表）
- ✅ 自动性能优化
- ✅ 渐进式验证流程
- ✅ 详细的结果报告

使用验证脚本可以安全地从小规模测试逐步扩展到完整实验，确保系统稳定性和结果可靠性。