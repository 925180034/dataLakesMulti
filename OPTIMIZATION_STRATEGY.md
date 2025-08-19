# 🚀 系统优化策略 - 基于实验结果

## 📊 当前性能分析

### JOIN任务问题（F1: 11.8%）
- **症状**: Precision和Recall都很低
- **原因分析**:
  - 数据集特征复杂（特殊列名、结构差异大）
  - L1层过滤太严格，导致召回率低
  - 阈值设置仍然偏高

### UNION任务问题（F1: 30.9%）
- **症状**: Precision很高(81.8%)但Recall低(20.4%)
- **原因分析**:
  - L3层阈值太高，过滤掉了很多正确答案
  - 需要更好的precision-recall平衡

## ✅ 优化方案

### 1. 🔧 参数优化（已更新）

#### JOIN任务新参数
```python
llm_confidence_threshold = 0.15  # 从0.25降到0.15
aggregator_min_score = 0.02      # 从0.05降到0.02
aggregator_max_results = 300     # 从150增到300
vector_top_k = 400               # 从250增到400
```

#### UNION任务新参数
```python
llm_confidence_threshold = 0.20  # 从0.35降到0.20
aggregator_min_score = 0.05      # 从0.08降到0.05
aggregator_max_results = 150     # 从80增到150
vector_top_k = 250               # 从150增到250
```

### 2. 💻 进程数优化

您有**128个CPU核心**，当前只用16个进程！

```bash
# 推荐配置（充分利用CPU）
--workers 32   # 保守方案，2-3倍加速
--workers 48   # 平衡方案，3-4倍加速
--workers 64   # 激进方案，4-5倍加速

# 示例命令
python three_layer_ablation_optimized.py \
    --task both \
    --dataset subset \
    --max-queries all \
    --workers 48
```

### 3. 🎯 L1层优化建议

修改 `src/tools/smd_enhanced_metadata_filter.py`:

```python
# 放宽过滤条件
COLUMN_COUNT_DIFF_THRESHOLD = 15  # 从10增加到15
MIN_COLUMN_OVERLAP_RATIO = 0.1    # 从0.2降到0.1
MIN_TYPE_SIMILARITY = 0.3         # 从0.5降到0.3
```

### 4. 🔍 L2层优化建议

修改向量搜索策略:

```python
# 增加候选数量
DEFAULT_TOP_K = 300     # 从100增加到300
SIMILARITY_THRESHOLD = 0.5  # 从0.7降到0.5
```

### 5. 📈 动态优化策略

创建自适应优化脚本：

```python
# adaptive_runtime_optimizer.py
class AdaptiveOptimizer:
    def optimize_based_on_feedback(self, precision, recall):
        if recall < 0.3:  # 召回率太低
            # 降低所有阈值
            self.lower_thresholds()
        elif precision < 0.3:  # 精确率太低
            # 提高阈值
            self.raise_thresholds()
        return new_config
```

## 🎯 验证动态参数是否生效

### 检查日志
```bash
grep -E "DEBUG.*parameters|confidence_threshold|aggregator" your_log_file.log
```

您应该看到：
- JOIN: `DEBUG: Applying JOIN-specific parameters (ULTRA LOW thresholds)`
- UNION: `DEBUG: Applying UNION-specific parameters (balanced for recall)`

### 预期改进

| 指标 | 当前值 | 目标值 | 优化后预期 |
|------|--------|--------|-----------|
| **JOIN F1** | 11.8% | 30%+ | 20-25% |
| **JOIN Recall** | 14.6% | 40%+ | 25-30% |
| **UNION F1** | 30.9% | 40%+ | 35-40% |
| **UNION Recall** | 20.4% | 35%+ | 30-35% |

## 🚀 立即执行的优化步骤

### Step 1: 测试新参数
```bash
# 快速测试（10个查询）
python three_layer_ablation_optimized.py \
    --task both \
    --dataset subset \
    --max-queries 10 \
    --workers 32
```

### Step 2: 如果改进明显，运行完整测试
```bash
# 完整测试（所有查询）
python three_layer_ablation_optimized.py \
    --task both \
    --dataset subset \
    --max-queries all \
    --workers 48
```

### Step 3: 进一步调优
如果JOIN仍然差，继续降低阈值：
```python
llm_confidence_threshold = 0.10  # 极限低阈值
aggregator_min_score = 0.01      # 几乎不过滤
```

## 📊 监控和调试

### 实时监控
```bash
# 监控CPU使用
htop

# 监控内存
watch -n 1 free -h

# 监控进程
ps aux | grep python | wc -l
```

### 性能分析
```python
# 添加到代码中
import cProfile
cProfile.run('your_function()', 'profile_stats')
```

## ⚠️ 注意事项

1. **API限制**: 降低阈值会增加API调用量
2. **内存使用**: 增加候选数量会占用更多内存
3. **处理时间**: 更多候选意味着更长处理时间
4. **成本控制**: 监控API使用量避免超支

## 🎉 总结

**动态参数调整确实在起作用**，但需要：
1. ✅ 更激进的参数（已调整）
2. ✅ 充分利用CPU（32-64个进程）
3. ⏳ L1/L2层优化（待实施）
4. ⏳ 自适应调优（可选）

通过这些优化，预期：
- JOIN F1-Score: 11.8% → 20-25%
- UNION F1-Score: 30.9% → 35-40%
- 处理速度: 提升3-4倍（使用48个进程）