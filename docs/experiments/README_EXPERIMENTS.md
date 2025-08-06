# 📊 实验指南 - WebTable JOIN & UNION 测试

## 快速开始

### 1️⃣ 最简单的测试方式
```bash
# 运行快速测试（各5个查询）
./quick_test.sh
```

### 2️⃣ 分别运行JOIN和UNION实验

#### JOIN实验（模式匹配/列连接）
```bash
# 小规模测试（10个查询）
python run_join_experiment.py \
    --dataset examples/separated_datasets/join_subset \
    --max-queries 10 \
    --verbose

# 完整测试（402个查询）
python run_join_experiment.py \
    --dataset examples/separated_datasets/join_subset \
    --verbose

# 大规模测试（使用complete数据集，5824个查询）
python run_join_experiment.py \
    --dataset examples/separated_datasets/join_complete \
    --max-queries 100
```

#### UNION实验（数据实例相似性）
```bash
# 小规模测试（10个查询）
python run_union_experiment.py \
    --dataset examples/separated_datasets/union_subset \
    --max-queries 10 \
    --verbose

# 完整测试（100个查询）
python run_union_experiment.py \
    --dataset examples/separated_datasets/union_subset \
    --verbose

# 大规模测试（使用complete数据集，1534个查询）
python run_union_experiment.py \
    --dataset examples/separated_datasets/union_complete \
    --max-queries 100
```

### 3️⃣ 运行全部实验（自动化）
```bash
# 运行所有配置的实验
python run_all_experiments.py
```

## 📁 数据集说明

### Subset数据集（快速测试用）
- **位置**: `examples/separated_datasets/join_subset/` 和 `union_subset/`
- **规模**: 100个表
- **JOIN**: 402个查询，84个ground truth
- **UNION**: 100个查询，110个ground truth
- **用途**: 算法调试和快速验证

### Complete数据集（完整评估用）
- **位置**: `examples/separated_datasets/join_complete/` 和 `union_complete/`
- **规模**: 1,534个表
- **JOIN**: 5,824个查询，6,805个ground truth
- **UNION**: 1,534个查询，8,248个ground truth
- **用途**: 性能评估和最终测试

## 📊 评估指标

实验会自动计算以下指标：

1. **准确性指标**
   - Precision（精确率）: 预测正确的比例
   - Recall（召回率）: 找到所有正确答案的比例
   - F1-Score: Precision和Recall的调和平均

2. **性能指标**
   - 平均查询时间
   - 总执行时间
   - 每秒查询数(QPS)

3. **匹配统计**
   - True Positives（真阳性）
   - False Positives（假阳性）
   - False Negatives（假阴性）

## 📈 实验结果

结果会自动保存在 `experiment_results/` 目录：
```
experiment_results/
├── join/
│   └── join_experiment_YYYYMMDD_HHMMSS.json
├── union/
│   └── union_experiment_YYYYMMDD_HHMMSS.json
└── experiment_summary_YYYYMMDD_HHMMSS.json
```

## 🔧 高级选项

### 自定义实验参数
```python
# 在Python脚本中自定义
from run_join_experiment import run_join_experiment

results = run_join_experiment(
    dataset_path="your/dataset/path",
    max_queries=50,      # 限制查询数量
    verbose=True         # 显示详细进度
)
```

### 调试模式
```bash
# 设置环境变量启用调试
export DEBUG=true
export CACHE_ENABLED=false  # 禁用缓存以测试真实性能

# 运行实验
python run_join_experiment.py --dataset examples/separated_datasets/join_subset --max-queries 5 --verbose
```

## ⚠️ 注意事项

1. **API密钥配置**
   - 确保 `.env` 文件中配置了正确的API密钥
   - 推荐使用 `GEMINI_API_KEY`（免费且稳定）

2. **内存使用**
   - 大规模数据集可能需要较多内存
   - 如遇到内存问题，减少 `--max-queries` 参数

3. **执行时间**
   - Subset测试：约1-5分钟
   - Complete测试：约10-30分钟（取决于API响应速度）

4. **并发限制**
   - 默认使用串行处理避免API限流
   - 可在 `config.yml` 中调整并发设置

## 📝 结果解读

### 好的结果标准
- **JOIN任务**: Precision > 0.8, Recall > 0.7, F1 > 0.75
- **UNION任务**: Precision > 0.7, Recall > 0.6, F1 > 0.65
- **查询速度**: < 5秒/查询（使用缓存时 < 1秒）

### 常见问题
1. **低Precision**: 系统返回了太多错误的候选
2. **低Recall**: 系统遗漏了正确的候选
3. **查询超时**: 检查API连接和配额

## 🚀 优化建议

1. **启用缓存**: 在 `config.yml` 中设置 `cache_enabled: true`
2. **使用索引**: 运行 `python run_cli.py index-tables` 预建索引
3. **批处理**: 使用 `--max-queries` 分批处理大数据集
4. **模型选择**: 在 `.env` 中选择合适的LLM模型

## 📧 问题反馈

如遇到问题，请检查：
1. 日志文件：`logs/` 目录
2. 错误信息：实验输出的错误提示
3. 配置文件：`config.yml` 和 `.env` 设置