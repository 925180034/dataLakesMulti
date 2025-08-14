# 自定义数据模板 / Custom Data Template

这个文件夹包含了准备您自己数据的模板文件。

## 📁 文件说明

1. **custom_tables.json** - 表数据模板
2. **custom_queries.json** - 查询任务模板  
3. **custom_ground_truth.json** - Ground Truth模板

## 🚀 快速开始

### 1. 准备您的数据

编辑模板文件，替换为您的实际数据：

```bash
# 编辑表数据
vim custom_tables.json

# 编辑查询任务
vim custom_queries.json

# （可选）编辑ground truth
vim custom_ground_truth.json
```

### 2. 验证数据格式

```python
import json

# 验证JSON格式
with open('custom_tables.json', 'r') as f:
    tables = json.load(f)
    print(f"✅ 成功加载 {len(tables)} 个表")

with open('custom_queries.json', 'r') as f:
    queries = json.load(f)
    print(f"✅ 成功加载 {len(queries)} 个查询")
```

### 3. 运行实验

```bash
# 使用自定义实验脚本
python ../../run_custom_experiment.py \
  --tables custom_tables.json \
  --queries custom_queries.json \
  --ground-truth custom_ground_truth.json \
  --task join \
  --max-queries 10
```

### 4. 查看结果

实验结果将保存在 `experiment_results/` 文件夹中。

## 💡 数据准备提示

### 表数据要点
- **table_name**: 必须唯一
- **columns**: 至少包含2个列
- **data_type**: 支持 string, numeric, date
- **sample_values**: 提供3-5个代表性样本值

### 查询任务要点
- **query_table**: 必须在tables中存在
- **query_type**: "join" 或 "union"
- **query_id**: 用于追踪结果

### Ground Truth要点
- **query_table**: 查询表名
- **candidate_table**: 正确的匹配表
- **label**: 1表示匹配，0表示不匹配

## 📊 期望输出

成功运行后，您将获得：
- **精确率、召回率、F1分数**
- **Hit@K指标**
- **查询时间和吞吐量**
- **详细的匹配结果**

## ❓ 常见问题

1. **内存不足**: 减少表数量或增加系统内存
2. **API限制**: 降低LLM并发数
3. **格式错误**: 使用JSON验证工具检查格式
4. **性能慢**: 启用缓存和批处理