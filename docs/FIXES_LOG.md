# 修复总结 - run_cached_experiments.py 评估指标问题

## 问题描述
运行 `python run_cached_experiments.py --task join --dataset subset --max-queries 10` 时，所有评估指标都显示为0。

## 问题原因
Ground truth 数据格式不匹配：
- **评估函数期望格式**: `candidate_tables`（复数，列表）
- **实际数据格式**: `candidate_table`（单数，单个值）

示例：
```json
// 实际格式（examples/separated_datasets/join_subset/ground_truth.json）
{
  "query_table": "csvData4676916__9.csv",
  "candidate_table": "csvData4676916__9.csv",  // 单数，单个值
  ...
}

// 评估函数期望的格式
{
  "query_table": "csvData4676916__9.csv",
  "candidate_tables": ["table1", "table2", ...],  // 复数，列表
  ...
}
```

## 修复方案

### 1. 添加格式转换函数
```python
def convert_ground_truth_format(ground_truth_list: List[Dict]) -> List[Dict]:
    """
    将ground truth从单个候选表格式转换为列表格式
    并按查询表聚合，同时过滤自匹配
    """
    query_to_candidates = {}
    
    for item in ground_truth_list:
        query_table = item.get('query_table', '')
        candidate_table = item.get('candidate_table', '')
        
        if query_table and candidate_table:
            # 过滤自匹配（查询表和候选表相同的情况）
            if query_table != candidate_table:
                if query_table not in query_to_candidates:
                    query_to_candidates[query_table] = set()
                query_to_candidates[query_table].add(candidate_table)
    
    # 转换为期望的格式
    converted = []
    for query_table, candidates in query_to_candidates.items():
        if candidates:  # 只保留有候选表的
            converted.append({
                'query_table': query_table,
                'candidate_tables': list(candidates)
            })
    
    return converted
```

### 2. 在计算指标前转换格式
```python
# 转换ground truth格式
converted_ground_truth = convert_ground_truth_format(ground_truth)
logger.info(f"Ground truth转换: {len(ground_truth)} 条 -> {len(converted_ground_truth)} 个查询表")

# 计算评估指标
from evaluate_with_metrics import calculate_metrics
metrics = calculate_metrics(results, converted_ground_truth)
```

### 3. 过滤预测结果中的自匹配
```python
# 获取预测结果并过滤自匹配
predictions = [r['table_name'] for r in result.get('results', [])[:10]]
filtered_predictions = [p for p in predictions if p != query_table_name]

results.append({
    'query_table': query_table_name,
    'predictions': filtered_predictions[:5],  # 保留top-5
    'time': elapsed
})
```

## 验证结果

修复后运行测试：
```bash
python run_cached_experiments.py --task join --dataset subset --max-queries 3
```

成功获得评估指标：
- Hit@5: 1.000
- Hit@10: 1.000
- Precision: 0.200
- Recall: 1.000
- F1-Score: 0.333

## 关键改进
1. **格式转换**: 将ground truth从单个候选表格式聚合为列表格式
2. **自匹配过滤**: 过滤掉查询表和候选表相同的情况
3. **数据聚合**: 将多条记录按查询表聚合，便于评估

## 使用方法
修复后的脚本可以正常运行：
```bash
# 运行10个查询的完整测试
python run_cached_experiments.py --task join --dataset subset --max-queries 10

# 运行所有查询（可能需要较长时间）
python run_cached_experiments.py --task join --dataset subset --max-queries 100
```

现在评估指标能够正确计算并显示实际的系统性能！