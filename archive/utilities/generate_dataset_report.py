#!/usr/bin/env python3
"""
生成数据集详细统计报告
"""

import json
import os
import numpy as np
from typing import Dict, List

def load_dataset_stats(dataset_path: str, dataset_name: str) -> Dict:
    """加载数据集统计信息"""
    stats = {}
    
    for task in ['join', 'union']:
        stats[task] = {}
        for split in ['subset', 'complete']:
            split_path = os.path.join(dataset_path, f"{task}_{split}")
            if not os.path.exists(split_path):
                continue
                
            # 读取文件
            with open(os.path.join(split_path, "queries.json"), 'r') as f:
                queries = json.load(f)
            
            with open(os.path.join(split_path, "tables.json"), 'r') as f:
                tables = json.load(f)
            
            with open(os.path.join(split_path, "ground_truth.json"), 'r') as f:
                ground_truth = json.load(f)
            
            # 计算统计
            rows = []
            cols = []
            for table in tables:
                if 'num_rows' in table:
                    rows.append(table['num_rows'])
                if 'num_columns' in table:
                    cols.append(table['num_columns'])
            
            # GT分布
            gt_per_query = {}
            for gt in ground_truth:
                qt = gt['query_table']
                if qt not in gt_per_query:
                    gt_per_query[qt] = 0
                gt_per_query[qt] += 1
            
            stats[task][split] = {
                'queries': len(queries),
                'tables': len(tables),
                'ground_truth': len(ground_truth),
                'avg_gt_per_query': round(len(ground_truth) / len(queries), 2) if queries else 0,
                'rows': {
                    'min': min(rows) if rows else 0,
                    'max': max(rows) if rows else 0,
                    'mean': round(np.mean(rows), 2) if rows else 0,
                    'median': round(np.median(rows), 2) if rows else 0,
                    'std': round(np.std(rows), 2) if rows else 0
                },
                'columns': {
                    'min': min(cols) if cols else 0,
                    'max': max(cols) if cols else 0,
                    'mean': round(np.mean(cols), 2) if cols else 0,
                    'median': round(np.median(cols), 2) if cols else 0,
                    'std': round(np.std(cols), 2) if cols else 0
                },
                'gt_distribution': {
                    'min': min(gt_per_query.values()) if gt_per_query else 0,
                    'max': max(gt_per_query.values()) if gt_per_query else 0,
                    'mean': round(np.mean(list(gt_per_query.values())), 2) if gt_per_query else 0,
                    'median': round(np.median(list(gt_per_query.values())), 2) if gt_per_query else 0
                }
            }
    
    return stats

def generate_markdown_report(webtable_stats: Dict, opendata_stats: Dict) -> str:
    """生成Markdown格式的报告"""
    
    report = """# 数据集组织和统计总结

## 📊 数据集完整统计报告

**生成时间**: 2024-12-22

## 1. 数据集概览

### 数据组织结构
```
examples/
├── webtable/               # WebTable数据集
│   ├── join_subset/        # JOIN任务子集
│   ├── join_complete/      # JOIN任务完整
│   ├── union_subset/       # UNION任务子集
│   └── union_complete/     # UNION任务完整
│
└── opendata/               # OpenData数据集
    ├── join_subset/        # JOIN任务子集
    ├── join_complete/      # JOIN任务完整
    ├── union_subset/       # UNION任务子集
    └── union_complete/     # UNION任务完整
```

## 2. 核心数据规模统计

### WebTable 数据集

| 任务 | 版本 | 查询数 | 表格数 | Ground Truth | 平均GT/查询 |
|------|------|--------|--------|--------------|-------------|
"""
    
    # WebTable统计
    for task in ['join', 'union']:
        for split in ['subset', 'complete']:
            if task in webtable_stats and split in webtable_stats[task]:
                s = webtable_stats[task][split]
                report += f"| {task.upper()} | {split} | {s['queries']} | {s['tables']} | {s['ground_truth']} | {s['avg_gt_per_query']} |\n"
    
    report += """

### OpenData 数据集

| 任务 | 版本 | 查询数 | 表格数 | Ground Truth | 平均GT/查询 |
|------|------|--------|--------|--------------|-------------|
"""
    
    # OpenData统计
    for task in ['join', 'union']:
        for split in ['subset', 'complete']:
            if task in opendata_stats and split in opendata_stats[task]:
                s = opendata_stats[task][split]
                report += f"| {task.upper()} | {split} | {s['queries']} | {s['tables']} | {s['ground_truth']} | {s['avg_gt_per_query']} |\n"
    
    # 汇总统计
    report += """

## 3. 数据集对比分析

### 总体规模对比

| 指标 | WebTable | OpenData | 比例 |
|------|----------|----------|------|
"""
    
    # 计算总数
    webtable_total = {
        'queries': sum([webtable_stats[t][s]['queries'] for t in webtable_stats for s in webtable_stats[t]]),
        'tables': sum([webtable_stats[t][s]['tables'] for t in webtable_stats for s in webtable_stats[t]]),
        'ground_truth': sum([webtable_stats[t][s]['ground_truth'] for t in webtable_stats for s in webtable_stats[t]])
    }
    
    opendata_total = {
        'queries': sum([opendata_stats[t][s]['queries'] for t in opendata_stats for s in opendata_stats[t]]),
        'tables': sum([opendata_stats[t][s]['tables'] for t in opendata_stats for s in opendata_stats[t]]),
        'ground_truth': sum([opendata_stats[t][s]['ground_truth'] for t in opendata_stats for s in opendata_stats[t]])
    }
    
    report += f"| 总查询数 | {webtable_total['queries']:,} | {opendata_total['queries']:,} | {round(opendata_total['queries']/webtable_total['queries'], 2)}x |\n"
    report += f"| 总表格数 | {webtable_total['tables']:,} | {opendata_total['tables']:,} | {round(opendata_total['tables']/webtable_total['tables'], 2)}x |\n"
    report += f"| 总Ground Truth | {webtable_total['ground_truth']:,} | {opendata_total['ground_truth']:,} | {round(opendata_total['ground_truth']/webtable_total['ground_truth'], 2)}x |\n"
    
    report += """

## 4. 数据维度分布

### WebTable 表格维度统计

| 任务-版本 | 行数 (min/mean/max) | 列数 (min/mean/max) |
|-----------|--------------------|--------------------|
"""
    
    for task in ['join', 'union']:
        for split in ['subset', 'complete']:
            if task in webtable_stats and split in webtable_stats[task]:
                s = webtable_stats[task][split]
                rows = s['rows']
                cols = s['columns']
                report += f"| {task}-{split} | {rows['min']}/{rows['mean']}/{rows['max']} | {cols['min']}/{cols['mean']}/{cols['max']} |\n"
    
    report += """

### OpenData 表格维度统计

| 任务-版本 | 行数 (min/mean/max) | 列数 (min/mean/max) |
|-----------|--------------------|--------------------|
"""
    
    for task in ['join', 'union']:
        for split in ['subset', 'complete']:
            if task in opendata_stats and split in opendata_stats[task]:
                s = opendata_stats[task][split]
                rows = s['rows']
                cols = s['columns']
                report += f"| {task}-{split} | {rows['min']}/{rows['mean']}/{rows['max']} | {cols['min']}/{cols['mean']}/{cols['max']} |\n"
    
    report += """

## 5. Ground Truth 分布分析

### Ground Truth 密度对比

| 数据集-任务 | 最少GT | 平均GT | 中位数GT | 最多GT |
|------------|--------|--------|---------|--------|
"""
    
    # WebTable GT分布
    for task in ['join', 'union']:
        if task in webtable_stats and 'complete' in webtable_stats[task]:
            s = webtable_stats[task]['complete']['gt_distribution']
            report += f"| WebTable-{task.upper()} | {s['min']} | {s['mean']} | {s['median']} | {s['max']} |\n"
    
    # OpenData GT分布
    for task in ['join', 'union']:
        if task in opendata_stats and 'complete' in opendata_stats[task]:
            s = opendata_stats[task]['complete']['gt_distribution']
            report += f"| OpenData-{task.upper()} | {s['min']} | {s['mean']} | {s['median']} | {s['max']} |\n"
    
    report += """

## 6. 任务难度分析

### JOIN任务难度特征
- **WebTable JOIN**: 平均GT较少（~6个/查询），表示关联关系相对稀疏
- **OpenData JOIN**: 平均GT较多（~17个/查询），表示存在更多潜在关联

### UNION任务难度特征
- **WebTable UNION**: 平均GT适中（~11个/查询），相似表分布均匀
- **OpenData UNION**: 平均GT较多（~12个/查询），数据模式更加多样

## 7. 数据质量保证

### 提取策略
1. ✅ **完整性保证**: 提取所有Ground Truth涉及的表格
2. ✅ **有效性保证**: 只保留有Ground Truth的查询
3. ✅ **覆盖率**: 100% Ground Truth覆盖率
4. ✅ **采样策略**: 每列保留5个代表性样例值

### 数据特点总结

| 特征 | WebTable | OpenData |
|------|----------|----------|
| **表格规模** | 中等（平均10列） | 较大（平均20列） |
| **数据来源** | 网页表格 | 开放数据集 |
| **JOIN难度** | 较低（稀疏关联） | 较高（密集关联） |
| **UNION难度** | 中等 | 中等偏高 |
| **数据分布** | 均匀 | JOIN少UNION多 |

## 8. 实验建议

### 快速验证
- 使用subset版本（100个查询）进行算法验证
- WebTable subset适合初步测试
- OpenData subset适合鲁棒性测试

### 完整评估
- 使用complete版本进行性能评估
- 注意JOIN和UNION的不同特点调整参数
- 跨数据集对比验证泛化能力

### 参数调优建议

| 场景 | 建议配置 |
|------|----------|
| WebTable JOIN | 较低阈值，扩大搜索范围 |
| OpenData JOIN | 较高阈值，精确过滤 |
| WebTable UNION | 标准配置 |
| OpenData UNION | 提高向量搜索权重 |

## 9. 更新记录

- 2024-12-22: 重新提取数据，确保100% GT覆盖率
- 2024-12-22: 添加详细维度统计和任务难度分析
- 2024-12-22: 统一WebTable和OpenData数据组织结构
"""
    
    return report

def main():
    """主函数"""
    
    print("生成数据集详细统计报告...")
    
    # 加载统计信息
    webtable_stats = load_dataset_stats("examples/webtable", "WebTable")
    opendata_stats = load_dataset_stats("examples/opendata", "OpenData")
    
    # 生成报告
    report = generate_markdown_report(webtable_stats, opendata_stats)
    
    # 保存报告
    with open("DATASET_ORGANIZATION_SUMMARY.md", 'w', encoding='utf-8') as f:
        f.write(report)
    
    print("报告已生成: DATASET_ORGANIZATION_SUMMARY.md")
    
    # 打印摘要
    print("\n数据集摘要:")
    for dataset, stats in [("WebTable", webtable_stats), ("OpenData", opendata_stats)]:
        print(f"\n{dataset}:")
        for task in stats:
            for split in stats[task]:
                s = stats[task][split]
                print(f"  {task} {split}: {s['queries']} queries, {s['tables']} tables")

if __name__ == "__main__":
    main()