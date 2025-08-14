# 高质量数据集使用说明

## 📁 数据集结构

```
separated_datasets/
├── join/              # JOIN任务核心数据（新创建，高质量）
├── union/             # UNION任务核心数据（新创建，高质量）
├── join_subset/       # JOIN兼容目录（与join/相同）
├── union_subset/      # UNION兼容目录（与union/相同）
└── dataset_statistics.json  # 统计信息
```

## ✅ 数据集质量

- **JOIN**: 77个查询，平均7个候选表/查询
- **UNION**: 100个查询，平均10.5个候选表/查询
- **100%覆盖率**: 所有查询都有ground truth
- **无自匹配**: 过滤了所有无效数据
- **表完整性**: 所有表都存在于数据集中

## 🚀 使用方法

```bash
# 测试JOIN任务
python run_cached_experiments.py --task join --dataset subset --max-queries 20

# 测试UNION任务
python run_cached_experiments.py --task union --dataset subset --max-queries 20

# 同时测试两个任务
python run_cached_experiments.py --task both --dataset subset --max-queries 10
```
