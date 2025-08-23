# 数据集组织和统计总结

## 📊 数据集完整统计报告

**生成时间**: 2025-8-22

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
| JOIN | subset | 100 | 195 | 645 | 6.45 |
| JOIN | complete | 1534 | 1534 | 6805 | 4.44 |
| UNION | subset | 100 | 397 | 726 | 7.26 |
| UNION | complete | 5487 | 5487 | 41903 | 7.64 |


### OpenData 数据集

| 任务 | 版本 | 查询数 | 表格数 | Ground Truth | 平均GT/查询 |
|------|------|--------|--------|--------------|-------------|
| JOIN | subset | 100 | 169 | 2534 | 25.34 |
| JOIN | complete | 500 | 500 | 8458 | 16.92 |
| UNION | subset | 100 | 288 | 855 | 8.55 |
| UNION | complete | 3095 | 3095 | 39040 | 12.61 |


## 3. 数据集对比分析

### 总体规模对比

| 指标 | WebTable | OpenData | 比例 |
|------|----------|----------|------|
| 总查询数 | 7,221 | 3,795 | 0.53x |
| 总表格数 | 7,613 | 4,052 | 0.53x |
| 总Ground Truth | 50,079 | 50,887 | 1.02x |


## 4. 数据维度分布

### WebTable 表格维度统计

| 任务-版本 | 行数 (min/mean/max) | 列数 (min/mean/max) |
|-----------|--------------------|--------------------|
| join-subset | 15/107.12/443 | 4/8.33/11 |
| join-complete | 5/77.09/477 | 3/9.51/24 |
| union-subset | 5/98.15/615 | 3/11.77/22 |
| union-complete | 5/61.08/640 | 3/10.24/24 |


### OpenData 表格维度统计

| 任务-版本 | 行数 (min/mean/max) | 列数 (min/mean/max) |
|-----------|--------------------|--------------------|
| join-subset | 554/984.37/1000 | 3/27.4/64 |
| join-complete | 49/982.72/1000 | 3/20.46/195 |
| union-subset | 160/997.08/1000 | 4/14.19/20 |
| union-complete | 5/969.37/1000 | 3/19.85/329 |


## 5. Ground Truth 分布分析

### Ground Truth 密度对比

| 数据集-任务 | 最少GT | 平均GT | 中位数GT | 最多GT |
|------------|--------|--------|---------|--------|
| WebTable-JOIN | 1 | 5.61 | 5.0 | 21 |
| WebTable-UNION | 1 | 8.0 | 7.0 | 29 |
| OpenData-JOIN | 1 | 19.49 | 7.0 | 137 |
| OpenData-UNION | 1 | 12.62 | 12.0 | 42 |


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


