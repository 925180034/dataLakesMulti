# 数据集组织和统计总结

## 📊 数据集完整统计报告

**更新时间**: 2025-8-26 (增加NLCTables数据集)

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
├── opendata/               # OpenData数据集
│   ├── join_subset/        # JOIN任务子集
│   ├── join_complete/      # JOIN任务完整
│   ├── union_subset/       # UNION任务子集
│   └── union_complete/     # UNION任务完整
│
└── nlctables/              # NLCTables数据集 (2025-8新增)
    ├── join_subset/        # JOIN任务子集 (精简版)
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


### NLCTables 数据集

| 任务 | 版本 | 查询数 | 表格数 | Ground Truth | 平均GT/查询 |
|------|------|--------|--------|--------------|-------------|
| JOIN | subset | 18 | 60 | 42 | 2.33 |
| JOIN | complete | 91 | 4821 | 211 | 2.32 |
| UNION | subset | 100 | 908 | 308 | 3.08 |
| UNION | complete | 255 | 3255 | 815 | 3.20 |


## 3. 数据集对比分析

### 总体规模对比

| 指标 | WebTable | OpenData | NLCTables | 
|------|----------|----------|-----------|
| 总查询数 | 7,221 | 3,795 | 464 |
| 总表格数 | 7,613 | 4,052 | 9,044 |
| 总Ground Truth | 50,079 | 50,887 | 1,376 |
| 平均GT/查询 | 6.93 | 13.41 | 2.97 |


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


### NLCTables 表格维度统计

| 任务-版本 | 行数 (min/mean/max) | 列数 (min/mean/max) |
|-----------|--------------------|--------------------|
| join-subset | 20/1500/10000 | 3/7.2/18 |
| join-complete | 20/3543.9/58900 | 3/9.4/28 |
| union-subset | 10/150/1000 | 3/8.5/20 |
| union-complete | 10/104.5/1000 | 3/13.5/25 |


## 5. Ground Truth 分布分析

### Ground Truth 密度对比

| 数据集-任务 | 最少GT | 平均GT | 中位数GT | 最多GT |
|------------|--------|--------|---------|--------|
| WebTable-JOIN | 1 | 5.61 | 5.0 | 21 |
| WebTable-UNION | 1 | 8.0 | 7.0 | 29 |
| OpenData-JOIN | 1 | 19.49 | 7.0 | 137 |
| OpenData-UNION | 1 | 12.62 | 12.0 | 42 |
| NLCTables-JOIN | 1 | 2.32 | 3.0 | 3 |
| NLCTables-UNION | 1 | 3.19 | 3.0 | 10 |


## 6. 任务难度分析

### JOIN任务难度特征
- **NLCTables JOIN**: 平均GT最少（~2.3个/查询），高精度匹配，关联关系明确
- **WebTable JOIN**: 平均GT较少（~6个/查询），表示关联关系相对稀疏
- **OpenData JOIN**: 平均GT较多（~17个/查询），表示存在更多潜在关联

### UNION任务难度特征  
- **NLCTables UNION**: 平均GT最少（~3个/查询），相似表数量较少，匹配精确
- **WebTable UNION**: 平均GT适中（~8个/查询），相似表分布均匀
- **OpenData UNION**: 平均GT较多（~12个/查询），数据模式更加多样

## 7. 数据质量保证

### 提取策略
1. ✅ **完整性保证**: 提取所有Ground Truth涉及的表格
2. ✅ **有效性保证**: 只保留有Ground Truth的查询
3. ✅ **覆盖率**: 100% Ground Truth覆盖率
4. ✅ **采样策略**: 每列保留5个代表性样例值

### 数据特点总结

| 特征 | WebTable | OpenData | NLCTables |
|------|----------|----------|-----------|
| **表格规模** | 中等（平均10列，77行） | 较大（平均20列，970行） | 中等（平均11列，2000+行） |
| **数据来源** | 网页表格 | 开放数据集 | 自然语言查询表格 |
| **JOIN难度** | 较低（稀疏关联） | 较高（密集关联） | 最低（精确关联） |
| **UNION难度** | 中等 | 中等偏高 | 较低 |
| **数据分布** | 均匀 | JOIN少UNION多 | UNION多JOIN少 |
| **行数范围** | 5-640 | 5-1000 | 10-58900 |
| **特点** | 规模最大 | GT最密集 | 行数最多，GT最精确 |

## 8. 实验建议

### 快速验证
- 使用subset版本（100个查询）进行算法验证
- NLCTables subset适合精度测试（最少GT，最精确）
- WebTable subset适合初步测试（规模适中）
- OpenData subset适合鲁棒性测试（GT最多）

### 完整评估
- 使用complete版本进行性能评估
- 注意JOIN和UNION的不同特点调整参数
- 跨数据集对比验证泛化能力
- NLCTables适合测试高精度匹配场景

### 参数调优建议

| 场景 | 建议配置 |
|------|----------|
| NLCTables JOIN | 极低阈值（0.10），精确匹配模式 |
| NLCTables UNION | 低阈值（0.15），小范围搜索 |
| WebTable JOIN | 较低阈值（0.20），扩大搜索范围 |
| WebTable UNION | 标准配置（0.25） |
| OpenData JOIN | 较高阈值（0.30），精确过滤 |
| OpenData UNION | 提高向量搜索权重，增大候选集 |


## 9. NLCTables数据集特点

### 数据集定位
NLCTables是一个专注于自然语言查询的表格匹配数据集，具有以下特点：

1. **最精确的Ground Truth**: 平均每个查询只有2-3个匹配表，适合测试高精度算法
2. **行数变化最大**: 从10行到58,900行，平均3500+行（JOIN），测试系统的可扩展性
3. **中等列数**: 平均9-13列，与实际应用场景接近
4. **查询数量较少**: JOIN只有18个subset查询，需要合理设计实验
5. **表格数量充足**: Complete版本有4821个JOIN表和3255个UNION表

### 适用场景
- **高精度测试**: 测试算法在精确匹配场景下的表现
- **大规模表格处理**: 测试系统处理大表格（>50K行）的能力
- **参数下限测试**: 测试系统在最低阈值下的表现
- **消融实验**: GT数量少，适合快速验证算法效果

### 数据特征对比
- **行数规模**: NLCTables > OpenData > WebTable
- **列数规模**: OpenData > NLCTables > WebTable  
- **GT密度**: OpenData > WebTable > NLCTables
- **GT精度**: NLCTables > WebTable > OpenData

### 注意事项
- JOIN subset只有18个查询，统计结果可能不够稳定
- Ground Truth数量少，需要调低阈值避免漏检
- 表格行数差异极大（10-58900），需要考虑内存和性能优化
- 原始数据包含metadata（标题、页面信息等），处理时需要提取data字段

