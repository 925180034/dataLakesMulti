# NLCTables 实验指南（基于论文）

## 📚 论文核心要点

### 数据集创新点
**NLCTables** 是首个结合**查询表格**和**自然语言条件**的表格发现数据集，定义了新的任务场景：
- **传统方法**：仅使用关键词或查询表格
- **nlcTD创新**：查询表格 + 自然语言需求 → 更精确的表格发现

### 数据集规模（论文 Table 2）
```
Dataset Type    Queries  Tables   GT      Pos:Neg Ratio
nlcTables_K     235      7,405    6,841   1:8.6
nlcTables-U     255      7,567    7,411   1:10.7
nlcTables-J     91       4,871    4,821   1:21.8
nlcTables-U-fz  39       1,620    1,560   1:11.7
nlcTables-J-fz  27       617      567     1:7.5
Total           647      22,080   21,200  -
```

## 🔬 实验方法论（基于论文 Section 5）

### 1. 评估指标

论文使用三个标准指标（Section 5.1）：

```python
def calculate_metrics(retrieved_tables, ground_truth, k):
    """
    计算 Precision@k, Recall@k, NDCG@k
    基于论文公式（第7页）
    """
    # Precision@k = |T_g ∩ T'| / |T'|
    precision = len(set(retrieved_tables[:k]) & set(ground_truth)) / k
    
    # Recall@k = |T_g ∩ T'| / |T_g|
    recall = len(set(retrieved_tables[:k]) & set(ground_truth)) / len(ground_truth)
    
    # NDCG@k = 1/Z_k * Σ(ρ_i / log2(i+1))
    dcg = sum(relevance_score / np.log2(i+2) 
              for i, relevance_score in enumerate(relevance_scores[:k]))
    idcg = calculate_ideal_dcg(ground_truth_scores, k)
    ndcg = dcg / idcg if idcg > 0 else 0
    
    return precision, recall, ndcg
```

### 2. 基线方法对比

论文测试了6个代表性方法（Section 5.1）：

| 类型 | 方法 | 适用任务 | 论文结果 |
|------|------|----------|----------|
| Keyword-based | GTR (SIGIR'21) | K, U, J | 整体表现较好 |
| Keyword-based | StruBERT (WWW'22) | K, U, J | 长句查询表现差 |
| Union Search | Santos (SIGMOD'23) | U | 召回高但精度低 |
| Union Search | Starmie (VLDB'23) | U | 语义敏感性好 |
| Join Search | Josie (SIGMOD'19) | J | 极低召回率 |
| Join Search | DeepJoin (VLDB'23) | J | 语义匹配差 |

### 3. 实验设置建议

基于论文 Section 5 的实验设计：

```python
class NLCTablesExperiment:
    def __init__(self):
        self.metrics_k = [5, 10, 15, 20]  # 论文使用的k值
        self.dataset_types = ['K', 'U', 'J', 'U-fz', 'J-fz']
        
    def run_experiments(self, method, dataset_type):
        """
        复现论文实验流程
        """
        # 1. 加载数据集
        queries, tables, ground_truth = self.load_nlctables(dataset_type)
        
        # 2. 对每个查询进行检索
        results = {}
        for query in queries:
            # 提取查询表格和NL条件
            query_table = query['query_table']
            nl_condition = query['nl_condition']
            
            # 执行检索
            retrieved = method.search(query_table, nl_condition, tables)
            
            # 计算指标
            for k in self.metrics_k:
                metrics = calculate_metrics(retrieved, ground_truth[query['id']], k)
                results[f"{query['id']}_k{k}"] = metrics
        
        return results
```

## 🎯 关键实验发现（论文 Section 5.2-5.4）

### RQ1: 整体性能分析
- **发现**：现有方法在nlcTD任务上表现很差
- **原因**：无法同时处理查询表格和自然语言条件
- **建议**：需要开发新的融合方法

### RQ2: 条件类型分析（Table 4）
不同NL条件类型的性能差异：

| 条件类型 | 性能 | 原因 |
|----------|------|------|
| Table Topic | 最好 | 与关键词对齐 |
| Table Size | 中等 | 容易量化 |
| Categorical | 中等 | 类别匹配简单 |
| String | 较差 | 语义理解需求 |
| Numerical | 差 | 阈值判断困难 |
| Date | 最差 | 时间推理复杂 |
| Mixed-mode | 极差 | 多条件组合 |

### RQ3: 数据集组成影响
1. **正负样本比例**：负样本增加导致性能下降
2. **数据规模**：大规模时需要索引结构
3. **模糊查询**：语义增强后性能下降明显

## 🚀 实验流程建议

### Phase 1: 基线评估（2-3天）

```bash
# 1. 数据准备
python convert_nlctables.py \
    --input /root/autodl-tmp/datalakes/nlcTables \
    --output examples/nlctables

# 2. 基线测试（复现论文 Figure 5）
python run_baseline_experiment.py \
    --dataset nlctables \
    --methods all \
    --metrics P@k,R@k,NDCG@k \
    --k 5,10,15,20
```

### Phase 2: 系统适配（3-5天）

针对你的三层架构系统的适配策略：

```python
class NLCTablesAdapter:
    """将nlcTables适配到三层架构"""
    
    def adapt_for_l1_metadata(self, nl_condition):
        """Layer 1: 元数据过滤适配"""
        # 提取表级条件
        table_conditions = extract_table_level_conditions(nl_condition)
        # 转换为元数据过滤规则
        return {
            'topic': table_conditions.get('topic'),
            'min_rows': table_conditions.get('size', {}).get('min_rows'),
            'min_cols': table_conditions.get('size', {}).get('min_cols')
        }
    
    def adapt_for_l2_vector(self, query_table, nl_condition):
        """Layer 2: 向量搜索适配"""
        # 组合查询表格和NL条件的嵌入
        table_embedding = self.encode_table(query_table)
        condition_embedding = self.encode_nl(nl_condition)
        # 加权组合
        combined = 0.6 * table_embedding + 0.4 * condition_embedding
        return combined
    
    def adapt_for_l3_llm(self, query_table, nl_condition, candidates):
        """Layer 3: LLM验证适配"""
        prompt = f"""
        Given query table: {query_table}
        NL condition: {nl_condition}
        
        Rank these candidate tables based on:
        1. Structural compatibility (union/join)
        2. Satisfaction of NL conditions
        3. Semantic relevance
        """
        return self.llm_rerank(prompt, candidates)
```

### Phase 3: 条件类型优化（5-7天）

基于论文 Table 4 的发现，针对性优化：

```python
class ConditionTypeOptimizer:
    """针对不同条件类型的优化策略"""
    
    def optimize_numerical_conditions(self, condition):
        """优化数值条件处理"""
        # 论文发现：数值条件性能差
        # 策略：增强阈值判断逻辑
        threshold = extract_numerical_threshold(condition)
        return {
            'operator': threshold['op'],  # >, <, =
            'value': threshold['value'],
            'column': threshold['column']
        }
    
    def optimize_mixed_mode(self, conditions):
        """优化混合模式条件"""
        # 论文发现：混合条件极具挑战性
        # 策略：分解为子条件，分别处理后聚合
        sub_conditions = decompose_conditions(conditions)
        sub_results = [self.process_single(c) for c in sub_conditions]
        return aggregate_results(sub_results)
```

### Phase 4: 消融实验（3-4天）

复现论文的消融研究：

```python
# 1. 模糊查询影响（Figure 6）
python ablation_fuzzy.py --compare original,fuzzy

# 2. 正负样本比例影响（Figure 7）
python ablation_ratio.py --ratios 1:3,1:6,1:12

# 3. 数据规模影响（Table 5）
python ablation_scale.py --scales small,medium,large
```

## 📊 预期结果与改进方向

### 基于论文的性能预期

| 数据集 | 方法类型 | NDCG@10预期 | 你的系统目标 |
|--------|----------|-------------|--------------|
| nlcTables-K | Keyword | 0.45-0.55 | 0.60+ |
| nlcTables-U | Union | 0.35-0.45 | 0.55+ |
| nlcTables-J | Join | 0.30-0.40 | 0.50+ |

### 论文指出的改进方向

1. **语义理解增强**：
   - 现有方法忽略NL条件语义
   - 建议：使用大语言模型理解复杂条件

2. **多模态融合**：
   - 查询表格和NL条件的有效结合
   - 建议：设计联合编码器

3. **条件分解策略**：
   - 混合条件处理困难
   - 建议：层次化条件处理

4. **列类型感知**：
   - 数值/日期列被忽略
   - 建议：类型特定的处理模块

## 🔧 实验脚本模板

```bash
#!/bin/bash
# nlctables_experiment.sh

# 环境准备
export DATASET_PATH=/root/autodl-tmp/datalakes/nlcTables
export OUTPUT_DIR=experiment_results/nlctables

# 实验1: 整体性能评估（复现Figure 5）
python three_layer_ablation_optimized.py \
    --dataset $DATASET_PATH/nlcTables-U \
    --task union \
    --layers L1+L2+L3 \
    --metrics P@5,P@10,P@15,P@20,R@5,R@10,R@15,R@20,NDCG@5,NDCG@10,NDCG@15,NDCG@20 \
    --output $OUTPUT_DIR/overall_performance.json

# 实验2: 条件类型分析（复现Table 4）
for condition_type in topic size categorical string numerical date mixed; do
    python condition_type_analysis.py \
        --dataset $DATASET_PATH \
        --condition_type $condition_type \
        --output $OUTPUT_DIR/condition_${condition_type}.json
done

# 实验3: 数据组成研究（复现Section 5.4）
python dataset_composition_study.py \
    --dataset $DATASET_PATH \
    --experiments fuzzy,ratio,scale \
    --output $OUTPUT_DIR/composition_study.json

# 结果汇总
python summarize_nlctables_results.py \
    --input $OUTPUT_DIR \
    --output nlctables_report.pdf
```

## 📈 论文关键洞察总结

1. **nlcTD是一个全新且具有挑战性的任务**
   - 需要同时理解表格结构和自然语言语义
   - 现有方法都存在显著不足

2. **数据集设计合理且全面**
   - 覆盖多种查询类型和条件类别
   - 提供了丰富的评估维度

3. **未来研究方向明确**
   - 语义理解是关键
   - 需要专门为nlcTD设计的新方法

4. **实验设置严谨**
   - 多维度评估（整体性能、条件类型、数据组成）
   - 提供了clear baseline和改进空间

## 🎯 下一步行动建议

1. **立即开始**：数据转换和基线测试（1-2天）
2. **系统适配**：将NL条件集成到三层架构（3-5天）
3. **针对性优化**：基于论文发现改进弱点（5-7天）
4. **论文撰写**：对比论文结果，展示改进（2-3天）

论文GitHub: https://github.com/SuDIS-ZJU/nlcTables