# NLCTables多智能体系统适配方案

## 📋 执行摘要

本文档详细说明如何将现有的多智能体数据湖发现系统扩展以支持NLCTables（自然语言条件表）数据集。由于系统采用多智能体架构，每个Agent都有独立的API调用和提示词，理论上通过适配各Agent即可实现对NLCTables的支持，无需重构整个系统。

## 🎯 目标与挑战

### 当前系统能力
- ✅ **WebTables**: 基于表结构相似性的匹配（列名、数据类型、样本值）
- ✅ **OpenData**: 同样基于结构化匹配
- ❌ **NLCTables**: 需要自然语言理解能力

### 核心差异对比

| 特征 | WebTables/OpenData | NLCTables |
|------|-------------------|-----------|
| **输入类型** | 查询表（结构化） | 自然语言描述 |
| **匹配依据** | 表结构相似性 | 语义条件满足 |
| **搜索方式** | 列名/类型匹配 | 关键词/主题搜索 |
| **示例** | "找与users表结构相似的表" | "找包含keyword X且topic是Y的表" |

### NLCTables查询结构示例
```json
{
    "query_id": "nlc_join_1",
    "query_text": "Can you find more joinable tables with keyword 0.363with the topic related to Sports_Team_Statistics",
    "seed_table": "q_table_118_j1_3",  // 注：此表不存在于实际表列表中
    "task_type": "join",
    "features": {
        "keywords": ["0.363with"],
        "topics": ["Sports_Team_Statistics"],
        "column_mentions": [],
        "value_mentions": [],
        "constraints": {}
    }
}
```

## 🏗️ 系统架构分析

### 现有多智能体架构
```
用户查询
    ↓
多智能体协同系统
├── OptimizerAgent: 系统优化配置
├── PlannerAgent: 策略规划决策
├── AnalyzerAgent: 数据结构分析
├── SearcherAgent: 候选搜索（Layer1+Layer2）
├── MatcherAgent: 精确匹配（Layer3）
└── AggregatorAgent: 结果聚合排序
    ↓
三层加速工具（Agent按需调用）
├── Layer 1: MetadataFilter (规则筛选)
├── Layer 2: VectorSearch (向量搜索)
└── Layer 3: SmartLLMMatcher (LLM验证)
```

### 适配优势
1. **模块化设计**: 每个Agent独立运作，可单独修改
2. **API独立性**: 每个Agent有自己的提示词和处理逻辑
3. **工具复用**: 三层加速工具可以适配新的搜索策略
4. **最小侵入**: 不影响现有WebTables/OpenData功能

## 🔧 Agent适配方案

### 1. PlannerAgent适配
**现状**: 根据任务类型选择处理策略
**适配内容**:
```python
def plan_strategy(query):
    # 新增：识别NLCTables查询
    if query.get('query_id', '').startswith('nlc_') or 'query_text' in query:
        return {
            'strategy': 'natural_language',
            'use_features': True,
            'search_mode': 'semantic'
        }
    # 原有逻辑保持不变
    elif 'query_table' in query:
        return {
            'strategy': 'structure_match',
            'use_features': False,
            'search_mode': 'structural'
        }
```

**提示词修改**:
```
原提示词：
"根据查询表的结构特征，选择最佳的搜索策略..."

新增提示词：
"如果查询包含自然语言描述（query_text），请：
1. 识别关键词(keywords)和主题(topics)
2. 选择语义搜索策略
3. 优先考虑内容匹配而非结构匹配"
```

### 2. AnalyzerAgent适配
**现状**: 分析表结构（列名、类型、样本值）
**适配内容**:
```python
def analyze_query(query):
    if 'features' in query:  # NLCTables
        return {
            'search_criteria': {
                'keywords': query['features'].get('keywords', []),
                'topics': query['features'].get('topics', []),
                'column_mentions': query['features'].get('column_mentions', [])
            },
            'search_type': 'content_based'
        }
    else:  # WebTables/OpenData
        return analyze_table_structure(query['query_table'])
```

**提示词修改**:
```
新增提示词：
"对于自然语言查询，请分析：
1. 关键词的重要性和搜索优先级
2. 主题词与表内容的相关性
3. 列名提及与实际列的映射关系
输出结构化的搜索条件供后续Agent使用"
```

### 3. SearcherAgent适配
**现状**: 基于结构相似性搜索候选表
**适配内容**:

#### Layer 1 (MetadataFilter) 适配
```python
def filter_l1(query, tables):
    if query.get('search_type') == 'content_based':
        keywords = query['search_criteria']['keywords']
        topics = query['search_criteria']['topics']
        
        candidates = []
        for table in tables:
            score = 0
            # 关键词匹配
            table_text = ' '.join([
                table.get('name', ''),
                ' '.join([col['name'] for col in table.get('columns', [])]),
                ' '.join([str(v) for col in table.get('columns', []) 
                         for v in col.get('sample_values', [])])
            ])
            
            for keyword in keywords:
                if keyword.lower() in table_text.lower():
                    score += 1
            
            # 主题匹配（可以通过表名或预定义的主题映射）
            for topic in topics:
                if topic.lower() in table.get('name', '').lower():
                    score += 0.5
            
            if score > 0:
                candidates.append((table['name'], score))
        
        return sorted(candidates, key=lambda x: x[1], reverse=True)[:10]
```

#### Layer 2 (VectorSearch) 适配
```python
def search_l2(query, vector_index):
    if query.get('search_type') == 'content_based':
        # 将query_text编码为向量
        query_text = query.get('query_text', '')
        if not query_text:
            # 从features构建查询文本
            features = query.get('search_criteria', {})
            query_text = ' '.join([
                ' '.join(features.get('keywords', [])),
                ' '.join(features.get('topics', [])),
                ' '.join(features.get('column_mentions', []))
            ])
        
        # 使用语义向量搜索
        query_vector = encode_text(query_text)
        candidates = vector_index.search_semantic(query_vector, top_k=20)
        return candidates
```

**向量化策略改进**:
```python
def create_table_embedding(table, mode='hybrid'):
    if mode == 'structure':  # WebTables模式
        # 基于结构信息
        text = ' '.join([
            table['name'],
            ' '.join([col['name'] + ' ' + col['type'] 
                     for col in table['columns']])
        ])
    elif mode == 'content':  # NLCTables模式
        # 基于内容信息
        text = ' '.join([
            table['name'],
            ' '.join([col['name'] for col in table['columns']]),
            ' '.join([str(v) for col in table['columns'] 
                     for v in col.get('sample_values', [])[:5]])
        ])
    else:  # hybrid模式
        # 结合两者
        structure_text = ...
        content_text = ...
        text = structure_text + ' ' + content_text
    
    return encode_text(text)
```

### 4. MatcherAgent适配
**现状**: 验证表是否可连接（基于结构）
**适配内容**:

**提示词完全重写（针对NLCTables）**:
```
当处理自然语言条件查询时：

输入：
- 查询文本：{query_text}
- 关键词：{keywords}
- 主题：{topics}
- 候选表：{candidate_table}

任务：
判断候选表是否满足查询中描述的条件。

评估标准：
1. 关键词覆盖度（0-40分）
   - 表中包含的关键词数量
   - 关键词在表中的位置（列名>样本值）

2. 主题相关性（0-30分）
   - 表名与主题的相关性
   - 表内容与主题的语义相似度

3. 整体语义匹配（0-30分）
   - 查询描述与表内容的整体相关性
   - 是否符合用户的查询意图

输出格式：
{
    "is_match": true/false,
    "confidence": 0.0-1.0,
    "reasoning": "匹配理由",
    "score_breakdown": {
        "keyword_score": 0-40,
        "topic_score": 0-30,
        "semantic_score": 0-30
    }
}
```

### 5. AggregatorAgent适配
**现状**: 聚合并排序结果
**适配内容**: 基本不需要修改，但可以优化排序策略

```python
def aggregate_results(results, query_type):
    if query_type == 'natural_language':
        # 对NLCTables结果，更重视语义匹配分数
        weight_config = {
            'l1_score': 0.2,  # 关键词匹配
            'l2_score': 0.4,  # 语义相似性
            'l3_score': 0.4   # LLM验证
        }
    else:
        # WebTables保持原权重
        weight_config = {
            'l1_score': 0.3,
            'l2_score': 0.3,
            'l3_score': 0.4
        }
```

## 📊 实现路线图

### 第一阶段：基础支持（2天）
**目标**: 实现基本的NLCTables查询处理

**任务清单**:
- [ ] 修改PlannerAgent识别NLCTables查询
- [ ] 修改AnalyzerAgent解析features字段
- [ ] 实现基于keywords的简单L1搜索
- [ ] 调整数据加载逻辑处理NLCTables格式

**预期效果**:
- 能够处理NLCTables查询
- 基于关键词匹配返回候选
- 准确率: 15-25%

### 第二阶段：语义增强（3天）
**目标**: 添加语义理解能力

**任务清单**:
- [ ] 实现query_text的向量化
- [ ] 创建内容感知的表嵌入
- [ ] 修改L2层支持语义搜索
- [ ] 优化向量索引结构

**预期效果**:
- 支持语义相似性搜索
- 更好的召回率
- 准确率: 35-45%

### 第三阶段：智能匹配（3天）
**目标**: 利用LLM的自然语言理解能力

**任务清单**:
- [ ] 重写MatcherAgent的NLCTables提示词
- [ ] 实现条件验证逻辑
- [ ] 优化评分和聚合策略
- [ ] 添加解释性输出

**预期效果**:
- 精确的条件匹配
- 可解释的匹配结果
- 准确率: 55-65%

### 第四阶段：优化调优（2天）
**目标**: 性能优化和准确率提升

**任务清单**:
- [ ] 参数调优（权重、阈值）
- [ ] 缓存优化
- [ ] 批处理优化
- [ ] A/B测试不同策略

**预期效果**:
- 稳定的性能表现
- 准确率: 65-75%

## 💻 具体实现示例

### 1. 查询路由器实现
```python
# src/agents/planner_agent.py 修改

class PlannerAgent(BaseAgent):
    def identify_query_type(self, query: Dict) -> str:
        """识别查询类型"""
        # NLCTables识别
        if query.get('query_id', '').startswith('nlc_'):
            return 'nlctables'
        elif 'query_text' in query and 'features' in query:
            return 'nlctables'
        # WebTables/OpenData识别
        elif 'query_table' in query:
            return 'webtables'
        else:
            return 'unknown'
    
    def plan_execution(self, query: Dict) -> Dict:
        query_type = self.identify_query_type(query)
        
        if query_type == 'nlctables':
            return {
                'strategy': 'natural_language_search',
                'agents_sequence': [
                    'AnalyzerAgent',  # 解析NL条件
                    'SearcherAgent',  # 语义搜索
                    'MatcherAgent',   # 条件验证
                    'AggregatorAgent' # 结果聚合
                ],
                'search_config': {
                    'use_semantic': True,
                    'use_keywords': True,
                    'use_structure': False
                }
            }
        else:
            # 原有WebTables逻辑
            return self.plan_webtables_execution(query)
```

### 2. NL条件解析器
```python
# src/agents/analyzer_agent.py 添加

class AnalyzerAgent(BaseAgent):
    def analyze_nl_query(self, query: Dict) -> Dict:
        """分析自然语言查询"""
        features = query.get('features', {})
        query_text = query.get('query_text', '')
        
        # 使用LLM增强理解（可选）
        if self.use_llm_enhancement:
            enhanced_features = self.llm_enhance_features(query_text, features)
        else:
            enhanced_features = features
        
        return {
            'search_criteria': {
                'must_have_keywords': enhanced_features.get('keywords', []),
                'should_have_topics': enhanced_features.get('topics', []),
                'column_hints': enhanced_features.get('column_mentions', []),
                'value_patterns': enhanced_features.get('value_mentions', [])
            },
            'search_strategy': self.determine_search_strategy(enhanced_features),
            'confidence_threshold': self.calculate_threshold(enhanced_features)
        }
    
    def llm_enhance_features(self, query_text: str, features: Dict) -> Dict:
        """使用LLM增强特征理解"""
        prompt = f"""
        查询文本：{query_text}
        已识别特征：{features}
        
        请分析这个查询的真实意图，并增强特征：
        1. 扩展同义词和相关词
        2. 识别隐含的约束条件
        3. 推断可能的表结构特征
        """
        
        response = self.llm_client.generate(prompt)
        return self.parse_llm_response(response)
```

### 3. 语义搜索实现
```python
# src/tools/semantic_search.py 新增

class SemanticSearchTool:
    def __init__(self, embedding_model='sentence-transformers/all-MiniLM-L6-v2'):
        self.encoder = SentenceTransformer(embedding_model)
        self.index = None
        
    def build_semantic_index(self, tables: List[Dict], mode='content'):
        """构建语义索引"""
        embeddings = []
        for table in tables:
            if mode == 'content':
                # 基于内容的嵌入
                text = self.extract_content_text(table)
            else:
                # 基于结构的嵌入
                text = self.extract_structure_text(table)
            
            embedding = self.encoder.encode(text)
            embeddings.append(embedding)
        
        # 构建FAISS索引
        self.index = faiss.IndexFlatIP(embeddings[0].shape[0])
        self.index.add(np.array(embeddings))
        
    def search(self, query_text: str, top_k: int = 20) -> List[Tuple[int, float]]:
        """语义搜索"""
        query_embedding = self.encoder.encode(query_text)
        distances, indices = self.index.search(
            query_embedding.reshape(1, -1), 
            top_k
        )
        return list(zip(indices[0], distances[0]))
    
    def extract_content_text(self, table: Dict) -> str:
        """提取表的内容文本用于语义编码"""
        parts = [
            table.get('name', ''),
            ' '.join([col['name'] for col in table.get('columns', [])]),
            ' '.join([
                str(val) 
                for col in table.get('columns', [])
                for val in col.get('sample_values', [])[:3]
            ])
        ]
        return ' '.join(parts)
```

## 📈 评估指标

### 保持统一的评估体系
- **Hit@K**: Top-K候选中包含正确答案的比例
- **Precision**: 返回结果的准确率
- **Recall**: 正确答案的召回率
- **F1-Score**: 精确率和召回率的调和平均

### NLCTables特定指标
- **语义相关性**: 查询与结果的语义相似度
- **条件满足度**: 满足的条件数/总条件数
- **关键词覆盖率**: 匹配的关键词数/总关键词数

### 分阶段目标
| 阶段 | Hit@1 | Hit@5 | F1-Score | 实现难度 |
|------|-------|-------|----------|---------|
| 基础支持 | 10% | 25% | 0.15 | 低 |
| 语义增强 | 25% | 45% | 0.35 | 中 |
| 智能匹配 | 40% | 65% | 0.55 | 高 |
| 优化调优 | 50% | 75% | 0.65 | 中 |

## 🚀 实施建议

### 1. 优先级建议
1. **高优先级**: PlannerAgent和AnalyzerAgent的适配（基础功能）
2. **中优先级**: SearcherAgent的L1/L2层适配（提升召回）
3. **低优先级**: MatcherAgent的精细化调优（提升精确率）

### 2. 风险与缓解
| 风险 | 影响 | 缓解策略 |
|------|------|---------|
| LLM理解偏差 | 准确率低 | 提供更详细的提示词和示例 |
| 性能下降 | 响应时间增加 | 实现缓存和批处理优化 |
| 向量质量 | 语义搜索效果差 | 尝试不同的嵌入模型 |
| 兼容性问题 | 影响现有功能 | 严格的查询类型隔离 |

### 3. 测试策略
```python
# 测试脚本示例
def test_nlctables_support():
    # 1. 单元测试：各Agent独立功能
    test_planner_routing()
    test_analyzer_parsing()
    test_searcher_semantic()
    
    # 2. 集成测试：端到端流程
    test_nlctables_pipeline()
    test_webtables_compatibility()  # 确保不影响现有功能
    
    # 3. 性能测试
    test_query_latency()
    test_throughput()
    
    # 4. 准确性测试
    test_accuracy_metrics()
```

## 💡 创新机会

### 1. 混合查询支持
支持同时包含结构和语义条件的查询：
```json
{
    "structure_constraint": "包含user_id列",
    "semantic_constraint": "关于体育统计的数据"
}
```

### 2. 交互式优化
让系统从用户反馈中学习：
- 记录查询和用户选择
- 优化Agent提示词
- 调整权重参数

### 3. 多语言支持
由于NLCTables基于自然语言，可以扩展到支持中文等其他语言的查询。

## 📝 总结

通过充分利用现有的多智能体架构，我们可以以最小的改动实现对NLCTables的支持。关键在于：

1. **识别并路由**：PlannerAgent识别查询类型
2. **理解并转换**：AnalyzerAgent将NL条件转为搜索指令
3. **搜索并匹配**：SearcherAgent和MatcherAgent执行语义搜索
4. **保持兼容**：不影响现有WebTables/OpenData功能

这种渐进式的实现方案允许我们逐步提升系统能力，同时保持系统的稳定性和可维护性。

## 🔗 相关文档
- [NLCTables数据集分析](./NLCTABLES_DATASET_ANALYSIS.md)
- [多智能体系统架构](../CLAUDE.md)
- [三层加速工具说明](./SYSTEM_ARCHITECTURE_AND_PLAN.md)