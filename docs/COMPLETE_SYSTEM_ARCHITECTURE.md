# 数据湖多智能体发现系统完整架构文档

## 目录
1. [系统概述](#系统概述)
2. [核心架构设计](#核心架构设计)
3. [多智能体系统详解](#多智能体系统详解)
4. [三层加速架构](#三层加速架构)
5. [技术实现细节](#技术实现细节)
6. [性能优化策略](#性能优化策略)
7. [系统部署与扩展](#系统部署与扩展)

---

## 1. 系统概述

### 1.1 项目定位
**数据湖多智能体发现系统**是一个创新的数据湖探索平台，专门解决大规模数据湖中的数据发现和关联分析问题。系统结合了**多智能体协同决策**和**三层加速架构**，实现高效、智能的数据湖发现。

### 1.2 核心能力
- **数据湖发现**: 自动发现相关表、可连接数据、相似数据集
- **智能决策**: 6个专门Agent协同工作
- **性能加速**: 三层递进式优化（<10ms → 10-50ms → 1-3s）
- **灵活扩展**: 支持新Agent和新策略

### 1.3 技术栈
```yaml
语言: Python 3.10+
框架: LangGraph, LangChain
LLM: Gemini 1.5 (主要) / OpenAI / Anthropic
向量数据库: FAISS (HNSW索引)
嵌入模型: Sentence-Transformers (all-MiniLM-L6-v2)
并发: AsyncIO + aiohttp
缓存: 三级缓存（内存/Redis/磁盘）
```

### 1.4 系统规模
- **数据规模**: 1,534个表，~7,000列
- **响应时间**: 5-10秒（端到端）
- **并发能力**: 10个并发查询
- **准确率**: Hit@10 36% (complete) / 44% (subset)

---

## 2. 核心架构设计

### 2.1 整体架构

系统采用**分层架构**设计，包含以下核心层次：

1. **接口层**: CLI、REST API、Python SDK
2. **协调层**: Orchestrator总协调器
3. **智能体层**: 6个专门Agent协同工作
4. **加速层**: 三层递进式筛选优化
5. **数据层**: 数据湖、索引、缓存

### 2.2 关键创新

#### 2.2.1 多智能体协同
- **分工明确**: 每个Agent专注特定任务
- **消息驱动**: 基于消息总线的松耦合通信
- **并行执行**: 独立任务并行处理

#### 2.2.2 三层加速架构
- **Layer 1**: 规则筛选，快速过滤（<10ms）
- **Layer 2**: 向量搜索，语义匹配（10-50ms）
- **Layer 3**: LLM验证，精确判断（1-3s）

#### 2.2.3 异步并发优化
- **并行LLM调用**: 从串行72s优化到并行3.6s
- **异步HTTP客户端**: 使用aiohttp替代requests
- **协程池管理**: 动态调整并发度

---

## 3. 多智能体系统详解

### 3.1 Agent角色定义

#### 3.1.1 OptimizerAgent（优化器）
```python
class OptimizerAgent:
    """系统优化配置Agent"""
    
    职责:
    - 动态调整系统参数
    - 选择最优并行度（1-20）
    - 配置缓存策略
    - 资源分配优化
    
    决策逻辑:
    - if query_complexity > 0.8: workers = 16
    - if data_size > 1000: enable_cache = True
    - if latency > 10s: increase_parallelism()
```

#### 3.1.2 PlannerAgent（规划器）
```python
class PlannerAgent:
    """策略规划Agent"""
    
    职责:
    - 选择执行策略（Bottom-Up/Top-Down）
    - 制定执行计划
    - 任务分解与调度
    
    策略选择:
    - JOIN任务 → Bottom-Up（列匹配优先）
    - UNION任务 → Top-Down（表相似优先）
    - 混合任务 → Hybrid（自适应）
```

#### 3.1.3 AnalyzerAgent（分析器）
```python
class AnalyzerAgent:
    """数据分析Agent"""
    
    职责:
    - 提取表结构特征
    - 识别关键列
    - 计算统计信息
    - 生成表指纹
    
    分析维度:
    - 结构特征: 列数、列名、数据类型
    - 内容特征: 样本值、分布、模式
    - 语义特征: 描述、注释、业务含义
```

#### 3.1.4 SearcherAgent（搜索器）
```python
class SearcherAgent:
    """候选搜索Agent"""
    
    职责:
    - 调用Layer 1元数据筛选
    - 调用Layer 2向量搜索
    - 管理搜索策略
    - 候选剪枝优化
    
    搜索流程:
    1. metadata_filter() → 500 candidates
    2. vector_search() → 50 candidates
    3. prune_candidates() → optimized list
```

#### 3.1.5 MatcherAgent（匹配器）
```python
class MatcherAgent:
    """精确匹配Agent"""
    
    职责:
    - 调用Layer 3 LLM验证
    - 并行处理候选表
    - 计算匹配分数
    - 收集匹配证据
    
    并行策略:
    - 批大小: 20个并发LLM调用
    - 超时控制: 30秒
    - 重试机制: 最多3次
```

#### 3.1.6 AggregatorAgent（聚合器）
```python
class AggregatorAgent:
    """结果聚合Agent"""
    
    职责:
    - 融合多维度分数
    - 结果排序
    - Top-K选择
    - 生成推荐理由
    
    分数融合公式:
    final_score = 0.3 * metadata_score + 
                  0.3 * vector_score + 
                  0.4 * llm_score
```

### 3.2 Agent通信机制

```python
# 消息总线实现
class MessageBus:
    def __init__(self):
        self.subscribers = defaultdict(list)
        self.message_queue = asyncio.Queue()
    
    async def publish(self, topic: str, message: Any):
        """发布消息到指定主题"""
        for subscriber in self.subscribers[topic]:
            await subscriber.handle_message(message)
    
    def subscribe(self, topic: str, agent: BaseAgent):
        """订阅指定主题"""
        self.subscribers[topic].append(agent)
```

---

## 4. 三层加速架构

### 4.1 Layer 1: MetadataFilter（元数据筛选）

```python
class MetadataFilter:
    """快速规则筛选，响应时间<10ms"""
    
    def filter(self, query_table, all_tables):
        candidates = []
        
        # 1. 列数匹配（±20%）
        col_count = len(query_table.columns)
        for table in all_tables:
            if 0.8 * col_count <= len(table.columns) <= 1.2 * col_count:
                candidates.append(table)
        
        # 2. 数据类型匹配
        query_types = {col.type for col in query_table.columns}
        candidates = [t for t in candidates 
                     if len(query_types & {c.type for c in t.columns}) > 0.5]
        
        # 3. 名称模式匹配
        keywords = extract_keywords(query_table.name)
        candidates = [t for t in candidates 
                     if any(kw in t.name for kw in keywords)]
        
        return candidates[:500]  # 最多返回500个
```

**优化技术**:
- 倒排索引: 按列数建立索引，O(1)查找
- 位图匹配: 数据类型用位图表示，快速求交
- 布隆过滤器: 快速排除不可能的候选

### 4.2 Layer 2: VectorSearch（向量搜索）

```python
class VectorSearch:
    """基于HNSW的向量相似度搜索，响应时间10-50ms"""
    
    def __init__(self):
        self.index = faiss.IndexHNSWFlat(384, 32)  # 384维，M=32
        self.embeddings = {}
    
    async def build_index(self, tables):
        """预计算所有表的嵌入向量"""
        embeddings = []
        for table in tables:
            emb = await self.generate_embedding(table)
            embeddings.append(emb)
            self.embeddings[table.name] = emb
        
        # 构建HNSW索引
        embeddings_array = np.array(embeddings)
        self.index.add(embeddings_array)
    
    def search(self, query_embedding, k=50):
        """搜索最相似的k个表"""
        distances, indices = self.index.search(query_embedding, k)
        return [(self.table_names[i], 1/(1+d)) for i, d in zip(indices[0], distances[0])]
```

**HNSW参数优化**:
- M=32: 每个节点的邻居数
- ef_construction=200: 构建时的搜索宽度
- ef_search=100: 查询时的搜索宽度

### 4.3 Layer 3: SmartLLMMatcher（LLM验证）

```python
class SmartLLMMatcher:
    """并行LLM精确匹配，响应时间1-3s"""
    
    async def match_batch(self, query_table, candidates):
        """并行验证多个候选表"""
        
        # 构建prompt
        prompts = []
        for candidate in candidates:
            prompt = self.build_prompt(query_table, candidate)
            prompts.append(prompt)
        
        # 并行调用LLM（关键优化！）
        tasks = []
        for prompt in prompts[:20]:  # 最多20个并发
            task = asyncio.create_task(
                self.llm_client.generate(prompt, timeout=30)
            )
            tasks.append(task)
        
        # 等待所有结果
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # 解析结果
        scores = []
        for result in results:
            if isinstance(result, Exception):
                scores.append(0.0)
            else:
                score = self.parse_llm_response(result)
                scores.append(score)
        
        return scores
```

**并行优化关键**:
```python
# ❌ Before: 串行调用（72秒）
for candidate in candidates:
    result = llm_client.generate(prompt)  # 阻塞！
    
# ✅ After: 并行调用（3.6秒）
tasks = [llm_client.generate(p) for p in prompts]
results = await asyncio.gather(*tasks)  # 并行！
```

---

## 5. 技术实现细节

### 5.1 异步HTTP客户端优化

```python
# src/utils/llm_client_proxy.py
class GeminiClientWithProxy:
    """异步LLM客户端，支持代理"""
    
    def __init__(self, proxy_url="http://127.0.0.1:7890"):
        self.proxy = proxy_url
        self.base_url = "https://generativelanguage.googleapis.com"
    
    async def generate(self, prompt: str):
        """异步生成响应"""
        
        # ✅ 使用aiohttp（非阻塞）
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self.base_url}/v1/models/gemini-1.5-flash:generateContent",
                json={"contents": [{"parts": [{"text": prompt}]}]},
                proxy=self.proxy,
                timeout=aiohttp.ClientTimeout(total=30)
            ) as response:
                result = await response.json()
                return result["candidates"][0]["content"]["parts"][0]["text"]
        
        # ❌ 不要使用requests（阻塞）
        # response = requests.post(url, json=data)  # 会阻塞事件循环！
```

### 5.2 数据格式转换

```python
def dict_to_table_info(table_dict: Dict[str, Any]) -> TableInfo:
    """将字典转换为TableInfo对象"""
    columns = []
    for col_dict in table_dict.get('columns', []):
        column_info = ColumnInfo(
            table_name=table_dict['table_name'],
            column_name=col_dict.get('column_name', col_dict.get('name', '')),
            data_type=col_dict.get('data_type', col_dict.get('type', 'unknown')),
            sample_values=col_dict.get('sample_values', [])[:5],
            null_count=col_dict.get('null_count'),
            unique_count=col_dict.get('unique_count')
        )
        columns.append(column_info)
    
    return TableInfo(
        table_name=table_dict['table_name'],
        columns=columns,
        row_count=table_dict.get('row_count'),
        description=table_dict.get('description')
    )
```

### 5.3 评价指标计算

```python
def calculate_metrics(predictions, ground_truth):
    """计算多维度评价指标"""
    
    metrics = {
        'precision': len(set(predictions) & set(ground_truth)) / len(predictions),
        'recall': len(set(predictions) & set(ground_truth)) / len(ground_truth),
        'f1_score': 2 * precision * recall / (precision + recall),
        'hit_at_1': 1 if predictions[0] in ground_truth else 0,
        'hit_at_3': 1 if any(p in ground_truth for p in predictions[:3]) else 0,
        'hit_at_5': 1 if any(p in ground_truth for p in predictions[:5]) else 0,
        'hit_at_10': 1 if any(p in ground_truth for p in predictions[:10]) else 0,
        'mrr': calculate_mrr(predictions, ground_truth)
    }
    
    return metrics
```

---

## 6. 性能优化策略

### 6.1 并行化优化

| 优化项 | Before | After | 提升 |
|--------|--------|-------|------|
| LLM调用 | 串行72s | 并行3.6s | 20x |
| 嵌入生成 | 串行15s | 批量2s | 7.5x |
| 候选搜索 | 顺序执行 | 并行执行 | 3x |

### 6.2 缓存策略

```python
class MultiLevelCache:
    """三级缓存系统"""
    
    def __init__(self):
        self.l1_cache = {}  # 内存缓存（最快）
        self.l2_cache = redis.Redis()  # Redis缓存（中等）
        self.l3_cache = DiskCache()  # 磁盘缓存（最慢）
    
    async def get(self, key):
        # L1查找
        if key in self.l1_cache:
            return self.l1_cache[key]
        
        # L2查找
        value = await self.l2_cache.get(key)
        if value:
            self.l1_cache[key] = value  # 提升到L1
            return value
        
        # L3查找
        value = await self.l3_cache.get(key)
        if value:
            await self.l2_cache.set(key, value)  # 提升到L2
            self.l1_cache[key] = value  # 提升到L1
            return value
        
        return None
```

### 6.3 索引优化

```yaml
HNSW索引参数:
  M: 32  # 每个节点的边数
  ef_construction: 200  # 构建时的动态列表大小
  ef_search: 100  # 搜索时的动态列表大小
  
性能指标:
  构建时间: 7-8秒（1,534个表）
  搜索时间: 10-50ms
  内存占用: ~200MB
  召回率: 85%@50
```

---

## 7. 系统部署与扩展

### 7.1 部署架构

```yaml
生产部署:
  负载均衡: Nginx
  应用服务: FastAPI (8 workers × 3 instances)
  缓存: Redis Cluster
  向量库: FAISS with persistence
  监控: Prometheus + Grafana
  日志: ELK Stack
```

### 7.2 扩展性设计

#### 7.2.1 水平扩展
- **无状态设计**: 所有Agent无状态，可水平扩展
- **分片策略**: 数据按hash分片到多个节点
- **负载均衡**: Round-robin或最少连接数

#### 7.2.2 垂直优化
- **GPU加速**: 向量计算使用GPU
- **内存优化**: 使用内存映射文件
- **JIT编译**: 使用Numba加速计算密集型代码

### 7.3 未来优化方向

1. **模型优化**
   - 微调专门的表匹配模型
   - 使用更轻量的嵌入模型
   - 实现模型量化压缩

2. **算法优化**
   - 实现增量索引更新
   - 使用近似算法加速
   - 引入强化学习优化策略

3. **系统优化**
   - 实现分布式Agent
   - 支持流式处理
   - 添加自适应并发控制

---

## 📊 系统性能总结

### 当前性能指标

| 指标 | Subset (100表) | Complete (1,534表) |
|------|---------------|-------------------|
| 响应时间 | 5.2秒 | 9.4秒 |
| 吞吐量 | 0.19 QPS | 0.11 QPS |
| Precision | 10.0% | 7.5% |
| Recall | 29.6% | 9.3% |
| Hit@10 | 44.4% | 36.0% |
| 内存使用 | ~1GB | ~2GB |
| CPU使用 | 4 cores | 8 cores |

### 关键成就

✅ **性能突破**: 查询响应从72秒优化到5-10秒（92%提升）
✅ **并发能力**: 支持20个并发LLM调用
✅ **系统稳定**: 100%查询成功率，无崩溃
✅ **可扩展性**: 成功处理15倍数据增长（100→1,534表）

---

*文档版本: 2.0*
*最后更新: 2025-08-12*
*作者: Multi-Agent System Team*