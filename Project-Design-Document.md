# 项目设计文档：混合策略多智能体框架
## 用于数据湖的数据发现与模式匹配

> 本文档描述了一个全新的、完整的项目设计文档。它将"先匹配，后发现"（Bottom-Up）作为核心的高精度策略，同时融入了"先发现，后匹配"（Top-Down）作为探索性策略，并由一个智能的规划器智能体（Planner Agent）进行动态调度。这形成了一个灵活、强大且智能的混合框架。

## 1. 愿景与设计哲学

本框架旨在构建一个先进的多智能体系统，以解决数据湖（Data Lakes）中两大核心挑战：**数据发现 (Data Discovery)** 与 **模式匹配 (Schema Matching)**。

核心设计哲学是 **"策略灵活性 (Strategic Flexibility)"**。我们认识到，不存在单一的"最佳"工作流程。因此，本系统将融合两种互补的策略，并由一个智能的**规划器智能体 (Planner Agent)** 根据用户意图动态选择最优路径：

### 1.1 高精度、由点及面的"自下而上 (Bottom-Up)"策略

- **核心思想：** 先匹配，后发现
- **流程：** 从最基础的单元——**列 (Column)** ——开始，在整个数据湖中寻找与查询表中的列相匹配的"连接点"。然后，基于这些强有力的证据，聚合推断出最可能相关的**表 (Table)**
- **适用场景：** 需要高精度结果、寻找可连接（Joinable）数据、数据集成任务

### 1.2 高召回、探索性的"自上而下 (Top-Down)"策略

- **核心思想：** 先发现，后匹配
- **流程：** 首先从**表 (Table)** 的整体语义出发，快速筛选出可能相关的候选表。然后，再对这些候选表进行详细的**列 (Column)** 匹配，以验证其相关性
- **适用场景：** 探索性分析、寻找可合并（Unionable）或主题相似的数据集

通过这种混合设计，系统能够在精确度和探索性之间取得最佳平衡，以适应多样化的用户需求。

## 2. 系统架构

系统基于`LangGraph`框架构建，包含三个核心智能体和一个共享知识库，由规划器进行统一协调。

### 2.1 核心组件

#### 规划器智能体 (Planner Agent)
系统的大脑。负责理解用户意图、选择执行策略（Bottom-Up 或 Top-Down）、编排任务流程，并最终生成报告。

#### 列发现与匹配模块 (Column Discovery & Matching Module)
- **列发现智能体 (Column Discovery Agent)：** "Bottom-Up"策略的核心。给定一个查询列，它负责在数据湖的所有列中找到匹配项
- **表聚合智能体 (Table Aggregation Agent)：** "Bottom-Up"策略的聚合器。接收零散的列匹配结果，推断出相关的表

#### 表发现与验证模块 (Table Discovery & Validation Module)
- **表发现智能体 (Table Discovery Agent)：** "Top-Down"策略的核心。给定一个查询表，它负责从语义上发现相似的候选表
- **表匹配智能体 (Table-to-Table Matching Agent)：** "Top-Down"策略的验证器。比较两个完整的表，进行详细的列匹配

#### 共享知识库与缓存 (Shared Knowledge Base & Cache)
存储预处理的索引（向量索引、倒排索引）和已完成任务的结果，用于加速计算和避免重复工作。

## 3. 核心组件与智能体设计

### 3.1 规划器智能体 (Planner Agent)

**核心职责：** 智能路由与任务编排

**工作流：**

1. 接收用户请求和查询表
2. **构建决策Prompt**，调用LLM API判断用户意图：

```prompt
You are an expert task planner. Analyze the user's request to determine the best strategy.

User Request: "{user_query}"

Choose a strategy:
1. **Bottom-Up (Match then Discover):** Best for finding joinable tables, precise connections, or when user asks to "match columns".
2. **Top-Down (Discover then Match):** Best for finding unionable tables, similar topics, or for general exploration.

Your decision (1 or 2):
```

3. 根据LLM的返回结果，决定启动 **策略A（Bottom-Up）** 还是 **策略B（Top-Down）** 的工作流
4. 在工作流的每一步完成后，接收中间结果，并规划下一步行动，直到任务完成
5. 最后，整合所有结果，生成一份对用户友好的报告

### 3.2 策略A: “自下而上”模块

#### 3.2.1 列发现智能体 (Column Discovery Agent)

**核心职责：** 给定一个查询列，返回数据湖中所有匹配的列

**输入：** 单个查询列对象（包含名称、类型、数据样本等）

**内部工具：**
- **语义搜索工具：** 使用列的Embedding向量，在向量索引库中进行ANN搜索
- **值重叠搜索工具：** 使用列的值，在倒排索引中进行关键词/实体搜索

**输出：** 匹配的列清单，附带置信度和原因

```json
{
  "source_column": "query_table.column_X",
  "matched_columns": [
    {"column": "table_A.column_Y", "confidence": 0.98},
    {"column": "table_B.column_Z", "confidence": 0.85}
  ]
}
```

#### 3.2.2 表聚合智能体 (Table Aggregation Agent)

**核心职责：** 从列匹配结果中推断出相关的表

**输入：** `ColumnDiscoveryAgent`产生的所有列匹配对的集合

**工作流：**

1. **聚合 (Group By)：** 按目标表名对所有匹配结果进行分组
2. **评分 (Scoring)：** 为每个聚合后的表打分，综合考虑匹配列的数量、平均置信度、是否有关键列（如ID）匹配等因素
3. **排序 (Ranking)：** 按分数对表进行排序

**输出：** 排好序的表清单，附带证据

```json
{
  "discovered_tables_ranked": [
    {"table": "table_A", "score": 95.5, "evidence_columns": ["column_Y", ...]},
    {"table": "table_B", "score": 89.0, "evidence_columns": ["column_Z", ...]}
  ]
}
```

### 3.3 策略B: “自上而下”模块

#### 3.3.1 表发现智能体 (Table Discovery Agent)

**核心职责：** 给定一个查询表，返回数据湖中语义上相似的表

**输入：** 一个查询表对象

**内部工具：** 主要使用 **语义搜索工具**。计算整个查询表的Embedding（或其关键列的平均Embedding），在 **表级别** 的向量索引库中进行搜索

**输出：** 一组候选表名

#### 3.3.2 表匹配智能体 (Table-to-Table Matching Agent)

**核心职责：** 详细比较两个表，找出所有匹配的列对

**输入：** 一对表（`table_A`, `table_B`）

**工作流：** 对两个表的列进行笛卡尔积组合，为每一对列调用LLM进行匹配判断（参考第一版设计文档中的`SchemaMatchingAgent`）

**输出：** 详细的列匹配报告

## 4. 详细工作流程场景

### 场景A: 用户寻找可连接的表 (启动“自下而上”策略)

1. **Planner：** 用户请求包含"连接"、"ID"，决定使用 **Bottom-Up** 策略。它将查询表的列`[id, name, email]`传递给`ColumnDiscoveryAgent`

2. **ColumnDiscoveryAgent (并行)：**
   - 任务1: 寻找`id`的匹配项，找到 `table_X.user_id`
   - 任务2: 寻找`name`的匹配项，找到 `table_Y.customer_name`
   - 任务3: 寻找`email`的匹配项，找到 `table_X.login_email` 和 `table_Y.contact_email`

3. **TableAggregationAgent：**
   - 接收到所有结果
   - 聚合：`table_X`有2个匹配（`user_id`, `login_email`），`table_Y`有2个匹配（`customer_name`, `contact_email`）
   - 评分：`table_X`的`id`匹配是关键列匹配，得分更高
   - 输出排序列表：`[table_X, table_Y]`

4. **Planner：** 接收到排好序的表和其匹配证据，生成最终报告

### 场景B: 用户寻找主题相似的表 (启动“自上而下”策略)

1. **Planner：** 用户请求包含"类似数据"、"其他部门的报告"，决定使用 **Top-Down** 策略。它将整个查询表传递给`TableDiscoveryAgent`

2. **TableDiscoveryAgent：** 计算查询表的Embedding，在表向量索引库中搜索，返回一个候选列表 `[table_C, table_D]`

3. **Planner：** 接收到候选列表，生成下一步计划：为`table_C`和`table_D`分别启动匹配任务

4. **TableMatchingAgent (并行)：**
   - 任务1: 比较 查询表 vs `table_C`，返回详细的列匹配结果
   - 任务2: 比较 查询表 vs `table_D`，返回详细的列匹配结果

5. **Planner：** 接收到所有详细的匹配报告，汇总后生成最终报告

## 5. 实现与技术栈

| 组件 | 技术/库 | 备注 |
|:---|:---|:---|
| **Orchestration** | `LangGraph` | 用于构建和执行有向无环图，实现条件路由和状态管理 |
| **LLM Interface** | OpenAI API, Anthropic, or others | 提供所有智能体进行推理、决策和文本生成的核心能力 |
| **Vector Search** | FAISS, ChromaDB, Pinecone | 存储列和表的Embedding向量，并进行高效的近似最近邻搜索 |
| **Value Indexing** | Inverted Index (e.g., Whoosh, Lucene) | 用于"自下而上"策略中的快速值重叠搜索 |
| **LLM Caching** | GPTCache, or custom Redis/KV store | 缓存LLM的请求和响应，降低成本，提高响应速度 |

### 5.1 关键实现：LangGraph中的条件路由

Planner的智能决策将通过`LangGraph`中的条件边（Conditional Edges）来实现。

```python
# Pseudocode for the graph
from langgraph.graph import StateGraph, END

# ... (AgentState and Agent nodes definition) ...

def planner_router(state: AgentState):
    """ Reads planner's decision and routes to the correct workflow. """
    if state['planner_decision'] == "BOTTOM_UP":
        return "column_discoverer"
    elif state['planner_decision'] == "TOP_DOWN":
        return "table_discoverer"
    else:
        return END

# Build the graph
workflow = StateGraph(AgentState)
workflow.add_node("planner", planner_node)

# --- Bottom-Up Path ---
workflow.add_node("column_discoverer", column_discoverer_node)
workflow.add_node("table_aggregator", table_aggregator_node)

# --- Top-Down Path ---
workflow.add_node("table_discoverer", table_discoverer_node)
workflow.add_node("table_matcher", table_matcher_node)

# Set the entry point and conditional routing
workflow.set_entry_point("planner")
workflow.add_conditional_edges("planner", planner_router)

# Define workflow paths
workflow.add_edge("column_discoverer", "table_aggregator")
workflow.add_edge("table_aggregator", "planner") # Return to planner for final report

workflow.add_edge("table_discoverer", "table_matcher")
workflow.add_edge("table_matcher", "planner") # Return to planner for final report

# Compile the graph
app = workflow.compile()
```

## 6. 结论

本设计文档提出了一个**混合策略、智能调度**的多智能体框架。其核心优势在于：

### 6.1 核心优势

- **灵活性：** 能够根据用户意图，在**高精度 (Bottom-Up)** 和 **高召回 (Top-Down)** 两种策略间动态切换
- **精确性：** "先匹配，后发现"的策略确保了结果的强相关性和可靠性
- **可解释性：** 结果天然地附带了发现的"证据"，让用户明白为何一个表是相关的
- **模块化：** 基于`LangGraph`的清晰架构，易于实现、扩展和维护

### 6.2 总结

该设计为您将现有研究成果扩展到复杂的数据湖环境，并构建一个真正实用的智能数据工具提供了坚实而完整的蓝图。