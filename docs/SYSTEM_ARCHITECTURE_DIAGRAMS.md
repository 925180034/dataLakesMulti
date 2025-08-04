# 数据湖多智能体系统架构图

本文档包含数据湖多智能体系统的完整架构图、数据流图和处理流程图。

## 1. 系统整体架构图

```mermaid
graph TB
    %% 用户交互层
    subgraph UI["用户交互层"]
        CLI[CLI命令行接口]
        API[FastAPI接口]
        CONFIG[配置管理]
    end
    
    %% 工作流引擎层
    subgraph WF["工作流引擎层"]
        LANG[LangGraph引擎]
        OPT[OptimizedWorkflow]
        ULTRA[UltraOptimizedWorkflow]
    end
    
    %% 多智能体层
    subgraph AGENTS["多智能体系统"]
        PA[PlannerAgent<br/>任务规划与路由]
        CDA[ColumnDiscoveryAgent<br/>列级别匹配]
        TAA[TableAggregationAgent<br/>表级别聚合]
        TDA[TableDiscoveryAgent<br/>表发现]
        TMA[TableMatchingAgent<br/>表匹配验证]
    end
    
    %% 三层加速架构
    subgraph ACC["三层加速架构"]
        L1[Layer 1: 元数据过滤<br/>MetadataFilter]
        L2[Layer 2: 向量搜索<br/>HNSW/FAISS Index]
        L3[Layer 3: LLM验证<br/>SmartLLMMatcher]
    end
    
    %% 优化组件层
    subgraph OPT_COMP["优化组件"]
        CACHE[多级缓存系统<br/>L1: Memory<br/>L2: Redis<br/>L3: Disk]
        BATCH[批处理器<br/>BatchLLMProcessor]
        PARALLEL[并行处理器<br/>ParallelBatchProcessor]
        MONITOR[性能监控<br/>PerformanceMonitor]
    end
    
    %% 数据存储层
    subgraph STORAGE["数据存储层"]
        TABLES[表数据存储<br/>JSON/Parquet]
        VECTOR_DB[向量数据库<br/>FAISS/ChromaDB]
        INDEX_DB[索引数据库<br/>元数据索引]
        CACHE_DB[缓存存储<br/>Redis/Disk]
    end
    
    %% LLM提供商
    subgraph LLM["LLM提供商"]
        GEMINI[Google Gemini]
        OPENAI[OpenAI GPT]
        ANTHROPIC[Anthropic Claude]
    end
    
    %% 连接关系
    CLI --> LANG
    API --> LANG
    CONFIG --> LANG
    
    LANG --> OPT
    LANG --> ULTRA
    
    OPT --> PA
    ULTRA --> PA
    
    PA --> CDA
    PA --> TDA
    CDA --> TAA
    TAA --> TMA
    TDA --> TMA
    
    CDA --> L1
    TDA --> L1
    L1 --> L2
    L2 --> L3
    L3 --> LLM
    
    CACHE -.-> L1
    CACHE -.-> L2
    CACHE -.-> L3
    
    BATCH --> LLM
    PARALLEL --> LLM
    MONITOR -.-> ACC
    
    L1 --> INDEX_DB
    L2 --> VECTOR_DB
    TABLES --> INDEX_DB
    TABLES --> VECTOR_DB
    CACHE --> CACHE_DB
    
    style UI fill:#e1f5fe
    style WF fill:#f3e5f5
    style AGENTS fill:#fff3e0
    style ACC fill:#e8f5e9
    style OPT_COMP fill:#fce4ec
    style STORAGE fill:#f5f5f5
    style LLM fill:#fff9c4
```

## 2. 数据流图

```mermaid
flowchart LR
    %% 输入
    subgraph INPUT["输入数据"]
        QUERY[用户查询<br/>query_table, query_column, query_type]
        TABLES_IN[表数据集<br/>tables.json]
        GT[Ground Truth<br/>真实标签]
    end
    
    %% 预处理
    subgraph PREPROCESS["预处理阶段"]
        PARSE[查询解析]
        LOAD[数据加载]
        CONVERT[数据转换<br/>TableInfo/ColumnInfo]
    end
    
    %% 索引构建
    subgraph INDEX["索引构建"]
        META_INDEX[元数据索引<br/>列名/类型/统计信息]
        VECTOR_INDEX[向量索引<br/>HNSW/FAISS]
        CACHE_INIT[缓存初始化]
    end
    
    %% 查询处理
    subgraph PROCESS["查询处理"]
        ROUTE[路由决策<br/>Bottom-Up/Top-Down]
        FILTER[候选筛选]
        RANK[相似度排序]
        VERIFY[LLM验证]
    end
    
    %% 结果生成
    subgraph OUTPUT["输出结果"]
        MATCHES[匹配表列表]
        SCORES[相似度分数]
        METRICS[评价指标<br/>Precision/Recall/F1]
    end
    
    %% 数据流连接
    QUERY --> PARSE
    TABLES_IN --> LOAD
    GT --> LOAD
    
    PARSE --> ROUTE
    LOAD --> CONVERT
    CONVERT --> META_INDEX
    CONVERT --> VECTOR_INDEX
    
    META_INDEX --> FILTER
    VECTOR_INDEX --> FILTER
    CACHE_INIT --> FILTER
    
    ROUTE --> FILTER
    FILTER --> RANK
    RANK --> VERIFY
    
    VERIFY --> MATCHES
    VERIFY --> SCORES
    MATCHES --> METRICS
    GT --> METRICS
    
    style INPUT fill:#e3f2fd
    style PREPROCESS fill:#f3e5f5
    style INDEX fill:#e8f5e9
    style PROCESS fill:#fff3e0
    style OUTPUT fill:#ffebee
```

## 3. 查询处理详细流程图

```mermaid
flowchart TD
    START[开始查询处理]
    
    %% 查询分析
    ANALYZE[查询分析<br/>提取table/column/type]
    
    %% 策略选择
    STRATEGY{选择策略}
    BOTTOM_UP[Bottom-Up策略<br/>列匹配优先]
    TOP_DOWN[Top-Down策略<br/>表匹配优先]
    
    %% Bottom-Up流程
    COL_SEARCH[列搜索<br/>ColumnDiscoveryAgent]
    COL_FILTER[元数据过滤<br/>列名/类型匹配]
    COL_VECTOR[向量相似度<br/>语义匹配]
    TABLE_AGG[表聚合<br/>TableAggregationAgent]
    
    %% Top-Down流程
    TABLE_SEARCH[表搜索<br/>TableDiscoveryAgent]
    TABLE_FILTER[元数据过滤<br/>表结构匹配]
    TABLE_VECTOR[向量相似度<br/>表级别匹配]
    
    %% 三层加速
    L1_META{Layer 1<br/>元数据过滤}
    L2_VECTOR{Layer 2<br/>向量搜索}
    L3_LLM{Layer 3<br/>LLM验证}
    
    %% 缓存检查
    CACHE_CHECK{缓存检查}
    CACHE_HIT[缓存命中<br/>直接返回]
    CACHE_MISS[缓存未命中<br/>继续处理]
    
    %% LLM处理
    LLM_BATCH[批量LLM调用<br/>并行处理]
    LLM_PARSE[结果解析]
    
    %% 结果处理
    MERGE[结果合并]
    RANK[结果排序]
    TOP_K[取Top-K结果]
    
    %% 评估
    EVALUATE[计算评价指标<br/>Precision/Recall/F1]
    
    END[返回结果]
    
    %% 连接关系
    START --> ANALYZE
    ANALYZE --> STRATEGY
    
    STRATEGY -->|Join查询| BOTTOM_UP
    STRATEGY -->|Union查询| TOP_DOWN
    
    BOTTOM_UP --> COL_SEARCH
    COL_SEARCH --> CACHE_CHECK
    CACHE_CHECK -->|命中| CACHE_HIT
    CACHE_CHECK -->|未命中| CACHE_MISS
    CACHE_MISS --> COL_FILTER
    COL_FILTER --> L1_META
    L1_META -->|通过| COL_VECTOR
    L1_META -->|过滤| END
    COL_VECTOR --> L2_VECTOR
    L2_VECTOR -->|候选| TABLE_AGG
    TABLE_AGG --> L3_LLM
    
    TOP_DOWN --> TABLE_SEARCH
    TABLE_SEARCH --> CACHE_CHECK
    TABLE_FILTER --> L1_META
    TABLE_VECTOR --> L2_VECTOR
    
    L3_LLM -->|需要验证| LLM_BATCH
    L3_LLM -->|高置信度| MERGE
    LLM_BATCH --> LLM_PARSE
    LLM_PARSE --> MERGE
    
    CACHE_HIT --> MERGE
    MERGE --> RANK
    RANK --> TOP_K
    TOP_K --> EVALUATE
    EVALUATE --> END
    
    style START fill:#4caf50
    style END fill:#f44336
    style CACHE_HIT fill:#ffc107
    style L1_META fill:#e8f5e9
    style L2_VECTOR fill:#e3f2fd
    style L3_LLM fill:#fff3e0
```

## 4. 多智能体协作流程图

```mermaid
sequenceDiagram
    participant U as 用户
    participant CLI as CLI接口
    participant WF as 工作流引擎
    participant PA as PlannerAgent
    participant CDA as ColumnDiscoveryAgent
    participant TAA as TableAggregationAgent
    participant TDA as TableDiscoveryAgent
    participant TMA as TableMatchingAgent
    participant ACC as 三层加速架构
    participant LLM as LLM服务
    
    U->>CLI: 提交查询
    CLI->>WF: 初始化工作流
    WF->>PA: 分析查询意图
    
    alt Bottom-Up策略 (Join查询)
        PA->>CDA: 执行列发现
        CDA->>ACC: 调用三层加速
        ACC->>ACC: L1: 元数据过滤
        ACC->>ACC: L2: 向量搜索
        ACC->>LLM: L3: LLM验证
        LLM-->>ACC: 返回验证结果
        ACC-->>CDA: 返回候选列
        CDA->>TAA: 聚合到表级别
        TAA->>TMA: 验证表匹配
    else Top-Down策略 (Union查询)
        PA->>TDA: 执行表发现
        TDA->>ACC: 调用三层加速
        ACC->>ACC: L1: 元数据过滤
        ACC->>ACC: L2: 向量搜索
        ACC->>LLM: L3: LLM验证
        LLM-->>ACC: 返回验证结果
        ACC-->>TDA: 返回候选表
        TDA->>TMA: 验证表匹配
    end
    
    TMA->>WF: 返回最终结果
    WF->>CLI: 格式化输出
    CLI->>U: 显示结果
```

## 5. 性能优化组件图

```mermaid
graph LR
    subgraph CACHE["多级缓存系统"]
        L1C[L1: 内存缓存<br/>最近查询]
        L2C[L2: Redis缓存<br/>热点数据]
        L3C[L3: 磁盘缓存<br/>历史结果]
        
        L1C --> L2C
        L2C --> L3C
    end
    
    subgraph BATCH["批处理优化"]
        QUEUE[查询队列]
        BATCHER[批量组合器<br/>最大10个/批]
        BATCH_LLM[批量LLM调用]
        
        QUEUE --> BATCHER
        BATCHER --> BATCH_LLM
    end
    
    subgraph PARALLEL["并行处理"]
        SPLITTER[任务分割器]
        W1[Worker 1]
        W2[Worker 2]
        W3[Worker N]
        MERGER[结果合并器]
        
        SPLITTER --> W1
        SPLITTER --> W2
        SPLITTER --> W3
        W1 --> MERGER
        W2 --> MERGER
        W3 --> MERGER
    end
    
    subgraph MONITOR["性能监控"]
        METRICS[指标收集<br/>延迟/吞吐量/成功率]
        PROFILER[性能分析器]
        OPTIMIZER[自适应优化器]
        
        METRICS --> PROFILER
        PROFILER --> OPTIMIZER
    end
    
    style CACHE fill:#e8f5e9
    style BATCH fill:#e3f2fd
    style PARALLEL fill:#fff3e0
    style MONITOR fill:#ffebee
```

## 6. 三层加速架构详细图

```mermaid
graph TB
    subgraph INPUT["查询输入"]
        Q[查询表/列信息]
        CAND[候选表集合<br/>10,000+表]
    end
    
    subgraph L1["Layer 1: 元数据过滤"]
        META[元数据索引]
        FILTER1[规则过滤器]
        
        META --> FILTER1
        
        subgraph RULES["过滤规则"]
            R1[列名完全匹配]
            R2[数据类型兼容]
            R3[表结构相似度>0.5]
            R4[统计特征匹配]
        end
        
        FILTER1 --> R1
        FILTER1 --> R2
        FILTER1 --> R3
        FILTER1 --> R4
    end
    
    subgraph L2["Layer 2: 向量搜索"]
        EMB[嵌入生成<br/>Sentence-BERT]
        HNSW[HNSW索引<br/>高维向量搜索]
        TOPK[Top-K选择<br/>K=50]
        
        EMB --> HNSW
        HNSW --> TOPK
    end
    
    subgraph L3["Layer 3: LLM验证"]
        SMART[智能匹配器]
        
        subgraph STRATEGY["验证策略"]
            S1[规则预判<br/>高置信度直接通过]
            S2[批量验证<br/>合并多个请求]
            S3[并行调用<br/>最大20并发]
        end
        
        SMART --> S1
        SMART --> S2
        SMART --> S3
        
        LLM_API[LLM API调用]
        S2 --> LLM_API
        S3 --> LLM_API
    end
    
    subgraph OUTPUT["输出"]
        RESULTS[最终匹配结果<br/>Top-10表]
        SCORES[匹配分数]
        EVIDENCE[匹配证据]
    end
    
    %% 数据流
    Q --> L1
    CAND --> L1
    
    L1 -->|1000表| L2
    L2 -->|50表| L3
    L3 -->|10表| OUTPUT
    
    OUTPUT --> RESULTS
    OUTPUT --> SCORES
    OUTPUT --> EVIDENCE
    
    %% 性能标注
    L1 -.->|"<10ms"| L2
    L2 -.->|"<100ms"| L3
    L3 -.->|"<3s"| OUTPUT
    
    style L1 fill:#e8f5e9
    style L2 fill:#e3f2fd
    style L3 fill:#fff3e0
    style OUTPUT fill:#ffebee
```

## 7. 系统部署架构图

```mermaid
graph TB
    subgraph CLIENT["客户端"]
        WEB[Web界面]
        CLI_CLIENT[CLI客户端]
        SDK[Python SDK]
    end
    
    subgraph GATEWAY["API网关"]
        NGINX[Nginx<br/>负载均衡]
        AUTH[认证服务]
        RATE[限流器]
    end
    
    subgraph APP["应用服务器"]
        API1[API Server 1<br/>FastAPI]
        API2[API Server 2<br/>FastAPI]
        APIX[API Server N<br/>FastAPI]
        
        subgraph WORKERS["工作进程"]
            W1[Worker 1]
            W2[Worker 2]
            WX[Worker N]
        end
    end
    
    subgraph CACHE_LAYER["缓存层"]
        REDIS1[Redis主节点]
        REDIS2[Redis从节点]
    end
    
    subgraph STORAGE_LAYER["存储层"]
        subgraph VECTOR["向量存储"]
            FAISS1[FAISS索引]
            CHROMA[ChromaDB]
        end
        
        subgraph DATA["数据存储"]
            S3[对象存储<br/>表数据]
            PG[PostgreSQL<br/>元数据]
        end
    end
    
    subgraph MONITOR_LAYER["监控层"]
        PROM[Prometheus<br/>指标收集]
        GRAF[Grafana<br/>可视化]
        LOG[ELK Stack<br/>日志分析]
    end
    
    %% 连接关系
    WEB --> NGINX
    CLI_CLIENT --> NGINX
    SDK --> NGINX
    
    NGINX --> AUTH
    AUTH --> RATE
    RATE --> API1
    RATE --> API2
    RATE --> APIX
    
    API1 --> W1
    API2 --> W2
    APIX --> WX
    
    API1 --> REDIS1
    API2 --> REDIS1
    APIX --> REDIS1
    REDIS1 --> REDIS2
    
    W1 --> FAISS1
    W2 --> CHROMA
    WX --> S3
    WX --> PG
    
    API1 --> PROM
    API2 --> PROM
    APIX --> PROM
    PROM --> GRAF
    API1 --> LOG
    
    style CLIENT fill:#e1f5fe
    style GATEWAY fill:#f3e5f5
    style APP fill:#fff3e0
    style CACHE_LAYER fill:#e8f5e9
    style STORAGE_LAYER fill:#f5f5f5
    style MONITOR_LAYER fill:#ffebee
```

## 系统特点总结

### 1. 多智能体协作
- **PlannerAgent**: 智能路由，选择最优策略
- **专业化Agent**: 每个Agent专注特定任务
- **协同工作**: Agent间无缝协作

### 2. 三层加速架构
- **Layer 1**: 元数据过滤，快速筛选（<10ms）
- **Layer 2**: 向量搜索，语义匹配（<100ms）
- **Layer 3**: LLM验证，精确匹配（<3s）

### 3. 性能优化
- **多级缓存**: L1/L2/L3三级缓存
- **批处理**: 减少LLM调用次数
- **并行处理**: 最大20并发
- **早停机制**: 高置信度直接返回

### 4. 可扩展性
- **模块化设计**: 易于添加新Agent
- **插件式架构**: 支持多种LLM和向量数据库
- **水平扩展**: 支持分布式部署

### 5. 评估指标
- **准确性**: Precision, Recall, F1-Score
- **性能**: 查询延迟 < 3秒（目标）
- **吞吐量**: 支持10+并发查询
- **可用性**: 99.9%正常运行时间

## 更新日志
- 2024-08-04: 创建初始架构图
- 包含系统架构、数据流、查询处理流程等7个详细图表
- 涵盖多智能体系统、三层加速架构、性能优化等核心组件