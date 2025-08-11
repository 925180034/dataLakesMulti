# 系统架构图表集合

## 1. 整体系统架构图

```mermaid
graph TB
    subgraph "📱 用户接口层"
        UI[Web UI/CLI]
        API[REST API]
        GQL[GraphQL]
    end
    
    subgraph "🤖 多智能体系统"
        ORCH[协调器 Orchestrator]
        
        subgraph "决策层 Agents"
            OPT[⚙️ OptimizerAgent<br/>性能优化]
            PLAN[📋 PlannerAgent<br/>策略规划]
            ANAL[🔬 AnalyzerAgent<br/>数据分析]
        end
        
        subgraph "执行层 Agents"
            SEARCH[🔍 SearcherAgent<br/>候选搜索]
            MATCH[✅ MatcherAgent<br/>精确匹配]
            AGG[📊 AggregatorAgent<br/>结果聚合]
        end
        
        MSG[消息总线<br/>Message Bus]
    end
    
    subgraph "⚡ 三层加速架构"
        L1[Layer 1: MetadataFilter<br/>规则筛选<br/>⏱️ <10ms]
        L2[Layer 2: VectorSearch<br/>向量搜索<br/>⏱️ 10-50ms]
        L3[Layer 3: SmartLLM<br/>智能验证<br/>⏱️ 1-3s]
    end
    
    subgraph "💾 数据存储层"
        DL[(数据湖<br/>10,000+ Tables)]
        IDX[(索引存储<br/>HNSW/Inverted)]
        CACHE[(缓存<br/>Multi-Level)]
    end
    
    subgraph "🧠 LLM服务"
        LLM[LLM Provider<br/>Gemini/GPT/Claude]
    end
    
    UI --> API
    API --> ORCH
    GQL --> ORCH
    
    ORCH <--> MSG
    MSG <--> OPT
    MSG <--> PLAN
    MSG <--> ANAL
    MSG <--> SEARCH
    MSG <--> MATCH
    MSG <--> AGG
    
    ANAL --> L1
    SEARCH --> L1
    SEARCH --> L2
    MATCH --> L3
    
    L1 --> IDX
    L2 --> IDX
    L3 --> LLM
    
    IDX --> DL
    CACHE --> DL
    
    style ORCH fill:#f9f,stroke:#333,stroke-width:4px
    style MSG fill:#bbf,stroke:#333,stroke-width:2px
    style L1 fill:#9f9,stroke:#333,stroke-width:2px
    style L2 fill:#9f9,stroke:#333,stroke-width:2px
    style L3 fill:#9f9,stroke:#333,stroke-width:2px
```

## 2. Agent协同工作流程图

```mermaid
flowchart LR
    subgraph "JOIN查询处理流程"
        J1[查询输入] --> J2[OptimizerAgent<br/>配置优化]
        J2 --> J3[PlannerAgent<br/>识别JOIN策略]
        J3 --> J4[AnalyzerAgent<br/>提取外键关系]
        J4 --> J5[SearcherAgent<br/>Layer1+Layer2搜索]
        J5 --> J6[MatcherAgent<br/>验证JOIN条件]
        J6 --> J7[AggregatorAgent<br/>排序输出]
        J7 --> J8[返回结果]
    end
```

```mermaid
flowchart LR
    subgraph "UNION查询处理流程"
        U1[查询输入] --> U2[OptimizerAgent<br/>配置优化]
        U2 --> U3[PlannerAgent<br/>识别UNION策略]
        U3 --> U4[AnalyzerAgent<br/>分析表结构]
        U4 --> U5[SearcherAgent<br/>向量相似搜索]
        U5 --> U6[MatcherAgent<br/>验证兼容性]
        U6 --> U7[AggregatorAgent<br/>合并结果]
        U7 --> U8[返回结果]
    end
```

## 3. 三层加速数据流图

```mermaid
graph TD
    Q[查询表] --> L1{Layer 1<br/>元数据筛选}
    
    L1 -->|列数匹配| L1A[规则1: 列数±2]
    L1 -->|类型匹配| L1B[规则2: 类型签名]
    L1 -->|命名模式| L1C[规则3: 命名相似]
    
    L1A --> L1R[1000候选]
    L1B --> L1R
    L1C --> L1R
    
    L1R --> L2{Layer 2<br/>向量搜索}
    
    L2 -->|表嵌入| L2A[BERT嵌入]
    L2 -->|相似度| L2B[余弦相似度]
    L2 -->|HNSW| L2C[近邻搜索]
    
    L2A --> L2R[100候选]
    L2B --> L2R
    L2C --> L2R
    
    L2R --> L3{Layer 3<br/>LLM验证}
    
    L3 -->|规则跳过| L3A[明显匹配]
    L3 -->|LLM验证| L3B[复杂匹配]
    L3 -->|批处理| L3C[批量验证]
    
    L3A --> L3R[20-30结果]
    L3B --> L3R
    L3C --> L3R
    
    L3R --> R[最终结果]
    
    style L1 fill:#e1f5fe,stroke:#01579b
    style L2 fill:#fff3e0,stroke:#e65100
    style L3 fill:#f3e5f5,stroke:#4a148c
```

## 4. 数据湖发现能力图

```mermaid
graph TB
    subgraph "数据湖发现类型"
        DLD[Data Lake Discovery]
        
        DLD --> ST[表结构发现<br/>Structure Discovery]
        DLD --> SE[语义关联发现<br/>Semantic Discovery]
        DLD --> IN[数据实例发现<br/>Instance Discovery]
        DLD --> CO[关系约束发现<br/>Constraint Discovery]
        
        ST --> ST1[列名匹配]
        ST --> ST2[类型匹配]
        ST --> ST3[结构相似]
        
        SE --> SE1[词向量相似]
        SE --> SE2[上下文理解]
        SE --> SE3[领域知识]
        
        IN --> IN1[值重叠]
        IN --> IN2[分布相似]
        IN --> IN3[模式识别]
        
        CO --> CO1[主外键]
        CO --> CO2[唯一性]
        CO --> CO3[参照完整性]
    end
    
    subgraph "Agent负责分工"
        A1[AnalyzerAgent] -.-> ST
        A2[SearcherAgent] -.-> SE
        A3[MatcherAgent] -.-> IN
        A3 -.-> CO
    end
```

## 5. 性能优化策略图

```mermaid
graph LR
    subgraph "优化层次"
        O[性能优化]
        
        O --> O1[索引优化]
        O --> O2[缓存优化]
        O --> O3[并发优化]
        O --> O4[批处理优化]
        
        O1 --> O11[HNSW索引]
        O1 --> O12[倒排索引]
        O1 --> O13[混合索引]
        
        O2 --> O21[L1内存缓存]
        O2 --> O22[L2磁盘缓存]
        O2 --> O23[L3分布式缓存]
        
        O3 --> O31[异步IO]
        O3 --> O32[线程池]
        O3 --> O33[协程并发]
        
        O4 --> O41[自适应批大小]
        O4 --> O42[流式处理]
        O4 --> O43[预取策略]
    end
```

## 6. Agent决策流程图

```mermaid
stateDiagram-v2
    [*] --> 查询输入
    
    查询输入 --> PlannerAgent: 分析查询
    
    state PlannerAgent {
        [*] --> 意图识别
        意图识别 --> JOIN判断: contains "join"
        意图识别 --> UNION判断: contains "union"
        意图识别 --> 复杂判断: 其他
        
        JOIN判断 --> BottomUp策略
        UNION判断 --> TopDown策略
        复杂判断 --> LLM分析
        LLM分析 --> 混合策略
    }
    
    PlannerAgent --> AnalyzerAgent: 传递策略
    
    state AnalyzerAgent {
        [*] --> 表分析
        表分析 --> 简单分析: 列数<20
        表分析 --> 复杂分析: 列数>=20
        
        简单分析 --> 规则提取
        复杂分析 --> LLM理解
        
        规则提取 --> 特征输出
        LLM理解 --> 特征输出
    }
    
    AnalyzerAgent --> SearcherAgent: 传递特征
    
    state SearcherAgent {
        [*] --> 搜索策略
        搜索策略 --> Layer1: 候选<100
        搜索策略 --> Layer1_2: 100-1000
        搜索策略 --> 全层搜索: >1000
        
        Layer1 --> 候选列表
        Layer1_2 --> 候选列表
        全层搜索 --> 候选列表
    }
    
    SearcherAgent --> MatcherAgent: 传递候选
    
    state MatcherAgent {
        [*] --> 匹配策略
        匹配策略 --> 规则验证: 明显匹配
        匹配策略 --> LLM验证: 需要验证
        
        规则验证 --> 匹配结果
        LLM验证 --> 匹配结果
    }
    
    MatcherAgent --> AggregatorAgent: 传递匹配
    
    state AggregatorAgent {
        [*] --> 聚合策略
        聚合策略 --> 简单排序: 结果>100
        聚合策略 --> 混合排序: 20-100
        聚合策略 --> 详细排序: <20
        
        简单排序 --> 最终结果
        混合排序 --> 最终结果
        详细排序 --> LLM重排
        LLM重排 --> 最终结果
    }
    
    AggregatorAgent --> [*]
```

## 7. 系统部署架构图

```mermaid
graph TB
    subgraph "客户端"
        C1[Web浏览器]
        C2[CLI工具]
        C3[SDK客户端]
    end
    
    subgraph "负载均衡层"
        LB[Nginx/HAProxy<br/>负载均衡器]
    end
    
    subgraph "应用服务器集群"
        subgraph "节点1"
            API1[API Server]
            MA1[Multi-Agent System]
            ACC1[三层加速]
        end
        
        subgraph "节点2"
            API2[API Server]
            MA2[Multi-Agent System]
            ACC2[三层加速]
        end
        
        subgraph "节点3"
            API3[API Server]
            MA3[Multi-Agent System]
            ACC3[三层加速]
        end
    end
    
    subgraph "存储层"
        subgraph "向量数据库集群"
            VDB1[(FAISS Master)]
            VDB2[(FAISS Slave)]
        end
        
        subgraph "缓存集群"
            RED1[(Redis Master)]
            RED2[(Redis Slave)]
        end
        
        subgraph "数据湖存储"
            S3[(对象存储<br/>S3/MinIO)]
        end
    end
    
    subgraph "LLM服务"
        LLM1[Gemini API]
        LLM2[OpenAI API]
        LLM3[Claude API]
    end
    
    subgraph "监控系统"
        PROM[Prometheus]
        GRAF[Grafana]
        ELK[ELK Stack]
    end
    
    C1 --> LB
    C2 --> LB
    C3 --> LB
    
    LB --> API1
    LB --> API2
    LB --> API3
    
    API1 --> MA1 --> ACC1
    API2 --> MA2 --> ACC2
    API3 --> MA3 --> ACC3
    
    ACC1 --> VDB1
    ACC2 --> VDB1
    ACC3 --> VDB1
    VDB1 --> VDB2
    
    ACC1 --> RED1
    ACC2 --> RED1
    ACC3 --> RED1
    RED1 --> RED2
    
    ACC1 --> S3
    ACC2 --> S3
    ACC3 --> S3
    
    MA1 --> LLM1
    MA2 --> LLM2
    MA3 --> LLM3
    
    API1 --> PROM
    API2 --> PROM
    API3 --> PROM
    PROM --> GRAF
    
    API1 --> ELK
    API2 --> ELK
    API3 --> ELK
```

## 8. 数据处理流程图

```mermaid
sequenceDiagram
    participant User as 用户
    participant API as API网关
    participant Orch as 协调器
    participant Cache as 缓存
    participant Agents as 多Agent系统
    participant L1 as Layer1
    participant L2 as Layer2
    participant L3 as Layer3
    participant DB as 数据湖
    
    User->>API: 发送查询请求
    API->>Orch: 转发请求
    
    Orch->>Cache: 检查缓存
    alt 缓存命中
        Cache-->>Orch: 返回缓存结果
        Orch-->>API: 直接返回
        API-->>User: 显示结果
    else 缓存未命中
        Orch->>Agents: 启动Agent协同
        
        Agents->>L1: 元数据筛选
        L1->>DB: 查询元数据索引
        DB-->>L1: 返回1000候选
        L1-->>Agents: 初步候选
        
        Agents->>L2: 向量搜索
        L2->>DB: HNSW搜索
        DB-->>L2: 返回100候选
        L2-->>Agents: 精选候选
        
        Agents->>L3: LLM验证
        L3->>L3: 智能匹配
        L3-->>Agents: 最终结果
        
        Agents-->>Orch: 聚合结果
        Orch->>Cache: 更新缓存
        Orch-->>API: 返回结果
        API-->>User: 显示结果
    end
```

## 9. 性能指标对比图

```mermaid
graph LR
    subgraph "纯三层加速"
        P1[查询] --> P2[Layer1<br/>10ms]
        P2 --> P3[Layer2<br/>50ms]
        P3 --> P4[Layer3<br/>2000ms]
        P4 --> P5[结果<br/>总计: 2060ms]
    end
    
    subgraph "多Agent+三层加速"
        M1[查询] --> M2[Agent决策<br/>5ms]
        M2 --> M3{智能路由}
        M3 -->|简单| M4A[仅Layer1<br/>10ms]
        M3 -->|中等| M4B[Layer1+2<br/>60ms]
        M3 -->|复杂| M4C[全部层<br/>2060ms]
        M4A --> M5[结果<br/>15ms]
        M4B --> M5[结果<br/>65ms]
        M4C --> M5[结果<br/>2065ms]
    end
    
    style P5 fill:#fcc,stroke:#f00
    style M5 fill:#cfc,stroke:#0f0
```

## 10. 数据湖表匹配工作流程

```mermaid
flowchart TD
    Start([开始]) --> Input[输入两个表]
    
    Input --> A1{AnalyzerAgent}
    A1 --> SM[结构匹配]
    
    SM --> SM1[列名完全匹配]
    SM --> SM2[列名模糊匹配]
    SM --> SM3[数据类型匹配]
    
    SM1 --> Score1[匹配分数+0.4]
    SM2 --> Score2[匹配分数+0.2]
    SM3 --> Score3[匹配分数+0.2]
    
    Score1 --> A2{SearcherAgent}
    Score2 --> A2
    Score3 --> A2
    
    A2 --> SEM[语义匹配]
    SEM --> Vec[生成词向量]
    Vec --> Sim[计算相似度]
    Sim --> Score4[匹配分数+0.2]
    
    Score4 --> A3{MatcherAgent}
    A3 --> IM[实例匹配]
    
    IM --> Val[样本值比较]
    Val --> Over[计算重叠度]
    Over --> Score5[匹配分数+0.2]
    
    Score5 --> Final[综合评分]
    
    Final --> Check{分数>0.7?}
    Check -->|是| Match[可以匹配]
    Check -->|否| NoMatch[不能匹配]
    
    Match --> End([结束])
    NoMatch --> End
    
    style A1 fill:#e3f2fd,stroke:#1976d2
    style A2 fill:#fff3e0,stroke:#f57c00
    style A3 fill:#f3e5f5,stroke:#7b1fa2
```

## 总结

这些架构图展示了系统的：
1. **整体架构**：多层次、模块化设计
2. **工作流程**：Agent协同和数据流动
3. **技术细节**：三层加速和数据湖发现
4. **部署方案**：分布式、高可用架构
5. **性能优化**：多维度优化策略

系统通过**6个智能Agent**的协同工作和**三层加速架构**的性能优化，实现了高效、准确的数据湖发现能力。