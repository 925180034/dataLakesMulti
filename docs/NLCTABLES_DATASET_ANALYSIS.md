# nlcTables Dataset Analysis and Experiment Guide

## Dataset Overview

**nlcTables** is a specialized dataset for evaluating table discovery systems with natural language queries. It contains two main components:

### Dataset Structure
```
nlcTables/
├── nlcTables-J/     # JOIN task dataset
│   ├── datalake-test/        # 4,872 tables
│   ├── queries-test.txt      # 91 natural language queries
│   ├── qtrels-test.txt       # Ground truth relevance scores
│   └── deepjoin.csv         # Deep join annotations
└── nlcTables-U/     # UNION task dataset
    ├── datalake-test/        # 7,567 tables
    ├── queries-test.txt      # 255 natural language queries
    └── qtrels-test.txt       # Ground truth relevance scores
```

## Dataset Characteristics

### 1. **Natural Language Queries**
Unlike WebTable and OpenData which use table-based queries, nlcTables uses **natural language descriptions**:

**JOIN Query Examples:**
- "Can you find more joinable tables with keyword 0.363 with the topic related to Sports_Team_Statistics"
- "I want to find more joinable tables, and these tables have topics related to stockport_contracts"
- "Can you find more tables with similar content to Ext. Applied to Contract and have more than 10 records"

**UNION Query Examples:**
- "I am searching for more unionable tables with the value of Given-name being Emma Frances and containing the keyword Henry"
- "I need to locate additional unionable tables where the value of Residence is Tossorontio"
- "Can you find more tables that related to keyword: Markham with the topic related to Baptismal_Records"

### 2. **Query Complexity Features**
- **Keyword matching**: Direct keyword search requirements
- **Topic constraints**: Domain-specific topic requirements
- **Column constraints**: Specific column name or value requirements
- **Size constraints**: Minimum rows/columns requirements
- **Temporal constraints**: Date/time range specifications

### 3. **Relevance Scoring**
Ground truth uses graded relevance scores (0-2):
- **0**: Not relevant
- **1**: Partially relevant
- **2**: Highly relevant

## Comparison with Existing Datasets

| Feature | WebTable | OpenData | nlcTables |
|---------|----------|----------|-----------|
| **Query Type** | Table-based | Table-based | Natural Language |
| **JOIN Queries** | 1,042 | 500 | 91 |
| **UNION Queries** | 3,222 | 3,095 | 255 |
| **Total Tables** | 6,487 | 3,595 | 12,439 |
| **Avg Tables/Query (JOIN)** | 1.47 | 3.06 | 2.32 |
| **Avg Tables/Query (UNION)** | 1.70 | 1.0 | 3.27 |
| **Query Complexity** | Low | Medium | High |
| **Real-world Relevance** | Medium | High | Very High |

## Integration Strategy for Your System

### 1. **Data Preparation**

```python
# Convert nlcTables to your system format
def convert_nlctables_to_system_format():
    """
    Convert nlcTables dataset to match your system's expected format
    """
    # Load tables
    tables = []
    for json_file in Path('datalake-test').glob('*.json'):
        with open(json_file, 'r') as f:
            table_data = json.load(f)
            tables.append({
                'name': json_file.stem,
                'columns': [
                    {
                        'name': col,
                        'type': 'numeric' if i in table_data.get('numericColumns', []) else 'string',
                        'sample_values': extract_column_values(table_data['data'], i)[:5]
                    }
                    for i, col in enumerate(table_data['title'])
                ]
            })
    
    # Load queries with ground truth
    queries = []
    for line in open('queries-test.txt'):
        qid, text, table_id = line.strip().split('\t')
        queries.append({
            'query_id': qid,
            'query_text': text,
            'seed_table': table_id  # Use as reference table
        })
    
    # Load ground truth
    ground_truth = defaultdict(list)
    for line in open('qtrels-test.txt'):
        qid, _, table_id, relevance = line.strip().split('\t')
        if int(relevance) > 0:
            ground_truth[qid].append(table_id)
    
    return tables, queries, ground_truth
```

### 2. **Natural Language Query Processing**

Your system needs enhancement to handle natural language queries:

```python
class NLQueryProcessor:
    """Process natural language queries for table discovery"""
    
    def extract_features(self, query_text):
        """Extract structured features from natural language query"""
        features = {
            'keywords': self.extract_keywords(query_text),
            'topics': self.extract_topics(query_text),
            'column_names': self.extract_column_mentions(query_text),
            'values': self.extract_value_mentions(query_text),
            'constraints': self.extract_constraints(query_text)
        }
        return features
    
    def to_structured_query(self, nl_query):
        """Convert natural language to structured query for your system"""
        features = self.extract_features(nl_query)
        
        # Create a synthetic seed table based on extracted features
        seed_table = {
            'columns': features['column_names'],
            'keywords': features['keywords'],
            'topic': features['topics']
        }
        
        return seed_table
```

### 3. **Experiment Design**

```bash
# Experiment script for nlcTables
python three_layer_ablation_optimized.py \
    --task join \
    --dataset /root/autodl-tmp/datalakes/nlcTables/nlcTables-J \
    --max-queries 91 \
    --layers L1+L2+L3 \
    --use-nl-queries
```

### 4. **Evaluation Metrics Enhancement**

For graded relevance (0-2 scores), use **NDCG (Normalized Discounted Cumulative Gain)**:

```python
def calculate_ndcg(predicted, ground_truth, relevance_scores, k=5):
    """Calculate NDCG@k for graded relevance"""
    dcg = 0.0
    for i, table_id in enumerate(predicted[:k]):
        if table_id in ground_truth:
            relevance = relevance_scores.get(table_id, 0)
            dcg += (2**relevance - 1) / np.log2(i + 2)
    
    # Calculate ideal DCG
    ideal_relevance = sorted(relevance_scores.values(), reverse=True)[:k]
    idcg = sum((2**rel - 1) / np.log2(i + 2) 
               for i, rel in enumerate(ideal_relevance))
    
    return dcg / idcg if idcg > 0 else 0.0
```

## Experimental Approach

### Phase 1: Baseline Evaluation
1. Convert nlcTables format to your system format
2. Run baseline without modifications
3. Measure performance degradation from table-based to NL queries

### Phase 2: NL Query Enhancement
1. Implement query understanding module
2. Add keyword extraction and topic modeling
3. Enhance L1 metadata filter with NL features

### Phase 3: Full System Adaptation
1. Fine-tune LLM prompts for NL queries
2. Adjust vector embeddings for query-table matching
3. Optimize threshold parameters for graded relevance

### Expected Challenges
1. **Query Understanding**: NL queries are more ambiguous than table queries
2. **Relevance Grading**: Binary classification needs adaptation for graded relevance
3. **Semantic Gap**: Bridging natural language to structured table matching
4. **Performance**: NL processing adds computational overhead

## Quick Start Commands

```bash
# 1. Extract datasets
cd /root/autodl-tmp/datalakes/nlcTables/nlcTables-J
unzip -q datalake-test-20250823T011404Z-1-001.zip

cd ../nlcTables-U
unzip -q datalake-test-20250823T013814Z-1-001.zip

# 2. Convert to system format
python convert_nlctables.py \
    --input /root/autodl-tmp/datalakes/nlcTables \
    --output /root/dataLakesMulti/examples/nlctables

# 3. Run experiments
python three_layer_ablation_optimized.py \
    --task both \
    --dataset examples/nlctables \
    --max-queries 50 \
    --layers L1+L2+L3

# 4. Analyze results
python analyze_nlctables_results.py \
    --results experiment_results/nlctables_*.json \
    --metrics ndcg,map,precision
```

## Key Insights

1. **Query Complexity**: nlcTables queries are significantly more complex, requiring semantic understanding
2. **Real-world Relevance**: Natural language queries better represent actual user needs
3. **Graded Relevance**: More nuanced evaluation than binary matching
4. **Smaller Scale**: Fewer queries but higher quality annotations
5. **Cross-domain**: Tables span multiple domains requiring broader understanding

## Recommendations

1. **Start Simple**: Begin with keyword-based matching before full NL understanding
2. **Leverage LLMs**: Use GPT/Gemini for query understanding and reformulation
3. **Hybrid Approach**: Combine NL features with existing vector search
4. **Incremental Development**: Build NL capabilities on top of existing system
5. **Careful Evaluation**: Use appropriate metrics for graded relevance (NDCG, MAP)

## Next Steps

1. Create data converter script for nlcTables → system format
2. Implement NL query processor module
3. Adapt evaluation metrics for graded relevance
4. Run baseline experiments to establish performance
5. Iterate on improvements based on error analysis