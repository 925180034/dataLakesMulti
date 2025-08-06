# Separated WebTable Datasets

This directory contains WebTable datasets separated by task type (JOIN and UNION) for experiments.

## Directory Structure

```
separated_datasets/
├── join_subset/          # Join task subset (100 tables)
│   ├── tables.json       # Table metadata and columns
│   ├── queries.json      # Join queries with column specifications
│   └── ground_truth.json # Expected matches for evaluation
├── union_subset/         # Union task subset (100 tables)
│   ├── tables.json       # Table metadata and columns
│   ├── queries.json      # Union queries (table-level)
│   └── ground_truth.json # Expected matches for evaluation
├── join_complete/        # Complete join dataset (1,534 tables)
│   ├── tables.json       # Table metadata and columns
│   ├── queries.json      # Join queries with column specifications
│   └── ground_truth.json # Expected matches for evaluation
└── union_complete/       # Complete union dataset (1,534 tables)
    ├── tables.json       # Table metadata and columns
    ├── queries.json      # Union queries (table-level)
    └── ground_truth.json # Expected matches for evaluation
```

## Dataset Statistics

### Subset Datasets (For Quick Testing)
- **Join Subset**: 100 tables, 402 queries, 84 ground truth entries
- **Union Subset**: 100 tables, 100 queries, 110 ground truth entries

### Complete Datasets (For Full Evaluation)
- **Join Complete**: 1,534 tables, 5,824 queries, 6,805 ground truth entries
- **Union Complete**: 1,534 tables, 1,534 queries, 8,248 ground truth entries

## Data Format

### Tables (tables.json)
```json
[
  {
    "table_name": "csvData12345.csv",
    "columns": [
      {
        "column_name": "Player",
        "data_type": "string",
        "sample_values": ["John Doe", "Jane Smith"]
      }
    ]
  }
]
```

### Join Queries (queries.json)
```json
[
  {
    "query_table": "csvData12345.csv",
    "query_column": "Player",
    "query_type": "join"
  }
]
```

### Union Queries (queries.json)
```json
[
  {
    "query_table": "csvData12345.csv",
    "query_type": "union"
  }
]
```

### Ground Truth (ground_truth.json)

#### Join Ground Truth
```json
[
  {
    "query_table": "csvData12345.csv",
    "candidate_table": "csvData67890.csv",
    "query_column": "Player",
    "candidate_column": "Name",
    "query_type": "join"
  }
]
```

#### Union Ground Truth
```json
[
  {
    "query_table": "csvData12345.csv",
    "candidate_table": "csvData67890.csv",
    "query_type": "union"
  }
]
```

## Usage Examples

### For Join Experiments
```python
import json

# Load join dataset
with open('join_subset/tables.json') as f:
    tables = json.load(f)
with open('join_subset/queries.json') as f:
    queries = json.load(f)
with open('join_subset/ground_truth.json') as f:
    ground_truth = json.load(f)

# Process join queries
for query in queries:
    query_table = query['query_table']
    query_column = query['query_column']
    # Find joinable tables based on column matching
```

### For Union Experiments
```python
import json

# Load union dataset
with open('union_subset/tables.json') as f:
    tables = json.load(f)
with open('union_subset/queries.json') as f:
    queries = json.load(f)
with open('union_subset/ground_truth.json') as f:
    ground_truth = json.load(f)

# Process union queries
for query in queries:
    query_table = query['query_table']
    # Find unionable tables based on data instance similarity
```

## Running Experiments

### With the Data Lake Multi-Agent System

```bash
# Join experiment with subset
python run_cli.py discover \
  -q "find joinable tables" \
  -t examples/separated_datasets/join_subset/tables.json \
  -f json

# Union experiment with subset
python run_cli.py discover \
  -q "find similar tables" \
  -t examples/separated_datasets/union_subset/tables.json \
  -f json
```

### Evaluation
```python
# Use the ground truth for computing metrics
from sklearn.metrics import precision_score, recall_score, f1_score

# Your predictions vs ground truth
predictions = your_model_predictions()
ground_truth = load_ground_truth()

# Calculate metrics
precision = precision_score(ground_truth, predictions)
recall = recall_score(ground_truth, predictions)
f1 = f1_score(ground_truth, predictions)
```

## Notes

1. **Table Overlap**: Many tables appear in both join and union datasets as they can be used for both types of operations.

2. **Query Types**: 
   - Join queries focus on column-level matching (schema matching)
   - Union queries focus on table-level similarity (data instance matching)

3. **Ground Truth**: Contains the expected matches for evaluation. Use this to compute precision, recall, and F1-score.

4. **Performance Testing**: Start with subset datasets for quick testing, then move to complete datasets for full evaluation.

## Validation

All datasets have been validated for:
- Correct query type labels
- Presence of required fields
- Referential integrity (all referenced tables exist)
- Format consistency

Run `python validate_datasets.py` to re-validate if needed.