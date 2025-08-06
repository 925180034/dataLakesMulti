# Unified Experiment System for Data Lake Multi-Agent

## Quick Start

### 1. Basic Usage

```bash
# Quick test with 5 queries each for JOIN and UNION
./run_experiments.sh --task both --dataset subset --max-queries 5 --skip-llm

# Full experiment with LLM enabled
./run_experiments.sh --task both --dataset subset

# Specific task only
./run_experiments.sh --task join --dataset subset --max-queries 10
```

### 2. Direct Python Usage

```bash
# Run JOIN experiment
python unified_experiment.py --task join --dataset subset --max-queries 10

# Run UNION experiment  
python unified_experiment.py --task union --dataset subset --max-queries 10

# Run both with verbose output
python unified_experiment.py --task both --dataset subset --verbose

# Skip LLM for faster testing
python unified_experiment.py --task both --dataset subset --skip-llm
```

## Environment Variables

```bash
# Skip LLM calls (faster but less accurate)
export SKIP_LLM=true

# Set LLM timeout (seconds)
export LLM_TIMEOUT=30

# Set LLM max retries
export LLM_MAX_RETRIES=3
```

## Command Line Options

| Option | Values | Description |
|--------|--------|-------------|
| `--task` | join, union, both | Task type to run |
| `--dataset` | subset, complete, both | Dataset size |
| `--max-queries` | integer | Limit number of queries |
| `--skip-llm` | flag | Skip LLM calls |
| `--verbose` | flag | Show detailed output |

## Dataset Structure

```
examples/separated_datasets/
├── join_subset/       # 100 tables, 402 queries
│   ├── tables.json
│   ├── queries.json
│   └── ground_truth.json
├── union_subset/      # 100 tables, 100 queries
│   ├── tables.json
│   ├── queries.json
│   └── ground_truth.json
├── join_complete/     # 1534 tables, 5824 queries
│   └── ...
└── union_complete/    # 1534 tables, 1534 queries
    └── ...
```

## Output Format

Results are saved to `experiment_results/` with timestamps:
```
experiment_results/
├── join_subset_20250806_120000.json
├── union_subset_20250806_120100.json
└── ...
```

Each result file contains:
- Task type and dataset info
- Configuration used
- Metrics: Precision, Recall, F1-Score
- Performance: Query times, QPS
- Detailed results (if verbose)

## Metrics Explanation

- **Precision**: % of returned results that are correct
- **Recall**: % of correct results that were found
- **F1-Score**: Harmonic mean of Precision and Recall
- **True Positives (TP)**: Correctly identified matches
- **False Positives (FP)**: Incorrectly identified as matches
- **False Negatives (FN)**: Missed correct matches

## Troubleshooting

### Vector Index Loading Failed
```bash
# Initialize vector indexes
python initialize_indexes.py --dataset subset

# Or use the run script with --init-index
./run_experiments.sh --init-index --task join --dataset subset
```

### Low Performance
- Use `--skip-llm` for faster testing without LLM calls
- Reduce `--max-queries` for quicker experiments
- Use `subset` dataset instead of `complete`

### API Issues
- Check `.env` file for correct API keys
- Verify network connectivity
- Check API rate limits and quotas

## Expected Performance

### With LLM (SKIP_LLM=false)
- JOIN Task: 2-5 seconds per query
- UNION Task: 2-5 seconds per query
- Accuracy: Precision ~0.7-0.9, Recall ~0.6-0.8

### Without LLM (SKIP_LLM=true)
- JOIN Task: <1 second per query
- UNION Task: <1 second per query
- Accuracy: Lower (depends on vector search quality)

## Workflow Steps

1. **Data Loading**: Load tables, queries, and ground truth
2. **Workflow Init**: Initialize vector search and agents
3. **Query Processing**: 
   - JOIN: Find joinable columns across tables
   - UNION: Find similar tables for union
4. **Evaluation**: Compare predictions with ground truth
5. **Results**: Save metrics and detailed results

## Common Commands

```bash
# Quick validation test
python unified_experiment.py --task join --dataset subset --max-queries 2 --skip-llm

# Standard evaluation
python unified_experiment.py --task both --dataset subset --max-queries 50

# Full evaluation with LLM
python unified_experiment.py --task both --dataset complete --max-queries 100

# Debug mode with verbose output
python unified_experiment.py --task join --dataset subset --max-queries 5 --verbose
```