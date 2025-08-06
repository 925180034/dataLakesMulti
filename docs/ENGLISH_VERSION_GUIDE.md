# English Version Guide

This guide explains how to use the English versions of the Data Lake Multi-Agent System documentation and experiments.

## English Documentation

### Academic README
- **File**: `README_EN.md`
- **Purpose**: Complete system documentation following academic paper standards
- **Contents**: Abstract, methodology, experimental setup, results, and references
- **Usage**: Share with international collaborators or use for paper submissions

### English Prompt Templates
- **File**: `src/config/prompts_en.py`
- **Purpose**: English versions of all agent prompts with academic terminology
- **Usage**: Import in your code to use English prompts:
  ```python
  from src.config.prompts_en import PROMPT_TEMPLATES as ENGLISH_PROMPTS
  ```

## Running Experiments in English

### Method 1: English Experiment Runner
Run experiments with English output directly:

```bash
# Basic usage
python run_experiment_english.py --dataset subset --max-queries 50

# Full evaluation
python run_experiment_english.py --dataset complete

# Without LLM (vector only)
python run_experiment_english.py --dataset subset --max-queries 10 --no-llm

# Custom output path
python run_experiment_english.py --dataset subset --output results/my_experiment.json
```

### Method 2: Convert Existing Results
Convert Chinese experiment outputs to English:

```bash
# Convert single file
python scripts/convert_experiment_output_to_english.py experiment_results/your_result.json

# Convert entire directory
python scripts/convert_experiment_output_to_english.py experiment_results/

# Specify output location
python scripts/convert_experiment_output_to_english.py input.json -o output_en.json
```

## English Output Format

The English experiment output follows standard academic conventions:

```json
{
  "experiment_metadata": {
    "experiment_id": "exp_20250805_sample",
    "timestamp": "2025-08-05T10:00:00Z",
    "system_version": "v2.1",
    "configuration": {...}
  },
  "performance_metrics": {
    "average_query_latency": 1.084,
    "queries_per_second": 0.922,
    ...
  },
  "accuracy_metrics": {
    "precision": 0.3,
    "recall": 0.75,
    "f1_score": 0.4286,
    ...
  }
}
```

## Academic Terminology Reference

| Chinese | English (Academic) |
|---------|-------------------|
| 精确率 | Precision |
| 召回率 | Recall |
| F1分数 | F1-Score |
| 查询延迟 | Query Latency |
| 吞吐量 | Throughput |
| 真阳性 | True Positives |
| 假阳性 | False Positives |
| 假阴性 | False Negatives |
| 元数据过滤 | Metadata Filtering |
| 向量搜索 | Vector Search |
| 语义相似度 | Semantic Similarity |
| 数据湖 | Data Lake |
| 模式匹配 | Schema Matching |
| 多智能体系统 | Multi-Agent System |

## Citation Format

When using this system in academic papers, please cite:

```bibtex
@software{datalakes_multiagent_2025,
  title={Multi-Agent System for Data Lake Schema Matching and Discovery},
  author={Your Name},
  year={2025},
  url={https://github.com/925180034/dataLakesMulti}
}
```

## Best Practices

1. **Consistency**: Use either Chinese or English throughout your experiment, not both
2. **Reproducibility**: Always include configuration details in your reports
3. **Metrics**: Report standard IR metrics (Precision, Recall, F1) with 4 decimal places
4. **Statistical Significance**: For paper submissions, run at least 3 trials and report mean ± std
5. **Baselines**: Compare against standard baselines when reporting results

## Troubleshooting

If you encounter encoding issues:
```bash
# Set UTF-8 encoding
export LANG=en_US.UTF-8
export LC_ALL=en_US.UTF-8
```

For missing translations, check `scripts/convert_experiment_output_to_english.py` and add new mappings as needed.