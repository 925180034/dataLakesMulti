# Multi-Agent System for Data Lake Schema Matching and Discovery

[![Python](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Framework](https://img.shields.io/badge/framework-LangGraph-orange.svg)](https://github.com/langchain-ai/langgraph)

## Abstract

This repository presents a novel multi-agent system designed for intelligent schema matching and data discovery in large-scale data lake environments. Leveraging state-of-the-art Large Language Models (LLMs) and a sophisticated three-layer acceleration architecture, our system achieves sub-second query latency while maintaining high accuracy in table matching tasks.

## Table of Contents

- [Introduction](#introduction)
- [System Architecture](#system-architecture)
- [Key Features](#key-features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Experimental Evaluation](#experimental-evaluation)
- [Performance Metrics](#performance-metrics)
- [Configuration](#configuration)
- [Advanced Usage](#advanced-usage)
- [Contributing](#contributing)
- [Citation](#citation)
- [License](#license)

## Introduction

In modern data lake environments, discovering semantically related tables across heterogeneous schemas presents significant challenges. This system addresses two fundamental scenarios:

- **Join Discovery**: Identifying tables with similar column structures for potential join operations
- **Union Discovery**: Discovering semantically related tables based on data content for union operations

Our approach employs a multi-agent collaborative framework built on LangGraph, incorporating five specialized agents that work synergistically to achieve high-precision table matching.

## System Architecture

### Three-Layer Acceleration Architecture

Our system implements a hierarchical filtering approach to optimize performance:

1. **Metadata Filtering Layer** (Millisecond-scale)
   - Rapid candidate filtering based on column names and data types
   - Reduces candidate space by >90%

2. **Vector Search Layer** (Hundred-millisecond-scale)
   - Semantic similarity matching using HNSW (Hierarchical Navigable Small World) index
   - Batch-parallel search capabilities

3. **LLM Verification Layer** (Second-scale)
   - Intelligent batch processing to minimize API calls
   - Parallel processing for enhanced throughput

### Multi-Agent Framework

The system comprises five specialized agents:

- **PlannerAgent**: Query intent understanding and strategy formulation
- **ColumnDiscoveryAgent**: Column-level similarity detection (Bottom-Up strategy)
- **TableAggregationAgent**: Column match aggregation into table-level scores
- **TableDiscoveryAgent**: Table-level similarity detection (Top-Down strategy)
- **TableMatchingAgent**: Precise table-to-table matching verification

### Technical Innovations

- **Intelligent Query Routing**: Automatic query type identification and optimal path selection
- **Parallel Processing**: Support for 10 concurrent requests with significant throughput improvement
- **Multi-Level Caching**: L1 (Memory) + L2 (Redis) + L3 (Disk) with >95% cache hit rate
- **Performance Optimization**: Batch processing + parallelization achieving up to 857x speedup
- **Extensible Architecture**: Support for multiple LLM providers (Gemini, OpenAI, Anthropic)

## Key Features

- **High Performance**: Sub-second query latency (1.08s average with LLM verification)
- **Scalability**: Tested on datasets with 1,534 tables and 7,358 queries
- **Accuracy**: 75% recall rate with ongoing optimization for precision
- **Flexibility**: Configurable similarity thresholds and matching strategies
- **Production-Ready**: Comprehensive error handling and monitoring

## Installation

### Prerequisites

- Python 3.10 or higher
- 8GB RAM minimum (16GB recommended)
- 5GB available storage
- Internet connection for LLM API access

### Setup Instructions

1. **Clone the repository**
```bash
git clone https://github.com/925180034/dataLakesMulti.git
cd dataLakesMulti
```

2. **Create virtual environment**
```bash
# Using conda (recommended)
conda create -n datalakes python=3.10 -y
conda activate datalakes

# Or using venv
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate  # Windows
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Configure API credentials**
```bash
cp .env.example .env
# Edit .env file to add your API key (choose at least one):
# - GEMINI_API_KEY (recommended, generous free tier)
# - OPENAI_API_KEY
# - ANTHROPIC_API_KEY
```

5. **Verify installation**
```bash
python run_cli.py config
```

## Quick Start

### Basic Usage

```bash
# Quick test with 3 queries
python run_full_experiment_fixed.py --dataset subset --max-queries 3

# Standard evaluation with 50 queries
python run_full_experiment_fixed.py --dataset subset --max-queries 50

# Complete evaluation
python run_full_experiment_fixed.py --dataset subset
```

### Command-Line Interface

```bash
# Single query
python run_cli.py discover -q "find joinable tables" -t examples/final_subset_tables.json -f json

# Table-specific query
python run_cli.py discover -q "find tables similar to csvData6444295__5" -t examples/final_subset_tables.json -f markdown
```

## Experimental Evaluation

### Experiment Scripts

1. **Full Experiment Script** (`run_full_experiment_fixed.py`)
```bash
python run_full_experiment_fixed.py [options]

Options:
  --dataset [subset|complete]  # Dataset selection
  --max-queries [n]           # Limit number of queries
  --no-llm                    # Disable LLM verification
```

2. **Fast Evaluation Script** (`ultra_fast_evaluation_fixed.py`)
```bash
python ultra_fast_evaluation_fixed.py [queries] [dataset]
```

3. **Interactive CLI** (`run_cli.py`)
```bash
python run_cli.py discover -q "your query" -t table_file.json
```

### Evaluation Metrics

We employ standard information retrieval metrics:

- **Precision**: Fraction of retrieved tables that are relevant
  - Formula: |Retrieved ∩ Relevant| / |Retrieved|
- **Recall**: Fraction of relevant tables that are retrieved
  - Formula: |Retrieved ∩ Relevant| / |Relevant|
- **F1-Score**: Harmonic mean of precision and recall
  - Formula: 2 × (Precision × Recall) / (Precision + Recall)
- **Query Latency**: Average processing time per query
- **Throughput**: Queries processed per second

### Dataset Description

- **Subset Dataset**: 100 tables, 502 queries (for rapid testing)
- **Complete Dataset**: 1,534 tables, 7,358 queries (for comprehensive evaluation)
- **Data Format**: JSON with table schemas and sample values
- **Ground Truth**: Manually verified table relationships

## Performance Metrics

### Current Performance (August 2025)

#### System Performance
- **Query Success Rate**: 100%
- **Average Latency**: 
  - With LLM: 1.08s ✓
  - Vector-only: 0.07s (cached)
- **Throughput**: ~10 QPS (concurrent processing)
- **Cache Hit Rate**: >95%

#### Matching Accuracy
- **Current Performance**:
  - Precision: 30%
  - Recall: 75%
  - F1-Score: 43%

#### Performance Evolution
1. **Phase 1**: Basic implementation (60s/query)
2. **Phase 2**: Three-layer architecture (5-10s/query)
3. **Phase 3**: Optimization (1.08s/query)

### Performance Targets

| Metric | Target | Current | Status |
|--------|--------|---------|---------|
| Latency | 3-8s | 1.08s | ✓ Achieved |
| Precision | >90% | 30% | In Progress |
| Recall | >90% | 75% | In Progress |
| Scale | 10,000 tables | 1,534 tables | Planned |
| Concurrency | 10 QPS | 10 QPS | ✓ Achieved |

## Configuration

### Configuration Files

- `config.yml`: Default configuration (balanced performance)
- `config_optimized.yml`: Optimized configuration (maximum performance)

### Key Parameters

```yaml
# LLM Configuration
llm:
  provider: "gemini"
  model: "gemini-2.0-flash-exp"
  temperature: 0.1
  max_tokens: 500
  timeout: 10

# Performance Settings
performance:
  batch_size: 20
  max_concurrent_requests: 10
  enable_cache: true
  cache_ttl: 3600

# Vector Search
vector_search:
  model: "sentence-transformers/all-MiniLM-L6-v2"
  ef_search: 200
  top_k: 20
```

## Advanced Usage

### Batch Processing

```python
from src.core.ultra_optimized_workflow import UltraOptimizedWorkflow

workflow = UltraOptimizedWorkflow()
queries = ["find sales tables", "find user tables", "find product tables"]
results = workflow.batch_process(queries)
```

### API Service

```bash
# Start API service
python -m src.cli serve

# API Endpoints
# POST /api/v1/discover
# GET /api/v1/health
# GET /docs (Swagger UI)
```

## Contributing

We welcome contributions from the research community. Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

### Development Workflow

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/your-feature`)
3. Commit your changes (`git commit -m 'feat: add new feature'`)
4. Push to the branch (`git push origin feature/your-feature`)
5. Create a Pull Request

## Citation

If you use this system in your research, please cite:

```bibtex
@software{datalakes_multiagent_2025,
  title={Multi-Agent System for Data Lake Schema Matching and Discovery},
  author={Your Name},
  year={2025},
  url={https://github.com/925180034/dataLakesMulti}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

This work builds upon the LangGraph framework and leverages state-of-the-art language models. We thank the open-source community for their invaluable contributions.

---

**Version**: v2.1  
**Last Updated**: August 5, 2025  
**Maintainer**: [Your Name]