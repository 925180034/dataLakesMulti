# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## ⚠️ CRITICAL: Development Guidelines

### 🎯 核心架构: 多智能体数据湖发现系统
**系统采用多智能体协同架构**：
- **主架构文档**: `docs/COMPLETE_SYSTEM_ARCHITECTURE.md`
- **6个专门Agent**: OptimizerAgent, PlannerAgent, AnalyzerAgent, SearcherAgent, MatcherAgent, AggregatorAgent
- **三层加速工具**: 作为可选工具供Agent按需使用
- **数据湖发现**: 自动发现相关表、可连接数据、相似数据集

### ⚠️ 关键要求：LLM验证层（L3）绝对不能跳过
**重要**：本系统的核心价值在于多智能体的智能协同，特别是L3层的LLM验证：
- **L1层（元数据过滤）**: 快速初筛，减少候选集
- **L2层（向量搜索）**: 语义相似性匹配
- **L3层（LLM验证）**: ⚡**必须启用**⚡ 智能验证和精确匹配

**禁止事项**：
- ❌ 不要设置 `SKIP_LLM=true`
- ❌ 不要设置 `use_llm: false`
- ❌ 不要创建跳过LLM的测试脚本
- ❌ 不要在消融实验外禁用LLM

**LLM配置要求**（config.yml）：
```yaml
llm_matcher:
  enable_llm: true  # 必须为true
agents:
  matcher:
    enable_llm: true  # 必须为true
  analyzer:
    enable_llm: true  # 必须为true
```

### 📊 Current System Status (December 2024)
**架构版本**: v2.0 - 多智能体系统

| Component | Status | Description |
|-----------|--------|-------------|
| Multi-Agent System | ✅ Complete | 6个Agent协同工作 |
| Three-Layer Tools | ✅ Complete | 集成为可选工具 |
| Schema Matching | ✅ Working | 支持多维度匹配 |
| Performance | 🔄 Optimizing | 目标3秒内响应 |

### 🏗️ 系统架构概览
```
用户查询
    ↓
多智能体协同系统
├── OptimizerAgent: 系统优化配置
├── PlannerAgent: 策略规划决策
├── AnalyzerAgent: 数据结构分析
├── SearcherAgent: 候选搜索（可用Layer1+Layer2）
├── MatcherAgent: 精确匹配（可用Layer3）
└── AggregatorAgent: 结果聚合排序
    ↓
三层加速工具（Agent按需调用）
├── Layer 1: MetadataFilter (规则筛选 <10ms)
├── Layer 2: VectorSearch (向量搜索 10-50ms)
└── Layer 3: SmartLLMMatcher (LLM验证 1-3s)
    ↓
Schema Matching结果
```

### ✅ Recent Achievements (2025-08-10)
1. **Architecture Correction** - Reverted to correct pre-computed vector index design
2. **Performance Verified** - 0.002-8s/query (average 2.5s with LLM)
3. **HNSW Index Working** - Pre-built index with no query-time embedding computation
4. **System Stability** - 100% success rate on test queries
5. **Code Cleanup Complete** - Removed all redundant workflows and temporary files

### 📈 Performance Metrics (Verified on correct-architecture-v2)
- **Query Speed**: 0.002s (no LLM) / 2-8s (with LLM) ✅
- **Initialization Time**: 7-8s for 100 tables (acceptable) ✅
- **QPS**: 219 (no LLM) / 0.4-0.7 (with LLM) ✅
- **Success Rate**: 100% for JOIN, 60% for UNION ✅
- **Architecture**: Three-layer with pre-computed indices ✅

### 📋 Next Development Priorities
1. **Three-Layer Ablation Experiments** - Validate each layer's contribution
2. **Accuracy Improvement** - Reach >90% precision/recall target
3. **Scale Testing** - Expand from 100 to 10,000+ tables
4. **Production Deployment** - Docker containerization and API server
5. **Monitoring & Observability** - Add metrics and logging dashboards

### 🧪 Three-Layer Ablation Experiment Plan
**Purpose**: Validate the contribution of each layer and identify optimization opportunities

#### Experiment Configurations:
1. **L1 Only**: Metadata filtering only (baseline)
2. **L1+L2**: Metadata + Vector search (no LLM)
3. **L1+L2+L3**: Full pipeline (all layers)
4. **L2 Only**: Vector search only (ablation)
5. **L3 Only**: Direct LLM matching (ablation)

#### Metrics to Collect:
- Query latency per layer
- Accuracy metrics (Hit@1/3/5, Precision, Recall)
- Resource usage (CPU, Memory, API calls)
- Cost analysis (LLM tokens used)

See `docs/THREE_LAYER_ABLATION_PLAN.md` for detailed experiment design.

## Project Overview

This is a **Data Lake Multi-Agent System** for schema matching and data discovery. The system uses Large Language Model APIs and LangGraph framework to implement intelligent data matching through multi-agent collaboration.

### Core Capabilities
- **Schema Matching**: Find tables with similar column structures (for Join operations)
- **Data Instance Matching**: Discover semantically related tables (for Union operations)
- **Intelligent Routing**: Automatically select optimal processing strategy based on user intent

### Target Performance (per SYSTEM_ARCHITECTURE_AND_PLAN.md)
- **Query Speed**: 3-8 seconds (for 10,000+ tables)
- **Matching Accuracy**: >90% (Precision & Recall)
- **System Scale**: Support 10,000-50,000 tables
- **Concurrency**: Support 10 concurrent queries
- **Resource Requirements**: Single machine with 16GB RAM

## Development Commands

### Environment Setup
```bash
# Create conda environment
conda create -n data_lakes_multi python=3.10 -y
conda activate data_lakes_multi

# Install dependencies
pip install -r requirements.txt

# Configure API keys
cp .env.example .env
# Edit .env file to add your API keys
```

### Running the System

**Recommended Methods**:

1. **Ultra Fast Evaluation with Metrics (推荐)**:
```bash
# Enhanced evaluation script with Hit@K metrics
python ultra_fast_evaluation_with_metrics.py --task join --dataset subset --max-queries 10

# Full evaluation with both tasks
python ultra_fast_evaluation_with_metrics.py --task both --dataset subset --max-queries 5

# Skip LLM for faster testing (L1+L2 only)
python ultra_fast_evaluation_with_metrics.py --task join --dataset subset --max-queries 10 --skip-llm

# Options:
# --task: join, union, or both
# --dataset: subset (100 tables) or complete (1,534 tables)
# --max-queries: limit number of queries to test
# --skip-llm: skip Layer 3 LLM matching
# --verbose: show detailed results
```

2. **Ablation Experiments (for layer validation)**:
```bash
# Run three-layer ablation study
python ablation_study.py --task join --dataset subset --layers all

# Test specific layer combinations
python ablation_study.py --task join --dataset subset --layers L1
python ablation_study.py --task join --dataset subset --layers L1+L2
python ablation_study.py --task join --dataset subset --layers L1+L2+L3
```

3. **CLI Commands**:
```bash
# Single query discovery
python run_cli.py discover -q "find joinable tables" -t examples/final_subset_tables.json -f json

# Index data for faster search
python run_cli.py index-tables -t examples/final_subset_tables.json

# View configuration
python run_cli.py config
```

4. **Quick Test Script** (If exists):
```bash
# Check if script exists before running
[ -f "./run_experiment.sh" ] && ./run_experiment.sh batch 3
```

**Alternative Methods (for debugging)**:
```bash
# Method 1: Direct module execution (requires PYTHONPATH=.)
export PYTHONPATH=.
python -m src.cli discover -q "find joinable tables" -t examples/sample_tables.json -f markdown

# Method 2: Using shell wrapper (if available)
./datalakes discover -q "find joinable tables" -t examples/sample_tables.json -f markdown
```

### Testing
```bash
# Run all tests (now organized in tests/ directory)
python -m pytest tests/

# Run with verbose output
python -m pytest tests/ -v

# Run specific test file
python -m pytest tests/test_workflow.py

# Run integration tests
python -m pytest tests/test_integration.py

# Run performance tests
python -m pytest tests/test_scalable_search.py
```

### Code Quality
```bash
# Format code
black src/ tests/

# Sort imports
isort src/ tests/

# Type checking (if mypy is configured)
mypy src/
```

## Architecture Overview

The system is built on **LangGraph** with a multi-agent collaboration pattern:

### Core Agents
1. **PlannerAgent** (`src/agents/planner.py`): Task understanding and strategy selection
2. **ColumnDiscoveryAgent** (`src/agents/column_discovery.py`): Find matching columns (Bottom-Up strategy)
3. **TableAggregationAgent** (`src/agents/table_aggregation.py`): Aggregate column matches into table scores
4. **TableDiscoveryAgent** (`src/agents/table_discovery.py`): Find similar tables (Top-Down strategy)
5. **TableMatchingAgent** (`src/agents/table_matching.py`): Detailed table-to-table comparison

### Processing Strategies
- **Bottom-Up**: Column matching → Table aggregation (for precise joins)
- **Top-Down**: Table discovery → Table matching (for semantic similarity)

### Key Components
- **Workflow Engine**: `src/core/workflow.py` - LangGraph-based orchestration
- **Data Models**: `src/core/models.py` - Pydantic models for type safety
- **Search Tools**: `src/tools/` - Vector search and value search implementations
- **Configuration**: `config.yml` and `.env` for system settings

## Configuration Files

### Main Configuration (`config.yml`)
- LLM provider settings (OpenAI/Anthropic)
- Vector database configuration (FAISS/ChromaDB)
- Matching thresholds and performance tuning
- Logging and caching settings

### Environment Variables (`.env`)
Required (choose one):
- `GEMINI_API_KEY` (recommended) - Google Gemini API key
- `OPENAI_API_KEY` - OpenAI API key (with optional `OPENAI_BASE_URL`)
- `ANTHROPIC_API_KEY` - Anthropic Claude API key

Optional:
- `VECTOR_DB_PATH`, `INDEX_DB_PATH`, `CACHE_DIR`
- `DEBUG`, `CACHE_ENABLED`

## Working with Data

### Input Data Formats
The system expects JSON files with specific schemas:

**Tables** (`examples/sample_tables.json`):
```json
[
  {
    "name": "users",
    "columns": [
      {"name": "id", "type": "int", "sample_values": ["1", "2", "3"]},
      {"name": "email", "type": "string", "sample_values": ["user@example.com"]}
    ]
  }
]
```

**Columns** (`examples/sample_columns.json`):
```json
[
  {
    "table_name": "users",
    "name": "user_id", 
    "type": "int",
    "sample_values": ["1", "2", "3"]
  }
]
```

### API Endpoints
When running `python -m src.cli serve`:
- `POST /api/v1/discover` - Main discovery endpoint
- `GET /docs` - Interactive API documentation

## Common Development Tasks

### Adding New Agents
1. Inherit from `BaseAgent` in `src/agents/base.py`
2. Implement the `process` method
3. Add to workflow in `src/core/workflow.py`
4. Update routing logic if needed

### Modifying Search Tools
- Vector search: `src/tools/vector_search.py`
- Value search: `src/tools/value_search.py`
- Embedding utilities: `src/tools/embedding.py`

### Configuration Changes
- Edit `config.yml` for system parameters
- Update `src/config/settings.py` for new config options
- Modify prompt templates in `src/config/prompts.py`

## Project Structure (Updated)

The project has been organized with clean separation and unified documentation:

```
/root/dataLakesMulti/
├── src/                        # Source code
│   ├── agents/                 # Multi-agent implementations
│   ├── core/                   # Core workflow and models
│   ├── tools/                  # Search and utility tools
│   ├── config/                 # Configuration management
│   └── cli.py                  # Command-line interface
├── tests/                      # All test files (organized)
│   ├── test_*.py               # Unit and integration tests
│   ├── full_pipeline_test.py   # End-to-end tests
│   └── README.md               # Test documentation
├── docs/                       # 📚 Comprehensive documentation center
│   ├── README.md               # Documentation index
│   ├── SYSTEM_ARCHITECTURE_AND_PLAN.md # Unified architecture & plan
│   ├── QUICK_START.md          # Quick start guide
│   ├── Project-Design-Document.md # Original design
│   ├── lakebench_analysis.md   # Technical analysis
│   ├── architecture_diagram.md # System design diagrams
│   └── WEBTABLE_TEST_REPORT.md # Test reports
├── examples/                   # Sample data and demos
│   ├── final_subset_*.json     # 100-table test dataset
│   ├── final_complete_*.json   # 1,534-table full dataset
│   └── demos/                  # Demo scripts and utilities
├── experiment_results/         # All experiment results (auto-saved)
├── unified_experiment.py       # Unified experiment script with metrics
└── config.yml                 # Main configuration
```

## Important Dependencies

- **langgraph==0.5.4**: Multi-agent workflow orchestration
- **langchain==0.3.26**: LLM integration framework
- **google-generativeai==0.8.5**: Google Gemini API client
- **langchain-openai==0.3.28**: OpenAI integration (optional)
- **langchain-anthropic==0.3.17**: Anthropic integration (optional)
- **faiss-cpu==1.7.4**: Vector similarity search
- **chromadb==0.4.22**: Alternative vector database
- **sentence-transformers==2.7.0**: Text embeddings
- **pydantic==2.11.7**: Data validation and settings
- **fastapi==0.110.2**: Web API framework

## Project Status

### ✅ Fully Functional
- **Configuration System**: All CLI configuration commands work
- **Gemini API Integration**: Text generation and JSON output working properly
- **CLI Interface**: All command-line functionality operational
- **Embedding System**: Offline mode with virtual vectors (network-independent)
- **Basic Testing**: All pytest tests passing and organized in tests/ directory
- **Multiple Launch Methods**: run_cli.py, direct module execution, shell wrapper
- **Project Structure**: Clean organization with proper file separation
- **Documentation System**: Comprehensive docs in docs/ directory with unified architecture upgrade plan

### ⚠️ Partially Functional
- **Data Discovery**: Basic functionality works, but workflow has minor bugs
- **Vector Search**: Uses virtual vectors in offline mode (lower accuracy)
- **API Server**: Can start but not fully tested

### 🔧 Areas for Improvement
- **Performance Optimization**: Ready for HNSW indexing and Hungarian algorithm implementation
- **Full Network Functionality**: Requires internet access to download HuggingFace models
- **Workflow Error Handling**: Some edge cases need optimization
- **Architecture Upgrade**: Comprehensive upgrade plan available in docs/COMPREHENSIVE_ARCHITECTURE_UPGRADE.md

## Documentation Structure

The project maintains comprehensive documentation in the `docs/` directory:

### 📁 Key Documents
- **docs/SYSTEM_ARCHITECTURE_AND_PLAN.md**: Unified system architecture and implementation plan (simplified)
- **docs/QUICK_START.md**: Quick start guide with setup instructions
- **docs/Project-Design-Document.md**: Original project design and multi-agent framework
- **docs/lakebench_analysis.md**: Technical analysis and performance optimization insights
- **docs/architecture_diagram.md**: System architecture diagrams and design
- **docs/WEBTABLE_TEST_REPORT.md**: Testing reports and validation results
- **docs/README.md**: Complete documentation index and navigation

### 🎯 Performance Optimization Roadmap
The system has a simplified 3-phase implementation plan:
- **Phase 1 (Complete)**: Core functionality with HNSW indexing
- **Phase 2 (In Progress)**: Performance optimization and caching
- **Phase 3 (Planned)**: Full evaluation and deployment

Current achieved performance:
- **Query Speed**: ~15-20 seconds (optimized from 60s+)
- **Matching Accuracy**: ~85% precision, ~78% recall
- **System Scalability**: Tested on 1,534 tables
- **Success Rate**: >95% query completion

See `docs/SYSTEM_ARCHITECTURE_AND_PLAN.md` for implementation details.

## Quick Start Reference

For detailed setup instructions, see `QUICK_START.md`:
1. Setup Python 3.10+ environment
2. Install dependencies: `pip install -r requirements.txt`
3. Configure API keys in `.env` file
4. Test installation: `python run_cli.py config`
5. Run discovery: `python run_cli.py discover -q "your query" -t examples/sample_tables.json`

## Troubleshooting

### Common Issues
1. **Missing API Keys**: Ensure `.env` file has correct API keys
   - Copy from `.env.example` and add your API key
   - Recommended: Use `GEMINI_API_KEY` (free and stable)

2. **Import Errors**: Verify all dependencies in `requirements.txt` are installed
   - Run: `pip install -r requirements.txt`
   - Use virtual environment to avoid conflicts

3. **Module Not Found**: Use recommended launch method
   - Recommended: `python run_cli.py` (handles PYTHONPATH automatically)
   - Alternative: `export PYTHONPATH=. && python -m src.cli`

4. **Network Issues**: System works offline with limited functionality
   - Embedding system uses virtual vectors in offline mode
   - For full functionality, ensure internet access to download models

5. **Database Issues**: Check that data directories exist
   - `./data/vector_db` and `./data/index_db` directories
   - System will create them automatically if missing

6. **Memory Issues**: Reduce `batch_size` in `config.yml` for large datasets

### Debugging
- Enable debug mode: Set `DEBUG=true` in `.env`
- Check logs in `./logs/` directory
- Use verbose CLI output: Add `-v` flag to commands
- Test API connection: Use test scripts in `QUICK_START.md`

## Recent Updates

### Project Cleanup (July 30, 2024)
- ✅ **File Organization**: All test files moved to `tests/` directory
- ✅ **Documentation Consolidation**: Technical docs organized in `docs/` directory  
- ✅ **Architecture Unification**: Merged performance improvement plans into single comprehensive upgrade document
- ✅ **Structure Optimization**: Demo scripts organized in `examples/demos/`
- ✅ **Cache Cleanup**: Removed all Python cache files and temporary data

### System Status Update (December 2024)
- ❌ **Critical Issue**: Core matching logic returns empty predictions (0% accuracy)
- 🔄 **Phase 2 In Progress**: 30% complete, focusing on three-layer acceleration
- ⚠️ **Performance Gap**: 8.75s/query vs 3-8s target for 10,000+ tables

### Next Steps (MUST FOLLOW docs/SYSTEM_ARCHITECTURE_AND_PLAN.md)
The system development must strictly follow the 3-phase plan:

1. **Phase 1: Basic Infrastructure** ✅ COMPLETE
   - HNSW indexing implemented
   - Basic workflow established
   - Configuration optimized

2. **Phase 2: Performance Acceleration** 🔄 IN PROGRESS (30%)
   - **2.1 Three-Layer Index** (40% done) - Complete metadata filter and HNSW optimization
   - **2.2 Parallelization & Cache** (35% done) - Enable concurrent queries and Redis L2 cache
   - **2.3 Monitoring & Tuning** (20% done) - Implement performance profiling

3. **Phase 3: Scale Deployment** 📋 PLANNED
   - Distributed indexing design
   - Load balancing implementation
   - High availability guarantee

**CRITICAL**: All development MUST align with `docs/SYSTEM_ARCHITECTURE_AND_PLAN.md`. Do NOT create alternative architectures or deviate from the three-layer acceleration design.

## 🎯 Implementation Requirements

### Strict Design Adherence
1. **Architecture**: MUST use three-layer acceleration (Metadata → Vector → LLM)
2. **Performance**: MUST achieve 3-8 second query time for 10,000+ tables
3. **Accuracy**: MUST achieve >90% Precision and Recall
4. **Concurrency**: MUST support 10 concurrent queries
5. **Scale**: MUST handle 10,000-50,000 tables

### Data Input/Output Specification
**Input Requirements**:
- Tables data: JSON format with table_name, columns, sample_values
- Query specification: User query or specific table to match
- Ground truth: Expected matches for evaluation

**Output Requirements**:
- Matched tables: List of candidate tables with scores
- Evaluation metrics: Precision, Recall, F1-Score
- Performance metrics: Query time, throughput, resource usage

### Current Blockers (MUST FIX FIRST)
1. **Empty Predictions Bug**: `discover_data()` returns no matches
2. **Workflow Integration**: Data not flowing correctly through agents
3. **Vector Search Issues**: HNSW not returning relevant candidates
4. **LLM Matcher Problems**: Not generating proper match scores

### Development Workflow
1. Always test with `evaluate_with_metrics.py` to verify metrics
2. Use `examples/final_subset_*.json` for testing (100 tables)
3. Scale to `examples/final_complete_*.json` only after fixing core issues
4. Monitor performance with built-in profiling tools
5. Validate against ground truth data for accuracy

**Remember**: The goal is a production-ready system that handles real data lakes at scale, not a prototype. Every change must move toward the performance and accuracy targets defined in SYSTEM_ARCHITECTURE_AND_PLAN.md.