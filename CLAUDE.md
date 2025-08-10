# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## ⚠️ CRITICAL: Development Guidelines

### 🎯 MUST FOLLOW: System Architecture and Plan
**All development MUST strictly adhere to**: `docs/SYSTEM_ARCHITECTURE_AND_PLAN.md`
- This is the authoritative design document
- Do NOT deviate from the three-layer acceleration architecture
- Do NOT create new architectures or approaches
- Always verify changes align with the plan

### 🚨 MANDATORY: Real System Validation - NO SIMULATION ALLOWED

**CRITICAL REQUIREMENT**: ALL experiments and evaluations MUST use REAL systems, REAL data, and REAL metrics. NO simulation or mock systems allowed.

#### ✅ Required Real System Components
**You MUST verify these components are actually running:**

1. **Real LLM API Calls** 
   ```python
   # ✅ REQUIRED: Real Gemini/OpenAI API calls
   llm_response = await self.llm_client.generate_json(prompt)
   
   # ❌ FORBIDDEN: Mock/simulation 
   await asyncio.sleep(0.05)  # This is FAKE!
   ```

2. **Real SentenceTransformer Models**
   ```python
   # ✅ REQUIRED: Real model loading
   from sentence_transformers import SentenceTransformer
   model = SentenceTransformer('all-MiniLM-L6-v2')
   embedding = model.encode(text)
   
   # ❌ FORBIDDEN: Hash-based fake vectors
   hash_obj = hashlib.sha256(text.encode())  # This is FAKE!
   ```

3. **Real Data Processing**
   - Use actual data from `examples/separated_datasets/`
   - Process real JSON files with real table structures
   - NO artificial or generated test data

4. **Real Evaluation Metrics**
   - Calculate metrics from actual prediction vs ground truth comparisons
   - NO hardcoded or simulated scores
   - Time measurements must reflect actual processing time

#### 🔍 Verification Checklist - Run BEFORE Every Experiment

**Before running ANY experiment, verify:**

```bash
# 1. Check real API keys are loaded
echo "API Key length: ${#GEMINI_API_KEY}"  # Should be 39+ characters

# 2. Check real model exists  
ls -la ~/.cache/torch/sentence_transformers/  # Should show downloaded models

# 3. Check real data exists
ls -la examples/separated_datasets/union_subset/  # Should show real JSON files

# 4. Verify NOT using simulation flags
echo "SKIP_LLM: $SKIP_LLM"  # Should be 'false' or empty
```

#### 📊 Expected Real Processing Times

**WARNING: If you see these times, you're using SIMULATION:**
- Layer2 (Vector): <10ms ❌ (Real should be 0.5-5s)
- Layer3 (LLM): <100ms ❌ (Real should be 1-8s)
- Total per query: <1s ❌ (Real should be 2-10s)

**REAL processing times should be:**
- Layer2 (SentenceTransformer): 0.5-5 seconds per query
- Layer3 (LLM API): 1-8 seconds per query  
- Total per query: 2-10 seconds for complete processing

#### 🚨 Red Flags - Signs of Simulation

**If you see ANY of these, STOP immediately:**
- Processing time <0.1s per query
- "模拟LLM" or "模拟" in logs
- `asyncio.sleep()` used for delays
- Hash-based vector generation
- Hardcoded similarity scores
- Perfect round numbers in results (50.0%, 100.0%)

#### ✅ Approved Real Systems Only

**ONLY use these verified real components:**
- `src/utils/llm_client.py` - GeminiClient with real API
- `src/tools/embedding.py` - SentenceTransformerEmbeddingGenerator  
- `src/core/workflow.py` - DataLakesWorkflow with real agents
- `real_three_layer_ablation.py` - Verified real system script

**NEVER use these simulation files:**
- `optimized_three_layer_system.py` - Contains simulation code
- Any file with "mock", "fake", "simulate" in comments

#### 📋 Mandatory Pre-Experiment Validation

**Run this validation BEFORE every experiment:**

```python
# Validation script - add to beginning of every experiment
def validate_real_system():
    # 1. Check API keys
    api_key = os.getenv('GEMINI_API_KEY')
    assert api_key and len(api_key) > 30, "Real API key required"
    
    # 2. Check SentenceTransformer model
    model_path = Path.home() / '.cache/torch/sentence_transformers'
    assert model_path.exists(), "Real SentenceTransformer cache required"
    
    # 3. Check no simulation flags
    skip_llm = os.getenv('SKIP_LLM', '').lower()
    assert skip_llm in ['false', ''], f"SKIP_LLM must be false, got: {skip_llm}"
    
    # 4. Check real data exists
    data_path = Path("examples/separated_datasets")
    assert data_path.exists(), "Real dataset path required"
    
    print("✅ Real system validation passed")
```

#### 🔬 Truth Verification Protocol

**For EVERY experiment result, provide proof:**

1. **Time Evidence**: Show actual processing times >1s per query
2. **API Evidence**: Show real API call logs/responses  
3. **Model Evidence**: Show SentenceTransformer model loading logs
4. **Data Evidence**: Show actual data being processed from real files
5. **Metric Evidence**: Show calculation from actual predictions vs ground truth

**Format for reporting:**
```
🔍 Real System Evidence:
- API Calls: X successful Gemini calls (Y seconds each)  
- Vector Generation: SentenceTransformer processed X tables (Y seconds total)
- Data Source: Loaded X tables from examples/separated_datasets/
- Processing Time: Total X.XX seconds (breakdown: L1=Xs, L2=Xs, L3=Xs)
- Metrics: Calculated from X predictions vs X ground truth entries
```

### 📊 Current System Status (August 2025)
**Overall Completion**: ~60% of target requirements
**Current Focus**: 🎯 **三层架构优化任务**

| Component | Status | Completion | Priority |
|-----------|--------|------------|----------|
| Phase 1: Basic Infrastructure | ✅ Complete | 100% | - |
| Phase 2: 三层架构优化 | 🔄 In Progress | 70% | **HIGH** |
| Phase 3: Scale Deployment | 📋 Planned | 0% | MEDIUM |

### ✅ Recent Achievements (2025-08-10)
1. **三层架构诊断完成** - 发现并诊断了三层架构的实现问题
2. **消融实验框架建立** - 完整的三层架构消融实验系统
3. **LLM层智能调用** - 基于置信度的智能LLM调用机制（30-50%调用率）
4. **问题根因分析** - 明确识别了各层存在的技术问题
5. **LakeBench技术分析** - 获得HNSW、匈牙利算法等优化方案

### 📈 Real System Performance Metrics (Verified 2025-08-10)

**🔥 TRUE Three-Layer Architecture (Real SentenceTransformer + Real LLM API)**:
- **Layer1 Only**: F1=0.0%, 查询时间=0.000s (元数据过滤基准)
- **Layer1+2 Real**: F1=0.0%, 查询时间=3.434s (真实SentenceTransformer向量搜索)
- **Complete Real**: F1=0.0%, 查询时间=4.788s (完整系统含真实Gemini LLM)

**Real Processing Time Breakdown per Query**:
- Layer1 (Metadata Filter): ~0.000s
- Layer2 (SentenceTransformer): ~3.434s (包含首次模型加载14.55s)
- Layer3 (Gemini LLM API): ~1.353s (2.89-4.73s range)

**🚨 CRITICAL ISSUES IDENTIFIED**:
1. **Accuracy Problem**: All configurations show 0% F1 score - Ground Truth parsing needs fix
2. **Performance Gap**: Real system is 84x slower than simulated (4.8s vs 0.057s)
3. **LLM Cost**: Each query requires 1-5 seconds of API time
4. **Vector Cost**: Real embedding generation needs 3+ seconds per query

**Next Priority**: Fix Ground Truth format parsing to get accurate F1 scores

### 🎯 当前开发优先级（基于真实系统验证结果）

**🚨 CRITICAL (Must Fix First)**:
1. **修复Ground Truth解析问题** - 当前0% F1分数的根本原因
   - 检查ground truth数据格式 (每个候选表一条记录 vs 列表格式)
   - 修复预测结果与ground truth的匹配逻辑
   - 验证表名标准化和ID匹配规则

**⚡ HIGH PRIORITY (Performance Optimization)**:
2. **Layer2向量搜索优化** - 从3.4秒优化到<1秒
   - 批量向量化处理 (减少API调用次数)
   - 向量缓存机制 (避免重复计算)
   - HNSW索引预构建 (启动时一次性构建)

3. **Layer3 LLM调用优化** - 从1.4秒优化到<0.5秒
   - 智能LLM调用策略 (仅在必要时调用)
   - 批量LLM处理 (多个查询合并)
   - 响应缓存机制 (相似查询复用结果)

**🔧 MEDIUM PRIORITY (Architecture Enhancement)**:
4. **实现HNSW预索引** - 系统启动时预构建全部向量索引
5. **添加匈牙利算法精确匹配** - Layer3精度提升
6. **元数据过滤层智能化** - Layer1候选数量自适应调整

**📊 SUCCESS CRITERIA**:
- Target F1: >80% (Fix ground truth parsing first)
- Target Query Time: <2 seconds per query
- Target LLM Usage: <30% of queries (intelligent calling)

## 🎯 IMMEDIATE NEXT STEPS

### Step 1: Fix Ground Truth Parsing (URGENT)
**Current Issue**: 0% F1 scores due to ground truth format mismatch
**Action Required**: 
```python
# Check ground truth format in examples/separated_datasets/union_subset/ground_truth.json
# Fix the parsing logic in real_three_layer_ablation.py lines 400-406
# Expected ground truth format vs actual format mismatch needs resolution
```

### Step 2: Optimize Real System Performance
**Current Issue**: 4.8s per query vs 2s target
**Action Required**:
- Layer2: Batch SentenceTransformer processing (reduce 3.4s → <1s)  
- Layer3: Smart LLM caching (reduce 1.4s → <0.5s)
- Overall: 4.8s → <2s per query target

**Success Validation**: Run `real_three_layer_ablation.py` and verify:
- ✅ F1 scores >50% (not 0%)
- ✅ Processing times <3s per query
- ✅ Real API calls (not simulation)

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

**🔥 REQUIRED: Use REAL System Only**

1. **Real Three-Layer Architecture Testing (RECOMMENDED)**:
```bash
# Verified real system with actual SentenceTransformer + LLM API
python real_three_layer_ablation.py

# This script GUARANTEES:
# ✅ Real SentenceTransformer model loading (~8s initialization)
# ✅ Real Gemini LLM API calls (2-5s per query)
# ✅ Real data from examples/separated_datasets/
# ✅ Real processing times (4-5s per query)
# ✅ True ablation study results

# Expected Real Results:
# - Layer1 Only: ~0.000s per query
# - Layer1+2 Real: ~3.434s per query (real vector search)
# - Full Real 3-Layer: ~4.788s per query (with LLM)
```

2. **Pre-Experiment Validation (MANDATORY)**:
```bash
# Run BEFORE any experiment to ensure real system
python -c "
import os
from pathlib import Path

# Validate real components exist
api_key = os.getenv('GEMINI_API_KEY')
print(f'API Key: {len(api_key) if api_key else 0} characters')

model_cache = Path.home() / '.cache/torch/sentence_transformers'
print(f'SentenceTransformer Cache: {model_cache.exists()}')

data_path = Path('examples/separated_datasets/union_subset')
print(f'Real Data Path: {data_path.exists()}')

skip_llm = os.getenv('SKIP_LLM', '').lower()
print(f'SKIP_LLM: {skip_llm} (should be false/empty)')
"
```

3. **⚠️ DEPRECATED/FORBIDDEN Methods**:
```bash
# ❌ DO NOT USE - Contains simulation code
# python evaluate_with_metrics.py  # May use mock systems
# python optimized_three_layer_system.py  # Contains asyncio.sleep() simulation
# python verify_three_layer_llm.py  # Previous verification, use real_three_layer_ablation.py instead
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