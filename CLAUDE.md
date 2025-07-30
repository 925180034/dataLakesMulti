# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a **Data Lake Multi-Agent System** for schema matching and data discovery. The system uses Large Language Model APIs and LangGraph framework to implement intelligent data matching through multi-agent collaboration.

### Core Capabilities
- **Schema Matching**: Find tables with similar column structures (for Join operations)
- **Data Instance Matching**: Discover semantically related tables (for Union operations)
- **Intelligent Routing**: Automatically select optimal processing strategy based on user intent

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

**Recommended Method** (using run_cli.py wrapper):
```bash
# CLI discovery command
python run_cli.py discover -q "find joinable tables" -t examples/sample_tables.json -f markdown

# Start API server
python run_cli.py serve

# Index data for faster search
python run_cli.py index-tables examples/sample_tables.json
python run_cli.py index-columns examples/sample_columns.json

# View configuration
python run_cli.py config

# Get help
python run_cli.py --help
```

**Alternative Methods**:
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

## Project Structure (Post-Cleanup)

The project has been recently organized with proper file separation:

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
├── docs/                       # Comprehensive documentation
│   ├── COMPREHENSIVE_ARCHITECTURE_UPGRADE.md # Main upgrade plan
│   ├── lakebench_analysis.md   # Technical analysis
│   ├── architecture_diagram.md # System design
│   └── README.md               # Documentation index
├── examples/                   # Sample data and demos
│   ├── demos/                  # Demo scripts and utilities
│   └── sample_*.json           # Example datasets
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
- **docs/COMPREHENSIVE_ARCHITECTURE_UPGRADE.md**: Main architecture upgrade plan (4-phase implementation)
- **docs/lakebench_analysis.md**: Technical analysis and performance optimization insights
- **docs/architecture_diagram.md**: System architecture and design documentation
- **docs/WEBTABLE_TEST_REPORT.md**: Testing reports and validation results
- **docs/environment_requirements.md**: Environment setup and configuration
- **docs/README.md**: Complete documentation index

### 🎯 Performance Optimization Roadmap
The system has a comprehensive 4-phase upgrade plan targeting:
- **Query Speed**: 2.5s → 10-50ms (98% improvement)
- **Matching Accuracy**: 80% → 95% (18.75% improvement)  
- **System Scalability**: 1,000 → 100,000 tables (100x expansion)
- **Memory Efficiency**: 80% reduction
- **System Availability**: 99.9% stability

See `docs/COMPREHENSIVE_ARCHITECTURE_UPGRADE.md` for detailed implementation plan.

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

### Next Steps
The system is ready for the next phase of development following the 4-phase architecture upgrade plan:
1. **Phase 1**: HNSW indexing and Hungarian algorithm integration (Weeks 1-3)
2. **Phase 2**: LSH prefiltering and vectorized computation (Weeks 4-6)
3. **Phase 3**: Multi-feature fusion and graph analysis (Weeks 7-9)  
4. **Phase 4**: Distributed architecture and high availability (Weeks 10-12)

Refer to `docs/COMPREHENSIVE_ARCHITECTURE_UPGRADE.md` for detailed implementation guidance.