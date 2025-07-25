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
```bash
# CLI discovery command
python -m src.cli discover -q "find joinable tables" -t examples/sample_tables.json -f markdown

# Start API server
python -m src.cli serve

# Index data for faster search
python -m src.cli index-tables examples/sample_tables.json
python -m src.cli index-columns examples/sample_columns.json

# View configuration
python -m src.cli config
```

### Testing
```bash
# Run all tests
python -m pytest tests/

# Run with verbose output
python -m pytest tests/ -v

# Run specific test file
python -m pytest tests/test_workflow.py
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

## Troubleshooting

### Common Issues
1. **Missing API Keys**: Ensure `.env` file has correct API keys
2. **Import Errors**: Verify all dependencies in `requirements.txt` are installed
3. **Database Issues**: Check that `./data/vector_db` and `./data/index_db` directories exist
4. **Memory Issues**: Reduce `batch_size` in `config.yml` for large datasets

### Debugging
- Enable debug mode: Set `DEBUG=true` in `.env`
- Check logs in `./logs/` directory
- Use verbose CLI output: Add `-v` flag to commands