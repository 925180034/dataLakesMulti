"""
AnalyzerAgent - Table structure and semantic analysis
"""
import asyncio
import json
from typing import Dict, List, Any
from src.agents.base_agent import BaseAgent
from src.core.state import WorkflowState, TableAnalysis
from src.config.prompts import get_agent_prompt, format_user_prompt


class AnalyzerAgent(BaseAgent):
    """
    Analyzer Agent responsible for understanding table structure and semantics
    """
    
    def __init__(self):
        super().__init__(
            name="AnalyzerAgent",
            description="Analyzes table structure, identifies patterns and key columns",
            use_llm=True  # Enable LLM for intelligent analysis
        )
        
        # Use centralized prompt from config
        self.system_prompt = get_agent_prompt("AnalyzerAgent", "system_prompt")
        
    def process(self, state: WorkflowState) -> WorkflowState:
        """
        Analyze the query table structure and characteristics
        
        Args:
            state: Current workflow state
            
        Returns:
            Updated state with table analysis
        """
        self.logger.info("Analyzing query table structure")
        
        # Get query table
        query_table = state.get('query_table')
        if not query_table:
            self.logger.error("No query table found in state")
            if 'errors' not in state:
                state['errors'] = []
            state['errors'].append("AnalyzerAgent: No query table to analyze")
            return state
        
        # Try LLM-based analysis first
        llm_analysis = self._get_llm_analysis(query_table)
        
        # Initialize analysis
        analysis = TableAnalysis(
            column_count=0,
            column_types={},
            key_columns=[],
            table_type='unknown',
            column_names=[],
            patterns={}
        )
        
        # Apply LLM insights if available
        if llm_analysis and 'table_type' in llm_analysis:
            analysis.table_type = llm_analysis.get('table_type', 'unknown')
            analysis.key_columns = llm_analysis.get('key_columns', [])
            analysis.patterns = {
                'semantic_domain': llm_analysis.get('semantic_domain', 'unknown'),
                'patterns': llm_analysis.get('patterns', []),
                'join_potential': llm_analysis.get('join_potential', {}),
                'insights': llm_analysis.get('insights', '')
            }
            
            self.logger.info(f"LLM Analysis Applied:")
            self.logger.info(f"  Table Type: {analysis.table_type}")
            self.logger.info(f"  Semantic Domain: {analysis.patterns.get('semantic_domain')}")
            self.logger.info(f"  Insights: {analysis.patterns.get('insights')}")
        else:
            self.logger.info("Using rule-based analysis (LLM unavailable or failed)")
        
        # Extract table name
        table_name = query_table.get('table_name', '').lower()
        
        # Analyze columns
        columns = query_table.get('columns', [])
        analysis.column_count = len(columns)
        
        for col in columns:
            # Get column name and type
            col_name = col.get('column_name', col.get('name', '')).lower()
            col_type = col.get('data_type', col.get('type', 'unknown')).lower()
            
            # Add to column names
            analysis.column_names.append(col_name)
            
            # Count column types
            if col_type not in analysis.column_types:
                analysis.column_types[col_type] = 0
            analysis.column_types[col_type] += 1
            
            # Identify key columns
            if any(key_pattern in col_name for key_pattern in ['_id', '_key', '_code', 'id', 'key']):
                analysis.key_columns.append(col_name)
                self.logger.debug(f"Identified key column: {col_name}")
            
            # Detect patterns
            if 'date' in col_name or 'time' in col_name:
                if 'temporal' not in analysis.patterns:
                    analysis.patterns['temporal'] = []
                analysis.patterns['temporal'].append(col_name)
            
            if 'amount' in col_name or 'price' in col_name or 'cost' in col_name:
                if 'monetary' not in analysis.patterns:
                    analysis.patterns['monetary'] = []
                analysis.patterns['monetary'].append(col_name)
            
            if 'name' in col_name or 'description' in col_name:
                if 'descriptive' not in analysis.patterns:
                    analysis.patterns['descriptive'] = []
                analysis.patterns['descriptive'].append(col_name)
        
        # Determine table type based on name and structure
        if 'dim_' in table_name or '_dim' in table_name:
            analysis.table_type = 'dimension'
            self.logger.info("Identified as DIMENSION table")
        elif 'fact_' in table_name or '_fact' in table_name:
            analysis.table_type = 'fact'
            self.logger.info("Identified as FACT table")
        elif 'lookup' in table_name or 'ref' in table_name:
            analysis.table_type = 'lookup'
            self.logger.info("Identified as LOOKUP table")
        else:
            # Try to infer from structure
            if len(analysis.key_columns) == 1 and analysis.column_count < 10:
                analysis.table_type = 'dimension'
                self.logger.info("Inferred as DIMENSION table (single key, few columns)")
            elif len(analysis.key_columns) > 1 and 'monetary' in analysis.patterns:
                analysis.table_type = 'fact'
                self.logger.info("Inferred as FACT table (multiple keys, monetary data)")
            else:
                analysis.table_type = 'unknown'
                self.logger.info("Could not determine table type")
        
        # Log analysis results
        self.logger.info(f"Table analysis complete:")
        self.logger.info(f"  - Column count: {analysis.column_count}")
        self.logger.info(f"  - Column types: {analysis.column_types}")
        self.logger.info(f"  - Key columns: {analysis.key_columns}")
        self.logger.info(f"  - Table type: {analysis.table_type}")
        self.logger.info(f"  - Patterns detected: {list(analysis.patterns.keys())}")
        
        # Update state
        state['analysis'] = analysis
        
        # Update metrics
        if state.get('metrics'):
            state['metrics'].candidates_generated = analysis.column_count
        
        return state
    
    def _get_llm_analysis(self, query_table: Dict[str, Any]) -> dict:
        """Use LLM to analyze table structure"""
        # Format table info for prompt
        columns = query_table.get('columns', [])
        column_info = []
        for col in columns[:20]:  # Limit to first 20 columns
            col_info = f"{col.get('column_name', col.get('name', 'unknown'))} ({col.get('data_type', col.get('type', 'unknown'))})"
            if col.get('sample_values'):
                samples = col['sample_values'][:3]
                col_info += f" [{', '.join(str(s) for s in samples)}]"
            column_info.append(col_info)
        
        # Basic statistics
        stats = {
            'column_count': len(columns),
            'has_id_columns': any('id' in str(col.get('column_name', col.get('name', ''))).lower() for col in columns),
            'has_date_columns': any('date' in str(col.get('data_type', col.get('type', ''))).lower() for col in columns)
        }
        
        # Use centralized user prompt template
        prompt = format_user_prompt(
            "AnalyzerAgent",
            table_name=query_table.get('table_name', 'unknown'),
            columns=', '.join(column_info),
            sample_data=str(columns[:3]) if columns else 'N/A',
            statistics=str(stats)
        )
        
        try:
            # Handle async call in sync context
            try:
                loop = asyncio.get_running_loop()
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(asyncio.run, self.call_llm_json(prompt, self.system_prompt))
                    response = future.result(timeout=10)
            except RuntimeError:
                response = asyncio.run(self.call_llm_json(prompt, self.system_prompt))
            
            return response
        except Exception as e:
            self.logger.warning(f"LLM analysis failed: {e}, using rule-based fallback")
            return {}
    
    def validate_input(self, state: WorkflowState) -> bool:
        """
        Validate required inputs
        """
        if 'query_table' not in state or state['query_table'] is None:
            self.logger.error("Missing query_table in state")
            return False
        
        return True