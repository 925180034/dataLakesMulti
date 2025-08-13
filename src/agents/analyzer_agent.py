"""
AnalyzerAgent - Table structure and semantic analysis
"""
from typing import Dict, List, Any
from src.agents.base_agent import BaseAgent
from src.core.state import WorkflowState, TableAnalysis


class AnalyzerAgent(BaseAgent):
    """
    Analyzer Agent responsible for understanding table structure and semantics
    """
    
    def __init__(self):
        super().__init__(
            name="AnalyzerAgent",
            description="Analyzes table structure, identifies patterns and key columns"
        )
        
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
        
        # Initialize analysis
        analysis = TableAnalysis(
            column_count=0,
            column_types={},
            key_columns=[],
            table_type='unknown',
            column_names=[],
            patterns={}
        )
        
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
    
    def validate_input(self, state: WorkflowState) -> bool:
        """
        Validate required inputs
        """
        if 'query_table' not in state or state['query_table'] is None:
            self.logger.error("Missing query_table in state")
            return False
        
        return True