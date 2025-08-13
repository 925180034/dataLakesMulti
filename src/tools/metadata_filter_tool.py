"""
MetadataFilterTool - Wrapper for Layer 1 metadata filtering
"""
import time
from typing import List, Dict, Any, Tuple
import logging
from src.tools.metadata_filter import MetadataFilter
from src.core.models import TableInfo, ColumnInfo


class MetadataFilterTool:
    """
    Wrapper for MetadataFilter to work with new agent architecture
    """
    
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.metadata_filter = MetadataFilter()
        self._initialized = False
        
    def initialize(self, tables: List[TableInfo]):
        """Initialize the metadata index"""
        if not self._initialized:
            self.metadata_filter.build_index(tables)
            self._initialized = True
            
    def filter(self, query_table: Dict[str, Any], 
               all_tables: List[Dict[str, Any]], 
               criteria: Dict[str, Any]) -> List[Tuple[str, float]]:
        """
        Filter tables based on metadata criteria
        
        Args:
            query_table: The query table dictionary
            all_tables: List of all candidate table dictionaries
            criteria: Filtering criteria
            
        Returns:
            List of (table_name, score) tuples
        """
        start_time = time.time()
        
        # Convert dict to TableInfo if needed
        if isinstance(query_table, dict):
            query_table_info = self._dict_to_table_info(query_table)
        else:
            query_table_info = query_table
            
        # If index not built yet, build it
        if not self._initialized and all_tables:
            table_infos = [self._dict_to_table_info(t) for t in all_tables]
            self.initialize(table_infos)
        
        # Get all table names for filtering
        table_names = [t.get('table_name') for t in all_tables if 'table_name' in t]
        
        # Use the existing MetadataFilter
        candidates = self.metadata_filter.filter_candidates(
            query_table_info,
            table_names,
            top_k=criteria.get('top_k', 100)
        )
        
        # Log performance
        elapsed_ms = (time.time() - start_time) * 1000
        self.logger.debug(f"Metadata filtering completed in {elapsed_ms:.1f}ms")
        
        return candidates
    
    def _dict_to_table_info(self, table_dict: Dict[str, Any]) -> TableInfo:
        """Convert table dictionary to TableInfo object"""
        columns = []
        for col_dict in table_dict.get('columns', []):
            col = ColumnInfo(
                table_name=table_dict.get('table_name', ''),
                column_name=col_dict.get('column_name', col_dict.get('name', '')),
                data_type=col_dict.get('data_type', col_dict.get('type', 'unknown')),
                sample_values=col_dict.get('sample_values', [])
            )
            columns.append(col)
        
        return TableInfo(
            table_name=table_dict.get('table_name', ''),
            columns=columns,
            row_count=table_dict.get('row_count', 0),
            file_path=table_dict.get('file_path', '')
        )