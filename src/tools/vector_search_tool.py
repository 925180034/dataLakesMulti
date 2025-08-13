"""
VectorSearchTool - Wrapper for Layer 2 vector similarity search
"""
import time
import asyncio
from typing import List, Dict, Any, Tuple
import logging
import numpy as np
from src.tools.vector_search import FAISSVectorSearch
from src.core.models import TableInfo, ColumnInfo
from src.tools.embedding import get_embedding_generator


class VectorSearchTool:
    """
    Wrapper for vector search to work with new agent architecture
    """
    
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.vector_engine = FAISSVectorSearch(dimension=384)  # Match all-MiniLM-L6-v2 embedding dimension
        self.embedding_model = get_embedding_generator()
        self._initialized = False
        
    def initialize(self, tables: List[TableInfo]):
        """Initialize the vector index"""
        if not self._initialized:
            asyncio.run(self._build_index(tables))
            self._initialized = True
            
    async def _build_index(self, tables: List[TableInfo]):
        """Build vector index for all tables"""
        for table in tables:
            # Create table embedding
            table_text = self._table_to_text(table)
            embedding = await self.embedding_model.generate_text_embedding(table_text)
            
            # Add to index
            await self.vector_engine.add_table_vector(table, embedding)
            
        self.logger.info(f"Built vector index for {len(tables)} tables")
        
    def search(self, query_table: Dict[str, Any], 
               all_tables: List[Dict[str, Any]], 
               top_k: int = 50) -> List[Tuple[str, float]]:
        """
        Search for similar tables using vector similarity
        
        Args:
            query_table: The query table dictionary
            all_tables: List of all candidate table dictionaries
            top_k: Number of top results to return
            
        Returns:
            List of (table_name, score) tuples
        """
        start_time = time.time()
        
        # Initialize if needed
        if not self._initialized and all_tables:
            table_infos = [self._dict_to_table_info(t) for t in all_tables]
            self.initialize(table_infos)
        
        # Create query embedding
        if isinstance(query_table, dict):
            query_table_info = self._dict_to_table_info(query_table)
        else:
            query_table_info = query_table
            
        query_text = self._table_to_text(query_table_info)
        query_embedding = asyncio.run(
            self.embedding_model.generate_text_embedding(query_text)
        )
        
        # Search similar tables
        results = asyncio.run(
            self.vector_engine.search_similar_tables(
                query_embedding,
                k=top_k,
                threshold=0.3  # Lower threshold for more candidates
            )
        )
        
        # Convert to expected format
        candidates = []
        for result in results:
            candidates.append((result.item_id, result.score))
        
        # Log performance
        elapsed_ms = (time.time() - start_time) * 1000
        self.logger.debug(f"Vector search completed in {elapsed_ms:.1f}ms")
        if elapsed_ms > 50:
            self.logger.warning(f"Vector search took {elapsed_ms:.1f}ms (target: 10-50ms)")
        
        return candidates
    
    def _table_to_text(self, table: TableInfo) -> str:
        """Convert table to text representation for embedding"""
        parts = [f"Table: {table.table_name}"]
        
        for col in table.columns[:20]:  # Limit to first 20 columns
            parts.append(f"Column: {col.column_name} ({col.data_type})")
            if col.sample_values:
                samples = ', '.join(str(v) for v in col.sample_values[:3])
                parts.append(f"  Samples: {samples}")
        
        return '\n'.join(parts)
    
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