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
        self._precomputed_index = None
        self._precomputed_embeddings = None
        self._table_names_list = []  # 保存表名顺序
        
    def initialize(self, tables: List[TableInfo]):
        """Initialize the vector index"""
        if not self._initialized:
            # 首先尝试加载预计算的索引
            if self._load_precomputed_index(tables):
                self._initialized = True
                return
            # 如果没有预计算索引，构建新的
            asyncio.run(self._build_index(tables))
            self._initialized = True
            
    def _load_precomputed_index(self, tables: List[TableInfo]) -> bool:
        """加载预计算的FAISS索引"""
        import os
        import pickle
        import faiss
        
        index_file = os.getenv('USE_PERSISTENT_INDEX')
        embeddings_file = os.getenv('USE_PRECOMPUTED_EMBEDDINGS')
        
        if index_file and embeddings_file and os.path.exists(index_file) and os.path.exists(embeddings_file):
            try:
                # 加载预计算的FAISS索引
                with open(index_file, 'rb') as f:
                    self._precomputed_index = pickle.load(f)
                with open(embeddings_file, 'rb') as f:
                    self._precomputed_embeddings = pickle.load(f)
                
                # 构建表名列表（保持顺序）
                self._table_names_list = [t.table_name for t in tables]
                
                self.logger.info(f"✅ 加载预计算FAISS索引: {self._precomputed_index.ntotal} 个向量")
                return True
            except Exception as e:
                self.logger.warning(f"加载预计算索引失败: {e}")
                return False
        return False
    
    async def _build_index(self, tables: List[TableInfo]):
        """Build vector index for all tables"""
        # 🔥 关键修复：检查是否有预计算的嵌入
        import os
        import pickle
        
        precomputed_file = os.getenv('USE_PRECOMPUTED_EMBEDDINGS')
        precomputed_embeddings = {}
        
        if precomputed_file and os.path.exists(precomputed_file):
            try:
                with open(precomputed_file, 'rb') as f:
                    precomputed_embeddings = pickle.load(f)
                self.logger.info(f"📥 加载预计算嵌入: {len(precomputed_embeddings)} 个向量")
            except Exception as e:
                self.logger.warning(f"预计算嵌入加载失败: {e}")
        
        for table in tables:
            table_name = table.table_name
            
            # 优先使用预计算的嵌入
            if table_name in precomputed_embeddings:
                embedding = precomputed_embeddings[table_name].tolist()
                self.logger.debug(f"✅ 使用预计算嵌入: {table_name}")
            else:
                # 降级到实时计算
                table_text = self._table_to_text(table)
                embedding = await self.embedding_model.generate_text_embedding(table_text)
                self.logger.debug(f"🔄 实时计算嵌入: {table_name}")
            
            # Add to index
            await self.vector_engine.add_table_vector(table, embedding)
            
        used_precomputed = sum(1 for t in tables if t.table_name in precomputed_embeddings)
        self.logger.info(f"Built vector index for {len(tables)} tables "
                        f"({used_precomputed} 预计算, {len(tables)-used_precomputed} 实时计算)")
        
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
        
        # 如果使用预计算索引，直接搜索
        if self._precomputed_index and self._precomputed_embeddings:
            return self._search_with_precomputed_index(query_table, top_k, start_time)
        
        # Create query embedding
        if isinstance(query_table, dict):
            query_table_info = self._dict_to_table_info(query_table)
        else:
            query_table_info = query_table
        
        # 🔥 关键修复：优先使用预计算的查询嵌入
        import os
        import pickle
        
        query_embedding = None
        precomputed_file = os.getenv('USE_PRECOMPUTED_EMBEDDINGS')
        
        if precomputed_file and os.path.exists(precomputed_file):
            try:
                with open(precomputed_file, 'rb') as f:
                    precomputed_embeddings = pickle.load(f)
                query_table_name = query_table_info.table_name
                
                if query_table_name in precomputed_embeddings:
                    query_embedding = precomputed_embeddings[query_table_name].tolist()
                    self.logger.debug(f"✅ 使用预计算查询嵌入: {query_table_name}")
            except Exception as e:
                self.logger.warning(f"查询嵌入加载失败: {e}")
        
        # 如果没有预计算的嵌入，则实时计算
        if query_embedding is None:
            query_text = self._table_to_text(query_table_info)
            query_embedding = asyncio.run(
                self.embedding_model.generate_text_embedding(query_text)
            )
            self.logger.debug(f"🔄 实时计算查询嵌入: {query_table_info.table_name}")
        
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
    
    def _search_with_precomputed_index(self, query_table: Dict[str, Any], 
                                      top_k: int, start_time: float) -> List[Tuple[str, float]]:
        """使用预计算索引进行搜索"""
        import numpy as np
        
        # 获取查询表名
        query_table_name = query_table.get('table_name', query_table.get('name', ''))
        
        # 获取查询向量
        if query_table_name in self._precomputed_embeddings:
            query_vector = np.array(self._precomputed_embeddings[query_table_name]).astype('float32')
        else:
            # 如果没有预计算的查询向量，实时计算
            query_text = self._table_to_text_dict(query_table)
            query_vector = np.array(self.embedding_model.generate_text_embedding_sync(query_text)).astype('float32')
        
        # 搜索
        query_vector = query_vector.reshape(1, -1)
        distances, indices = self._precomputed_index.search(query_vector, min(top_k, self._precomputed_index.ntotal))
        
        # 构建结果
        results = []
        for idx, dist in zip(indices[0], distances[0]):
            if idx < len(self._table_names_list):
                table_name = self._table_names_list[idx]
                # 转换距离为相似度分数 (0-1)
                score = 1.0 / (1.0 + dist)  # 简单的距离到相似度转换
                results.append((table_name, score))
        
        elapsed = (time.time() - start_time) * 1000
        self.logger.info(f"Vector search with precomputed index took {elapsed:.1f}ms")
        
        return results
    
    def _table_to_text_dict(self, table: Dict[str, Any]) -> str:
        """将表字典转换为文本表示"""
        parts = [f"Table: {table.get('table_name', table.get('name', ''))}"]  
        
        for col in table.get('columns', [])[:20]:  # 限制前20列
            col_name = col.get('column_name', col.get('name', ''))
            col_type = col.get('data_type', col.get('type', 'unknown'))
            parts.append(f"Column: {col_name} ({col_type})")
            
            samples = col.get('sample_values', [])
            if samples:
                samples_str = ', '.join(str(v) for v in samples[:3])
                parts.append(f"  Samples: {samples_str}")
        
        return '\n'.join(parts)
    
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