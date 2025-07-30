"""
ChromaDB向量搜索引擎实现 - 开发友好的选择
"""

from typing import List, Dict, Any, Optional, Tuple
import logging
import asyncio
from pathlib import Path

from src.tools.vector_search import VectorSearchEngine
from src.core.models import VectorSearchResult, ColumnInfo, TableInfo
from src.config.settings import settings

logger = logging.getLogger(__name__)


class ChromaVectorSearch(VectorSearchEngine):
    """基于ChromaDB的向量搜索实现
    
    优势:
    - 开发体验极佳，API简洁直观
    - 自动处理嵌入生成（可选）
    - 内置元数据过滤功能
    - 轻量级，适合中小规模数据
    """
    
    def __init__(self, dimension: int = 384, db_path: str = "./data/chroma_db"):
        self.dimension = dimension
        self.db_path = Path(db_path)
        self.db_path.mkdir(parents=True, exist_ok=True)
        
        self.client = None
        self.column_collection = None
        self.table_collection = None
        self._initialize_client()
    
    def _initialize_client(self):
        """初始化ChromaDB客户端"""
        try:
            import chromadb
            from chromadb.config import Settings as ChromaSettings
            
            # 创建持久化客户端
            self.client = chromadb.PersistentClient(
                path=str(self.db_path),
                settings=ChromaSettings(
                    anonymized_telemetry=False,
                    is_persistent=True
                )
            )
            
            # 获取或创建集合
            self.column_collection = self.client.get_or_create_collection(
                name="data_lakes_columns",
                metadata={"description": "Data lake column vectors"}
            )
            
            self.table_collection = self.client.get_or_create_collection(
                name="data_lakes_tables", 
                metadata={"description": "Data lake table vectors"}
            )
            
            logger.info(f"ChromaDB客户端初始化完成，路径: {self.db_path}")
            
        except ImportError:
            logger.error("ChromaDB库未安装，请安装: pip install chromadb")
            raise
        except Exception as e:
            logger.error(f"ChromaDB客户端初始化失败: {e}")
            raise
    
    async def add_column_vector(self, column_info: ColumnInfo, embedding: List[float]) -> None:
        """添加列向量到索引"""
        try:
            # 生成唯一ID
            doc_id = f"{column_info.table_name}.{column_info.column_name}"
            
            # 构建元数据
            metadata = {
                "table_name": column_info.table_name,
                "column_name": column_info.column_name,
                "data_type": column_info.data_type or "unknown",
                "sample_count": len(column_info.sample_values),
                "full_name": column_info.full_name
            }
            
            # 构建文档内容（用于全文搜索）
            document = f"Table: {column_info.table_name}, Column: {column_info.column_name}, Type: {column_info.data_type}"
            if column_info.sample_values:
                document += f", Samples: {', '.join(str(v) for v in column_info.sample_values[:3])}"
            
            # 添加到集合
            self.column_collection.upsert(
                ids=[doc_id],
                embeddings=[embedding],
                metadatas=[metadata],
                documents=[document]
            )
            
            logger.debug(f"添加列向量: {column_info.full_name}")
            
        except Exception as e:
            logger.error(f"添加列向量失败 {column_info.full_name}: {e}")
            raise
    
    async def add_table_vector(self, table_info: TableInfo, embedding: List[float]) -> None:
        """添加表向量到索引"""
        try:
            # 生成唯一ID
            doc_id = table_info.table_name
            
            # 构建元数据
            metadata = {
                "table_name": table_info.table_name,
                "column_count": len(table_info.columns),
                "row_count": table_info.row_count or 0,
                "data_types": ",".join(set(col.data_type or "unknown" for col in table_info.columns))
            }
            
            # 构建文档内容
            column_names = [col.column_name for col in table_info.columns]
            document = f"Table: {table_info.table_name}, Columns: {', '.join(column_names)}"
            
            # 添加到集合
            self.table_collection.upsert(
                ids=[doc_id],
                embeddings=[embedding],
                metadatas=[metadata],
                documents=[document]
            )
            
            logger.debug(f"添加表向量: {table_info.table_name}")
            
        except Exception as e:
            logger.error(f"添加表向量失败 {table_info.table_name}: {e}")
            raise
    
    async def search_similar_columns(
        self, 
        query_embedding: List[float], 
        k: int = 10,
        threshold: float = 0.7,
        data_type_filter: Optional[str] = None
    ) -> List[VectorSearchResult]:
        """搜索相似的列（支持元数据过滤）"""
        try:
            # 构建where条件
            where_condition = {}
            if data_type_filter:
                where_condition["data_type"] = data_type_filter
            
            # 执行搜索
            results = self.column_collection.query(
                query_embeddings=[query_embedding],
                n_results=k,
                where=where_condition if where_condition else None,
                include=["metadatas", "documents", "distances"]
            )
            
            # 转换结果
            search_results = []
            if results["ids"] and results["ids"][0]:
                for i, doc_id in enumerate(results["ids"][0]):
                    # ChromaDB返回距离，需要转换为相似度
                    distance = results["distances"][0][i]
                    similarity = 1 / (1 + distance)  # 距离转相似度
                    
                    if similarity >= threshold:
                        metadata = results["metadatas"][0][i]
                        result = VectorSearchResult(
                            id=doc_id,
                            content=metadata["full_name"],
                            similarity=similarity,
                            metadata=metadata
                        )
                        search_results.append(result)
            
            logger.debug(f"列搜索返回 {len(search_results)} 个结果，阈值: {threshold}")
            return search_results
            
        except Exception as e:
            logger.error(f"列搜索失败: {e}")
            return []
    
    async def search_similar_tables(
        self, 
        query_embedding: List[float], 
        k: int = 10,
        threshold: float = 0.7,
        min_columns: Optional[int] = None
    ) -> List[VectorSearchResult]:
        """搜索相似的表（支持列数过滤）"""
        try:
            # 构建where条件
            where_condition = {}
            if min_columns:
                where_condition["column_count"] = {"$gte": min_columns}
            
            # 执行搜索
            results = self.table_collection.query(
                query_embeddings=[query_embedding],
                n_results=k,
                where=where_condition if where_condition else None,
                include=["metadatas", "documents", "distances"]
            )
            
            # 转换结果
            search_results = []
            if results["ids"] and results["ids"][0]:
                for i, doc_id in enumerate(results["ids"][0]):
                    distance = results["distances"][0][i]
                    similarity = 1 / (1 + distance)
                    
                    if similarity >= threshold:
                        metadata = results["metadatas"][0][i]
                        result = VectorSearchResult(
                            id=doc_id,
                            content=metadata["table_name"],
                            similarity=similarity,
                            metadata=metadata
                        )
                        search_results.append(result)
            
            logger.debug(f"表搜索返回 {len(search_results)} 个结果，阈值: {threshold}")
            return search_results
            
        except Exception as e:
            logger.error(f"表搜索失败: {e}")
            return []
    
    async def save_index(self, file_path: str) -> None:
        """ChromaDB自动持久化，无需手动保存"""
        logger.info("ChromaDB自动持久化，索引已保存")
    
    async def load_index(self, file_path: str) -> None:
        """从持久化存储加载索引"""
        try:
            column_count = self.column_collection.count()
            table_count = self.table_collection.count()
            logger.info(f"加载索引完成 - 列: {column_count}, 表: {table_count}")
        except Exception as e:
            logger.warning(f"加载索引统计失败: {e}")
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """获取集合统计信息"""
        try:
            return {
                "columns": {
                    "count": self.column_collection.count(),
                    "status": "ready"
                },
                "tables": {
                    "count": self.table_collection.count(),
                    "status": "ready"
                }
            }
        except Exception as e:
            logger.error(f"获取统计信息失败: {e}")
            return {}


# 创建全局实例
def create_chroma_search() -> ChromaVectorSearch:
    """创建ChromaDB搜索引擎实例"""
    db_path = settings.vector_db.db_path.replace("vector_db", "chroma_db")
    return ChromaVectorSearch(db_path=db_path)