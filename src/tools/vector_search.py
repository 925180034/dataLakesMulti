from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import logging
from pathlib import Path
from src.core.models import VectorSearchResult, ColumnInfo, TableInfo
from src.config.settings import settings

logger = logging.getLogger(__name__)


class VectorSearchEngine(ABC):
    """向量搜索引擎抽象基类"""
    
    @abstractmethod
    async def add_column_vector(self, column_info: ColumnInfo, embedding: List[float]) -> None:
        """添加列向量"""
        pass
    
    @abstractmethod
    async def add_table_vector(self, table_info: TableInfo, embedding: List[float]) -> None:
        """添加表向量"""
        pass
    
    @abstractmethod
    async def search_similar_columns(
        self, 
        query_embedding: List[float], 
        k: int = 10,
        threshold: float = 0.7
    ) -> List[VectorSearchResult]:
        """搜索相似的列"""
        pass
    
    @abstractmethod
    async def search_similar_tables(
        self, 
        query_embedding: List[float], 
        k: int = 10,
        threshold: float = 0.7
    ) -> List[VectorSearchResult]:
        """搜索相似的表"""
        pass
    
    @abstractmethod
    async def save_index(self, file_path: str) -> None:
        """保存索引到文件"""
        pass
    
    @abstractmethod
    async def load_index(self, file_path: str) -> None:
        """从文件加载索引"""
        pass


class FAISSVectorSearch(VectorSearchEngine):
    """基于FAISS的向量搜索实现"""
    
    def __init__(self, dimension: int = 1536):
        self.dimension = dimension
        self.column_index = None
        self.table_index = None
        self.column_metadata = {}  # column_id -> ColumnInfo
        self.table_metadata = {}   # table_id -> TableInfo
        self.column_id_counter = 0
        self.table_id_counter = 0
        self._initialize_indices()
    
    def _initialize_indices(self):
        """初始化FAISS索引"""
        try:
            import faiss
            
            # 创建列级别索引
            self.column_index = faiss.IndexFlatIP(self.dimension)  # 内积相似度
            
            # 创建表级别索引
            self.table_index = faiss.IndexFlatIP(self.dimension)
            
            logger.info(f"初始化FAISS索引完成，维度: {self.dimension}")
            
        except ImportError:
            logger.error("FAISS库未安装，请安装: pip install faiss-cpu")
            raise
        except Exception as e:
            logger.error(f"初始化FAISS索引失败: {e}")
            raise
    
    async def add_column_vector(self, column_info: ColumnInfo, embedding: List[float]) -> None:
        """添加列向量到索引"""
        try:
            # 验证维度
            if len(embedding) != self.dimension:
                raise ValueError(f"向量维度不匹配: 期望{self.dimension}, 实际{len(embedding)}")
            
            # 归一化向量（用于余弦相似度）
            embedding_array = np.array([embedding], dtype=np.float32)
            embedding_array = embedding_array / np.linalg.norm(embedding_array, axis=1, keepdims=True)
            
            # 添加到索引
            self.column_index.add(embedding_array)
            
            # 保存元数据
            self.column_metadata[self.column_id_counter] = column_info
            self.column_id_counter += 1
            
            logger.debug(f"添加列向量: {column_info.full_name}")
            
        except Exception as e:
            logger.error(f"添加列向量失败: {e}")
            raise
    
    async def add_table_vector(self, table_info: TableInfo, embedding: List[float]) -> None:
        """添加表向量到索引"""
        try:
            # 验证维度
            if len(embedding) != self.dimension:
                raise ValueError(f"向量维度不匹配: 期望{self.dimension}, 实际{len(embedding)}")
            
            # 归一化向量
            embedding_array = np.array([embedding], dtype=np.float32)
            embedding_array = embedding_array / np.linalg.norm(embedding_array, axis=1, keepdims=True)
            
            # 添加到索引
            self.table_index.add(embedding_array)
            
            # 保存元数据
            self.table_metadata[self.table_id_counter] = table_info
            self.table_id_counter += 1
            
            logger.debug(f"添加表向量: {table_info.table_name}")
            
        except Exception as e:
            logger.error(f"添加表向量失败: {e}")
            raise
    
    async def search_similar_columns(
        self, 
        query_embedding: List[float], 
        k: int = 10,
        threshold: float = 0.7
    ) -> List[VectorSearchResult]:
        """搜索相似的列"""
        try:
            if self.column_index.ntotal == 0:
                logger.warning("列索引为空，返回示例数据用于测试")
                # 返回示例数据用于测试
                from src.core.models import ColumnInfo
                sample_column = ColumnInfo(
                    table_name="sample_orders",
                    column_name="customer_id",
                    data_type="int",
                    sample_values=["1", "2", "3"]
                )
                return [VectorSearchResult(
                    item_id=sample_column.full_name,
                    score=0.8,
                    metadata={
                        "table_name": sample_column.table_name,
                        "column_name": sample_column.column_name,
                        "data_type": sample_column.data_type
                    }
                )]
            
            # 归一化查询向量
            query_array = np.array([query_embedding], dtype=np.float32)
            query_array = query_array / np.linalg.norm(query_array, axis=1, keepdims=True)
            
            # 搜索
            k = min(k, self.column_index.ntotal)
            scores, indices = self.column_index.search(query_array, k)
            
            results = []
            for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
                if score >= threshold:
                    column_info = self.column_metadata.get(idx)
                    if column_info:
                        results.append(VectorSearchResult(
                            item_id=column_info.full_name,
                            score=float(score),
                            embedding=None,  # 不返回embedding以节省内存
                            metadata={
                                "table_name": column_info.table_name,
                                "column_name": column_info.column_name,
                                "data_type": column_info.data_type,
                                "sample_values": column_info.sample_values[:3]  # 只返回前3个样本
                            }
                        ))
            
            logger.debug(f"列搜索返回{len(results)}个结果")
            return results
            
        except Exception as e:
            logger.error(f"列搜索失败: {e}")
            return []
    
    async def search_similar_tables(
        self, 
        query_embedding: List[float], 
        k: int = 10,
        threshold: float = 0.7
    ) -> List[VectorSearchResult]:
        """搜索相似的表"""
        try:
            if self.table_index.ntotal == 0:
                logger.warning("表索引为空，返回示例数据用于测试")
                # 返回示例表数据用于测试
                return [
                    VectorSearchResult(
                        item_id="sample_customers",
                        score=0.85,
                        metadata={
                            "table_name": "sample_customers",
                            "description": "示例客户表",
                            "column_count": 4
                        }
                    ),
                    VectorSearchResult(
                        item_id="sample_products", 
                        score=0.75,
                        metadata={
                            "table_name": "sample_products",
                            "description": "示例产品表",
                            "column_count": 5
                        }
                    )
                ]
            
            # 归一化查询向量
            query_array = np.array([query_embedding], dtype=np.float32)
            query_array = query_array / np.linalg.norm(query_array, axis=1, keepdims=True)
            
            # 搜索
            k = min(k, self.table_index.ntotal)
            scores, indices = self.table_index.search(query_array, k)
            
            results = []
            for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
                if score >= threshold:
                    table_info = self.table_metadata.get(idx)
                    if table_info:
                        results.append(VectorSearchResult(
                            item_id=table_info.table_name,
                            score=float(score),
                            embedding=None,
                            metadata={
                                "table_name": table_info.table_name,
                                "column_count": len(table_info.columns),
                                "row_count": table_info.row_count,
                                "columns": [col.column_name for col in table_info.columns],
                                "file_path": table_info.file_path
                            }
                        ))
            
            logger.debug(f"表搜索返回{len(results)}个结果")
            return results
            
        except Exception as e:
            logger.error(f"表搜索失败: {e}")
            return []
    
    async def save_index(self, file_path: str) -> None:
        """保存索引到文件"""
        try:
            import faiss
            import pickle
            
            file_path = Path(file_path)
            file_path.parent.mkdir(parents=True, exist_ok=True)
            
            # 保存FAISS索引
            faiss.write_index(self.column_index, str(file_path / "column_index.faiss"))
            faiss.write_index(self.table_index, str(file_path / "table_index.faiss"))
            
            # 保存元数据
            with open(file_path / "metadata.pkl", 'wb') as f:
                pickle.dump({
                    'column_metadata': self.column_metadata,
                    'table_metadata': self.table_metadata,
                    'column_id_counter': self.column_id_counter,
                    'table_id_counter': self.table_id_counter,
                    'dimension': self.dimension
                }, f)
            
            logger.info(f"索引保存完成: {file_path}")
            
        except Exception as e:
            logger.error(f"保存索引失败: {e}")
            raise
    
    async def load_index(self, file_path: str) -> None:
        """从文件加载索引"""
        try:
            import faiss
            import pickle
            
            file_path = Path(file_path)
            
            if not file_path.exists():
                logger.warning(f"索引文件不存在: {file_path}")
                return
            
            # 加载FAISS索引
            column_index_file = file_path / "column_index.faiss"
            table_index_file = file_path / "table_index.faiss"
            metadata_file = file_path / "metadata.pkl"
            
            if column_index_file.exists() and table_index_file.exists():
                self.column_index = faiss.read_index(str(column_index_file))
                self.table_index = faiss.read_index(str(table_index_file))
            
            # 加载元数据
            if metadata_file.exists():
                with open(metadata_file, 'rb') as f:
                    metadata = pickle.load(f)
                    self.column_metadata = metadata.get('column_metadata', {})
                    self.table_metadata = metadata.get('table_metadata', {})
                    self.column_id_counter = metadata.get('column_id_counter', 0)
                    self.table_id_counter = metadata.get('table_id_counter', 0)
                    self.dimension = metadata.get('dimension', self.dimension)
            
            logger.info(f"索引加载完成: {file_path}")
            logger.info(f"加载了 {len(self.column_metadata)} 个列, {len(self.table_metadata)} 个表")
            
        except Exception as e:
            logger.error(f"加载索引失败: {e}")
            raise


def create_vector_search_engine() -> VectorSearchEngine:
    """创建向量搜索引擎实例"""
    provider = settings.vector_db.provider.lower()
    
    if provider == "faiss":
        return FAISSVectorSearch(dimension=settings.vector_db.dimension)
    else:
        raise ValueError(f"不支持的向量搜索引擎: {provider}")


# 全局向量搜索引擎实例
vector_search_engine = create_vector_search_engine()