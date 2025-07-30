"""
混合索引策略 - 结合向量搜索和传统索引的优势
"""

from typing import List, Dict, Any, Optional, Tuple, Union
import logging
import asyncio
from pathlib import Path
import sqlite3
import json

from src.tools.vector_search import VectorSearchEngine
from src.core.models import VectorSearchResult, ColumnInfo, TableInfo
from src.config.settings import settings

logger = logging.getLogger(__name__)


class HybridIndexEngine:
    """混合索引引擎
    
    结合多种索引技术:
    1. 向量索引 - 语义相似度搜索
    2. 倒排索引 - 精确文本匹配
    3. SQLite索引 - 结构化查询和过滤
    4. Bloom过滤器 - 快速存在性检查
    """
    
    def __init__(self, 
                 vector_engine: VectorSearchEngine,
                 db_path: str = "./data/hybrid_index.db"):
        self.vector_engine = vector_engine
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 文本索引
        self.text_index = {}  # 简单的倒排索引
        self.bloom_filter = set()  # 简化的Bloom过滤器
        
        # 初始化SQLite
        self._init_sqlite()
    
    def _init_sqlite(self):
        """初始化SQLite数据库用于结构化查询"""
        try:
            self.conn = sqlite3.connect(str(self.db_path))
            self.conn.execute("PRAGMA journal_mode=WAL")  # 提升并发性能
            
            # 创建列表
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS columns (
                    id INTEGER PRIMARY KEY,
                    table_name TEXT NOT NULL,
                    column_name TEXT NOT NULL,
                    data_type TEXT,
                    sample_values TEXT,
                    full_name TEXT UNIQUE,
                    vector_id TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # 创建表表
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS tables (
                    id INTEGER PRIMARY KEY,
                    table_name TEXT UNIQUE NOT NULL,
                    column_count INTEGER,
                    row_count INTEGER,
                    column_names TEXT,
                    data_types TEXT,
                    vector_id TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # 创建索引
            self.conn.execute("CREATE INDEX IF NOT EXISTS idx_column_table ON columns(table_name)")
            self.conn.execute("CREATE INDEX IF NOT EXISTS idx_column_type ON columns(data_type)")
            self.conn.execute("CREATE INDEX IF NOT EXISTS idx_table_columns ON tables(column_count)")
            
            # 创建全文搜索表
            self.conn.execute("""
                CREATE VIRTUAL TABLE IF NOT EXISTS column_fts USING fts5(
                    full_name, table_name, column_name, sample_values,
                    content='columns', content_rowid='id'
                )
            """)
            
            self.conn.commit()
            logger.info("SQLite索引初始化完成")
            
        except Exception as e:
            logger.error(f"SQLite初始化失败: {e}")
            raise
    
    async def add_column(self, column_info: ColumnInfo, embedding: List[float]) -> None:
        """添加列到混合索引"""
        try:
            # 1. 添加到向量索引
            await self.vector_engine.add_column_vector(column_info, embedding)
            
            # 2. 添加到SQLite
            sample_values_json = json.dumps(column_info.sample_values)
            self.conn.execute("""
                INSERT OR REPLACE INTO columns 
                (table_name, column_name, data_type, sample_values, full_name, vector_id)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                column_info.table_name,
                column_info.column_name,
                column_info.data_type,
                sample_values_json,
                column_info.full_name,
                column_info.full_name
            ))
            
            # 3. 添加到全文搜索
            self.conn.execute("""
                INSERT OR REPLACE INTO column_fts 
                (full_name, table_name, column_name, sample_values)
                VALUES (?, ?, ?, ?)
            """, (
                column_info.full_name,
                column_info.table_name,
                column_info.column_name,
                ' '.join(str(v) for v in column_info.sample_values[:5])
            ))
            
            # 4. 更新文本索引
            self._update_text_index(column_info)
            
            # 5. 添加到Bloom过滤器
            self.bloom_filter.add(column_info.full_name)
            
            self.conn.commit()
            logger.debug(f"添加列到混合索引: {column_info.full_name}")
            
        except Exception as e:
            logger.error(f"添加列到混合索引失败 {column_info.full_name}: {e}")
            raise
    
    async def add_table(self, table_info: TableInfo, embedding: List[float]) -> None:
        """添加表到混合索引"""
        try:
            # 1. 添加到向量索引
            await self.vector_engine.add_table_vector(table_info, embedding)
            
            # 2. 添加到SQLite
            column_names_json = json.dumps([col.column_name for col in table_info.columns])
            data_types_json = json.dumps([col.data_type for col in table_info.columns])
            
            self.conn.execute("""
                INSERT OR REPLACE INTO tables 
                (table_name, column_count, row_count, column_names, data_types, vector_id)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                table_info.table_name,
                len(table_info.columns),
                table_info.row_count,
                column_names_json,
                data_types_json,
                table_info.table_name
            ))
            
            self.conn.commit()
            logger.debug(f"添加表到混合索引: {table_info.table_name}")
            
        except Exception as e:
            logger.error(f"添加表到混合索引失败 {table_info.table_name}: {e}")
            raise
    
    def _update_text_index(self, column_info: ColumnInfo):
        """更新倒排索引"""
        terms = [
            column_info.table_name.lower(),
            column_info.column_name.lower(),
            column_info.data_type.lower() if column_info.data_type else ""
        ]
        
        # 添加样本值
        for value in column_info.sample_values[:5]:
            terms.append(str(value).lower())
        
        # 构建倒排索引
        for term in terms:
            if term and len(term) > 1:
                if term not in self.text_index:
                    self.text_index[term] = set()
                self.text_index[term].add(column_info.full_name)
    
    async def search_columns_hybrid(
        self,
        query: str = None,
        query_embedding: List[float] = None,
        k: int = 10,
        threshold: float = 0.7,
        filters: Dict[str, Any] = None
    ) -> List[VectorSearchResult]:
        """混合搜索列
        
        Args:
            query: 文本查询
            query_embedding: 向量查询
            k: 返回结果数量
            threshold: 相似度阈值
            filters: 过滤条件 {"data_type": "string", "table_name": "users"}
        """
        results = []
        
        try:
            # 1. 快速过滤 - 使用Bloom过滤器和SQLite
            candidate_ids = await self._get_candidates(filters)
            
            if query_embedding:
                # 2. 向量搜索
                vector_results = await self.vector_engine.search_similar_columns(
                    query_embedding, k * 2, threshold * 0.8  # 放宽条件获取更多候选
                )
                
                # 3. 合并结果
                for result in vector_results:
                    if not candidate_ids or result.id in candidate_ids:
                        results.append(result)
            
            # 4. 文本搜索补充
            if query and len(results) < k:
                text_results = await self._text_search_columns(query, k - len(results))
                results.extend(text_results)
            
            # 5. 重排序和去重
            results = self._deduplicate_and_rank(results)
            
            return results[:k]
            
        except Exception as e:
            logger.error(f"混合列搜索失败: {e}")
            return []
    
    async def _get_candidates(self, filters: Dict[str, Any] = None) -> Optional[set]:
        """基于过滤条件获取候选ID"""
        if not filters:
            return None
        
        try:
            conditions = []
            params = []
            
            for key, value in filters.items():
                if key in ["data_type", "table_name"]:
                    conditions.append(f"{key} = ?")
                    params.append(value)
            
            if not conditions:
                return None
            
            query = f"SELECT full_name FROM columns WHERE {' AND '.join(conditions)}"
            cursor = self.conn.execute(query, params)
            
            candidates = {row[0] for row in cursor.fetchall()}
            logger.debug(f"SQLite过滤返回 {len(candidates)} 个候选")
            
            return candidates
            
        except Exception as e:
            logger.error(f"候选过滤失败: {e}")
            return None
    
    async def _text_search_columns(self, query: str, limit: int) -> List[VectorSearchResult]:
        """全文搜索补充"""
        try:
            # 使用FTS5进行全文搜索
            cursor = self.conn.execute("""
                SELECT c.full_name, c.table_name, c.column_name, c.data_type, c.sample_values,
                       fts.rank
                FROM column_fts fts
                JOIN columns c ON c.id = fts.rowid
                WHERE column_fts MATCH ?
                ORDER BY rank
                LIMIT ?
            """, (query, limit))
            
            results = []
            for row in cursor.fetchall():
                full_name, table_name, column_name, data_type, sample_values_json, rank = row
                
                result = VectorSearchResult(
                    id=full_name,
                    content=full_name,
                    similarity=min(0.9, 1.0 / (1.0 + abs(rank))),  # 转换为相似度
                    metadata={
                        "table_name": table_name,
                        "column_name": column_name,
                        "data_type": data_type,
                        "sample_values": json.loads(sample_values_json) if sample_values_json else [],
                        "search_type": "text"
                    }
                )
                results.append(result)
            
            logger.debug(f"全文搜索返回 {len(results)} 个结果")
            return results
            
        except Exception as e:
            logger.error(f"全文搜索失败: {e}")
            return []
    
    def _deduplicate_and_rank(self, results: List[VectorSearchResult]) -> List[VectorSearchResult]:
        """去重和重新排序"""
        seen = set()
        deduplicated = []
        
        # 按相似度排序
        results.sort(key=lambda x: x.similarity, reverse=True)
        
        for result in results:
            if result.id not in seen:
                seen.add(result.id)
                deduplicated.append(result)
        
        return deduplicated
    
    def get_stats(self) -> Dict[str, Any]:
        """获取索引统计信息"""
        try:
            # SQLite统计
            cursor = self.conn.execute("SELECT COUNT(*) FROM columns")
            column_count = cursor.fetchone()[0]
            
            cursor = self.conn.execute("SELECT COUNT(*) FROM tables")
            table_count = cursor.fetchone()[0]
            
            # 向量引擎统计
            vector_stats = {}
            if hasattr(self.vector_engine, 'get_collection_stats'):
                vector_stats = self.vector_engine.get_collection_stats()
            
            return {
                "sqlite": {
                    "columns": column_count,
                    "tables": table_count
                },
                "vector": vector_stats,
                "text_index_terms": len(self.text_index),
                "bloom_filter_size": len(self.bloom_filter)
            }
            
        except Exception as e:
            logger.error(f"获取统计信息失败: {e}")
            return {}
    
    def close(self):
        """关闭数据库连接"""
        if hasattr(self, 'conn'):
            self.conn.close()


# 工厂函数
def create_hybrid_index(vector_engine: VectorSearchEngine) -> HybridIndexEngine:
    """创建混合索引引擎"""
    return HybridIndexEngine(vector_engine)