"""
批量向量搜索优化器 - 三层加速架构的第二层
优化HNSW批量搜索性能，支持并行处理
"""

import logging
import asyncio
from typing import List, Dict, Any, Tuple, Optional
import numpy as np
from collections import defaultdict
import time

from src.tools.vector_search import VectorSearchEngine
from src.core.models import VectorSearchResult, TableInfo
from src.tools.embedding import get_embedding_generator
from src.config.settings import settings

logger = logging.getLogger(__name__)


class BatchVectorSearch:
    """批量向量搜索优化器
    
    优化策略：
    1. 批量编码：一次编码多个查询
    2. 并行搜索：多线程HNSW搜索
    3. 结果聚合：智能合并搜索结果
    4. 缓存复用：避免重复编码
    """
    
    def __init__(self, vector_engine: VectorSearchEngine):
        self.vector_engine = vector_engine
        self.embedding_gen = get_embedding_generator()
        self.embedding_cache = {}  # 缓存已编码的向量
        self.batch_size = getattr(settings.performance, 'batch_size', 50)
        self.max_workers = getattr(settings.performance, 'max_workers', 4)
        
    async def batch_search_tables(
        self,
        query_tables: List[TableInfo],
        candidate_table_names: List[str],
        k: int = 100,
        threshold: float = 0.7
    ) -> Dict[str, List[VectorSearchResult]]:
        """批量搜索表
        
        Args:
            query_tables: 查询表列表
            candidate_table_names: 候选表名列表（从元数据筛选得到）
            k: 每个查询返回的结果数
            threshold: 相似度阈值
            
        Returns:
            {query_table_name: [VectorSearchResult, ...]}
        """
        start_time = time.time()
        
        # 1. 批量生成查询向量
        query_embeddings = await self._batch_generate_embeddings(query_tables)
        
        # 2. 并行执行向量搜索
        search_tasks = []
        for i, query_table in enumerate(query_tables):
            if i < len(query_embeddings):
                task = self._search_with_candidates(
                    query_table.table_name,
                    query_embeddings[i],
                    candidate_table_names,
                    k,
                    threshold
                )
                search_tasks.append(task)
        
        # 3. 收集结果
        search_results = await asyncio.gather(*search_tasks, return_exceptions=True)
        
        # 4. 组织结果
        results_dict = {}
        for i, query_table in enumerate(query_tables):
            if i < len(search_results) and not isinstance(search_results[i], Exception):
                results_dict[query_table.table_name] = search_results[i]
            else:
                logger.error(f"搜索失败 {query_table.table_name}: {search_results[i] if i < len(search_results) else 'No result'}")
                results_dict[query_table.table_name] = []
        
        elapsed_time = time.time() - start_time
        logger.info(f"批量向量搜索完成: {len(query_tables)}个查询, 耗时{elapsed_time:.2f}秒")
        
        return results_dict
    
    async def _batch_generate_embeddings(self, tables: List[TableInfo]) -> List[np.ndarray]:
        """批量生成表的嵌入向量"""
        embeddings = []
        
        # 分批处理以避免内存溢出
        for i in range(0, len(tables), self.batch_size):
            batch = tables[i:i + self.batch_size]
            
            # 检查缓存
            batch_embeddings = []
            uncached_tables = []
            uncached_indices = []
            
            for j, table in enumerate(batch):
                cache_key = self._get_table_cache_key(table)
                if cache_key in self.embedding_cache:
                    batch_embeddings.append(self.embedding_cache[cache_key])
                else:
                    batch_embeddings.append(None)
                    uncached_tables.append(table)
                    uncached_indices.append(j)
            
            # 生成未缓存的嵌入
            if uncached_tables:
                new_embeddings = await self._generate_table_embeddings_batch(uncached_tables)
                
                # 更新结果和缓存
                for idx, embedding in zip(uncached_indices, new_embeddings):
                    batch_embeddings[idx] = embedding
                    cache_key = self._get_table_cache_key(batch[idx])
                    self.embedding_cache[cache_key] = embedding
            
            embeddings.extend(batch_embeddings)
        
        return embeddings
    
    async def _generate_table_embeddings_batch(self, tables: List[TableInfo]) -> List[np.ndarray]:
        """批量生成表嵌入向量"""
        # 构建批量文本
        texts = []
        for table in tables:
            # 组合表信息为文本
            text_parts = [f"Table: {table.table_name}"]
            
            # 添加列信息
            for column in table.columns[:20]:  # 限制列数
                col_text = f"Column: {column.column_name} ({column.data_type or 'unknown'})"
                if column.sample_values:
                    samples = ", ".join(str(v) for v in column.sample_values[:3])
                    col_text += f" - Examples: {samples}"
                text_parts.append(col_text)
            
            texts.append("\n".join(text_parts))
        
        # 批量编码
        try:
            # 假设embedding_gen支持批量编码
            if hasattr(self.embedding_gen, 'batch_encode'):
                embeddings = await self.embedding_gen.batch_encode(texts)
            else:
                # 如果不支持批量，则逐个编码（但使用并发）
                tasks = [self.embedding_gen.generate_text_embedding(text) for text in texts]
                embeddings = await asyncio.gather(*tasks)
            
            return [np.array(emb) for emb in embeddings]
            
        except Exception as e:
            logger.error(f"批量编码失败: {e}")
            # 返回空向量
            return [np.zeros(self.embedding_gen.dimension) for _ in tables]
    
    async def _search_with_candidates(
        self,
        query_table_name: str,
        query_embedding: np.ndarray,
        candidate_table_names: List[str],
        k: int,
        threshold: float
    ) -> List[VectorSearchResult]:
        """在候选表中搜索"""
        try:
            # 执行向量搜索
            all_results = await self.vector_engine.search_similar_tables(
                query_embedding=query_embedding.tolist(),
                k=min(k * 2, 500),  # 搜索更多结果以便筛选
                threshold=threshold
            )
            
            # 筛选出在候选列表中的结果
            candidate_set = set(candidate_table_names)
            filtered_results = []
            
            for result in all_results:
                if result.item_id in candidate_set:
                    filtered_results.append(result)
                    if len(filtered_results) >= k:
                        break
            
            return filtered_results
            
        except Exception as e:
            logger.error(f"向量搜索失败 {query_table_name}: {e}")
            return []
    
    def _get_table_cache_key(self, table: TableInfo) -> str:
        """生成表的缓存键"""
        # 使用表名和列数作为缓存键
        col_names = sorted([col.column_name for col in table.columns])
        return f"{table.table_name}:{len(table.columns)}:{':'.join(col_names[:5])}"
    
    async def parallel_search_columns(
        self,
        query_columns: List[Dict[str, Any]],
        k: int = 50
    ) -> Dict[str, List[VectorSearchResult]]:
        """并行搜索列（用于Bottom-Up策略）"""
        # 分组查询列（按表分组）
        table_columns = defaultdict(list)
        for col in query_columns:
            table_columns[col.get("table_name", "unknown")].append(col)
        
        # 并行搜索每组
        search_tasks = []
        for table_name, columns in table_columns.items():
            task = self._batch_search_columns_for_table(columns, k)
            search_tasks.append((table_name, task))
        
        # 收集结果
        results = {}
        for table_name, task in search_tasks:
            try:
                table_results = await task
                results[table_name] = table_results
            except Exception as e:
                logger.error(f"列搜索失败 {table_name}: {e}")
                results[table_name] = []
        
        return results
    
    async def _batch_search_columns_for_table(
        self,
        columns: List[Dict[str, Any]],
        k: int
    ) -> List[Dict[str, Any]]:
        """批量搜索一个表的列"""
        results = []
        
        # 批量生成列嵌入
        embeddings = []
        for col in columns:
            # 构建列文本
            col_text = f"Column: {col.get('column_name', '')} Type: {col.get('data_type', '')}"
            if col.get('sample_values'):
                samples = ", ".join(str(v) for v in col.get('sample_values', [])[:5])
                col_text += f" Examples: {samples}"
            
            # 生成嵌入
            embedding = await self.embedding_gen.generate_text_embedding(col_text)
            embeddings.append(embedding)
        
        # 批量搜索
        search_tasks = []
        for i, col in enumerate(columns):
            if i < len(embeddings):
                task = self.vector_engine.search_similar_columns(
                    query_embedding=embeddings[i],
                    k=k,
                    threshold=0.6
                )
                search_tasks.append(task)
        
        # 收集结果
        search_results = await asyncio.gather(*search_tasks, return_exceptions=True)
        
        for i, col in enumerate(columns):
            if i < len(search_results) and not isinstance(search_results[i], Exception):
                results.append({
                    "query_column": col,
                    "matches": search_results[i]
                })
        
        return results
    
    def aggregate_batch_results(
        self,
        batch_results: Dict[str, List[VectorSearchResult]],
        top_k: int = 10
    ) -> List[Tuple[str, float]]:
        """聚合批量搜索结果
        
        Returns:
            [(table_name, aggregated_score), ...]
        """
        # 统计每个表的出现次数和平均分数
        table_scores = defaultdict(list)
        
        for query_table, results in batch_results.items():
            for result in results:
                table_scores[result.item_id].append(result.score)
        
        # 计算聚合分数
        aggregated = []
        for table_name, scores in table_scores.items():
            # 综合考虑出现频率和平均分数
            frequency_score = len(scores) / len(batch_results)
            avg_score = sum(scores) / len(scores)
            
            # 加权聚合
            final_score = 0.6 * avg_score + 0.4 * frequency_score
            aggregated.append((table_name, final_score))
        
        # 排序并返回Top-K
        aggregated.sort(key=lambda x: x[1], reverse=True)
        return aggregated[:top_k]