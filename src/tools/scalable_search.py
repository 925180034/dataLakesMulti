"""
可扩展搜索引擎 - 针对万级表格规模的分层索引架构
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Set, Tuple
import numpy as np
import logging
from pathlib import Path
from collections import defaultdict
import hashlib
import json
from datetime import datetime

from src.core.models import VectorSearchResult, ValueSearchResult, ColumnInfo, TableInfo
from src.config.settings import settings

logger = logging.getLogger(__name__)


class TableSignature:
    """表签名 - 用于快速表级别匹配和聚类"""
    
    def __init__(self, table_info: TableInfo):
        self.table_name = table_info.table_name
        self.column_count = len(table_info.columns)
        self.column_types = self._extract_column_types(table_info)
        self.schema_pattern = self._generate_schema_pattern(table_info)
        self.domain_keywords = self._extract_domain_keywords(table_info)
        
    def _extract_column_types(self, table_info: TableInfo) -> Dict[str, int]:
        """提取列类型分布"""
        type_counts = defaultdict(int)
        for col in table_info.columns:
            if col.data_type:
                normalized_type = self._normalize_type(col.data_type)
                type_counts[normalized_type] += 1
        return dict(type_counts)
    
    def _normalize_type(self, data_type: str) -> str:
        """标准化数据类型"""
        data_type = data_type.lower()
        if any(t in data_type for t in ['int', 'bigint', 'smallint']):
            return 'integer'
        elif any(t in data_type for t in ['float', 'double', 'decimal', 'numeric']):
            return 'numeric'
        elif any(t in data_type for t in ['varchar', 'char', 'text', 'string']):
            return 'text'
        elif any(t in data_type for t in ['date', 'timestamp', 'datetime']):
            return 'temporal'
        elif any(t in data_type for t in ['bool', 'boolean']):
            return 'boolean'
        else:
            return 'other'
    
    def _generate_schema_pattern(self, table_info: TableInfo) -> str:
        """生成表结构模式签名"""
        column_names = [col.column_name.lower() for col in table_info.columns]
        column_names.sort()  # 排序以确保一致性
        
        # 提取关键词模式
        patterns = []
        for name in column_names:
            if 'id' in name:
                patterns.append('ID')
            elif any(kw in name for kw in ['name', 'title', 'label']):
                patterns.append('NAME')
            elif any(kw in name for kw in ['date', 'time', 'created', 'updated']):
                patterns.append('TIME')
            elif any(kw in name for kw in ['email', 'phone', 'address']):
                patterns.append('CONTACT')
            elif any(kw in name for kw in ['price', 'cost', 'amount', 'value']):
                patterns.append('MONEY')
            else:
                patterns.append('OTHER')
        
        return '|'.join(patterns)
    
    def _extract_domain_keywords(self, table_info: TableInfo) -> Set[str]:
        """提取领域关键词"""
        keywords = set()
        
        # 从表名提取
        table_words = self._extract_keywords(table_info.table_name)
        keywords.update(table_words)
        
        # 从列名提取
        for col in table_info.columns:
            col_words = self._extract_keywords(col.column_name)
            keywords.update(col_words)
        
        return keywords
    
    def _extract_keywords(self, text: str) -> Set[str]:
        """从文本中提取关键词"""
        import re
        
        # 分割驼峰命名和下划线
        words = re.findall(r'[A-Z]?[a-z]+|[A-Z]+(?=[A-Z][a-z]|\b)|[0-9]+', text)
        
        # 过滤常见词汇
        stop_words = {'id', 'name', 'type', 'data', 'info', 'table', 'column'}
        keywords = {word.lower() for word in words if len(word) > 2 and word.lower() not in stop_words}
        
        return keywords
    
    def calculate_similarity(self, other: 'TableSignature') -> float:
        """计算与另一个表签名的相似度"""
        similarities = []
        
        # 1. 列数量相似度
        col_count_sim = 1.0 - abs(self.column_count - other.column_count) / max(self.column_count, other.column_count)
        similarities.append(col_count_sim * 0.2)
        
        # 2. 类型分布相似度
        type_sim = self._calculate_type_similarity(other.column_types)
        similarities.append(type_sim * 0.3)
        
        # 3. 模式相似度
        pattern_sim = self._calculate_pattern_similarity(other.schema_pattern)
        similarities.append(pattern_sim * 0.3)
        
        # 4. 领域关键词相似度
        domain_sim = self._calculate_domain_similarity(other.domain_keywords)
        similarities.append(domain_sim * 0.2)
        
        return sum(similarities)
    
    def _calculate_type_similarity(self, other_types: Dict[str, int]) -> float:
        """计算类型分布相似度"""
        all_types = set(self.column_types.keys()) | set(other_types.keys())
        if not all_types:
            return 1.0
        
        similarities = []
        for type_name in all_types:
            count1 = self.column_types.get(type_name, 0)
            count2 = other_types.get(type_name, 0)
            
            if count1 == 0 and count2 == 0:
                similarities.append(1.0)
            else:
                sim = 1.0 - abs(count1 - count2) / max(count1 + count2, 1)
                similarities.append(sim)
        
        return np.mean(similarities)
    
    def _calculate_pattern_similarity(self, other_pattern: str) -> float:
        """计算模式相似度"""
        if not self.schema_pattern and not other_pattern:
            return 1.0
        if not self.schema_pattern or not other_pattern:
            return 0.0
        
        patterns1 = set(self.schema_pattern.split('|'))
        patterns2 = set(other_pattern.split('|'))
        
        intersection = patterns1 & patterns2
        union = patterns1 | patterns2
        
        return len(intersection) / len(union) if union else 0.0
    
    def _calculate_domain_similarity(self, other_keywords: Set[str]) -> float:
        """计算领域相似度"""
        if not self.domain_keywords and not other_keywords:
            return 1.0
        if not self.domain_keywords or not other_keywords:
            return 0.0
        
        intersection = self.domain_keywords & other_keywords
        union = self.domain_keywords | other_keywords
        
        return len(intersection) / len(union) if union else 0.0


class MetadataIndex:
    """元数据快速查找索引"""
    
    def __init__(self):
        # 按不同维度建立索引
        self.by_schema_pattern = defaultdict(list)  # schema_pattern -> [table_ids]
        self.by_domain_keywords = defaultdict(set)  # keyword -> {table_ids}
        self.by_column_count = defaultdict(list)    # count_range -> [table_ids]
        self.by_table_size = defaultdict(list)      # size_range -> [table_ids]
        
        # 表元数据存储
        self.table_signatures = {}  # table_id -> TableSignature
        self.table_metadata = {}    # table_id -> TableInfo
        
        logger.info("初始化元数据索引")
    
    def add_table(self, table_id: str, table_info: TableInfo) -> None:
        """添加表到元数据索引"""
        try:
            # 生成表签名
            signature = TableSignature(table_info)
            self.table_signatures[table_id] = signature
            self.table_metadata[table_id] = table_info
            
            # 更新各维度索引
            self.by_schema_pattern[signature.schema_pattern].append(table_id)
            
            for keyword in signature.domain_keywords:
                self.by_domain_keywords[keyword].add(table_id)
            
            # 按列数分组
            col_count_range = self._get_column_count_range(signature.column_count)
            self.by_column_count[col_count_range].append(table_id)
            
            # 按表大小分组（如果有行数信息）
            if table_info.row_count:
                size_range = self._get_size_range(table_info.row_count)
                self.by_table_size[size_range].append(table_id)
            
            logger.debug(f"添加表到元数据索引: {table_id}")
            
        except Exception as e:
            logger.error(f"添加表到元数据索引失败 {table_id}: {e}")
    
    def _get_column_count_range(self, count: int) -> str:
        """获取列数范围标识"""
        if count <= 5:
            return "small"
        elif count <= 15:
            return "medium"
        elif count <= 30:
            return "large"
        else:
            return "xlarge"
    
    def _get_size_range(self, row_count: int) -> str:
        """获取表大小范围标识"""
        if row_count <= 1000:
            return "tiny"
        elif row_count <= 10000:
            return "small"
        elif row_count <= 100000:
            return "medium"
        elif row_count <= 1000000:
            return "large"
        else:
            return "huge"
    
    def prefilter_tables(self, 
                        query_keywords: Set[str] = None,
                        target_column_count: int = None,
                        schema_pattern: str = None,
                        max_candidates: int = 1000) -> List[str]:
        """基于元数据预筛选表"""
        try:
            candidate_sets = []
            
            # 1. 基于关键词筛选
            if query_keywords:
                keyword_candidates = set()
                for keyword in query_keywords:
                    if keyword in self.by_domain_keywords:
                        keyword_candidates.update(self.by_domain_keywords[keyword])
                if keyword_candidates:
                    candidate_sets.append(keyword_candidates)
            
            # 2. 基于列数筛选
            if target_column_count:
                target_range = self._get_column_count_range(target_column_count)
                
                # 包含目标范围和相邻范围
                ranges_to_include = [target_range]
                range_order = ["small", "medium", "large", "xlarge"]
                
                if target_range in range_order:
                    idx = range_order.index(target_range)
                    if idx > 0:
                        ranges_to_include.append(range_order[idx - 1])
                    if idx < len(range_order) - 1:
                        ranges_to_include.append(range_order[idx + 1])
                
                count_candidates = set()
                for range_name in ranges_to_include:
                    count_candidates.update(self.by_column_count[range_name])
                
                if count_candidates:
                    candidate_sets.append(count_candidates)
            
            # 3. 基于模式筛选
            if schema_pattern:
                pattern_candidates = set()
                
                # 精确匹配
                if schema_pattern in self.by_schema_pattern:
                    pattern_candidates.update(self.by_schema_pattern[schema_pattern])
                
                # 模糊匹配 - 找相似模式
                for existing_pattern in self.by_schema_pattern.keys():
                    if self._pattern_similarity(schema_pattern, existing_pattern) > 0.5:
                        pattern_candidates.update(self.by_schema_pattern[existing_pattern])
                
                if pattern_candidates:
                    candidate_sets.append(pattern_candidates)
            
            # 取交集或并集
            if candidate_sets:
                if len(candidate_sets) == 1:
                    final_candidates = candidate_sets[0]
                else:
                    # 优先取交集，如果交集太小则取并集
                    intersection = set.intersection(*candidate_sets)
                    if len(intersection) >= max_candidates * 0.1:  # 至少10%
                        final_candidates = intersection
                    else:
                        final_candidates = set.union(*candidate_sets)
            else:
                # 没有筛选条件，返回所有表
                final_candidates = set(self.table_signatures.keys())
            
            # 限制候选数量
            candidates_list = list(final_candidates)[:max_candidates]
            
            logger.debug(f"元数据预筛选: {len(self.table_signatures)} -> {len(candidates_list)}")
            return candidates_list
            
        except Exception as e:
            logger.error(f"元数据预筛选失败: {e}")
            return list(self.table_signatures.keys())[:max_candidates]
    
    def _pattern_similarity(self, pattern1: str, pattern2: str) -> float:
        """计算模式相似度"""
        if not pattern1 or not pattern2:
            return 0.0
        
        parts1 = set(pattern1.split('|'))
        parts2 = set(pattern2.split('|'))
        
        intersection = parts1 & parts2
        union = parts1 | parts2
        
        return len(intersection) / len(union) if union else 0.0
    
    def get_similar_tables_by_signature(self, 
                                      target_signature: TableSignature, 
                                      k: int = 100) -> List[Tuple[str, float]]:
        """基于表签名找相似表"""
        similarities = []
        
        for table_id, signature in self.table_signatures.items():
            similarity = target_signature.calculate_similarity(signature)
            similarities.append((table_id, similarity))
        
        # 排序并返回top-k
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:k]


class HierarchicalVectorSearch:
    """分层向量搜索引擎"""
    
    def __init__(self, dimension: int = 1536):
        self.dimension = dimension
        
        # 表级索引（粗筛）
        self.table_index = None
        self.table_embeddings = {}  # table_id -> embedding
        
        # 列级索引（按表分组）
        self.column_indices = {}    # table_id -> faiss_index
        self.column_metadata = {}   # (table_id, col_idx) -> ColumnInfo
        
        # 元数据索引
        self.metadata_index = MetadataIndex()
        
        # 全局列索引（备用）
        self.global_column_index = None
        self.global_column_metadata = {}  # global_col_id -> ColumnInfo
        self.global_col_id_counter = 0
        
        self._initialize_indices()
        
        logger.info(f"初始化分层向量搜索引擎，维度: {dimension}")
    
    def _initialize_indices(self):
        """初始化FAISS索引"""
        try:
            import faiss
            
            # 表级索引 - 使用IVF索引提升大规模性能
            quantizer = faiss.IndexFlatIP(self.dimension)
            self.table_index = faiss.IndexIVFFlat(quantizer, self.dimension, min(100, 1000))
            self.table_index.nprobe = 10  # 搜索时的探测数量
            
            # 全局列索引
            column_quantizer = faiss.IndexFlatIP(self.dimension)
            self.global_column_index = faiss.IndexIVFFlat(column_quantizer, self.dimension, min(500, 5000))
            self.global_column_index.nprobe = 20
            
            logger.info("分层FAISS索引初始化完成")
            
        except ImportError:
            logger.error("FAISS库未安装，请安装: pip install faiss-cpu")
            raise
        except Exception as e:
            logger.error(f"初始化分层FAISS索引失败: {e}")
            raise
    
    async def add_table_with_columns(self, 
                                   table_info: TableInfo, 
                                   table_embedding: List[float],
                                   column_embeddings: List[List[float]]) -> str:
        """添加表及其所有列到分层索引"""
        try:
            table_id = f"{table_info.table_name}_{hash(table_info.table_name) % 10000}"
            
            # 1. 添加到元数据索引
            self.metadata_index.add_table(table_id, table_info)
            
            # 2. 添加表级向量
            await self._add_table_vector(table_id, table_embedding)
            
            # 3. 为该表创建专门的列索引
            await self._create_table_column_index(table_id, table_info.columns, column_embeddings)
            
            # 4. 同时添加到全局列索引（用于跨表搜索）
            await self._add_columns_to_global_index(table_info.columns, column_embeddings)
            
            logger.debug(f"添加表到分层索引: {table_id}, 列数: {len(table_info.columns)}")
            return table_id
            
        except Exception as e:
            logger.error(f"添加表到分层索引失败: {e}")
            raise
    
    async def _add_table_vector(self, table_id: str, embedding: List[float]) -> None:
        """添加表向量到表级索引"""
        try:
            import faiss
            
            # 归一化
            embedding_array = np.array([embedding], dtype=np.float32)
            embedding_array = embedding_array / np.linalg.norm(embedding_array, axis=1, keepdims=True)
            
            # 训练索引（如果还没训练）
            if not self.table_index.is_trained:
                # 需要积累一定数量的向量再训练
                if len(self.table_embeddings) == 0:
                    # 临时存储，等待训练
                    self.table_embeddings[table_id] = embedding_array[0]
                    return
                
                # 收集所有向量进行训练
                all_embeddings = list(self.table_embeddings.values())
                all_embeddings.append(embedding_array[0])
                
                if len(all_embeddings) >= self.table_index.nlist:
                    train_data = np.array(all_embeddings, dtype=np.float32)
                    self.table_index.train(train_data)
                    
                    # 添加之前存储的向量
                    for stored_embedding in all_embeddings[:-1]:
                        self.table_index.add(np.array([stored_embedding], dtype=np.float32))
            
            # 添加当前向量
            if self.table_index.is_trained:
                self.table_index.add(embedding_array)
            
            self.table_embeddings[table_id] = embedding_array[0]
            
        except Exception as e:
            logger.error(f"添加表向量失败: {e}")
            raise
    
    async def _create_table_column_index(self, 
                                       table_id: str, 
                                       columns: List[ColumnInfo],
                                       embeddings: List[List[float]]) -> None:
        """为特定表创建列索引"""
        try:
            import faiss
            
            if len(embeddings) != len(columns):
                raise ValueError(f"列数和嵌入向量数不匹配: {len(columns)} vs {len(embeddings)}")
            
            # 创建该表专用的列索引
            table_column_index = faiss.IndexFlatIP(self.dimension)
            
            # 添加所有列向量
            for i, (column, embedding) in enumerate(zip(columns, embeddings)):
                # 归一化
                embedding_array = np.array([embedding], dtype=np.float32)
                embedding_array = embedding_array / np.linalg.norm(embedding_array, axis=1, keepdims=True)
                
                # 添加到表级列索引
                table_column_index.add(embedding_array)
                
                # 保存列元数据
                self.column_metadata[(table_id, i)] = column
            
            self.column_indices[table_id] = table_column_index
            
        except Exception as e:
            logger.error(f"创建表列索引失败 {table_id}: {e}")
            raise
    
    async def _add_columns_to_global_index(self, 
                                         columns: List[ColumnInfo],
                                         embeddings: List[List[float]]) -> None:
        """添加列到全局列索引"""
        try:
            import faiss
            
            for column, embedding in zip(columns, embeddings):
                # 归一化
                embedding_array = np.array([embedding], dtype=np.float32)
                embedding_array = embedding_array / np.linalg.norm(embedding_array, axis=1, keepdims=True)
                
                # 训练索引（如果需要）
                if not self.global_column_index.is_trained:
                    if self.global_col_id_counter < self.global_column_index.nlist:
                        # 临时存储
                        self.global_column_metadata[self.global_col_id_counter] = column
                        self.global_col_id_counter += 1
                        continue
                    else:
                        # 训练索引
                        train_embeddings = []
                        for col_id in range(self.global_col_id_counter):
                            if col_id in self.global_column_metadata:
                                # 这里需要重新生成嵌入，简化处理先跳过训练
                                pass
                        # TODO: 完善训练逻辑
                
                # 添加到全局索引
                if self.global_column_index.is_trained:
                    self.global_column_index.add(embedding_array)
                
                self.global_column_metadata[self.global_col_id_counter] = column
                self.global_col_id_counter += 1
                
        except Exception as e:
            logger.error(f"添加列到全局索引失败: {e}")
    
    async def hierarchical_search_tables(self, 
                                       query_embedding: List[float],
                                       query_keywords: Set[str] = None,
                                       target_column_count: int = None,
                                       k: int = 50) -> List[VectorSearchResult]:
        """分层搜索相似表"""
        try:
            # 第一阶段：元数据预筛选
            candidate_table_ids = self.metadata_index.prefilter_tables(
                query_keywords=query_keywords,
                target_column_count=target_column_count,
                max_candidates=min(1000, len(self.table_embeddings))
            )
            
            logger.debug(f"元数据预筛选候选表: {len(candidate_table_ids)}")
            
            # 第二阶段：向量相似度搜索
            if not candidate_table_ids:
                return []
            
            # 构建候选表的嵌入矩阵
            candidate_embeddings = []
            candidate_ids = []
            
            for table_id in candidate_table_ids:
                if table_id in self.table_embeddings:
                    candidate_embeddings.append(self.table_embeddings[table_id])
                    candidate_ids.append(table_id)
            
            if not candidate_embeddings:
                return []
            
            # 计算相似度
            query_array = np.array([query_embedding], dtype=np.float32)
            query_array = query_array / np.linalg.norm(query_array, axis=1, keepdims=True)
            
            candidate_matrix = np.array(candidate_embeddings, dtype=np.float32)
            
            # 计算余弦相似度
            similarities = np.dot(query_array, candidate_matrix.T)[0]
            
            # 排序并返回top-k
            top_indices = np.argsort(similarities)[::-1][:k]
            
            results = []
            for idx in top_indices:
                table_id = candidate_ids[idx]
                score = float(similarities[idx])
                
                table_info = self.metadata_index.table_metadata.get(table_id)
                if table_info:
                    results.append(VectorSearchResult(
                        item_id=table_id,
                        score=score,
                        metadata={
                            "table_name": table_info.table_name,
                            "column_count": len(table_info.columns),
                            "row_count": table_info.row_count,
                            "file_path": table_info.file_path
                        }
                    ))
            
            logger.debug(f"分层表搜索返回 {len(results)} 个结果")
            return results
            
        except Exception as e:
            logger.error(f"分层表搜索失败: {e}")
            return []
    
    async def hierarchical_search_columns(self,
                                        query_embedding: List[float],
                                        candidate_table_ids: List[str] = None,
                                        k: int = 50) -> List[VectorSearchResult]:
        """分层搜索相似列"""
        try:
            results = []
            
            if candidate_table_ids:
                # 在指定表中搜索列
                for table_id in candidate_table_ids:
                    if table_id in self.column_indices:
                        table_results = await self._search_columns_in_table(
                            table_id, query_embedding, k=min(k, 20)
                        )
                        results.extend(table_results)
            else:
                # 全局列搜索
                results = await self._global_column_search(query_embedding, k=k)
            
            # 排序并返回top-k
            results.sort(key=lambda x: x.score, reverse=True)
            return results[:k]
            
        except Exception as e:
            logger.error(f"分层列搜索失败: {e}")
            return []
    
    async def _search_columns_in_table(self, 
                                     table_id: str, 
                                     query_embedding: List[float],
                                     k: int = 20) -> List[VectorSearchResult]:
        """在指定表中搜索列"""
        try:
            if table_id not in self.column_indices:
                return []
            
            table_index = self.column_indices[table_id]
            
            # 归一化查询向量
            query_array = np.array([query_embedding], dtype=np.float32)
            query_array = query_array / np.linalg.norm(query_array, axis=1, keepdims=True)
            
            # 搜索
            k = min(k, table_index.ntotal)
            if k == 0:
                return []
            
            scores, indices = table_index.search(query_array, k)
            
            results = []
            for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
                column_info = self.column_metadata.get((table_id, idx))
                if column_info:
                    results.append(VectorSearchResult(
                        item_id=column_info.full_name,
                        score=float(score),
                        metadata={
                            "table_name": column_info.table_name,
                            "column_name": column_info.column_name,
                            "data_type": column_info.data_type,
                            "sample_values": column_info.sample_values[:3]
                        }
                    ))
            
            return results
            
        except Exception as e:
            logger.error(f"表内列搜索失败 {table_id}: {e}")
            return []
    
    async def _global_column_search(self, 
                                  query_embedding: List[float],
                                  k: int = 50) -> List[VectorSearchResult]:
        """全局列搜索"""
        try:
            if not self.global_column_index.is_trained or self.global_column_index.ntotal == 0:
                logger.warning("全局列索引未训练或为空")
                return []
            
            # 归一化查询向量
            query_array = np.array([query_embedding], dtype=np.float32)
            query_array = query_array / np.linalg.norm(query_array, axis=1, keepdims=True)
            
            # 搜索
            k = min(k, self.global_column_index.ntotal)
            scores, indices = self.global_column_index.search(query_array, k)
            
            results = []
            for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
                column_info = self.global_column_metadata.get(idx)
                if column_info:
                    results.append(VectorSearchResult(
                        item_id=column_info.full_name,
                        score=float(score),
                        metadata={
                            "table_name": column_info.table_name,
                            "column_name": column_info.column_name,
                            "data_type": column_info.data_type,
                            "sample_values": column_info.sample_values[:3]
                        }
                    ))
            
            return results
            
        except Exception as e:
            logger.error(f"全局列搜索失败: {e}")
            return []


# 创建全局实例
hierarchical_search_engine = HierarchicalVectorSearch()