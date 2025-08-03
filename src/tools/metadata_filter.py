"""
元数据预筛选器 - 三层加速架构的第一层
基于表元数据快速过滤候选表，无需向量计算
"""

import logging
from typing import List, Dict, Any, Set, Tuple
from collections import defaultdict
import re
from src.core.models import TableInfo, ColumnInfo

logger = logging.getLogger(__name__)


class MetadataFilter:
    """基于元数据的快速预筛选器
    
    筛选策略：
    1. 领域分类：基于表名前缀/后缀识别业务领域
    2. 表规模：列数、数据类型分布
    3. 命名模式：识别相似的命名规范
    4. 关键列：主键、外键等特殊列
    """
    
    def __init__(self):
        # 领域索引
        self.domain_index: Dict[str, Set[str]] = defaultdict(set)
        # 表规模索引
        self.size_index: Dict[int, Set[str]] = defaultdict(set)
        # 列数索引
        self.column_count_index: Dict[int, Set[str]] = defaultdict(set)
        # 命名模式索引
        self.pattern_index: Dict[str, Set[str]] = defaultdict(set)
        # 数据类型分布索引
        self.type_distribution_index: Dict[str, Set[str]] = defaultdict(set)
        # 表元数据缓存
        self.table_metadata: Dict[str, Dict[str, Any]] = {}
        
    def build_index(self, tables: List[TableInfo]) -> None:
        """构建元数据索引"""
        logger.info(f"开始构建元数据索引，表数量: {len(tables)}")
        
        for table in tables:
            table_name = table.table_name
            
            # 1. 提取领域信息
            domain = self._extract_domain(table_name)
            self.domain_index[domain].add(table_name)
            
            # 2. 表规模分类
            size_category = self._categorize_table_size(table)
            self.size_index[size_category].add(table_name)
            
            # 3. 列数索引
            column_count = len(table.columns)
            column_bucket = self._get_column_count_bucket(column_count)
            self.column_count_index[column_bucket].add(table_name)
            
            # 4. 命名模式
            pattern = self._extract_naming_pattern(table_name)
            self.pattern_index[pattern].add(table_name)
            
            # 5. 数据类型分布
            type_dist = self._get_type_distribution(table)
            type_signature = self._get_type_signature(type_dist)
            self.type_distribution_index[type_signature].add(table_name)
            
            # 6. 缓存表元数据
            self.table_metadata[table_name] = {
                "domain": domain,
                "size_category": size_category,
                "column_count": column_count,
                "pattern": pattern,
                "type_distribution": type_dist,
                "key_columns": self._extract_key_columns(table)
            }
            
        logger.info(f"元数据索引构建完成，领域数: {len(self.domain_index)}")
        
    def filter_candidates(
        self, 
        query_table: TableInfo, 
        all_tables: List[str],
        top_k: int = 1000
    ) -> List[Tuple[str, float]]:
        """快速筛选候选表
        
        返回: [(table_name, score), ...]
        """
        # 提取查询表特征
        query_features = self._extract_table_features(query_table)
        
        # 计算每个表的相似度得分
        table_scores = {}
        
        for table_name in all_tables:
            if table_name == query_table.table_name:
                continue
                
            score = self._calculate_metadata_similarity(
                query_features, 
                self.table_metadata.get(table_name, {})
            )
            
            if score > 0:
                table_scores[table_name] = score
        
        # 排序并返回Top-K
        sorted_tables = sorted(
            table_scores.items(), 
            key=lambda x: x[1], 
            reverse=True
        )[:top_k]
        
        logger.info(f"元数据筛选完成: {len(all_tables)} -> {len(sorted_tables)}")
        return sorted_tables
    
    def _extract_domain(self, table_name: str) -> str:
        """提取表的业务领域"""
        # 常见领域前缀/后缀
        domain_patterns = {
            "user": ["user", "customer", "member", "account"],
            "order": ["order", "purchase", "transaction", "payment"],
            "product": ["product", "item", "goods", "sku"],
            "log": ["log", "event", "history", "audit"],
            "config": ["config", "setting", "param", "option"],
            "dim": ["dim_", "dimension_", "d_"],
            "fact": ["fact_", "f_", "agg_"],
            "tmp": ["tmp_", "temp_", "staging_"]
        }
        
        table_lower = table_name.lower()
        
        for domain, patterns in domain_patterns.items():
            for pattern in patterns:
                if pattern in table_lower:
                    return domain
                    
        # 默认领域
        return "general"
    
    def _categorize_table_size(self, table: TableInfo) -> int:
        """表规模分类"""
        # 基于列数和预估行数
        column_count = len(table.columns)
        
        if column_count <= 5:
            return 0  # 小表
        elif column_count <= 20:
            return 1  # 中表
        else:
            return 2  # 大表
    
    def _get_column_count_bucket(self, count: int) -> int:
        """列数分桶"""
        if count <= 5:
            return 5
        elif count <= 10:
            return 10
        elif count <= 20:
            return 20
        elif count <= 50:
            return 50
        else:
            return 100
    
    def _extract_naming_pattern(self, table_name: str) -> str:
        """提取命名模式"""
        # 识别常见命名模式
        if "_" in table_name:
            parts = table_name.split("_")
            if len(parts) >= 2:
                # 保留前缀和后缀模式
                return f"{parts[0]}_*_{parts[-1]}"
        
        # 驼峰命名
        if any(c.isupper() for c in table_name[1:]):
            # 转换为模式
            pattern = re.sub(r'[A-Z]', '*', table_name)
            return pattern
            
        return "simple"
    
    def _get_type_distribution(self, table: TableInfo) -> Dict[str, int]:
        """获取数据类型分布"""
        type_count = defaultdict(int)
        
        for column in table.columns:
            data_type = column.data_type or "unknown"
            # 标准化数据类型
            normalized_type = self._normalize_data_type(data_type)
            type_count[normalized_type] += 1
            
        return dict(type_count)
    
    def _normalize_data_type(self, data_type: str) -> str:
        """标准化数据类型"""
        type_lower = data_type.lower()
        
        # 数值类型
        if any(t in type_lower for t in ["int", "number", "numeric", "decimal", "float", "double"]):
            return "numeric"
        # 字符串类型
        elif any(t in type_lower for t in ["string", "text", "varchar", "char"]):
            return "string"
        # 日期时间类型
        elif any(t in type_lower for t in ["date", "time", "timestamp"]):
            return "datetime"
        # 布尔类型
        elif any(t in type_lower for t in ["bool", "boolean"]):
            return "boolean"
        else:
            return "other"
    
    def _get_type_signature(self, type_dist: Dict[str, int]) -> str:
        """生成类型分布签名"""
        # 创建标准化的类型签名
        signature_parts = []
        for dtype in ["numeric", "string", "datetime", "boolean", "other"]:
            count = type_dist.get(dtype, 0)
            if count > 0:
                signature_parts.append(f"{dtype}:{count}")
        
        return "|".join(signature_parts)
    
    def _extract_key_columns(self, table: TableInfo) -> List[str]:
        """提取关键列（主键、外键等）"""
        key_columns = []
        
        for column in table.columns:
            col_name_lower = column.column_name.lower()
            # 识别可能的主键
            if any(kw in col_name_lower for kw in ["id", "key", "code", "no", "num"]):
                key_columns.append(column.column_name)
            # 识别可能的外键
            elif col_name_lower.endswith("_id") or col_name_lower.endswith("_key"):
                key_columns.append(column.column_name)
                
        return key_columns
    
    def _extract_table_features(self, table: TableInfo) -> Dict[str, Any]:
        """提取表特征"""
        return {
            "domain": self._extract_domain(table.table_name),
            "size_category": self._categorize_table_size(table),
            "column_count": len(table.columns),
            "pattern": self._extract_naming_pattern(table.table_name),
            "type_distribution": self._get_type_distribution(table),
            "key_columns": self._extract_key_columns(table)
        }
    
    def _calculate_metadata_similarity(
        self, 
        query_features: Dict[str, Any],
        target_metadata: Dict[str, Any]
    ) -> float:
        """计算元数据相似度"""
        if not target_metadata:
            return 0.0
            
        score = 0.0
        
        # 1. 领域匹配 (权重: 0.3)
        if query_features["domain"] == target_metadata.get("domain"):
            score += 0.3
        
        # 2. 表规模相似 (权重: 0.2)
        size_diff = abs(
            query_features["size_category"] - 
            target_metadata.get("size_category", 0)
        )
        score += 0.2 * (1 - size_diff / 3)
        
        # 3. 列数相似 (权重: 0.2)
        query_cols = query_features["column_count"]
        target_cols = target_metadata.get("column_count", 0)
        if target_cols > 0:
            col_similarity = 1 - abs(query_cols - target_cols) / max(query_cols, target_cols)
            score += 0.2 * col_similarity
        
        # 4. 类型分布相似 (权重: 0.2)
        type_sim = self._calculate_type_distribution_similarity(
            query_features["type_distribution"],
            target_metadata.get("type_distribution", {})
        )
        score += 0.2 * type_sim
        
        # 5. 关键列匹配 (权重: 0.1)
        query_keys = set(query_features.get("key_columns", []))
        target_keys = set(target_metadata.get("key_columns", []))
        if query_keys and target_keys:
            key_overlap = len(query_keys & target_keys) / len(query_keys | target_keys)
            score += 0.1 * key_overlap
            
        return score
    
    def _calculate_type_distribution_similarity(
        self, 
        dist1: Dict[str, int], 
        dist2: Dict[str, int]
    ) -> float:
        """计算类型分布相似度"""
        if not dist1 or not dist2:
            return 0.0
            
        # 获取所有类型
        all_types = set(dist1.keys()) | set(dist2.keys())
        
        # 计算余弦相似度
        dot_product = 0
        norm1 = 0
        norm2 = 0
        
        for dtype in all_types:
            v1 = dist1.get(dtype, 0)
            v2 = dist2.get(dtype, 0)
            
            dot_product += v1 * v2
            norm1 += v1 * v1
            norm2 += v2 * v2
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
            
        return dot_product / (norm1 ** 0.5 * norm2 ** 0.5)