from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Set, Tuple
import logging
from collections import defaultdict, Counter
from pathlib import Path
import re
from src.core.models import ValueSearchResult, ColumnInfo
from src.config.settings import settings

logger = logging.getLogger(__name__)


class ValueSearchEngine(ABC):
    """值搜索引擎抽象基类"""
    
    @abstractmethod
    async def add_column_values(self, column_info: ColumnInfo) -> None:
        """添加列的值到索引"""
        pass
    
    @abstractmethod
    async def search_by_values(
        self, 
        query_values: List[Any], 
        min_overlap_ratio: float = 0.3,
        k: int = 10
    ) -> List[ValueSearchResult]:
        """根据值重叠搜索相似列"""
        pass
    
    @abstractmethod
    async def save_index(self, file_path: str) -> None:
        """保存索引到文件"""
        pass
    
    @abstractmethod
    async def load_index(self, file_path: str) -> None:
        """从文件加载索引"""
        pass


class InMemoryValueSearch(ValueSearchEngine):
    """基于内存的值搜索实现"""
    
    def __init__(self):
        # 值到列的倒排索引: value -> set(column_full_names)
        self.value_to_columns = defaultdict(set)
        
        # 列的元数据: column_full_name -> ColumnInfo
        self.column_metadata = {}
        
        # 列的值集合: column_full_name -> set(values)
        self.column_values = {}
        
        logger.info("初始化内存值搜索引擎")
    
    def _normalize_value(self, value) -> Optional[str]:
        """标准化值，用于索引和比较"""
        if value is None or value == "":
            return None
        
        # 转换为字符串并标准化
        str_value = str(value).strip().lower()
        
        # 过滤掉太短或无意义的值
        if len(str_value) < 2:
            return None
        
        # 移除多余的空白字符
        str_value = re.sub(r'\s+', ' ', str_value)
        
        return str_value
    
    async def add_column_values(self, column_info: ColumnInfo) -> None:
        """添加列的值到索引"""
        try:
            column_full_name = column_info.full_name
            
            # 保存列元数据
            self.column_metadata[column_full_name] = column_info
            
            # 处理样本值
            normalized_values = set()
            for value in column_info.sample_values:
                normalized_value = self._normalize_value(value)
                if normalized_value:
                    normalized_values.add(normalized_value)
                    # 建立倒排索引
                    self.value_to_columns[normalized_value].add(column_full_name)
            
            # 保存列的值集合
            self.column_values[column_full_name] = normalized_values
            
            logger.debug(f"添加列值索引: {column_full_name}, 值数量: {len(normalized_values)}")
            
        except Exception as e:
            logger.error(f"添加列值索引失败: {e}")
            raise
    
    async def search_by_values(
        self, 
        query_values: List[Any], 
        min_overlap_ratio: float = 0.3,
        k: int = 10
    ) -> List[ValueSearchResult]:
        """根据值重叠搜索相似列"""
        try:
            if not query_values:
                logger.warning("查询值列表为空")
                return []
            
            # 标准化查询值
            normalized_query_values = set()
            for value in query_values:
                normalized_value = self._normalize_value(value)
                if normalized_value:
                    normalized_query_values.add(normalized_value)
            
            if not normalized_query_values:
                logger.warning("标准化后的查询值为空")
                return []
            
            # 统计候选列的重叠情况
            candidate_overlap = defaultdict(set)  # column_name -> set(overlapped_values)
            
            for query_value in normalized_query_values:
                if query_value in self.value_to_columns:
                    for column_name in self.value_to_columns[query_value]:
                        candidate_overlap[column_name].add(query_value)
            
            # 计算重叠比例并排序
            results = []
            for column_name, overlapped_values in candidate_overlap.items():
                if column_name in self.column_values:
                    column_values = self.column_values[column_name]
                    
                    # 计算重叠比例（基于查询值）
                    overlap_ratio = len(overlapped_values) / len(normalized_query_values)
                    
                    # 可选：也考虑目标列的重叠比例
                    target_overlap_ratio = len(overlapped_values) / len(column_values) if column_values else 0
                    
                    # 综合评分（可以调整权重）
                    combined_score = 0.7 * overlap_ratio + 0.3 * target_overlap_ratio
                    
                    if overlap_ratio >= min_overlap_ratio:
                        column_info = self.column_metadata.get(column_name)
                        if column_info:
                            results.append(ValueSearchResult(
                                item_id=column_name,
                                score=combined_score,
                                matched_values=list(overlapped_values),
                                overlap_ratio=overlap_ratio,
                                metadata={
                                    "table_name": column_info.table_name,
                                    "column_name": column_info.column_name,
                                    "data_type": column_info.data_type,
                                    "total_query_values": len(normalized_query_values),
                                    "total_column_values": len(column_values),
                                    "overlapped_count": len(overlapped_values),
                                    "target_overlap_ratio": target_overlap_ratio
                                }
                            ))
            
            # 按分数排序并返回top-k结果
            results.sort(key=lambda x: x.score, reverse=True)
            results = results[:k]
            
            logger.debug(f"值搜索返回{len(results)}个结果")
            return results
            
        except Exception as e:
            logger.error(f"值搜索失败: {e}")
            return []
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取索引统计信息"""
        return {
            "total_columns": len(self.column_metadata),
            "total_unique_values": len(self.value_to_columns),
            "avg_values_per_column": sum(len(values) for values in self.column_values.values()) / len(self.column_values) if self.column_values else 0,
            "top_frequent_values": dict(Counter(
                len(columns) for columns in self.value_to_columns.values()
            ).most_common(10))
        }
    
    async def search_by_pattern(
        self, 
        pattern: str, 
        k: int = 10
    ) -> List[ValueSearchResult]:
        """根据正则表达式模式搜索"""
        try:
            compiled_pattern = re.compile(pattern, re.IGNORECASE)
            matching_columns = defaultdict(list)  # column_name -> [matched_values]
            
            for value, columns in self.value_to_columns.items():
                if compiled_pattern.search(value):
                    for column_name in columns:
                        matching_columns[column_name].append(value)
            
            results = []
            for column_name, matched_values in matching_columns.items():
                column_info = self.column_metadata.get(column_name)
                if column_info:
                    # 计算匹配比例
                    total_values = len(self.column_values.get(column_name, set()))
                    match_ratio = len(matched_values) / total_values if total_values > 0 else 0
                    
                    results.append(ValueSearchResult(
                        item_id=column_name,
                        score=match_ratio,
                        matched_values=matched_values[:10],  # 限制返回的匹配值数量
                        overlap_ratio=match_ratio,
                        metadata={
                            "table_name": column_info.table_name,
                            "column_name": column_info.column_name,
                            "pattern": pattern,
                            "total_matches": len(matched_values),
                            "total_values": total_values
                        }
                    ))
            
            results.sort(key=lambda x: x.score, reverse=True)
            return results[:k]
            
        except Exception as e:
            logger.error(f"模式搜索失败: {e}")
            return []
    
    async def save_index(self, file_path: str) -> None:
        """保存索引到文件"""
        try:
            import pickle
            
            file_path = Path(file_path)
            file_path.parent.mkdir(parents=True, exist_ok=True)
            
            index_data = {
                'value_to_columns': dict(self.value_to_columns),
                'column_metadata': self.column_metadata,
                'column_values': self.column_values
            }
            
            with open(file_path / "value_index.pkl", 'wb') as f:
                pickle.dump(index_data, f)
            
            logger.info(f"值索引保存完成: {file_path}")
            
        except Exception as e:
            logger.error(f"保存值索引失败: {e}")
            raise
    
    async def load_index(self, file_path: str) -> None:
        """从文件加载索引"""
        try:
            import pickle
            
            file_path = Path(file_path)
            index_file = file_path / "value_index.pkl"
            
            if not index_file.exists():
                logger.warning(f"值索引文件不存在: {index_file}")
                return
            
            with open(index_file, 'rb') as f:
                index_data = pickle.load(f)
            
            # 恢复数据结构
            self.value_to_columns = defaultdict(set)
            for value, columns in index_data.get('value_to_columns', {}).items():
                self.value_to_columns[value] = set(columns)
            
            self.column_metadata = index_data.get('column_metadata', {})
            self.column_values = index_data.get('column_values', {})
            
            logger.info(f"值索引加载完成: {file_path}")
            logger.info(f"加载了 {len(self.column_metadata)} 个列, {len(self.value_to_columns)} 个唯一值")
            
        except Exception as e:
            logger.error(f"加载值索引失败: {e}")
            raise


class WhooshValueSearch(ValueSearchEngine):
    """基于Whoosh的值搜索实现（可选扩展）"""
    
    def __init__(self, index_path: str):
        self.index_path = Path(index_path)
        self.index = None
        self.writer = None
        logger.info(f"初始化Whoosh值搜索引擎: {index_path}")
        # TODO: 实现Whoosh索引逻辑
    
    async def add_column_values(self, column_info: ColumnInfo) -> None:
        # TODO: 实现Whoosh索引添加
        pass
    
    async def search_by_values(
        self, 
        query_values: List[Any], 
        min_overlap_ratio: float = 0.3,
        k: int = 10
    ) -> List[ValueSearchResult]:
        # TODO: 实现Whoosh搜索
        return []
    
    async def save_index(self, file_path: str) -> None:
        # TODO: Whoosh索引自动持久化
        pass
    
    async def load_index(self, file_path: str) -> None:
        # TODO: 实现Whoosh索引加载
        pass


def create_value_search_engine() -> ValueSearchEngine:
    """创建值搜索引擎实例"""
    provider = settings.index.provider.lower()
    
    if provider == "memory" or provider == "whoosh":
        # 目前只实现内存版本，后续可扩展Whoosh
        return InMemoryValueSearch()
    else:
        raise ValueError(f"不支持的值搜索引擎: {provider}")


# 全局值搜索引擎实例
value_search_engine = create_value_search_engine()