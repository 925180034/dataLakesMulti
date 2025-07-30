"""
LSH预过滤器 - Phase 2架构升级核心组件
基于局部敏感哈希的快速预筛选系统
"""

import numpy as np
import logging
from typing import List, Dict, Any, Optional, Tuple, Set
from dataclasses import dataclass
from pathlib import Path
import pickle
import hashlib
from src.config.settings import settings

logger = logging.getLogger(__name__)


@dataclass
class LSHConfig:
    """LSH配置类"""
    num_hash_functions: int = 64    # LSH哈希函数数量
    num_hash_tables: int = 8        # 哈希表数量
    signature_length: int = 32      # MinHash签名长度
    bands: int = 16                 # MinHash分带数量
    rows_per_band: int = 2          # 每带行数
    similarity_threshold: float = 0.5  # 相似度阈值
    max_candidates: int = 1000      # 最大候选数量


class MinHashLSH:
    """MinHash LSH实现 - 用于快速相似度预筛选"""
    
    def __init__(self, config: Optional[LSHConfig] = None):
        self.config = config or LSHConfig()
        self.hash_functions = []
        self.hash_tables = []
        self.signatures = {}  # item_id -> signature
        self.metadata = {}    # item_id -> metadata
        self.is_initialized = False
        
        self._initialize_hash_functions()
        self._initialize_hash_tables()
        
        logger.info(f"MinHash LSH初始化完成: {self.config.num_hash_functions}个哈希函数, "
                   f"{self.config.num_hash_tables}个哈希表")
    
    def _initialize_hash_functions(self):
        """初始化哈希函数"""
        np.random.seed(42)  # 确保可重复性
        
        # MinHash使用随机排列
        for _ in range(self.config.num_hash_functions):
            # 生成大质数参数
            a = np.random.randint(1, 2**31 - 1)
            b = np.random.randint(0, 2**31 - 1)
            prime = 2**31 - 1  # Mersenne prime
            
            self.hash_functions.append((a, b, prime))
    
    def _initialize_hash_tables(self):
        """初始化哈希表"""
        for _ in range(self.config.num_hash_tables):
            self.hash_tables.append({})
    
    def _compute_signature(self, features: Set[str]) -> List[int]:
        """计算MinHash签名"""
        if not features:
            return [0] * self.config.num_hash_functions
        
        signature = []
        
        for a, b, prime in self.hash_functions:
            min_hash = float('inf')
            
            for feature in features:
                # 使用字符串哈希
                feature_hash = int(hashlib.md5(feature.encode()).hexdigest(), 16)
                hash_value = (a * feature_hash + b) % prime
                min_hash = min(min_hash, hash_value)
            
            signature.append(int(min_hash))
        
        return signature
    
    def _get_bands(self, signature: List[int]) -> List[int]:
        """将签名分成带"""
        bands = []
        for i in range(0, len(signature), self.config.rows_per_band):
            band = tuple(signature[i:i + self.config.rows_per_band])
            bands.append(hash(band))
        return bands
    
    def add_item(self, item_id: str, features: Set[str], metadata: Optional[Dict] = None):
        """添加项目到LSH索引"""
        try:
            # 计算MinHash签名
            signature = self._compute_signature(features)
            self.signatures[item_id] = signature
            
            if metadata:
                self.metadata[item_id] = metadata
            
            # 添加到哈希表
            bands = self._get_bands(signature)
            
            for i, (band_hash, hash_table) in enumerate(zip(bands, self.hash_tables)):
                if i >= len(self.hash_tables):
                    break
                    
                if band_hash not in hash_table:
                    hash_table[band_hash] = set()
                hash_table[band_hash].add(item_id)
            
            logger.debug(f"添加项目到LSH: {item_id}")
            
        except Exception as e:
            logger.error(f"添加LSH项目失败 {item_id}: {e}")
    
    def query_candidates(self, query_features: Set[str], top_k: Optional[int] = None) -> List[str]:
        """查询候选项目"""
        try:
            if not query_features:
                return []
            
            # 计算查询签名
            query_signature = self._compute_signature(query_features)
            query_bands = self._get_bands(query_signature)
            
            # 收集候选项目
            candidates = set()
            
            for i, (band_hash, hash_table) in enumerate(zip(query_bands, self.hash_tables)):
                if i >= len(self.hash_tables):
                    break
                    
                if band_hash in hash_table:
                    candidates.update(hash_table[band_hash])
            
            # 转换为列表并限制数量
            candidate_list = list(candidates)
            
            if top_k:
                candidate_list = candidate_list[:top_k]
            
            logger.debug(f"LSH查询返回 {len(candidate_list)} 个候选")
            return candidate_list
            
        except Exception as e:
            logger.error(f"LSH查询失败: {e}")
            return []
    
    def compute_jaccard_similarity(self, signature1: List[int], signature2: List[int]) -> float:
        """计算Jaccard相似度估计"""
        if len(signature1) != len(signature2):
            return 0.0
        
        matches = sum(1 for s1, s2 in zip(signature1, signature2) if s1 == s2)
        return matches / len(signature1)
    
    def get_similar_items(self, item_id: str, threshold: Optional[float] = None) -> List[Tuple[str, float]]:
        """获取相似项目及其相似度"""
        try:
            if item_id not in self.signatures:
                return []
            
            threshold = threshold or self.config.similarity_threshold
            item_signature = self.signatures[item_id]
            
            # 获取候选项目
            item_features = set()  # 简化处理，实际应从元数据获取
            candidates = self.query_candidates(item_features)
            
            # 计算相似度
            similar_items = []
            
            for candidate_id in candidates:
                if candidate_id == item_id:
                    continue
                
                candidate_signature = self.signatures[candidate_id]
                similarity = self.compute_jaccard_similarity(item_signature, candidate_signature)
                
                if similarity >= threshold:
                    similar_items.append((candidate_id, similarity))
            
            # 按相似度排序
            similar_items.sort(key=lambda x: x[1], reverse=True)
            
            return similar_items
            
        except Exception as e:
            logger.error(f"获取相似项目失败 {item_id}: {e}")
            return []
    
    def save_index(self, file_path: str):
        """保存LSH索引"""
        try:
            file_path = Path(file_path)
            file_path.parent.mkdir(parents=True, exist_ok=True)
            
            index_data = {
                'config': self.config,
                'hash_functions': self.hash_functions,
                'hash_tables': self.hash_tables,
                'signatures': self.signatures,
                'metadata': self.metadata
            }
            
            with open(file_path, 'wb') as f:
                pickle.dump(index_data, f)
            
            logger.info(f"LSH索引保存完成: {file_path}")
            
        except Exception as e:
            logger.error(f"保存LSH索引失败: {e}")
            raise
    
    def load_index(self, file_path: str):
        """加载LSH索引"""
        try:
            file_path = Path(file_path)
            
            if not file_path.exists():
                logger.warning(f"LSH索引文件不存在: {file_path}")
                return
            
            with open(file_path, 'rb') as f:
                index_data = pickle.load(f)
            
            self.config = index_data['config']
            self.hash_functions = index_data['hash_functions']
            self.hash_tables = index_data['hash_tables']
            self.signatures = index_data['signatures']
            self.metadata = index_data['metadata']
            self.is_initialized = True
            
            logger.info(f"LSH索引加载完成: {file_path}")
            logger.info(f"加载了 {len(self.signatures)} 个项目")
            
        except Exception as e:
            logger.error(f"加载LSH索引失败: {e}")
            raise
    
    def get_stats(self) -> Dict[str, Any]:
        """获取LSH统计信息"""
        total_buckets = sum(len(table) for table in self.hash_tables)
        avg_bucket_size = 0
        if total_buckets > 0:
            total_items_in_buckets = sum(
                len(bucket) for table in self.hash_tables 
                for bucket in table.values()
            )
            avg_bucket_size = total_items_in_buckets / total_buckets
        
        return {
            'total_items': len(self.signatures),
            'hash_tables': len(self.hash_tables),
            'hash_functions': len(self.hash_functions),
            'total_buckets': total_buckets,
            'avg_bucket_size': avg_bucket_size,
            'config': {
                'num_hash_functions': self.config.num_hash_functions,
                'num_hash_tables': self.config.num_hash_tables,
                'similarity_threshold': self.config.similarity_threshold
            }
        }


class LSHPrefilter:
    """LSH预过滤器 - 集成到搜索管道"""
    
    def __init__(self, config: Optional[LSHConfig] = None):
        self.config = config or LSHConfig()
        self.column_lsh = MinHashLSH(self.config)
        self.table_lsh = MinHashLSH(self.config)
        self.is_built = False
        
        logger.info("LSH预过滤器初始化完成")
    
    def build_column_index(self, columns_data: List[Dict[str, Any]]):
        """构建列级别LSH索引"""
        try:
            logger.info(f"构建列LSH索引，处理 {len(columns_data)} 个列")
            
            for column_data in columns_data:
                column_id = column_data.get('column_id', '')
                column_name = column_data.get('column_name', '')
                data_type = column_data.get('data_type', '')
                sample_values = column_data.get('sample_values', [])
                
                # 构建特征集合
                features = set()
                
                # 添加列名特征
                if column_name:
                    features.add(f"name:{column_name.lower()}")
                    # 添加列名的词汇特征
                    words = column_name.lower().replace('_', ' ').split()
                    for word in words:
                        features.add(f"word:{word}")
                
                # 添加数据类型特征
                if data_type:
                    features.add(f"type:{data_type.lower()}")
                
                # 添加值分布特征
                for value in sample_values[:10]:  # 限制样本数量
                    if value is not None:
                        # 添加具体值
                        features.add(f"value:{str(value).lower()}")
                        
                        # 添加值类型特征
                        try:
                            float(value)
                            features.add("value_type:numeric")
                        except:
                            features.add("value_type:text")
                
                # 添加到LSH索引
                metadata = {
                    'table_name': column_data.get('table_name', ''),
                    'column_name': column_name,
                    'data_type': data_type,
                    'sample_count': len(sample_values)
                }
                
                self.column_lsh.add_item(column_id, features, metadata)
            
            self.is_built = True
            logger.info("列LSH索引构建完成")
            
        except Exception as e:
            logger.error(f"构建列LSH索引失败: {e}")
            raise
    
    def build_table_index(self, tables_data: List[Dict[str, Any]]):
        """构建表级别LSH索引"""
        try:
            logger.info(f"构建表LSH索引，处理 {len(tables_data)} 个表")
            
            for table_data in tables_data:
                table_id = table_data.get('table_name', '')
                table_name = table_data.get('table_name', '')
                columns = table_data.get('columns', [])
                
                # 构建表级别特征集合
                features = set()
                
                # 添加表名特征
                if table_name:
                    features.add(f"table:{table_name.lower()}")
                    words = table_name.lower().replace('_', ' ').split()
                    for word in words:
                        features.add(f"table_word:{word}")
                
                # 添加列名特征
                for column in columns:
                    column_name = column.get('column_name', '')
                    if column_name:
                        features.add(f"has_column:{column_name.lower()}")
                        
                        # 添加数据类型特征
                        data_type = column.get('data_type', '')
                        if data_type:
                            features.add(f"has_type:{data_type.lower()}")
                
                # 添加结构特征
                features.add(f"column_count:{len(columns)}")
                if len(columns) <= 5:
                    features.add("table_size:small")
                elif len(columns) <= 15:
                    features.add("table_size:medium")
                else:
                    features.add("table_size:large")
                
                # 添加到LSH索引
                metadata = {
                    'table_name': table_name,
                    'column_count': len(columns),
                    'columns': [col.get('column_name', '') for col in columns]
                }
                
                self.table_lsh.add_item(table_id, features, metadata)
            
            logger.info("表LSH索引构建完成")
            
        except Exception as e:
            logger.error(f"构建表LSH索引失败: {e}")
            raise
    
    def prefilter_columns(self, query_column: Dict[str, Any], max_candidates: Optional[int] = None) -> List[str]:
        """预过滤列候选"""
        try:
            # 构建查询特征
            column_name = query_column.get('column_name', '')
            data_type = query_column.get('data_type', '')
            sample_values = query_column.get('sample_values', [])
            
            features = set()
            
            if column_name:
                features.add(f"name:{column_name.lower()}")
                words = column_name.lower().replace('_', ' ').split()
                for word in words:
                    features.add(f"word:{word}")
            
            if data_type:
                features.add(f"type:{data_type.lower()}")
            
            for value in sample_values[:10]:
                if value is not None:
                    features.add(f"value:{str(value).lower()}")
                    try:
                        float(value)
                        features.add("value_type:numeric")
                    except:
                        features.add("value_type:text")
            
            # 查询候选
            max_candidates = max_candidates or self.config.max_candidates
            candidates = self.column_lsh.query_candidates(features, top_k=max_candidates)
            
            logger.debug(f"列预过滤返回 {len(candidates)} 个候选")
            return candidates
            
        except Exception as e:
            logger.error(f"列预过滤失败: {e}")
            return []
    
    def prefilter_tables(self, query_table: Dict[str, Any], max_candidates: Optional[int] = None) -> List[str]:
        """预过滤表候选"""
        try:
            # 构建查询特征
            table_name = query_table.get('table_name', '')
            columns = query_table.get('columns', [])
            
            features = set()
            
            if table_name:
                features.add(f"table:{table_name.lower()}")
                words = table_name.lower().replace('_', ' ').split()
                for word in words:
                    features.add(f"table_word:{word}")
            
            for column in columns:
                column_name = column.get('column_name', '')
                if column_name:
                    features.add(f"has_column:{column_name.lower()}")
                    
                data_type = column.get('data_type', '')
                if data_type:
                    features.add(f"has_type:{data_type.lower()}")
            
            features.add(f"column_count:{len(columns)}")
            if len(columns) <= 5:
                features.add("table_size:small")
            elif len(columns) <= 15:
                features.add("table_size:medium")
            else:
                features.add("table_size:large")
            
            # 查询候选
            max_candidates = max_candidates or self.config.max_candidates
            candidates = self.table_lsh.query_candidates(features, top_k=max_candidates)
            
            logger.debug(f"表预过滤返回 {len(candidates)} 个候选")
            return candidates
            
        except Exception as e:
            logger.error(f"表预过滤失败: {e}")
            return []
    
    def save_indices(self, base_path: str):
        """保存所有LSH索引"""
        try:
            base_path = Path(base_path)
            base_path.mkdir(parents=True, exist_ok=True)
            
            self.column_lsh.save_index(str(base_path / "column_lsh.pkl"))
            self.table_lsh.save_index(str(base_path / "table_lsh.pkl"))
            
            logger.info(f"LSH索引保存完成: {base_path}")
            
        except Exception as e:
            logger.error(f"保存LSH索引失败: {e}")
            raise
    
    def load_indices(self, base_path: str):
        """加载所有LSH索引"""
        try:
            base_path = Path(base_path)
            
            column_path = base_path / "column_lsh.pkl"
            table_path = base_path / "table_lsh.pkl"
            
            if column_path.exists():
                self.column_lsh.load_index(str(column_path))
            
            if table_path.exists():
                self.table_lsh.load_index(str(table_path))
                
            self.is_built = True
            logger.info(f"LSH索引加载完成: {base_path}")
            
        except Exception as e:
            logger.error(f"加载LSH索引失败: {e}")
            raise
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """获取性能统计"""
        return {
            'is_built': self.is_built,
            'config': {
                'num_hash_functions': self.config.num_hash_functions,
                'num_hash_tables': self.config.num_hash_tables,
                'similarity_threshold': self.config.similarity_threshold,
                'max_candidates': self.config.max_candidates
            },
            'column_lsh_stats': self.column_lsh.get_stats(),
            'table_lsh_stats': self.table_lsh.get_stats()
        }


def create_lsh_prefilter(config: Optional[LSHConfig] = None) -> LSHPrefilter:
    """创建LSH预过滤器实例"""
    return LSHPrefilter(config)


# 全局LSH预过滤器实例
_global_lsh_prefilter = None

def get_lsh_prefilter() -> LSHPrefilter:
    """获取全局LSH预过滤器实例"""
    global _global_lsh_prefilter
    if _global_lsh_prefilter is None:
        _global_lsh_prefilter = create_lsh_prefilter()
    return _global_lsh_prefilter