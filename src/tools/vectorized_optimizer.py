"""
向量化计算优化器 - Phase 2架构升级组件
基于NumPy/CuPy的高性能批量计算系统
"""

import numpy as np
import logging
from typing import List, Dict, Any, Optional, Tuple, Union
import time
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing as mp
from src.config.settings import settings

logger = logging.getLogger(__name__)

# 尝试导入CUDA支持
try:
    import cupy as cp
    CUDA_AVAILABLE = True
    logger.info("CUDA支持已启用")
except ImportError:
    cp = None
    CUDA_AVAILABLE = False
    logger.info("CUDA不可用，使用CPU计算")


@dataclass
class VectorizedConfig:
    """向量化计算配置"""
    batch_size: int = 1000           # 批处理大小
    use_gpu: bool = False            # 是否使用GPU
    max_workers: int = mp.cpu_count()  # 最大工作进程数
    chunk_size: int = 100            # 分块大小
    parallel_threshold: int = 10000   # 并行处理阈值
    memory_limit_mb: int = 2048      # 内存限制(MB)
    precision: str = 'float32'       # 计算精度


class VectorizedCalculator:
    """向量化相似度计算器"""
    
    def __init__(self, config: Optional[VectorizedConfig] = None):
        self.config = config or VectorizedConfig()
        self.xp = cp if (self.config.use_gpu and CUDA_AVAILABLE) else np
        
        # 性能统计
        self.stats = {
            'total_computations': 0,
            'gpu_computations': 0,
            'cpu_computations': 0,
            'avg_computation_time': 0.0,
            'memory_usage_mb': 0.0
        }
        
        logger.info(f"向量化计算器初始化: {'GPU' if self.xp == cp else 'CPU'}模式")
    
    def batch_cosine_similarity(
        self, 
        vectors1: np.ndarray, 
        vectors2: np.ndarray
    ) -> np.ndarray:
        """批量计算余弦相似度"""
        try:
            start_time = time.time()
            
            # 转换为配置的数据类型
            if vectors1.dtype != self.config.precision:
                vectors1 = vectors1.astype(self.config.precision)
            if vectors2.dtype != self.config.precision:
                vectors2 = vectors2.astype(self.config.precision)
            
            # 转移到GPU（如果可用）
            if self.xp == cp:
                vectors1 = cp.asarray(vectors1)
                vectors2 = cp.asarray(vectors2)
            
            # 归一化向量
            norm1 = self.xp.linalg.norm(vectors1, axis=1, keepdims=True)
            norm2 = self.xp.linalg.norm(vectors2, axis=1, keepdims=True)
            
            # 避免除零
            norm1 = self.xp.where(norm1 == 0, 1, norm1)
            norm2 = self.xp.where(norm2 == 0, 1, norm2)
            
            normalized1 = vectors1 / norm1
            normalized2 = vectors2 / norm2
            
            # 计算余弦相似度
            similarity_matrix = self.xp.dot(normalized1, normalized2.T)
            
            # 转回CPU（如果使用GPU）
            if self.xp == cp:
                similarity_matrix = cp.asnumpy(similarity_matrix)
                self.stats['gpu_computations'] += 1
            else:
                self.stats['cpu_computations'] += 1
            
            # 更新统计
            computation_time = time.time() - start_time
            self.stats['total_computations'] += 1
            self.stats['avg_computation_time'] = (
                (self.stats['avg_computation_time'] * (self.stats['total_computations'] - 1) + 
                 computation_time) / self.stats['total_computations']
            )
            
            logger.debug(f"批量余弦相似度计算完成: {vectors1.shape} x {vectors2.shape}, "
                        f"用时 {computation_time:.3f}s")
            
            return similarity_matrix
            
        except Exception as e:
            logger.error(f"批量余弦相似度计算失败: {e}")
            # 降级到普通计算
            return self._fallback_cosine_similarity(vectors1, vectors2)
    
    def _fallback_cosine_similarity(self, vectors1: np.ndarray, vectors2: np.ndarray) -> np.ndarray:
        """降级余弦相似度计算"""
        try:
            from sklearn.metrics.pairwise import cosine_similarity
            return cosine_similarity(vectors1, vectors2)
        except ImportError:
            # 手动实现
            similarity_matrix = np.zeros((vectors1.shape[0], vectors2.shape[0]))
            for i, v1 in enumerate(vectors1):
                for j, v2 in enumerate(vectors2):
                    norm1 = np.linalg.norm(v1)
                    norm2 = np.linalg.norm(v2)
                    if norm1 > 0 and norm2 > 0:
                        similarity_matrix[i, j] = np.dot(v1, v2) / (norm1 * norm2)
            return similarity_matrix
    
    def batch_euclidean_distance(
        self, 
        vectors1: np.ndarray, 
        vectors2: np.ndarray
    ) -> np.ndarray:
        """批量计算欧几里得距离"""
        try:
            start_time = time.time()
            
            if self.xp == cp:
                vectors1 = cp.asarray(vectors1.astype(self.config.precision))
                vectors2 = cp.asarray(vectors2.astype(self.config.precision))
            
            # 使用广播计算距离矩阵
            # ||a - b||^2 = ||a||^2 + ||b||^2 - 2<a,b>
            sq_norms1 = self.xp.sum(vectors1**2, axis=1, keepdims=True)
            sq_norms2 = self.xp.sum(vectors2**2, axis=1, keepdims=True)
            
            dot_product = self.xp.dot(vectors1, vectors2.T)
            
            distances = sq_norms1 + sq_norms2.T - 2 * dot_product
            distances = self.xp.sqrt(self.xp.maximum(distances, 0))  # 避免负数开方
            
            if self.xp == cp:
                distances = cp.asnumpy(distances)
                self.stats['gpu_computations'] += 1
            else:
                self.stats['cpu_computations'] += 1
            
            computation_time = time.time() - start_time
            logger.debug(f"批量欧几里得距离计算完成: {vectors1.shape} x {vectors2.shape}, "
                        f"用时 {computation_time:.3f}s")
            
            return distances
            
        except Exception as e:
            logger.error(f"批量欧几里得距离计算失败: {e}")
            return np.zeros((vectors1.shape[0], vectors2.shape[0]))
    
    def parallel_similarity_computation(
        self,
        query_vectors: List[np.ndarray],
        candidate_vectors: List[np.ndarray],
        similarity_func: str = 'cosine'
    ) -> List[np.ndarray]:
        """并行相似度计算"""
        try:
            if len(query_vectors) * len(candidate_vectors) < self.config.parallel_threshold:
                # 小规模数据直接计算
                return self._sequential_computation(query_vectors, candidate_vectors, similarity_func)
            
            # 大规模数据并行处理
            logger.info(f"启动并行计算: {len(query_vectors)} x {len(candidate_vectors)}")
            
            with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
                futures = []
                
                # 分批处理
                for i in range(0, len(query_vectors), self.config.chunk_size):
                    chunk_queries = query_vectors[i:i + self.config.chunk_size]
                    
                    future = executor.submit(
                        self._compute_chunk_similarities,
                        chunk_queries,
                        candidate_vectors,
                        similarity_func
                    )
                    futures.append(future)
                
                # 收集结果
                results = []
                for future in futures:
                    chunk_results = future.result()
                    results.extend(chunk_results)
                
                logger.info("并行计算完成")
                return results
                
        except Exception as e:
            logger.error(f"并行相似度计算失败: {e}")
            return self._sequential_computation(query_vectors, candidate_vectors, similarity_func)
    
    def _sequential_computation(
        self,
        query_vectors: List[np.ndarray],
        candidate_vectors: List[np.ndarray],
        similarity_func: str
    ) -> List[np.ndarray]:
        """顺序计算"""
        results = []
        
        for query_vector in query_vectors:
            query_array = np.array([query_vector])
            candidate_array = np.vstack(candidate_vectors)
            
            if similarity_func == 'cosine':
                similarity_matrix = self.batch_cosine_similarity(query_array, candidate_array)
            elif similarity_func == 'euclidean':
                similarity_matrix = self.batch_euclidean_distance(query_array, candidate_array)
            else:
                raise ValueError(f"不支持的相似度函数: {similarity_func}")
            
            results.append(similarity_matrix[0])  # 取第一行
        
        return results
    
    def _compute_chunk_similarities(
        self,
        chunk_queries: List[np.ndarray],
        candidate_vectors: List[np.ndarray],
        similarity_func: str
    ) -> List[np.ndarray]:
        """计算块的相似度"""
        chunk_results = []
        
        query_array = np.vstack(chunk_queries)
        candidate_array = np.vstack(candidate_vectors)
        
        if similarity_func == 'cosine':
            similarity_matrix = self.batch_cosine_similarity(query_array, candidate_array)
        elif similarity_func == 'euclidean':
            similarity_matrix = self.batch_euclidean_distance(query_array, candidate_array)
        else:
            raise ValueError(f"不支持的相似度函数: {similarity_func}")
        
        # 分解矩阵为单独的查询结果
        for i in range(len(chunk_queries)):
            chunk_results.append(similarity_matrix[i])
        
        return chunk_results
    
    def optimize_memory_usage(self, vectors: List[np.ndarray]) -> List[np.ndarray]:
        """优化内存使用"""
        try:
            # 估算内存使用
            total_elements = sum(v.size for v in vectors)
            estimated_memory_mb = (total_elements * 4) / (1024 * 1024)  # float32
            
            if estimated_memory_mb > self.config.memory_limit_mb:
                logger.warning(f"内存使用预计超限: {estimated_memory_mb:.1f}MB > {self.config.memory_limit_mb}MB")
                
                # 降低精度
                if self.config.precision == 'float64':
                    self.config.precision = 'float32'
                    logger.info("降低计算精度为float32以节省内存")
                elif self.config.precision == 'float32':
                    self.config.precision = 'float16'
                    logger.info("降低计算精度为float16以节省内存")
            
            # 转换精度
            optimized_vectors = []
            for vector in vectors:
                if vector.dtype != self.config.precision:
                    optimized_vectors.append(vector.astype(self.config.precision))
                else:
                    optimized_vectors.append(vector)
            
            self.stats['memory_usage_mb'] = estimated_memory_mb
            return optimized_vectors
            
        except Exception as e:
            logger.error(f"内存优化失败: {e}")
            return vectors
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """获取性能统计"""
        gpu_ratio = 0
        if self.stats['total_computations'] > 0:
            gpu_ratio = self.stats['gpu_computations'] / self.stats['total_computations']
        
        return {
            'config': {
                'use_gpu': self.config.use_gpu,
                'cuda_available': CUDA_AVAILABLE,
                'batch_size': self.config.batch_size,
                'max_workers': self.config.max_workers,
                'precision': self.config.precision
            },
            'stats': self.stats.copy(),
            'gpu_utilization_ratio': gpu_ratio,
            'backend': 'CuPy' if self.xp == cp else 'NumPy'
        }


class VectorizedHybridSimilarity:
    """向量化混合相似度计算器"""
    
    def __init__(self, config: Optional[VectorizedConfig] = None):
        self.calculator = VectorizedCalculator(config)
        self.weights = {
            'cosine': 0.4,
            'euclidean': 0.2,
            'jaccard': 0.2,
            'semantic': 0.2
        }
    
    def compute_hybrid_similarity(
        self,
        vectors1: np.ndarray,
        vectors2: np.ndarray,
        metadata1: Optional[List[Dict]] = None,
        metadata2: Optional[List[Dict]] = None
    ) -> np.ndarray:
        """计算混合相似度"""
        try:
            # 余弦相似度
            cosine_sim = self.calculator.batch_cosine_similarity(vectors1, vectors2)
            
            # 欧几里得距离 (转换为相似度)
            euclidean_dist = self.calculator.batch_euclidean_distance(vectors1, vectors2)
            euclidean_sim = 1 / (1 + euclidean_dist)  # 转换为相似度
            
            # 组合相似度
            hybrid_similarity = (
                self.weights['cosine'] * cosine_sim +
                self.weights['euclidean'] * euclidean_sim
            )
            
            # 如果有元数据，添加语义相似度
            if metadata1 and metadata2:
                semantic_sim = self._compute_semantic_similarity(metadata1, metadata2)
                hybrid_similarity += self.weights['semantic'] * semantic_sim
            
            return hybrid_similarity
            
        except Exception as e:
            logger.error(f"混合相似度计算失败: {e}")
            return np.zeros((vectors1.shape[0], vectors2.shape[0]))
    
    def _compute_semantic_similarity(
        self, 
        metadata1: List[Dict], 
        metadata2: List[Dict]
    ) -> np.ndarray:
        """计算语义相似度"""
        # 简化实现，基于元数据字段匹配
        similarity_matrix = np.zeros((len(metadata1), len(metadata2)))
        
        for i, meta1 in enumerate(metadata1):
            for j, meta2 in enumerate(metadata2):
                # 数据类型匹配
                type_match = 0
                if meta1.get('data_type') == meta2.get('data_type'):
                    type_match = 1
                
                # 列名相似度（简化）
                name1 = meta1.get('column_name', '').lower()
                name2 = meta2.get('column_name', '').lower()
                name_match = 1 if name1 == name2 else 0
                
                # 组合语义分数
                similarity_matrix[i, j] = 0.6 * type_match + 0.4 * name_match
        
        return similarity_matrix


def create_vectorized_optimizer(config: Optional[VectorizedConfig] = None) -> VectorizedCalculator:
    """创建向量化优化器"""
    return VectorizedCalculator(config)


def create_hybrid_similarity_calculator(config: Optional[VectorizedConfig] = None) -> VectorizedHybridSimilarity:
    """创建混合相似度计算器"""
    return VectorizedHybridSimilarity(config)


# 全局实例
_global_vectorized_calculator = None
_global_hybrid_calculator = None

def get_vectorized_calculator() -> VectorizedCalculator:
    """获取全局向量化计算器"""
    global _global_vectorized_calculator
    if _global_vectorized_calculator is None:
        _global_vectorized_calculator = create_vectorized_optimizer()
    return _global_vectorized_calculator

def get_hybrid_similarity_calculator() -> VectorizedHybridSimilarity:
    """获取全局混合相似度计算器"""
    global _global_hybrid_calculator
    if _global_hybrid_calculator is None:
        _global_hybrid_calculator = create_hybrid_similarity_calculator()
    return _global_hybrid_calculator