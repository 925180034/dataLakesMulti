"""
优化版向量化计算器 - Phase 2性能优化
解决大规模矩阵计算性能问题，实现智能分块和自适应算法选择
"""

import numpy as np
import logging
from typing import List, Dict, Any, Optional, Tuple, Union
import time
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
import multiprocessing as mp
from functools import lru_cache
import gc
from src.config.settings import settings
from src.tools.performance_profiler import profile_component

logger = logging.getLogger(__name__)

# 尝试导入高性能BLAS库
try:
    import scipy.linalg.lapack
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

# 尝试导入CUDA支持
try:
    import cupy as cp
    CUDA_AVAILABLE = True
    logger.info("CUDA支持已启用")
except ImportError:
    cp = None
    CUDA_AVAILABLE = False


@dataclass
class OptimizedVectorizedConfig:
    """优化版向量化计算配置"""
    # 基础配置
    batch_size: int = 1000
    use_gpu: bool = False
    max_workers: int = mp.cpu_count()
    precision: str = 'float32'
    
    # 优化配置
    enable_chunking: bool = True          # 启用智能分块
    chunk_size_mb: int = 64               # 分块大小(MB)
    memory_pool_size_mb: int = 512        # 内存池大小(MB)
    enable_adaptive_algorithm: bool = True # 自适应算法选择
    cache_intermediate_results: bool = True # 缓存中间结果
    
    # 性能阈值
    large_matrix_threshold: int = 1000     # 大矩阵阈值
    gpu_threshold: int = 5000             # GPU加速阈值
    parallel_threshold: int = 100         # 并行处理阈值
    
    # 算法选择
    small_matrix_algorithm: str = 'direct'    # small: direct, chunked
    medium_matrix_algorithm: str = 'chunked'  # medium: direct, chunked, parallel
    large_matrix_algorithm: str = 'parallel'  # large: chunked, parallel, gpu


class MemoryPool:
    """内存池管理器"""
    
    def __init__(self, pool_size_mb: int = 512):
        self.pool_size_bytes = pool_size_mb * 1024 * 1024
        self.allocated_memory = {}
        self.free_memory = []
        self.total_allocated = 0
        
        logger.info(f"内存池初始化: {pool_size_mb}MB")
    
    def allocate(self, shape: Tuple[int, ...], dtype: str = 'float32') -> np.ndarray:
        """分配内存"""
        try:
            # 计算所需内存
            required_bytes = np.prod(shape) * np.dtype(dtype).itemsize
            
            # 检查是否有可复用的内存
            for i, (array, allocated_shape, allocated_dtype) in enumerate(self.free_memory):
                if (allocated_shape == shape and allocated_dtype == dtype):
                    # 复用内存
                    self.free_memory.pop(i)
                    return array
            
            # 检查内存池容量
            if self.total_allocated + required_bytes > self.pool_size_bytes:
                self._cleanup_memory()
            
            # 分配新内存
            array = np.empty(shape, dtype=dtype)
            self.total_allocated += required_bytes
            
            return array
            
        except Exception as e:
            logger.warning(f"内存池分配失败，使用常规分配: {e}")
            return np.empty(shape, dtype=dtype)
    
    def deallocate(self, array: np.ndarray):
        """释放内存到池中"""
        try:
            shape = array.shape
            dtype = str(array.dtype)
            
            # 添加到空闲内存列表
            if len(self.free_memory) < 10:  # 限制缓存数量
                self.free_memory.append((array, shape, dtype))
            else:
                # 释放内存
                array_bytes = array.nbytes
                self.total_allocated -= array_bytes
                del array
                
        except Exception as e:
            logger.warning(f"内存释放失败: {e}")
    
    def _cleanup_memory(self):
        """清理内存池"""
        # 清理最老的内存
        if self.free_memory:
            array, _, _ = self.free_memory.pop(0)
            self.total_allocated -= array.nbytes
            del array
            gc.collect()


class OptimizedVectorizedCalculator:
    """优化版向量化相似度计算器"""
    
    def __init__(self, config: Optional[OptimizedVectorizedConfig] = None):
        self.config = config or OptimizedVectorizedConfig()
        self.xp = cp if (self.config.use_gpu and CUDA_AVAILABLE) else np
        
        # 内存池
        self.memory_pool = MemoryPool(self.config.memory_pool_size_mb)
        
        # 性能统计
        self.stats = {
            'total_computations': 0,
            'gpu_computations': 0,
            'cpu_computations': 0,
            'chunked_computations': 0,
            'parallel_computations': 0,
            'cache_hits': 0,
            'avg_computation_time': 0.0,
            'memory_saved_mb': 0.0
        }
        
        # 结果缓存
        self.result_cache = {} if self.config.cache_intermediate_results else None
        
        logger.info(f"优化向量化计算器初始化: {'GPU' if self.xp == cp else 'CPU'}模式, "
                   f"分块={'启用' if self.config.enable_chunking else '禁用'}")
    
    @profile_component("optimized_vectorized")
    def optimized_cosine_similarity(
        self, 
        vectors1: np.ndarray, 
        vectors2: np.ndarray
    ) -> np.ndarray:
        """优化版余弦相似度计算"""
        try:
            start_time = time.time()
            
            # 数据预处理
            vectors1, vectors2 = self._preprocess_vectors(vectors1, vectors2)
            
            # 选择最优算法
            algorithm = self._select_algorithm(vectors1.shape, vectors2.shape)
            
            # 执行计算
            if algorithm == 'direct':
                result = self._direct_cosine_similarity(vectors1, vectors2)
            elif algorithm == 'chunked':
                result = self._chunked_cosine_similarity(vectors1, vectors2)
            elif algorithm == 'parallel':
                result = self._parallel_cosine_similarity(vectors1, vectors2)
            elif algorithm == 'gpu':
                result = self._gpu_cosine_similarity(vectors1, vectors2)
            else:
                result = self._fallback_cosine_similarity(vectors1, vectors2)
            
            # 更新统计
            computation_time = time.time() - start_time
            self._update_stats(algorithm, computation_time)
            
            logger.debug(f"优化余弦相似度: {vectors1.shape} x {vectors2.shape}, "
                        f"算法={algorithm}, 用时={computation_time:.3f}s")
            
            return result
            
        except Exception as e:
            logger.error(f"优化余弦相似度计算失败: {e}")
            return self._fallback_cosine_similarity(vectors1, vectors2)
    
    def _preprocess_vectors(self, vectors1: np.ndarray, vectors2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """向量预处理"""
        # 类型转换
        if vectors1.dtype != self.config.precision:
            vectors1 = vectors1.astype(self.config.precision)
        if vectors2.dtype != self.config.precision:
            vectors2 = vectors2.astype(self.config.precision)
        
        # 内存连续性优化
        if not vectors1.flags['C_CONTIGUOUS']:
            vectors1 = np.ascontiguousarray(vectors1)
        if not vectors2.flags['C_CONTIGUOUS']:
            vectors2 = np.ascontiguousarray(vectors2)
        
        return vectors1, vectors2
    
    def _select_algorithm(self, shape1: Tuple[int, ...], shape2: Tuple[int, ...]) -> str:
        """自适应算法选择"""
        if not self.config.enable_adaptive_algorithm:
            return 'direct'
        
        rows1, cols = shape1
        rows2 = shape2[0]
        
        # 计算矩阵规模
        total_elements = rows1 * rows2 * cols
        result_elements = rows1 * rows2
        
        # GPU优先（如果可用且数据足够大）
        if (self.config.use_gpu and CUDA_AVAILABLE and 
            result_elements > self.config.gpu_threshold ** 2):
            return 'gpu'
        
        # 小矩阵直接计算
        if result_elements < self.config.parallel_threshold ** 2:
            return self.config.small_matrix_algorithm
        
        # 中等矩阵
        if result_elements < self.config.large_matrix_threshold ** 2:
            return self.config.medium_matrix_algorithm
        
        # 大矩阵
        return self.config.large_matrix_algorithm
    
    def _direct_cosine_similarity(self, vectors1: np.ndarray, vectors2: np.ndarray) -> np.ndarray:
        """直接计算余弦相似度"""
        # 归一化
        norm1 = np.linalg.norm(vectors1, axis=1, keepdims=True)
        norm2 = np.linalg.norm(vectors2, axis=1, keepdims=True)
        
        # 避免除零
        norm1 = np.where(norm1 == 0, 1, norm1)
        norm2 = np.where(norm2 == 0, 1, norm2)
        
        normalized1 = vectors1 / norm1
        normalized2 = vectors2 / norm2
        
        # 计算相似度矩阵
        return np.dot(normalized1, normalized2.T)
    
    def _chunked_cosine_similarity(self, vectors1: np.ndarray, vectors2: np.ndarray) -> np.ndarray:
        """分块计算余弦相似度"""
        self.stats['chunked_computations'] += 1
        
        rows1, dim = vectors1.shape
        rows2 = vectors2.shape[0]
        
        # 计算最优分块大小
        chunk_size = self._calculate_optimal_chunk_size(vectors1.dtype, dim)
        
        # 预分配结果矩阵
        result = self.memory_pool.allocate((rows1, rows2), self.config.precision)
        
        try:
            # 预先归一化所有向量
            norm1 = np.linalg.norm(vectors1, axis=1, keepdims=True)
            norm2 = np.linalg.norm(vectors2, axis=1, keepdims=True)
            
            norm1 = np.where(norm1 == 0, 1, norm1)
            norm2 = np.where(norm2 == 0, 1, norm2)
            
            normalized1 = vectors1 / norm1
            normalized2 = vectors2 / norm2
            
            # 分块计算
            for i in range(0, rows1, chunk_size):
                end_i = min(i + chunk_size, rows1)
                chunk1 = normalized1[i:end_i]
                
                for j in range(0, rows2, chunk_size):
                    end_j = min(j + chunk_size, rows2)
                    chunk2 = normalized2[j:end_j]
                    
                    # 计算块的相似度
                    chunk_result = np.dot(chunk1, chunk2.T)
                    result[i:end_i, j:end_j] = chunk_result
            
            return result.copy()  # 返回副本以便内存池管理
            
        finally:
            self.memory_pool.deallocate(result)
    
    def _parallel_cosine_similarity(self, vectors1: np.ndarray, vectors2: np.ndarray) -> np.ndarray:
        """并行计算余弦相似度"""
        self.stats['parallel_computations'] += 1
        
        rows1 = vectors1.shape[0]
        chunk_size = max(1, rows1 // self.config.max_workers)
        
        # 预先归一化
        norm1 = np.linalg.norm(vectors1, axis=1, keepdims=True)
        norm2 = np.linalg.norm(vectors2, axis=1, keepdims=True)
        
        norm1 = np.where(norm1 == 0, 1, norm1)
        norm2 = np.where(norm2 == 0, 1, norm2)
        
        normalized1 = vectors1 / norm1
        normalized2 = vectors2 / norm2
        
        # 并行处理
        with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
            futures = []
            
            for i in range(0, rows1, chunk_size):
                end_i = min(i + chunk_size, rows1)
                chunk1 = normalized1[i:end_i]
                
                future = executor.submit(self._compute_similarity_chunk, chunk1, normalized2)
                futures.append((i, end_i, future))
            
            # 收集结果
            result = np.zeros((rows1, vectors2.shape[0]), dtype=self.config.precision)
            for i, end_i, future in futures:
                chunk_result = future.result()
                result[i:end_i] = chunk_result
        
        return result
    
    def _compute_similarity_chunk(self, chunk1: np.ndarray, normalized2: np.ndarray) -> np.ndarray:
        """计算相似度块（在线程中执行）"""
        return np.dot(chunk1, normalized2.T)
    
    def _gpu_cosine_similarity(self, vectors1: np.ndarray, vectors2: np.ndarray) -> np.ndarray:
        """GPU加速余弦相似度计算"""
        if not CUDA_AVAILABLE:
            return self._chunked_cosine_similarity(vectors1, vectors2)
        
        self.stats['gpu_computations'] += 1
        
        try:
            # 转移到GPU
            gpu_vectors1 = cp.asarray(vectors1)
            gpu_vectors2 = cp.asarray(vectors2)
            
            # GPU归一化
            norm1 = cp.linalg.norm(gpu_vectors1, axis=1, keepdims=True)
            norm2 = cp.linalg.norm(gpu_vectors2, axis=1, keepdims=True)
            
            norm1 = cp.where(norm1 == 0, 1, norm1)
            norm2 = cp.where(norm2 == 0, 1, norm2)
            
            normalized1 = gpu_vectors1 / norm1
            normalized2 = gpu_vectors2 / norm2
            
            # GPU计算
            gpu_result = cp.dot(normalized1, normalized2.T)
            
            # 转回CPU
            result = cp.asnumpy(gpu_result)
            
            return result
            
        except Exception as e:
            logger.warning(f"GPU计算失败，降级到CPU: {e}")
            return self._chunked_cosine_similarity(vectors1, vectors2)
    
    def _fallback_cosine_similarity(self, vectors1: np.ndarray, vectors2: np.ndarray) -> np.ndarray:
        """降级余弦相似度计算"""
        try:
            if SCIPY_AVAILABLE:
                from sklearn.metrics.pairwise import cosine_similarity
                return cosine_similarity(vectors1, vectors2)
            else:
                return self._direct_cosine_similarity(vectors1, vectors2)
        except ImportError:
            return self._direct_cosine_similarity(vectors1, vectors2)
    
    def _calculate_optimal_chunk_size(self, dtype: np.dtype, dim: int) -> int:
        """计算最优分块大小"""
        element_size = np.dtype(dtype).itemsize
        chunk_size_bytes = self.config.chunk_size_mb * 1024 * 1024
        
        # 估算每行所需内存（包括中间结果）
        memory_per_row = dim * element_size * 3  # 原始向量 + 归一化 + 中间结果
        
        chunk_size = max(1, chunk_size_bytes // memory_per_row)
        
        # 限制在合理范围内
        return min(chunk_size, 1000)
    
    def _update_stats(self, algorithm: str, computation_time: float):
        """更新统计信息"""
        self.stats['total_computations'] += 1
        
        if algorithm == 'gpu':
            self.stats['gpu_computations'] += 1
        else:
            self.stats['cpu_computations'] += 1
        
        if algorithm == 'chunked':
            self.stats['chunked_computations'] += 1
        elif algorithm == 'parallel':
            self.stats['parallel_computations'] += 1
        
        # 更新平均时间
        total_time = self.stats['avg_computation_time'] * (self.stats['total_computations'] - 1) + computation_time
        self.stats['avg_computation_time'] = total_time / self.stats['total_computations']
    
    def get_optimization_stats(self) -> Dict[str, Any]:
        """获取优化统计信息"""
        return {
            'config': {
                'enable_chunking': self.config.enable_chunking,
                'chunk_size_mb': self.config.chunk_size_mb,
                'enable_adaptive_algorithm': self.config.enable_adaptive_algorithm,
                'use_gpu': self.config.use_gpu,
                'cuda_available': CUDA_AVAILABLE,
                'precision': self.config.precision
            },
            'performance_stats': self.stats.copy(),
            'memory_pool': {
                'pool_size_mb': self.config.memory_pool_size_mb,
                'allocated_mb': self.memory_pool.total_allocated / (1024 * 1024),
                'free_arrays': len(self.memory_pool.free_memory)
            },
            'algorithm_distribution': {
                'gpu_ratio': self.stats['gpu_computations'] / max(1, self.stats['total_computations']),
                'chunked_ratio': self.stats['chunked_computations'] / max(1, self.stats['total_computations']),
                'parallel_ratio': self.stats['parallel_computations'] / max(1, self.stats['total_computations'])
            }
        }
    
    def cleanup(self):
        """清理资源"""
        if self.result_cache:
            self.result_cache.clear()
        
        # 清理内存池
        self.memory_pool.free_memory.clear()
        self.memory_pool.total_allocated = 0
        
        # 强制垃圾回收
        gc.collect()
        
        logger.info("优化向量化计算器资源清理完成")


def create_optimized_vectorized_calculator(
    config: Optional[OptimizedVectorizedConfig] = None
) -> OptimizedVectorizedCalculator:
    """创建优化版向量化计算器"""
    return OptimizedVectorizedCalculator(config)


# 全局优化实例
_global_optimized_calculator = None

def get_optimized_vectorized_calculator() -> OptimizedVectorizedCalculator:
    """获取全局优化向量化计算器"""
    global _global_optimized_calculator
    if _global_optimized_calculator is None:
        _global_optimized_calculator = create_optimized_vectorized_calculator()
    return _global_optimized_calculator