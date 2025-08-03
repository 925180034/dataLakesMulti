"""
优化的向量嵌入生成器 - 批量和缓存优化
"""

import logging
import asyncio
import hashlib
import pickle
from pathlib import Path
from typing import List, Dict, Any, Optional
import numpy as np

logger = logging.getLogger(__name__)


class OptimizedEmbeddingGenerator:
    """优化的嵌入生成器
    
    核心优化：
    1. 批量生成嵌入
    2. 磁盘缓存
    3. 并行处理
    4. 预计算优化
    """
    
    def __init__(self, cache_dir: str = "./cache/embeddings"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.model = None
        self._model_initialized = False
        self.dimension = 384
        
        # 内存缓存
        self.memory_cache = {}
        self.cache_hits = 0
        self.cache_misses = 0
        
    def _initialize_model(self):
        """延迟初始化模型"""
        if not self._model_initialized:
            try:
                from sentence_transformers import SentenceTransformer
                logger.info("初始化SentenceTransformer模型...")
                self.model = SentenceTransformer('all-MiniLM-L6-v2')
                self._model_initialized = True
                logger.info("模型初始化完成")
            except Exception as e:
                logger.warning(f"无法初始化SentenceTransformer: {e}")
                self.model = None
                self._model_initialized = True
    
    def _get_cache_key(self, text: str) -> str:
        """生成缓存键"""
        return hashlib.md5(text.encode()).hexdigest()
    
    def _get_cache_path(self, cache_key: str) -> Path:
        """获取缓存文件路径"""
        return self.cache_dir / f"{cache_key}.pkl"
    
    def _load_from_cache(self, cache_key: str) -> Optional[np.ndarray]:
        """从缓存加载嵌入"""
        # 先检查内存缓存
        if cache_key in self.memory_cache:
            self.cache_hits += 1
            return self.memory_cache[cache_key]
        
        # 检查磁盘缓存
        cache_path = self._get_cache_path(cache_key)
        if cache_path.exists():
            try:
                with open(cache_path, 'rb') as f:
                    embedding = pickle.load(f)
                # 加载到内存缓存
                self.memory_cache[cache_key] = embedding
                self.cache_hits += 1
                return embedding
            except Exception as e:
                logger.warning(f"缓存加载失败: {e}")
        
        self.cache_misses += 1
        return None
    
    def _save_to_cache(self, cache_key: str, embedding: np.ndarray):
        """保存嵌入到缓存"""
        # 保存到内存缓存
        self.memory_cache[cache_key] = embedding
        
        # 保存到磁盘缓存
        cache_path = self._get_cache_path(cache_key)
        try:
            with open(cache_path, 'wb') as f:
                pickle.dump(embedding, f)
        except Exception as e:
            logger.warning(f"缓存保存失败: {e}")
    
    async def generate_batch_embeddings(
        self,
        texts: List[str],
        batch_size: int = 32,
        use_cache: bool = True
    ) -> List[np.ndarray]:
        """批量生成嵌入向量
        
        Args:
            texts: 文本列表
            batch_size: 批处理大小
            use_cache: 是否使用缓存
            
        Returns:
            嵌入向量列表
        """
        if not self._model_initialized:
            self._initialize_model()
        
        embeddings = []
        uncached_texts = []
        uncached_indices = []
        
        # 检查缓存
        for i, text in enumerate(texts):
            if use_cache:
                cache_key = self._get_cache_key(text)
                cached_embedding = self._load_from_cache(cache_key)
                if cached_embedding is not None:
                    embeddings.append(cached_embedding)
                else:
                    embeddings.append(None)
                    uncached_texts.append(text)
                    uncached_indices.append(i)
            else:
                embeddings.append(None)
                uncached_texts.append(text)
                uncached_indices.append(i)
        
        # 批量生成未缓存的嵌入
        if uncached_texts:
            if self.model is not None:
                # 使用真实模型批量生成
                logger.info(f"批量生成 {len(uncached_texts)} 个嵌入向量")
                
                # 分批处理
                for batch_start in range(0, len(uncached_texts), batch_size):
                    batch_end = min(batch_start + batch_size, len(uncached_texts))
                    batch_texts = uncached_texts[batch_start:batch_end]
                    
                    # 生成嵌入
                    batch_embeddings = self.model.encode(
                        batch_texts,
                        convert_to_numpy=True,
                        show_progress_bar=False
                    )
                    
                    # 保存到缓存并更新结果
                    for j, (text, embedding) in enumerate(zip(batch_texts, batch_embeddings)):
                        idx = uncached_indices[batch_start + j]
                        embeddings[idx] = embedding
                        
                        if use_cache:
                            cache_key = self._get_cache_key(text)
                            self._save_to_cache(cache_key, embedding)
            else:
                # 使用虚拟嵌入（离线模式）
                logger.info(f"使用虚拟嵌入（离线模式）: {len(uncached_texts)} 个")
                for i, text in zip(uncached_indices, uncached_texts):
                    embedding = self._generate_virtual_embedding(text)
                    embeddings[i] = embedding
                    
                    if use_cache:
                        cache_key = self._get_cache_key(text)
                        self._save_to_cache(cache_key, embedding)
        
        logger.info(f"嵌入生成完成 - 缓存命中率: {self.cache_hits}/{self.cache_hits + self.cache_misses}")
        
        return embeddings
    
    def _generate_virtual_embedding(self, text: str) -> np.ndarray:
        """生成虚拟嵌入（离线模式）"""
        # 基于文本生成确定性的虚拟向量
        np.random.seed(hash(text) % (2**32))
        embedding = np.random.randn(self.dimension)
        # 归一化
        embedding = embedding / np.linalg.norm(embedding)
        return embedding
    
    async def precompute_embeddings(
        self,
        texts: List[str],
        batch_size: int = 32,
        max_workers: int = 4
    ):
        """预计算并缓存所有嵌入"""
        logger.info(f"开始预计算 {len(texts)} 个文本的嵌入向量")
        
        # 分批并行处理
        tasks = []
        for i in range(0, len(texts), batch_size * max_workers):
            batch_texts = texts[i:i + batch_size * max_workers]
            task = self.generate_batch_embeddings(batch_texts, batch_size, use_cache=True)
            tasks.append(task)
        
        # 等待所有任务完成
        await asyncio.gather(*tasks)
        
        logger.info(f"预计算完成 - 缓存命中率: {self.cache_hits}/{self.cache_hits + self.cache_misses}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "cache_hit_rate": self.cache_hits / (self.cache_hits + self.cache_misses) 
                            if (self.cache_hits + self.cache_misses) > 0 else 0,
            "memory_cache_size": len(self.memory_cache),
            "disk_cache_size": len(list(self.cache_dir.glob("*.pkl")))
        }
    
    def clear_memory_cache(self):
        """清空内存缓存"""
        self.memory_cache.clear()
        logger.info("内存缓存已清空")
    
    def clear_disk_cache(self):
        """清空磁盘缓存"""
        for cache_file in self.cache_dir.glob("*.pkl"):
            cache_file.unlink()
        logger.info("磁盘缓存已清空")