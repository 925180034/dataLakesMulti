"""
多级缓存系统 - 支持三层加速架构
L1: 内存缓存（LRU）
L2: Redis缓存（可选）
L3: 磁盘缓存
"""

import logging
import json
import pickle
import hashlib
import time
from pathlib import Path
from typing import Any, Optional, Dict, List, Tuple
from collections import OrderedDict
import asyncio

logger = logging.getLogger(__name__)


class LRUCache:
    """LRU缓存实现"""
    
    def __init__(self, max_size: int = 1000):
        self.max_size = max_size
        self.cache = OrderedDict()
        self.hits = 0
        self.misses = 0
    
    def get(self, key: str) -> Optional[Any]:
        """获取缓存值"""
        if key in self.cache:
            # 移到最后（最近使用）
            self.cache.move_to_end(key)
            self.hits += 1
            return self.cache[key]
        
        self.misses += 1
        return None
    
    def set(self, key: str, value: Any) -> None:
        """设置缓存值"""
        if key in self.cache:
            # 更新并移到最后
            self.cache.move_to_end(key)
        else:
            # 新增
            if len(self.cache) >= self.max_size:
                # 移除最旧的
                self.cache.popitem(last=False)
        
        self.cache[key] = value
    
    def get_stats(self) -> Dict[str, Any]:
        """获取缓存统计"""
        total = self.hits + self.misses
        hit_rate = self.hits / total if total > 0 else 0
        
        return {
            "size": len(self.cache),
            "max_size": self.max_size,
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": hit_rate
        }


class MultiLevelCache:
    """多级缓存系统"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # L1: 内存缓存
        self.l1_cache = LRUCache(
            max_size=config.get("l1_memory_size", 1000)
        )
        
        # L2: Redis缓存（可选）
        self.l2_enabled = config.get("l2_redis_enabled", False)
        self.redis_client = None
        if self.l2_enabled:
            self._init_redis(config.get("l2_redis_url"))
        
        # L3: 磁盘缓存
        self.l3_enabled = config.get("l3_disk_enabled", True)
        self.l3_path = Path(config.get("l3_disk_path", "./cache/l3"))
        if self.l3_enabled:
            self.l3_path.mkdir(parents=True, exist_ok=True)
        
        # TTL设置
        self.default_ttl = config.get("default_ttl", 3600)  # 1小时
        
        # 统计信息
        self.stats = {
            "l1_requests": 0,
            "l2_requests": 0,
            "l3_requests": 0,
            "total_requests": 0
        }
    
    def _init_redis(self, redis_url: str) -> None:
        """初始化Redis连接"""
        try:
            import redis
            self.redis_client = redis.from_url(redis_url)
            self.redis_client.ping()
            logger.info("Redis缓存连接成功")
        except Exception as e:
            logger.warning(f"Redis连接失败，禁用L2缓存: {e}")
            self.l2_enabled = False
    
    def _generate_key(self, namespace: str, key: str) -> str:
        """生成缓存键"""
        return f"{namespace}:{key}"
    
    def _hash_key(self, key: str) -> str:
        """对长键进行哈希"""
        if len(key) > 100:
            return hashlib.md5(key.encode()).hexdigest()
        return key
    
    async def get(
        self,
        namespace: str,
        key: str,
        level: Optional[int] = None
    ) -> Optional[Any]:
        """从缓存获取值
        
        Args:
            namespace: 命名空间（如 "vector_search", "llm_results"）
            key: 缓存键
            level: 指定缓存级别（None表示自动查找）
        """
        self.stats["total_requests"] += 1
        cache_key = self._generate_key(namespace, self._hash_key(key))
        
        # L1查找
        if level is None or level == 1:
            self.stats["l1_requests"] += 1
            value = self.l1_cache.get(cache_key)
            if value is not None:
                return value
        
        # L2查找
        if self.l2_enabled and (level is None or level == 2):
            self.stats["l2_requests"] += 1
            value = await self._get_from_redis(cache_key)
            if value is not None:
                # 提升到L1
                self.l1_cache.set(cache_key, value)
                return value
        
        # L3查找
        if self.l3_enabled and (level is None or level == 3):
            self.stats["l3_requests"] += 1
            value = await self._get_from_disk(cache_key)
            if value is not None:
                # 提升到L1和L2
                self.l1_cache.set(cache_key, value)
                if self.l2_enabled:
                    await self._set_to_redis(cache_key, value)
                return value
        
        return None
    
    async def set(
        self,
        namespace: str,
        key: str,
        value: Any,
        ttl: Optional[int] = None,
        levels: Optional[List[int]] = None
    ) -> None:
        """设置缓存值
        
        Args:
            namespace: 命名空间
            key: 缓存键
            value: 缓存值
            ttl: 过期时间（秒）
            levels: 指定缓存级别列表（None表示所有级别）
        """
        cache_key = self._generate_key(namespace, self._hash_key(key))
        ttl = ttl or self.default_ttl
        
        if levels is None:
            levels = [1, 2, 3]
        
        # L1设置
        if 1 in levels:
            self.l1_cache.set(cache_key, value)
        
        # L2设置
        if 2 in levels and self.l2_enabled:
            await self._set_to_redis(cache_key, value, ttl)
        
        # L3设置
        if 3 in levels and self.l3_enabled:
            await self._set_to_disk(cache_key, value, ttl)
    
    async def _get_from_redis(self, key: str) -> Optional[Any]:
        """从Redis获取"""
        if not self.redis_client:
            return None
        
        try:
            data = self.redis_client.get(key)
            if data:
                return pickle.loads(data)
        except Exception as e:
            logger.error(f"Redis读取失败: {e}")
        
        return None
    
    async def _set_to_redis(self, key: str, value: Any, ttl: int) -> None:
        """设置到Redis"""
        if not self.redis_client:
            return
        
        try:
            data = pickle.dumps(value)
            self.redis_client.setex(key, ttl, data)
        except Exception as e:
            logger.error(f"Redis写入失败: {e}")
    
    async def _get_from_disk(self, key: str) -> Optional[Any]:
        """从磁盘获取"""
        try:
            # 使用键的前两个字符作为子目录
            subdir = key[:2] if len(key) >= 2 else "00"
            file_path = self.l3_path / subdir / f"{key}.pkl"
            
            if file_path.exists():
                # 检查过期时间
                mtime = file_path.stat().st_mtime
                if time.time() - mtime > self.default_ttl:
                    file_path.unlink()  # 删除过期文件
                    return None
                
                with open(file_path, 'rb') as f:
                    return pickle.load(f)
        except Exception as e:
            logger.error(f"磁盘读取失败: {e}")
        
        return None
    
    async def _set_to_disk(self, key: str, value: Any, ttl: int) -> None:
        """设置到磁盘"""
        try:
            # 使用键的前两个字符作为子目录
            subdir = key[:2] if len(key) >= 2 else "00"
            dir_path = self.l3_path / subdir
            dir_path.mkdir(exist_ok=True)
            
            file_path = dir_path / f"{key}.pkl"
            
            with open(file_path, 'wb') as f:
                pickle.dump(value, f)
        except Exception as e:
            logger.error(f"磁盘写入失败: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """获取缓存统计信息"""
        stats = {
            "total_requests": self.stats["total_requests"],
            "l1": {
                **self.l1_cache.get_stats(),
                "requests": self.stats["l1_requests"]
            },
            "l2": {
                "enabled": self.l2_enabled,
                "requests": self.stats["l2_requests"]
            },
            "l3": {
                "enabled": self.l3_enabled,
                "requests": self.stats["l3_requests"]
            }
        }
        
        # 计算整体命中率
        if self.stats["total_requests"] > 0:
            l1_hit_rate = self.l1_cache.hits / self.stats["l1_requests"] if self.stats["l1_requests"] > 0 else 0
            stats["overall_hit_rate"] = l1_hit_rate  # 简化计算
        
        return stats
    
    async def clear(self, namespace: Optional[str] = None) -> None:
        """清除缓存
        
        Args:
            namespace: 只清除特定命名空间（None表示清除所有）
        """
        if namespace:
            # 清除特定命名空间
            # 这需要更复杂的实现来跟踪命名空间
            logger.warning("命名空间清除功能暂未实现")
        else:
            # 清除所有缓存
            self.l1_cache = LRUCache(self.config.get("l1_memory_size", 1000))
            
            if self.l2_enabled and self.redis_client:
                try:
                    self.redis_client.flushdb()
                except Exception as e:
                    logger.error(f"清除Redis缓存失败: {e}")
            
            if self.l3_enabled:
                try:
                    import shutil
                    shutil.rmtree(self.l3_path)
                    self.l3_path.mkdir(parents=True, exist_ok=True)
                except Exception as e:
                    logger.error(f"清除磁盘缓存失败: {e}")
    
    async def warm_up(
        self,
        namespace: str,
        keys: List[str],
        generator_func
    ) -> None:
        """预热缓存
        
        Args:
            namespace: 命名空间
            keys: 需要预热的键列表
            generator_func: 生成缓存值的函数
        """
        logger.info(f"开始预热缓存 {namespace}，键数量: {len(keys)}")
        
        # 批量生成和缓存
        tasks = []
        for key in keys:
            # 检查是否已存在
            if await self.get(namespace, key) is None:
                tasks.append(self._warm_up_key(namespace, key, generator_func))
        
        # 并行执行预热
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        
        logger.info(f"缓存预热完成 {namespace}")
    
    async def _warm_up_key(self, namespace: str, key: str, generator_func) -> None:
        """预热单个键"""
        try:
            value = await generator_func(key)
            if value is not None:
                await self.set(namespace, key, value)
        except Exception as e:
            logger.error(f"预热键失败 {key}: {e}")


class CacheManager:
    """缓存管理器 - 提供便捷的缓存接口"""
    
    def __init__(self, cache_config: Dict[str, Any]):
        self.cache = MultiLevelCache(cache_config)
        
    async def get_or_compute(
        self,
        namespace: str,
        key: str,
        compute_func,
        ttl: Optional[int] = None
    ) -> Any:
        """获取或计算并缓存
        
        Args:
            namespace: 缓存命名空间
            key: 缓存键
            compute_func: 计算函数（当缓存未命中时调用）
            ttl: 缓存过期时间
        """
        # 尝试从缓存获取
        value = await self.cache.get(namespace, key)
        if value is not None:
            return value
        
        # 计算新值
        value = await compute_func()
        
        # 缓存结果
        if value is not None:
            await self.cache.set(namespace, key, value, ttl)
        
        return value
    
    async def batch_get_or_compute(
        self,
        namespace: str,
        keys: List[str],
        compute_func,
        ttl: Optional[int] = None
    ) -> Dict[str, Any]:
        """批量获取或计算
        
        Args:
            namespace: 缓存命名空间
            keys: 缓存键列表
            compute_func: 批量计算函数（接收未命中的键列表）
            ttl: 缓存过期时间
        """
        results = {}
        uncached_keys = []
        
        # 检查缓存
        for key in keys:
            value = await self.cache.get(namespace, key)
            if value is not None:
                results[key] = value
            else:
                uncached_keys.append(key)
        
        # 批量计算未缓存的
        if uncached_keys:
            computed_values = await compute_func(uncached_keys)
            
            # 缓存新计算的值
            for key, value in computed_values.items():
                if value is not None:
                    await self.cache.set(namespace, key, value, ttl)
                    results[key] = value
        
        return results