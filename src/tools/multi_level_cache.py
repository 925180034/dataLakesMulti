"""
多级缓存系统 - Phase 2架构升级组件
L1内存缓存 + L2磁盘缓存 + 可选Redis缓存
"""

import json
import pickle
import hashlib
import logging
import time
from typing import Any, Optional, Dict, List, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path
from threading import Lock
import sqlite3
from src.config.settings import settings

logger = logging.getLogger(__name__)

# 尝试导入Redis支持
try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    redis = None
    REDIS_AVAILABLE = False


@dataclass
class CacheEntry:
    """缓存条目"""
    key: str
    value: Any
    timestamp: float
    ttl: int
    access_count: int = 0
    size_bytes: int = 0


class L1MemoryCache:
    """L1内存缓存 - LRU策略"""
    
    def __init__(self, max_size: int = 1000):
        self.max_size = max_size
        self.cache: Dict[str, CacheEntry] = {}
        self.access_order: List[str] = []  # LRU顺序
        self.lock = Lock()
        
        # 统计信息
        self.stats = {
            'hits': 0,
            'misses': 0,
            'evictions': 0,
            'total_requests': 0
        }
        
        logger.info(f"L1内存缓存初始化: 最大容量 {max_size}")
    
    def get(self, key: str) -> Optional[Any]:
        """获取缓存值"""
        with self.lock:
            self.stats['total_requests'] += 1
            
            if key in self.cache:
                entry = self.cache[key]
                
                # 检查TTL
                if time.time() - entry.timestamp > entry.ttl:
                    self._remove(key)
                    self.stats['misses'] += 1
                    return None
                
                # 更新访问顺序
                self._move_to_end(key)
                entry.access_count += 1
                
                self.stats['hits'] += 1
                logger.debug(f"L1缓存命中: {key}")
                return entry.value
            
            self.stats['misses'] += 1
            return None
    
    def put(self, key: str, value: Any, ttl: int = 3600):
        """存储缓存值"""
        with self.lock:
            # 计算值大小
            try:
                size_bytes = len(pickle.dumps(value))
            except:
                size_bytes = 1024  # 估算值
            
            # 如果已存在，更新
            if key in self.cache:
                self.cache[key] = CacheEntry(
                    key=key,
                    value=value,
                    timestamp=time.time(),
                    ttl=ttl,
                    size_bytes=size_bytes
                )
                self._move_to_end(key)
                logger.debug(f"L1缓存更新: {key}")
                return
            
            # 检查容量
            if len(self.cache) >= self.max_size:
                self._evict_lru()
            
            # 添加新条目
            self.cache[key] = CacheEntry(
                key=key,
                value=value,
                timestamp=time.time(),
                ttl=ttl,
                size_bytes=size_bytes
            )
            self.access_order.append(key)
            
            logger.debug(f"L1缓存存储: {key}")
    
    def _move_to_end(self, key: str):
        """移动到访问列表末尾"""
        if key in self.access_order:
            self.access_order.remove(key)
        self.access_order.append(key)
    
    def _evict_lru(self):
        """淘汰最少使用的条目"""
        if self.access_order:
            lru_key = self.access_order[0]
            self._remove(lru_key)
            self.stats['evictions'] += 1
            logger.debug(f"L1缓存淘汰: {lru_key}")
    
    def _remove(self, key: str):
        """移除条目"""
        if key in self.cache:
            del self.cache[key]
        if key in self.access_order:
            self.access_order.remove(key)
    
    def clear(self):
        """清空缓存"""
        with self.lock:
            self.cache.clear()
            self.access_order.clear()
            logger.info("L1缓存已清空")
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        hit_rate = 0
        if self.stats['total_requests'] > 0:
            hit_rate = self.stats['hits'] / self.stats['total_requests']
        
        return {
            'level': 'L1_Memory',
            'current_size': len(self.cache),
            'max_size': self.max_size,
            'hit_rate': hit_rate,
            'stats': self.stats.copy()
        }


class L2DiskCache:
    """L2磁盘缓存 - SQLite实现"""
    
    def __init__(self, cache_dir: str = "./cache/l2"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.db_path = self.cache_dir / "cache.db"
        self.lock = Lock()
        
        # 统计信息
        self.stats = {
            'hits': 0,
            'misses': 0,
            'writes': 0,
            'cleanups': 0,
            'total_requests': 0
        }
        
        self._init_database()
        logger.info(f"L2磁盘缓存初始化: {self.cache_dir}")
    
    def _init_database(self):
        """初始化数据库"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS cache_entries (
                    key TEXT PRIMARY KEY,
                    value BLOB,
                    timestamp REAL,
                    ttl INTEGER,
                    access_count INTEGER DEFAULT 0,
                    size_bytes INTEGER DEFAULT 0
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_timestamp 
                ON cache_entries(timestamp)
            """)
    
    def get(self, key: str) -> Optional[Any]:
        """获取缓存值"""
        with self.lock:
            self.stats['total_requests'] += 1
            
            try:
                with sqlite3.connect(self.db_path) as conn:
                    cursor = conn.execute(
                        "SELECT value, timestamp, ttl, access_count FROM cache_entries WHERE key = ?",
                        (key,)
                    )
                    row = cursor.fetchone()
                    
                    if row:
                        value_blob, timestamp, ttl, access_count = row
                        
                        # 检查TTL
                        if time.time() - timestamp > ttl:
                            conn.execute("DELETE FROM cache_entries WHERE key = ?", (key,))
                            self.stats['misses'] += 1
                            return None
                        
                        # 更新访问计数
                        conn.execute(
                            "UPDATE cache_entries SET access_count = ? WHERE key = ?",
                            (access_count + 1, key)
                        )
                        
                        # 反序列化值
                        value = pickle.loads(value_blob)
                        
                        self.stats['hits'] += 1
                        logger.debug(f"L2缓存命中: {key}")
                        return value
                
                self.stats['misses'] += 1
                return None
                
            except Exception as e:
                logger.error(f"L2缓存读取失败 {key}: {e}")
                self.stats['misses'] += 1
                return None
    
    def put(self, key: str, value: Any, ttl: int = 3600):
        """存储缓存值"""
        with self.lock:
            try:
                # 序列化值
                value_blob = pickle.dumps(value)
                size_bytes = len(value_blob)
                timestamp = time.time()
                
                with sqlite3.connect(self.db_path) as conn:
                    conn.execute("""
                        INSERT OR REPLACE INTO cache_entries 
                        (key, value, timestamp, ttl, access_count, size_bytes)
                        VALUES (?, ?, ?, ?, 0, ?)
                    """, (key, value_blob, timestamp, ttl, size_bytes))
                
                self.stats['writes'] += 1
                logger.debug(f"L2缓存存储: {key}")
                
            except Exception as e:
                logger.error(f"L2缓存写入失败 {key}: {e}")
    
    def cleanup_expired(self):
        """清理过期条目"""
        with self.lock:
            try:
                current_time = time.time()
                with sqlite3.connect(self.db_path) as conn:
                    cursor = conn.execute(
                        "DELETE FROM cache_entries WHERE (timestamp + ttl) < ?",
                        (current_time,)
                    )
                    deleted_count = cursor.rowcount
                
                if deleted_count > 0:
                    self.stats['cleanups'] += deleted_count
                    logger.info(f"L2缓存清理了 {deleted_count} 个过期条目")
                
            except Exception as e:
                logger.error(f"L2缓存清理失败: {e}")
    
    def clear(self):
        """清空缓存"""
        with self.lock:
            try:
                with sqlite3.connect(self.db_path) as conn:
                    conn.execute("DELETE FROM cache_entries")
                logger.info("L2缓存已清空")
            except Exception as e:
                logger.error(f"L2缓存清空失败: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        hit_rate = 0
        if self.stats['total_requests'] > 0:
            hit_rate = self.stats['hits'] / self.stats['total_requests']
        
        # 获取数据库统计
        db_stats = {'total_entries': 0, 'total_size_mb': 0}
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("SELECT COUNT(*), SUM(size_bytes) FROM cache_entries")
                row = cursor.fetchone()
                if row:
                    db_stats['total_entries'] = row[0] or 0
                    total_bytes = row[1] or 0
                    db_stats['total_size_mb'] = total_bytes / (1024 * 1024)
        except Exception as e:
            logger.error(f"获取L2缓存统计失败: {e}")
        
        return {
            'level': 'L2_Disk',
            'hit_rate': hit_rate,
            'stats': self.stats.copy(),
            'db_stats': db_stats
        }


class L3RedisCache:
    """L3 Redis缓存（可选）"""
    
    def __init__(self, redis_url: str = "redis://localhost:6379", prefix: str = "datalakes"):
        self.redis_url = redis_url
        self.prefix = prefix
        self.client = None
        self.available = False
        
        # 统计信息
        self.stats = {
            'hits': 0,
            'misses': 0,
            'writes': 0,
            'errors': 0,
            'total_requests': 0
        }
        
        if REDIS_AVAILABLE:
            self._connect()
        else:
            logger.warning("Redis不可用，L3缓存已禁用")
    
    def _connect(self):
        """连接Redis"""
        try:
            self.client = redis.from_url(self.redis_url, decode_responses=False)
            self.client.ping()
            self.available = True
            logger.info(f"L3 Redis缓存已连接: {self.redis_url}")
        except Exception as e:
            logger.warning(f"Redis连接失败: {e}")
            self.available = False
    
    def _make_key(self, key: str) -> str:
        """生成Redis键"""
        return f"{self.prefix}:{key}"
    
    def get(self, key: str) -> Optional[Any]:
        """获取缓存值"""
        if not self.available:
            return None
        
        self.stats['total_requests'] += 1
        
        try:
            redis_key = self._make_key(key)
            value_blob = self.client.get(redis_key)
            
            if value_blob:
                value = pickle.loads(value_blob)
                self.stats['hits'] += 1
                logger.debug(f"L3缓存命中: {key}")
                return value
            
            self.stats['misses'] += 1
            return None
            
        except Exception as e:
            logger.error(f"L3缓存读取失败 {key}: {e}")
            self.stats['errors'] += 1
            return None
    
    def put(self, key: str, value: Any, ttl: int = 3600):
        """存储缓存值"""
        if not self.available:
            return
        
        try:
            redis_key = self._make_key(key)
            value_blob = pickle.dumps(value)
            self.client.setex(redis_key, ttl, value_blob)
            
            self.stats['writes'] += 1
            logger.debug(f"L3缓存存储: {key}")
            
        except Exception as e:
            logger.error(f"L3缓存写入失败 {key}: {e}")
            self.stats['errors'] += 1
    
    def clear(self):
        """清空缓存"""
        if not self.available:
            return
        
        try:
            pattern = f"{self.prefix}:*"
            keys = self.client.keys(pattern)
            if keys:
                self.client.delete(*keys)
            logger.info("L3缓存已清空")
        except Exception as e:
            logger.error(f"L3缓存清空失败: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        hit_rate = 0
        if self.stats['total_requests'] > 0:
            hit_rate = self.stats['hits'] / self.stats['total_requests']
        
        return {
            'level': 'L3_Redis',
            'available': self.available,
            'hit_rate': hit_rate,
            'stats': self.stats.copy()
        }


class MultiLevelCache:
    """多级缓存管理器"""
    
    def __init__(self):
        # 从配置加载参数
        cache_config = settings.cache.multi_level_cache
        
        # 初始化各级缓存
        self.l1_cache = L1MemoryCache(max_size=cache_config.get('l1_memory_size', 1000))
        
        if cache_config.get('l3_disk_enabled', True):
            l3_path = cache_config.get('l3_disk_path', './cache/l3')
            self.l2_cache = L2DiskCache(cache_dir=l3_path)
        else:
            self.l2_cache = None
        
        if cache_config.get('l2_redis_enabled', False):
            redis_url = cache_config.get('l2_redis_url', 'redis://localhost:6379')
            self.l3_cache = L3RedisCache(redis_url=redis_url)
        else:
            self.l3_cache = None
        
        # 全局统计
        self.global_stats = {
            'total_requests': 0,
            'l1_hits': 0,
            'l2_hits': 0,
            'l3_hits': 0,
            'total_misses': 0
        }
        
        logger.info("多级缓存系统初始化完成")
    
    def get(self, key: str) -> Optional[Any]:
        """获取缓存值 - 从L1到L3依次查找"""
        self.global_stats['total_requests'] += 1
        
        # L1缓存查找
        value = self.l1_cache.get(key)
        if value is not None:
            self.global_stats['l1_hits'] += 1
            return value
        
        # L2缓存查找
        if self.l2_cache:
            value = self.l2_cache.get(key)
            if value is not None:
                self.global_stats['l2_hits'] += 1
                # 提升到L1缓存
                self.l1_cache.put(key, value)
                return value
        
        # L3缓存查找
        if self.l3_cache:
            value = self.l3_cache.get(key)
            if value is not None:
                self.global_stats['l3_hits'] += 1
                # 提升到L1和L2缓存
                self.l1_cache.put(key, value)
                if self.l2_cache:
                    self.l2_cache.put(key, value)
                return value
        
        self.global_stats['total_misses'] += 1
        return None
    
    def put(self, key: str, value: Any, ttl: int = 3600):
        """存储缓存值 - 同时写入所有级别"""
        # 写入L1缓存
        self.l1_cache.put(key, value, ttl)
        
        # 写入L2缓存
        if self.l2_cache:
            self.l2_cache.put(key, value, ttl)
        
        # 写入L3缓存
        if self.l3_cache:
            self.l3_cache.put(key, value, ttl)
    
    def generate_cache_key(self, prefix: str, *args) -> str:
        """生成缓存键"""
        # 创建唯一键
        key_data = f"{prefix}:" + ":".join(str(arg) for arg in args)
        
        # 如果键太长，使用哈希
        if len(key_data) > 200:
            key_hash = hashlib.md5(key_data.encode()).hexdigest()
            return f"{prefix}:{key_hash}"
        
        return key_data
    
    def clear_all(self):
        """清空所有级别缓存"""
        self.l1_cache.clear()
        
        if self.l2_cache:
            self.l2_cache.clear()
        
        if self.l3_cache:
            self.l3_cache.clear()
        
        logger.info("所有级别缓存已清空")
    
    def cleanup_expired(self):
        """清理过期条目"""
        if self.l2_cache:
            self.l2_cache.cleanup_expired()
    
    def get_comprehensive_stats(self) -> Dict[str, Any]:
        """获取综合统计信息"""
        stats = {
            'global_stats': self.global_stats.copy(),
            'l1_cache': self.l1_cache.get_stats()
        }
        
        if self.l2_cache:
            stats['l2_cache'] = self.l2_cache.get_stats()
        
        if self.l3_cache:
            stats['l3_cache'] = self.l3_cache.get_stats()
        
        # 计算总体命中率
        total_hits = self.global_stats['l1_hits'] + self.global_stats['l2_hits'] + self.global_stats['l3_hits']
        if self.global_stats['total_requests'] > 0:
            stats['overall_hit_rate'] = total_hits / self.global_stats['total_requests']
        else:
            stats['overall_hit_rate'] = 0
        
        return stats


# 全局多级缓存实例
_global_multi_cache = None

def get_multi_level_cache() -> MultiLevelCache:
    """获取全局多级缓存实例"""
    global _global_multi_cache
    if _global_multi_cache is None:
        _global_multi_cache = MultiLevelCache()
    return _global_multi_cache