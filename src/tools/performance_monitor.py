"""
性能监控工具 - 实时监控三层架构的性能表现
"""

import logging
import time
import asyncio
from typing import Dict, Any, List, Optional
from collections import defaultdict, deque
from datetime import datetime
import json

logger = logging.getLogger(__name__)


class PerformanceMonitor:
    """性能监控器
    
    监控指标：
    1. 响应时间（各层耗时）
    2. 吞吐量（QPS）
    3. 资源使用（内存、缓存）
    4. 成功率和错误率
    """
    
    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        
        # 响应时间记录
        self.response_times = defaultdict(lambda: deque(maxlen=window_size))
        
        # 层级耗时记录
        self.layer_times = {
            "metadata_filter": deque(maxlen=window_size),
            "vector_search": deque(maxlen=window_size),
            "llm_match": deque(maxlen=window_size)
        }
        
        # 查询计数
        self.query_count = 0
        self.success_count = 0
        self.error_count = 0
        
        # 缓存命中统计
        self.cache_hits = 0
        self.cache_misses = 0
        
        # LLM调用统计
        self.llm_calls = 0
        self.llm_tokens_used = 0
        
        # 开始时间
        self.start_time = time.time()
        
        # 慢查询日志
        self.slow_queries = deque(maxlen=50)
        self.slow_query_threshold = 10.0  # 10秒
        
    def record_query(
        self,
        query_id: str,
        total_time: float,
        layer_times: Dict[str, float],
        success: bool = True,
        error_msg: Optional[str] = None
    ) -> None:
        """记录查询性能数据"""
        self.query_count += 1
        
        if success:
            self.success_count += 1
        else:
            self.error_count += 1
        
        # 记录总响应时间
        self.response_times["total"].append(total_time)
        
        # 记录各层耗时
        for layer, layer_time in layer_times.items():
            if layer in self.layer_times:
                self.layer_times[layer].append(layer_time)
        
        # 检查慢查询
        if total_time > self.slow_query_threshold:
            self.slow_queries.append({
                "query_id": query_id,
                "total_time": total_time,
                "layer_times": layer_times,
                "timestamp": datetime.now().isoformat(),
                "error": error_msg
            })
    
    def record_cache_access(self, hit: bool) -> None:
        """记录缓存访问"""
        if hit:
            self.cache_hits += 1
        else:
            self.cache_misses += 1
    
    def record_llm_call(self, tokens: int = 0) -> None:
        """记录LLM调用"""
        self.llm_calls += 1
        self.llm_tokens_used += tokens
    
    def get_metrics(self) -> Dict[str, Any]:
        """获取当前性能指标"""
        # 计算运行时间
        runtime = time.time() - self.start_time
        
        # 计算QPS
        qps = self.query_count / runtime if runtime > 0 else 0
        
        # 计算成功率
        success_rate = self.success_count / self.query_count if self.query_count > 0 else 0
        
        # 计算缓存命中率
        total_cache_access = self.cache_hits + self.cache_misses
        cache_hit_rate = self.cache_hits / total_cache_access if total_cache_access > 0 else 0
        
        # 计算平均响应时间
        avg_response_time = (
            sum(self.response_times["total"]) / len(self.response_times["total"])
            if self.response_times["total"] else 0
        )
        
        # 计算各层平均耗时
        layer_avg_times = {}
        for layer, times in self.layer_times.items():
            if times:
                layer_avg_times[layer] = sum(times) / len(times)
            else:
                layer_avg_times[layer] = 0
        
        # 计算P50, P90, P99
        percentiles = self._calculate_percentiles(list(self.response_times["total"]))
        
        return {
            "summary": {
                "total_queries": self.query_count,
                "success_rate": f"{success_rate:.2%}",
                "qps": f"{qps:.2f}",
                "avg_response_time": f"{avg_response_time:.2f}s",
                "cache_hit_rate": f"{cache_hit_rate:.2%}",
                "runtime": f"{runtime:.0f}s"
            },
            "response_times": {
                "average": f"{avg_response_time:.2f}s",
                "p50": f"{percentiles['p50']:.2f}s",
                "p90": f"{percentiles['p90']:.2f}s",
                "p99": f"{percentiles['p99']:.2f}s",
                "max": f"{max(self.response_times['total']) if self.response_times['total'] else 0:.2f}s"
            },
            "layer_performance": {
                layer: f"{avg_time:.2f}s" 
                for layer, avg_time in layer_avg_times.items()
            },
            "llm_stats": {
                "total_calls": self.llm_calls,
                "avg_calls_per_query": f"{self.llm_calls / self.query_count if self.query_count > 0 else 0:.1f}",
                "total_tokens": self.llm_tokens_used
            },
            "slow_queries": len(self.slow_queries),
            "errors": self.error_count
        }
    
    def _calculate_percentiles(self, values: List[float]) -> Dict[str, float]:
        """计算百分位数"""
        if not values:
            return {"p50": 0, "p90": 0, "p99": 0}
        
        sorted_values = sorted(values)
        n = len(sorted_values)
        
        return {
            "p50": sorted_values[int(n * 0.5)],
            "p90": sorted_values[int(n * 0.9)],
            "p99": sorted_values[int(n * 0.99)] if n >= 100 else sorted_values[-1]
        }
    
    def get_slow_queries(self) -> List[Dict[str, Any]]:
        """获取慢查询列表"""
        return list(self.slow_queries)
    
    def export_metrics(self, filepath: str) -> None:
        """导出性能指标到文件"""
        metrics = self.get_metrics()
        metrics["slow_queries_detail"] = self.get_slow_queries()
        metrics["timestamp"] = datetime.now().isoformat()
        
        with open(filepath, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        logger.info(f"性能指标已导出到: {filepath}")
    
    def reset(self) -> None:
        """重置所有统计"""
        self.__init__(self.window_size)


class RealtimeMonitor:
    """实时性能监控器（支持异步更新）"""
    
    def __init__(self, update_interval: float = 1.0):
        self.monitor = PerformanceMonitor()
        self.update_interval = update_interval
        self.running = False
        self._monitor_task = None
    
    async def start(self) -> None:
        """启动实时监控"""
        self.running = True
        self._monitor_task = asyncio.create_task(self._monitor_loop())
        logger.info("实时性能监控已启动")
    
    async def stop(self) -> None:
        """停止监控"""
        self.running = False
        if self._monitor_task:
            await self._monitor_task
        logger.info("实时性能监控已停止")
    
    async def _monitor_loop(self) -> None:
        """监控循环"""
        while self.running:
            try:
                # 获取并显示指标
                metrics = self.monitor.get_metrics()
                self._display_metrics(metrics)
                
                # 检查告警
                self._check_alerts(metrics)
                
                await asyncio.sleep(self.update_interval)
                
            except Exception as e:
                logger.error(f"监控循环错误: {e}")
    
    def _display_metrics(self, metrics: Dict[str, Any]) -> None:
        """显示性能指标（控制台输出）"""
        # 清屏（可选）
        # print("\033[H\033[J")
        
        print("\n" + "="*60)
        print("📊 实时性能监控")
        print("="*60)
        
        summary = metrics["summary"]
        print(f"总查询数: {summary['total_queries']} | "
              f"成功率: {summary['success_rate']} | "
              f"QPS: {summary['qps']}")
        print(f"平均响应: {summary['avg_response_time']} | "
              f"缓存命中率: {summary['cache_hit_rate']}")
        
        print("\n📈 响应时间分布:")
        rt = metrics["response_times"]
        print(f"平均: {rt['average']} | P50: {rt['p50']} | "
              f"P90: {rt['p90']} | P99: {rt['p99']}")
        
        print("\n⚡ 各层性能:")
        for layer, time in metrics["layer_performance"].items():
            print(f"  {layer}: {time}")
        
        print("\n🤖 LLM统计:")
        llm = metrics["llm_stats"]
        print(f"总调用: {llm['total_calls']} | "
              f"平均每查询: {llm['avg_calls_per_query']}")
        
        if metrics["slow_queries"] > 0:
            print(f"\n⚠️  慢查询: {metrics['slow_queries']}")
        
        if metrics["errors"] > 0:
            print(f"\n❌ 错误数: {metrics['errors']}")
    
    def _check_alerts(self, metrics: Dict[str, Any]) -> None:
        """检查并触发告警"""
        # 成功率告警
        success_rate = float(metrics["summary"]["success_rate"].rstrip('%')) / 100
        if success_rate < 0.95:
            logger.warning(f"⚠️ 成功率低于95%: {metrics['summary']['success_rate']}")
        
        # 响应时间告警
        avg_time = float(metrics["response_times"]["average"].rstrip('s'))
        if avg_time > 10:
            logger.warning(f"⚠️ 平均响应时间超过10秒: {avg_time}s")
        
        # 缓存命中率告警
        cache_hit_rate = float(metrics["summary"]["cache_hit_rate"].rstrip('%')) / 100
        if cache_hit_rate < 0.3:
            logger.warning(f"⚠️ 缓存命中率低于30%: {metrics['summary']['cache_hit_rate']}")
    
    def record_query(self, *args, **kwargs) -> None:
        """代理到底层监控器"""
        self.monitor.record_query(*args, **kwargs)
    
    def record_cache_access(self, *args, **kwargs) -> None:
        """代理到底层监控器"""
        self.monitor.record_cache_access(*args, **kwargs)
    
    def record_llm_call(self, *args, **kwargs) -> None:
        """代理到底层监控器"""
        self.monitor.record_llm_call(*args, **kwargs)


# 全局监控实例
_global_monitor = None


def get_monitor() -> PerformanceMonitor:
    """获取全局监控实例"""
    global _global_monitor
    if _global_monitor is None:
        _global_monitor = PerformanceMonitor()
    return _global_monitor


def reset_monitor() -> None:
    """重置全局监控器"""
    global _global_monitor
    if _global_monitor:
        _global_monitor.reset()


# 便捷装饰器
def monitor_performance(layer_name: str):
    """性能监控装饰器"""
    def decorator(func):
        async def wrapper(*args, **kwargs):
            monitor = get_monitor()
            start_time = time.time()
            
            try:
                result = await func(*args, **kwargs)
                elapsed = time.time() - start_time
                
                # 记录层级性能
                monitor.layer_times[layer_name].append(elapsed)
                
                return result
                
            except Exception as e:
                elapsed = time.time() - start_time
                monitor.record_query(
                    query_id=f"{layer_name}_{int(start_time)}",
                    total_time=elapsed,
                    layer_times={layer_name: elapsed},
                    success=False,
                    error_msg=str(e)
                )
                raise
        
        return wrapper
    return decorator