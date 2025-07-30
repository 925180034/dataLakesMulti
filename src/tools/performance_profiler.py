"""
性能分析器 - Phase 2优化工具
用于诊断性能瓶颈和指导优化策略
"""

import time
import logging
import psutil
import functools
from typing import Dict, Any, Optional, Callable, List
from dataclasses import dataclass, field
from contextlib import contextmanager
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetrics:
    """性能指标"""
    execution_time: float = 0.0
    memory_usage_mb: float = 0.0
    cpu_percent: float = 0.0
    io_operations: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    function_calls: int = 0
    peak_memory_mb: float = 0.0


@dataclass
class ComponentProfile:
    """组件性能分析"""
    name: str
    metrics: PerformanceMetrics = field(default_factory=PerformanceMetrics)
    sub_components: Dict[str, 'ComponentProfile'] = field(default_factory=dict)
    overhead_ratio: float = 0.0  # 开销比例
    efficiency_score: float = 0.0  # 效率分数


class PerformanceProfiler:
    """性能分析器"""
    
    def __init__(self):
        self.profiles: Dict[str, ComponentProfile] = {}
        self.active_profiles: List[str] = []
        self.baseline_metrics: Optional[PerformanceMetrics] = None
        
        # 系统基准测试
        self._establish_baseline()
        
        logger.info("性能分析器初始化完成")
    
    def _establish_baseline(self):
        """建立系统基准"""
        # 简单的计算基准测试
        start_time = time.time()
        start_memory = psutil.virtual_memory().used / (1024 * 1024)
        
        # 执行标准计算任务
        for _ in range(1000):
            np.random.rand(100, 100).dot(np.random.rand(100, 100))
        
        end_time = time.time()
        end_memory = psutil.virtual_memory().used / (1024 * 1024)
        
        self.baseline_metrics = PerformanceMetrics(
            execution_time=end_time - start_time,
            memory_usage_mb=end_memory - start_memory,
            cpu_percent=psutil.cpu_percent()
        )
        
        logger.info(f"系统基准: 计算={self.baseline_metrics.execution_time:.3f}s, "
                   f"内存={self.baseline_metrics.memory_usage_mb:.1f}MB")
    
    @contextmanager
    def profile_component(self, component_name: str):
        """组件性能分析上下文管理器"""
        if component_name not in self.profiles:
            self.profiles[component_name] = ComponentProfile(name=component_name)
        
        profile = self.profiles[component_name]
        self.active_profiles.append(component_name)
        
        # 记录开始状态
        start_time = time.time()
        start_memory = psutil.virtual_memory().used / (1024 * 1024)
        start_cpu = psutil.cpu_percent()
        
        try:
            yield profile
        finally:
            # 记录结束状态
            end_time = time.time()
            end_memory = psutil.virtual_memory().used / (1024 * 1024)
            end_cpu = psutil.cpu_percent()
            
            # 更新指标
            profile.metrics.execution_time += end_time - start_time
            profile.metrics.memory_usage_mb = max(profile.metrics.memory_usage_mb, 
                                                 end_memory - start_memory)
            profile.metrics.cpu_percent = (start_cpu + end_cpu) / 2
            profile.metrics.function_calls += 1
            profile.metrics.peak_memory_mb = max(profile.metrics.peak_memory_mb, end_memory)
            
            # 计算效率分数
            if self.baseline_metrics:
                time_ratio = profile.metrics.execution_time / self.baseline_metrics.execution_time
                memory_ratio = profile.metrics.memory_usage_mb / max(1, self.baseline_metrics.memory_usage_mb)
                profile.efficiency_score = 1.0 / (time_ratio * memory_ratio + 0.1)
            
            self.active_profiles.remove(component_name)
    
    def profile_function(self, component_name: str):
        """函数装饰器版本的性能分析"""
        def decorator(func: Callable) -> Callable:
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                with self.profile_component(f"{component_name}.{func.__name__}") as profile:
                    return func(*args, **kwargs)
            return wrapper
        return decorator
    
    def analyze_bottlenecks(self) -> Dict[str, Any]:
        """分析性能瓶颈"""
        bottlenecks = {
            'time_bottlenecks': [],
            'memory_bottlenecks': [],
            'inefficient_components': [],
            'optimization_recommendations': []
        }
        
        # 按执行时间排序
        time_sorted = sorted(self.profiles.items(), 
                           key=lambda x: x[1].metrics.execution_time, reverse=True)
        
        # 识别时间瓶颈（占总时间的20%以上）
        total_time = sum(p.metrics.execution_time for _, p in self.profiles.items())
        for name, profile in time_sorted[:3]:
            time_percentage = (profile.metrics.execution_time / total_time) * 100
            if time_percentage > 20:
                bottlenecks['time_bottlenecks'].append({
                    'component': name,
                    'time': profile.metrics.execution_time,
                    'percentage': time_percentage
                })
        
        # 识别内存瓶颈
        memory_sorted = sorted(self.profiles.items(),
                             key=lambda x: x[1].metrics.memory_usage_mb, reverse=True)
        
        for name, profile in memory_sorted[:3]:
            if profile.metrics.memory_usage_mb > 100:  # 超过100MB
                bottlenecks['memory_bottlenecks'].append({
                    'component': name,
                    'memory_mb': profile.metrics.memory_usage_mb,
                    'peak_memory_mb': profile.metrics.peak_memory_mb
                })
        
        # 识别低效组件
        for name, profile in self.profiles.items():
            if profile.efficiency_score < 0.5:  # 效率分数低于0.5
                bottlenecks['inefficient_components'].append({
                    'component': name,
                    'efficiency_score': profile.efficiency_score,
                    'function_calls': profile.metrics.function_calls
                })
        
        # 生成优化建议
        bottlenecks['optimization_recommendations'] = self._generate_recommendations(bottlenecks)
        
        return bottlenecks
    
    def _generate_recommendations(self, bottlenecks: Dict[str, Any]) -> List[str]:
        """生成优化建议"""
        recommendations = []
        
        # 时间瓶颈建议
        for bottleneck in bottlenecks['time_bottlenecks']:
            component = bottleneck['component']
            if 'vectorized' in component.lower():
                recommendations.append(f"优化{component}: 考虑分块处理或GPU加速")
            elif 'lsh' in component.lower():
                recommendations.append(f"优化{component}: 减少哈希函数数量或调整分桶策略")
            elif 'cache' in component.lower():
                recommendations.append(f"优化{component}: 调整缓存策略或增加L1缓存大小")
        
        # 内存瓶颈建议
        for bottleneck in bottlenecks['memory_bottlenecks']:
            component = bottleneck['component']
            recommendations.append(f"优化{component}: 实现内存流式处理或数据压缩")
        
        # 低效组件建议
        for bottleneck in bottlenecks['inefficient_components']:
            component = bottleneck['component']
            recommendations.append(f"重构{component}: 效率分数过低，需要算法优化")
        
        return recommendations
    
    def get_performance_report(self) -> Dict[str, Any]:
        """获取性能报告"""
        report = {
            'summary': {
                'total_components': len(self.profiles),
                'total_execution_time': sum(p.metrics.execution_time for p in self.profiles.values()),
                'total_memory_usage': sum(p.metrics.memory_usage_mb for p in self.profiles.values()),
                'avg_efficiency_score': np.mean([p.efficiency_score for p in self.profiles.values()])
            },
            'component_details': {},
            'bottleneck_analysis': self.analyze_bottlenecks(),
            'system_baseline': {
                'computation_time': self.baseline_metrics.execution_time if self.baseline_metrics else 0,
                'memory_overhead': self.baseline_metrics.memory_usage_mb if self.baseline_metrics else 0
            }
        }
        
        # 详细组件信息
        for name, profile in self.profiles.items():
            report['component_details'][name] = {
                'execution_time': profile.metrics.execution_time,
                'memory_usage_mb': profile.metrics.memory_usage_mb,
                'cpu_percent': profile.metrics.cpu_percent,
                'function_calls': profile.metrics.function_calls,
                'efficiency_score': profile.efficiency_score,
                'avg_time_per_call': profile.metrics.execution_time / max(1, profile.metrics.function_calls)
            }
        
        return report
    
    def reset_profiles(self):
        """重置所有性能分析数据"""
        self.profiles.clear()
        logger.info("性能分析数据已重置")
    
    def export_profile_data(self, file_path: str):
        """导出性能分析数据"""
        import json
        from pathlib import Path
        
        report = self.get_performance_report()
        
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        logger.info(f"性能分析数据已导出: {file_path}")


# 全局性能分析器实例
_global_profiler = None

def get_performance_profiler() -> PerformanceProfiler:
    """获取全局性能分析器"""
    global _global_profiler
    if _global_profiler is None:
        _global_profiler = PerformanceProfiler()
    return _global_profiler


def profile_component(component_name: str):
    """组件性能分析装饰器"""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            profiler = get_performance_profiler()
            with profiler.profile_component(f"{component_name}.{func.__name__}"):
                return await func(*args, **kwargs)
        
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            profiler = get_performance_profiler()
            with profiler.profile_component(f"{component_name}.{func.__name__}"):
                return func(*args, **kwargs)
        
        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
    return decorator


# 导入asyncio用于异步函数检测
import asyncio