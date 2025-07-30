"""
自适应配置系统 - Phase 2智能优化
根据系统负载和性能指标自动调整配置参数
"""

import logging
import time
import json
import threading
from typing import Dict, Any, Optional, List, Tuple, Callable
from dataclasses import dataclass, asdict
from pathlib import Path
import numpy as np
import psutil
from collections import deque, defaultdict
from src.config.settings import settings
from src.tools.performance_profiler import get_performance_profiler

logger = logging.getLogger(__name__)


@dataclass
class SystemMetrics:
    """系统性能指标"""
    timestamp: float
    cpu_percent: float
    memory_percent: float
    memory_available_mb: float
    query_throughput: float
    avg_response_time: float
    cache_hit_rate: float
    error_rate: float


@dataclass
class ConfigurationProfile:
    """配置档案"""
    name: str
    description: str
    config_values: Dict[str, Any]
    performance_score: float = 0.0
    usage_count: int = 0
    last_used: float = 0.0
    system_conditions: Dict[str, Any] = None


class AdaptiveConfigurationEngine:
    """自适应配置引擎"""
    
    def __init__(self, config_file: str = "./data/adaptive_configs.json"):
        self.config_file = Path(config_file)
        self.config_file.parent.mkdir(parents=True, exist_ok=True)
        
        # 性能指标历史
        self.metrics_history = deque(maxlen=1000)  # 保留最近1000个指标
        self.performance_baselines = {}
        
        # 配置档案
        self.configuration_profiles: Dict[str, ConfigurationProfile] = {}
        self.current_profile = None
        
        # 学习参数
        self.learning_rate = 0.1
        self.exploration_rate = 0.1  # 探索新配置的概率
        self.adaptation_interval = 60  # 配置调整间隔(秒)
        self.last_adaptation_time = time.time()
        
        # 参数调整范围
        self.parameter_ranges = {
            'lsh_prefilter': {
                'num_hash_functions': (32, 128),
                'num_hash_tables': (4, 16),
                'similarity_threshold': (0.3, 0.8)
            },
            'vectorized_optimizer': {
                'batch_size': (100, 2000),
                'chunk_size_mb': (32, 256),
                'max_workers': (2, 8)
            },
            'multi_cache': {
                'l1_memory_size': (500, 5000),
                'ttl_seconds': (1800, 7200)
            },
            'pipeline': {
                'fast_path_threshold': (20, 100),
                'lsh_activation_threshold': (100, 500),
                'vectorized_activation_threshold': (50, 200)
            }
        }
        
        # 监控线程
        self.monitoring_thread = None
        self.monitoring_active = False
        
        # 加载已保存的配置
        self._load_configurations()
        
        # 初始化默认档案
        self._initialize_default_profiles()
        
        logger.info("自适应配置引擎初始化完成")
    
    def start_monitoring(self):
        """启动性能监控"""
        if self.monitoring_active:
            return
        
        self.monitoring_active = True
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitoring_thread.start()
        logger.info("自适应配置监控已启动")
    
    def stop_monitoring(self):
        """停止性能监控"""
        self.monitoring_active = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5)
        logger.info("自适应配置监控已停止")
    
    def _monitoring_loop(self):
        """监控循环"""
        while self.monitoring_active:
            try:
                # 收集系统指标
                metrics = self._collect_system_metrics()
                self.metrics_history.append(metrics)
                
                # 检查是否需要调整配置
                if time.time() - self.last_adaptation_time > self.adaptation_interval:
                    self._consider_configuration_adjustment()
                    self.last_adaptation_time = time.time()
                
                time.sleep(10)  # 每10秒收集一次指标
                
            except Exception as e:
                logger.error(f"监控循环异常: {e}")
                time.sleep(30)  # 异常时延长间隔
    
    def _collect_system_metrics(self) -> SystemMetrics:
        """收集系统性能指标"""
        # 基础系统指标
        cpu_percent = psutil.cpu_percent()
        memory = psutil.virtual_memory()
        memory_percent = memory.percent
        memory_available_mb = memory.available / (1024 * 1024)
        
        # 应用性能指标
        profiler = get_performance_profiler()
        performance_report = profiler.get_performance_report()
        
        # 计算查询吞吐量
        query_throughput = 0.0
        avg_response_time = 0.0
        cache_hit_rate = 0.0
        error_rate = 0.0
        
        if self.metrics_history:
            recent_metrics = list(self.metrics_history)[-10:]  # 最近10个指标
            if len(recent_metrics) > 1:
                time_span = recent_metrics[-1].timestamp - recent_metrics[0].timestamp
                if time_span > 0:
                    # 简化的吞吐量计算
                    query_count = len(performance_report.get('component_details', {}))
                    query_throughput = query_count / time_span
        
        # 从性能报告中提取指标
        if 'summary' in performance_report:
            summary = performance_report['summary']
            avg_response_time = summary.get('total_execution_time', 0) / max(1, summary.get('total_components', 1))
        
        return SystemMetrics(
            timestamp=time.time(),
            cpu_percent=cpu_percent,
            memory_percent=memory_percent,
            memory_available_mb=memory_available_mb,
            query_throughput=query_throughput,
            avg_response_time=avg_response_time,
            cache_hit_rate=cache_hit_rate,
            error_rate=error_rate
        )
    
    def _consider_configuration_adjustment(self):
        """考虑配置调整"""
        if len(self.metrics_history) < 10:
            return  # 数据不足
        
        # 分析最近的性能趋势
        recent_metrics = list(self.metrics_history)[-10:]
        performance_trend = self._analyze_performance_trend(recent_metrics)
        
        # 根据性能趋势决定调整策略
        if performance_trend['needs_optimization']:
            logger.info(f"检测到性能下降: {performance_trend['reason']}")
            self._adjust_configuration(performance_trend)
    
    def _analyze_performance_trend(self, metrics: List[SystemMetrics]) -> Dict[str, Any]:
        """分析性能趋势"""
        if len(metrics) < 5:
            return {'needs_optimization': False, 'reason': '数据不足'}
        
        # 计算趋势指标
        cpu_trend = np.polyfit(range(len(metrics)), [m.cpu_percent for m in metrics], 1)[0]
        memory_trend = np.polyfit(range(len(metrics)), [m.memory_percent for m in metrics], 1)[0]
        response_time_trend = np.polyfit(range(len(metrics)), [m.avg_response_time for m in metrics], 1)[0]
        
        # 当前性能状态
        current_cpu = metrics[-1].cpu_percent
        current_memory = metrics[-1].memory_percent
        current_response_time = metrics[-1].avg_response_time
        
        # 判断是否需要优化
        needs_optimization = False
        reasons = []
        
        if current_cpu > 80:
            needs_optimization = True
            reasons.append("CPU使用率过高")
        
        if current_memory > 85:
            needs_optimization = True
            reasons.append("内存使用率过高")
        
        if cpu_trend > 2:  # CPU使用率快速上升
            needs_optimization = True
            reasons.append("CPU使用率上升趋势明显")
        
        if response_time_trend > 0.1:  # 响应时间快速增长
            needs_optimization = True
            reasons.append("响应时间上升趋势明显")
        
        return {
            'needs_optimization': needs_optimization,
            'reason': '; '.join(reasons),
            'cpu_trend': cpu_trend,
            'memory_trend': memory_trend,
            'response_time_trend': response_time_trend,
            'current_state': {
                'cpu': current_cpu,
                'memory': current_memory,
                'response_time': current_response_time
            }
        }
    
    def _adjust_configuration(self, performance_trend: Dict[str, Any]):
        """调整配置"""
        current_state = performance_trend['current_state']
        
        # 根据当前状态选择调整策略
        if current_state['cpu'] > 80:
            self._optimize_for_cpu_reduction()
        elif current_state['memory'] > 85:
            self._optimize_for_memory_reduction()
        elif current_state['response_time'] > 1.0:
            self._optimize_for_response_time()
        else:
            self._general_performance_optimization()
    
    def _optimize_for_cpu_reduction(self):
        """优化CPU使用"""
        logger.info("执行CPU优化配置调整")
        
        adjustments = {
            'vectorized_optimizer.max_workers': min(4, psutil.cpu_count() // 2),
            'pipeline.parallel_activation_threshold': 100,  # 提高并行阈值
            'lsh_prefilter.num_hash_functions': 32  # 减少哈希函数数量
        }
        
        self._apply_configuration_adjustments(adjustments, "cpu_optimization")
    
    def _optimize_for_memory_reduction(self):
        """优化内存使用"""
        logger.info("执行内存优化配置调整")
        
        adjustments = {
            'vectorized_optimizer.chunk_size_mb': 32,  # 减少分块大小
            'multi_cache.l1_memory_size': 500,  # 减少L1缓存大小
            'vectorized_optimizer.batch_size': 500  # 减少批处理大小
        }
        
        self._apply_configuration_adjustments(adjustments, "memory_optimization")
    
    def _optimize_for_response_time(self):
        """优化响应时间"""
        logger.info("执行响应时间优化配置调整")
        
        adjustments = {
            'pipeline.fast_path_threshold': 100,  # 增加快速路径使用
            'lsh_prefilter.similarity_threshold': 0.4,  # 降低LSH阈值
            'vectorized_optimizer.use_gpu': True  # 启用GPU（如果可用）
        }
        
        self._apply_configuration_adjustments(adjustments, "response_time_optimization")
    
    def _general_performance_optimization(self):
        """通用性能优化"""
        logger.info("执行通用性能优化配置调整")
        
        # 基于历史最佳配置进行微调
        best_profile = self._get_best_performing_profile()
        if best_profile:
            self._apply_configuration_profile(best_profile)
        else:
            # 探索性调整
            self._explore_new_configuration()
    
    def _apply_configuration_adjustments(self, adjustments: Dict[str, Any], profile_name: str):
        """应用配置调整"""
        try:
            # 记录当前配置作为新档案
            current_config = self._get_current_configuration()
            
            # 应用调整
            for key, value in adjustments.items():
                self._set_configuration_value(key, value)
            
            # 创建新的配置档案
            new_profile = ConfigurationProfile(
                name=profile_name,
                description=f"自动调整配置 - {profile_name}",
                config_values=adjustments,
                performance_score=0.0,  # 将在使用后评估
                last_used=time.time(),
                system_conditions=self._get_current_system_conditions()
            )
            
            self.configuration_profiles[profile_name] = new_profile
            self.current_profile = profile_name
            
            logger.info(f"配置调整已应用: {profile_name}")
            
        except Exception as e:
            logger.error(f"配置调整失败: {e}")
    
    def _get_current_configuration(self) -> Dict[str, Any]:
        """获取当前配置"""
        # 这里应该从settings对象中读取当前配置
        # 简化实现，返回关键配置项
        return {
            'lsh_prefilter.num_hash_functions': settings.lsh_prefilter.num_hash_functions,
            'vectorized_optimizer.batch_size': settings.vectorized_optimizer.batch_size,
            'multi_cache.l1_memory_size': settings.cache.multi_level_cache.get('l1_memory_size', 1000)
        }
    
    def _set_configuration_value(self, key: str, value: Any):
        """设置配置值"""
        # 解析配置键路径
        parts = key.split('.')
        if len(parts) != 2:
            logger.warning(f"无效的配置键: {key}")
            return
        
        component, param = parts
        
        # 应用配置（这里需要根据实际的配置对象结构来实现）
        try:
            if component == 'lsh_prefilter' and hasattr(settings, 'lsh_prefilter'):
                setattr(settings.lsh_prefilter, param, value)
            elif component == 'vectorized_optimizer' and hasattr(settings, 'vectorized_optimizer'):
                setattr(settings.vectorized_optimizer, param, value)
            elif component == 'multi_cache' and hasattr(settings, 'cache'):
                if param in settings.cache.multi_level_cache:
                    settings.cache.multi_level_cache[param] = value
            
            logger.debug(f"配置已更新: {key} = {value}")
            
        except Exception as e:
            logger.error(f"配置更新失败 {key}: {e}")
    
    def _get_current_system_conditions(self) -> Dict[str, Any]:
        """获取当前系统条件"""
        if not self.metrics_history:
            return {}
        
        latest_metrics = self.metrics_history[-1]
        return {
            'cpu_percent': latest_metrics.cpu_percent,
            'memory_percent': latest_metrics.memory_percent,
            'memory_available_mb': latest_metrics.memory_available_mb,
            'timestamp': latest_metrics.timestamp
        }
    
    def _get_best_performing_profile(self) -> Optional[ConfigurationProfile]:
        """获取表现最佳的配置档案"""
        if not self.configuration_profiles:
            return None
        
        # 按性能分数排序
        sorted_profiles = sorted(
            self.configuration_profiles.values(),
            key=lambda p: p.performance_score,
            reverse=True
        )
        
        return sorted_profiles[0] if sorted_profiles else None
    
    def _explore_new_configuration(self):
        """探索新配置"""
        logger.info("探索新的配置组合")
        
        # 随机调整一些参数（在安全范围内）
        exploration_adjustments = {}
        
        # 随机选择要调整的参数
        all_params = []
        for component, params in self.parameter_ranges.items():
            for param, (min_val, max_val) in params.items():
                all_params.append((component, param, min_val, max_val))
        
        # 随机选择2-3个参数进行调整
        import random
        selected_params = random.sample(all_params, min(3, len(all_params)))
        
        for component, param, min_val, max_val in selected_params:
            # 在范围内随机选择值
            if isinstance(min_val, int) and isinstance(max_val, int):
                new_value = random.randint(min_val, max_val)
            else:
                new_value = random.uniform(min_val, max_val)
            
            key = f"{component}.{param}"
            exploration_adjustments[key] = new_value
        
        self._apply_configuration_adjustments(exploration_adjustments, "exploration")
    
    def _apply_configuration_profile(self, profile: ConfigurationProfile):
        """应用配置档案"""
        logger.info(f"应用配置档案: {profile.name}")
        
        for key, value in profile.config_values.items():
            self._set_configuration_value(key, value)
        
        profile.usage_count += 1
        profile.last_used = time.time()
        self.current_profile = profile.name
    
    def _initialize_default_profiles(self):
        """初始化默认配置档案"""
        # 高性能档案
        high_performance = ConfigurationProfile(
            name="high_performance",
            description="高性能配置，适用于大规模数据处理",
            config_values={
                'vectorized_optimizer.batch_size': 1500,
                'vectorized_optimizer.chunk_size_mb': 128,
                'lsh_prefilter.num_hash_functions': 64,
                'pipeline.fast_path_threshold': 30
            }
        )
        
        # 节能档案
        energy_efficient = ConfigurationProfile(
            name="energy_efficient",
            description="节能配置，适用于资源受限环境",
            config_values={
                'vectorized_optimizer.batch_size': 500,
                'vectorized_optimizer.max_workers': 2,
                'lsh_prefilter.num_hash_functions': 32,
                'pipeline.fast_path_threshold': 80
            }
        )
        
        # 平衡档案
        balanced = ConfigurationProfile(
            name="balanced",
            description="平衡配置，性能和资源使用的折衷",
            config_values={
                'vectorized_optimizer.batch_size': 1000,
                'vectorized_optimizer.chunk_size_mb': 64,
                'lsh_prefilter.num_hash_functions': 48,
                'pipeline.fast_path_threshold': 50
            }
        )
        
        self.configuration_profiles.update({
            "high_performance": high_performance,
            "energy_efficient": energy_efficient,
            "balanced": balanced
        })
        
        # 设置默认档案
        self.current_profile = "balanced"
    
    def _load_configurations(self):
        """加载保存的配置"""
        try:
            if self.config_file.exists():
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                for name, profile_data in data.get('profiles', {}).items():
                    profile = ConfigurationProfile(**profile_data)
                    self.configuration_profiles[name] = profile
                
                logger.info(f"已加载 {len(self.configuration_profiles)} 个配置档案")
        
        except Exception as e:
            logger.warning(f"加载配置失败: {e}")
    
    def save_configurations(self):
        """保存配置到文件"""
        try:
            data = {
                'profiles': {
                    name: asdict(profile) 
                    for name, profile in self.configuration_profiles.items()
                },
                'current_profile': self.current_profile,
                'last_save_time': time.time()
            }
            
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"配置已保存到 {self.config_file}")
            
        except Exception as e:
            logger.error(f"保存配置失败: {e}")
    
    def get_optimization_recommendations(self) -> List[Dict[str, Any]]:
        """获取优化建议"""
        recommendations = []
        
        if len(self.metrics_history) < 5:
            return [{"type": "info", "message": "数据收集中，暂无优化建议"}]
        
        # 分析最近的性能指标
        recent_metrics = list(self.metrics_history)[-10:]
        avg_cpu = np.mean([m.cpu_percent for m in recent_metrics])
        avg_memory = np.mean([m.memory_percent for m in recent_metrics])
        avg_response_time = np.mean([m.avg_response_time for m in recent_metrics])
        
        # 生成建议
        if avg_cpu > 75:
            recommendations.append({
                "type": "warning",
                "category": "cpu",
                "message": f"CPU使用率较高 ({avg_cpu:.1f}%)，建议减少并行工作进程数",
                "suggested_config": {"vectorized_optimizer.max_workers": 2},
                "priority": "high"
            })
        
        if avg_memory > 80:
            recommendations.append({
                "type": "warning", 
                "category": "memory",
                "message": f"内存使用率较高 ({avg_memory:.1f}%)，建议减少缓存大小",
                "suggested_config": {"multi_cache.l1_memory_size": 500},
                "priority": "high"
            })
        
        if avg_response_time > 2.0:
            recommendations.append({
                "type": "suggestion",
                "category": "performance",
                "message": f"响应时间较慢 ({avg_response_time:.2f}s)，建议增加快速路径使用",
                "suggested_config": {"pipeline.fast_path_threshold": 100},
                "priority": "medium"
            })
        
        if not recommendations:
            recommendations.append({
                "type": "success",
                "message": "系统运行状态良好，无需调整配置"
            })
        
        return recommendations
    
    def get_system_status(self) -> Dict[str, Any]:
        """获取系统状态"""
        if not self.metrics_history:
            return {"status": "initializing", "message": "系统初始化中"}
        
        latest_metrics = self.metrics_history[-1]
        
        return {
            "status": "running",
            "monitoring_active": self.monitoring_active,
            "current_profile": self.current_profile,
            "metrics": {
                "cpu_percent": latest_metrics.cpu_percent,
                "memory_percent": latest_metrics.memory_percent,
                "memory_available_mb": latest_metrics.memory_available_mb,
                "query_throughput": latest_metrics.query_throughput,
                "avg_response_time": latest_metrics.avg_response_time
            },
            "profiles_count": len(self.configuration_profiles),
            "metrics_history_length": len(self.metrics_history)
        }


# 全局自适应配置引擎实例
_global_adaptive_engine = None

def get_adaptive_config_engine() -> AdaptiveConfigurationEngine:
    """获取全局自适应配置引擎"""
    global _global_adaptive_engine
    if _global_adaptive_engine is None:
        _global_adaptive_engine = AdaptiveConfigurationEngine()
    return _global_adaptive_engine