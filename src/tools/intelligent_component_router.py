"""
智能组件路由器 - Phase 2优化核心
统一管理和智能协调所有优化组件
"""

import logging
import time
import asyncio
from typing import Dict, Any, Optional, List, Tuple, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
from collections import defaultdict, deque
from src.config.settings import settings
from src.tools.performance_profiler import profile_component, get_performance_profiler
from src.tools.adaptive_config_system import get_adaptive_config_engine
from src.core.models import ColumnInfo, TableInfo, MatchResult, TableMatchResult

logger = logging.getLogger(__name__)


class ComponentType(Enum):
    """组件类型枚举"""
    LSH_PREFILTER = "lsh_prefilter"
    VECTORIZED_CALCULATOR = "vectorized_calculator"
    OPTIMIZED_VECTORIZED = "optimized_vectorized"
    MULTI_LEVEL_CACHE = "multi_level_cache"
    PARALLEL_PROCESSING = "parallel_processing"
    FAST_PATH = "fast_path"


class RouteStrategy(Enum):
    """路由策略枚举"""
    PERFORMANCE_OPTIMIZED = "performance_optimized"
    RESOURCE_CONSERVING = "resource_conserving"
    BALANCED = "balanced"
    ADAPTIVE = "adaptive"


@dataclass
class ComponentMetrics:
    """组件性能指标"""
    activation_count: int = 0
    total_processing_time: float = 0.0
    avg_processing_time: float = 0.0
    success_rate: float = 1.0
    resource_usage: float = 0.0
    speedup_factor: float = 1.0
    error_count: int = 0
    last_used: float = 0.0


@dataclass
class RoutingDecision:
    """路由决策"""
    selected_components: List[ComponentType] = field(default_factory=list)
    processing_strategy: str = "standard"
    estimated_performance: Dict[str, float] = field(default_factory=dict)
    reasoning: List[str] = field(default_factory=list)
    fallback_plan: Optional['RoutingDecision'] = None


@dataclass
class QueryContext:
    """查询上下文"""
    query_type: str  # column_search, table_search
    data_size: int
    complexity_hint: Optional[str] = None
    user_priority: str = "balanced"  # performance, resource, balanced
    time_constraint: Optional[float] = None
    quality_requirement: float = 0.8


class IntelligentComponentRouter:
    """智能组件路由器"""
    
    def __init__(self):
        # 组件实例（延迟初始化）
        self._components = {}
        self._component_metrics = defaultdict(ComponentMetrics)
        
        # 路由策略
        self.current_strategy = RouteStrategy.ADAPTIVE
        self.strategy_history = deque(maxlen=100)
        
        # 性能预测模型
        self.performance_models = {}
        self.decision_history = deque(maxlen=1000)
        
        # 自适应配置引擎
        self.adaptive_engine = get_adaptive_config_engine()
        
        # 路由规则权重
        self.routing_weights = {
            'data_size': 0.3,
            'resource_usage': 0.25,
            'historical_performance': 0.2,
            'system_load': 0.15,
            'user_priority': 0.1
        }
        
        # 组件激活阈值（动态调整）
        self.activation_thresholds = {
            ComponentType.LSH_PREFILTER: {'min_data_size': 200, 'max_cpu': 80},
            ComponentType.VECTORIZED_CALCULATOR: {'min_data_size': 100, 'max_memory': 80},
            ComponentType.OPTIMIZED_VECTORIZED: {'min_data_size': 500, 'max_memory': 70},
            ComponentType.MULTI_LEVEL_CACHE: {'min_queries': 5, 'max_memory': 85},
            ComponentType.PARALLEL_PROCESSING: {'min_data_size': 50, 'max_cpu': 75},
            ComponentType.FAST_PATH: {'max_data_size': 100, 'min_cpu': 20}
        }
        
        logger.info("智能组件路由器初始化完成")
    
    def get_component(self, component_type: ComponentType):
        """获取组件实例（延迟加载）"""
        if component_type not in self._components:
            self._components[component_type] = self._load_component(component_type)
        return self._components[component_type]
    
    def _load_component(self, component_type: ComponentType):
        """加载组件实例"""
        try:
            if component_type == ComponentType.LSH_PREFILTER:
                from src.tools.lsh_prefilter import get_lsh_prefilter
                return get_lsh_prefilter()
            
            elif component_type == ComponentType.VECTORIZED_CALCULATOR:
                from src.tools.vectorized_optimizer import get_vectorized_calculator
                return get_vectorized_calculator()
            
            elif component_type == ComponentType.OPTIMIZED_VECTORIZED:
                from src.tools.optimized_vectorized_calculator import get_optimized_vectorized_calculator
                return get_optimized_vectorized_calculator()
            
            elif component_type == ComponentType.MULTI_LEVEL_CACHE:
                from src.tools.multi_level_cache import get_multi_level_cache
                return get_multi_level_cache()
            
            else:
                logger.warning(f"未知组件类型: {component_type}")
                return None
                
        except Exception as e:
            logger.error(f"加载组件失败 {component_type}: {e}")
            return None
    
    @profile_component("intelligent_router")
    async def route_query(
        self, 
        query_context: QueryContext,
        query_data: Any = None
    ) -> RoutingDecision:
        """智能路由查询"""
        start_time = time.time()
        
        try:
            # 分析查询上下文
            context_analysis = self._analyze_query_context(query_context)
            
            # 评估系统状态
            system_status = self._evaluate_system_status()
            
            # 制定路由决策
            decision = self._make_routing_decision(
                query_context, context_analysis, system_status
            )
            
            # 记录决策历史
            decision_record = {
                'timestamp': start_time,
                'query_context': query_context,
                'decision': decision,
                'system_status': system_status
            }
            self.decision_history.append(decision_record)
            
            # 调整激活阈值（自适应学习）
            self._adjust_activation_thresholds(context_analysis, system_status)
            
            logger.debug(f"路由决策完成: {len(decision.selected_components)} 个组件, "
                        f"策略={decision.processing_strategy}")
            
            return decision
            
        except Exception as e:
            logger.error(f"路由决策失败: {e}")
            return self._create_fallback_decision(query_context)
    
    def _analyze_query_context(self, context: QueryContext) -> Dict[str, Any]:
        """分析查询上下文"""
        analysis = {
            'complexity_score': 0.0,
            'resource_requirement': 'medium',
            'urgency_level': 'normal',
            'optimization_potential': 0.5
        }
        
        # 计算复杂度分数
        size_factor = min(1.0, context.data_size / 1000)  # 标准化到[0,1]
        analysis['complexity_score'] = size_factor
        
        # 评估资源需求
        if context.data_size > 1000:
            analysis['resource_requirement'] = 'high'
        elif context.data_size < 100:
            analysis['resource_requirement'] = 'low'
        
        # 评估紧急程度
        if context.time_constraint and context.time_constraint < 1.0:
            analysis['urgency_level'] = 'high'
        elif context.time_constraint and context.time_constraint > 5.0:
            analysis['urgency_level'] = 'low'
        
        # 评估优化潜力
        historical_performance = self._get_historical_performance(context)
        analysis['optimization_potential'] = 1.0 - historical_performance
        
        return analysis
    
    def _evaluate_system_status(self) -> Dict[str, Any]:
        """评估系统状态"""
        # 获取系统指标
        adaptive_status = self.adaptive_engine.get_system_status()
        
        status = {
            'cpu_load': adaptive_status.get('metrics', {}).get('cpu_percent', 50),
            'memory_usage': adaptive_status.get('metrics', {}).get('memory_percent', 50),
            'query_throughput': adaptive_status.get('metrics', {}).get('query_throughput', 0),
            'component_health': {}
        }
        
        # 评估组件健康状态
        for component_type, metrics in self._component_metrics.items():
            health_score = self._calculate_component_health(metrics)
            status['component_health'][component_type.value] = health_score
        
        return status
    
    def _calculate_component_health(self, metrics: ComponentMetrics) -> float:
        """计算组件健康分数"""
        if metrics.activation_count == 0:
            return 1.0  # 未使用的组件认为是健康的
        
        # 综合考虑成功率、性能和资源使用
        success_score = metrics.success_rate
        performance_score = min(1.0, metrics.speedup_factor / 2.0)  # 期望2x加速
        resource_score = max(0.0, 1.0 - metrics.resource_usage / 100)
        
        health = (success_score * 0.5 + performance_score * 0.3 + resource_score * 0.2)
        return max(0.0, min(1.0, health))
    
    def _make_routing_decision(
        self,
        context: QueryContext,
        analysis: Dict[str, Any],
        system_status: Dict[str, Any]
    ) -> RoutingDecision:
        """制定路由决策"""
        decision = RoutingDecision()
        
        # 根据当前策略选择组件
        if self.current_strategy == RouteStrategy.PERFORMANCE_OPTIMIZED:
            decision = self._performance_optimized_routing(context, analysis, system_status)
        elif self.current_strategy == RouteStrategy.RESOURCE_CONSERVING:
            decision = self._resource_conserving_routing(context, analysis, system_status)
        elif self.current_strategy == RouteStrategy.BALANCED:
            decision = self._balanced_routing(context, analysis, system_status)
        else:  # ADAPTIVE
            decision = self._adaptive_routing(context, analysis, system_status)
        
        # 创建后备计划
        decision.fallback_plan = self._create_fallback_decision(context)
        
        return decision
    
    def _performance_optimized_routing(
        self, context: QueryContext, analysis: Dict[str, Any], system_status: Dict[str, Any]
    ) -> RoutingDecision:
        """性能优化路由策略"""
        decision = RoutingDecision()
        decision.processing_strategy = "performance_optimized"
        
        # 激进地使用所有可用的优化组件
        if context.data_size >= 200:
            decision.selected_components.append(ComponentType.LSH_PREFILTER)
            decision.reasoning.append("使用LSH预过滤以减少候选数量")
        
        if context.data_size >= 500:
            decision.selected_components.append(ComponentType.OPTIMIZED_VECTORIZED)
            decision.reasoning.append("使用优化向量化计算以提升大规模计算性能")
        elif context.data_size >= 100:
            decision.selected_components.append(ComponentType.VECTORIZED_CALCULATOR)
            decision.reasoning.append("使用标准向量化计算")
        
        if context.data_size >= 50:
            decision.selected_components.append(ComponentType.PARALLEL_PROCESSING)
            decision.reasoning.append("启用并行处理以提升吞吐量")
        
        # 总是启用缓存
        decision.selected_components.append(ComponentType.MULTI_LEVEL_CACHE)
        decision.reasoning.append("启用多级缓存以减少重复计算")
        
        # 估算性能提升
        decision.estimated_performance = {
            'speedup_factor': 3.5,
            'resource_usage': 80,
            'accuracy': 0.9
        }
        
        return decision
    
    def _resource_conserving_routing(
        self, context: QueryContext, analysis: Dict[str, Any], system_status: Dict[str, Any]
    ) -> RoutingDecision:
        """资源节约路由策略"""
        decision = RoutingDecision()
        decision.processing_strategy = "resource_conserving"
        
        # 只在确实需要时启用组件
        cpu_load = system_status.get('cpu_load', 50)
        memory_usage = system_status.get('memory_usage', 50)
        
        # 小数据优先使用快速路径
        if context.data_size < 100:
            decision.selected_components.append(ComponentType.FAST_PATH)
            decision.reasoning.append("小数据集使用快速路径处理")
        else:
            # 保守使用组件
            if context.data_size >= 500 and cpu_load < 60:
                decision.selected_components.append(ComponentType.LSH_PREFILTER)
                decision.reasoning.append("数据量大且CPU负载低时使用LSH预过滤")
            
            if context.data_size >= 200 and memory_usage < 70:
                decision.selected_components.append(ComponentType.VECTORIZED_CALCULATOR)
                decision.reasoning.append("内存充足时使用向量化计算")
            
            # 适度使用缓存
            if memory_usage < 75:
                decision.selected_components.append(ComponentType.MULTI_LEVEL_CACHE)
                decision.reasoning.append("内存允许时启用缓存")
        
        decision.estimated_performance = {
            'speedup_factor': 1.8,
            'resource_usage': 40,
            'accuracy': 0.85
        }
        
        return decision
    
    def _balanced_routing(
        self, context: QueryContext, analysis: Dict[str, Any], system_status: Dict[str, Any]
    ) -> RoutingDecision:
        """平衡路由策略"""
        decision = RoutingDecision()
        decision.processing_strategy = "balanced"
        
        # 平衡性能和资源使用
        cpu_load = system_status.get('cpu_load', 50)
        memory_usage = system_status.get('memory_usage', 50)
        
        # 基于动态阈值决策
        if context.data_size >= self.activation_thresholds[ComponentType.LSH_PREFILTER]['min_data_size']:
            if cpu_load < self.activation_thresholds[ComponentType.LSH_PREFILTER]['max_cpu']:
                decision.selected_components.append(ComponentType.LSH_PREFILTER)
                decision.reasoning.append("满足LSH激活条件")
        
        if context.data_size >= self.activation_thresholds[ComponentType.VECTORIZED_CALCULATOR]['min_data_size']:
            if memory_usage < self.activation_thresholds[ComponentType.VECTORIZED_CALCULATOR]['max_memory']:
                decision.selected_components.append(ComponentType.VECTORIZED_CALCULATOR)
                decision.reasoning.append("满足向量化计算激活条件")
        
        if context.data_size >= self.activation_thresholds[ComponentType.PARALLEL_PROCESSING]['min_data_size']:
            if cpu_load < self.activation_thresholds[ComponentType.PARALLEL_PROCESSING]['max_cpu']:
                decision.selected_components.append(ComponentType.PARALLEL_PROCESSING)
                decision.reasoning.append("满足并行处理激活条件")
        
        # 智能缓存决策
        cache_metrics = self._component_metrics.get(ComponentType.MULTI_LEVEL_CACHE)
        if cache_metrics and cache_metrics.speedup_factor > 1.2:
            decision.selected_components.append(ComponentType.MULTI_LEVEL_CACHE)
            decision.reasoning.append("缓存历史表现良好")
        
        decision.estimated_performance = {
            'speedup_factor': 2.5,
            'resource_usage': 60,
            'accuracy': 0.87
        }
        
        return decision
    
    def _adaptive_routing(
        self, context: QueryContext, analysis: Dict[str, Any], system_status: Dict[str, Any]
    ) -> RoutingDecision:
        """自适应路由策略"""
        decision = RoutingDecision()
        decision.processing_strategy = "adaptive"
        
        # 基于历史学习和当前状态做出决策
        historical_success = self._get_historical_success_rate(context)
        system_capacity = self._estimate_system_capacity(system_status)
        
        # 动态权重计算
        performance_weight = context.user_priority == "performance" and 0.6 or 0.4
        resource_weight = context.user_priority == "resource" and 0.6 or 0.4
        
        # 组件选择评分
        component_scores = {}
        
        for component_type in ComponentType:
            score = self._calculate_component_score(
                component_type, context, analysis, system_status,
                performance_weight, resource_weight
            )
            component_scores[component_type] = score
        
        # 选择评分最高的组件
        sorted_components = sorted(
            component_scores.items(), 
            key=lambda x: x[1], 
            reverse=True
        )
        
        selected_count = 0
        max_components = 4  # 最多选择4个组件
        
        for component_type, score in sorted_components:
            if score > 0.5 and selected_count < max_components:
                decision.selected_components.append(component_type)
                decision.reasoning.append(f"{component_type.value} 评分: {score:.2f}")
                selected_count += 1
        
        # 估算性能
        total_speedup = 1.0
        total_resource = 0.0
        
        for component_type in decision.selected_components:
            metrics = self._component_metrics.get(component_type)
            if metrics:
                total_speedup *= min(metrics.speedup_factor, 2.0)  # 限制单个组件的贡献
                total_resource += metrics.resource_usage
        
        decision.estimated_performance = {
            'speedup_factor': min(total_speedup, 5.0),  # 总体限制
            'resource_usage': min(total_resource, 100),
            'accuracy': 0.85 + (len(decision.selected_components) * 0.02)
        }
        
        return decision
    
    def _calculate_component_score(
        self,
        component_type: ComponentType,
        context: QueryContext,
        analysis: Dict[str, Any],
        system_status: Dict[str, Any],
        performance_weight: float,
        resource_weight: float
    ) -> float:
        """计算组件评分"""
        # 基础适用性评分
        base_score = self._get_component_applicability(component_type, context)
        
        # 历史性能评分
        metrics = self._component_metrics.get(component_type)
        performance_score = 0.5
        if metrics and metrics.activation_count > 0:
            performance_score = min(1.0, metrics.speedup_factor / 2.0)
        
        # 资源影响评分
        resource_score = 1.0
        if metrics:
            resource_impact = metrics.resource_usage / 100
            resource_score = max(0.0, 1.0 - resource_impact)
        
        # 系统状态适应性
        system_score = self._get_system_compatibility_score(component_type, system_status)
        
        # 综合评分
        final_score = (
            base_score * 0.3 +
            performance_score * performance_weight * 0.4 +
            resource_score * resource_weight * 0.2 +
            system_score * 0.1
        )
        
        return max(0.0, min(1.0, final_score))
    
    def _get_component_applicability(self, component_type: ComponentType, context: QueryContext) -> float:
        """获取组件适用性评分"""
        applicability_rules = {
            ComponentType.LSH_PREFILTER: lambda ctx: min(1.0, ctx.data_size / 500),
            ComponentType.VECTORIZED_CALCULATOR: lambda ctx: min(1.0, ctx.data_size / 200),
            ComponentType.OPTIMIZED_VECTORIZED: lambda ctx: min(1.0, ctx.data_size / 1000),
            ComponentType.MULTI_LEVEL_CACHE: lambda ctx: 0.8,  # 缓存总是有用的
            ComponentType.PARALLEL_PROCESSING: lambda ctx: min(1.0, ctx.data_size / 100),
            ComponentType.FAST_PATH: lambda ctx: max(0.0, 1.0 - ctx.data_size / 100)
        }
        
        rule = applicability_rules.get(component_type)
        return rule(context) if rule else 0.5
    
    def _get_system_compatibility_score(self, component_type: ComponentType, system_status: Dict[str, Any]) -> float:
        """获取系统兼容性评分"""
        cpu_load = system_status.get('cpu_load', 50)
        memory_usage = system_status.get('memory_usage', 50)
        
        # 不同组件对系统资源的敏感性不同
        if component_type in [ComponentType.PARALLEL_PROCESSING, ComponentType.OPTIMIZED_VECTORIZED]:
            # CPU密集型组件
            return max(0.0, 1.0 - cpu_load / 100)
        elif component_type in [ComponentType.MULTI_LEVEL_CACHE, ComponentType.VECTORIZED_CALCULATOR]:
            # 内存密集型组件
            return max(0.0, 1.0 - memory_usage / 100)
        else:
            # 通用组件
            system_load = (cpu_load + memory_usage) / 200
            return max(0.0, 1.0 - system_load)
    
    def _get_historical_performance(self, context: QueryContext) -> float:
        """获取历史性能"""
        # 简化实现：基于决策历史计算平均性能
        if not self.decision_history:
            return 0.5
        
        relevant_decisions = [
            d for d in self.decision_history 
            if d['query_context'].query_type == context.query_type
        ]
        
        if not relevant_decisions:
            return 0.5
        
        # 计算平均性能分数
        total_score = 0.0
        for decision_record in relevant_decisions[-10:]:  # 最近10次
            estimated_perf = decision_record['decision'].estimated_performance
            speedup = estimated_perf.get('speedup_factor', 1.0)
            accuracy = estimated_perf.get('accuracy', 0.8)
            score = (speedup / 3.0 + accuracy) / 2.0  # 标准化评分
            total_score += score
        
        return total_score / len(relevant_decisions)
    
    def _get_historical_success_rate(self, context: QueryContext) -> float:
        """获取历史成功率"""
        # 简化实现
        total_decisions = len(self.decision_history)
        if total_decisions == 0:
            return 0.9  # 默认乐观估计
        
        # 计算最近决策的成功率
        recent_decisions = list(self.decision_history)[-50:]  # 最近50次
        successful_decisions = len([
            d for d in recent_decisions 
            if d['decision'].estimated_performance.get('speedup_factor', 1.0) > 1.5
        ])
        
        return successful_decisions / len(recent_decisions)
    
    def _estimate_system_capacity(self, system_status: Dict[str, Any]) -> float:
        """估算系统容量"""
        cpu_load = system_status.get('cpu_load', 50)
        memory_usage = system_status.get('memory_usage', 50)
        
        # 计算剩余容量
        cpu_capacity = max(0.0, 100 - cpu_load) / 100
        memory_capacity = max(0.0, 100 - memory_usage) / 100
        
        # 返回综合容量评分
        return (cpu_capacity + memory_capacity) / 2.0
    
    def _adjust_activation_thresholds(self, analysis: Dict[str, Any], system_status: Dict[str, Any]):
        """调整激活阈值（自适应学习）"""
        # 基于系统状态动态调整阈值
        cpu_load = system_status.get('cpu_load', 50)
        memory_usage = system_status.get('memory_usage', 50)
        
        # 如果系统负载高，提高激活阈值
        if cpu_load > 80:
            for component_type in [ComponentType.PARALLEL_PROCESSING, ComponentType.OPTIMIZED_VECTORIZED]:
                if 'max_cpu' in self.activation_thresholds[component_type]:
                    self.activation_thresholds[component_type]['max_cpu'] = max(70, 
                        self.activation_thresholds[component_type]['max_cpu'] - 2)
        
        if memory_usage > 85:
            for component_type in [ComponentType.MULTI_LEVEL_CACHE, ComponentType.VECTORIZED_CALCULATOR]:
                if 'max_memory' in self.activation_thresholds[component_type]:
                    self.activation_thresholds[component_type]['max_memory'] = max(75,
                        self.activation_thresholds[component_type]['max_memory'] - 2)
    
    def _create_fallback_decision(self, context: QueryContext) -> RoutingDecision:
        """创建后备决策"""
        decision = RoutingDecision()
        decision.processing_strategy = "fallback"
        
        # 最小化的安全组件组合
        if context.data_size < 50:
            decision.selected_components = [ComponentType.FAST_PATH]
        else:
            decision.selected_components = [ComponentType.VECTORIZED_CALCULATOR]
        
        decision.reasoning = ["后备处理策略"]
        decision.estimated_performance = {
            'speedup_factor': 1.2,
            'resource_usage': 30,
            'accuracy': 0.8
        }
        
        return decision
    
    def update_component_metrics(self, component_type: ComponentType, 
                               processing_time: float, success: bool, 
                               speedup_factor: float = 1.0, resource_usage: float = 0.0):
        """更新组件性能指标"""
        metrics = self._component_metrics[component_type]
        
        metrics.activation_count += 1
        metrics.total_processing_time += processing_time
        metrics.avg_processing_time = metrics.total_processing_time / metrics.activation_count
        metrics.last_used = time.time()
        
        if success:
            # 更新成功率（指数移动平均）
            alpha = 0.1  # 学习率
            metrics.success_rate = metrics.success_rate * (1 - alpha) + alpha
        else:
            metrics.error_count += 1
            metrics.success_rate = metrics.success_rate * (1 - alpha)
        
        # 更新加速因子和资源使用（指数移动平均）
        metrics.speedup_factor = metrics.speedup_factor * 0.9 + speedup_factor * 0.1
        metrics.resource_usage = metrics.resource_usage * 0.9 + resource_usage * 0.1
    
    def get_routing_analytics(self) -> Dict[str, Any]:
        """获取路由分析报告"""
        return {
            'current_strategy': self.current_strategy.value,
            'total_decisions': len(self.decision_history),
            'component_metrics': {
                comp_type.value: asdict(metrics) 
                for comp_type, metrics in self._component_metrics.items()
            },
            'activation_thresholds': {
                comp_type.value: thresholds 
                for comp_type, thresholds in self.activation_thresholds.items()
            },
            'routing_weights': self.routing_weights,
            'loaded_components': list(self._components.keys())
        }
    
    def cleanup(self):
        """清理资源"""
        # 清理组件实例
        for component in self._components.values():
            if hasattr(component, 'cleanup'):
                try:
                    component.cleanup()
                except Exception as e:
                    logger.error(f"组件清理失败: {e}")
        
        self._components.clear()
        logger.info("智能组件路由器资源清理完成")


# 全局智能路由器实例
_global_intelligent_router = None

def get_intelligent_component_router() -> IntelligentComponentRouter:
    """获取全局智能组件路由器"""
    global _global_intelligent_router
    if _global_intelligent_router is None:
        _global_intelligent_router = IntelligentComponentRouter()
    return _global_intelligent_router