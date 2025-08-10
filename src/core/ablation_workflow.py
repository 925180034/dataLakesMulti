"""
消融实验专用工作流 - 支持层级指标收集
"""

import logging
import time
import numpy as np
from typing import Dict, Any, List, Tuple, Optional
from dataclasses import dataclass, field
from collections import defaultdict

from src.core.models import AgentState, TableInfo, ColumnInfo
from src.core.ultra_optimized_workflow import UltraOptimizedWorkflow
from src.tools.metadata_filter import MetadataFilter
from src.tools.batch_vector_search import BatchVectorSearch

logger = logging.getLogger(__name__)


@dataclass
class LayerPerformance:
    """单层性能数据"""
    layer_name: str
    input_candidates: List[str] = field(default_factory=list)
    output_candidates: List[str] = field(default_factory=list)
    processing_time: float = 0.0
    scores: Dict[str, float] = field(default_factory=dict)
    
    # 指标
    filter_ratio: float = 0.0
    retention_rate: float = 0.0  # 真实标签保留率
    precision_at_k: Dict[int, float] = field(default_factory=dict)
    recall_at_k: Dict[int, float] = field(default_factory=dict)
    hit_at_k: Dict[int, float] = field(default_factory=dict)
    
    def calculate_metrics(self, ground_truth: List[str]):
        """计算层级指标"""
        # 计算过滤比
        if self.input_candidates:
            self.filter_ratio = 1 - (len(self.output_candidates) / len(self.input_candidates))
        
        # 计算真实标签保留率
        if ground_truth:
            output_set = set(self.output_candidates)
            truth_set = set(ground_truth)
            retained = len(output_set & truth_set)
            self.retention_rate = retained / len(truth_set)
            
            # 计算不同K值的指标
            for k in [1, 3, 5, 10, 20]:
                if len(self.output_candidates) >= k:
                    top_k = set(self.output_candidates[:k])
                    correct = len(top_k & truth_set)
                    
                    # Precision@K
                    self.precision_at_k[k] = correct / k
                    
                    # Recall@K
                    self.recall_at_k[k] = correct / len(truth_set)
                    
                    # Hit@K
                    self.hit_at_k[k] = 1.0 if correct > 0 else 0.0


@dataclass
class PipelineMetrics:
    """完整管道的指标"""
    query_id: str
    ground_truth: List[str]
    layer_performances: Dict[str, LayerPerformance] = field(default_factory=dict)
    final_candidates: List[str] = field(default_factory=list)
    total_time: float = 0.0
    
    # 最终指标
    final_precision: float = 0.0
    final_recall: float = 0.0
    final_f1: float = 0.0
    final_hit_at_k: Dict[int, float] = field(default_factory=dict)
    
    def calculate_final_metrics(self):
        """计算最终指标"""
        if not self.ground_truth:
            return
        
        final_set = set(self.final_candidates)
        truth_set = set(self.ground_truth)
        correct = len(final_set & truth_set)
        
        # 精确率和召回率
        if self.final_candidates:
            self.final_precision = correct / len(self.final_candidates)
        if self.ground_truth:
            self.final_recall = correct / len(self.ground_truth)
        
        # F1分数
        if self.final_precision + self.final_recall > 0:
            self.final_f1 = 2 * self.final_precision * self.final_recall / (
                self.final_precision + self.final_recall
            )
        
        # Hit@K
        for k in [1, 3, 5, 10]:
            if len(self.final_candidates) >= k:
                top_k = set(self.final_candidates[:k])
                self.final_hit_at_k[k] = 1.0 if len(top_k & truth_set) > 0 else 0.0


class AblationWorkflow(UltraOptimizedWorkflow):
    """消融实验工作流 - 支持详细的层级指标收集"""
    
    def __init__(self, config: Optional[Dict] = None):
        super().__init__()
        
        # 配置参数
        self.config = config or {}
        self.metadata_top_k = self.config.get('metadata_top_k', 50)
        self.vector_top_k = self.config.get('vector_top_k', 20)
        self.llm_top_k = self.config.get('llm_top_k', 5)
        self.enable_llm = self.config.get('enable_llm', False)
        
        # 指标收集
        self.pipeline_metrics: List[PipelineMetrics] = []
        self.layer_statistics = defaultdict(list)
    
    async def run_with_metrics(
        self,
        query_info: Dict,
        all_tables: List[str],
        ground_truth: List[str]
    ) -> PipelineMetrics:
        """运行工作流并收集详细指标"""
        
        pipeline_metrics = PipelineMetrics(
            query_id=query_info.get('query_id', 'unknown'),
            ground_truth=ground_truth
        )
        
        start_time = time.time()
        
        # 层1：元数据过滤
        metadata_perf = await self._run_metadata_layer(
            query_info, all_tables, ground_truth
        )
        pipeline_metrics.layer_performances['metadata'] = metadata_perf
        
        # 层2：向量搜索
        vector_input = metadata_perf.output_candidates
        vector_perf = await self._run_vector_layer(
            query_info, vector_input, ground_truth
        )
        pipeline_metrics.layer_performances['vector'] = vector_perf
        
        # 层3：LLM匹配（可选）
        if self.enable_llm:
            llm_input = vector_perf.output_candidates
            llm_perf = await self._run_llm_layer(
                query_info, llm_input, ground_truth
            )
            pipeline_metrics.layer_performances['llm'] = llm_perf
            pipeline_metrics.final_candidates = llm_perf.output_candidates
        else:
            # 不使用LLM，直接使用向量搜索结果
            pipeline_metrics.final_candidates = vector_perf.output_candidates[:self.llm_top_k]
        
        # 计算总时间和最终指标
        pipeline_metrics.total_time = time.time() - start_time
        pipeline_metrics.calculate_final_metrics()
        
        # 保存到历史记录
        self.pipeline_metrics.append(pipeline_metrics)
        
        # 更新统计信息
        self._update_statistics(pipeline_metrics)
        
        return pipeline_metrics
    
    async def _run_metadata_layer(
        self,
        query_info: Dict,
        all_tables: List[str],
        ground_truth: List[str]
    ) -> LayerPerformance:
        """运行元数据层并收集指标"""
        
        perf = LayerPerformance(layer_name="metadata")
        perf.input_candidates = all_tables
        
        start_time = time.time()
        
        # 执行元数据过滤
        if hasattr(self, 'metadata_filter') and self.metadata_filter:
            # 找到查询表的TableInfo
            query_table_name = query_info.get('table_name', '')
            query_table_info = None
            for table in self.table_metadata_cache.values():
                if table.table_name == query_table_name:
                    query_table_info = table
                    break
            
            if not query_table_info:
                # 创建一个简单的TableInfo
                from src.core.models import TableInfo
                query_table_info = TableInfo(table_name=query_table_name, columns=[])
            
            results = self.metadata_filter.filter_candidates(
                query_table=query_table_info,
                all_tables=all_tables,
                top_k=self.metadata_top_k
            )
            perf.output_candidates = [name for name, score in results]
            perf.scores = {name: score for name, score in results}
        else:
            # 简单的基于名称的过滤
            query_table = query_info.get('table_name', '')
            scored_tables = []
            
            for table in all_tables:
                # 计算简单的相似度分数
                score = self._calculate_name_similarity(query_table, table)
                scored_tables.append((table, score))
            
            # 排序并取前K个
            scored_tables.sort(key=lambda x: x[1], reverse=True)
            top_tables = scored_tables[:self.metadata_top_k]
            
            perf.output_candidates = [name for name, _ in top_tables]
            perf.scores = {name: score for name, score in top_tables}
        
        perf.processing_time = time.time() - start_time
        
        # 计算指标
        perf.calculate_metrics(ground_truth)
        
        logger.info(f"Metadata layer: {len(perf.input_candidates)} -> {len(perf.output_candidates)} "
                   f"(filter={perf.filter_ratio:.1%}, retention={perf.retention_rate:.1%})")
        
        return perf
    
    async def _run_vector_layer(
        self,
        query_info: Dict,
        input_tables: List[str],
        ground_truth: List[str]
    ) -> LayerPerformance:
        """运行向量层并收集指标"""
        
        perf = LayerPerformance(layer_name="vector")
        perf.input_candidates = input_tables
        
        start_time = time.time()
        
        # 执行向量搜索
        if hasattr(self, 'vector_search') and self.vector_search:
            results = await self.vector_search.search_similar_tables(
                query_tables=[query_info.get('table_name', '')],
                candidate_tables=input_tables,
                k=self.vector_top_k
            )
            perf.output_candidates = [name for name, score in results]
            perf.scores = {name: score for name, score in results}
        else:
            # 简单的随机选择作为后备
            import random
            selected = random.sample(
                input_tables, 
                min(self.vector_top_k, len(input_tables))
            )
            perf.output_candidates = selected
            perf.scores = {name: 0.5 for name in selected}
        
        perf.processing_time = time.time() - start_time
        
        # 计算指标
        perf.calculate_metrics(ground_truth)
        
        logger.info(f"Vector layer: {len(perf.input_candidates)} -> {len(perf.output_candidates)} "
                   f"(filter={perf.filter_ratio:.1%}, retention={perf.retention_rate:.1%})")
        
        return perf
    
    async def _run_llm_layer(
        self,
        query_info: Dict,
        input_tables: List[str],
        ground_truth: List[str]
    ) -> LayerPerformance:
        """运行LLM层并收集指标"""
        
        perf = LayerPerformance(layer_name="llm")
        perf.input_candidates = input_tables
        
        start_time = time.time()
        
        # 这里应该调用实际的LLM API进行匹配
        # 现在简化处理，仅返回前K个
        perf.output_candidates = input_tables[:self.llm_top_k]
        perf.scores = {name: 1.0 - i*0.1 for i, name in enumerate(perf.output_candidates)}
        
        perf.processing_time = time.time() - start_time
        
        # 计算指标
        perf.calculate_metrics(ground_truth)
        
        logger.info(f"LLM layer: {len(perf.input_candidates)} -> {len(perf.output_candidates)} "
                   f"(filter={perf.filter_ratio:.1%}, retention={perf.retention_rate:.1%})")
        
        return perf
    
    def _calculate_name_similarity(self, name1: str, name2: str) -> float:
        """计算表名相似度（简单版本）"""
        # 转换为小写
        name1 = name1.lower()
        name2 = name2.lower()
        
        # 完全匹配
        if name1 == name2:
            return 1.0
        
        # 包含关系
        if name1 in name2 or name2 in name1:
            return 0.8
        
        # 共同前缀
        common_prefix_len = 0
        for i in range(min(len(name1), len(name2))):
            if name1[i] == name2[i]:
                common_prefix_len += 1
            else:
                break
        
        if common_prefix_len > 0:
            return 0.5 * (common_prefix_len / max(len(name1), len(name2)))
        
        # 编辑距离（简化版）
        return 0.1
    
    def _update_statistics(self, metrics: PipelineMetrics):
        """更新统计信息"""
        for layer_name, perf in metrics.layer_performances.items():
            self.layer_statistics[layer_name].append({
                'filter_ratio': perf.filter_ratio,
                'retention_rate': perf.retention_rate,
                'processing_time': perf.processing_time,
                'precision_at_5': perf.precision_at_k.get(5, 0),
                'recall_at_5': perf.recall_at_k.get(5, 0),
                'hit_at_5': perf.hit_at_k.get(5, 0)
            })
    
    def get_summary_statistics(self) -> Dict:
        """获取汇总统计信息"""
        summary = {}
        
        # 计算每层的平均指标
        for layer_name, stats_list in self.layer_statistics.items():
            if not stats_list:
                continue
            
            summary[layer_name] = {
                'avg_filter_ratio': np.mean([s['filter_ratio'] for s in stats_list]),
                'avg_retention_rate': np.mean([s['retention_rate'] for s in stats_list]),
                'avg_processing_time': np.mean([s['processing_time'] for s in stats_list]),
                'avg_precision_at_5': np.mean([s['precision_at_5'] for s in stats_list]),
                'avg_recall_at_5': np.mean([s['recall_at_5'] for s in stats_list]),
                'avg_hit_at_5': np.mean([s['hit_at_5'] for s in stats_list])
            }
        
        # 计算整体指标
        if self.pipeline_metrics:
            summary['overall'] = {
                'avg_total_time': np.mean([m.total_time for m in self.pipeline_metrics]),
                'avg_final_precision': np.mean([m.final_precision for m in self.pipeline_metrics]),
                'avg_final_recall': np.mean([m.final_recall for m in self.pipeline_metrics]),
                'avg_final_f1': np.mean([m.final_f1 for m in self.pipeline_metrics]),
                'avg_hit_at_1': np.mean([m.final_hit_at_k.get(1, 0) for m in self.pipeline_metrics]),
                'avg_hit_at_3': np.mean([m.final_hit_at_k.get(3, 0) for m in self.pipeline_metrics]),
                'avg_hit_at_5': np.mean([m.final_hit_at_k.get(5, 0) for m in self.pipeline_metrics])
            }
        
        return summary
    
    def print_summary(self):
        """打印汇总信息"""
        summary = self.get_summary_statistics()
        
        print("\n" + "="*60)
        print("LAYER PERFORMANCE SUMMARY")
        print("="*60)
        
        for layer_name in ['metadata', 'vector', 'llm']:
            if layer_name in summary:
                stats = summary[layer_name]
                print(f"\n{layer_name.upper()} Layer:")
                print(f"  Filter Ratio: {stats['avg_filter_ratio']:.1%}")
                print(f"  GT Retention: {stats['avg_retention_rate']:.1%}")
                print(f"  Processing Time: {stats['avg_processing_time']:.3f}s")
                print(f"  Precision@5: {stats['avg_precision_at_5']:.2%}")
                print(f"  Recall@5: {stats['avg_recall_at_5']:.2%}")
                print(f"  Hit@5: {stats['avg_hit_at_5']:.2%}")
        
        if 'overall' in summary:
            stats = summary['overall']
            print("\nOVERALL PERFORMANCE:")
            print(f"  Total Time: {stats['avg_total_time']:.3f}s")
            print(f"  Final Precision: {stats['avg_final_precision']:.2%}")
            print(f"  Final Recall: {stats['avg_final_recall']:.2%}")
            print(f"  Final F1: {stats['avg_final_f1']:.2%}")
            print(f"  Hit@1: {stats['avg_hit_at_1']:.2%}")
            print(f"  Hit@3: {stats['avg_hit_at_3']:.2%}")
            print(f"  Hit@5: {stats['avg_hit_at_5']:.2%}")
        
        print("="*60)