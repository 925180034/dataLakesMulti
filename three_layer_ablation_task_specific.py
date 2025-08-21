#!/usr/bin/env python
"""
三层加速架构消融实验 - 任务特定配置版本
支持JOIN和UNION使用不同的优化配置
"""

import os
import sys
import json
import time
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Any, Tuple
from datetime import datetime
import numpy as np
from collections import defaultdict
import yaml

# 添加项目根目录到路径
sys.path.append(str(Path(__file__).parent))

from src.core.models import TableInfo, ColumnInfo
from src.tools.metadata_filter import MetadataFilter
from src.tools.vector_search import get_vector_search_engine
from src.tools.smart_llm_matcher import SmartLLMMatcher
from src.tools.batch_embedding import BatchEmbeddingProcessor
from src.config.settings import Settings

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class TaskSpecificThreeLayerAcceleration:
    """任务特定的三层加速架构"""
    
    def __init__(self, config_file: str = None, use_llm: bool = True):
        """初始化
        
        Args:
            config_file: 配置文件路径 (config_join.yml 或 config_union.yml)
            use_llm: 是否使用LLM验证层
        """
        self.use_llm = use_llm
        
        # 加载任务特定配置
        if config_file:
            logger.info(f"Loading task-specific config from: {config_file}")
            with open(config_file, 'r', encoding='utf-8') as f:
                config_data = yaml.safe_load(f)
            
            # 创建Settings对象并更新配置
            self.settings = Settings()
            self._update_settings_from_config(config_data)
            self.task_type = config_data.get('task_type', 'general')
        else:
            # 使用默认配置
            self.settings = Settings()
            self.task_type = 'general'
        
        # 初始化三层组件
        self.metadata_filter = MetadataFilter()
        self.vector_search = get_vector_search_engine()
        self.embedding_processor = BatchEmbeddingProcessor()
        
        if self.use_llm:
            self.llm_matcher = SmartLLMMatcher()
        else:
            self.llm_matcher = None
            
        # 缓存
        self.embedding_cache = {}
        self.llm_cache = {}
        
        logger.info(f"Initialized {self.task_type} task-specific three-layer acceleration")
        
    def _update_settings_from_config(self, config_data: Dict[str, Any]):
        """从配置数据更新Settings对象"""
        # 更新向量搜索设置
        if 'vector_search' in config_data:
            vs_config = config_data['vector_search']
            self.settings.vector_db.similarity_threshold = vs_config.get('similarity_threshold', 0.35)
            self.settings.search.top_k = vs_config.get('top_k', 80)
            
        # 更新元数据过滤设置
        if 'metadata_filter' in config_data:
            mf_config = config_data['metadata_filter']
            self.settings.search.column_similarity_threshold = mf_config.get('column_similarity_threshold', 0.35)
            self.settings.search.min_column_overlap = mf_config.get('min_column_overlap', 2)
            
        # 更新LLM设置
        if 'llm_matcher' in config_data:
            llm_config = config_data['llm_matcher']
            self.settings.search.confidence_threshold = llm_config.get('confidence_threshold', 0.55)
            
        # 更新评分权重
        if 'scoring' in config_data and 'weights' in config_data['scoring']:
            weights = config_data['scoring']['weights']
            self.L1_weight = weights.get('metadata', 0.25)
            self.L2_weight = weights.get('vector', 0.40)
            self.L3_weight = weights.get('llm', 0.35)
        else:
            self.L1_weight = 0.25
            self.L2_weight = 0.40
            self.L3_weight = 0.35
    
    def build_index(self, tables: List[TableInfo]):
        """构建索引"""
        logger.info(f"Building index for {len(tables)} tables...")
        
        # Layer 1: 构建元数据索引
        self.metadata_filter.build_index(tables)
        
        # Layer 2: 构建向量索引
        # 这里假设向量已经预计算好了
        logger.info("Vector index assumed to be pre-built")
        
    def search(
        self, 
        query_table: TableInfo,
        all_tables: List[TableInfo],
        layers: str = "L1_L2_L3",
        top_k: int = 10
    ) -> List[Tuple[str, float, Dict[str, float]]]:
        """执行搜索
        
        Returns:
            List of (table_name, final_score, layer_scores)
        """
        start_time = time.time()
        
        # 获取所有表名
        all_table_names = [t.table_name for t in all_tables]
        
        # Layer 1: 元数据过滤
        if "L1" in layers:
            # 根据任务类型调整候选数量
            if self.task_type == 'join':
                l1_top_k = min(200, len(all_table_names))  # JOIN需要更多候选
            elif self.task_type == 'union':
                l1_top_k = min(80, len(all_table_names))   # UNION需要较少候选
            else:
                l1_top_k = min(150, len(all_table_names))
                
            l1_results = self.metadata_filter.filter_candidates(
                query_table, 
                all_table_names,
                top_k=l1_top_k
            )
            logger.debug(f"L1 returned {len(l1_results)} candidates")
        else:
            l1_results = [(name, 1.0) for name in all_table_names]
        
        # Layer 2: 向量搜索
        if "L2" in layers and len(l1_results) > 0:
            l2_scores = {}
            
            # 获取查询表的embedding
            query_key = f"table_{query_table.table_name}"
            if query_key in self.embedding_cache:
                query_embedding = self.embedding_cache[query_key]
            else:
                # 创建表描述
                query_desc = self._create_table_description(query_table)
                query_embedding = self.embedding_processor.process_single(query_desc)
                self.embedding_cache[query_key] = query_embedding
            
            # 搜索相似表
            try:
                # 根据任务类型调整搜索参数
                if self.task_type == 'join':
                    threshold = 0.25  # 更低阈值
                    search_k = min(120, len(l1_results))
                elif self.task_type == 'union':
                    threshold = 0.50  # 更高阈值
                    search_k = min(40, len(l1_results))
                else:
                    threshold = 0.35
                    search_k = min(80, len(l1_results))
                    
                similar_tables = self.vector_search.search_similar_tables(
                    query_embedding,
                    k=search_k,
                    threshold=threshold
                )
                
                for result in similar_tables:
                    l2_scores[result.item_id] = result.score
                    
            except Exception as e:
                logger.warning(f"Vector search failed: {e}, using uniform scores")
                for table_name, _ in l1_results:
                    l2_scores[table_name] = 0.5
        else:
            l2_scores = {name: score for name, score in l1_results}
        
        # Layer 3: LLM验证
        if "L3" in layers and self.use_llm and len(l1_results) > 0:
            # 根据任务类型选择候选数量
            if self.task_type == 'join':
                llm_candidates = l1_results[:30]  # JOIN验证更多
            elif self.task_type == 'union':
                llm_candidates = l1_results[:15]  # UNION验证较少
            else:
                llm_candidates = l1_results[:20]
                
            l3_scores = self._llm_verify(query_table, llm_candidates, all_tables)
        else:
            l3_scores = {}
        
        # 组合得分
        final_results = []
        
        for table_name, l1_score in l1_results[:top_k*3]:  # 考虑更多候选以防过滤
            # 获取各层得分
            l2_score = l2_scores.get(table_name, 0.0)
            l3_score = l3_scores.get(table_name, 0.0)
            
            # 根据启用的层计算最终得分
            layer_scores = {'L1': l1_score}
            total_weight = 0
            final_score = 0
            
            if "L1" in layers:
                final_score += l1_score * self.L1_weight
                total_weight += self.L1_weight
                
            if "L2" in layers:
                final_score += l2_score * self.L2_weight
                total_weight += self.L2_weight
                layer_scores['L2'] = l2_score
                
            if "L3" in layers and self.use_llm:
                final_score += l3_score * self.L3_weight
                total_weight += self.L3_weight
                layer_scores['L3'] = l3_score
            
            # 归一化
            if total_weight > 0:
                final_score /= total_weight
            
            # 任务特定的boost factors
            if self.task_type == 'join':
                # JOIN任务的特殊加分
                if self._has_foreign_key_pattern(table_name, query_table.table_name):
                    final_score *= 1.5  # 外键模式加分
                    
            elif self.task_type == 'union':
                # UNION任务的特殊加分
                if self._has_same_prefix(table_name, query_table.table_name):
                    final_score *= 2.0  # 同前缀大幅加分
            
            final_results.append((table_name, final_score, layer_scores))
        
        # 排序并返回top_k
        final_results.sort(key=lambda x: x[1], reverse=True)
        
        elapsed = time.time() - start_time
        logger.debug(f"Search completed in {elapsed:.2f}s")
        
        return final_results[:top_k]
    
    def _has_foreign_key_pattern(self, table1: str, table2: str) -> bool:
        """检查是否有潜在的外键关系"""
        # 简单的外键模式检测
        t1_lower = table1.lower()
        t2_lower = table2.lower()
        
        # 检查常见的外键模式
        fk_patterns = ['_id', '_key', '_code', '_no']
        for pattern in fk_patterns:
            if pattern in t1_lower or pattern in t2_lower:
                return True
        return False
    
    def _has_same_prefix(self, table1: str, table2: str) -> bool:
        """检查是否有相同前缀"""
        # 提取前缀
        def get_prefix(name):
            if '__' in name:
                return name.split('__')[0]
            elif '_' in name:
                parts = name.split('_')
                if len(parts) > 1:
                    return parts[0]
            return name[:min(10, len(name))]
        
        return get_prefix(table1) == get_prefix(table2)
    
    def _create_table_description(self, table: TableInfo) -> str:
        """创建表描述用于embedding"""
        desc_parts = [f"Table: {table.table_name}"]
        
        # 添加列信息
        for col in table.columns[:10]:  # 最多10列
            col_desc = f"- {col.column_name} ({col.data_type})"
            if col.sample_values:
                samples = ', '.join(str(v) for v in col.sample_values[:3])
                col_desc += f": {samples}"
            desc_parts.append(col_desc)
        
        return '\n'.join(desc_parts)
    
    def _llm_verify(
        self, 
        query_table: TableInfo,
        candidates: List[Tuple[str, float]],
        all_tables: List[TableInfo]
    ) -> Dict[str, float]:
        """LLM验证"""
        if not self.llm_matcher:
            return {}
        
        scores = {}
        table_map = {t.table_name: t for t in all_tables}
        
        for table_name, _ in candidates:
            if table_name not in table_map:
                continue
                
            candidate_table = table_map[table_name]
            
            # 检查缓存
            cache_key = f"{query_table.table_name}_{table_name}"
            if cache_key in self.llm_cache:
                scores[table_name] = self.llm_cache[cache_key]
                continue
            
            try:
                # 根据任务类型调整prompt
                if self.task_type == 'join':
                    score = self.llm_matcher.match_tables_for_join(
                        query_table, candidate_table
                    )
                elif self.task_type == 'union':
                    score = self.llm_matcher.match_tables_for_union(
                        query_table, candidate_table
                    )
                else:
                    score = self.llm_matcher.match_tables(
                        query_table, candidate_table
                    )
                    
                scores[table_name] = score
                self.llm_cache[cache_key] = score
                
            except Exception as e:
                logger.warning(f"LLM matching failed for {table_name}: {e}")
                scores[table_name] = 0.0
        
        return scores


def evaluate_with_metrics(predictions: Dict, ground_truth: Dict, k_values=[1, 3, 5]) -> Dict:
    """计算评估指标"""
    metrics = {}
    
    # Hit@K
    for k in k_values:
        hits = 0
        for query, pred_list in predictions.items():
            if query in ground_truth:
                true_tables = ground_truth[query]
                top_k_preds = pred_list[:k] if len(pred_list) >= k else pred_list
                if any(pred in true_tables for pred in top_k_preds):
                    hits += 1
        metrics[f'hit@{k}'] = hits / len(predictions) if predictions else 0
    
    # Precision, Recall, F1
    total_tp = 0
    total_fp = 0
    total_fn = 0
    
    for query, pred_list in predictions.items():
        if query in ground_truth:
            true_set = set(ground_truth[query])
            pred_set = set(pred_list[:5])  # Top-5 for P/R/F1
            
            tp = len(pred_set & true_set)
            fp = len(pred_set - true_set)
            fn = len(true_set - pred_set)
            
            total_tp += tp
            total_fp += fp
            total_fn += fn
    
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    metrics['precision'] = precision
    metrics['recall'] = recall
    metrics['f1_score'] = f1
    
    return metrics


def run_experiment(
    task: str,
    dataset: str,
    layers: str,
    max_queries: int = None,
    use_llm: bool = True,
    use_task_config: bool = True
):
    """运行实验"""
    
    # 选择配置文件
    if use_task_config:
        if task == 'join':
            config_file = 'config_join.yml'
        elif task == 'union':
            config_file = 'config_union.yml'
        else:
            config_file = None
    else:
        config_file = None
    
    # 初始化系统
    system = TaskSpecificThreeLayerAcceleration(
        config_file=config_file,
        use_llm=use_llm
    )
    
    # 加载数据
    if dataset == 'subset':
        table_file = 'examples/final_subset_tables.json'
        column_file = 'examples/final_subset_columns.json'
        if task == 'join':
            query_file = 'examples/final_subset_join_queries.json'
            ground_truth_file = 'examples/final_subset_join_ground_truth.json'
        else:
            query_file = 'examples/final_subset_union_queries.json'
            ground_truth_file = 'examples/final_subset_union_ground_truth.json'
    else:
        table_file = 'examples/final_complete_tables.json'
        column_file = 'examples/final_complete_columns.json'
        if task == 'join':
            query_file = 'examples/final_complete_join_queries.json'
            ground_truth_file = 'examples/final_complete_join_ground_truth.json'
        else:
            query_file = 'examples/final_complete_union_queries.json'
            ground_truth_file = 'examples/final_complete_union_ground_truth.json'
    
    # 加载表数据
    with open(table_file, 'r') as f:
        tables_data = json.load(f)
    
    with open(column_file, 'r') as f:
        columns_data = json.load(f)
    
    # 构建TableInfo对象
    table_map = {}
    for table_data in tables_data:
        table_name = table_data['table_name']
        table_columns = [
            ColumnInfo(
                table_name=col['table_name'],
                column_name=col['column_name'],
                data_type=col.get('data_type', 'unknown'),
                sample_values=col.get('sample_values', [])
            )
            for col in columns_data if col['table_name'] == table_name
        ]
        
        table_map[table_name] = TableInfo(
            table_name=table_name,
            columns=table_columns,
            row_count=table_data.get('row_count', 0),
            file_path=table_data.get('file_path', '')
        )
    
    all_tables = list(table_map.values())
    
    # 构建索引
    system.build_index(all_tables)
    
    # 加载查询和ground truth
    with open(query_file, 'r') as f:
        queries = json.load(f)
    
    with open(ground_truth_file, 'r') as f:
        ground_truth = json.load(f)
    
    # 限制查询数量
    if max_queries:
        queries = queries[:max_queries]
    
    # 执行查询
    predictions = {}
    total_time = 0
    
    for i, query_table_name in enumerate(queries):
        if query_table_name not in table_map:
            continue
        
        query_table = table_map[query_table_name]
        
        start_time = time.time()
        results = system.search(query_table, all_tables, layers=layers, top_k=10)
        elapsed = time.time() - start_time
        total_time += elapsed
        
        # 提取预测的表名
        pred_tables = [name for name, score, _ in results]
        predictions[query_table_name] = pred_tables
        
        if (i + 1) % 10 == 0:
            logger.info(f"Processed {i+1}/{len(queries)} queries")
    
    # 计算指标
    metrics = evaluate_with_metrics(predictions, ground_truth)
    
    # 计算性能指标
    avg_time = total_time / len(predictions) if predictions else 0
    qps = 1 / avg_time if avg_time > 0 else 0
    
    metrics['avg_query_time'] = avg_time
    metrics['qps'] = qps
    metrics['total_queries'] = len(predictions)
    
    return {
        'task': task,
        'dataset': dataset,
        'layers': layers,
        'use_task_config': use_task_config,
        'config_file': config_file,
        'metrics': metrics,
        'predictions': predictions
    }


def main():
    parser = argparse.ArgumentParser(description='Task-specific three-layer ablation experiment')
    parser.add_argument('--task', choices=['join', 'union', 'both'], default='both',
                        help='Task type to evaluate')
    parser.add_argument('--dataset', choices=['subset', 'complete'], default='subset',
                        help='Dataset to use')
    parser.add_argument('--layers', default='all',
                        help='Layers to test (L1, L1_L2, L1_L2_L3, or all)')
    parser.add_argument('--max-queries', type=int, default=None,
                        help='Maximum number of queries to test')
    parser.add_argument('--skip-llm', action='store_true',
                        help='Skip LLM layer (L3)')
    parser.add_argument('--compare-configs', action='store_true',
                        help='Compare task-specific vs general config')
    
    args = parser.parse_args()
    
    # 确定要测试的层配置
    if args.layers == 'all':
        layer_configs = ['L1', 'L1_L2', 'L1_L2_L3']
    else:
        layer_configs = [args.layers]
    
    # 确定要测试的任务
    if args.task == 'both':
        tasks = ['join', 'union']
    else:
        tasks = [args.task]
    
    # 运行实验
    all_results = {}
    
    for task in tasks:
        logger.info(f"\n{'='*80}")
        logger.info(f"Testing {task.upper()} task")
        logger.info(f"{'='*80}")
        
        task_results = {}
        
        for layers in layer_configs:
            # 如果skip-llm，跳过L3层
            if args.skip_llm and 'L3' in layers:
                continue
            
            logger.info(f"\nTesting {layers} layers...")
            
            # 测试任务特定配置
            logger.info("Using task-specific configuration...")
            result_specific = run_experiment(
                task=task,
                dataset=args.dataset,
                layers=layers,
                max_queries=args.max_queries,
                use_llm=not args.skip_llm,
                use_task_config=True
            )
            
            # 如果需要比较，也测试通用配置
            if args.compare_configs:
                logger.info("Using general configuration...")
                result_general = run_experiment(
                    task=task,
                    dataset=args.dataset,
                    layers=layers,
                    max_queries=args.max_queries,
                    use_llm=not args.skip_llm,
                    use_task_config=False
                )
                
                task_results[layers] = {
                    'specific': result_specific,
                    'general': result_general
                }
            else:
                task_results[layers] = result_specific
        
        all_results[task] = task_results
    
    # 打印结果
    print("\n" + "="*80)
    print("EXPERIMENT RESULTS")
    print("="*80)
    
    for task, task_results in all_results.items():
        print(f"\n{task.upper()} Task Results:")
        print("-"*60)
        
        for layers, results in task_results.items():
            if isinstance(results, dict) and 'specific' in results:
                # 比较模式
                print(f"\n{layers} (Comparison Mode):")
                
                spec_metrics = results['specific']['metrics']
                gen_metrics = results['general']['metrics']
                
                print(f"{'Metric':<15} {'Task-Specific':<15} {'General':<15} {'Improvement':<15}")
                print("-"*60)
                
                for metric in ['hit@1', 'hit@3', 'hit@5', 'precision', 'recall', 'f1_score']:
                    spec_val = spec_metrics[metric]
                    gen_val = gen_metrics[metric]
                    improvement = ((spec_val - gen_val) / gen_val * 100) if gen_val > 0 else 0
                    print(f"{metric:<15} {spec_val:<15.3f} {gen_val:<15.3f} {improvement:+.1f}%")
            else:
                # 单一模式
                print(f"\n{layers}:")
                metrics = results['metrics']
                
                print(f"{'Metric':<15} {'Value':<15}")
                print("-"*30)
                
                for metric in ['hit@1', 'hit@3', 'hit@5', 'precision', 'recall', 'f1_score']:
                    print(f"{metric:<15} {metrics[metric]:<15.3f}")
                
                print(f"{'avg_query_time':<15} {metrics['avg_query_time']:<15.3f}s")
                print(f"{'qps':<15} {metrics['qps']:<15.2f}")
    
    # 保存结果
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = f'experiment_results/task_specific_{args.task}_{args.dataset}_{timestamp}.json'
    
    os.makedirs('experiment_results', exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    
    print(f"\nResults saved to: {output_file}")


if __name__ == '__main__':
    main()