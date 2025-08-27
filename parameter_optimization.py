#!/usr/bin/env python3
"""
参数优化脚本 - 系统地搜索最佳配置参数
通过网格搜索找到JOIN和UNION任务在三个数据集上的最佳参数组合
"""

import json
import os
import sys
import time
import logging
import itertools
from pathlib import Path
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass
from datetime import datetime
import yaml
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class ParameterConfig:
    """参数配置类"""
    # 阈值参数
    metadata_threshold: float
    vector_threshold: float
    llm_threshold: float
    
    # 权重参数
    l1_weight: float
    l2_weight: float
    l3_weight: float
    
    # 策略参数
    layer_combination: str
    vector_top_k: int
    min_column_overlap: int
    
    # 特殊优化（可选）
    allow_self_match: bool = False
    use_semantic: bool = False
    prompt_enhancement: str = None

@dataclass
class ExperimentResult:
    """实验结果类"""
    config: ParameterConfig
    dataset: str
    task: str
    hit_at_1: float
    hit_at_3: float
    hit_at_5: float
    precision_at_5: float
    recall_at_5: float
    avg_time: float
    
    @property
    def score(self) -> float:
        """综合评分 - 用于排序"""
        # 主要关注Hit@1，但也考虑其他指标
        return (self.hit_at_1 * 0.5 + 
                self.hit_at_3 * 0.2 + 
                self.hit_at_5 * 0.15 +
                self.precision_at_5 * 0.1 +
                self.recall_at_5 * 0.05)

class ParameterOptimizer:
    """参数优化器"""
    
    def __init__(self, max_queries: int = 5):
        self.max_queries = max_queries
        self.results = []
        
        # 定义参数搜索空间 - 极度宽松的阈值确保有结果
        self.search_space = {
            'join': {
                # JOIN任务参数范围 - 极低阈值确保高召回率
                'metadata_threshold': [0.001, 0.01, 0.02, 0.05],
                'vector_threshold': [0.001, 0.01, 0.05, 0.10],
                'llm_threshold': [0.001, 0.005, 0.01, 0.02],
                'l1_weight': [0.20, 0.25, 0.30],
                'l2_weight': [0.35, 0.40, 0.45],
                'l3_weight': [0.25, 0.30, 0.35],
                'layer_combination': ['union'],  # 使用最宽松的组合策略
                'vector_top_k': [100, 150],
                'min_column_overlap': [1]  # 最宽松的重叠要求
            },
            'union': {
                # UNION任务参数范围 - 极低阈值
                'metadata_threshold': [0.001, 0.005, 0.01, 0.02],
                'vector_threshold': [0.001, 0.005, 0.01, 0.05],
                'llm_threshold': [0.001, 0.003, 0.005, 0.01],
                'l1_weight': [0.15, 0.20],
                'l2_weight': [0.50, 0.55],
                'l3_weight': [0.25, 0.30],
                'layer_combination': ['union'],
                'vector_top_k': [150, 200],
                'min_column_overlap': [1],
                'allow_self_match': [True],
                'use_semantic': [True]
            }
        }
        
        # 数据集特定的调整因子
        self.dataset_factors = {
            'nlctables': {'quality': 1.0, 'prompt_enhance': True, 'name': 'NLCTables'},
            'opendata': {'quality': 1.0, 'self_match': True, 'name': 'OpenData'},
            'webtable': {'quality': 1.0, 'noise_tolerant': True, 'name': 'WebTable'}
        }
    
    def generate_configs(self, task: str, dataset: str = None) -> List[ParameterConfig]:
        """生成参数配置组合"""
        space = self.search_space[task]
        
        # 智能采样策略 - 不是全部组合，而是智能选择
        configs = []
        
        if task == 'join':
            # JOIN任务的关键组合
            key_combos = [
                # 高精度组合
                {'metadata': 0.30, 'vector': 0.40, 'llm': 0.05, 'strategy': 'weighted_union'},
                # 平衡组合  
                {'metadata': 0.25, 'vector': 0.35, 'llm': 0.03, 'strategy': 'weighted_union'},
                # 宽松组合（WebTable）
                {'metadata': 0.10, 'vector': 0.25, 'llm': 0.02, 'strategy': 'union'},
                # 严格组合（NLCTables）
                {'metadata': 0.35, 'vector': 0.45, 'llm': 0.08, 'strategy': 'intersection'},
            ]
            
            for combo in key_combos:
                # 尝试不同的权重组合
                for l1_w, l2_w, l3_w in [(0.30, 0.40, 0.30), 
                                          (0.25, 0.45, 0.30),
                                          (0.35, 0.35, 0.30),
                                          (0.20, 0.50, 0.30)]:
                    # 确保权重和为1
                    total = l1_w + l2_w + l3_w
                    l1_w, l2_w, l3_w = l1_w/total, l2_w/total, l3_w/total
                    
                    configs.append(ParameterConfig(
                        metadata_threshold=combo['metadata'],
                        vector_threshold=combo['vector'],
                        llm_threshold=combo['llm'],
                        l1_weight=l1_w,
                        l2_weight=l2_w,
                        l3_weight=l3_w,
                        layer_combination=combo['strategy'],
                        vector_top_k=60,
                        min_column_overlap=2
                    ))
        
        else:  # union
            # UNION任务的关键组合
            key_combos = [
                # 极宽松（最大召回）
                {'metadata': 0.05, 'vector': 0.15, 'llm': 0.005, 'strategy': 'union'},
                # 宽松
                {'metadata': 0.08, 'vector': 0.20, 'llm': 0.01, 'strategy': 'union'},
                # 平衡
                {'metadata': 0.10, 'vector': 0.25, 'llm': 0.015, 'strategy': 'union'},
                # 稍严格
                {'metadata': 0.15, 'vector': 0.30, 'llm': 0.02, 'strategy': 'weighted_union'},
            ]
            
            for combo in key_combos:
                # UNION更看重语义（L2）
                for l1_w, l2_w, l3_w in [(0.15, 0.60, 0.25),
                                          (0.10, 0.65, 0.25),
                                          (0.20, 0.55, 0.25),
                                          (0.15, 0.55, 0.30)]:
                    # 确保权重和为1
                    total = l1_w + l2_w + l3_w
                    l1_w, l2_w, l3_w = l1_w/total, l2_w/total, l3_w/total
                    
                    configs.append(ParameterConfig(
                        metadata_threshold=combo['metadata'],
                        vector_threshold=combo['vector'],
                        llm_threshold=combo['llm'],
                        l1_weight=l1_w,
                        l2_weight=l2_w,
                        l3_weight=l3_w,
                        layer_combination=combo['strategy'],
                        vector_top_k=100,
                        min_column_overlap=1,
                        allow_self_match=True,
                        use_semantic=True
                    ))
        
        # 如果指定了数据集，应用数据集特定的调整
        if dataset:
            adjusted_configs = []
            for config in configs:
                adjusted = self._apply_dataset_adjustment(config, dataset, task)
                adjusted_configs.append(adjusted)
            return adjusted_configs
        
        return configs
    
    def _apply_dataset_adjustment(self, config: ParameterConfig, 
                                  dataset: str, task: str) -> ParameterConfig:
        """应用数据集特定的调整"""
        factor = self.dataset_factors[dataset]['quality']
        
        # 调整阈值
        adjusted = ParameterConfig(
            metadata_threshold=config.metadata_threshold * factor,
            vector_threshold=config.vector_threshold * factor,
            llm_threshold=config.llm_threshold * factor,
            l1_weight=config.l1_weight,
            l2_weight=config.l2_weight,
            l3_weight=config.l3_weight,
            layer_combination=config.layer_combination,
            vector_top_k=config.vector_top_k,
            min_column_overlap=config.min_column_overlap,
            allow_self_match=config.allow_self_match,
            use_semantic=config.use_semantic
        )
        
        # 特殊处理
        if dataset == 'nlctables' and self.dataset_factors[dataset].get('prompt_enhance'):
            adjusted.prompt_enhancement = 'nlc_optimized'
        elif dataset == 'opendata' and task == 'union':
            adjusted.allow_self_match = True
        elif dataset == 'webtable':
            adjusted.layer_combination = 'union'  # WebTable总是用union
            adjusted.llm_threshold = min(adjusted.llm_threshold, 0.02)  # 更宽松
        
        return adjusted
    
    def run_experiment(self, config: ParameterConfig, dataset: str, 
                       task: str) -> ExperimentResult:
        """运行单个实验"""
        # 创建临时配置文件
        temp_config_file = f'temp_config_{task}_{dataset}_{time.time()}.yml'
        
        try:
            # 生成配置文件
            self._create_config_file(temp_config_file, config, task)
            
            # 设置环境变量
            os.environ['TEMP_CONFIG'] = temp_config_file
            os.environ['MAX_PREDICTIONS'] = '20'
            os.environ['SKIP_LLM'] = 'false'
            
            # 运行实验
            from three_layer_ablation_optimized import run_ablation_experiment_optimized
            
            # 构建正确的dataset_type路径
            if dataset == 'nlctables':
                # NLCTables使用特殊格式
                dataset_path = 'nlctables_subset'
            else:
                # OpenData和WebTable使用标准路径格式
                dataset_path = f"examples/{dataset}/{task}_subset"
            
            results = run_ablation_experiment_optimized(
                task_type=task,
                dataset_type=dataset_path,  # 传递完整路径
                max_queries=self.max_queries,
                max_workers=1,
                use_challenging=False
            )
            
            # 提取所有层的结果，找最好的
            best_result = None
            best_score = 0
            
            for layer_name, layer_result in results.items():
                if 'metrics' in layer_result:
                    metrics = layer_result['metrics']
                    score = (metrics.get('hit@1', 0) * 0.5 + 
                            metrics.get('hit@3', 0) * 0.3 +
                            metrics.get('hit@5', 0) * 0.2)
                    
                    if score > best_score:
                        best_score = score
                        best_result = ExperimentResult(
                            config=config,
                            dataset=dataset,
                            task=task,
                            hit_at_1=metrics.get('hit@1', 0),
                            hit_at_3=metrics.get('hit@3', 0),
                            hit_at_5=metrics.get('hit@5', 0),
                            precision_at_5=metrics.get('precision', 0),  # 修正字段名
                            recall_at_5=metrics.get('recall', 0),  # 修正字段名
                            avg_time=layer_result.get('avg_time', 0)
                        )
            
            return best_result if best_result else ExperimentResult(
                config=config,
                dataset=dataset,
                task=task,
                hit_at_1=0, hit_at_3=0, hit_at_5=0,
                precision_at_5=0, recall_at_5=0, avg_time=999
            )
            
        except Exception as e:
            logger.error(f"实验失败: {e}")
            import traceback
            logger.error(traceback.format_exc())
            # 返回失败结果
            return ExperimentResult(
                config=config,
                dataset=dataset,
                task=task,
                hit_at_1=0, hit_at_3=0, hit_at_5=0,
                precision_at_5=0, recall_at_5=0, avg_time=999
            )
        finally:
            # 清理临时文件
            if Path(temp_config_file).exists():
                Path(temp_config_file).unlink()
    
    def _create_config_file(self, filename: str, config: ParameterConfig, task: str):
        """创建临时配置文件"""
        yaml_config = {
            'task_configs': {
                task: {
                    'layer_combination': config.layer_combination,
                    'metadata_filter': {
                        'column_similarity_threshold': config.metadata_threshold,
                        'value_overlap_threshold': config.metadata_threshold * 0.4,
                        'min_column_overlap': config.min_column_overlap,
                        'max_candidates_per_table': 50 if task == 'join' else 80,
                        'allow_self_match': config.allow_self_match,
                        'fuzzy_column_matching': True,
                    },
                    'vector_search': {
                        'similarity_threshold': config.vector_threshold,
                        'top_k': config.vector_top_k,
                        'embedding_model': 'all-MiniLM-L6-v2'
                    },
                    'llm_matcher': {
                        'confidence_threshold': config.llm_threshold,
                        'scoring_mode': 'adaptive',
                        'max_candidates_to_evaluate': 40,
                        'temperature': 0.3,
                        'check_semantic_compatibility': config.use_semantic
                    },
                    'aggregator': {
                        'final_combination': 'weighted_average',
                        'layer_weights': {
                            'L1': config.l1_weight,
                            'L2': config.l2_weight,
                            'L3': config.l3_weight
                        },
                        'ranking_weights': {
                            'metadata_score': config.l1_weight,
                            'vector_score': config.l2_weight,
                            'llm_score': config.l3_weight
                        },
                        'max_results': 20 if task == 'join' else 30
                    }
                }
            },
            'optimization_config': {
                'enable_caching': True,
                'parallel_processing': True,
                'max_workers': 8
            }
        }
        
        # 如果有prompt优化，添加到配置
        if config.prompt_enhancement == 'nlc_optimized':
            yaml_config['task_configs'][task]['llm_matcher']['nlc_prompt_optimization'] = True
        
        with open(filename, 'w', encoding='utf-8') as f:
            yaml.dump(yaml_config, f, default_flow_style=False, allow_unicode=True)
    
    def optimize(self, task: str, datasets: List[str] = None):
        """运行优化流程"""
        if datasets is None:
            datasets = ['nlctables', 'opendata', 'webtable']
        
        all_results = []
        
        for dataset in datasets:
            logger.info(f"\n{'='*60}")
            logger.info(f"开始优化 {dataset.upper()} - {task.upper()} 任务")
            logger.info(f"{'='*60}")
            
            # 生成配置
            configs = self.generate_configs(task, dataset)
            logger.info(f"生成了 {len(configs)} 个配置组合")
            
            # 运行实验
            dataset_results = []
            for i, config in enumerate(configs, 1):
                logger.info(f"\n测试配置 {i}/{len(configs)}:")
                logger.info(f"  阈值: meta={config.metadata_threshold:.2f}, "
                          f"vec={config.vector_threshold:.2f}, "
                          f"llm={config.llm_threshold:.3f}")
                logger.info(f"  权重: L1={config.l1_weight:.2f}, "
                          f"L2={config.l2_weight:.2f}, "
                          f"L3={config.l3_weight:.2f}")
                logger.info(f"  策略: {config.layer_combination}")
                
                result = self.run_experiment(config, dataset, task)
                dataset_results.append(result)
                
                logger.info(f"  结果: Hit@1={result.hit_at_1:.1%}, "
                          f"Hit@5={result.hit_at_5:.1%}, "
                          f"时间={result.avg_time:.2f}s")
            
            # 找出最佳配置
            best_result = max(dataset_results, key=lambda x: x.score)
            all_results.append(best_result)
            
            logger.info(f"\n{dataset.upper()} 最佳配置:")
            self._print_result(best_result)
        
        return all_results
    
    def _print_result(self, result: ExperimentResult):
        """打印结果"""
        logger.info(f"  数据集: {result.dataset}")
        logger.info(f"  任务: {result.task}")
        logger.info(f"  Hit@1: {result.hit_at_1:.1%}")
        logger.info(f"  Hit@3: {result.hit_at_3:.1%}")
        logger.info(f"  Hit@5: {result.hit_at_5:.1%}")
        logger.info(f"  Precision@5: {result.precision_at_5:.1%}")
        logger.info(f"  Recall@5: {result.recall_at_5:.1%}")
        logger.info(f"  平均时间: {result.avg_time:.2f}s")
        logger.info(f"  综合评分: {result.score:.3f}")
        logger.info(f"  配置详情:")
        logger.info(f"    - 阈值: meta={result.config.metadata_threshold:.2f}, "
                  f"vec={result.config.vector_threshold:.2f}, "
                  f"llm={result.config.llm_threshold:.3f}")
        logger.info(f"    - 权重: L1={result.config.l1_weight:.2f}, "
                  f"L2={result.config.l2_weight:.2f}, "
                  f"L3={result.config.l3_weight:.2f}")
        logger.info(f"    - 策略: {result.config.layer_combination}")
    
    def save_best_configs(self, results: List[ExperimentResult], output_dir: str = 'optimized_configs'):
        """保存最佳配置"""
        Path(output_dir).mkdir(exist_ok=True)
        
        # 按任务分组
        by_task = {}
        for result in results:
            if result.task not in by_task:
                by_task[result.task] = []
            by_task[result.task].append(result)
        
        # 为每个任务创建最优配置
        for task, task_results in by_task.items():
            # 如果没有有效结果，跳过
            if not task_results:
                logger.warning(f"任务{task}没有有效结果，跳过配置生成")
                continue
                
            # 计算平均最优参数（加权平均，按得分加权）
            total_weight = sum(r.score for r in task_results)
            
            # 如果所有分数都是0，使用简单平均
            if total_weight == 0:
                logger.warning(f"任务{task}的所有实验分数都是0，使用简单平均")
                num_results = len(task_results)
                avg_meta = sum(r.config.metadata_threshold for r in task_results) / num_results
                avg_vec = sum(r.config.vector_threshold for r in task_results) / num_results
                avg_llm = sum(r.config.llm_threshold for r in task_results) / num_results
                avg_l1 = sum(r.config.l1_weight for r in task_results) / num_results
                avg_l2 = sum(r.config.l2_weight for r in task_results) / num_results
                avg_l3 = sum(r.config.l3_weight for r in task_results) / num_results
            else:
                # 加权平均
                avg_meta = sum(r.config.metadata_threshold * r.score for r in task_results) / total_weight
                avg_vec = sum(r.config.vector_threshold * r.score for r in task_results) / total_weight
                avg_llm = sum(r.config.llm_threshold * r.score for r in task_results) / total_weight
                avg_l1 = sum(r.config.l1_weight * r.score for r in task_results) / total_weight
                avg_l2 = sum(r.config.l2_weight * r.score for r in task_results) / total_weight
                avg_l3 = sum(r.config.l3_weight * r.score for r in task_results) / total_weight
            
            # 选择最常见的策略
            strategies = [r.config.layer_combination for r in task_results]
            most_common_strategy = max(set(strategies), key=strategies.count)
            
            # 创建优化后的配置文件
            optimized_config = {
                'task_configs': {
                    task: {
                        'layer_combination': most_common_strategy,
                        'metadata_filter': {
                            'column_similarity_threshold': round(avg_meta, 2),
                            'value_overlap_threshold': round(avg_meta * 0.4, 2),
                            'type_matching_weight': 0.3 if task == 'join' else 0.2,
                            'min_column_overlap': 2 if task == 'join' else 1,
                            'max_candidates_per_table': 50 if task == 'join' else 80,
                            'fuzzy_column_matching': True,
                            'allow_self_match': task == 'union',
                            'allow_subset_matching': task == 'union',
                            'allow_type_coercion': task == 'union'
                        },
                        'vector_search': {
                            'similarity_threshold': round(avg_vec, 2),
                            'top_k': 60 if task == 'join' else 100,
                            'embedding_model': 'all-MiniLM-L6-v2',
                            'column_name_weight': 0.3,
                            'value_embedding_weight': 0.5 if task == 'join' else 0.6,
                            'context_weight': 0.2 if task == 'join' else 0.1
                        },
                        'llm_matcher': {
                            'confidence_threshold': round(avg_llm, 3),
                            'scoring_mode': 'adaptive',
                            'max_candidates_to_evaluate': 40 if task == 'join' else 60,
                            'batch_size': 10,
                            'temperature': 0.3,
                            'focus_on_join_keys': task == 'join',
                            'check_semantic_compatibility': task == 'union',
                            'allow_partial_matches': task == 'union',
                            'value_similarity_weight': 0.5 if task == 'join' else 0.6
                        },
                        'aggregator': {
                            'final_combination': 'weighted_average',
                            'layer_weights': {
                                'L1': round(avg_l1, 2),
                                'L2': round(avg_l2, 2),
                                'L3': round(avg_l3, 2)
                            },
                            'ranking_weights': {
                                'metadata_score': round(avg_l1, 2),
                                'vector_score': round(avg_l2, 2),
                                'llm_score': round(avg_l3, 2)
                            },
                            'score_normalization': 'min_max',
                            'max_results': 20 if task == 'join' else 30
                        }
                    }
                },
                'optimization_config': {
                    'enable_caching': True,
                    'cache_ttl': 3600,
                    'parallel_processing': True,
                    'max_workers': 8,
                    'adaptive_adjustment': {
                        'enable': True,
                        'factors': {
                            'high_quality_data': 1.15,  # NLCTables
                            'medium_quality': 1.0,       # OpenData
                            'noisy_data': 0.85,          # WebTable
                            'has_self_matches': 0.95,    # OpenData UNION
                            'high_diversity': 0.9,       # WebTable
                            'clean_data': 1.1            # NLCTables UNION
                        }
                    }
                },
                'performance_targets': {
                    'hit_at_1': 0.70 if task == 'join' else 0.65,
                    'hit_at_5': 0.85 if task == 'join' else 0.80,
                    'precision_at_5': 0.30 if task == 'join' else 0.25,
                    'recall_at_5': 0.60 if task == 'join' else 0.55
                }
            }
            
            # 保存配置
            config_path = Path(output_dir) / f'config_{task}_optimized_auto.yml'
            with open(config_path, 'w', encoding='utf-8') as f:
                yaml.dump(optimized_config, f, default_flow_style=False, allow_unicode=True)
            
            logger.info(f"\n保存优化配置到: {config_path}")
            
            # 同时生成数据集特定的微调版本
            for dataset_result in task_results:
                dataset = dataset_result.dataset
                dataset_config = optimized_config.copy()
                
                # 应用数据集特定的调整
                factor = self.dataset_factors[dataset]['quality']
                task_cfg = dataset_config['task_configs'][task]
                task_cfg['metadata_filter']['column_similarity_threshold'] *= factor
                task_cfg['vector_search']['similarity_threshold'] *= factor
                task_cfg['llm_matcher']['confidence_threshold'] *= factor
                
                # 特殊处理
                if dataset == 'nlctables':
                    task_cfg['llm_matcher']['nlc_prompt_optimization'] = True
                elif dataset == 'webtable':
                    task_cfg['layer_combination'] = 'union'
                elif dataset == 'opendata' and task == 'union':
                    task_cfg['metadata_filter']['allow_self_match'] = True
                
                # 保存数据集特定配置
                dataset_config_path = Path(output_dir) / f'config_{dataset}_{task}_optimized.yml'
                with open(dataset_config_path, 'w', encoding='utf-8') as f:
                    yaml.dump(dataset_config, f, default_flow_style=False, allow_unicode=True)
                
                logger.info(f"保存{dataset}特定配置到: {dataset_config_path}")


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='参数优化脚本')
    parser.add_argument('--task', choices=['join', 'union', 'both'], 
                       default='both', help='优化的任务类型')
    parser.add_argument('--datasets', nargs='+',
                       default=['nlctables', 'opendata', 'webtable'],
                       help='要优化的数据集')
    parser.add_argument('--max-queries', type=int, default=5,
                       help='每个实验使用的查询数')
    parser.add_argument('--output-dir', default='optimized_configs',
                       help='输出目录')
    
    args = parser.parse_args()
    
    # 创建优化器
    optimizer = ParameterOptimizer(max_queries=args.max_queries)
    
    # 运行优化
    all_results = []
    tasks = ['join', 'union'] if args.task == 'both' else [args.task]
    
    for task in tasks:
        logger.info(f"\n{'='*70}")
        logger.info(f"开始优化 {task.upper()} 任务")
        logger.info(f"{'='*70}")
        
        task_results = optimizer.optimize(task, args.datasets)
        all_results.extend(task_results)
    
    # 保存最佳配置
    logger.info(f"\n{'='*70}")
    logger.info("保存最佳配置")
    logger.info(f"{'='*70}")
    
    optimizer.save_best_configs(all_results, args.output_dir)
    
    # 打印总结
    logger.info(f"\n{'='*70}")
    logger.info("优化完成 - 总结")
    logger.info(f"{'='*70}")
    
    for result in sorted(all_results, key=lambda x: x.score, reverse=True):
        logger.info(f"\n{result.dataset.upper()} - {result.task.upper()}:")
        logger.info(f"  综合评分: {result.score:.3f}")
        logger.info(f"  Hit@1: {result.hit_at_1:.1%}, Hit@5: {result.hit_at_5:.1%}")
    
    logger.info("\n优化配置已保存到optimized_configs目录")
    logger.info("使用方法：将生成的配置文件复制为config_join_universal.yml或config_union_universal.yml")


if __name__ == '__main__':
    main()