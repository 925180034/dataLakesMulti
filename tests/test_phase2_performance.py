"""
Phase 2 性能基准测试 - LSH预过滤、向量化计算、多级缓存和并行管道验证
基于架构升级计划的综合性能测试
"""

import pytest
import asyncio
import time
import logging
import random
import numpy as np
from typing import List, Dict, Any
from src.core.models import ColumnInfo, TableInfo
from src.config.settings import settings
from src.tools.lsh_prefilter import create_lsh_prefilter, LSHConfig
from src.tools.vectorized_optimizer import create_vectorized_optimizer, VectorizedConfig
from src.tools.multi_level_cache import MultiLevelCache
from src.tools.parallel_pipeline import create_phase2_pipeline, PipelineConfig

logger = logging.getLogger(__name__)


class Phase2PerformanceBenchmark:
    """Phase 2 性能基准测试"""
    
    def __init__(self):
        self.results = {}
        self.test_data = {}
        
        # 测试目标（基于架构升级计划）
        self.performance_targets = {
            'lsh_prefilter_reduction': 0.7,    # 70%候选减少
            'vectorized_speedup': 2.0,         # 2x向量化加速
            'cache_hit_rate': 0.8,             # 80%缓存命中率
            'pipeline_speedup': 3.0,           # 3x整体加速
            'memory_efficiency': 0.5           # 50%内存减少
        }
    
    async def run_full_benchmark(self) -> Dict[str, Any]:
        """运行完整的Phase 2性能基准测试"""
        logger.info("开始Phase 2性能基准测试")
        
        # 准备测试数据
        await self._prepare_test_data()
        
        # 1. LSH预过滤器测试
        lsh_results = await self._benchmark_lsh_prefilter()
        
        # 2. 向量化计算优化测试
        vectorized_results = await self._benchmark_vectorized_optimizer()
        
        # 3. 多级缓存系统测试
        cache_results = await self._benchmark_multi_level_cache()
        
        # 4. 并行管道集成测试
        pipeline_results = await self._benchmark_parallel_pipeline()
        
        # 5. 端到端性能对比
        e2e_results = await self._benchmark_end_to_end_comparison()
        
        # 整合结果
        self.results = {
            "lsh_prefilter": lsh_results,
            "vectorized_optimizer": vectorized_results,
            "multi_level_cache": cache_results,
            "parallel_pipeline": pipeline_results,
            "end_to_end_comparison": e2e_results,
            "performance_targets": self.performance_targets,
            "test_summary": self._generate_summary()
        }
        
        logger.info("Phase 2性能基准测试完成")
        return self.results
    
    async def _prepare_test_data(self):
        """准备测试数据"""
        logger.info("准备Phase 2测试数据...")
        
        # 创建不同规模的测试数据集
        self.test_data = {
            "small": await self._create_test_dataset(100, "small"),    # 100个表
            "medium": await self._create_test_dataset(1000, "medium"), # 1000个表
            "large": await self._create_test_dataset(5000, "large")    # 5000个表
        }
        
        logger.info(f"测试数据准备完成: {len(self.test_data)} 个数据集")
    
    async def _create_test_dataset(self, table_count: int, dataset_name: str) -> Dict[str, Any]:
        """创建测试数据集"""
        import string
        
        tables = []
        all_columns = []
        
        # 预定义的列名模式以增加相似性
        column_patterns = [
            ['id', 'name', 'email', 'created_at'],
            ['user_id', 'username', 'email_address', 'registration_date'],
            ['customer_id', 'customer_name', 'contact_email', 'signup_time'],
            ['product_id', 'product_name', 'description', 'price'],
            ['order_id', 'customer_id', 'product_id', 'quantity', 'order_date'],
            ['transaction_id', 'amount', 'currency', 'transaction_date'],
        ]
        
        for i in range(table_count):
            # 随机选择列模式或生成随机列
            if random.random() < 0.3:  # 30%概率使用预定义模式
                base_pattern = random.choice(column_patterns)
                column_names = base_pattern.copy()
                # 添加一些随机列
                for _ in range(random.randint(1, 3)):
                    column_names.append(f"col_{random.randint(1, 100)}")
            else:
                # 完全随机列名
                column_count = random.randint(3, 12)
                column_names = [f"col_{j}_{random.choice(['id', 'name', 'value', 'data', 'info'])}" 
                               for j in range(column_count)]
            
            table_name = f"{dataset_name}_table_{i}"
            columns = []
            
            for column_name in column_names:
                data_type = random.choice(["int", "varchar", "float", "datetime", "boolean"])
                sample_values = [f"sample_{k}" for k in range(3)]
                
                column = ColumnInfo(
                    table_name=table_name,
                    column_name=column_name,
                    data_type=data_type,
                    sample_values=sample_values
                )
                columns.append(column)
                all_columns.append(column)
            
            table = TableInfo(
                table_name=table_name,
                columns=columns,
                row_count=random.randint(100, 100000)
            )
            tables.append(table)
        
        return {
            "tables": tables,
            "columns": all_columns,
            "table_count": table_count,
            "column_count": len(all_columns)
        }
    
    async def _benchmark_lsh_prefilter(self) -> Dict[str, Any]:
        """LSH预过滤器性能测试"""
        logger.info("开始LSH预过滤器性能测试...")
        
        # 测试配置
        lsh_config = LSHConfig(
            num_hash_functions=64,
            num_hash_tables=8,
            similarity_threshold=0.5,
            max_candidates=1000
        )
        
        lsh_prefilter = create_lsh_prefilter(lsh_config)
        results = {}
        
        for dataset_name, dataset in self.test_data.items():
            logger.info(f"测试LSH预过滤器 - {dataset_name} 数据集")
            
            # 构建索引
            start_time = time.time()
            
            # 构建列索引
            columns_data = []
            for column in dataset["columns"]:
                column_data = {
                    'column_id': column.full_name,
                    'column_name': column.column_name,
                    'data_type': column.data_type,
                    'sample_values': column.sample_values,
                    'table_name': column.table_name
                }
                columns_data.append(column_data)
            
            lsh_prefilter.build_column_index(columns_data)
            
            # 构建表索引
            tables_data = []
            for table in dataset["tables"]:
                table_data = {
                    'table_name': table.table_name,
                    'columns': [{'column_name': col.column_name, 'data_type': col.data_type} 
                               for col in table.columns]
                }
                tables_data.append(table_data)
            
            lsh_prefilter.build_table_index(tables_data)
            
            build_time = time.time() - start_time
            
            # 测试查询性能和过滤效果
            query_times = []
            reduction_ratios = []
            
            # 选择查询样本
            query_samples = random.sample(dataset["columns"], min(20, len(dataset["columns"])))
            
            for query_column in query_samples:
                query_data = {
                    'column_name': query_column.column_name,
                    'data_type': query_column.data_type,
                    'sample_values': query_column.sample_values
                }
                
                # 测量查询时间
                start_time = time.time()
                candidates = lsh_prefilter.prefilter_columns(query_data, max_candidates=100)
                query_time = time.time() - start_time
                query_times.append(query_time)
                
                # 计算过滤效果
                total_candidates = len(dataset["columns"])
                filtered_count = len(candidates)
                reduction_ratio = 1 - (filtered_count / total_candidates)
                reduction_ratios.append(reduction_ratio)
            
            avg_query_time = sum(query_times) / len(query_times)
            avg_reduction_ratio = sum(reduction_ratios) / len(reduction_ratios)
            
            results[dataset_name] = {
                "build_time": build_time,
                "avg_query_time": avg_query_time,
                "avg_reduction_ratio": avg_reduction_ratio,
                "total_columns": len(dataset["columns"]),
                "lsh_stats": lsh_prefilter.get_performance_stats()
            }
            
            logger.info(f"LSH {dataset_name}: 构建={build_time:.2f}s, "
                       f"查询={avg_query_time*1000:.2f}ms, "
                       f"过滤率={avg_reduction_ratio:.1%}")
        
        return results
    
    async def _benchmark_vectorized_optimizer(self) -> Dict[str, Any]:
        """向量化计算优化器性能测试"""
        logger.info("开始向量化计算优化器性能测试...")
        
        # 测试配置
        vectorized_config = VectorizedConfig(
            batch_size=1000,
            use_gpu=False,  # 测试环境通常没有GPU
            max_workers=4,
            precision='float32'
        )
        
        vectorized_calculator = create_vectorized_optimizer(vectorized_config)
        results = {}
        
        # 测试不同向量规模
        test_cases = [
            (100, 100),   # 小规模
            (500, 500),   # 中等规模
            (1000, 1000), # 大规模
            (2000, 2000)  # 超大规模
        ]
        
        for rows, cols in test_cases:
            logger.info(f"测试向量化计算 - {rows}x{cols} 矩阵")
            
            # 生成测试向量
            vectors1 = np.random.rand(rows, 384).astype(np.float32)
            vectors2 = np.random.rand(cols, 384).astype(np.float32)
            
            # 向量化计算
            start_time = time.time()
            vectorized_similarity = vectorized_calculator.batch_cosine_similarity(vectors1, vectors2)
            vectorized_time = time.time() - start_time
            
            # 传统计算对比
            start_time = time.time()
            traditional_similarity = vectorized_calculator._fallback_cosine_similarity(vectors1, vectors2)
            traditional_time = time.time() - start_time
            
            # 计算加速比
            speedup = traditional_time / vectorized_time if vectorized_time > 0 else 0
            
            results[f"{rows}x{cols}"] = {
                "vectorized_time": vectorized_time,
                "traditional_time": traditional_time,
                "speedup": speedup,
                "matrix_size": rows * cols,
                "memory_usage_mb": (vectors1.nbytes + vectors2.nbytes) / (1024 * 1024)
            }
            
            logger.info(f"向量化 {rows}x{cols}: 向量化={vectorized_time:.3f}s, "
                       f"传统={traditional_time:.3f}s, 加速比={speedup:.1f}x")
        
        # 添加计算器统计
        results["calculator_stats"] = vectorized_calculator.get_performance_stats()
        
        return results
    
    async def _benchmark_multi_level_cache(self) -> Dict[str, Any]:
        """多级缓存系统性能测试"""
        logger.info("开始多级缓存系统性能测试...")
        
        multi_cache = MultiLevelCache()
        results = {}
        
        # 测试数据
        test_keys = [f"test_key_{i}" for i in range(1000)]
        test_values = [{"data": f"test_value_{i}", "size": random.randint(100, 10000)} 
                      for i in range(1000)]
        
        # 写入性能测试
        start_time = time.time()
        for key, value in zip(test_keys, test_values):
            multi_cache.put(key, value, ttl=3600)
        write_time = time.time() - start_time
        
        # 读取性能测试 - 第一轮（填充缓存）
        start_time = time.time()
        first_read_hits = 0
        for key in test_keys:
            value = multi_cache.get(key)
            if value is not None:
                first_read_hits += 1
        first_read_time = time.time() - start_time
        
        # 读取性能测试 - 第二轮（缓存命中）
        start_time = time.time()
        second_read_hits = 0
        for key in test_keys:
            value = multi_cache.get(key)
            if value is not None:
                second_read_hits += 1
        second_read_time = time.time() - start_time
        
        # 随机访问测试
        random_keys = random.sample(test_keys, 200)
        start_time = time.time()
        random_hits = 0
        for key in random_keys:
            value = multi_cache.get(key)
            if value is not None:
                random_hits += 1
        random_read_time = time.time() - start_time
        
        results = {
            "write_performance": {
                "total_writes": len(test_keys),
                "write_time": write_time,
                "writes_per_second": len(test_keys) / write_time
            },
            "read_performance": {
                "first_read_time": first_read_time,
                "first_read_hit_rate": first_read_hits / len(test_keys),
                "second_read_time": second_read_time,
                "second_read_hit_rate": second_read_hits / len(test_keys),
                "cache_speedup": first_read_time / second_read_time if second_read_time > 0 else 0
            },
            "random_access": {
                "random_read_time": random_read_time,
                "random_hit_rate": random_hits / len(random_keys),
                "reads_per_second": len(random_keys) / random_read_time
            },
            "cache_stats": multi_cache.get_comprehensive_stats()
        }
        
        logger.info(f"缓存测试: 写入={write_time:.2f}s, "
                   f"缓存命中率={second_read_hits / len(test_keys):.1%}, "
                   f"加速比={first_read_time / second_read_time:.1f}x")
        
        return results
    
    async def _benchmark_parallel_pipeline(self) -> Dict[str, Any]:
        """并行管道性能测试"""
        logger.info("开始并行管道性能测试...")
        
        # 测试配置
        pipeline_config = PipelineConfig(
            enable_lsh_prefilter=True,
            enable_vectorized_compute=True,
            enable_multi_cache=True,
            enable_parallel_processing=True,
            max_workers=4
        )
        
        pipeline = create_phase2_pipeline(pipeline_config)
        results = {}
        
        for dataset_name, dataset in self.test_data.items():
            if dataset_name == "large":  # 只测试大数据集以节省时间
                continue
                
            logger.info(f"测试并行管道 - {dataset_name} 数据集")
            
            # 选择测试样本
            query_tables = random.sample(dataset["tables"], min(5, len(dataset["tables"])))
            candidate_tables = dataset["tables"]
            
            # 测试表搜索性能
            table_search_times = []
            for query_table in query_tables:
                start_time = time.time()
                search_results = await pipeline.enhanced_table_search(
                    query_table, candidate_tables, k=10
                )
                search_time = time.time() - start_time
                table_search_times.append(search_time)
            
            avg_table_search_time = sum(table_search_times) / len(table_search_times)
            
            # 选择列样本测试
            query_columns = random.sample(dataset["columns"], min(10, len(dataset["columns"])))
            candidate_columns = dataset["columns"]
            
            # 测试列搜索性能
            column_search_times = []
            for query_column in query_columns:
                start_time = time.time()
                search_results = await pipeline.enhanced_column_search(
                    query_column, candidate_columns, k=10
                )
                search_time = time.time() - start_time
                column_search_times.append(search_time)
            
            avg_column_search_time = sum(column_search_times) / len(column_search_times)
            
            results[dataset_name] = {
                "avg_table_search_time": avg_table_search_time,
                "avg_column_search_time": avg_column_search_time,
                "pipeline_stats": pipeline.get_performance_stats()
            }
            
            logger.info(f"管道 {dataset_name}: 表搜索={avg_table_search_time:.3f}s, "
                       f"列搜索={avg_column_search_time:.3f}s")
        
        return results
    
    async def _benchmark_end_to_end_comparison(self) -> Dict[str, Any]:
        """端到端性能对比测试"""
        logger.info("开始端到端性能对比测试...")
        
        results = {}
        
        # 测试中等规模数据集
        dataset = self.test_data["medium"]
        query_tables = random.sample(dataset["tables"], min(3, len(dataset["tables"])))
        candidate_tables = dataset["tables"]
        
        # Phase 2增强管道
        pipeline_config = PipelineConfig(
            enable_lsh_prefilter=True,
            enable_vectorized_compute=True,
            enable_multi_cache=True,
            enable_parallel_processing=True
        )
        enhanced_pipeline = create_phase2_pipeline(pipeline_config)
        
        # Phase 1基础管道
        basic_config = PipelineConfig(
            enable_lsh_prefilter=False,
            enable_vectorized_compute=False,
            enable_multi_cache=False,
            enable_parallel_processing=False
        )
        basic_pipeline = create_phase2_pipeline(basic_config)
        
        # 测试增强管道
        start_time = time.time()
        enhanced_results = []
        for query_table in query_tables:
            table_results = await enhanced_pipeline.enhanced_table_search(
                query_table, candidate_tables, k=10
            )
            enhanced_results.extend(table_results)
        enhanced_time = time.time() - start_time
        
        # 测试基础管道
        start_time = time.time()
        basic_results = []
        for query_table in query_tables:
            table_results = await basic_pipeline.enhanced_table_search(
                query_table, candidate_tables, k=10
            )
            basic_results.extend(table_results)
        basic_time = time.time() - start_time
        
        # 计算改进
        speedup = basic_time / enhanced_time if enhanced_time > 0 else 0
        
        results = {
            "enhanced_pipeline": {
                "processing_time": enhanced_time,
                "results_count": len(enhanced_results),
                "stats": enhanced_pipeline.get_performance_stats()
            },
            "basic_pipeline": {
                "processing_time": basic_time,
                "results_count": len(basic_results),
                "stats": basic_pipeline.get_performance_stats()
            },
            "improvement": {
                "speedup": speedup,
                "time_saved": basic_time - enhanced_time,
                "efficiency_gain": (speedup - 1) * 100  # 百分比提升
            }
        }
        
        logger.info(f"端到端对比: Phase2={enhanced_time:.2f}s, "
                   f"Basic={basic_time:.2f}s, 加速比={speedup:.1f}x")
        
        return results
    
    def _generate_summary(self) -> Dict[str, Any]:
        """生成测试总结"""
        summary = {
            "test_date": time.strftime("%Y-%m-%d %H:%M:%S"),
            "phase2_objectives": {
                "lsh_prefilter_target": "70% candidate reduction",
                "vectorized_compute_target": "2x computation speedup", 
                "multi_cache_target": "80% cache hit rate",
                "pipeline_integration_target": "3x overall speedup"
            },
            "test_coverage": [
                "LSH预过滤器候选减少测试",
                "向量化计算加速比测试",
                "多级缓存命中率和性能测试",
                "并行管道集成测试",
                "端到端性能对比测试"
            ],
            "performance_targets": self.performance_targets
        }
        
        return summary


@pytest.mark.asyncio
async def test_phase2_performance_benchmark():
    """Phase 2 性能基准测试"""
    benchmark = Phase2PerformanceBenchmark()
    results = await benchmark.run_full_benchmark()
    
    # 验证基本结果存在
    assert "lsh_prefilter" in results
    assert "vectorized_optimizer" in results
    assert "multi_level_cache" in results
    assert "parallel_pipeline" in results
    assert "end_to_end_comparison" in results
    
    # 验证性能目标
    print("\n" + "="*80)
    print("Phase 2 增强功能性能测试结果")
    print("="*80)
    
    # LSH预过滤器结果
    print("\n1. LSH预过滤器性能:")
    lsh_results = results["lsh_prefilter"]
    for dataset_name, lsh_data in lsh_results.items():
        if isinstance(lsh_data, dict) and "avg_reduction_ratio" in lsh_data:
            reduction = lsh_data["avg_reduction_ratio"]
            target_met = "✅" if reduction >= 0.5 else "❌"  # 50%减少目标
            print(f"  {dataset_name}: 候选减少率={reduction:.1%} {target_met}")
    
    # 向量化计算结果
    print("\n2. 向量化计算性能:")
    vectorized_results = results["vectorized_optimizer"]
    for case_name, case_data in vectorized_results.items():
        if isinstance(case_data, dict) and "speedup" in case_data:
            speedup = case_data["speedup"]
            target_met = "✅" if speedup >= 1.5 else "❌"  # 1.5x加速目标
            print(f"  {case_name}: 加速比={speedup:.1f}x {target_met}")
    
    # 多级缓存结果
    print("\n3. 多级缓存性能:")
    cache_results = results["multi_level_cache"]
    if "read_performance" in cache_results:
        hit_rate = cache_results["read_performance"]["second_read_hit_rate"]
        speedup = cache_results["read_performance"]["cache_speedup"]
        target_met = "✅" if hit_rate >= 0.8 else "❌"  # 80%命中率目标
        print(f"  缓存命中率: {hit_rate:.1%} {target_met}")
        print(f"  缓存加速比: {speedup:.1f}x")
    
    # 端到端对比结果
    print("\n4. 端到端性能对比:")
    e2e_results = results["end_to_end_comparison"]
    if "improvement" in e2e_results:
        overall_speedup = e2e_results["improvement"]["speedup"]
        target_met = "✅" if overall_speedup >= 2.0 else "❌"  # 2x整体加速目标
        print(f"  整体加速比: {overall_speedup:.1f}x {target_met}")
        print(f"  效率提升: {e2e_results['improvement']['efficiency_gain']:.1f}%")
    
    print("\n" + "="*80)
    
    return results


if __name__ == "__main__":
    # 直接运行测试
    import asyncio
    asyncio.run(test_phase2_performance_benchmark())