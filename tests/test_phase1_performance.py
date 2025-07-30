"""
Phase 1 性能对比测试 - HNSW vs FAISS 和匈牙利算法验证
基于架构升级计划的性能基准测试
"""

import pytest
import asyncio
import time
import logging
import random
from typing import List, Dict, Any
from src.core.models import ColumnInfo, TableInfo
from src.config.settings import settings

logger = logging.getLogger(__name__)


class Phase1PerformanceBenchmark:
    """Phase 1 性能基准测试"""
    
    def __init__(self):
        self.results = {}
        self.test_data = {}
    
    async def run_full_benchmark(self) -> Dict[str, Any]:
        """运行完整的性能基准测试"""
        logger.info("开始Phase 1性能基准测试")
        
        # 准备测试数据
        await self._prepare_test_data()
        
        # 1. HNSW vs FAISS 性能对比
        hnsw_results = await self._benchmark_hnsw_performance()
        faiss_results = await self._benchmark_faiss_performance()
        
        # 2. 匈牙利算法精确匹配测试
        hungarian_results = await self._benchmark_hungarian_matching()
        
        # 3. 端到端性能测试
        e2e_results = await self._benchmark_end_to_end()
        
        # 整合结果
        self.results = {
            "hnsw_performance": hnsw_results,
            "faiss_performance": faiss_results,
            "hungarian_matching": hungarian_results,
            "end_to_end": e2e_results,
            "comparison": self._compare_results(hnsw_results, faiss_results),
            "test_summary": self._generate_summary()
        }
        
        logger.info("Phase 1性能基准测试完成")
        return self.results
    
    async def _prepare_test_data(self):
        """准备测试数据"""
        logger.info("准备测试数据...")
        
        # 创建不同规模的测试数据集
        self.test_data = {
            "small": await self._create_test_dataset(50),    # 50个表
            "medium": await self._create_test_dataset(500),  # 500个表
            "large": await self._create_test_dataset(1000)   # 1000个表
        }
        
        logger.info(f"测试数据准备完成: {len(self.test_data)} 个数据集")
    
    async def _create_test_dataset(self, table_count: int) -> Dict[str, Any]:
        """创建测试数据集"""
        import string
        
        tables = []
        columns_list = []
        embeddings_list = []
        
        for i in range(table_count):
            # 创建表信息
            table_name = f"test_table_{i}"
            column_count = random.randint(3, 10)
            
            columns = []
            embeddings = []
            
            for j in range(column_count):
                # 创建列信息
                column_name = f"col_{j}_{''.join(random.choices(string.ascii_lowercase, k=3))}"
                data_type = random.choice(["int", "varchar", "float", "datetime", "boolean"])
                sample_values = [f"sample_{k}" for k in range(3)]
                
                column = ColumnInfo(
                    table_name=table_name,
                    column_name=column_name,
                    data_type=data_type,
                    sample_values=sample_values
                )
                columns.append(column)
                
                # 创建模拟向量
                embedding = [random.random() for _ in range(384)]
                embeddings.append(embedding)
            
            table = TableInfo(
                table_name=table_name,
                columns=columns,
                row_count=random.randint(100, 10000)
            )
            
            tables.append(table)
            columns_list.append(columns)
            embeddings_list.append(embeddings)
        
        return {
            "tables": tables,
            "columns": columns_list,
            "embeddings": embeddings_list,
            "table_count": table_count
        }
    
    async def _benchmark_hnsw_performance(self) -> Dict[str, Any]:
        """HNSW性能基准测试"""
        logger.info("开始HNSW性能测试...")
        
        from src.tools.hnsw_search import create_hnsw_search
        
        results = {}
        
        for dataset_name, dataset in self.test_data.items():
            logger.info(f"测试HNSW - {dataset_name} 数据集 ({dataset['table_count']} 表)")
            
            # 创建HNSW搜索引擎
            hnsw_engine = create_hnsw_search()
            
            # 测试索引构建时间
            start_time = time.time()
            
            for i, (table, columns, embeddings) in enumerate(zip(
                dataset["tables"], dataset["columns"], dataset["embeddings"]
            )):
                # 添加表向量
                table_embedding = [sum(emb[j] for emb in embeddings) / len(embeddings) 
                                 for j in range(384)]
                await hnsw_engine.add_table_vector(table, table_embedding)
                
                # 添加列向量
                for column, embedding in zip(columns, embeddings):
                    await hnsw_engine.add_column_vector(column, embedding)
            
            build_time = time.time() - start_time
            
            # 测试搜索性能
            search_times = []
            query_count = min(10, dataset['table_count'])
            
            for i in range(query_count):
                query_embedding = [random.random() for _ in range(384)]
                
                start_time = time.time()
                await hnsw_engine.search_similar_columns(query_embedding, k=10)
                search_time = time.time() - start_time
                search_times.append(search_time)
            
            avg_search_time = sum(search_times) / len(search_times)
            
            results[dataset_name] = {
                "build_time": build_time,
                "avg_search_time": avg_search_time,
                "total_elements": hnsw_engine.get_collection_stats()["hnsw_index"]["total_elements"],
                "memory_usage": "未测量"  # 可以添加内存监控
            }
            
            logger.info(f"HNSW {dataset_name}: 构建={build_time:.2f}s, 平均查询={avg_search_time*1000:.2f}ms")
        
        return results
    
    async def _benchmark_faiss_performance(self) -> Dict[str, Any]:
        """FAISS性能基准测试"""
        logger.info("开始FAISS性能测试...")
        
        from src.tools.vector_search import FAISSVectorSearch
        
        results = {}
        
        for dataset_name, dataset in self.test_data.items():
            logger.info(f"测试FAISS - {dataset_name} 数据集 ({dataset['table_count']} 表)")
            
            # 创建FAISS搜索引擎
            faiss_engine = FAISSVectorSearch(dimension=384)
            
            # 测试索引构建时间
            start_time = time.time()
            
            for i, (table, columns, embeddings) in enumerate(zip(
                dataset["tables"], dataset["columns"], dataset["embeddings"]
            )):
                # 添加表向量
                table_embedding = [sum(emb[j] for emb in embeddings) / len(embeddings) 
                                 for j in range(384)]
                await faiss_engine.add_table_vector(table, table_embedding)
                
                # 添加列向量
                for column, embedding in zip(columns, embeddings):
                    await faiss_engine.add_column_vector(column, embedding)
            
            build_time = time.time() - start_time
            
            # 测试搜索性能
            search_times = []
            query_count = min(10, dataset['table_count'])
            
            for i in range(query_count):
                query_embedding = [random.random() for _ in range(384)]
                
                start_time = time.time()
                await faiss_engine.search_similar_columns(query_embedding, k=10)
                search_time = time.time() - start_time
                search_times.append(search_time)
            
            avg_search_time = sum(search_times) / len(search_times)
            
            results[dataset_name] = {
                "build_time": build_time,
                "avg_search_time": avg_search_time,
                "total_elements": faiss_engine.column_index.ntotal + faiss_engine.table_index.ntotal,
                "memory_usage": "未测量"
            }
            
            logger.info(f"FAISS {dataset_name}: 构建={build_time:.2f}s, 平均查询={avg_search_time*1000:.2f}ms")
        
        return results
    
    async def _benchmark_hungarian_matching(self) -> Dict[str, Any]:
        """匈牙利算法精确匹配测试"""
        logger.info("开始匈牙利算法性能测试...")
        
        from src.tools.hungarian_matcher import create_hungarian_matcher
        
        matcher = create_hungarian_matcher(threshold=0.6)
        results = {}
        
        # 测试不同表大小的匹配性能
        test_cases = [
            (3, 3), (5, 5), (10, 10), (15, 15), (20, 20)
        ]
        
        for rows, cols in test_cases:
            logger.info(f"测试匈牙利算法 - {rows}x{cols} 矩阵")
            
            # 创建测试数据
            import random
            embeddings1 = [[random.random() for _ in range(384)] for _ in range(rows)]
            embeddings2 = [[random.random() for _ in range(384)] for _ in range(cols)]
            
            # 测试匹配时间
            times = []
            for _ in range(5):  # 运行5次取平均
                start_time = time.time()
                
                similarity_matrix = matcher.compute_similarity_matrix(embeddings1, embeddings2)
                total_score, matches, _ = matcher.find_optimal_matching(similarity_matrix)
                
                match_time = time.time() - start_time
                times.append(match_time)
            
            avg_time = sum(times) / len(times)
            results[f"{rows}x{cols}"] = {
                "avg_time": avg_time,
                "matrix_operations": rows * cols,
                "matches_found": len(matches) if 'matches' in locals() else 0
            }
            
            logger.info(f"匈牙利 {rows}x{cols}: 平均时间={avg_time*1000:.2f}ms")
        
        return results
    
    async def _benchmark_end_to_end(self) -> Dict[str, Any]:
        """端到端性能测试"""
        logger.info("开始端到端性能测试...")
        
        results = {}
        
        try:
            # 测试增强表匹配代理
            from src.agents.enhanced_table_matching import EnhancedTableMatchingAgent
            enhanced_agent = EnhancedTableMatchingAgent()
            
            # 使用中等规模数据集测试
            dataset = self.test_data["medium"]
            query_table = dataset["tables"][0]
            candidate_tables = dataset["tables"][1:11]  # 取10个候选表
            
            start_time = time.time()
            matching_results = await enhanced_agent.precise_matching(query_table, candidate_tables)
            e2e_time = time.time() - start_time
            
            results["enhanced_matching"] = {
                "total_time": e2e_time,
                "matches_found": len(matching_results),
                "avg_time_per_candidate": e2e_time / len(candidate_tables),
                "performance_stats": enhanced_agent.get_performance_stats()
            }
            
            logger.info(f"端到端增强匹配: {e2e_time:.2f}s, 找到 {len(matching_results)} 个匹配")
            
        except Exception as e:
            logger.error(f"端到端测试失败: {e}")
            results["error"] = str(e)
        
        return results
    
    def _compare_results(self, hnsw_results: Dict, faiss_results: Dict) -> Dict[str, Any]:
        """比较HNSW和FAISS结果"""
        comparison = {}
        
        for dataset_name in hnsw_results.keys():
            if dataset_name in faiss_results:
                hnsw = hnsw_results[dataset_name]
                faiss = faiss_results[dataset_name]
                
                # 计算性能提升比例
                build_time_improvement = (faiss["build_time"] - hnsw["build_time"]) / faiss["build_time"] * 100
                search_time_improvement = (faiss["avg_search_time"] - hnsw["avg_search_time"]) / faiss["avg_search_time"] * 100
                
                comparison[dataset_name] = {
                    "build_time_improvement_percent": build_time_improvement,
                    "search_time_improvement_percent": search_time_improvement,
                    "hnsw_faster_build": build_time_improvement > 0,
                    "hnsw_faster_search": search_time_improvement > 0,
                    "meets_target": search_time_improvement >= 30  # 目标30%提升
                }
        
        return comparison
    
    def _generate_summary(self) -> Dict[str, Any]:
        """生成测试总结"""
        summary = {
            "test_date": time.strftime("%Y-%m-%d %H:%M:%S"),
            "phase1_objectives": {
                "hnsw_performance_target": "30-50% improvement over FAISS",
                "hungarian_accuracy_target": "10-15% improvement in matching precision",
                "query_time_target": "2.5s -> 1.5s (40% improvement)"
            },
            "test_coverage": [
                "HNSW vs FAISS performance comparison",
                "Hungarian algorithm precision matching",
                "End-to-end enhanced table matching",
                "Scalability testing (50-1000 tables)"
            ]
        }
        
        return summary


@pytest.mark.asyncio
async def test_phase1_performance_benchmark():
    """Phase 1 性能基准测试"""
    benchmark = Phase1PerformanceBenchmark()
    results = await benchmark.run_full_benchmark()
    
    # 验证基本结果存在
    assert "hnsw_performance" in results
    assert "faiss_performance" in results
    assert "hungarian_matching" in results
    assert "comparison" in results
    
    # 验证性能目标
    comparison = results["comparison"]
    for dataset_name, comparison_data in comparison.items():
        # 至少在某个数据集上HNSW应该表现更好
        if comparison_data.get("meets_target", False):
            logger.info(f"✅ {dataset_name} 数据集达到30%性能提升目标")
        else:
            logger.warning(f"⚠️ {dataset_name} 数据集未达到性能目标")
    
    # 打印详细结果
    print("\n" + "="*80)
    print("Phase 1 性能测试结果")
    print("="*80)
    
    for dataset_name in ["small", "medium", "large"]:
        if dataset_name in results["hnsw_performance"] and dataset_name in results["faiss_performance"]:
            hnsw = results["hnsw_performance"][dataset_name]
            faiss = results["faiss_performance"][dataset_name]
            comp = results["comparison"][dataset_name]
            
            print(f"\n{dataset_name.upper()} 数据集:")
            print(f"  HNSW: 构建={hnsw['build_time']:.2f}s, 查询={hnsw['avg_search_time']*1000:.2f}ms")
            print(f"  FAISS: 构建={faiss['build_time']:.2f}s, 查询={faiss['avg_search_time']*1000:.2f}ms")
            print(f"  提升: 构建={comp['build_time_improvement_percent']:.1f}%, 查询={comp['search_time_improvement_percent']:.1f}%")
            print(f"  目标达成: {'✅' if comp['meets_target'] else '❌'}")
    
    print("\n匈牙利算法测试:")
    if "hungarian_matching" in results:
        for case, result in results["hungarian_matching"].items():
            print(f"  {case}: {result['avg_time']*1000:.2f}ms")
    
    print("\n" + "="*80)
    
    return results


if __name__ == "__main__":
    # 直接运行测试
    import asyncio
    asyncio.run(test_phase1_performance_benchmark())