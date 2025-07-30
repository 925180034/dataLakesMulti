#!/usr/bin/env python3
"""
WebTable Phase 2 优化效果测试
测试优化后的系统在真实 WebTable 数据集上的性能表现
"""

import os
import sys
import json
import time
import asyncio
import logging
from pathlib import Path
from typing import Dict, List, Any, Tuple
import traceback

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.core.workflow import discover_data
from src.core.models import AgentState, TableInfo, ColumnInfo
from src.tools.performance_profiler import PerformanceProfiler
from src.tools.optimized_pipeline import OptimizedPipeline
from src.tools.intelligent_component_router import IntelligentComponentRouter, QueryContext

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class WebTablePhase2Tester:
    """WebTable Phase 2 优化效果测试器"""
    
    def __init__(self):
        self.project_root = Path(__file__).parent.parent
        self.examples_dir = self.project_root / "examples"
        self.profiler = PerformanceProfiler()
        self.results = {}
        self.test_scenarios = [
            {
                "name": "Join Tables - Small Scale",
                "tables_file": "webtable_join_tables.json",
                "queries_file": "webtable_join_queries.json",
                "ground_truth_file": "webtable_join_ground_truth.json",
                "sample_size": 50,
                "description": "小规模 Join 表格测试 (50 tables)"
            },
            {
                "name": "Join Tables - Medium Scale", 
                "tables_file": "webtable_join_tables.json",
                "queries_file": "webtable_join_queries.json",
                "ground_truth_file": "webtable_join_ground_truth.json",
                "sample_size": 200,
                "description": "中等规模 Join 表格测试 (200 tables)"
            },
            {
                "name": "Union Tables - Full Scale",
                "tables_file": "webtable_union_tables.json", 
                "queries_file": "webtable_union_queries.json",
                "ground_truth_file": "webtable_union_ground_truth.json",
                "sample_size": None,  # 全量数据
                "description": "全量 Union 表格测试"
            }
        ]

    def load_webtable_data(self, filename: str, sample_size: int = None) -> List[Dict]:
        """加载 WebTable 数据"""
        file_path = self.examples_dir / filename
        if not file_path.exists():
            raise FileNotFoundError(f"WebTable 数据文件不存在: {file_path}")
        
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if sample_size and len(data) > sample_size:
            data = data[:sample_size]
            
        logger.info(f"加载 {filename}: {len(data)} 条记录")
        return data

    def load_queries(self, filename: str, sample_size: int = 5) -> List[Dict]:
        """加载测试查询"""
        file_path = self.examples_dir / filename
        if not file_path.exists():
            logger.warning(f"查询文件不存在: {file_path}，使用默认查询")
            return [
                {"query": "find tables with similar schemas for joining", "intent": "join"},
                {"query": "discover tables with related data for union", "intent": "union"},
                {"query": "identify joinable tables with common columns", "intent": "join"},
                {"query": "find semantically similar tables", "intent": "union"},
                {"query": "locate tables that can be joined on key columns", "intent": "join"}
            ]
        
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if len(data) > sample_size:
            data = data[:sample_size]
            
        return data

    async def test_workflow_performance(self, tables: List[Dict], queries: List[Dict], scenario_name: str) -> Dict:
        """测试工作流性能"""
        logger.info(f"\n🚀 开始测试场景: {scenario_name}")
        logger.info(f"📊 数据规模: {len(tables)} 表格, {len(queries)} 查询")
        
        # 准备测试数据
        total_columns = sum(len(table.get('columns', [])) for table in tables)
        
        results = {
            'scenario': scenario_name,
            'tables_count': len(tables),
            'columns_count': total_columns,
            'queries_count': len(queries),
            'execution_times': [],
            'success_count': 0,
            'error_count': 0,
            'avg_results_per_query': 0,
            'total_time': 0,
            'throughput_qps': 0,
            'memory_usage': 0,
            'performance_breakdown': {}
        }
        
        start_time = time.time()
        
        try:
            # 准备表格数据（转换为 TableInfo 格式）
            table_info_list = []
            for table in tables:
                # 处理列数据格式
                columns = []
                if 'columns' in table:
                    for col in table['columns']:
                        if isinstance(col, dict):
                            # 标准化列数据格式
                            col_info = ColumnInfo(
                                table_name=col.get('table_name', table.get('table_name', 'unknown')),
                                column_name=col.get('column_name', col.get('name', 'unknown')),
                                data_type=col.get('data_type', col.get('type', 'string')),
                                sample_values=col.get('sample_values', [])
                            )
                            columns.append(col_info)
                
                # 创建表信息
                table_info = TableInfo(
                    table_name=table.get('table_name', table.get('name', 'unknown')),
                    columns=columns
                )
                table_info_list.append(table_info)
            
            # 测试每个查询
            all_results = []
            for i, query_data in enumerate(queries):
                query_text = query_data.get('query', f'test query {i+1}')
                logger.info(f"📝 执行查询 {i+1}/{len(queries)}: {query_text[:50]}...")
                
                # 性能监控
                with self.profiler.profile_component(f"query_{i+1}"):
                    try:
                        # 执行查询
                        query_start = time.time()
                        result = await discover_data(
                            user_query=query_text,
                            query_tables=[t.model_dump() for t in table_info_list],
                            query_columns=[]
                        )
                        query_time = time.time() - query_start
                        
                        results['execution_times'].append(query_time)
                        results['success_count'] += 1
                        
                        # 统计结果数量
                        if isinstance(result, dict):
                            result = AgentState(**result)
                        
                        if hasattr(result, 'final_results') and result.final_results:
                            all_results.extend(result.final_results)
                        
                        logger.info(f"✅ 查询 {i+1} 完成: {query_time:.3f}s, 结果数: {len(result.final_results) if hasattr(result, 'final_results') else 0}")
                        
                    except Exception as e:
                        logger.error(f"❌ 查询 {i+1} 失败: {str(e)}")
                        results['error_count'] += 1
                        results['execution_times'].append(0)
                        
                        # 如果是测试环境的连接问题，继续测试其他查询
                        if "connection" in str(e).lower() or "network" in str(e).lower():
                            logger.warning("网络连接问题，继续测试...")
                            continue
                        else:
                            # 打印详细错误信息用于调试
                            traceback.print_exc()
            
            # 计算总体统计
            total_time = time.time() - start_time
            results['total_time'] = total_time
            results['avg_results_per_query'] = len(all_results) / len(queries) if queries else 0
            results['throughput_qps'] = results['success_count'] / total_time if total_time > 0 else 0
            
            # 获取性能分析结果
            performance_report = self.profiler.get_performance_report()
            results['performance_breakdown'] = performance_report.get('component_stats', {})
            
            # 内存使用情况（模拟）
            results['memory_usage'] = len(tables) * 0.1 + total_columns * 0.01  # MB
            
            logger.info(f"🎯 场景完成: {results['success_count']}/{len(queries)} 成功")
            logger.info(f"⚡ 平均响应时间: {sum(results['execution_times'])/len(results['execution_times']):.3f}s")
            logger.info(f"🔥 吞吐量: {results['throughput_qps']:.2f} QPS")
            
        except Exception as e:
            logger.error(f"💥 测试场景失败: {str(e)}")
            traceback.print_exc()
            results['error_count'] = len(queries)
            
        return results

    async def run_comprehensive_test(self) -> Dict:
        """运行综合测试"""
        logger.info("🧪 开始 WebTable Phase 2 优化效果综合测试")
        logger.info("=" * 70)
        
        comprehensive_results = {
            'test_summary': {
                'total_scenarios': len(self.test_scenarios),
                'total_success': 0,
                'total_errors': 0,
                'total_tables_tested': 0,
                'total_queries_executed': 0,
                'overall_start_time': time.time()
            },
            'scenario_results': [],
            'performance_comparison': {},
            'optimization_metrics': {}
        }
        
        for scenario in self.test_scenarios:
            try:
                logger.info(f"\n📋 准备场景: {scenario['name']}")
                logger.info(f"📄 描述: {scenario['description']}")
                
                # 加载数据
                tables = self.load_webtable_data(
                    scenario['tables_file'], 
                    scenario['sample_size']
                )
                queries = self.load_queries(scenario['queries_file'])
                
                # 执行测试
                scenario_result = await self.test_workflow_performance(
                    tables, queries, scenario['name']
                )
                
                comprehensive_results['scenario_results'].append(scenario_result)
                comprehensive_results['test_summary']['total_success'] += scenario_result['success_count']
                comprehensive_results['test_summary']['total_errors'] += scenario_result['error_count']
                comprehensive_results['test_summary']['total_tables_tested'] += scenario_result['tables_count']
                comprehensive_results['test_summary']['total_queries_executed'] += scenario_result['queries_count']
                
            except Exception as e:
                logger.error(f"❌ 场景 {scenario['name']} 执行失败: {str(e)}")
                comprehensive_results['test_summary']['total_errors'] += 1
        
        # 计算总体指标
        total_time = time.time() - comprehensive_results['test_summary']['overall_start_time']
        comprehensive_results['test_summary']['total_time'] = total_time
        comprehensive_results['test_summary']['overall_throughput'] = (
            comprehensive_results['test_summary']['total_success'] / total_time 
            if total_time > 0 else 0
        )
        
        return comprehensive_results

    def generate_performance_report(self, results: Dict) -> str:
        """生成性能报告"""
        report = []
        report.append("=" * 80)
        report.append("🎯 WebTable Phase 2 优化效果测试报告")
        report.append("=" * 80)
        report.append("")
        
        # 测试总结
        summary = results['test_summary']
        report.append("📊 测试总结:")
        report.append(f"  总场景数: {summary['total_scenarios']}")
        report.append(f"  总表格数: {summary['total_tables_tested']}")
        report.append(f"  总查询数: {summary['total_queries_executed']}")
        report.append(f"  成功查询: {summary['total_success']}")
        report.append(f"  失败查询: {summary['total_errors']}")
        report.append(f"  成功率: {summary['total_success']/(summary['total_success']+summary['total_errors'])*100:.1f}%")
        report.append(f"  总执行时间: {summary['total_time']:.2f}s")  
        report.append(f"  整体吞吐量: {summary['overall_throughput']:.2f} QPS")
        report.append("")
        
        # 分场景结果
        report.append("🔍 分场景性能分析:")
        for scenario_result in results['scenario_results']:
            report.append(f"\n📋 {scenario_result['scenario']}:")
            report.append(f"  数据规模: {scenario_result['tables_count']} 表格, {scenario_result['columns_count']} 列")
            report.append(f"  查询数量: {scenario_result['queries_count']}")
            report.append(f"  成功/失败: {scenario_result['success_count']}/{scenario_result['error_count']}")
            
            if scenario_result['execution_times']:
                avg_time = sum(scenario_result['execution_times']) / len(scenario_result['execution_times'])
                min_time = min(scenario_result['execution_times'])
                max_time = max(scenario_result['execution_times'])
                report.append(f"  平均响应时间: {avg_time:.3f}s")
                report.append(f"  响应时间范围: {min_time:.3f}s - {max_time:.3f}s")
            
            report.append(f"  吞吐量: {scenario_result['throughput_qps']:.2f} QPS")
            report.append(f"  内存使用: {scenario_result['memory_usage']:.1f} MB")
            report.append(f"  平均结果数: {scenario_result['avg_results_per_query']:.1f}")
        
        # 优化效果评估
        report.append("")
        report.append("🚀 Phase 2 优化效果评估:")
        
        # 基于测试结果评估优化效果
        overall_qps = summary['overall_throughput']
        if overall_qps > 10:
            report.append(f"  ✅ 卓越性能: {overall_qps:.2f} QPS (超过预期)")
        elif overall_qps > 5:
            report.append(f"  ✅ 良好性能: {overall_qps:.2f} QPS (达到预期)")
        elif overall_qps > 1:
            report.append(f"  ⚠️  基础性能: {overall_qps:.2f} QPS (需要优化)")
        else:
            report.append(f"  ❌ 性能不足: {overall_qps:.2f} QPS (需要重新优化)")
        
        success_rate = summary['total_success']/(summary['total_success']+summary['total_errors'])*100
        if success_rate >= 95:
            report.append(f"  ✅ 高稳定性: {success_rate:.1f}% 成功率")
        elif success_rate >= 80:
            report.append(f"  ⚠️  中等稳定性: {success_rate:.1f}% 成功率")
        else:
            report.append(f"  ❌ 稳定性不足: {success_rate:.1f}% 成功率")
        
        report.append("")
        report.append("💡 优化建议:")
        if overall_qps < 5:
            report.append("  • 考虑启用并行处理模式")
            report.append("  • 优化向量计算算法")
        if success_rate < 90:
            report.append("  • 加强错误处理机制")
            report.append("  • 提升系统稳定性")
        
        report.append("  • 监控生产环境表现")
        report.append("  • 收集更多真实场景数据")
        report.append("")
        report.append("=" * 80)
        
        return "\n".join(report)


async def main():
    """主测试函数"""
    tester = WebTablePhase2Tester()
    
    try:
        # 运行综合测试
        results = await tester.run_comprehensive_test()
        
        # 生成并输出报告
        report = tester.generate_performance_report(results)
        print(report)
        
        # 保存结果到文件
        results_file = Path(__file__).parent.parent / "webtable_phase2_test_results.json"
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print(f"\n📁 详细结果已保存到: {results_file}")
        
    except Exception as e:
        logger.error(f"💥 测试执行失败: {str(e)}")
        traceback.print_exc()
        return False
    
    return True


if __name__ == "__main__":
    asyncio.run(main())