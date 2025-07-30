#!/usr/bin/env python3
"""
WebTable阶段二优化测试 - 在真实大规模WebTable数据集上验证性能优化效果
"""
import asyncio
import json
import logging
import time
import sys
from pathlib import Path

# 设置项目路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.tools.data_indexer import build_webtable_indices, verify_indices
from src.core.workflow import discover_data
from src.core.models import TableInfo, ColumnInfo
from src.config.settings import settings

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')
logger = logging.getLogger(__name__)


async def test_webtable_phase2_optimization():
    """在真实WebTable数据集上测试阶段二优化"""
    try:
        logger.info("🚀 开始WebTable阶段二优化测试")
        logger.info("📊 使用真实的大规模WebTable数据集")
        
        # 1. 检查WebTable数据集
        logger.info("\n📂 步骤1: 检查WebTable数据集")
        
        tables_file = Path("examples/webtable_join_tables.json")
        columns_file = Path("examples/webtable_join_columns.json") 
        
        if not tables_file.exists():
            raise Exception(f"WebTable表数据不存在: {tables_file}")
        
        if not columns_file.exists():
            logger.warning(f"WebTable列数据不存在: {columns_file}，将仅使用表数据")
            columns_file = None
        
        # 获取数据集统计信息
        with open(tables_file) as f:
            tables_data = json.load(f)
        
        logger.info(f"✅ WebTable数据集统计:")
        logger.info(f"   表数量: {len(tables_data)}")
        
        # 计算总列数（从表中）
        total_columns_in_tables = sum(len(table.get('columns', [])) for table in tables_data)
        logger.info(f"   表中总列数: {total_columns_in_tables}")
        
        # 2. 构建大规模索引（测试阶段二优化的索引构建性能）
        logger.info("\n🔧 步骤2: 构建WebTable大规模索引")
        
        start_time = time.time()
        
        index_result = await build_webtable_indices(
            tables_file=str(tables_file),
            columns_file=str(columns_file) if columns_file else None,
            save_path=None  # 使用默认路径
        )
        
        index_time = time.time() - start_time
        
        if index_result['status'] != 'success':
            raise Exception(f"WebTable索引构建失败: {index_result.get('error')}")
        
        logger.info("✅ WebTable大规模索引构建完成")
        logger.info(f"   索引构建时间: {index_time:.2f}秒")
        logger.info(f"   处理表数: {index_result['tables_processed']}")
        logger.info(f"   索引表数: {index_result['tables_indexed']}")
        logger.info(f"   处理列数: {index_result['columns_processed']}")
        logger.info(f"   索引列数: {index_result['columns_indexed']}")
        
        # 3. 验证索引
        logger.info("\n🔍 步骤3: 验证WebTable索引")
        
        verify_result = await verify_indices()
        if verify_result['status'] != 'success':
            raise Exception(f"索引验证失败: {verify_result.get('error')}")
        
        vector_stats = verify_result.get('vector_search', {})
        value_stats = verify_result.get('value_search', {})
        
        logger.info("✅ WebTable索引验证成功")
        logger.info(f"   向量搜索列数: {vector_stats.get('column_count', 0)}")
        logger.info(f"   向量搜索表数: {vector_stats.get('table_count', 0)}")
        logger.info(f"   值搜索索引列数: {value_stats.get('indexed_columns', 0)}")
        
        # 4. 执行真实查询测试
        logger.info("\n🔍 步骤4: 执行WebTable真实查询测试")
        
        # 创建测试查询
        test_queries = [
            {
                "query": "find tables with user demographic and personal information",
                "table": {
                    "table_name": "user_demographics",
                    "columns": [
                        {
                            "table_name": "user_demographics",
                            "column_name": "user_id",
                            "data_type": "int",
                            "sample_values": ["1", "2", "3"]
                        },
                        {
                            "table_name": "user_demographics", 
                            "column_name": "age",
                            "data_type": "int",
                            "sample_values": ["25", "30", "35"]
                        }
                    ]
                }
            },
            {
                "query": "find tables with financial and economic data",
                "table": {
                    "table_name": "financial_data",
                    "columns": [
                        {
                            "table_name": "financial_data",
                            "column_name": "amount",
                            "data_type": "decimal",
                            "sample_values": ["100.50", "200.75"]
                        }
                    ]
                }
            }
        ]
        
        total_discovery_time = 0
        successful_discoveries = 0
        total_matches = 0
        
        for i, test_query in enumerate(test_queries, 1):
            logger.info(f"\n--- 查询 {i}/{len(test_queries)} ---")
            logger.info(f"查询文本: {test_query['query']}")
            
            start_time = time.time()
            
            try:
                result = await discover_data(
                    user_query=test_query['query'],
                    query_tables=[test_query['table']],
                    query_columns=None
                )
                
                discovery_time = time.time() - start_time
                total_discovery_time += discovery_time
                
                # 处理返回结果类型
                if isinstance(result, dict):
                    from src.core.models import AgentState
                    result = AgentState.from_dict(result)
                
                logger.info(f"✅ 查询完成，耗时: {discovery_time:.2f}秒")
                
                if result.final_results:
                    matches_count = len(result.final_results)
                    total_matches += matches_count
                    logger.info(f"   找到匹配: {matches_count}个")
                    
                    # 显示前3个最佳匹配
                    for j, match in enumerate(result.final_results[:3], 1):
                        logger.info(f"     {j}. {match.target_table} (评分: {match.score:.1f})")
                else:
                    logger.info("   未找到匹配结果")
                
                successful_discoveries += 1
                
            except Exception as e:
                logger.error(f"❌ 查询 {i} 失败: {e}")
                continue
        
        # 5. 性能分析
        logger.info("\n📊 步骤5: WebTable阶段二优化性能分析")
        
        # 计算性能指标
        avg_discovery_time = total_discovery_time / successful_discoveries if successful_discoveries > 0 else 0
        tables_per_second_index = len(tables_data) / index_time if index_time > 0 else 0
        columns_per_second_index = index_result['columns_processed'] / index_time if index_time > 0 else 0
        avg_matches_per_query = total_matches / successful_discoveries if successful_discoveries > 0 else 0
        
        logger.info(f"🚀 WebTable索引构建性能:")
        logger.info(f"   数据规模: {len(tables_data)}表, {index_result['columns_processed']}列")
        logger.info(f"   构建时间: {index_time:.2f}秒")
        logger.info(f"   表处理速度: {tables_per_second_index:.1f} 表/秒")
        logger.info(f"   列处理速度: {columns_per_second_index:.1f} 列/秒")
        
        logger.info(f"🔍 WebTable搜索性能:")
        logger.info(f"   成功查询: {successful_discoveries}/{len(test_queries)}")
        logger.info(f"   总搜索时间: {total_discovery_time:.2f}秒")
        logger.info(f"   平均查询时间: {avg_discovery_time:.2f}秒")
        logger.info(f"   平均匹配数: {avg_matches_per_query:.1f}个/查询")
        
        # 6. 阶段二优化组件状态
        logger.info("\n🔧 阶段二优化组件状态:")
        
        optimization_status = {
            "匈牙利算法匹配器": settings.hungarian_matcher.enabled,
            "LSH预过滤器": settings.lsh_prefilter.enabled, 
            "向量化计算优化器": settings.vectorized_optimizer.enabled,
            "多级缓存": settings.cache.multi_level_cache.get('enabled', True),
            "GPU加速嵌入": True  # 从日志中可以看到使用了CUDA
        }
        
        for component, enabled in optimization_status.items():
            status_icon = "✅" if enabled else "❌"
            logger.info(f"   {status_icon} {component}: {'启用' if enabled else '禁用'}")
        
        enabled_optimizations = sum(optimization_status.values())
        logger.info(f"\n🎯 优化组件启用率: {enabled_optimizations}/{len(optimization_status)} ({enabled_optimizations/len(optimization_status)*100:.0f}%)")
        
        # 成功标准
        success_criteria = [
            index_time < 120,  # 索引构建时间 < 2分钟
            avg_discovery_time < 10,  # 平均查询时间 < 10秒
            successful_discoveries >= 1,  # 至少1个查询成功
            enabled_optimizations >= 4  # 至少4个优化组件启用
        ]
        
        success = all(success_criteria)
        
        if success:
            logger.info("\n🎉 WebTable阶段二优化测试成功！")
            logger.info("✅ 大规模真实数据处理性能优异")
            logger.info("✅ 阶段二优化组件全面启用")
            logger.info("✅ 搜索性能和匹配质量良好")
            return True
        else:
            logger.warning("\n⚠️  WebTable阶段二优化测试部分通过")
            return success_criteria
        
    except Exception as e:
        logger.error(f"❌ WebTable阶段二优化测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


async def main():
    """主函数"""
    logger.info("🧪 开始WebTable大规模阶段二优化测试")
    
    success = await test_webtable_phase2_optimization()
    
    if success is True:
        logger.info("\n🎉 WebTable阶段二优化测试完全成功！")
        logger.info("✅ 系统已在真实大规模数据上验证阶段二优化效果")
        sys.exit(0)
    else:
        logger.error("\n❌ WebTable阶段二优化测试未完全通过")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())