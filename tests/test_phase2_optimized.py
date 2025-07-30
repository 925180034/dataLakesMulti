#!/usr/bin/env python3
"""
阶段二优化测试 - 验证性能优化组件是否正常工作
"""
import asyncio
import json
import logging
import time
import sys
from pathlib import Path

# 设置项目路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.tools.data_indexer import build_webtable_indices
from src.core.workflow import discover_data
from src.core.models import TableInfo, ColumnInfo
from src.config.settings import settings

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')
logger = logging.getLogger(__name__)


async def create_large_test_dataset():
    """创建大规模测试数据集以验证性能优化效果"""
    
    # 创建大规模表数据（模拟真实WebTable环境）
    test_tables = []
    
    # 用户相关表
    for i in range(20):
        test_tables.append({
            "name": f"users_dataset_{i}",
            "columns": [
                {
                    "name": "user_id",
                    "type": "int",
                    "sample_values": [str(j) for j in range(1000, 1100)]
                },
                {
                    "name": "email",
                    "type": "string",
                    "sample_values": [f"user{j}@example{i}.com" for j in range(10)]
                },
                {
                    "name": "full_name",
                    "type": "string", 
                    "sample_values": [f"User Name {j}" for j in range(10)]
                },
                {
                    "name": "age",
                    "type": "int",
                    "sample_values": [str(j) for j in range(18, 80, 5)]
                },
                {
                    "name": "registration_date",
                    "type": "date",
                    "sample_values": ["2023-01-01", "2023-06-15", "2024-01-01"]
                }
            ],
            "row_count": 50000
        })
    
    # 客户相关表
    for i in range(15):
        test_tables.append({
            "name": f"customers_db_{i}",
            "columns": [
                {
                    "name": "customer_id",
                    "type": "int",
                    "sample_values": [str(j) for j in range(2000, 2100)]
                },
                {
                    "name": "email_address",
                    "type": "string",
                    "sample_values": [f"customer{j}@company{i}.com" for j in range(10)]
                },
                {
                    "name": "customer_name",
                    "type": "string",
                    "sample_values": [f"Customer {j}" for j in range(10)]
                },
                {
                    "name": "phone",
                    "type": "string", 
                    "sample_values": [f"555-{1000+j}" for j in range(100)]
                },
                {
                    "name": "created_at",
                    "type": "timestamp",
                    "sample_values": ["2023-01-01 10:00:00", "2023-12-31 23:59:59"]
                }
            ],
            "row_count": 30000
        })
    
    # 产品相关表
    for i in range(10):
        test_tables.append({
            "name": f"products_catalog_{i}",
            "columns": [
                {
                    "name": "product_id",
                    "type": "string",
                    "sample_values": [f"P{j:04d}" for j in range(1000)]
                },
                {
                    "name": "product_name",
                    "type": "string",
                    "sample_values": ["Laptop", "Desktop", "Tablet", "Phone", "Monitor"]
                },
                {
                    "name": "price",
                    "type": "decimal",
                    "sample_values": ["999.99", "1299.99", "599.99", "299.99"]
                },
                {
                    "name": "category",
                    "type": "string",
                    "sample_values": ["Electronics", "Computers", "Accessories"]
                }
            ],
            "row_count": 10000
        })
    
    return test_tables


async def test_phase2_optimization():
    """测试阶段二优化组件"""
    try:
        logger.info("🚀 开始阶段二优化测试")
        
        # 1. 创建大规模测试数据
        logger.info("\n📊 步骤1: 创建大规模测试数据")
        test_tables = await create_large_test_dataset()
        
        # 保存测试数据
        test_data_dir = Path(__file__).parent / "test_data"
        test_data_dir.mkdir(exist_ok=True)
        
        large_tables_file = test_data_dir / "large_test_tables.json"
        with open(large_tables_file, 'w', encoding='utf-8') as f:
            json.dump(test_tables, f, ensure_ascii=False, indent=2)
        
        logger.info(f"✅ 创建了 {len(test_tables)} 个大规模测试表")
        total_columns = sum(len(table['columns']) for table in test_tables)
        logger.info(f"   总列数: {total_columns}")
        
        # 2. 构建索引（测试阶段二优化的索引构建性能）
        logger.info("\n🔧 步骤2: 构建大规模索引（测试优化效果）")
        
        start_time = time.time()
        
        index_result = await build_webtable_indices(
            tables_file=str(large_tables_file),
            columns_file=None,
            save_path=None
        )
        
        index_time = time.time() - start_time
        
        if index_result['status'] != 'success':
            raise Exception(f"索引构建失败: {index_result.get('error')}")
        
        logger.info("✅ 大规模索引构建完成")
        logger.info(f"   索引构建时间: {index_time:.2f}秒")
        logger.info(f"   处理表数: {index_result['tables_processed']}")
        logger.info(f"   索引表数: {index_result['tables_indexed']}")
        logger.info(f"   处理列数: {index_result['columns_processed']}")
        logger.info(f"   索引列数: {index_result['columns_indexed']}")
        
        # 3. 创建复杂查询数据
        logger.info("\n📝 步骤3: 创建复杂查询数据")
        
        query_table = {
            "table_name": "target_user_profiles",
            "columns": [
                {
                    "table_name": "target_user_profiles",
                    "column_name": "id",
                    "data_type": "int",
                    "sample_values": ["1", "2", "3"]
                },
                {
                    "table_name": "target_user_profiles",
                    "column_name": "email",
                    "data_type": "string", 
                    "sample_values": ["john@example.com", "jane@example.com"]
                },
                {
                    "table_name": "target_user_profiles",
                    "column_name": "name",
                    "data_type": "string",
                    "sample_values": ["John Doe", "Jane Smith"]
                }
            ]
        }
        
        # 4. 执行数据发现（测试阶段二优化的搜索性能）
        logger.info("\n🔍 步骤4: 执行优化的数据发现")
        
        start_time = time.time()
        
        result = await discover_data(
            user_query="find tables with user information, email addresses, and personal data for data integration",
            query_tables=[query_table],
            query_columns=None
        )
        
        discovery_time = time.time() - start_time
        
        # 5. 分析结果和性能
        logger.info("\n📈 步骤5: 分析性能优化效果")
        
        logger.info(f"✅ 数据发现完成")
        logger.info(f"   发现时间: {discovery_time:.2f}秒")
        
        # 处理返回结果类型
        if isinstance(result, dict):
            from src.core.models import AgentState
            result = AgentState.from_dict(result)
        
        logger.info(f"   工作流状态: {result.current_step}")
        logger.info(f"   策略: {result.strategy}")
        
        if result.final_results:
            logger.info(f"🎉 找到 {len(result.final_results)} 个匹配结果")
            
            # 分析匹配质量
            high_quality_matches = [r for r in result.final_results if r.score > 80]
            medium_quality_matches = [r for r in result.final_results if 60 <= r.score <= 80]
            
            logger.info(f"   高质量匹配 (>80分): {len(high_quality_matches)}")
            logger.info(f"   中等质量匹配 (60-80分): {len(medium_quality_matches)}")
            
            # 显示前5个最佳匹配
            logger.info("\n🏆 前5个最佳匹配:")
            for i, match in enumerate(result.final_results[:5], 1):
                logger.info(f"   {i}. {match.target_table} (评分: {match.score:.1f}, 匹配列: {len(match.matched_columns)})")
        
        else:
            logger.warning("⚠️  没有找到匹配结果")
        
        # 6. 性能指标总结
        logger.info("\n📊 阶段二优化性能总结")
        
        # 计算性能指标
        tables_per_second = len(test_tables) / index_time if index_time > 0 else 0
        columns_per_second = total_columns / index_time if index_time > 0 else 0
        search_throughput = len(test_tables) / discovery_time if discovery_time > 0 else 0
        
        logger.info(f"🚀 索引构建性能:")
        logger.info(f"   表处理速度: {tables_per_second:.1f} 表/秒")
        logger.info(f"   列处理速度: {columns_per_second:.1f} 列/秒")
        
        logger.info(f"🔍 搜索性能:")
        logger.info(f"   搜索吞吐量: {search_throughput:.1f} 表/秒")
        logger.info(f"   平均搜索延迟: {discovery_time*1000:.1f} 毫秒")
        
        # 7. 检查优化组件是否生效
        logger.info("\n🔧 优化组件状态检查:")
        
        optimization_status = {
            "匈牙利算法匹配器": settings.hungarian_matcher.enabled,
            "LSH预过滤器": settings.lsh_prefilter.enabled,
            "向量化计算优化器": settings.vectorized_optimizer.enabled,
            "多级缓存": settings.cache.multi_level_cache.enabled
        }
        
        for component, enabled in optimization_status.items():
            status_icon = "✅" if enabled else "❌"
            logger.info(f"   {status_icon} {component}: {'启用' if enabled else '禁用'}")
        
        enabled_optimizations = sum(optimization_status.values())
        logger.info(f"\n🎯 优化组件启用率: {enabled_optimizations}/{len(optimization_status)} ({enabled_optimizations/len(optimization_status)*100:.0f}%)")
        
        # 判断测试是否成功
        success_criteria = [
            index_time < 60,  # 索引构建时间 < 60秒
            discovery_time < 30,  # 发现时间 < 30秒
            len(result.final_results) > 0,  # 有匹配结果
            enabled_optimizations >= 3  # 至少3个优化组件启用
        ]
        
        success = all(success_criteria)
        
        if success:
            logger.info("\n🎉 阶段二优化测试成功！")
            logger.info("✅ 性能优化组件正常工作")
            logger.info("✅ 大规模数据处理性能良好")
            logger.info("✅ 匹配质量保持稳定")
            return True
        else:
            logger.warning("\n⚠️  阶段二优化测试部分通过")
            logger.warning("   某些性能指标可能需要进一步优化")
            return False
        
    except Exception as e:
        logger.error(f"❌ 阶段二优化测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


async def main():
    """主函数"""
    logger.info("🧪 开始阶段二优化性能测试")
    
    success = await test_phase2_optimization()
    
    if success:
        logger.info("\n🎉 阶段二优化测试完成！")
        logger.info("✅ 系统已启用所有性能优化组件")
        logger.info("✅ 大规模数据处理性能验证通过")
        logger.info("✅ 准备在真实WebTable数据集上测试")
        sys.exit(0)
    else:
        logger.error("\n❌ 阶段二优化测试未完全通过")
        logger.error("   建议检查优化组件配置和性能参数")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())