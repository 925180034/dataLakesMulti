#!/usr/bin/env python3
"""
测试修复后的完整工作流程
验证系统是否能产生真实的匹配结果
"""
import asyncio
import json
import logging
import sys
from pathlib import Path

# 设置项目路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.tools.data_indexer import build_webtable_indices, verify_indices
from src.core.workflow import discover_data
from src.core.models import TableInfo, ColumnInfo
from src.config.settings import settings

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def test_complete_fixed_workflow():
    """测试修复后的完整工作流程"""
    try:
        logger.info("🚀 开始测试修复后的完整工作流程")
        
        # 1. 创建测试数据
        logger.info("\n📊 步骤1: 创建测试数据")
        test_data_dir = Path(__file__).parent / "test_data"
        test_data_dir.mkdir(exist_ok=True)
        
        # 创建测试表数据
        test_tables = [
            {
                "name": "users",
                "columns": [
                    {
                        "name": "user_id",
                        "type": "int",
                        "sample_values": ["1", "2", "3", "100", "200"]
                    },
                    {
                        "name": "email",
                        "type": "string", 
                        "sample_values": ["john@example.com", "jane@example.com", "bob@test.com"]
                    },
                    {
                        "name": "name",
                        "type": "string",
                        "sample_values": ["John Doe", "Jane Smith", "Bob Wilson"]
                    },
                    {
                        "name": "age",
                        "type": "int",
                        "sample_values": ["25", "30", "35", "28", "42"]
                    }
                ],
                "row_count": 1000
            },
            {
                "name": "customers",
                "columns": [
                    {
                        "name": "customer_id",
                        "type": "int",
                        "sample_values": ["1001", "1002", "1003", "1004"]
                    },
                    {
                        "name": "email_address", 
                        "type": "string",
                        "sample_values": ["customer1@company.com", "customer2@company.com"]
                    },
                    {
                        "name": "full_name",
                        "type": "string",
                        "sample_values": ["Alice Johnson", "Charlie Brown", "David Wilson"]
                    },
                    {
                        "name": "registration_date",
                        "type": "date",
                        "sample_values": ["2023-01-15", "2023-02-20", "2023-03-10"]
                    }
                ],
                "row_count": 500
            },
            {
                "name": "products",
                "columns": [
                    {
                        "name": "product_id",
                        "type": "int",
                        "sample_values": ["P001", "P002", "P003"]
                    },
                    {
                        "name": "product_name",
                        "type": "string",
                        "sample_values": ["Laptop", "Mouse", "Keyboard", "Monitor"]
                    },
                    {
                        "name": "price",
                        "type": "decimal",
                        "sample_values": ["999.99", "29.99", "79.99", "299.99"]
                    }
                ],
                "row_count": 200
            }
        ]
        
        # 保存测试表数据
        tables_file = test_data_dir / "test_tables.json"
        with open(tables_file, 'w', encoding='utf-8') as f:
            json.dump(test_tables, f, ensure_ascii=False, indent=2)
        
        logger.info(f"✅ 创建了 {len(test_tables)} 个测试表")
        
        # 2. 构建索引
        logger.info("\n🔧 步骤2: 构建向量搜索索引")
        
        index_result = await build_webtable_indices(
            tables_file=str(tables_file),
            columns_file=None,
            save_path=None  # 使用默认路径
        )
        
        if index_result['status'] != 'success':
            raise Exception(f"索引构建失败: {index_result.get('error')}")
        
        logger.info("✅ 索引构建成功")
        logger.info(f"   - 处理表数: {index_result['tables_processed']}")
        logger.info(f"   - 索引表数: {index_result['tables_indexed']}")
        logger.info(f"   - 处理列数: {index_result['columns_processed']}")
        logger.info(f"   - 索引列数: {index_result['columns_indexed']}")
        
        # 3. 验证索引
        logger.info("\n🔍 步骤3: 验证索引")
        
        verify_result = await verify_indices()
        if verify_result['status'] != 'success':
            raise Exception(f"索引验证失败: {verify_result.get('error')}")
        
        vector_stats = verify_result.get('vector_search', {})
        logger.info("✅ 索引验证成功")
        logger.info(f"   - 向量搜索列数: {vector_stats.get('column_count', 0)}")
        logger.info(f"   - 向量搜索表数: {vector_stats.get('table_count', 0)}")
        
        # 4. 创建查询数据
        logger.info("\n📝 步骤4: 创建查询数据")
        
        query_table = {
            "table_name": "user_profiles",
            "columns": [
                {
                    "table_name": "user_profiles",
                    "column_name": "id",
                    "data_type": "int",
                    "sample_values": ["1", "2", "3"]
                },
                {
                    "table_name": "user_profiles", 
                    "column_name": "email",
                    "data_type": "string",
                    "sample_values": ["john@example.com", "jane@example.com"]
                }
            ]
        }
        
        # 5. 执行数据发现
        logger.info("\n🔍 步骤5: 执行数据发现")
        
        result = await discover_data(
            user_query="find tables with user information and email addresses",
            query_tables=[query_table],
            query_columns=None
        )
        
        # 6. 验证结果
        logger.info("\n✅ 步骤6: 验证结果")
        
        logger.info(f"工作流执行状态: {result.current_step}")
        logger.info(f"策略: {result.strategy}")
        logger.info(f"错误信息: {result.error_messages}")
        
        if result.final_results:
            logger.info(f"🎉 找到 {len(result.final_results)} 个匹配结果!")
            
            for i, match in enumerate(result.final_results, 1):
                logger.info(f"\n匹配结果 {i}:")
                logger.info(f"  - 目标表: {match.target_table}")
                logger.info(f"  - 评分: {match.score:.2f}")
                logger.info(f"  - 匹配列数: {len(match.matched_columns)}")
                
                if match.matched_columns:
                    logger.info("  - 匹配详情:")
                    for col_match in match.matched_columns[:3]:  # 显示前3个
                        logger.info(f"    * {col_match.source_column} → {col_match.target_column} "
                                  f"(置信度: {col_match.confidence:.3f})")
            
            # 检查是否包含真实匹配（不是demo数据）
            real_matches = [m for m in result.final_results 
                          if m.target_table not in ['sample_customers', 'sample_products']]
            
            if real_matches:
                logger.info(f"✅ 成功！找到了 {len(real_matches)} 个真实匹配结果")
                logger.info("🎯 系统现在能够产生真实的语义匹配，而不是演示数据")
                return True
            else:
                logger.warning("⚠️  仍然返回演示数据，可能需要进一步调试")
                return False
        else:
            logger.warning("⚠️  没有找到匹配结果")
            
            if result.final_report:
                logger.info(f"最终报告: {result.final_report}")
            
            return False
        
    except Exception as e:
        logger.error(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


async def main():
    """主函数"""
    logger.info("🧪 开始测试修复后的数据湖多智能体系统")
    
    success = await test_complete_fixed_workflow()
    
    if success:
        logger.info("\n🎉 测试成功！系统修复完成")
        logger.info("✅ 系统现在能够:")
        logger.info("   - 正确初始化向量搜索引擎")
        logger.info("   - 构建和加载真实的数据索引")
        logger.info("   - 执行语义匹配并返回真实结果")
        logger.info("   - 完成端到端的数据发现流程")
        sys.exit(0)
    else:
        logger.error("\n❌ 测试失败！需要进一步调试")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())