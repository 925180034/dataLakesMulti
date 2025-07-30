#!/usr/bin/env python3
"""
集成测试脚本 - 测试混合相似度与现有系统的集成
"""

import asyncio
import logging
from src.core.models import ColumnInfo, TableInfo, AgentState
from src.agents.column_discovery import ColumnDiscoveryAgent
from src.tools.hybrid_similarity import hybrid_similarity_engine

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_sample_data():
    """创建样本数据用于测试"""
    
    # 创建查询表（用户想要匹配的表）
    query_table = TableInfo(
        table_name="customers",
        columns=[
            ColumnInfo(
                table_name="customers",
                column_name="customer_id",
                data_type="int",
                sample_values=[1, 2, 3, 4, 5, 10, 15, 20]
            ),
            ColumnInfo(
                table_name="customers", 
                column_name="customer_name",
                data_type="string",
                sample_values=["John Doe", "Jane Smith", "Bob Johnson", "Alice Brown"]
            ),
            ColumnInfo(
                table_name="customers",
                column_name="email",
                data_type="string", 
                sample_values=["john@email.com", "jane@email.com", "bob@email.com"]
            )
        ]
    )
    
    # 创建数据湖中的候选表（要搜索的数据）
    candidate_tables = [
        TableInfo(
            table_name="users",
            columns=[
                ColumnInfo(
                    table_name="users",
                    column_name="user_id", 
                    data_type="integer",
                    sample_values=[10, 20, 30, 40, 50, 100, 150, 200]
                ),
                ColumnInfo(
                    table_name="users",
                    column_name="full_name",
                    data_type="varchar",
                    sample_values=["Alice Johnson", "Charlie Brown", "David Wilson", "Emma Davis"]
                ),
                ColumnInfo(
                    table_name="users",
                    column_name="email_address",
                    data_type="varchar", 
                    sample_values=["alice@test.com", "charlie@test.com", "david@test.com"]
                )
            ]
        ),
        TableInfo(
            table_name="orders",
            columns=[
                ColumnInfo(
                    table_name="orders",
                    column_name="order_id",
                    data_type="int",
                    sample_values=[1001, 1002, 1003, 1004]
                ),
                ColumnInfo(
                    table_name="orders", 
                    column_name="customer_ref",
                    data_type="int",
                    sample_values=[1, 2, 3, 4, 5]
                ),
                ColumnInfo(
                    table_name="orders",
                    column_name="order_date",
                    data_type="date",
                    sample_values=["2023-01-01", "2023-01-02", "2023-01-03"]
                )
            ]
        ),
        TableInfo(
            table_name="products",
            columns=[
                ColumnInfo(
                    table_name="products",
                    column_name="product_id",
                    data_type="int", 
                    sample_values=[101, 102, 103, 104]
                ),
                ColumnInfo(
                    table_name="products",
                    column_name="product_name",
                    data_type="string",
                    sample_values=["iPhone 12", "Samsung Galaxy", "Google Pixel"]
                ),
                ColumnInfo(
                    table_name="products",
                    column_name="price", 
                    data_type="decimal",
                    sample_values=[999.99, 899.99, 799.99, 699.99]
                )
            ]
        )
    ]
    
    return query_table, candidate_tables


async def test_column_discovery_with_hybrid_similarity():
    """测试ColumnDiscoveryAgent与混合相似度的集成"""
    logger.info("=== 测试列发现智能体与混合相似度集成 ===")
    
    try:
        # 创建测试数据
        query_table, candidate_tables = create_sample_data()
        
        # 创建ColumnDiscoveryAgent实例
        agent = ColumnDiscoveryAgent()
        
        # 准备候选列数据（模拟已索引的数据）
        all_candidate_columns = []
        for table in candidate_tables:
            all_candidate_columns.extend(table.columns)
        
        # 初始化索引（模拟实际系统中的索引构建过程）
        logger.info("初始化搜索索引...")
        await agent.initialize_indices(all_candidate_columns)
        
        # 创建智能体状态
        state = AgentState(
            user_query="查找与customers表结构相似的表",
            query_columns=query_table.columns,
            strategy="BOTTOM_UP"
        )
        
        # 执行列发现
        logger.info("执行列发现...")
        result_state = await agent.process(state)
        
        # 分析结果
        logger.info(f"发现 {len(result_state.column_matches)} 个列匹配")
        
        # 按查询列分组显示结果
        matches_by_query = {}
        for match in result_state.column_matches:
            query_col = match.source_column
            if query_col not in matches_by_query:
                matches_by_query[query_col] = []
            matches_by_query[query_col].append(match)
        
        for query_col, matches in matches_by_query.items():
            logger.info(f"\n查询列: {query_col}")
            logger.info(f"找到 {len(matches)} 个匹配:")
            
            for match in matches[:3]:  # 显示前3个最佳匹配
                logger.info(f"  -> {match.target_column}")
                logger.info(f"     置信度: {match.confidence:.3f}")
                logger.info(f"     类型: {match.match_type}")
                logger.info(f"     原因: {match.reason}")
                logger.info(f"     ---")
        
        return result_state
        
    except Exception as e:
        logger.error(f"集成测试失败: {e}")
        raise


def test_direct_similarity_comparison():
    """测试直接相似度比较"""
    logger.info("=== 直接相似度比较测试 ===")
    
    query_table, candidate_tables = create_sample_data()
    
    # 取第一个查询列进行测试
    query_column = query_table.columns[0]  # customer_id
    logger.info(f"查询列: {query_column.full_name}")
    
    # 与所有候选列进行比较
    all_candidates = []
    for table in candidate_tables:
        all_candidates.extend(table.columns)
    
    results = []
    for candidate in all_candidates:
        # SMD场景比较
        smd_result = hybrid_similarity_engine.calculate_column_similarity(
            query_column, candidate, "SMD"
        )
        
        # SLD场景比较  
        sld_result = hybrid_similarity_engine.calculate_column_similarity(
            query_column, candidate, "SLD"
        )
        
        results.append({
            'target': candidate.full_name,
            'smd_score': smd_result['combined_similarity'],
            'sld_score': sld_result['combined_similarity'],
            'name_sim': smd_result['name_similarity'],
            'struct_sim': smd_result['structural_similarity'],
            'semantic_sim': sld_result['semantic_similarity']
        })
    
    # 按SLD分数排序
    results.sort(key=lambda x: x['sld_score'], reverse=True)
    
    logger.info("相似度排名:")
    for i, result in enumerate(results, 1):
        logger.info(f"{i}. {result['target']}")
        logger.info(f"   SMD: {result['smd_score']:.3f}, SLD: {result['sld_score']:.3f}")
        logger.info(f"   [名称: {result['name_sim']:.3f}, 结构: {result['struct_sim']:.3f}, 语义: {result['semantic_sim']:.3f}]")


async def test_scenario_comparison():
    """测试不同场景的性能差异"""
    logger.info("=== 场景性能比较 ===")
    
    # 创建有完整数据的列
    col_with_data = ColumnInfo(
        table_name="test", 
        column_name="user_id",
        data_type="int",
        sample_values=[1, 2, 3, 4, 5, 10, 15, 20, 25, 30]
    )
    
    # 创建仅有元数据的列
    col_metadata_only = ColumnInfo(
        table_name="test",
        column_name="user_id", 
        data_type="int",
        sample_values=[]
    )
    
    # 目标列
    target_col = ColumnInfo(
        table_name="target",
        column_name="customer_id",
        data_type="integer", 
        sample_values=[2, 4, 6, 8, 10, 12, 16, 24, 28, 32]
    )
    
    # 测试SMD场景
    smd_result = hybrid_similarity_engine.calculate_column_similarity(
        col_metadata_only, target_col, "SMD"
    )
    
    # 测试SLD场景
    sld_result = hybrid_similarity_engine.calculate_column_similarity(
        col_with_data, target_col, "SLD"
    )
    
    logger.info("场景对比:")
    logger.info(f"SMD (仅元数据): {smd_result['combined_similarity']:.3f}")
    logger.info(f"  权重: {smd_result['weights_used']}")
    logger.info(f"SLD (完整数据): {sld_result['combined_similarity']:.3f}")
    logger.info(f"  权重: {sld_result['weights_used']}")
    
    logger.info(f"语义特征提升: {sld_result['semantic_similarity']:.3f}")


async def main():
    """主测试函数"""
    logger.info("开始混合相似度集成测试")
    
    try:
        # 直接相似度比较
        test_direct_similarity_comparison()
        
        # 场景比较
        await test_scenario_comparison()
        
        # 集成测试（需要更多系统组件）
        # await test_column_discovery_with_hybrid_similarity()
        
        logger.info("集成测试完成！")
        
    except Exception as e:
        logger.error(f"测试失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())