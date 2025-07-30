#!/usr/bin/env python3
"""
混合相似度计算测试脚本
"""

import logging
from src.core.models import ColumnInfo, TableInfo
from src.tools.hybrid_similarity import hybrid_similarity_engine

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_test_columns():
    """创建测试列数据"""
    
    # 测试用例1：相似的数值列
    col1_numeric = ColumnInfo(
        table_name="customers",
        column_name="customer_id", 
        data_type="int",
        sample_values=[1, 2, 3, 4, 5, 10, 15, 20]
    )
    
    col2_numeric = ColumnInfo(
        table_name="orders",
        column_name="cust_id",
        data_type="integer", 
        sample_values=[2, 4, 6, 8, 12, 18, 25, 30]
    )
    
    # 测试用例2：相似的文本列
    col1_text = ColumnInfo(
        table_name="products",
        column_name="product_name",
        data_type="string",
        sample_values=["iPhone 12", "Samsung Galaxy", "Google Pixel", "OnePlus 9", "Xiaomi Mi"]
    )
    
    col2_text = ColumnInfo(
        table_name="inventory", 
        column_name="productName",
        data_type="varchar",
        sample_values=["iPhone 13", "Samsung Note", "Google Pixel", "OnePlus 8", "Xiaomi Redmi"]
    )
    
    # 测试用例3：不相似的列
    col3_different = ColumnInfo(
        table_name="employees",
        column_name="salary",
        data_type="decimal",
        sample_values=[50000, 60000, 75000, 80000, 90000]
    )
    
    # 测试用例4：仅元数据的列（SMD场景）
    col1_metadata_only = ColumnInfo(
        table_name="users",
        column_name="user_email",
        data_type="string",
        sample_values=[]  # 无样本数据
    )
    
    col2_metadata_only = ColumnInfo(
        table_name="accounts",
        column_name="email_address", 
        data_type="varchar",
        sample_values=[]  # 无样本数据
    )
    
    return [
        (col1_numeric, col2_numeric, "数值列匹配"),
        (col1_text, col2_text, "文本列匹配"),
        (col1_numeric, col3_different, "不相关列"),
        (col1_metadata_only, col2_metadata_only, "仅元数据匹配")
    ]


def test_text_similarity():
    """测试文本相似度计算"""
    logger.info("=== 测试文本相似度计算 ===")
    
    text_calc = hybrid_similarity_engine.text_calculator
    
    test_cases = [
        ("customer_id", "cust_id", "预期高相似度"),
        ("product_name", "productName", "驼峰vs下划线"),
        ("user_email", "email_address", "相关概念"),
        ("order_date", "salary", "不相关概念")
    ]
    
    for name1, name2, description in test_cases:
        similarity = text_calc.calculate_name_similarity(name1, name2)
        logger.info(f"{description}: '{name1}' vs '{name2}' = {similarity:.3f}")


def test_statistical_similarity():
    """测试统计相似度计算"""
    logger.info("=== 测试统计相似度计算 ===")
    
    stat_calc = hybrid_similarity_engine.statistical_calculator
    
    # 数值相似度测试
    values1 = [1, 2, 3, 4, 5, 10, 15, 20]
    values2 = [2, 4, 6, 8, 12, 18, 25, 30]
    values3 = [100, 200, 500, 1000, 5000]
    
    numeric_sim1 = stat_calc.calculate_numeric_similarity(values1, values2)
    numeric_sim2 = stat_calc.calculate_numeric_similarity(values1, values3)
    
    logger.info(f"相似数值分布: {numeric_sim1:.3f}")
    logger.info(f"不同数值分布: {numeric_sim2:.3f}")
    
    # 分类相似度测试
    cat_values1 = ["iPhone", "Samsung", "Google", "OnePlus", "Xiaomi"]
    cat_values2 = ["iPhone", "Samsung", "Google", "Huawei", "Sony"] 
    cat_values3 = ["Red", "Blue", "Green", "Yellow", "Purple"]
    
    cat_sim1 = stat_calc.calculate_categorical_similarity(cat_values1, cat_values2)
    cat_sim2 = stat_calc.calculate_categorical_similarity(cat_values1, cat_values3)
    
    logger.info(f"相似分类分布: {cat_sim1:.3f}")
    logger.info(f"不同分类分布: {cat_sim2:.3f}")


def test_hybrid_similarity():
    """测试混合相似度计算"""
    logger.info("=== 测试混合相似度计算 ===")
    
    test_columns = create_test_columns()
    
    for col1, col2, description in test_columns:
        # SMD场景测试
        smd_result = hybrid_similarity_engine.calculate_column_similarity(col1, col2, "SMD")
        logger.info(f"{description} (SMD场景):")
        logger.info(f"  组合相似度: {smd_result['combined_similarity']:.3f}")
        logger.info(f"  名称相似度: {smd_result['name_similarity']:.3f}")
        logger.info(f"  结构相似度: {smd_result['structural_similarity']:.3f}")
        
        # SLD场景测试（如果有样本数据）
        if col1.sample_values and col2.sample_values:
            sld_result = hybrid_similarity_engine.calculate_column_similarity(col1, col2, "SLD")
            logger.info(f"{description} (SLD场景):")
            logger.info(f"  组合相似度: {sld_result['combined_similarity']:.3f}")
            logger.info(f"  名称相似度: {sld_result['name_similarity']:.3f}")
            logger.info(f"  结构相似度: {sld_result['structural_similarity']:.3f}")
            logger.info(f"  语义相似度: {sld_result['semantic_similarity']:.3f}")
        
        logger.info("-" * 50)


def test_table_similarity():
    """测试表相似度计算"""
    logger.info("=== 测试表相似度计算 ===")
    
    # 创建测试表
    table1 = TableInfo(
        table_name="customers",
        columns=[
            ColumnInfo(table_name="customers", column_name="customer_id", data_type="int", sample_values=[1, 2, 3]),
            ColumnInfo(table_name="customers", column_name="name", data_type="string", sample_values=["John", "Jane", "Bob"]),
            ColumnInfo(table_name="customers", column_name="email", data_type="string", sample_values=["john@email.com", "jane@email.com"])
        ]
    )
    
    table2 = TableInfo(
        table_name="users", 
        columns=[
            ColumnInfo(table_name="users", column_name="user_id", data_type="integer", sample_values=[10, 20, 30]),
            ColumnInfo(table_name="users", column_name="full_name", data_type="varchar", sample_values=["Alice", "Charlie", "David"]),
            ColumnInfo(table_name="users", column_name="email_address", data_type="varchar", sample_values=["alice@email.com", "charlie@email.com"])
        ]
    )
    
    # 计算表相似度
    smd_result = hybrid_similarity_engine.calculate_table_similarity(table1, table2, "SMD")
    sld_result = hybrid_similarity_engine.calculate_table_similarity(table1, table2, "SLD")
    
    logger.info("表相似度 (SMD场景):")
    logger.info(f"  表相似度: {smd_result['table_similarity']:.3f}")
    logger.info(f"  匹配列数: {smd_result['matched_count']}")
    logger.info(f"  匹配比例: {smd_result['match_ratio']:.3f}")
    
    logger.info("表相似度 (SLD场景):")
    logger.info(f"  表相似度: {sld_result['table_similarity']:.3f}")
    logger.info(f"  匹配列数: {sld_result['matched_count']}")
    logger.info(f"  匹配比例: {sld_result['match_ratio']:.3f}")
    
    logger.info("匹配详情:")
    for match in sld_result['matched_columns']:
        logger.info(f"  {match['source_column']} -> {match['target_column']} ({match['similarity']:.3f})")


def main():
    """主测试函数"""
    logger.info("开始混合相似度计算测试")
    
    try:
        # 测试各个组件
        test_text_similarity()
        test_statistical_similarity()  
        test_hybrid_similarity()
        test_table_similarity()
        
        logger.info("测试完成！")
        
    except Exception as e:
        logger.error(f"测试失败: {e}")
        raise


if __name__ == "__main__":
    main()