#!/usr/bin/env python3
"""
可扩展搜索引擎测试脚本 - 验证分层索引性能提升
"""

import time
import logging
import random
from typing import List, Dict
from src.core.models import ColumnInfo, TableInfo
from src.tools.scalable_search import (
    HierarchicalVectorSearch, 
    MetadataIndex, 
    TableSignature,
    hierarchical_search_engine
)
from src.tools.query_preprocessor import query_preprocessor, smart_prefilter

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def generate_mock_tables(num_tables: int = 1000) -> List[TableInfo]:
    """生成模拟表数据用于测试"""
    logger.info(f"生成 {num_tables} 个模拟表...")
    
    # 表名模板和领域
    table_templates = {
        'customer': ['customers', 'users', 'clients', 'accounts', 'members'],
        'order': ['orders', 'purchases', 'transactions', 'sales', 'invoices'],
        'product': ['products', 'items', 'inventory', 'catalog', 'goods'],
        'employee': ['employees', 'staff', 'workers', 'personnel', 'team'],
        'financial': ['payments', 'billing', 'revenue', 'expenses', 'budget']
    }
    
    # 列名模板
    column_templates = {
        'identifier': ['id', 'uuid', 'key', 'reference', 'code'],
        'name': ['name', 'title', 'label', 'description'],
        'contact': ['email', 'phone', 'address', 'contact'],
        'temporal': ['created_at', 'updated_at', 'date', 'timestamp'],
        'financial': ['price', 'cost', 'amount', 'value', 'total'],
        'status': ['status', 'state', 'active', 'enabled']
    }
    
    data_types = ['int', 'varchar', 'text', 'decimal', 'datetime', 'boolean']
    
    tables = []
    
    for i in range(num_tables):
        # 随机选择领域和表名
        domain = random.choice(list(table_templates.keys()))
        base_name = random.choice(table_templates[domain])
        
        # 生成唯一表名
        table_name = f"{base_name}_{i // 100}_{i % 100}"
        
        # 生成列数（3-20列）
        num_columns = random.randint(3, 20)
        columns = []
        
        # 总是包含一个ID列
        id_col = ColumnInfo(
            table_name=table_name,
            column_name=f"{base_name[:-1]}_id",
            data_type="int",
            sample_values=[j for j in range(1, 11)]
        )
        columns.append(id_col)
        
        # 生成其他列
        for j in range(num_columns - 1):
            col_category = random.choice(list(column_templates.keys()))
            col_base_name = random.choice(column_templates[col_category])
            
            # 添加随机后缀避免重复
            col_name = f"{col_base_name}_{j}" if j > 0 else col_base_name
            
            col = ColumnInfo(
                table_name=table_name,
                column_name=col_name,
                data_type=random.choice(data_types),
                sample_values=[f"sample_{j}_{k}" for k in range(5)]
            )
            columns.append(col)
        
        # 创建表
        table = TableInfo(
            table_name=table_name,
            columns=columns,
            row_count=random.randint(100, 100000),
            description=f"Mock table for {domain} domain"
        )
        
        tables.append(table)
    
    logger.info(f"生成模拟表完成: {len(tables)} 个表")
    return tables


def generate_mock_embeddings(tables: List[TableInfo]) -> Dict[str, Dict]:
    """生成模拟嵌入向量"""
    logger.info("生成模拟嵌入向量...")
    
    dimension = 1536
    embeddings = {}
    
    for table in tables:
        # 表级嵌入
        table_embedding = [random.random() for _ in range(dimension)]
        
        # 列级嵌入
        column_embeddings = []
        for col in table.columns:
            col_embedding = [random.random() for _ in range(dimension)]
            column_embeddings.append(col_embedding)
        
        embeddings[table.table_name] = {
            'table_embedding': table_embedding,
            'column_embeddings': column_embeddings
        }
    
    logger.info(f"嵌入向量生成完成: {len(embeddings)} 个表")
    return embeddings


def test_table_signature_similarity():
    """测试表签名相似度计算"""
    logger.info("=== 测试表签名相似度计算 ===")
    
    # 创建测试表
    table1 = TableInfo(
        table_name="customers",
        columns=[
            ColumnInfo(table_name="customers", column_name="customer_id", data_type="int"),
            ColumnInfo(table_name="customers", column_name="customer_name", data_type="varchar"),
            ColumnInfo(table_name="customers", column_name="email", data_type="varchar"),
            ColumnInfo(table_name="customers", column_name="created_at", data_type="datetime")
        ]
    )
    
    table2 = TableInfo(
        table_name="users",
        columns=[
            ColumnInfo(table_name="users", column_name="user_id", data_type="int"),
            ColumnInfo(table_name="users", column_name="full_name", data_type="varchar"),
            ColumnInfo(table_name="users", column_name="email_address", data_type="varchar"),
            ColumnInfo(table_name="users", column_name="registration_date", data_type="datetime")
        ]
    )
    
    table3 = TableInfo(
        table_name="products",
        columns=[
            ColumnInfo(table_name="products", column_name="product_id", data_type="int"),
            ColumnInfo(table_name="products", column_name="product_name", data_type="varchar"),
            ColumnInfo(table_name="products", column_name="price", data_type="decimal")
        ]
    )
    
    # 生成签名
    sig1 = TableSignature(table1)
    sig2 = TableSignature(table2)
    sig3 = TableSignature(table3)
    
    # 计算相似度
    sim_1_2 = sig1.calculate_similarity(sig2)
    sim_1_3 = sig1.calculate_similarity(sig3)
    sim_2_3 = sig2.calculate_similarity(sig3)
    
    logger.info(f"customers vs users: {sim_1_2:.3f}")
    logger.info(f"customers vs products: {sim_1_3:.3f}")
    logger.info(f"users vs products: {sim_2_3:.3f}")
    
    logger.info(f"表签名测试完成")


def test_metadata_index():
    """测试元数据索引"""
    logger.info("=== 测试元数据索引 ===")
    
    # 创建小规模测试数据
    test_tables = generate_mock_tables(100)
    
    # 初始化元数据索引
    metadata_index = MetadataIndex()
    
    # 添加表到索引
    start_time = time.time()
    for i, table in enumerate(test_tables):
        table_id = f"table_{i}"
        metadata_index.add_table(table_id, table)
    
    index_time = time.time() - start_time
    logger.info(f"索引构建时间: {index_time:.3f}s")
    
    # 测试预筛选
    test_queries = [
        {'query_keywords': {'customer', 'user'}, 'target_column_count': 5},
        {'query_keywords': {'order', 'purchase'}, 'target_column_count': 8},
        {'query_keywords': set(), 'target_column_count': 12}
    ]
    
    for i, query in enumerate(test_queries):
        start_time = time.time()
        candidates = metadata_index.prefilter_tables(**query)
        filter_time = time.time() - start_time
        
        logger.info(f"查询 {i+1}: 筛选出 {len(candidates)} 个候选表 (耗时: {filter_time*1000:.1f}ms)")
    
    logger.info("元数据索引测试完成")


def test_query_preprocessing():
    """测试查询预处理"""
    logger.info("=== 测试查询预处理 ===")
    
    test_queries = [
        "find tables that can join with customer data",
        "search for similar product inventory tables",
        "analyze order transaction patterns",
        "discover user behavior tracking tables"
    ]
    
    # 创建查询表
    query_table = TableInfo(
        table_name="customer_orders",
        columns=[
            ColumnInfo(table_name="customer_orders", column_name="customer_id", data_type="int"),
            ColumnInfo(table_name="customer_orders", column_name="order_date", data_type="datetime"),
            ColumnInfo(table_name="customer_orders", column_name="total_amount", data_type="decimal")
        ]
    )
    
    for query in test_queries:
        start_time = time.time()
        analysis = query_preprocessor.analyze_query(query, [query_table])
        process_time = time.time() - start_time
        
        logger.info(f"查询: '{query}'")
        logger.info(f"  意图: {analysis['operation_intent']}")
        logger.info(f"  领域: {analysis['domain_hints']}")
        logger.info(f"  复杂度: {analysis['complexity_score']:.2f}")
        logger.info(f"  处理时间: {process_time*1000:.1f}ms")
        
        # 生成搜索策略
        strategy = smart_prefilter.generate_search_strategy(query, [query_table])
        logger.info(f"  搜索策略: 分层搜索={strategy['use_hierarchical_search']}")
        logger.info("---")
    
    logger.info("查询预处理测试完成")


async def test_hierarchical_search_performance():
    """测试分层搜索性能"""
    logger.info("=== 测试分层搜索性能 ===")
    
    # 生成测试数据
    num_tables = 500  # 中等规模测试
    test_tables = generate_mock_tables(num_tables)
    embeddings = generate_mock_embeddings(test_tables)
    
    # 初始化分层搜索引擎
    search_engine = HierarchicalVectorSearch()
    
    # 批量添加数据
    logger.info("添加数据到分层索引...")
    start_time = time.time()
    
    for table in test_tables:
        table_data = embeddings[table.table_name]
        table_id = await search_engine.add_table_with_columns(
            table,
            table_data['table_embedding'],
            table_data['column_embeddings']
        )
    
    build_time = time.time() - start_time
    logger.info(f"索引构建时间: {build_time:.2f}s ({num_tables} 个表)")
    
    # 性能测试查询
    query_embedding = [random.random() for _ in range(1536)]
    
    # 测试表级搜索
    logger.info("测试表级搜索性能...")
    table_search_times = []
    
    for i in range(10):
        start_time = time.time()
        results = await search_engine.hierarchical_search_tables(
            query_embedding,
            query_keywords={'customer', 'user'},
            target_column_count=5,
            k=20
        )
        search_time = time.time() - start_time
        table_search_times.append(search_time)
        
        if i == 0:
            logger.info(f"  首次搜索返回 {len(results)} 个结果")
    
    avg_table_time = sum(table_search_times) / len(table_search_times)
    logger.info(f"表级搜索平均时间: {avg_table_time*1000:.1f}ms")
    
    # 测试列级搜索
    logger.info("测试列级搜索性能...")
    column_search_times = []
    
    # 获取候选表ID
    candidate_tables = [f"table_{i}" for i in range(min(50, num_tables))]
    
    for i in range(10):
        start_time = time.time()
        results = await search_engine.hierarchical_search_columns(
            query_embedding,
            candidate_table_ids=candidate_tables[:20],
            k=30
        )
        search_time = time.time() - start_time
        column_search_times.append(search_time)
        
        if i == 0:
            logger.info(f"  首次搜索返回 {len(results)} 个结果")
    
    avg_column_time = sum(column_search_times) / len(column_search_times)
    logger.info(f"列级搜索平均时间: {avg_column_time*1000:.1f}ms")
    
    # 性能总结
    logger.info("=== 性能总结 ===")
    logger.info(f"数据规模: {num_tables} 个表")
    logger.info(f"索引构建: {build_time:.2f}s")
    logger.info(f"表级搜索: {avg_table_time*1000:.1f}ms")
    logger.info(f"列级搜索: {avg_column_time*1000:.1f}ms")


async def main():
    """主测试函数"""
    logger.info("开始可扩展搜索引擎测试")
    
    try:
        # 基础功能测试
        test_table_signature_similarity()
        test_metadata_index()
        test_query_preprocessing()
        
        # 性能测试
        await test_hierarchical_search_performance()
        
        logger.info("所有测试完成！")
        
    except Exception as e:
        logger.error(f"测试失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())