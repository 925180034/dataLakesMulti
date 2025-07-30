#!/usr/bin/env python3
"""
简化的修复验证测试 - 使用FAISS避免HNSW问题
"""
import asyncio
import json
import logging
import sys
from pathlib import Path

# 设置项目路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.tools.vector_search import get_vector_search_engine
from src.tools.embedding import get_embedding_generator
from src.core.models import TableInfo, ColumnInfo

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')
logger = logging.getLogger(__name__)


async def test_core_fixes():
    """测试核心修复是否成功"""
    try:
        logger.info("🧪 测试核心修复")
        
        # 1. 测试向量搜索引擎初始化
        logger.info("\n📊 步骤1: 测试向量搜索引擎初始化")
        vector_search = get_vector_search_engine()
        
        if vector_search is None:
            raise Exception("向量搜索引擎初始化失败")
        
        logger.info("✅ 向量搜索引擎初始化成功")
        logger.info(f"   引擎类型: {type(vector_search).__name__}")
        
        # 2. 测试嵌入生成器
        logger.info("\n🔧 步骤2: 测试嵌入生成器")
        embedding_gen = get_embedding_generator()
        
        if embedding_gen is None:
            raise Exception("嵌入生成器初始化失败")
        
        logger.info("✅ 嵌入生成器初始化成功")
        
        # 3. 测试简单的数据添加和搜索
        logger.info("\n📝 步骤3: 测试数据添加和搜索")
        
        # 创建测试表
        test_table = TableInfo(
            table_name="test_users",
            columns=[
                ColumnInfo(
                    table_name="test_users",
                    column_name="user_id",
                    data_type="int",
                    sample_values=["1", "2", "3"]
                ),
                ColumnInfo(
                    table_name="test_users", 
                    column_name="email",
                    data_type="string",
                    sample_values=["test@example.com"]
                )
            ]
        )
        
        # 生成嵌入向量
        logger.info("生成表嵌入向量...")
        table_embedding = await embedding_gen.generate_table_embedding(test_table)
        
        if not table_embedding or len(table_embedding) == 0:
            raise Exception("表嵌入向量生成失败")
        
        logger.info(f"✅ 表嵌入向量生成成功，维度: {len(table_embedding)}")
        
        # 添加到向量搜索
        logger.info("添加表到向量搜索索引...")
        await vector_search.add_table_vector(test_table, table_embedding)
        logger.info("✅ 表成功添加到向量搜索索引")
        
        # 测试搜索
        logger.info("测试表搜索...")
        search_results = await vector_search.search_similar_tables(
            query_embedding=table_embedding,
            k=5,
            threshold=0.1
        )
        
        logger.info(f"✅ 搜索完成，找到 {len(search_results)} 个结果")
        
        if search_results:
            for i, result in enumerate(search_results, 1):
                logger.info(f"   结果{i}: {result.item_id} (评分: {result.score:.3f})")
        
        # 4. 测试列级别操作
        logger.info("\n🔍 步骤4: 测试列级别操作")
        
        test_column = test_table.columns[0]  # user_id列
        
        logger.info("生成列嵌入向量...")
        column_embedding = await embedding_gen.generate_column_embedding(test_column)
        
        if not column_embedding or len(column_embedding) == 0:
            raise Exception("列嵌入向量生成失败")
        
        logger.info(f"✅ 列嵌入向量生成成功，维度: {len(column_embedding)}")
        
        # 添加到向量搜索
        logger.info("添加列到向量搜索索引...")
        await vector_search.add_column_vector(test_column, column_embedding)
        logger.info("✅ 列成功添加到向量搜索索引")
        
        # 测试列搜索
        logger.info("测试列搜索...")
        column_search_results = await vector_search.search_similar_columns(
            query_embedding=column_embedding,
            k=5,
            threshold=0.1
        )
        
        logger.info(f"✅ 列搜索完成，找到 {len(column_search_results)} 个结果")
        
        if column_search_results:
            for i, result in enumerate(column_search_results, 1):
                logger.info(f"   结果{i}: {result.item_id} (评分: {result.score:.3f})")
        
        # 5. 检查是否返回真实结果而不是demo数据
        logger.info("\n✅ 步骤5: 验证结果真实性")
        
        # 检查表搜索结果
        real_table_results = [r for r in search_results 
                            if r.item_id not in ['sample_customers', 'sample_products']]
        
        # 检查列搜索结果  
        real_column_results = [r for r in column_search_results
                             if 'sample_' not in r.item_id]
        
        if real_table_results or real_column_results:
            logger.info("🎉 成功！系统返回真实匹配结果，不是演示数据")
            logger.info(f"   真实表结果: {len(real_table_results)}")
            logger.info(f"   真实列结果: {len(real_column_results)}")
            return True
        else:
            logger.warning("⚠️  系统仍在返回演示数据")
            return False
        
    except Exception as e:
        logger.error(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


async def main():
    """主函数"""
    logger.info("🧪 开始简化的修复验证测试")
    
    success = await test_core_fixes()
    
    if success:
        logger.info("\n🎉 测试成功！核心修复完成")
        logger.info("✅ 修复内容:")
        logger.info("   - 向量搜索引擎初始化问题已修复")
        logger.info("   - 嵌入向量生成正常工作")
        logger.info("   - 数据索引添加和搜索功能正常")
        logger.info("   - 系统能够处理真实数据而不是演示数据")
        sys.exit(0)
    else:
        logger.error("\n❌ 测试失败！需要进一步调试")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())