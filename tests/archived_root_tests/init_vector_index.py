#!/usr/bin/env python
"""
初始化向量索引
确保系统在运行查询之前有必要的索引文件
"""

import asyncio
import json
from pathlib import Path
import logging
from typing import List

from src.config.settings import settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def initialize_vector_index(tables_file: str, force_rebuild: bool = False):
    """初始化向量索引
    
    Args:
        tables_file: 表数据文件路径
        force_rebuild: 是否强制重建索引
    """
    logger.info("🚀 开始初始化向量索引")
    
    # 检查索引目录
    vector_db_path = Path(settings.vector_db.db_path)
    index_db_path = Path(settings.index.index_path)
    
    # 检查是否已有索引
    if not force_rebuild:
        if vector_db_path.exists() and any(vector_db_path.iterdir()):
            logger.info("✅ 向量索引已存在，跳过初始化")
            logger.info("   如需重建，请使用 --force 参数")
            return
    
    # 使用便捷函数构建索引
    logger.info(f"📊 从文件加载表数据: {tables_file}")
    from src.tools.data_indexer import build_webtable_indices
    
    result = await build_webtable_indices(
        tables_file=tables_file,
        columns_file=None,
        save_path=None
    )
    
    if result.get("status") == "success":
        logger.info("✅ 向量索引初始化成功!")
        logger.info(f"   表索引: {result.get('tables_indexed', 0)} 个")
        logger.info(f"   列索引: {result.get('columns_indexed', 0)} 个")
        logger.info(f"   索引路径: {result.get('index_path', 'N/A')}")
    else:
        logger.error(f"❌ 向量索引初始化失败: {result.get('error', 'Unknown error')}")
        raise Exception("索引初始化失败")


async def verify_index():
    """验证索引是否可用"""
    try:
        from src.tools.vector_search import get_vector_search_engine
        from src.tools.embedding import get_embedding_generator
        
        logger.info("\n🔍 验证向量索引...")
        vector_search = get_vector_search_engine()
        embedding_gen = get_embedding_generator()
        
        # 生成测试查询的嵌入向量
        test_text = "test table with columns"
        test_embedding = await embedding_gen.generate_text_embedding(test_text)
        
        # 尝试搜索 - 使用正确的参数
        test_results = await vector_search.search_similar_tables(
            query_embedding=test_embedding,
            k=1
        )
        
        logger.info("✅ 向量索引验证成功")
        logger.info(f"   找到 {len(test_results)} 个结果")
        return True
        
    except Exception as e:
        logger.error(f"❌ 向量索引验证失败: {e}")
        import traceback
        logger.error(f"错误详情: {traceback.format_exc()}")
        return False


async def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="初始化向量索引")
    parser.add_argument(
        "--tables",
        default="examples/final_subset_tables.json",
        help="表数据文件路径"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="强制重建索引"
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="初始化后验证索引"
    )
    
    args = parser.parse_args()
    
    # 检查文件是否存在
    tables_path = Path(args.tables)
    if not tables_path.exists():
        logger.error(f"❌ 文件不存在: {args.tables}")
        return
    
    try:
        # 初始化索引
        await initialize_vector_index(args.tables, args.force)
        
        # 验证索引
        if args.verify:
            await verify_index()
            
    except Exception as e:
        logger.error(f"❌ 错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())