from typing import List, Dict, Any
import json
import logging
import asyncio
from pathlib import Path
from src.core.models import TableInfo, ColumnInfo
from src.tools.vector_search import get_vector_search_engine
from src.tools.value_search import value_search_engine
from src.tools.embedding import get_embedding_generator
from src.config.settings import settings

logger = logging.getLogger(__name__)


class DataIndexer:
    """数据索引构建器 - 将WebTable数据构建为向量搜索索引"""
    
    def __init__(self):
        self.vector_search = get_vector_search_engine()
        self.value_search = value_search_engine
        self.embedding_gen = get_embedding_generator()
        
    async def build_indices_from_webtable_data(
        self, 
        tables_file: str, 
        columns_file: str = None,
        save_path: str = None
    ) -> Dict[str, Any]:
        """从WebTable数据构建索引
        
        Args:
            tables_file: 表数据JSON文件路径
            columns_file: 列数据JSON文件路径（可选）
            save_path: 索引保存路径（可选，默认使用配置路径）
        
        Returns:
            构建统计信息
        """
        try:
            logger.info("开始构建WebTable数据索引")
            
            # 加载数据
            tables_data = await self._load_tables_data(tables_file)
            columns_data = await self._load_columns_data(columns_file) if columns_file else []
            
            logger.info(f"加载了 {len(tables_data)} 个表和 {len(columns_data)} 个列")
            
            # 构建表级别索引
            table_stats = await self._build_table_indices(tables_data)
            
            # 构建列级别索引
            column_stats = await self._build_column_indices(tables_data, columns_data)
            
            # 保存索引
            if save_path is None:
                save_path = settings.vector_db.db_path
            
            await self._save_indices(save_path)
            
            stats = {
                "tables_processed": table_stats["processed"],
                "tables_indexed": table_stats["indexed"],
                "columns_processed": column_stats["processed"], 
                "columns_indexed": column_stats["indexed"],
                "index_path": save_path,
                "status": "success"
            }
            
            logger.info(f"索引构建完成: {stats}")
            return stats
            
        except Exception as e:
            logger.error(f"构建索引失败: {e}")
            return {
                "status": "error",
                "error": str(e)
            }
    
    async def _load_tables_data(self, file_path: str) -> List[TableInfo]:
        """加载表数据"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                raw_data = json.load(f)
            
            tables = []
            for table_data in raw_data:
                # 确保表数据包含必要字段 - 优先使用WebTable格式的字段名
                table_name = table_data.get('table_name') or table_data.get('name', 'unknown')
                columns_data = table_data.get('columns', [])
                
                columns = []
                for col_data in columns_data:
                    column = ColumnInfo(
                        table_name=table_name,
                        column_name=col_data.get('column_name') or col_data.get('name', 'unknown'),
                        data_type=col_data.get('data_type') or col_data.get('type'),
                        sample_values=col_data.get('sample_values', [])
                    )
                    columns.append(column)
                
                table = TableInfo(
                    table_name=table_name,
                    columns=columns,
                    row_count=table_data.get('row_count'),
                    file_path=table_data.get('file_path')
                )
                tables.append(table)
            
            logger.info(f"成功解析 {len(tables)} 个表")
            return tables
            
        except Exception as e:
            logger.error(f"加载表数据失败: {e}")
            raise
    
    async def _load_columns_data(self, file_path: str) -> List[ColumnInfo]:
        """加载列数据"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                raw_data = json.load(f)
            
            columns = []
            for col_data in raw_data:
                column = ColumnInfo(
                    table_name=col_data.get('table_name', 'unknown'),
                    column_name=col_data.get('name') or col_data.get('column_name', 'unknown'),
                    data_type=col_data.get('type') or col_data.get('data_type'),
                    sample_values=col_data.get('sample_values', [])
                )
                columns.append(column)
            
            logger.info(f"成功解析 {len(columns)} 个列")
            return columns
            
        except Exception as e:
            logger.error(f"加载列数据失败: {e}")
            raise
    
    async def _build_table_indices(self, tables_data: List[TableInfo]) -> Dict[str, int]:
        """构建表级别索引"""
        try:
            logger.info("开始构建表级别索引")
            
            processed = 0
            indexed = 0
            batch_size = settings.performance.batch_size
            
            for i in range(0, len(tables_data), batch_size):
                batch_tables = tables_data[i:i + batch_size]
                
                # 并行处理批次中的表
                tasks = []
                for table_info in batch_tables:
                    task = self._add_table_to_index(table_info)
                    tasks.append(task)
                
                # 执行批次任务
                results = await asyncio.gather(*tasks, return_exceptions=True)
                
                for j, result in enumerate(results):
                    processed += 1
                    if not isinstance(result, Exception):
                        indexed += 1
                    else:
                        logger.error(f"表 {batch_tables[j].table_name} 索引构建失败: {result}")
                
                logger.info(f"表索引进度: {processed}/{len(tables_data)}")
            
            logger.info(f"表级别索引构建完成: 处理{processed}个，成功索引{indexed}个")
            return {"processed": processed, "indexed": indexed}
            
        except Exception as e:
            logger.error(f"构建表索引失败: {e}")
            raise
    
    async def _build_column_indices(
        self, 
        tables_data: List[TableInfo], 
        additional_columns: List[ColumnInfo] = None
    ) -> Dict[str, int]:
        """构建列级别索引"""
        try:
            logger.info("开始构建列级别索引")
            
            # 收集所有列信息
            all_columns = []
            
            # 从表数据中提取列
            for table in tables_data:
                all_columns.extend(table.columns)
            
            # 添加额外的列数据
            if additional_columns:
                all_columns.extend(additional_columns)
            
            logger.info(f"总共需要索引 {len(all_columns)} 个列")
            
            processed = 0
            indexed = 0
            batch_size = settings.performance.batch_size
            
            for i in range(0, len(all_columns), batch_size):
                batch_columns = all_columns[i:i + batch_size]
                
                # 并行处理批次中的列
                tasks = []
                for column_info in batch_columns:
                    task = self._add_column_to_index(column_info)
                    tasks.append(task)
                
                # 执行批次任务
                results = await asyncio.gather(*tasks, return_exceptions=True)
                
                for j, result in enumerate(results):
                    processed += 1
                    if not isinstance(result, Exception):
                        indexed += 1
                    else:
                        logger.error(f"列 {batch_columns[j].full_name} 索引构建失败: {result}")
                
                logger.info(f"列索引进度: {processed}/{len(all_columns)}")
            
            logger.info(f"列级别索引构建完成: 处理{processed}个，成功索引{indexed}个")
            return {"processed": processed, "indexed": indexed}
            
        except Exception as e:
            logger.error(f"构建列索引失败: {e}")
            raise
    
    async def _add_table_to_index(self, table_info: TableInfo) -> None:
        """添加表到索引"""
        try:
            # 生成表的嵌入向量
            embedding = await self.embedding_gen.generate_table_embedding(table_info)
            
            # 添加到向量索引
            await self.vector_search.add_table_vector(table_info, embedding)
            
            logger.debug(f"成功添加表到索引: {table_info.table_name}")
            
        except Exception as e:
            logger.error(f"添加表 {table_info.table_name} 到索引失败: {e}")
            raise
    
    async def _add_column_to_index(self, column_info: ColumnInfo) -> None:
        """添加列到索引"""
        try:
            # 生成列的嵌入向量
            embedding = await self.embedding_gen.generate_column_embedding(column_info)
            
            # 添加到向量索引
            await self.vector_search.add_column_vector(column_info, embedding)
            
            # 如果有样本值，添加到值搜索索引
            if column_info.sample_values:
                await self.value_search.add_column_values(column_info)
            
            logger.debug(f"成功添加列到索引: {column_info.full_name}")
            
        except Exception as e:
            logger.error(f"添加列 {column_info.full_name} 到索引失败: {e}")
            raise
    
    async def _save_indices(self, save_path: str) -> None:
        """保存索引到文件"""
        try:
            logger.info(f"保存索引到: {save_path}")
            
            # 确保目录存在
            Path(save_path).mkdir(parents=True, exist_ok=True)
            
            # 保存向量搜索索引 - 修正：save_path是目录，需要添加文件名
            vector_index_path = str(Path(save_path) / "hnsw_index.bin")
            await self.vector_search.save_index(vector_index_path)
            
            # 保存值搜索索引
            value_index_path = str(Path(save_path) / "value_index.pkl")
            await self.value_search.save_index(value_index_path)
            
            logger.info("索引保存完成")
            
        except Exception as e:
            logger.error(f"保存索引失败: {e}")
            raise
    
    async def verify_indices(self, save_path: str = None) -> Dict[str, Any]:
        """验证索引是否正确构建"""
        try:
            if save_path is None:
                save_path = settings.vector_db.db_path
            
            # 加载索引 - 修正：save_path是目录，需要添加文件名
            vector_index_path = str(Path(save_path) / "hnsw_index.bin")
            await self.vector_search.load_index(vector_index_path)
            
            value_index_path = str(Path(save_path) / "value_index.pkl")
            await self.value_search.load_index(value_index_path)
            
            # 获取统计信息
            stats = {
                "vector_search": {
                    "column_count": len(self.vector_search.column_metadata),
                    "table_count": len(self.vector_search.table_metadata)
                },
                "value_search": {
                    "indexed_columns": getattr(self.value_search, 'indexed_columns_count', 0)
                },
                "status": "success"
            }
            
            logger.info(f"索引验证结果: {stats}")
            return stats
            
        except Exception as e:
            logger.error(f"索引验证失败: {e}")
            return {
                "status": "error",
                "error": str(e)
            }


# 便捷函数
async def build_webtable_indices(
    tables_file: str,
    columns_file: str = None, 
    save_path: str = None
) -> Dict[str, Any]:
    """构建WebTable索引的便捷函数"""
    indexer = DataIndexer()
    return await indexer.build_indices_from_webtable_data(
        tables_file=tables_file,
        columns_file=columns_file,
        save_path=save_path
    )


async def verify_indices(save_path: str = None) -> Dict[str, Any]:
    """验证索引的便捷函数"""
    indexer = DataIndexer()
    return await indexer.verify_indices(save_path)