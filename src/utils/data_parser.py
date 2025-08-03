"""
数据解析工具函数
统一处理不同格式的表和列数据
"""

from typing import List, Dict, Any
from src.core.models import TableInfo, ColumnInfo
from src.utils.table_name_utils import normalize_table_name
import logging

logger = logging.getLogger(__name__)


def parse_table_data(table_data: Dict[str, Any]) -> TableInfo:
    """解析单个表数据
    
    处理不同的数据格式：
    - WebTable格式: {table_name: str, columns: [...]}
    - 简化格式: {name: str, columns: [...]}
    """
    # 获取表名 - 兼容不同格式，并标准化
    raw_table_name = table_data.get('table_name') or table_data.get('name', 'unknown')
    table_name = normalize_table_name(raw_table_name)
    
    # 解析列数据
    columns_data = table_data.get('columns', [])
    columns = []
    
    for col_data in columns_data:
        # 对于嵌套的列数据，table_name可能已经包含在列数据中
        raw_col_table_name = col_data.get('table_name', raw_table_name)
        col_table_name = normalize_table_name(raw_col_table_name)
        
        column = ColumnInfo(
            table_name=col_table_name,
            column_name=col_data.get('column_name') or col_data.get('name', 'unknown'),
            data_type=col_data.get('data_type') or col_data.get('type', 'string'),
            sample_values=col_data.get('sample_values', [])
        )
        columns.append(column)
    
    # 创建表对象 - 使用标准化的表名
    table = TableInfo(
        table_name=table_name,
        columns=columns,
        row_count=table_data.get('row_count'),
        file_path=table_data.get('file_path')
    )
    
    return table


def parse_tables_data(tables_data: List[Dict[str, Any]]) -> List[TableInfo]:
    """解析表数据列表"""
    tables = []
    for table_data in tables_data:
        try:
            table = parse_table_data(table_data)
            tables.append(table)
        except Exception as e:
            logger.warning(f"解析表数据失败: {e}, 数据: {table_data}")
            continue
    return tables


def parse_column_data(col_data: Dict[str, Any]) -> ColumnInfo:
    """解析单个列数据"""
    # 标准化表名
    raw_table_name = col_data.get('table_name', 'unknown')
    table_name = normalize_table_name(raw_table_name)
    
    return ColumnInfo(
        table_name=table_name,
        column_name=col_data.get('column_name') or col_data.get('name', 'unknown'),
        data_type=col_data.get('data_type') or col_data.get('type', 'string'),
        sample_values=col_data.get('sample_values', [])
    )


def parse_columns_data(columns_data: List[Dict[str, Any]]) -> List[ColumnInfo]:
    """解析列数据列表"""
    columns = []
    for col_data in columns_data:
        try:
            column = parse_column_data(col_data)
            columns.append(column)
        except Exception as e:
            logger.warning(f"解析列数据失败: {e}, 数据: {col_data}")
            continue
    return columns