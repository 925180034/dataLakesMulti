"""
表名处理工具
统一处理表名格式，确保系统一致性
"""

from typing import Optional


def normalize_table_name(table_name: Optional[str]) -> str:
    """标准化表名
    
    统一处理规则：
    1. 保留.csv后缀（如果有）以匹配数据集格式
    2. 去除前后空格
    3. 保持大小写不变
    
    Args:
        table_name: 原始表名
        
    Returns:
        标准化后的表名
    """
    if not table_name:
        return ""
    
    # 去除前后空格
    name = table_name.strip()
    
    # 不再去除.csv后缀，保持原始格式
    # if name.endswith('.csv'):
    #     name = name[:-4]
    
    return name


def add_csv_extension(table_name: Optional[str]) -> str:
    """添加.csv扩展名
    
    Args:
        table_name: 表名
        
    Returns:
        带.csv扩展名的表名
    """
    if not table_name:
        return ""
    
    name = table_name.strip()
    
    # 如果已经有.csv后缀，直接返回
    if name.endswith('.csv'):
        return name
    
    return f"{name}.csv"


def table_names_match(name1: Optional[str], name2: Optional[str]) -> bool:
    """检查两个表名是否匹配
    
    忽略.csv后缀的差异
    
    Args:
        name1: 第一个表名
        name2: 第二个表名
        
    Returns:
        是否匹配
    """
    return normalize_table_name(name1) == normalize_table_name(name2)


def ensure_consistent_format(table_name: Optional[str], include_extension: bool = True) -> str:
    """确保表名格式一致
    
    Args:
        table_name: 原始表名
        include_extension: 是否包含.csv扩展名
        
    Returns:
        格式一致的表名
    """
    normalized = normalize_table_name(table_name)
    
    if include_extension and normalized:
        return add_csv_extension(normalized)
    
    return normalized