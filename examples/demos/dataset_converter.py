#!/usr/bin/env python3
"""
数据集转换工具 - 将WebTable数据集转换为系统支持的格式
"""

import json
import csv
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any
import logging

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def read_csv_with_encoding(file_path: str) -> pd.DataFrame:
    """读取CSV文件，自动处理编码问题"""
    encodings = ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252']
    
    for encoding in encodings:
        try:
            return pd.read_csv(file_path, encoding=encoding)
        except UnicodeDecodeError:
            continue
        except Exception as e:
            logger.warning(f"使用编码 {encoding} 读取 {file_path} 失败: {e}")
            continue
    
    raise Exception(f"无法读取文件 {file_path}，尝试了所有编码")


def convert_csv_to_table_info(csv_file_path: str, limit_rows: int = 10) -> Dict[str, Any]:
    """将CSV文件转换为TableInfo格式"""
    try:
        # 读取CSV文件
        df = read_csv_with_encoding(csv_file_path)
        
        # 获取表名（从文件名提取）
        table_name = Path(csv_file_path).stem
        
        # 构建列信息
        columns = []
        for col_name in df.columns:
            # 获取样本值，排除空值和NaN
            sample_values = []
            col_data = df[col_name].dropna()
            
            # 转换为字符串并去除空白
            for val in col_data.head(limit_rows):
                if pd.notna(val) and str(val).strip():
                    sample_values.append(str(val).strip())
            
            # 推断数据类型
            data_type = "string"  # 默认为字符串
            if len(sample_values) > 0:
                # 尝试判断是否为数值类型
                try:
                    float(sample_values[0])
                    data_type = "numeric"
                except:
                    pass
            
            columns.append({
                "table_name": table_name,
                "column_name": col_name,
                "data_type": data_type,
                "sample_values": sample_values[:8]  # 限制样本值数量
            })
        
        return {
            "table_name": table_name,
            "columns": columns
        }
        
    except Exception as e:
        logger.error(f"转换CSV文件失败 {csv_file_path}: {e}")
        return None


def create_tables_dataset(tables_dir: str, output_file: str, max_tables: int = 50):
    """创建表数据集"""
    tables_path = Path(tables_dir)
    
    if not tables_path.exists():
        logger.error(f"表目录不存在: {tables_dir}")
        return
    
    tables_data = []
    processed_count = 0
    
    # 遍历所有CSV文件
    for csv_file in tables_path.glob("*.csv"):
        if processed_count >= max_tables:
            break
            
        logger.info(f"处理表文件: {csv_file.name}")
        table_info = convert_csv_to_table_info(str(csv_file))
        
        if table_info:
            tables_data.append(table_info)
            processed_count += 1
            
        if processed_count % 10 == 0:
            logger.info(f"已处理 {processed_count} 个表")
    
    # 保存到JSON文件
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(tables_data, f, ensure_ascii=False, indent=2)
    
    logger.info(f"成功创建表数据集: {output_file}，包含 {len(tables_data)} 个表")


def create_columns_dataset(tables_dir: str, output_file: str, max_tables: int = 50):
    """创建列数据集"""
    tables_path = Path(tables_dir)
    
    if not tables_path.exists():
        logger.error(f"表目录不存在: {tables_dir}")
        return
    
    columns_data = []
    processed_count = 0
    
    # 遍历所有CSV文件
    for csv_file in tables_path.glob("*.csv"):
        if processed_count >= max_tables:
            break
            
        logger.info(f"处理表文件: {csv_file.name}")
        table_info = convert_csv_to_table_info(str(csv_file))
        
        if table_info:
            table_name = table_info["table_name"]
            
            # 为每个列创建条目
            for col in table_info["columns"]:
                columns_data.append({
                    "table_name": table_name,
                    "column_name": col["column_name"],
                    "data_type": col["data_type"],
                    "sample_values": col["sample_values"]
                })
            
            processed_count += 1
    
    # 保存到JSON文件
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(columns_data, f, ensure_ascii=False, indent=2)
    
    logger.info(f"成功创建列数据集: {output_file}，包含 {len(columns_data)} 个列")


def load_query_data(query_file: str) -> List[Dict[str, Any]]:
    """加载查询数据"""
    queries = []
    
    try:
        with open(query_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                queries.append(row)
    except Exception as e:
        logger.error(f"加载查询文件失败 {query_file}: {e}")
    
    return queries


def load_ground_truth(gt_file: str) -> List[Dict[str, Any]]:
    """加载真实匹配数据"""
    ground_truth = []
    
    try:
        with open(gt_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                ground_truth.append(row)
    except Exception as e:
        logger.error(f"加载真实匹配文件失败 {gt_file}: {e}")
    
    return ground_truth


def main():
    """主函数"""
    base_dir = Path("Datasets/webtable")
    
    # 创建Join场景数据集
    logger.info("=== 创建Join场景数据集 ===")
    join_tables_dir = base_dir / "join" / "tables"
    
    # 创建表数据集
    create_tables_dataset(
        str(join_tables_dir), 
        "examples/webtable_join_tables.json",
        max_tables=30  # 限制表数量以避免过大
    )
    
    # 创建列数据集
    create_columns_dataset(
        str(join_tables_dir),
        "examples/webtable_join_columns.json", 
        max_tables=30
    )
    
    # 加载查询和真实匹配数据
    join_queries = load_query_data(str(base_dir / "join" / "webtable_join_query.csv"))
    join_ground_truth = load_ground_truth(str(base_dir / "join" / "webtable_join_ground_truth.csv"))
    
    logger.info(f"Join查询数量: {len(join_queries)}")
    logger.info(f"Join真实匹配数量: {len(join_ground_truth)}")
    
    # 保存查询和真实匹配数据
    with open("examples/webtable_join_queries.json", 'w', encoding='utf-8') as f:
        json.dump(join_queries, f, ensure_ascii=False, indent=2)
    
    with open("examples/webtable_join_ground_truth.json", 'w', encoding='utf-8') as f:
        json.dump(join_ground_truth, f, ensure_ascii=False, indent=2)
    
    # Union场景处理
    logger.info("=== 创建Union场景数据集 ===")
    union_queries = load_query_data(str(base_dir / "union" / "webtable_union_query.csv"))
    union_ground_truth = load_ground_truth(str(base_dir / "union" / "webtable_union_ground_truth.csv"))
    
    logger.info(f"Union查询数量: {len(union_queries)}")
    logger.info(f"Union真实匹配数量: {len(union_ground_truth)}")
    
    # 保存Union查询和真实匹配数据
    with open("examples/webtable_union_queries.json", 'w', encoding='utf-8') as f:
        json.dump(union_queries, f, ensure_ascii=False, indent=2)
    
    with open("examples/webtable_union_ground_truth.json", 'w', encoding='utf-8') as f:
        json.dump(union_ground_truth, f, ensure_ascii=False, indent=2)
    
    logger.info("=== 数据集转换完成 ===")
    

if __name__ == "__main__":
    main()