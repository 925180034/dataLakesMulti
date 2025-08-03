#!/usr/bin/env python3
"""
完整WebTable数据集提取和索引工具
"""
import asyncio
import json
import pandas as pd
import sys
from pathlib import Path
from typing import List, Dict, Any
import logging

# 设置项目路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.tools.data_indexer import DataIndexer
from src.config.settings import settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FullWebTableExtractor:
    """完整WebTable数据集提取器"""
    
    def __init__(self):
        self.webtable_dir = Path("Datasets/webtable/join/tables")
        self.output_dir = Path("examples")
        self.batch_size = 100
        
    def extract_table_metadata(self, csv_file_path: Path) -> Dict[str, Any]:
        """从CSV文件提取表元数据"""
        try:
            # 读取CSV文件 (只读前几行以获取结构)
            df = pd.read_csv(csv_file_path, nrows=10)
            
            columns = []
            for col_name in df.columns:
                # 推断数据类型
                col_data = df[col_name].dropna()
                if col_data.empty:
                    data_type = "string"
                elif col_data.dtype in ['int64', 'float64']:
                    data_type = "numeric"
                else:
                    data_type = "string"
                
                # 获取样本值
                sample_values = col_data.head(5).astype(str).tolist()
                
                columns.append({
                    "table_name": csv_file_path.stem,
                    "column_name": col_name,
                    "data_type": data_type,
                    "sample_values": sample_values
                })
            
            return {
                "table_name": csv_file_path.stem,
                "columns": columns,
                "row_count": len(df),
                "column_count": len(df.columns)
            }
            
        except Exception as e:
            logger.error(f"处理文件 {csv_file_path} 失败: {e}")
            return None
    
    def extract_all_tables(self) -> List[Dict[str, Any]]:
        """提取所有表的元数据"""
        logger.info(f"开始提取WebTable数据集: {self.webtable_dir}")
        
        csv_files = list(self.webtable_dir.glob("*.csv"))
        logger.info(f"找到 {len(csv_files)} 个CSV文件")
        
        all_tables = []
        failed_files = []
        
        for i, csv_file in enumerate(csv_files):
            if i % 100 == 0:
                logger.info(f"处理进度: {i}/{len(csv_files)} ({i/len(csv_files)*100:.1f}%)")
            
            table_metadata = self.extract_table_metadata(csv_file)
            if table_metadata:
                all_tables.append(table_metadata)
            else:
                failed_files.append(str(csv_file))
        
        logger.info(f"成功提取 {len(all_tables)} 个表")
        if failed_files:
            logger.warning(f"失败文件数: {len(failed_files)}")
        
        return all_tables, failed_files
    
    def save_extracted_data(self, tables_data: List[Dict[str, Any]], 
                           failed_files: List[str]):
        """保存提取的数据"""
        # 保存表数据
        tables_file = self.output_dir / "webtable_full_tables.json"
        with open(tables_file, 'w', encoding='utf-8') as f:
            json.dump(tables_data, f, indent=2, ensure_ascii=False)
        logger.info(f"表数据保存到: {tables_file}")
        
        # 保存列数据
        all_columns = []
        for table in tables_data:
            all_columns.extend(table["columns"])
        
        columns_file = self.output_dir / "webtable_full_columns.json"
        with open(columns_file, 'w', encoding='utf-8') as f:
            json.dump(all_columns, f, indent=2, ensure_ascii=False)
        logger.info(f"列数据保存到: {columns_file}")
        
        # 保存统计信息
        stats = {
            "total_tables": len(tables_data),
            "total_columns": len(all_columns),
            "failed_files": len(failed_files),
            "success_rate": len(tables_data) / (len(tables_data) + len(failed_files)) * 100,
            "avg_columns_per_table": len(all_columns) / len(tables_data) if tables_data else 0,
            "failed_files_list": failed_files[:10]  # 只保存前10个失败文件
        }
        
        stats_file = self.output_dir / "webtable_extraction_stats.json"
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)
        logger.info(f"统计信息保存到: {stats_file}")
        
        return stats
    
    async def create_full_index(self, tables_data: List[Dict[str, Any]]):
        """创建完整的向量索引"""
        logger.info("开始创建完整向量索引")
        
        indexer = DataIndexer()
        
        # 分批处理表数据
        for i in range(0, len(tables_data), self.batch_size):
            batch = tables_data[i:i + self.batch_size]
            logger.info(f"索引批次 {i//self.batch_size + 1}: {len(batch)} 个表")
            
            try:
                await indexer.index_tables(batch)
            except Exception as e:
                logger.error(f"索引批次失败: {e}")
                continue
        
        logger.info("完整索引创建完成")


async def main():
    """主函数"""
    print("🚀 WebTable完整数据集提取工具")
    print("=" * 50)
    
    extractor = FullWebTableExtractor()
    
    # 检查数据目录
    if not extractor.webtable_dir.exists():
        print(f"❌ WebTable数据目录不存在: {extractor.webtable_dir}")
        return
    
    # 提取数据
    print("📊 第一阶段: 提取表元数据")
    tables_data, failed_files = extractor.extract_all_tables()
    
    if not tables_data:
        print("❌ 未能提取到任何表数据")
        return
    
    # 保存数据
    print("💾 第二阶段: 保存提取数据")
    stats = extractor.save_extracted_data(tables_data, failed_files)
    
    print(f"\n📈 提取统计:")
    print(f"   总表数: {stats['total_tables']}")
    print(f"   总列数: {stats['total_columns']}")
    print(f"   成功率: {stats['success_rate']:.1f}%")
    print(f"   平均每表列数: {stats['avg_columns_per_table']:.1f}")
    
    # 询问是否创建索引
    response = input("\n🔧 是否立即创建向量索引? (y/n): ")
    if response.lower() == 'y':
        print("⚡ 第三阶段: 创建向量索引")
        await extractor.create_full_index(tables_data)
        print("✅ 完整索引创建完成")
    else:
        print("⏭️  跳过索引创建。可以稍后使用以下命令创建:")
        print("   python run_cli.py index-tables examples/webtable_full_tables.json")
    
    print(f"\n🎯 完成! 使用以下文件进行测试:")
    print(f"   表数据: examples/webtable_full_tables.json")
    print(f"   列数据: examples/webtable_full_columns.json")


if __name__ == "__main__":
    asyncio.run(main())