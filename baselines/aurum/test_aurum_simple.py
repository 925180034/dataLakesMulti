#!/usr/bin/env python3
"""
简化版Aurum测试脚本
测试Aurum在转换后的数据上的基本功能

使用你的三个数据集:
- NLCTables 
- WebTable
- OpenData
"""

import os
import sys
import json
import pandas as pd
from pathlib import Path
import time
import logging
from datasketch import MinHash
import pickle

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class AurumSimpleTest:
    """简化版Aurum测试器"""
    
    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)
        self.hash_cache = {}
        
    def create_minhash(self, table_df: pd.DataFrame, num_perm: int = 128) -> MinHash:
        """为表格创建MinHash"""
        mh = MinHash(num_perm=num_perm)
        
        # 将所有列的所有值添加到MinHash中
        for col in table_df.columns:
            for value in table_df[col].dropna().astype(str):
                if value.strip():  # 忽略空字符串
                    mh.update(value.strip().lower().encode('utf-8'))
        
        return mh
    
    def build_index(self, dataset: str, task: str) -> dict:
        """为指定数据集构建MinHash索引"""
        dataset_path = self.data_dir / dataset / task
        
        if not dataset_path.exists():
            logging.error(f"数据集路径不存在: {dataset_path}")
            return {}
        
        logging.info(f"构建索引: {dataset}-{task}")
        
        index = {}
        csv_files = list(dataset_path.glob("*.csv"))
        
        if len(csv_files) == 0:
            logging.warning(f"没有找到CSV文件: {dataset_path}")
            return {}
        
        logging.info(f"找到 {len(csv_files)} 个CSV文件")
        
        for csv_file in csv_files:
            try:
                # 读取CSV文件
                df = pd.read_csv(csv_file)
                
                if len(df) == 0:
                    logging.warning(f"空表格: {csv_file.name}")
                    continue
                
                # 创建MinHash
                mh = self.create_minhash(df)
                
                # 存储索引
                index[csv_file.name] = {
                    'minhash': mh,
                    'num_rows': len(df),
                    'num_cols': len(df.columns),
                    'columns': list(df.columns)
                }
                
            except Exception as e:
                logging.error(f"处理文件 {csv_file.name} 时出错: {e}")
                continue
        
        logging.info(f"成功构建索引，包含 {len(index)} 个表格")
        return index
    
    def query_similar_tables(self, query_table_name: str, index: dict, 
                           threshold: float = 0.1, top_k: int = 10) -> list:
        """查询相似表格"""
        if query_table_name not in index:
            logging.error(f"查询表格不存在于索引中: {query_table_name}")
            return []
        
        query_mh = index[query_table_name]['minhash']
        similarities = []
        
        for table_name, table_info in index.items():
            if table_name == query_table_name:
                continue  # 跳过自己
            
            # 计算Jaccard相似度
            similarity = query_mh.jaccard(table_info['minhash'])
            
            if similarity >= threshold:
                similarities.append({
                    'table_name': table_name,
                    'similarity': similarity,
                    'num_rows': table_info['num_rows'],
                    'num_cols': table_info['num_cols'],
                    'columns': table_info['columns']
                })
        
        # 按相似度排序
        similarities.sort(key=lambda x: x['similarity'], reverse=True)
        
        return similarities[:top_k]
    
    def load_queries(self, dataset: str, task: str) -> list:
        """加载查询数据"""
        queries_file = self.data_dir / dataset / task / "queries.json"
        
        if not queries_file.exists():
            logging.warning(f"查询文件不存在: {queries_file}")
            return []
        
        with open(queries_file, 'r', encoding='utf-8') as f:
            queries = json.load(f)
        
        return queries
    
    def load_ground_truth(self, dataset: str, task: str) -> dict:
        """加载ground truth数据"""
        gt_file = self.data_dir / dataset / task / "ground_truth.json"
        
        if not gt_file.exists():
            logging.warning(f"Ground truth文件不存在: {gt_file}")
            return {}
        
        with open(gt_file, 'r', encoding='utf-8') as f:
            ground_truth = json.load(f)
        
        return ground_truth
    
    def evaluate_performance(self, dataset: str, task: str, max_queries: int = 5):
        """评估性能"""
        logging.info(f"\n=== 评估 {dataset}-{task} ===")
        
        # 构建索引
        start_time = time.time()
        index = self.build_index(dataset, task)
        index_time = time.time() - start_time
        
        if len(index) == 0:
            logging.error("索引构建失败")
            return
        
        logging.info(f"索引构建耗时: {index_time:.2f}秒")
        
        # 加载查询和ground truth
        queries = self.load_queries(dataset, task)
        ground_truth = self.load_ground_truth(dataset, task)
        
        if len(queries) == 0:
            logging.warning("没有查询数据，使用索引中的表格进行测试")
            # 使用前5个表格作为查询
            query_tables = list(index.keys())[:max_queries]
            queries = [{"query_table": table} for table in query_tables]
        
        logging.info(f"处理 {min(len(queries), max_queries)} 个查询")
        
        # 执行查询
        results = []
        total_query_time = 0
        
        for i, query in enumerate(queries[:max_queries]):
            query_table = query.get('query_table', query.get('seed_table'))
            
            if not query_table:
                logging.warning(f"查询 {i} 缺少表格名称")
                continue
            
            # 确保查询表格名称格式正确
            if not query_table.endswith('.csv'):
                query_table += '.csv'
            
            logging.info(f"查询 {i+1}: {query_table}")
            
            # 执行查询
            start_time = time.time()
            similar_tables = self.query_similar_tables(
                query_table, index, threshold=0.05, top_k=10
            )
            query_time = time.time() - start_time
            total_query_time += query_time
            
            results.append({
                'query_table': query_table,
                'similar_tables': similar_tables,
                'query_time': query_time
            })
            
            # 显示结果
            if similar_tables:
                logging.info(f"  找到 {len(similar_tables)} 个相似表格:")
                for j, sim_table in enumerate(similar_tables[:3]):
                    logging.info(f"    {j+1}. {sim_table['table_name']} (相似度: {sim_table['similarity']:.3f})")
            else:
                logging.info("  未找到相似表格")
        
        # 计算平均性能
        if results:
            avg_query_time = total_query_time / len(results)
            logging.info(f"\n性能统计:")
            logging.info(f"  平均查询时间: {avg_query_time:.3f}秒")
            logging.info(f"  总查询时间: {total_query_time:.2f}秒")
            logging.info(f"  索引构建时间: {index_time:.2f}秒")
            logging.info(f"  处理的查询数: {len(results)}")
        
        return results

def main():
    # Aurum数据目录
    data_dir = "/root/dataLakesMulti/baselines/data/aurum"
    
    tester = AurumSimpleTest(data_dir)
    
    # 测试数据集
    datasets = [
        ('nlctables', 'join'),
        ('webtable', 'join'), 
        ('opendata', 'join')
    ]
    
    print("🔍 Aurum Baseline 测试")
    print("=" * 50)
    
    all_results = {}
    
    for dataset, task in datasets:
        try:
            results = tester.evaluate_performance(dataset, task, max_queries=3)
            all_results[f"{dataset}-{task}"] = results
        except Exception as e:
            logging.error(f"测试 {dataset}-{task} 时出错: {e}")
            continue
    
    print("\n📊 总体结果摘要:")
    for key, results in all_results.items():
        if results:
            avg_time = sum(r['query_time'] for r in results) / len(results)
            avg_results = sum(len(r['similar_tables']) for r in results) / len(results)
            print(f"  {key}: {len(results)}个查询, 平均{avg_time:.3f}s/查询, 平均{avg_results:.1f}个结果")
    
    print("\n✅ Aurum测试完成!")

if __name__ == "__main__":
    main()