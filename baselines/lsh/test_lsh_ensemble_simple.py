#!/usr/bin/env python3
"""
简化版LSH Ensemble测试脚本
测试LSH Ensemble在转换后的数据上的基本功能

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
import pickle
import farmhash

# 确保使用本地datasketch版本
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

from datasketch import MinHashLSHEnsemble, MinHash

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def _hash_32(d):
    """使用farmhash的32位哈希函数"""
    return farmhash.hash32(d)

class LSHEnsembleSimpleTest:
    """简化版LSH Ensemble测试器"""
    
    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)
        
    def create_minhash(self, table_df: pd.DataFrame, num_perm: int = 128) -> MinHash:
        """为表格创建MinHash"""
        mh = MinHash(num_perm=num_perm, hashfunc=_hash_32)
        
        # 将所有列的所有值添加到MinHash中
        for col in table_df.columns:
            for value in table_df[col].dropna().astype(str):
                if value.strip():  # 忽略空字符串
                    mh.update(value.strip().lower().encode('utf-8'))
        
        return mh
    
    def build_lsh_ensemble(self, dataset: str, task: str, threshold: float = 0.1, num_perm: int = 128, num_part: int = 8, m: int = 4) -> MinHashLSHEnsemble:
        """为指定数据集构建LSH Ensemble索引"""
        dataset_path = self.data_dir / dataset / task
        
        if not dataset_path.exists():
            logging.error(f"数据集路径不存在: {dataset_path}")
            return None
        
        logging.info(f"构建LSH Ensemble索引: {dataset}-{task}")
        
        csv_files = list(dataset_path.glob("*.csv"))
        
        if len(csv_files) == 0:
            logging.warning(f"没有找到CSV文件: {dataset_path}")
            return None
        
        logging.info(f"找到 {len(csv_files)} 个CSV文件")
        
        # 第一遍：计算所有列的大小
        logging.info("第一遍：计算列大小...")
        sizes = []
        keys = []
        
        for csv_file in csv_files:
            try:
                df = pd.read_csv(csv_file, dtype=str).dropna()
                if len(df) == 0:
                    continue
                    
                # 为每一列创建键值和计算大小
                for column in df.columns:
                    vals = df[column].dropna().astype(str).tolist()
                    vals = list(set(vals))  # 去重
                    
                    key = f"{csv_file.name}.{column}"
                    keys.append(key)
                    sizes.append(len(vals))
                    
            except Exception as e:
                logging.error(f"处理文件 {csv_file.name} 时出错: {e}")
                continue
        
        if len(sizes) == 0:
            logging.error("没有有效的数据")
            return None
        
        logging.info(f"总共 {len(sizes)} 列")
        
        # 创建LSH Ensemble并分区
        lsh = MinHashLSHEnsemble(
            threshold=threshold, 
            num_perm=num_perm,
            num_part=num_part, 
            m=m
        )
        
        # 分区计数
        lsh.count_partition(sizes)
        
        # 第二遍：创建MinHash并索引
        logging.info("第二遍：创建MinHash并构建索引...")
        key_idx = 0
        
        for csv_file in csv_files:
            try:
                df = pd.read_csv(csv_file, dtype=str).dropna()
                if len(df) == 0:
                    continue
                    
                for column in df.columns:
                    vals = df[column].dropna().astype(str).tolist()
                    vals = list(set(vals))  # 去重
                    
                    # 创建MinHash
                    mh = MinHash(num_perm, hashfunc=_hash_32)
                    for val in vals:
                        if val.strip():
                            mh.update(val.strip().lower().encode('utf-8'))
                    
                    # 添加到索引
                    key = keys[key_idx]
                    size = sizes[key_idx]
                    lsh.index((key, mh, size))
                    
                    key_idx += 1
                    
                    if key_idx % 100 == 0:
                        logging.info(f"已索引 {key_idx} 列")
                        
            except Exception as e:
                logging.error(f"索引文件 {csv_file.name} 时出错: {e}")
                continue
        
        logging.info(f"成功构建LSH Ensemble索引，包含 {len(keys)} 列")
        return lsh
    
    def query_similar_tables(self, query_table_name: str, query_column: str, 
                           lsh: MinHashLSHEnsemble, dataset: str, task: str,
                           threshold: float = 0.1, top_k: int = 10) -> list:
        """查询相似表格列"""
        dataset_path = self.data_dir / dataset / task
        csv_file = dataset_path / query_table_name
        
        if not csv_file.exists():
            logging.error(f"查询表格文件不存在: {csv_file}")
            return []
        
        try:
            df = pd.read_csv(csv_file, dtype=str).dropna()
            if query_column not in df.columns:
                logging.error(f"查询列不存在: {query_column}")
                return []
            
            # 为查询列创建MinHash
            vals = df[query_column].dropna().astype(str).tolist()
            vals = list(set(vals))  # 去重
            
            mh = MinHash(128, hashfunc=_hash_32)  # 使用默认参数
            for val in vals:
                if val.strip():
                    mh.update(val.strip().lower().encode('utf-8'))
            
            # 查询相似列
            results = list(lsh.query(mh, len(vals)))
            
            # 解析结果并排序
            similar_columns = []
            for result_key in results:
                if result_key == f"{query_table_name}.{query_column}":
                    continue  # 跳过自己
                
                parts = result_key.split('.')
                if len(parts) >= 2:
                    table_name = parts[0]
                    column_name = '.'.join(parts[1:])  # 处理列名中可能包含点的情况
                    
                    similar_columns.append({
                        'table_name': table_name,
                        'column_name': column_name,
                        'key': result_key
                    })
            
            return similar_columns[:top_k]
            
        except Exception as e:
            logging.error(f"查询时出错: {e}")
            return []
    
    def load_queries(self, dataset: str, task: str) -> list:
        """加载查询数据"""
        queries_file = self.data_dir / dataset / task / "queries.json"
        
        if not queries_file.exists():
            logging.warning(f"查询文件不存在: {queries_file}")
            return []
        
        with open(queries_file, 'r', encoding='utf-8') as f:
            queries = json.load(f)
        
        return queries
    
    def evaluate_performance(self, dataset: str, task: str, max_queries: int = 5):
        """评估性能"""
        logging.info(f"\n=== 评估 {dataset}-{task} ===")
        
        # 构建索引
        start_time = time.time()
        lsh = self.build_lsh_ensemble(dataset, task, threshold=0.1, num_perm=128, num_part=8, m=4)
        index_time = time.time() - start_time
        
        if lsh is None:
            logging.error("LSH Ensemble索引构建失败")
            return
        
        logging.info(f"索引构建耗时: {index_time:.2f}秒")
        
        # 加载查询
        queries = self.load_queries(dataset, task)
        
        if len(queries) == 0:
            logging.warning("没有查询数据，使用随机表格和列进行测试")
            # 使用一些示例查询
            dataset_path = self.data_dir / dataset / task
            csv_files = list(dataset_path.glob("*.csv"))[:max_queries]
            
            queries = []
            for csv_file in csv_files:
                try:
                    df = pd.read_csv(csv_file, dtype=str).dropna()
                    if len(df.columns) > 0:
                        queries.append({
                            "query_table": csv_file.name,
                            "query_column": df.columns[0]  # 使用第一列
                        })
                except:
                    continue
        
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
            
            # 获取查询列（如果没有指定，使用第一列）
            query_column = query.get('query_column', query.get('column'))
            if not query_column:
                # 尝试读取表格获取第一列
                try:
                    csv_file = self.data_dir / dataset / task / query_table
                    df = pd.read_csv(csv_file, dtype=str).dropna()
                    if len(df.columns) > 0:
                        query_column = df.columns[0]
                    else:
                        continue
                except:
                    continue
            
            logging.info(f"查询 {i+1}: {query_table}.{query_column}")
            
            # 执行查询
            start_time = time.time()
            similar_columns = self.query_similar_tables(
                query_table, query_column, lsh, dataset, task, threshold=0.1, top_k=10
            )
            query_time = time.time() - start_time
            total_query_time += query_time
            
            results.append({
                'query_table': query_table,
                'query_column': query_column,
                'similar_columns': similar_columns,
                'query_time': query_time
            })
            
            # 显示结果
            if similar_columns:
                logging.info(f"  找到 {len(similar_columns)} 个相似列:")
                for j, sim_col in enumerate(similar_columns[:3]):
                    logging.info(f"    {j+1}. {sim_col['table_name']}.{sim_col['column_name']}")
            else:
                logging.info("  未找到相似列")
        
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
    # LSH数据目录
    data_dir = "/root/dataLakesMulti/baselines/data/lsh"
    
    tester = LSHEnsembleSimpleTest(data_dir)
    
    # 测试数据集
    datasets = [
        ('nlctables', 'join'),
        ('webtable', 'join'), 
        ('opendata', 'join')
    ]
    
    print("🔍 LSH Ensemble Baseline 测试")
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
            avg_results = sum(len(r['similar_columns']) for r in results) / len(results)
            print(f"  {key}: {len(results)}个查询, 平均{avg_time:.3f}s/查询, 平均{avg_results:.1f}个结果")
    
    print("\n✅ LSH Ensemble测试完成!")

if __name__ == "__main__":
    main()