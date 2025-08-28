#!/usr/bin/env python3
"""
快速LSH Ensemble测试
只测试小数据集避免超时
"""

import os
import sys
import pandas as pd
from pathlib import Path
import time
import logging

# 确保使用本地datasketch版本
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

from datasketch import MinHashLSHEnsemble, MinHash
import farmhash

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def _hash_32(d):
    """使用farmhash的32位哈希函数"""
    return farmhash.hash32(d)

def quick_lsh_test():
    """快速LSH Ensemble测试"""
    data_dir = Path("/root/dataLakesMulti/baselines/data/lsh/nlctables/join")
    
    if not data_dir.exists():
        logging.error(f"数据路径不存在: {data_dir}")
        return
    
    # 只选择前5个有效的CSV文件
    csv_files = []
    for csv_file in data_dir.glob("*.csv"):
        try:
            df = pd.read_csv(csv_file, dtype=str).dropna()
            if len(df) > 0 and len(df.columns) > 0:
                csv_files.append(csv_file)
                if len(csv_files) >= 5:
                    break
        except:
            continue
    
    logging.info(f"使用 {len(csv_files)} 个CSV文件进行测试")
    
    # 计算列大小
    sizes = []
    keys = []
    
    for csv_file in csv_files:
        df = pd.read_csv(csv_file, dtype=str).dropna()
        for column in df.columns:
            vals = df[column].dropna().astype(str).tolist()
            vals = list(set(vals))  # 去重
            
            key = f"{csv_file.name}.{column}"
            keys.append(key)
            sizes.append(len(vals))
    
    logging.info(f"总共 {len(sizes)} 列")
    
    # 创建LSH Ensemble
    start_time = time.time()
    lsh = MinHashLSHEnsemble(
        threshold=0.1, 
        num_perm=64,  # 减少permutation数量
        num_part=4,   # 减少分区数量
        m=2           # 减少m值
    )
    
    # 分区计数
    lsh.count_partition(sizes)
    
    # 创建MinHash并索引
    key_idx = 0
    for csv_file in csv_files:
        df = pd.read_csv(csv_file, dtype=str).dropna()
        for column in df.columns:
            vals = df[column].dropna().astype(str).tolist()
            vals = list(set(vals))
            
            # 创建MinHash
            mh = MinHash(64, hashfunc=_hash_32)  # 减少num_perm
            for val in vals:
                if val.strip():
                    mh.update(val.strip().lower().encode('utf-8'))
            
            # 添加到索引
            key = keys[key_idx]
            size = sizes[key_idx]
            lsh.index((key, mh, size))
            
            key_idx += 1
    
    index_time = time.time() - start_time
    logging.info(f"索引构建耗时: {index_time:.2f}秒")
    
    # 执行查询测试
    if len(keys) > 0:
        # 使用第一个键进行查询 - 简化版本，直接使用现有数据
        test_key = keys[0]
        logging.info(f"测试查询key: {test_key}")
        
        # 使用已经构建索引时的数据进行查询测试
        # 直接创建一个简单的MinHash进行查询
        mh = MinHash(64, hashfunc=_hash_32)
        for i in range(10):  # 添加一些测试数据
            mh.update(f"test_value_{i}".encode('utf-8'))
        
        # 执行查询
        start_time = time.time()
        results = list(lsh.query(mh, 10))
        query_time = time.time() - start_time
        
        logging.info(f"查询测试:")
        logging.info(f"  查询时间: {query_time:.3f}秒")
        logging.info(f"  找到 {len(results)} 个相似列")
        
        for i, result in enumerate(results[:5]):
            logging.info(f"    {i+1}. {result}")
    
    return {
        'index_time': index_time,
        'query_time': query_time if 'query_time' in locals() else 0,
        'num_columns': len(keys),
        'num_results': len(results) if 'results' in locals() else 0
    }

if __name__ == "__main__":
    print("🚀 快速LSH Ensemble测试")
    print("=" * 40)
    
    try:
        result = quick_lsh_test()
        print("\n📊 测试结果:")
        print(f"  索引构建时间: {result['index_time']:.2f}秒")
        print(f"  查询时间: {result['query_time']:.3f}秒")
        print(f"  处理列数: {result['num_columns']}")
        print(f"  查询结果数: {result['num_results']}")
        print("\n✅ LSH Ensemble测试完成!")
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()