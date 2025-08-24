#!/usr/bin/env python
"""
预计算NLCTables向量嵌入
一次性为所有表生成向量嵌入并保存
避免每次查询都重新生成
"""
import os
import sys
import json
import pickle
import time
import logging
from pathlib import Path
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def precompute_nlctables_embeddings(dataset_type='subset', task_type='union'):
    """
    为NLCTables数据集预计算所有向量嵌入
    """
    logger.info("="*80)
    logger.info(f"预计算NLCTables向量嵌入: {task_type.upper()} - {dataset_type.upper()}")
    logger.info("="*80)
    
    # 加载数据
    base_dir = Path(f"examples/nlctables/{task_type}_{dataset_type}")
    
    with open(base_dir / "tables.json", 'r') as f:
        tables = json.load(f)
    
    logger.info(f"加载 {len(tables)} 个表")
    
    # 创建缓存目录
    cache_dir = Path("cache") / "nlctables" / f"{task_type}_{dataset_type}"
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    # 缓存文件路径
    index_file = cache_dir / f"vector_index_{len(tables)}.pkl"
    embeddings_file = cache_dir / f"table_embeddings_{len(tables)}.pkl"
    
    # 检查是否已存在
    if index_file.exists() and embeddings_file.exists():
        logger.info(f"向量索引已存在: {index_file}")
        logger.info("如需重新生成，请删除缓存文件")
        return str(index_file), str(embeddings_file)
    
    # 初始化嵌入模型
    logger.info("初始化SentenceTransformer模型...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # 准备文本
    logger.info("准备表文本...")
    table_texts = []
    table_names = []
    
    for table in tables:
        # 创建表的文本表示
        table_name = table.get('table_name', table.get('name', ''))
        table_names.append(table_name)
        
        # 组合表名和列信息
        text_parts = [f"Table: {table_name}"]
        
        columns = table.get('columns', [])
        for col in columns:
            col_name = col.get('name', '')
            col_type = col.get('type', '')
            text_parts.append(f"Column: {col_name} ({col_type})")
            
            # 添加样本值（如果有）
            sample_values = col.get('sample_values', [])
            if sample_values:
                # 限制样本值数量和长度
                values_str = ', '.join(str(v)[:50] for v in sample_values[:3])
                text_parts.append(f"  Samples: {values_str}")
        
        # NLCTables特殊：添加NL条件（如果有）
        if 'nl_condition' in table:
            text_parts.append(f"Condition: {table['nl_condition']}")
        
        table_text = ' '.join(text_parts)
        table_texts.append(table_text)
    
    logger.info(f"准备了 {len(table_texts)} 个表文本")
    
    # 批量生成嵌入
    logger.info("开始批量生成嵌入...")
    start_time = time.time()
    
    # 批量生成，每批32个
    batch_size = 32
    all_embeddings = []
    
    for i in range(0, len(table_texts), batch_size):
        batch = table_texts[i:i + batch_size]
        batch_embeddings = model.encode(batch, convert_to_numpy=True)
        all_embeddings.append(batch_embeddings)
        
        if (i + batch_size) % 100 == 0:
            logger.info(f"  已处理 {min(i + batch_size, len(table_texts))}/{len(table_texts)} 个表")
    
    # 合并所有嵌入
    embeddings = np.vstack(all_embeddings).astype('float32')
    logger.info(f"生成嵌入完成，形状: {embeddings.shape}")
    
    # 构建FAISS索引
    logger.info("构建FAISS索引...")
    dimension = embeddings.shape[1]
    
    # 使用HNSW索引，更快的搜索
    index = faiss.IndexHNSWFlat(dimension, 32)
    index.hnsw.efConstruction = 200
    index.hnsw.efSearch = 100
    
    # 添加向量到索引
    index.add(embeddings)
    logger.info(f"FAISS索引构建完成，包含 {index.ntotal} 个向量")
    
    # 保存索引和嵌入
    logger.info("保存索引和嵌入...")
    
    # 保存FAISS索引
    with open(index_file, 'wb') as f:
        pickle.dump({
            'index': faiss.serialize_index(index),
            'table_names': table_names,
            'dimension': dimension
        }, f)
    
    # 保存原始嵌入
    with open(embeddings_file, 'wb') as f:
        pickle.dump({
            'embeddings': embeddings,
            'table_names': table_names,
            'table_texts': table_texts
        }, f)
    
    elapsed = time.time() - start_time
    logger.info(f"✅ 预计算完成，耗时: {elapsed:.2f}秒")
    logger.info(f"  索引文件: {index_file}")
    logger.info(f"  嵌入文件: {embeddings_file}")
    
    return str(index_file), str(embeddings_file)


def precompute_all_nlctables():
    """预计算所有NLCTables数据集的嵌入"""
    datasets = [
        ('union', 'subset'),
        ('union', 'complete'),
        ('join', 'subset'),
        ('join', 'complete')
    ]
    
    for task, dataset in datasets:
        # 检查数据集是否存在
        base_dir = Path(f"examples/nlctables/{task}_{dataset}")
        if base_dir.exists():
            logger.info(f"\n处理 {task}_{dataset}...")
            precompute_nlctables_embeddings(dataset, task)
        else:
            logger.info(f"跳过不存在的数据集: {task}_{dataset}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='union', help='Task type: union or join')
    parser.add_argument('--dataset', type=str, default='subset', help='Dataset type: subset or complete')
    parser.add_argument('--all', action='store_true', help='Precompute all datasets')
    
    args = parser.parse_args()
    
    if args.all:
        precompute_all_nlctables()
    else:
        precompute_nlctables_embeddings(args.dataset, args.task)