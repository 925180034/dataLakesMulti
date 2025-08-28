#!/usr/bin/env python
"""
预计算向量嵌入
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

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def precompute_dataset_embeddings(dataset_type='complete', task_type='join'):
    """
    为指定数据集预计算所有向量嵌入
    """
    logger.info("="*80)
    logger.info(f"预计算向量嵌入: {task_type.upper()} - {dataset_type.upper()}")
    logger.info("="*80)
    
    # 加载数据
    base_dir = Path(f"examples/separated_datasets/{task_type}_{dataset_type}")
    
    with open(base_dir / "tables.json", 'r') as f:
        tables = json.load(f)
    
    logger.info(f"加载 {len(tables)} 个表")
    
    # 创建缓存目录
    cache_dir = Path("cache") / dataset_type
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    # 缓存文件路径
    index_file = cache_dir / f"vector_index_{len(tables)}.pkl"
    embeddings_file = cache_dir / f"table_embeddings_{len(tables)}.pkl"
    
    # 检查是否已存在
    if index_file.exists() and embeddings_file.exists():
        logger.info(f"向量索引已存在: {index_file}")
        logger.info("如需重新生成，请删除缓存文件")
        return
    
    # 初始化嵌入生成器
    from src.tools.batch_embedding import BatchEmbeddingGenerator
    
    logger.info("初始化嵌入生成器...")
    generator = BatchEmbeddingGenerator()
    generator.initialize()
    
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
                values_str = ', '.join(str(v) for v in sample_values[:3])
                text_parts.append(f"  Samples: {values_str}")
        
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
        batch = table_texts[i:i+batch_size]
        batch_embeddings = generator.generate_batch(batch, batch_size=batch_size)
        all_embeddings.extend(batch_embeddings)
        
        progress = min(i + batch_size, len(table_texts))
        logger.info(f"  进度: {progress}/{len(table_texts)} ({progress*100/len(table_texts):.1f}%)")
    
    elapsed = time.time() - start_time
    logger.info(f"嵌入生成完成，耗时: {elapsed:.2f}秒")
    logger.info(f"平均速度: {len(table_texts)/elapsed:.1f} 个表/秒")
    
    # 创建嵌入字典
    logger.info("创建嵌入字典...")
    embeddings_dict = {}
    for i, name in enumerate(table_names):
        embeddings_dict[name] = all_embeddings[i]
    
    # 构建FAISS索引
    logger.info("构建FAISS HNSW索引...")
    embeddings_array = np.array(all_embeddings).astype('float32')
    dimension = embeddings_array.shape[1]
    
    # 使用HNSW索引（性能最优）
    index = faiss.IndexHNSWFlat(dimension, 32)
    index.hnsw.efConstruction = 200
    index.hnsw.efSearch = 160
    
    # 添加向量到索引
    index.add(embeddings_array)
    
    logger.info(f"HNSW索引构建完成: {len(tables)} 个向量，维度: {dimension}")
    
    # 保存到文件
    logger.info("保存索引和嵌入...")
    
    with open(index_file, 'wb') as f:
        pickle.dump(index, f)
    logger.info(f"  索引已保存: {index_file}")
    
    with open(embeddings_file, 'wb') as f:
        pickle.dump(embeddings_dict, f)
    logger.info(f"  嵌入已保存: {embeddings_file}")
    
    # 验证
    logger.info("验证保存的文件...")
    
    with open(index_file, 'rb') as f:
        loaded_index = pickle.load(f)
    with open(embeddings_file, 'rb') as f:
        loaded_embeddings = pickle.load(f)
    
    logger.info(f"  验证成功: 索引包含 {loaded_index.ntotal} 个向量")
    logger.info(f"  验证成功: 嵌入字典包含 {len(loaded_embeddings)} 个表")
    
    logger.info("\n✅ 向量预计算完成！")
    logger.info(f"缓存位置: {cache_dir}")
    
    return str(index_file), str(embeddings_file)


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='预计算向量嵌入')
    parser.add_argument('--dataset', 
                       choices=['subset', 'complete', 'both'],
                       default='both',
                       help='数据集类型')
    parser.add_argument('--task',
                       choices=['join', 'union', 'both'],
                       default='both',
                       help='任务类型')
    
    args = parser.parse_args()
    
    # 确定要处理的数据集和任务
    datasets = ['subset', 'complete'] if args.dataset == 'both' else [args.dataset]
    tasks = ['join', 'union'] if args.task == 'both' else [args.task]
    
    logger.info("="*80)
    logger.info("批量预计算向量嵌入")
    logger.info("="*80)
    
    for dataset in datasets:
        for task in tasks:
            try:
                logger.info(f"\n处理: {task}_{dataset}")
                precompute_dataset_embeddings(dataset, task)
            except Exception as e:
                logger.error(f"处理 {task}_{dataset} 失败: {e}")
                continue
    
    logger.info("\n✅ 所有数据集处理完成！")


def precompute_all_embeddings(tables, dataset_type):
    """
    预计算所有表的嵌入向量并保存
    供其他脚本调用的接口函数
    """
    cache_dir = Path("cache") / dataset_type
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    # 检查缓存文件
    index_file = cache_dir / f"vector_index_{len(tables)}.pkl"
    embeddings_file = cache_dir / f"table_embeddings_{len(tables)}.pkl"
    
    # 如果缓存已存在，直接返回
    if index_file.exists() and embeddings_file.exists():
        logger.info(f"✅ 使用已存在的嵌入缓存: {cache_dir}")
        return
    
    logger.info(f"⚙️ 生成新的嵌入向量 ({len(tables)} 个表)...")
    
    # 默认使用真实嵌入（除非明确禁用）
    use_real_embeddings = os.environ.get('USE_REAL_EMBEDDINGS', 'true').lower() != 'false'
    
    if use_real_embeddings:
        # 使用优化的真实嵌入生成器
        logger.info("📊 使用真实SentenceTransformer模型...")
        from optimized_real_embedding import OptimizedRealEmbeddingGenerator
        
        generator = OptimizedRealEmbeddingGenerator()
        embeddings_array, table_names = generator.batch_generate_embeddings(tables, batch_size=32)
        
        # 构建嵌入字典
        embeddings_data = {}
        for i, name in enumerate(table_names):
            embeddings_data[name] = embeddings_array[i]
    else:
        # 使用虚拟嵌入（仅用于快速测试，不应用于生产）
        logger.info("⚠️ 警告：使用虚拟嵌入（无语义意义，仅用于测试）...")
        embeddings_data = {}
        for table in tables:
            table_name = table.get('table_name', '')
            columns = table.get('columns', [])
            
            # 创建表的文本表示
            column_names = [col['name'] for col in columns]
            table_text = f"{table_name} {' '.join(column_names)}"
            
            # 使用简单的虚拟嵌入
            import hashlib
            hash_obj = hashlib.md5(table_text.encode())
            hash_bytes = hash_obj.digest()
            # 转换为384维向量
            embedding = []
            for i in range(384):
                byte_idx = i % len(hash_bytes)
                embedding.append(float(hash_bytes[byte_idx]) / 255.0)
            
            embeddings_data[table_name] = embedding
    
    # 保存嵌入
    import pickle
    embeddings_file = cache_dir / f"table_embeddings_{len(tables)}.pkl"
    with open(embeddings_file, 'wb') as f:
        pickle.dump(embeddings_data, f)
    
    # 创建FAISS索引
    import numpy as np
    try:
        import faiss
        
        # 转换为numpy数组
        table_names = list(embeddings_data.keys())
        embeddings_array = np.array(list(embeddings_data.values()), dtype=np.float32)
        
        # 创建FAISS索引
        dimension = embeddings_array.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(embeddings_array)
        
        # 保存索引
        index_file = cache_dir / f"vector_index_{len(tables)}.pkl"
        with open(index_file, 'wb') as f:
            pickle.dump({'index': index, 'table_names': table_names}, f)
        
        print(f"✅ 创建向量索引: {index_file}")
    except ImportError:
        print("⚠️ FAISS未安装，跳过索引创建")
    
    return str(embeddings_file)

if __name__ == "__main__":
    main()