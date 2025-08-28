#!/usr/bin/env python
"""
优化的真实嵌入生成器 - 解决多进程重复初始化问题
核心问题和解决方案：
1. 问题：64个进程每个都初始化SentenceTransformer模型，导致内存爆炸和初始化时间极长
   解决：预先生成所有嵌入并缓存，进程只读取缓存

2. 问题：每次调用都检查和加载模型
   解决：单次预计算，后续直接使用缓存

3. 问题：批处理大小不合理
   解决：根据内存动态调整批处理大小
"""
import os
import sys
import json
import pickle
import time
import logging
from pathlib import Path
import numpy as np
from typing import List, Dict, Optional, Tuple
from sentence_transformers import SentenceTransformer
import faiss
import gc

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class OptimizedRealEmbeddingGenerator:
    """优化的真实嵌入生成器 - 单次初始化，批量处理"""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.model = None
        self.dimension = 384  # all-MiniLM-L6-v2的维度
        
    def initialize_model(self):
        """初始化模型（只调用一次）"""
        if self.model is not None:
            return
        
        logger.info(f"🚀 初始化SentenceTransformer模型: {self.model_name}")
        start_time = time.time()
        
        try:
            # 设置缓存目录
            cache_folder = "/root/.cache/huggingface/hub"
            os.makedirs(cache_folder, exist_ok=True)
            
            # 加载模型
            self.model = SentenceTransformer(
                self.model_name, 
                cache_folder=cache_folder,
                device='cpu'  # 明确使用CPU避免GPU内存问题
            )
            
            elapsed = time.time() - start_time
            logger.info(f"✅ 模型初始化完成，耗时: {elapsed:.2f}秒")
            
        except Exception as e:
            logger.error(f"❌ 模型初始化失败: {e}")
            raise
    
    def generate_table_text(self, table: Dict) -> str:
        """为表生成文本表示（优化版）"""
        table_name = table.get('table_name', table.get('name', ''))
        columns = table.get('columns', [])
        
        # 构建文本表示
        text_parts = [f"Table: {table_name}"]
        
        # 添加列信息（限制数量避免过长）
        for col in columns[:20]:  # 最多20列
            col_name = col.get('name', col.get('column_name', ''))
            col_type = col.get('type', col.get('data_type', ''))
            if col_name:
                text_parts.append(f"Column: {col_name}")
                if col_type:
                    text_parts[-1] += f" ({col_type})"
            
            # 添加样本值（限制数量）
            sample_values = col.get('sample_values', [])
            if sample_values:
                samples = [str(v) for v in sample_values[:3] if v]
                if samples:
                    text_parts.append(f"  Values: {', '.join(samples)}")
        
        return ' | '.join(text_parts)
    
    def batch_generate_embeddings(self, tables: List[Dict], batch_size: int = 32) -> Tuple[np.ndarray, List[str]]:
        """批量生成嵌入（核心优化）"""
        # 初始化模型
        self.initialize_model()
        
        # 生成文本表示
        logger.info("📝 生成表文本表示...")
        texts = []
        table_names = []
        
        for table in tables:
            text = self.generate_table_text(table)
            texts.append(text)
            table_names.append(table.get('table_name', table.get('name', f'table_{len(table_names)}')))
        
        logger.info(f"📊 批量生成 {len(texts)} 个嵌入向量...")
        start_time = time.time()
        
        # 批量编码
        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            
            # 生成嵌入
            batch_embeddings = self.model.encode(
                batch_texts,
                show_progress_bar=False,  # 关闭进度条避免输出混乱
                convert_to_numpy=True,
                normalize_embeddings=True
            )
            
            all_embeddings.append(batch_embeddings)
            
            # 进度报告
            progress = min(i + batch_size, len(texts))
            elapsed = time.time() - start_time
            speed = progress / elapsed if elapsed > 0 else 0
            eta = (len(texts) - progress) / speed if speed > 0 else 0
            
            logger.info(f"  进度: {progress}/{len(texts)} ({progress*100/len(texts):.1f}%) "
                       f"速度: {speed:.1f} 表/秒, 剩余: {eta:.0f}秒")
            
            # 定期清理内存
            if i % (batch_size * 10) == 0:
                gc.collect()
        
        # 合并所有批次
        embeddings_array = np.vstack(all_embeddings)
        
        elapsed = time.time() - start_time
        logger.info(f"✅ 嵌入生成完成: {len(texts)} 个表, 耗时: {elapsed:.2f}秒 "
                   f"({len(texts)/elapsed:.1f} 表/秒)")
        
        return embeddings_array, table_names


def precompute_embeddings_for_dataset(dataset_name: str, dataset_type: str = 'subset',
                                     task_type: str = 'join', force_regenerate: bool = False):
    """为整个数据集预计算嵌入"""
    logger.info("="*80)
    logger.info(f"🚀 预计算真实嵌入: {dataset_name}/{task_type}_{dataset_type}")
    logger.info("="*80)
    
    # 构建路径
    data_path = Path("examples") / dataset_name / f"{task_type}_{dataset_type}" / "tables.json"
    
    if not data_path.exists():
        logger.error(f"❌ 找不到数据文件: {data_path}")
        return False
    
    # 加载数据
    with open(data_path, 'r') as f:
        tables = json.load(f)
    logger.info(f"📊 加载了 {len(tables)} 个表")
    
    # 缓存路径
    cache_dir = Path("cache") / dataset_name
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    index_file = cache_dir / f"vector_index_{len(tables)}.pkl"
    embeddings_file = cache_dir / f"table_embeddings_{len(tables)}.pkl"
    
    # 检查缓存
    if not force_regenerate and index_file.exists() and embeddings_file.exists():
        logger.info(f"✅ 缓存已存在: {cache_dir}")
        logger.info(f"   索引文件: {index_file.stat().st_size / (1024*1024):.2f} MB")
        logger.info(f"   嵌入文件: {embeddings_file.stat().st_size / (1024*1024):.2f} MB")
        return True
    
    # 创建生成器
    generator = OptimizedRealEmbeddingGenerator()
    
    # 根据数据规模调整批处理大小
    if len(tables) > 1000:
        batch_size = 16  # 大数据集用小批次
    elif len(tables) > 500:
        batch_size = 32
    else:
        batch_size = 64  # 小数据集可以用大批次
    
    logger.info(f"⚙️ 使用批处理大小: {batch_size}")
    
    # 生成嵌入
    embeddings_array, table_names = generator.batch_generate_embeddings(tables, batch_size)
    
    # 构建FAISS索引
    logger.info("📊 构建FAISS索引...")
    dimension = embeddings_array.shape[1]
    
    if len(tables) < 1000:
        # 小数据集：使用FlatL2（精确搜索）
        index = faiss.IndexFlatL2(dimension)
        logger.info("  使用FlatL2索引（精确搜索）")
    else:
        # 大数据集：使用HNSW（近似搜索，更快）
        M = 32  # HNSW的连接数
        index = faiss.IndexHNSWFlat(dimension, M)
        index.hnsw.efConstruction = 100  # 构建时的搜索宽度
        index.hnsw.efSearch = 50  # 查询时的搜索宽度
        logger.info(f"  使用HNSW索引（近似搜索，M={M}）")
    
    # 添加向量到索引
    index.add(embeddings_array)
    logger.info(f"  添加了 {len(embeddings_array)} 个向量到索引")
    
    # 创建嵌入字典
    embeddings_dict = {}
    for i, name in enumerate(table_names):
        embeddings_dict[name] = embeddings_array[i]
    
    # 保存到文件
    logger.info("💾 保存缓存...")
    
    with open(index_file, 'wb') as f:
        pickle.dump(index, f)
    logger.info(f"  索引保存到: {index_file} ({index_file.stat().st_size / (1024*1024):.2f} MB)")
    
    with open(embeddings_file, 'wb') as f:
        pickle.dump(embeddings_dict, f)
    logger.info(f"  嵌入保存到: {embeddings_file} ({embeddings_file.stat().st_size / (1024*1024):.2f} MB)")
    
    # 清理内存
    del generator
    gc.collect()
    
    logger.info("✅ 预计算完成！")
    return True


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='优化的真实嵌入生成器')
    parser.add_argument('--dataset', required=True, 
                       choices=['nlctables', 'opendata', 'webtable'],
                       help='数据集名称')
    parser.add_argument('--type', default='subset',
                       choices=['subset', 'complete'],
                       help='数据集类型')
    parser.add_argument('--task', default='join',
                       choices=['join', 'union', 'both'],
                       help='任务类型')
    parser.add_argument('--force', action='store_true',
                       help='强制重新生成（忽略缓存）')
    
    args = parser.parse_args()
    
    # 处理任务
    tasks = ['join', 'union'] if args.task == 'both' else [args.task]
    
    for task in tasks:
        success = precompute_embeddings_for_dataset(
            args.dataset,
            args.type,
            task,
            args.force
        )
        
        if not success:
            logger.error(f"❌ {args.dataset}/{task}_{args.type} 处理失败")
            sys.exit(1)
    
    logger.info("\n✅ 所有任务完成！")


if __name__ == "__main__":
    main()