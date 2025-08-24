"""
增强的向量搜索工具 - 使用预构建的FAISS索引
"""
import numpy as np
import faiss
import pickle
import logging
from typing import List, Dict, Tuple
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


class EnhancedVectorSearch:
    """使用预构建FAISS索引的向量搜索"""
    
    def __init__(self, index_data: Dict = None, embeddings_data: Dict = None):
        self.index = None
        self.table_names = []
        self.embeddings = None
        self.model = None
        
        if index_data and embeddings_data:
            self.load_prebuilt_index(index_data, embeddings_data)
    
    def load_prebuilt_index(self, index_data: Dict, embeddings_data: Dict):
        """加载预构建的索引"""
        # 反序列化FAISS索引
        if 'index' in index_data:
            self.index = faiss.deserialize_index(index_data['index'])
            self.table_names = index_data.get('table_names', [])
            logger.info(f"加载FAISS索引: {self.index.ntotal} 个向量")
        
        # 加载嵌入
        if 'embeddings' in embeddings_data:
            self.embeddings = embeddings_data['embeddings']
            logger.info(f"加载嵌入: {self.embeddings.shape}")
    
    def get_model(self):
        """延迟加载模型（自动选择最佳设备）"""
        if self.model is None:
            # 让SentenceTransformer自动选择设备
            # 在主进程中会使用CUDA（如果可用），在子进程中会自动处理
            self.model = SentenceTransformer('all-MiniLM-L6-v2')
            # 获取实际使用的设备
            import torch
            device = 'cuda' if torch.cuda.is_available() and hasattr(self.model, '_target_device') and 'cuda' in str(self.model._target_device) else 'cpu'
            logger.info(f"初始化SentenceTransformer，设备: {device}")
        return self.model
    
    def search(self, query_table: Dict, top_k: int = 20, 
               filter_candidates: List[str] = None) -> List[Tuple[str, float]]:
        """
        搜索相似表
        
        Args:
            query_table: 查询表
            top_k: 返回的候选数
            filter_candidates: 如果提供，只从这些候选中返回结果
        
        Returns:
            表名和相似度分数的列表
        """
        if self.index is None:
            logger.warning("索引未初始化")
            return []
        
        # 生成查询向量
        query_text = self._table_to_text(query_table)
        model = self.get_model()
        query_vector = model.encode([query_text], convert_to_numpy=True).astype('float32')
        
        # 搜索（搜索更多以便过滤）
        search_k = min(top_k * 3, self.index.ntotal)
        distances, indices = self.index.search(query_vector, search_k)
        
        # 收集结果
        results = []
        for idx, dist in zip(indices[0], distances[0]):
            if idx < 0 or idx >= len(self.table_names):
                continue
            
            table_name = self.table_names[idx]
            # 转换距离为相似度分数（FAISS返回的是L2距离）
            score = 1.0 / (1.0 + dist)  # 距离越小，相似度越高
            
            # 如果有过滤候选，只保留在候选中的
            if filter_candidates is None or table_name in filter_candidates:
                results.append((table_name, score))
        
        # 排序并返回top_k
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]
    
    def _table_to_text(self, table: Dict) -> str:
        """将表转换为文本表示"""
        text_parts = []
        
        # 表名
        table_name = table.get('table_name', table.get('name', ''))
        text_parts.append(f"Table: {table_name}")
        
        # 列信息
        columns = table.get('columns', [])
        for col in columns[:20]:  # 限制列数
            col_name = col.get('name', '')
            col_type = col.get('type', '')
            text_parts.append(f"Column: {col_name} ({col_type})")
            
            # 样本值
            sample_values = col.get('sample_values', [])
            if sample_values:
                values_str = ', '.join(str(v)[:30] for v in sample_values[:3])
                text_parts.append(f"  Samples: {values_str}")
        
        # NL条件（如果有）
        if 'nl_condition' in table:
            text_parts.append(f"Condition: {table['nl_condition']}")
        
        return ' '.join(text_parts)