"""
Value Similarity Tool - 基于实际数据值的相似性计算
用于增强JOIN/UNION匹配的准确性
"""
import logging
from typing import List, Dict, Any, Tuple, Set
import numpy as np
from collections import Counter


class ValueSimilarityTool:
    """
    基于样本数据值计算表之间的相似性
    这是对元数据和向量搜索的重要补充
    """
    
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def calculate_value_similarity(
        self,
        query_table: Dict[str, Any],
        candidate_table: Dict[str, Any],
        task_type: str = 'join'
    ) -> float:
        """
        计算两个表基于数据值的相似性
        
        Args:
            query_table: 查询表
            candidate_table: 候选表
            task_type: 'join' 或 'union'
            
        Returns:
            相似性分数 (0.0-1.0)
        """
        if task_type == 'join':
            return self._calculate_join_value_similarity(query_table, candidate_table)
        else:
            return self._calculate_union_value_similarity(query_table, candidate_table)
    
    def _calculate_join_value_similarity(
        self,
        query_table: Dict[str, Any],
        candidate_table: Dict[str, Any]
    ) -> float:
        """
        计算JOIN任务的值相似性
        重点检查：潜在的连接键是否有重叠值
        """
        query_cols = query_table.get('columns', [])
        candidate_cols = candidate_table.get('columns', [])
        
        if not query_cols or not candidate_cols:
            return 0.0
        
        similarities = []
        
        # 1. 检查相同列名的值重叠
        same_name_similarity = self._check_same_column_value_overlap(
            query_cols, candidate_cols
        )
        if same_name_similarity > 0:
            similarities.append(('same_name', same_name_similarity, 0.5))
        
        # 2. 检查潜在ID列的值重叠
        id_similarity = self._check_id_column_value_overlap(
            query_cols, candidate_cols
        )
        if id_similarity > 0:
            similarities.append(('id_overlap', id_similarity, 0.3))
        
        # 3. 检查数值范围重叠
        numeric_similarity = self._check_numeric_range_overlap(
            query_cols, candidate_cols
        )
        if numeric_similarity > 0:
            similarities.append(('numeric_range', numeric_similarity, 0.2))
        
        # 加权计算最终分数
        if not similarities:
            return 0.0
        
        total_weight = sum(s[2] for s in similarities)
        weighted_score = sum(s[1] * s[2] for s in similarities) / total_weight
        
        self.logger.debug(f"JOIN value similarities: {similarities}")
        return weighted_score
    
    def _calculate_union_value_similarity(
        self,
        query_table: Dict[str, Any],
        candidate_table: Dict[str, Any]
    ) -> float:
        """
        计算UNION任务的值相似性
        重点检查：数据分布是否相似
        """
        query_cols = query_table.get('columns', [])
        candidate_cols = candidate_table.get('columns', [])
        
        if not query_cols or not candidate_cols:
            return 0.0
        
        # 按列名匹配
        query_col_map = {c.get('column_name', c.get('name', '')): c for c in query_cols}
        candidate_col_map = {c.get('column_name', c.get('name', '')): c for c in candidate_cols}
        
        common_cols = set(query_col_map.keys()) & set(candidate_col_map.keys())
        
        if not common_cols:
            return 0.0
        
        # 计算每个共同列的值分布相似性
        similarities = []
        for col_name in common_cols:
            query_col = query_col_map[col_name]
            candidate_col = candidate_col_map[col_name]
            
            # 比较值分布
            dist_sim = self._compare_value_distributions(
                query_col.get('sample_values', []),
                candidate_col.get('sample_values', [])
            )
            similarities.append(dist_sim)
        
        # 返回平均相似性
        return np.mean(similarities) if similarities else 0.0
    
    def _check_same_column_value_overlap(
        self,
        query_cols: List[Dict],
        candidate_cols: List[Dict]
    ) -> float:
        """
        检查相同列名的值重叠程度
        """
        query_col_map = {c.get('column_name', c.get('name', '')): c for c in query_cols}
        candidate_col_map = {c.get('column_name', c.get('name', '')): c for c in candidate_cols}
        
        common_cols = set(query_col_map.keys()) & set(candidate_col_map.keys())
        
        if not common_cols:
            return 0.0
        
        overlaps = []
        for col_name in common_cols:
            query_values = set(query_col_map[col_name].get('sample_values', []))
            candidate_values = set(candidate_col_map[col_name].get('sample_values', []))
            
            # 过滤掉None值
            query_values = {v for v in query_values if v is not None}
            candidate_values = {v for v in candidate_values if v is not None}
            
            if query_values and candidate_values:
                # 计算Jaccard相似度
                intersection = query_values & candidate_values
                union = query_values | candidate_values
                if union:
                    overlap = len(intersection) / len(union)
                    overlaps.append(overlap)
        
        return np.mean(overlaps) if overlaps else 0.0
    
    def _check_id_column_value_overlap(
        self,
        query_cols: List[Dict],
        candidate_cols: List[Dict]
    ) -> float:
        """
        检查潜在ID列（以_id结尾或名为id）的值重叠
        """
        # 找出ID列
        query_id_cols = []
        for col in query_cols:
            col_name = col.get('column_name', col.get('name', '')).lower()
            if col_name == 'id' or col_name.endswith('_id'):
                query_id_cols.append(col)
        
        candidate_id_cols = []
        for col in candidate_cols:
            col_name = col.get('column_name', col.get('name', '')).lower()
            if col_name == 'id' or col_name.endswith('_id'):
                candidate_id_cols.append(col)
        
        if not query_id_cols or not candidate_id_cols:
            return 0.0
        
        # 检查所有ID列组合的值重叠
        max_overlap = 0.0
        for q_col in query_id_cols:
            q_values = set(q_col.get('sample_values', []))
            q_values = {v for v in q_values if v is not None}
            
            for c_col in candidate_id_cols:
                c_values = set(c_col.get('sample_values', []))
                c_values = {v for v in c_values if v is not None}
                
                if q_values and c_values:
                    intersection = q_values & c_values
                    if intersection:
                        # 有重叠值，可能可以JOIN
                        overlap = len(intersection) / min(len(q_values), len(c_values))
                        max_overlap = max(max_overlap, overlap)
        
        return max_overlap
    
    def _check_numeric_range_overlap(
        self,
        query_cols: List[Dict],
        candidate_cols: List[Dict]
    ) -> float:
        """
        检查数值列的范围重叠
        """
        overlaps = []
        
        for q_col in query_cols:
            q_type = q_col.get('data_type', q_col.get('type', '')).lower()
            if not any(t in q_type for t in ['int', 'float', 'num', 'double']):
                continue
            
            q_values = q_col.get('sample_values', [])
            q_numeric = [v for v in q_values if isinstance(v, (int, float))]
            
            if not q_numeric:
                continue
            
            q_min, q_max = min(q_numeric), max(q_numeric)
            
            for c_col in candidate_cols:
                c_type = c_col.get('data_type', c_col.get('type', '')).lower()
                if not any(t in c_type for t in ['int', 'float', 'num', 'double']):
                    continue
                
                c_values = c_col.get('sample_values', [])
                c_numeric = [v for v in c_values if isinstance(v, (int, float))]
                
                if not c_numeric:
                    continue
                
                c_min, c_max = min(c_numeric), max(c_numeric)
                
                # 计算范围重叠
                overlap_start = max(q_min, c_min)
                overlap_end = min(q_max, c_max)
                
                if overlap_start <= overlap_end:
                    # 有重叠
                    q_range = q_max - q_min
                    c_range = c_max - c_min
                    overlap_range = overlap_end - overlap_start
                    
                    if q_range > 0 and c_range > 0:
                        overlap_ratio = overlap_range / min(q_range, c_range)
                        overlaps.append(min(overlap_ratio, 1.0))
        
        return np.mean(overlaps) if overlaps else 0.0
    
    def _compare_value_distributions(
        self,
        values1: List[Any],
        values2: List[Any]
    ) -> float:
        """
        比较两个值列表的分布相似性
        """
        if not values1 or not values2:
            return 0.0
        
        # 过滤None值
        values1 = [v for v in values1 if v is not None]
        values2 = [v for v in values2 if v is not None]
        
        if not values1 or not values2:
            return 0.0
        
        # 检查数据类型
        types1 = set(type(v).__name__ for v in values1)
        types2 = set(type(v).__name__ for v in values2)
        
        # 类型必须兼容
        if not (types1 & types2):
            return 0.0
        
        # 对于字符串，比较唯一值的重叠
        if 'str' in types1 or 'str' in types2:
            set1 = set(str(v) for v in values1)
            set2 = set(str(v) for v in values2)
            
            intersection = set1 & set2
            union = set1 | set2
            
            if union:
                return len(intersection) / len(union)
            return 0.0
        
        # 对于数值，比较分布
        try:
            nums1 = [float(v) for v in values1 if isinstance(v, (int, float))]
            nums2 = [float(v) for v in values2 if isinstance(v, (int, float))]
            
            if nums1 and nums2:
                # 比较均值和标准差
                mean1, std1 = np.mean(nums1), np.std(nums1)
                mean2, std2 = np.mean(nums2), np.std(nums2)
                
                # 归一化差异
                mean_diff = abs(mean1 - mean2) / max(abs(mean1), abs(mean2), 1)
                std_diff = abs(std1 - std2) / max(std1, std2, 1)
                
                # 相似性分数
                similarity = 1 - (mean_diff * 0.6 + std_diff * 0.4)
                return max(0, similarity)
        except:
            pass
        
        return 0.0
    
    def find_best_join_columns(
        self,
        query_table: Dict[str, Any],
        candidate_table: Dict[str, Any]
    ) -> List[Tuple[str, str, float]]:
        """
        找出最佳的JOIN列对
        
        Returns:
            [(query_col, candidate_col, confidence), ...]
        """
        query_cols = query_table.get('columns', [])
        candidate_cols = candidate_table.get('columns', [])
        
        join_pairs = []
        
        # 1. 检查同名列
        query_col_map = {c.get('column_name', c.get('name', '')): c for c in query_cols}
        candidate_col_map = {c.get('column_name', c.get('name', '')): c for c in candidate_cols}
        
        common_cols = set(query_col_map.keys()) & set(candidate_col_map.keys())
        
        for col_name in common_cols:
            if col_name:  # 跳过空列名
                # 检查值重叠
                q_values = set(query_col_map[col_name].get('sample_values', []))
                c_values = set(candidate_col_map[col_name].get('sample_values', []))
                
                q_values = {v for v in q_values if v is not None}
                c_values = {v for v in c_values if v is not None}
                
                if q_values and c_values and (q_values & c_values):
                    overlap = len(q_values & c_values) / min(len(q_values), len(c_values))
                    join_pairs.append((col_name, col_name, overlap))
        
        # 2. 检查ID列配对（如user表的id和order表的user_id）
        for q_col in query_cols:
            q_name = q_col.get('column_name', q_col.get('name', ''))
            if q_name.lower() == 'id':
                # 查找候选表中的外键
                for c_col in candidate_cols:
                    c_name = c_col.get('column_name', c_col.get('name', ''))
                    if c_name.lower().endswith('_id') and c_name.lower() != 'id':
                        # 检查值重叠
                        q_values = set(q_col.get('sample_values', []))
                        c_values = set(c_col.get('sample_values', []))
                        
                        q_values = {v for v in q_values if v is not None}
                        c_values = {v for v in c_values if v is not None}
                        
                        if q_values and c_values and (q_values & c_values):
                            overlap = len(q_values & c_values) / min(len(q_values), len(c_values))
                            join_pairs.append((q_name, c_name, overlap * 0.8))  # 略低置信度
        
        # 排序并返回
        join_pairs.sort(key=lambda x: x[2], reverse=True)
        return join_pairs[:3]  # 返回前3个最佳匹配