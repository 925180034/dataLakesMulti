"""
匈牙利算法表匹配模块 - 基于LakeBench最佳实践
借鉴Aurum、DeepJoin、Starmie等项目的精确匹配策略
"""

from typing import List, Dict, Any, Optional, Tuple
import logging
import numpy as np
from numpy.linalg import norm

from src.core.models import ColumnInfo, TableInfo

logger = logging.getLogger(__name__)


class HungarianMatcher:
    """基于匈牙利算法的精确表匹配器
    
    技术特点：
    - 使用二分图最大权重匹配算法
    - 解决表间列的最优对应关系
    - 提供精确的相似度评分
    - 支持阈值过滤和部分匹配
    """
    
    def __init__(self, similarity_threshold: float = 0.6):
        self.similarity_threshold = similarity_threshold
        self._munkres = None
        self._init_munkres()
    
    def _init_munkres(self):
        """初始化匈牙利算法求解器"""
        try:
            from munkres import Munkres, make_cost_matrix, DISALLOWED
            self._munkres = Munkres()
            self._make_cost_matrix = make_cost_matrix
            self._DISALLOWED = DISALLOWED
            logger.info("匈牙利算法模块初始化完成")
            
        except ImportError:
            logger.error("munkres库未安装，请安装: pip install munkres")
            raise
    
    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """计算余弦相似度"""
        try:
            # 确保向量是numpy数组
            if not isinstance(vec1, np.ndarray):
                vec1 = np.array(vec1)
            if not isinstance(vec2, np.ndarray):
                vec2 = np.array(vec2)
            
            # 处理零向量情况
            norm1, norm2 = norm(vec1), norm(vec2)
            if norm1 == 0 or norm2 == 0:
                return 0.0
            
            return np.dot(vec1, vec2) / (norm1 * norm2)
            
        except Exception as e:
            logger.warning(f"余弦相似度计算失败: {e}")
            return 0.0
    
    def compute_similarity_matrix(
        self, 
        table1_embeddings: List[List[float]], 
        table2_embeddings: List[List[float]]
    ) -> np.ndarray:
        """计算两个表之间所有列对的相似度矩阵
        
        Args:
            table1_embeddings: 表1的列向量列表
            table2_embeddings: 表2的列向量列表
            
        Returns:
            相似度矩阵 (n_rows × n_cols)
        """
        n_rows = len(table1_embeddings)
        n_cols = len(table2_embeddings)
        
        similarity_matrix = np.zeros((n_rows, n_cols), dtype=float)
        
        for i, emb1 in enumerate(table1_embeddings):
            for j, emb2 in enumerate(table2_embeddings):
                similarity = self._cosine_similarity(emb1, emb2)
                similarity_matrix[i, j] = similarity
        
        logger.debug(f"计算相似度矩阵: {n_rows}×{n_cols}")
        return similarity_matrix
    
    def find_optimal_matching(
        self, 
        similarity_matrix: np.ndarray,
        threshold: Optional[float] = None
    ) -> Tuple[float, List[Tuple[int, int, float]], np.ndarray]:
        """使用匈牙利算法找到最优匹配
        
        Args:
            similarity_matrix: 相似度矩阵
            threshold: 相似度阈值，低于此值的匹配将被忽略
            
        Returns:
            (总分数, 匹配对列表, 过滤后的相似度矩阵)
        """
        if threshold is None:
            threshold = self.similarity_threshold
        
        n_rows, n_cols = similarity_matrix.shape
        
        # 应用阈值过滤，创建成本矩阵
        filtered_matrix = similarity_matrix.copy()
        filtered_matrix[filtered_matrix < threshold] = 0
        
        # 转换为成本矩阵（匈牙利算法求最小成本）
        max_similarity = similarity_matrix.max()
        cost_matrix = self._make_cost_matrix(
            filtered_matrix,
            lambda cost: (max_similarity - cost) if cost != self._DISALLOWED else self._DISALLOWED
        )
        
        # 求解最优匹配
        optimal_assignments = self._munkres.compute(cost_matrix)
        
        # 计算结果
        total_score = 0.0
        valid_matches = []
        
        for row, col in optimal_assignments:
            similarity = similarity_matrix[row, col]
            if similarity >= threshold:
                total_score += similarity
                valid_matches.append((row, col, similarity))
        
        logger.debug(f"匈牙利算法找到 {len(valid_matches)} 个有效匹配，总分: {total_score:.3f}")
        
        return total_score, valid_matches, filtered_matrix
    
    def match_tables(
        self,
        table1_columns: List[ColumnInfo],
        table2_columns: List[ColumnInfo], 
        table1_embeddings: List[List[float]],
        table2_embeddings: List[List[float]],
        threshold: Optional[float] = None
    ) -> Dict[str, Any]:
        """精确匹配两个表
        
        Args:
            table1_columns: 表1的列信息
            table2_columns: 表2的列信息
            table1_embeddings: 表1的列向量
            table2_embeddings: 表2的列向量
            threshold: 相似度阈值
            
        Returns:
            匹配结果字典
        """
        try:
            if len(table1_columns) != len(table1_embeddings):
                raise ValueError("表1列信息和向量数量不匹配")
            if len(table2_columns) != len(table2_embeddings):
                raise ValueError("表2列信息和向量数量不匹配")
            
            # 计算相似度矩阵
            similarity_matrix = self.compute_similarity_matrix(table1_embeddings, table2_embeddings)
            
            # 寻找最优匹配
            total_score, matches, filtered_matrix = self.find_optimal_matching(
                similarity_matrix, threshold
            )
            
            # 构建详细的匹配结果
            detailed_matches = []
            for row, col, similarity in matches:
                match_info = {
                    "table1_column": {
                        "index": row,
                        "name": table1_columns[row].column_name,
                        "table_name": table1_columns[row].table_name,
                        "data_type": table1_columns[row].data_type,
                        "full_name": table1_columns[row].full_name
                    },
                    "table2_column": {
                        "index": col,
                        "name": table2_columns[col].column_name,
                        "table_name": table2_columns[col].table_name,
                        "data_type": table2_columns[col].data_type,
                        "full_name": table2_columns[col].full_name
                    },
                    "similarity": similarity,
                    "data_type_match": table1_columns[row].data_type == table2_columns[col].data_type
                }
                detailed_matches.append(match_info)
            
            # 计算匹配统计
            n1, n2 = len(table1_columns), len(table2_columns)
            match_ratio = len(matches) / max(n1, n2) if max(n1, n2) > 0 else 0
            avg_similarity = total_score / len(matches) if matches else 0
            
            # 计算不同的匹配分数
            scores = self._calculate_matching_scores(total_score, matches, n1, n2)
            
            result = {
                "total_score": total_score,
                "average_similarity": avg_similarity,
                "match_count": len(matches),
                "table1_columns": n1,
                "table2_columns": n2,
                "match_ratio": match_ratio,
                "detailed_matches": detailed_matches,
                "scores": scores,
                "similarity_matrix": similarity_matrix.tolist(),
                "threshold_used": threshold or self.similarity_threshold
            }
            
            logger.info(f"表匹配完成: {n1}×{n2} 列，找到 {len(matches)} 个匹配")
            return result
            
        except Exception as e:
            logger.error(f"表匹配失败: {e}")
            return {
                "total_score": 0,
                "average_similarity": 0,
                "match_count": 0,
                "error": str(e)
            }
    
    def _calculate_matching_scores(
        self, 
        total_score: float, 
        matches: List[Tuple[int, int, float]], 
        n1: int, 
        n2: int
    ) -> Dict[str, float]:
        """计算各种匹配评分指标"""
        scores = {}
        
        # 1. 原始总分
        scores["raw_total"] = total_score
        
        # 2. 标准化分数（除以最小列数）
        min_cols = min(n1, n2)
        scores["normalized_min"] = total_score / min_cols if min_cols > 0 else 0
        
        # 3. 标准化分数（除以最大列数） 
        max_cols = max(n1, n2)
        scores["normalized_max"] = total_score / max_cols if max_cols > 0 else 0
        
        # 4. Jaccard风格分数（考虑未匹配的列）
        matched_cols = len(matches)
        total_cols = n1 + n2 - matched_cols  # 并集大小
        scores["jaccard_style"] = total_score / total_cols if total_cols > 0 else 0
        
        # 5. 平均相似度
        scores["average_similarity"] = total_score / matched_cols if matched_cols > 0 else 0
        
        # 6. 加权分数（考虑匹配比例）
        match_ratio = matched_cols / max_cols if max_cols > 0 else 0
        scores["weighted"] = scores["average_similarity"] * match_ratio
        
        return scores
    
    def batch_match_tables(
        self,
        query_table: Tuple[List[ColumnInfo], List[List[float]]],
        candidate_tables: List[Tuple[List[ColumnInfo], List[List[float]]]],
        threshold: Optional[float] = None,
        top_k: int = 10
    ) -> List[Dict[str, Any]]:
        """批量匹配多个候选表
        
        Args:
            query_table: (列信息, 列向量) 的查询表
            candidate_tables: 候选表列表
            threshold: 相似度阈值
            top_k: 返回前k个最佳匹配
            
        Returns:
            按匹配分数排序的结果列表
        """
        query_columns, query_embeddings = query_table
        results = []
        
        for i, (candidate_columns, candidate_embeddings) in enumerate(candidate_tables):
            try:
                result = self.match_tables(
                    query_columns, candidate_columns,
                    query_embeddings, candidate_embeddings,
                    threshold
                )
                result["candidate_index"] = i
                result["candidate_table_name"] = candidate_columns[0].table_name if candidate_columns else f"table_{i}"
                results.append(result)
                
            except Exception as e:
                logger.warning(f"候选表 {i} 匹配失败: {e}")
                continue
        
        # 按加权分数排序
        results.sort(key=lambda x: x.get("scores", {}).get("weighted", 0), reverse=True)
        
        logger.info(f"批量匹配完成: {len(candidate_tables)} 个候选表，返回前 {top_k} 个结果")
        return results[:top_k]
    
    def explain_matching(self, matching_result: Dict[str, Any]) -> str:
        """生成匹配结果的文字解释"""
        if "error" in matching_result:
            return f"匹配失败: {matching_result['error']}"
        
        matches = matching_result["detailed_matches"]
        scores = matching_result["scores"]
        
        explanation = []
        explanation.append(f"表匹配分析:")
        explanation.append(f"- 表1列数: {matching_result['table1_columns']}")
        explanation.append(f"- 表2列数: {matching_result['table2_columns']}")
        explanation.append(f"- 成功匹配: {matching_result['match_count']} 列")
        explanation.append(f"- 匹配比例: {matching_result['match_ratio']:.1%}")
        explanation.append(f"- 平均相似度: {scores['average_similarity']:.3f}")
        explanation.append(f"- 加权分数: {scores['weighted']:.3f}")
        
        if matches:
            explanation.append(f"\\n最佳匹配:")
            # 显示前3个最佳匹配
            top_matches = sorted(matches, key=lambda x: x['similarity'], reverse=True)[:3]
            for i, match in enumerate(top_matches, 1):
                col1 = match["table1_column"]
                col2 = match["table2_column"]
                explanation.append(
                    f"  {i}. {col1['name']} ↔ {col2['name']} "
                    f"(相似度: {match['similarity']:.3f})"
                )
        
        return "\\n".join(explanation)


# 工厂函数和辅助工具
def create_hungarian_matcher(threshold: float = 0.6) -> HungarianMatcher:
    """创建匈牙利匹配器实例"""
    return HungarianMatcher(similarity_threshold=threshold)


class MatchingBenchmark:
    """匹配算法性能基准测试"""
    
    @staticmethod
    def benchmark_matching_speed(
        matcher: HungarianMatcher,
        test_cases: List[Tuple[int, int]]  # (table1_size, table2_size)
    ) -> Dict[str, float]:
        """测试不同表大小下的匹配速度"""
        import time
        import random
        
        results = {}
        
        for n1, n2 in test_cases:
            # 生成测试数据
            embeddings1 = [[random.random() for _ in range(384)] for _ in range(n1)]
            embeddings2 = [[random.random() for _ in range(384)] for _ in range(n2)]
            
            # 测试匹配速度
            start_time = time.time()
            similarity_matrix = matcher.compute_similarity_matrix(embeddings1, embeddings2)
            matcher.find_optimal_matching(similarity_matrix)
            end_time = time.time()
            
            case_name = f"{n1}x{n2}"
            results[case_name] = end_time - start_time
        
        return results
    
    @staticmethod
    def compare_thresholds(
        matcher: HungarianMatcher,
        table1_embeddings: List[List[float]],
        table2_embeddings: List[List[float]],
        thresholds: List[float] = [0.3, 0.5, 0.7, 0.8, 0.9]
    ) -> Dict[float, Dict[str, Any]]:
        """比较不同阈值下的匹配效果"""
        similarity_matrix = matcher.compute_similarity_matrix(table1_embeddings, table2_embeddings)
        results = {}
        
        for threshold in thresholds:
            total_score, matches, _ = matcher.find_optimal_matching(similarity_matrix, threshold)
            results[threshold] = {
                "match_count": len(matches),
                "total_score": total_score,
                "average_similarity": total_score / len(matches) if matches else 0
            }
        
        return results