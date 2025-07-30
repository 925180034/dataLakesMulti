"""
混合相似度计算引擎
基于论文方法实现SMD和SLD两种场景的相似度计算
"""

from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import difflib
import re
from collections import Counter
import logging
from abc import ABC, abstractmethod

from src.core.models import ColumnInfo, TableInfo

logger = logging.getLogger(__name__)


class SimilarityCalculator(ABC):
    """相似度计算器抽象基类"""
    
    @abstractmethod
    def calculate_similarity(self, col1: ColumnInfo, col2: ColumnInfo) -> float:
        """计算两列之间的相似度"""
        pass


class TextSimilarityCalculator:
    """文本特征相似度计算器"""
    
    def __init__(self, 
                 levenshtein_weight: float = 0.4,
                 sequence_weight: float = 0.3, 
                 jaccard_weight: float = 0.3):
        """
        初始化文本相似度计算器
        
        Args:
            levenshtein_weight: Levenshtein相似度权重
            sequence_weight: 序列相似度权重  
            jaccard_weight: Jaccard相似度权重
        """
        self.levenshtein_weight = levenshtein_weight
        self.sequence_weight = sequence_weight
        self.jaccard_weight = jaccard_weight
        
        # 权重归一化
        total_weight = levenshtein_weight + sequence_weight + jaccard_weight
        self.levenshtein_weight /= total_weight
        self.sequence_weight /= total_weight
        self.jaccard_weight /= total_weight
    
    def _levenshtein_similarity(self, s1: str, s2: str) -> float:
        """计算Levenshtein相似度"""
        if not s1 or not s2:
            return 0.0
        
        # 转换为小写进行比较
        s1, s2 = s1.lower(), s2.lower()
        
        # 计算编辑距离
        len1, len2 = len(s1), len(s2)
        if len1 == 0:
            return 0.0 if len2 > 0 else 1.0
        if len2 == 0:
            return 0.0
        
        # 动态规划计算编辑距离
        dp = [[0] * (len2 + 1) for _ in range(len1 + 1)]
        
        for i in range(len1 + 1):
            dp[i][0] = i
        for j in range(len2 + 1):
            dp[0][j] = j
        
        for i in range(1, len1 + 1):
            for j in range(1, len2 + 1):
                if s1[i-1] == s2[j-1]:
                    dp[i][j] = dp[i-1][j-1]
                else:
                    dp[i][j] = 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])
        
        edit_distance = dp[len1][len2]
        max_len = max(len1, len2)
        
        return 1.0 - (edit_distance / max_len)
    
    def _sequence_similarity(self, s1: str, s2: str) -> float:
        """计算序列相似度（使用difflib）"""
        if not s1 or not s2:
            return 0.0
        
        return difflib.SequenceMatcher(None, s1.lower(), s2.lower()).ratio()
    
    def _tokenize(self, text: str) -> set:
        """将文本分词为token集合"""
        # 分割驼峰命名、下划线、点号等
        tokens = re.findall(r'[A-Z]?[a-z]+|[A-Z]+(?=[A-Z][a-z]|\b)|[0-9]+', text)
        return set(token.lower() for token in tokens if len(token) > 1)
    
    def _jaccard_similarity(self, s1: str, s2: str) -> float:
        """计算基于词汇的Jaccard相似度"""
        if not s1 or not s2:
            return 0.0
        
        tokens1 = self._tokenize(s1)
        tokens2 = self._tokenize(s2)
        
        if not tokens1 and not tokens2:
            return 1.0
        if not tokens1 or not tokens2:
            return 0.0
        
        intersection = tokens1.intersection(tokens2)
        union = tokens1.union(tokens2)
        
        return len(intersection) / len(union)
    
    def calculate_name_similarity(self, name1: str, name2: str) -> float:
        """
        计算列名相似度（论文公式1）
        Simname(asi, atj) = α1·SimLEV + α2·Simseq + α3·SimJAC
        """
        if not name1 or not name2:
            return 0.0
        
        lev_sim = self._levenshtein_similarity(name1, name2)
        seq_sim = self._sequence_similarity(name1, name2)
        jac_sim = self._jaccard_similarity(name1, name2)
        
        combined_similarity = (
            self.levenshtein_weight * lev_sim +
            self.sequence_weight * seq_sim +
            self.jaccard_weight * jac_sim
        )
        
        logger.debug(f"Name similarity '{name1}' vs '{name2}': "
                    f"LEV={lev_sim:.3f}, SEQ={seq_sim:.3f}, JAC={jac_sim:.3f}, "
                    f"Combined={combined_similarity:.3f}")
        
        return combined_similarity


class StatisticalSimilarityCalculator:
    """统计分布相似度计算器"""
    
    def __init__(self,
                 mean_weight: float = 0.4,
                 std_weight: float = 0.3,
                 quantile_weight: float = 0.3):
        """初始化统计相似度计算器"""
        self.mean_weight = mean_weight
        self.std_weight = std_weight  
        self.quantile_weight = quantile_weight
        
        # 权重归一化
        total_weight = mean_weight + std_weight + quantile_weight
        self.mean_weight /= total_weight
        self.std_weight /= total_weight
        self.quantile_weight /= total_weight
    
    def _extract_numeric_values(self, values: List[Any]) -> List[float]:
        """提取数值类型的值"""
        numeric_values = []
        for val in values:
            try:
                if val is not None and val != "":
                    numeric_val = float(str(val).strip())
                    if not np.isnan(numeric_val) and not np.isinf(numeric_val):
                        numeric_values.append(numeric_val)
            except (ValueError, TypeError):
                continue
        return numeric_values
    
    def _normalized_difference(self, val1: float, val2: float, scale: float = 1.0) -> float:
        """计算归一化差异（0-1之间，越接近0越相似）"""
        if scale == 0:
            return 0.0 if val1 == val2 else 1.0
        
        diff = abs(val1 - val2) / scale
        return min(diff, 1.0)
    
    def _calculate_mean_similarity(self, values1: List[float], values2: List[float]) -> float:
        """计算均值相似度"""
        if not values1 or not values2:
            return 0.0
        
        mean1 = np.mean(values1)
        mean2 = np.mean(values2)
        
        # 使用两个数据集的标准差作为缩放因子
        all_values = values1 + values2
        scale = max(np.std(all_values), abs(mean1), abs(mean2), 1.0)
        
        diff = self._normalized_difference(mean1, mean2, scale)
        return 1.0 - diff
    
    def _calculate_std_similarity(self, values1: List[float], values2: List[float]) -> float:
        """计算标准差相似度"""
        if not values1 or not values2:
            return 0.0
        
        std1 = np.std(values1)
        std2 = np.std(values2)
        
        # 使用较大的标准差作为缩放因子
        scale = max(std1, std2, 1.0)
        
        diff = self._normalized_difference(std1, std2, scale)
        return 1.0 - diff
    
    def _calculate_quantile_similarity(self, values1: List[float], values2: List[float]) -> float:
        """计算分位数相似度"""
        if not values1 or not values2:
            return 0.0
        
        # 计算关键分位数
        quantiles = [0.25, 0.5, 0.75]
        similarities = []
        
        for q in quantiles:
            q1 = np.percentile(values1, q * 100)
            q2 = np.percentile(values2, q * 100)
            
            # 使用数据范围作为缩放因子
            range1 = np.max(values1) - np.min(values1)
            range2 = np.max(values2) - np.min(values2)
            scale = max(range1, range2, abs(q1), abs(q2), 1.0)
            
            diff = self._normalized_difference(q1, q2, scale)
            similarities.append(1.0 - diff)
        
        return np.mean(similarities)
    
    def calculate_numeric_similarity(self, values1: List[Any], values2: List[Any]) -> float:
        """
        计算数值属性相似度（论文公式2）
        Simnumeric = β1·Simmean + β2·Simstd + β3·Simquantile
        """
        numeric_values1 = self._extract_numeric_values(values1)
        numeric_values2 = self._extract_numeric_values(values2)
        
        if not numeric_values1 or not numeric_values2:
            logger.debug("No numeric values found for comparison")
            return 0.0
        
        mean_sim = self._calculate_mean_similarity(numeric_values1, numeric_values2)
        std_sim = self._calculate_std_similarity(numeric_values1, numeric_values2)
        quantile_sim = self._calculate_quantile_similarity(numeric_values1, numeric_values2)
        
        combined_similarity = (
            self.mean_weight * mean_sim +
            self.std_weight * std_sim +
            self.quantile_weight * quantile_sim
        )
        
        logger.debug(f"Numeric similarity: MEAN={mean_sim:.3f}, STD={std_sim:.3f}, "
                    f"QUANTILE={quantile_sim:.3f}, Combined={combined_similarity:.3f}")
        
        return combined_similarity
    
    def _extract_categorical_values(self, values: List[Any]) -> List[str]:
        """提取分类值"""
        categorical_values = []
        for val in values:
            if val is not None and val != "":
                str_val = str(val).strip().lower()
                if str_val:
                    categorical_values.append(str_val)
        return categorical_values
    
    def _jaccard_similarity(self, set1: set, set2: set) -> float:
        """计算Jaccard相似度"""
        if not set1 and not set2:
            return 1.0
        if not set1 or not set2:
            return 0.0
        
        intersection = set1.intersection(set2)
        union = set1.union(set2)
        
        return len(intersection) / len(union)
    
    def _distribution_similarity(self, values1: List[str], values2: List[str]) -> float:
        """计算分布相似度（基于值频率）"""
        if not values1 or not values2:
            return 0.0
        
        counter1 = Counter(values1)
        counter2 = Counter(values2)
        
        # 获取所有唯一值
        all_values = set(counter1.keys()) | set(counter2.keys())
        
        # 计算频率向量
        freq1 = np.array([counter1.get(val, 0) for val in all_values])
        freq2 = np.array([counter2.get(val, 0) for val in all_values])
        
        # 归一化为概率分布
        freq1 = freq1 / np.sum(freq1)
        freq2 = freq2 / np.sum(freq2)
        
        # 计算余弦相似度
        dot_product = np.dot(freq1, freq2)
        norm1 = np.linalg.norm(freq1)
        norm2 = np.linalg.norm(freq2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2)
    
    def calculate_categorical_similarity(self, values1: List[Any], values2: List[Any], 
                                       jaccard_weight: float = 0.6,
                                       distribution_weight: float = 0.4) -> float:
        """
        计算分类属性相似度（论文公式3）
        Simcat = γ1·SimJAC + γ2·Simdist
        """
        cat_values1 = self._extract_categorical_values(values1)
        cat_values2 = self._extract_categorical_values(values2)
        
        if not cat_values1 or not cat_values2:
            logger.debug("No categorical values found for comparison")
            return 0.0
        
        # Jaccard相似度
        set1 = set(cat_values1)
        set2 = set(cat_values2)
        jaccard_sim = self._jaccard_similarity(set1, set2)
        
        # 分布相似度
        dist_sim = self._distribution_similarity(cat_values1, cat_values2)
        
        combined_similarity = jaccard_weight * jaccard_sim + distribution_weight * dist_sim
        
        logger.debug(f"Categorical similarity: JAC={jaccard_sim:.3f}, DIST={dist_sim:.3f}, "
                    f"Combined={combined_similarity:.3f}")
        
        return combined_similarity


class HybridSimilarityEngine:
    """混合相似度计算引擎 - 支持SMD和SLD两种场景"""
    
    def __init__(self):
        self.text_calculator = TextSimilarityCalculator()
        self.statistical_calculator = StatisticalSimilarityCalculator()
        
        # SMD场景权重配置（仅元数据）
        self.smd_weights = {
            'name': 1.0,        # 只有名称相似度
            'structural': 0.0,   # 无结构信息
            'semantic': 0.0      # 无实例数据
        }
        
        # SLD场景权重配置（完整数据）
        self.sld_weights = {
            'name': 0.4,         # 名称相似度权重降低
            'structural': 0.2,   # 结构特征权重
            'semantic': 0.4      # 语义特征权重提高
        }
    
    def _is_numeric_column(self, column_info: ColumnInfo) -> bool:
        """判断列是否为数值类型"""
        if column_info.data_type:
            numeric_types = ['int', 'integer', 'float', 'double', 'decimal', 'number', 'numeric']
            return any(t in column_info.data_type.lower() for t in numeric_types)
        
        # 基于样本值判断
        if column_info.sample_values:
            numeric_count = 0
            for val in column_info.sample_values[:5]:  # 检查前5个值
                try:
                    if val is not None and val != "":
                        float(str(val).strip())
                        numeric_count += 1
                except (ValueError, TypeError):
                    continue
            
            return numeric_count >= len(column_info.sample_values) * 0.8
        
        return False
    
    def _calculate_structural_similarity(self, col1: ColumnInfo, col2: ColumnInfo) -> float:
        """计算结构特征相似度"""
        similarities = []
        
        # 数据类型相似度
        if col1.data_type and col2.data_type:
            type1 = col1.data_type.lower()
            type2 = col2.data_type.lower()
            
            if type1 == type2:
                type_sim = 1.0
            elif any(t in type1 and t in type2 for t in ['int', 'float', 'number']):
                type_sim = 0.8  # 数值类型之间相似
            elif any(t in type1 and t in type2 for t in ['str', 'text', 'char']):
                type_sim = 0.8  # 文本类型之间相似
            else:
                type_sim = 0.0
            
            similarities.append(type_sim)
        
        # 空值比例相似度（如果有统计信息）
        if hasattr(col1, 'null_count') and hasattr(col2, 'null_count'):
            if col1.null_count is not None and col2.null_count is not None:
                # 简化的空值比例比较
                null_sim = 1.0 - abs(col1.null_count - col2.null_count) / 100.0
                similarities.append(max(0.0, null_sim))
        
        return np.mean(similarities) if similarities else 0.0
    
    def _calculate_semantic_similarity(self, col1: ColumnInfo, col2: ColumnInfo) -> float:
        """计算语义特征相似度（基于实例数据）"""
        if not col1.sample_values or not col2.sample_values:
            return 0.0
        
        # 判断数据类型并选择合适的相似度计算方法
        is_numeric1 = self._is_numeric_column(col1)
        is_numeric2 = self._is_numeric_column(col2)
        
        if is_numeric1 and is_numeric2:
            # 两列都是数值型，使用统计分布相似度
            return self.statistical_calculator.calculate_numeric_similarity(
                col1.sample_values, col2.sample_values
            )
        elif not is_numeric1 and not is_numeric2:
            # 两列都是分类型，使用分类相似度
            return self.statistical_calculator.calculate_categorical_similarity(
                col1.sample_values, col2.sample_values
            )
        else:
            # 类型不匹配，相似度为0
            return 0.0
    
    def calculate_column_similarity(self, col1: ColumnInfo, col2: ColumnInfo, 
                                  scenario: str = "SLD") -> Dict[str, float]:
        """
        计算列相似度
        
        Args:
            col1: 源列信息
            col2: 目标列信息
            scenario: 场景类型 ("SMD" 或 "SLD")
            
        Returns:
            包含各种相似度分数的字典
        """
        results = {}
        
        # 计算名称相似度
        name_sim = self.text_calculator.calculate_name_similarity(
            col1.column_name, col2.column_name
        )
        results['name_similarity'] = name_sim
        
        # 计算结构相似度
        struct_sim = self._calculate_structural_similarity(col1, col2)
        results['structural_similarity'] = struct_sim
        
        # 计算语义相似度（仅在SLD场景下）
        if scenario == "SLD":
            semantic_sim = self._calculate_semantic_similarity(col1, col2)
            results['semantic_similarity'] = semantic_sim
        else:
            results['semantic_similarity'] = 0.0
        
        # 场景特定的权重组合
        weights = self.smd_weights if scenario == "SMD" else self.sld_weights
        
        combined_similarity = (
            weights['name'] * name_sim +
            weights['structural'] * struct_sim +
            weights['semantic'] * results['semantic_similarity']
        )
        
        results['combined_similarity'] = combined_similarity
        results['scenario'] = scenario
        results['weights_used'] = weights.copy()
        
        logger.debug(f"Column similarity ({scenario}): {col1.full_name} vs {col2.full_name} = {combined_similarity:.3f}")
        
        return results
    
    def calculate_table_similarity(self, table1: TableInfo, table2: TableInfo,
                                 scenario: str = "SLD") -> Dict[str, Any]:
        """
        计算表相似度（基于列匹配聚合）
        
        Args:
            table1: 源表信息
            table2: 目标表信息  
            scenario: 场景类型
            
        Returns:
            表相似度结果
        """
        if not table1.columns or not table2.columns:
            return {
                'table_similarity': 0.0,
                'matched_columns': [],
                'scenario': scenario
            }
        
        # 计算所有列对的相似度
        column_similarities = []
        matched_columns = []
        
        for col1 in table1.columns:
            best_match = None
            best_score = 0.0
            
            for col2 in table2.columns:
                sim_result = self.calculate_column_similarity(col1, col2, scenario)
                score = sim_result['combined_similarity']
                
                if score > best_score:
                    best_score = score
                    best_match = {
                        'source_column': col1.full_name,
                        'target_column': col2.full_name,
                        'similarity': score,
                        'details': sim_result
                    }
            
            if best_match and best_score > 0.5:  # 阈值可配置
                column_similarities.append(best_score)
                matched_columns.append(best_match)
        
        # 计算表级别相似度（匹配列的平均相似度）
        if column_similarities:
            table_similarity = np.mean(column_similarities)
        else:
            table_similarity = 0.0
        
        return {
            'table_similarity': table_similarity,
            'matched_columns': matched_columns,
            'match_ratio': len(matched_columns) / len(table1.columns),
            'scenario': scenario,
            'total_source_columns': len(table1.columns),
            'total_target_columns': len(table2.columns),
            'matched_count': len(matched_columns)
        }


# 全局实例
hybrid_similarity_engine = HybridSimilarityEngine()

# 别名兼容性
HybridSimilarityCalculator = HybridSimilarityEngine