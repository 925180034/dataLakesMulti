"""
放松限制的优化工作流 - 提高召回率
通过放松metadata到vector的过滤条件来提升精度
"""

import logging
import asyncio
import time
import numpy as np
from typing import Dict, Any, List, Tuple, Optional, Set
from dataclasses import dataclass
from collections import defaultdict

from src.core.models import AgentState, TableInfo, ColumnInfo
from src.core.ultra_optimized_workflow import UltraOptimizedWorkflow, EvaluationMetrics

logger = logging.getLogger(__name__)


class RelaxedOptimizedWorkflow(UltraOptimizedWorkflow):
    """放松限制的优化工作流
    
    核心改进：
    1. 增加metadata候选数量
    2. 降低筛选阈值
    3. 增加vector候选数量
    4. 更宽松的相似度阈值
    5. 保留更多中等质量候选
    """
    
    def __init__(self):
        super().__init__()
        
        # 适度放松参数设置（平衡性能和召回率）
        self.max_metadata_candidates = 80   # 从50增加到80（适度增加）
        self.max_vector_candidates = 15     # 从20减少到15（控制时间）
        self.max_llm_candidates = 3         # 从5减少到3（加快速度）
        self.early_stop_threshold = 0.92    # 从0.90提高到0.92（略微提高）
        
        # 适度放松阈值
        self.metadata_threshold = 0.35      # 从0.5降低到0.35（适度放松）
        self.vector_threshold = 0.25        # 从0.3降低到0.25（适度放松）
        self.final_threshold = 0.2          # 最终阈值适度放松
        
        logger.info(f"RelaxedOptimizedWorkflow initialized with relaxed parameters:")
        logger.info(f"  - Metadata candidates: {self.max_metadata_candidates}")
        logger.info(f"  - Vector candidates: {self.max_vector_candidates}")
        logger.info(f"  - Metadata threshold: {self.metadata_threshold}")
        logger.info(f"  - Vector threshold: {self.vector_threshold}")
    
    async def _ultra_metadata_filter(
        self,
        query_tables: List[TableInfo],
        all_table_names: List[str],
        top_k: int = 100  # 增加默认值
    ) -> List[Tuple[str, float]]:
        """放松的元数据筛选"""
        results = []
        
        for query_table in query_tables:
            # 初步筛选更多候选（3倍）
            candidates = self.metadata_filter.filter_candidates(
                query_table,
                all_table_names,
                top_k=top_k * 3  # 从2倍增加到3倍
            )
            
            # 更宽松的二次筛选
            filtered = []
            for table_name, score in candidates:
                # 降低阈值，保留更多候选
                if score > self.metadata_threshold:  # 使用更低的阈值
                    filtered.append((table_name, score))
                elif score > self.metadata_threshold * 0.5 and len(filtered) < top_k // 2:
                    # 即使分数较低，也保留一些候选以提高召回率
                    filtered.append((table_name, score * 0.8))  # 轻微降低分数
                
                if len(filtered) >= top_k:
                    break
            
            # 如果筛选结果太少，补充一些低分候选
            if len(filtered) < 20:  # 确保至少有20个候选
                for table_name, score in candidates[len(filtered):]:
                    filtered.append((table_name, score * 0.6))  # 大幅降低分数
                    if len(filtered) >= 20:
                        break
            
            results.extend(filtered)
        
        # 去重并排序
        unique_results = {}
        for name, score in results:
            if name not in unique_results or score > unique_results[name]:
                unique_results[name] = score
        
        # 返回更多候选
        sorted_results = sorted(unique_results.items(), key=lambda x: x[1], reverse=True)
        return sorted_results[:min(top_k, len(sorted_results))]
    
    async def _ultra_vector_search(
        self,
        query_tables: List[TableInfo],
        candidate_names: List[str],
        k: int = 30  # 增加默认值
    ) -> Dict[str, List[Tuple[str, float]]]:
        """放松的向量搜索"""
        # 批量向量搜索，返回更多结果
        results = await self.batch_vector_search.batch_search_tables(
            query_tables,
            candidate_names,
            k=min(k, len(candidate_names)),  # 确保不超过候选数
            threshold=self.vector_threshold  # 使用更低的阈值
        )
        
        # 格式化结果，保留更多候选
        formatted = {}
        for query_name, search_results in results.items():
            filtered_results = []
            
            for r in search_results:
                # 主要候选：高分结果
                if r.score > self.vector_threshold:
                    filtered_results.append((r.item_id, r.score))
                # 补充候选：中等分数也保留
                elif r.score > self.vector_threshold * 0.5 and len(filtered_results) < k:
                    filtered_results.append((r.item_id, r.score * 0.9))
            
            # 确保有足够的候选
            if len(filtered_results) < 10:  # 至少保留10个
                for r in search_results[len(filtered_results):]:
                    if r.score > 0.1:  # 极低阈值
                        filtered_results.append((r.item_id, r.score * 0.7))
                        if len(filtered_results) >= 10:
                            break
            
            formatted[query_name] = filtered_results[:k]
        
        return formatted
    
    def _should_early_stop(
        self,
        vector_results: Dict[str, List[Tuple[str, float]]]
    ) -> bool:
        """更严格的早期终止条件（减少误判）"""
        for query_name, candidates in vector_results.items():
            if candidates:
                # 需要更高的分数才触发早停
                if candidates[0][1] > self.early_stop_threshold:
                    # 额外检查：第二个候选也要有高分
                    if len(candidates) > 1 and candidates[1][1] > self.early_stop_threshold * 0.8:
                        return True
        return False
    
    def _format_vector_results(
        self,
        vector_results: Dict[str, List[Tuple[str, float]]]
    ) -> List:
        """格式化向量搜索结果，保留更多候选"""
        from src.core.models import TableMatchResult
        results = []
        
        for query_name, candidates in vector_results.items():
            # 保留更多结果（前20个而不是前10个）
            for table_name, score in candidates[:20]:
                # 使用更宽松的阈值
                if score > self.final_threshold:
                    results.append(TableMatchResult(
                        source_table=query_name,
                        target_table=table_name,
                        score=score * 100,
                        matched_columns=[],
                        evidence={
                            "match_type": "vector_similarity",
                            "method": "relaxed_filtering",
                            "threshold": self.final_threshold
                        }
                    ))
        
        return results
    
    def _format_final_results(
        self,
        llm_results: Dict[str, List[Dict[str, Any]]]
    ) -> List:
        """格式化LLM结果，保留更多候选"""
        from src.core.models import TableMatchResult
        results = []
        
        for query_name, matches in llm_results.items():
            for match in matches[:10]:  # 保留前10个
                # 更宽松的分数阈值
                if match.get('score', 0) > 0.1:  # 从0.3降低到0.1
                    results.append(TableMatchResult(
                        source_table=query_name,
                        target_table=match['target_table'],
                        score=match.get('score', 0.5) * 100,
                        matched_columns=match.get('matched_columns', []),
                        evidence=match.get('evidence', {
                            "method": "llm_validation",
                            "relaxed": True
                        })
                    ))
        
        return results


def create_relaxed_workflow():
    """创建放松限制的工作流实例"""
    return RelaxedOptimizedWorkflow()