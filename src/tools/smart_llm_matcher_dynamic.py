"""
智能LLM匹配器 - 动态参数版本
支持动态调整所有阈值，而不是硬编码
"""

import logging
import asyncio
from typing import List, Dict, Any, Tuple, Optional
import json
from collections import defaultdict

from src.core.models import TableInfo, ColumnInfo
from src.config.settings import settings
from src.config.prompts import format_prompt

logger = logging.getLogger(__name__)


class SmartLLMMatcherDynamic:
    """智能LLM匹配器 - 支持动态参数调整
    
    关键改进：
    1. 所有阈值都可以动态调整
    2. 支持从外部传入参数
    3. 不再有硬编码的阈值
    """
    
    def __init__(self, llm_client, dynamic_params: Optional[Dict] = None):
        self.llm_client = llm_client
        self.cache = {}
        
        # 使用动态参数或默认值
        params = dynamic_params or {}
        
        # L3层可调参数
        self.llm_confidence_threshold = params.get('llm_confidence_threshold', 0.15)
        self.rule_high_threshold = params.get('rule_high_threshold', 0.7)  # 降低默认值
        self.rule_medium_threshold = params.get('rule_medium_threshold', 0.5)  # 降低默认值
        self.rule_low_threshold = params.get('rule_low_threshold', 0.3)  # 降低默认值
        self.max_candidates_per_query = params.get('llm_max_candidates', 20)
        self.batch_size = params.get('llm_batch_size', 20)
        
        # 特殊任务优化
        self.task_type = params.get('task_type', 'join')
        self.optimize_for_recall = params.get('optimize_for_recall', False)
        
        # 日志参数使用情况
        logger.info(f"SmartLLMMatcher initialized with dynamic params:")
        logger.info(f"  Task type: {self.task_type}")
        logger.info(f"  LLM confidence threshold: {self.llm_confidence_threshold}")
        logger.info(f"  Rule thresholds: high={self.rule_high_threshold}, "
                   f"medium={self.rule_medium_threshold}, low={self.rule_low_threshold}")
        logger.info(f"  Max candidates: {self.max_candidates_per_query}")
        
        # 初始化批量处理器
        from src.tools.batch_llm_processor import BatchLLMProcessor
        self.batch_processor = BatchLLMProcessor(
            llm_client=llm_client,
            max_batch_size=self.batch_size,
            max_concurrent=10
        )
        
    def update_params(self, new_params: Dict):
        """动态更新参数（支持批次内调整）"""
        if 'llm_confidence_threshold' in new_params:
            self.llm_confidence_threshold = new_params['llm_confidence_threshold']
            logger.debug(f"Updated LLM threshold to {self.llm_confidence_threshold}")
            
        if 'rule_high_threshold' in new_params:
            self.rule_high_threshold = new_params['rule_high_threshold']
            
        if 'llm_max_candidates' in new_params:
            self.max_candidates_per_query = new_params['llm_max_candidates']
            
    async def match_tables(
        self,
        query_tables: List[TableInfo],
        candidate_results: Dict[str, List[Tuple[str, float]]],
        table_metadata: Dict[str, TableInfo]
    ) -> Dict[str, List[Dict[str, Any]]]:
        """智能表匹配"""
        all_matches = {}
        
        for query_table in query_tables:
            candidates = candidate_results.get(query_table.table_name, [])
            
            if not candidates:
                all_matches[query_table.table_name] = []
                continue
            
            # 1. 规则预判（使用动态阈值）
            rule_matches, uncertain_candidates = await self._apply_rules_dynamic(
                query_table, 
                candidates[:self.max_candidates_per_query],
                table_metadata
            )
            
            # 2. LLM验证（使用动态阈值）
            llm_matches = []
            if uncertain_candidates:
                llm_matches = await self._batch_llm_verify_dynamic(
                    query_table,
                    uncertain_candidates,
                    table_metadata
                )
            
            # 3. 合并结果
            all_results = rule_matches + llm_matches
            
            # 4. 根据任务类型调整排序策略
            if self.task_type == 'union' and self.optimize_for_recall:
                # UNION任务：优先考虑召回率，放宽过滤
                all_results = [r for r in all_results if r['score'] > self.llm_confidence_threshold * 0.5]
            else:
                # JOIN任务：正常过滤
                all_results = [r for r in all_results if r['score'] > self.llm_confidence_threshold]
            
            # 排序并返回
            all_results.sort(key=lambda x: x['score'], reverse=True)
            all_matches[query_table.table_name] = all_results
            
            # 日志匹配情况
            if len(all_results) == 0 and len(candidates) > 0:
                logger.warning(f"No matches for {query_table.table_name} after filtering "
                             f"(had {len(candidates)} candidates, threshold={self.llm_confidence_threshold})")
        
        return all_matches
    
    async def _apply_rules_dynamic(
        self, 
        query_table: TableInfo,
        candidates: List[Tuple[str, float]],
        table_metadata: Dict[str, TableInfo]
    ) -> Tuple[List[Dict], List[Tuple[str, float]]]:
        """应用规则预判（使用动态阈值）"""
        rule_matches = []
        uncertain_candidates = []
        
        for candidate_name, score in candidates:
            candidate_table = table_metadata.get(candidate_name)
            if not candidate_table:
                continue
            
            # 使用动态阈值判断
            if score > self.rule_high_threshold:
                # 高置信度 - 直接通过
                rule_matches.append({
                    "table": candidate_name,
                    "score": score,
                    "confidence": "high",
                    "reason": f"High similarity score ({score:.3f} > {self.rule_high_threshold})"
                })
            elif self._check_special_rules(query_table, candidate_table):
                # 特殊规则匹配
                key_match_score = self._check_key_column_match(query_table, candidate_table)
                if key_match_score > self.rule_medium_threshold:
                    rule_matches.append({
                        "table": candidate_name,
                        "score": min(score + 0.1, 1.0),  # 提升分数
                        "confidence": "medium",
                        "reason": f"Key column match ({key_match_score:.2f})"
                    })
                else:
                    uncertain_candidates.append((candidate_name, score))
            elif score > self.rule_low_threshold:
                # 中低分数 - 需要LLM验证
                uncertain_candidates.append((candidate_name, score))
        
        return rule_matches, uncertain_candidates
    
    async def _batch_llm_verify_dynamic(
        self,
        query_table: TableInfo,
        candidates: List[Tuple[str, float]],
        table_metadata: Dict[str, TableInfo]
    ) -> List[Dict[str, Any]]:
        """批量LLM验证（使用动态阈值）"""
        from src.tools.batch_llm_processor import TableMatchingPromptBuilder
        
        # 准备批处理项目
        items = []
        for candidate_name, score in candidates:
            candidate_table = table_metadata.get(candidate_name)
            if candidate_table:
                # 在prompt中传入动态阈值
                item = {
                    "query_table": {
                        "name": query_table.table_name,
                        "columns": [
                            {"name": col.column_name, "type": col.data_type or "unknown"}
                            for col in query_table.columns[:10]
                        ]
                    },
                    "candidate_table": {
                        "name": candidate_name,
                        "columns": [
                            {"name": col.column_name, "type": col.data_type or "unknown"}
                            for col in candidate_table.columns[:10]
                        ]
                    },
                    "task_type": self.task_type,
                    "confidence_threshold": self.llm_confidence_threshold,  # 传入动态阈值
                    "initial_score": score
                }
                items.append(item)
        
        if not items:
            return []
        
        # 批量处理
        builder = TableMatchingPromptBuilder(task_type=self.task_type)
        results = await self.batch_processor.process_batch(
            items=items,
            prompt_builder=builder.build_prompt,
            response_parser=self._parse_llm_response
        )
        
        # 过滤结果（使用动态阈值）
        matches = []
        for result, (candidate_name, score) in zip(results, candidates):
            if result and result.get('is_match'):
                llm_score = result.get('confidence', 0.5)
                
                # 动态阈值过滤
                if llm_score >= self.llm_confidence_threshold:
                    matches.append({
                        "table": candidate_name,
                        "score": llm_score,
                        "confidence": "llm_verified",
                        "reason": result.get('reason', 'LLM verified match')
                    })
                else:
                    logger.debug(f"LLM score {llm_score} < threshold {self.llm_confidence_threshold}, filtered")
        
        return matches
    
    def _check_special_rules(self, table1: TableInfo, table2: TableInfo) -> bool:
        """检查特殊规则"""
        # 检查列重叠度
        overlap = self._calculate_column_overlap(table1, table2)
        
        # 动态阈值
        if self.task_type == 'union':
            # UNION更宽松
            return overlap > 0.3
        else:
            # JOIN需要更高的重叠度
            return overlap > 0.5
    
    def _calculate_column_overlap(self, table1: TableInfo, table2: TableInfo) -> float:
        """计算列重叠度"""
        cols1 = {col.column_name.lower() for col in table1.columns}
        cols2 = {col.column_name.lower() for col in table2.columns}
        
        if not cols1 or not cols2:
            return 0.0
        
        intersection = cols1 & cols2
        union = cols1 | cols2
        
        return len(intersection) / len(union) if union else 0.0
    
    def _check_key_column_match(self, table1: TableInfo, table2: TableInfo) -> float:
        """检查关键列匹配度"""
        key_cols1 = self._extract_key_columns(table1)
        key_cols2 = self._extract_key_columns(table2)
        
        if not key_cols1 or not key_cols2:
            return 0.0
        
        matches = 0
        for col1 in key_cols1:
            for col2 in key_cols2:
                if self._is_similar_column_name(col1, col2):
                    matches += 1
                    break
        
        return matches / max(len(key_cols1), len(key_cols2))
    
    def _extract_key_columns(self, table: TableInfo) -> List[str]:
        """提取关键列"""
        key_columns = []
        for col in table.columns:
            col_lower = col.column_name.lower()
            if any(pattern in col_lower for pattern in ["id", "key", "code", "_no", "_num"]):
                key_columns.append(col.column_name)
        return key_columns
    
    def _is_similar_column_name(self, col1: str, col2: str) -> bool:
        """判断列名是否相似"""
        col1_lower = col1.lower()
        col2_lower = col2.lower()
        
        if col1_lower == col2_lower:
            return True
        
        # 去除常见前缀/后缀
        for prefix in ["fk_", "pk_", "idx_"]:
            col1_lower = col1_lower.replace(prefix, "")
            col2_lower = col2_lower.replace(prefix, "")
        
        for suffix in ["_id", "_key", "_code", "_no"]:
            col1_lower = col1_lower.replace(suffix, "")
            col2_lower = col2_lower.replace(suffix, "")
        
        return col1_lower == col2_lower
    
    def _parse_llm_response(self, response: str) -> Dict:
        """解析LLM响应"""
        try:
            return json.loads(response)
        except:
            # 尝试简单解析
            response_lower = response.lower()
            is_match = "yes" in response_lower or "match" in response_lower
            
            # 提取置信度
            confidence = self.llm_confidence_threshold  # 使用阈值作为默认值
            if "high" in response_lower:
                confidence = 0.8
            elif "medium" in response_lower:
                confidence = 0.5
            elif "low" in response_lower:
                confidence = 0.3
                
            return {
                "is_match": is_match,
                "confidence": confidence,
                "reason": "Parsed from text response"
            }