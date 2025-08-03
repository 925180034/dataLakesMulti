"""
智能LLM匹配器 - 三层加速架构的第三层
通过规则预判和批量调用优化LLM使用
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


class SmartLLMMatcher:
    """智能LLM匹配器
    
    优化策略：
    1. 规则预判：使用启发式规则快速判断
    2. 批量验证：合并多个验证请求
    3. 智能截断：限制输入长度
    4. 结果缓存：避免重复调用
    """
    
    def __init__(self, llm_client):
        self.llm_client = llm_client
        self.cache = {}  # LLM调用缓存
        self.batch_size = 10  # 增加批量大小
        self.max_candidates_per_query = 10  # 减少每个查询的候选数，提高速度
        
        # 初始化批量处理器
        from src.tools.batch_llm_processor import BatchLLMProcessor
        self.batch_processor = BatchLLMProcessor(
            llm_client=llm_client,
            max_batch_size=10,
            max_concurrent=5
        )
        
    async def match_tables(
        self,
        query_tables: List[TableInfo],
        candidate_results: Dict[str, List[Tuple[str, float]]],
        table_metadata: Dict[str, TableInfo]
    ) -> Dict[str, List[Dict[str, Any]]]:
        """智能表匹配
        
        Args:
            query_tables: 查询表列表
            candidate_results: {query_table_name: [(candidate_name, score), ...]}
            table_metadata: 所有表的元数据
            
        Returns:
            {query_table_name: [{"table": name, "score": score, "reason": reason}, ...]}
        """
        all_matches = {}
        
        for query_table in query_tables:
            candidates = candidate_results.get(query_table.table_name, [])
            
            if not candidates:
                all_matches[query_table.table_name] = []
                continue
            
            # 1. 规则预判
            rule_matches, uncertain_candidates = await self._apply_rules(
                query_table, 
                candidates[:self.max_candidates_per_query],
                table_metadata
            )
            
            # 2. LLM验证（仅对不确定的）
            llm_matches = []
            if uncertain_candidates:
                llm_matches = await self._batch_llm_verify(
                    query_table,
                    uncertain_candidates,
                    table_metadata
                )
            
            # 3. 合并结果
            final_matches = self._merge_results(rule_matches, llm_matches)
            all_matches[query_table.table_name] = final_matches
        
        return all_matches
    
    async def _apply_rules(
        self,
        query_table: TableInfo,
        candidates: List[Tuple[str, float]],
        table_metadata: Dict[str, TableInfo]
    ) -> Tuple[List[Dict[str, Any]], List[Tuple[str, float]]]:
        """应用启发式规则
        
        Returns:
            (确定匹配的表, 需要LLM验证的表)
        """
        rule_matches = []
        uncertain = []
        
        for candidate_name, score in candidates:
            candidate_table = table_metadata.get(candidate_name)
            if not candidate_table:
                continue
            
            # 规则1: 高相似度 + 列重叠
            if score > 0.9:
                overlap_ratio = self._calculate_column_overlap(query_table, candidate_table)
                if overlap_ratio > 0.8:
                    rule_matches.append({
                        "table": candidate_name,
                        "score": score,
                        "reason": f"High similarity ({score:.2f}) with {overlap_ratio:.0%} column overlap",
                        "method": "rule"
                    })
                    continue
            
            # 规则2: 完全相同的列结构
            if self._has_identical_structure(query_table, candidate_table):
                rule_matches.append({
                    "table": candidate_name,
                    "score": min(score * 1.2, 1.0),  # 提升分数
                    "reason": "Identical table structure",
                    "method": "rule"
                })
                continue
            
            # 规则3: 关键列匹配
            key_match_score = self._check_key_column_match(query_table, candidate_table)
            if key_match_score > 0.8:
                rule_matches.append({
                    "table": candidate_name,
                    "score": score,
                    "reason": f"Key columns match (score: {key_match_score:.2f})",
                    "method": "rule"
                })
                continue
            
            # 规则4: 特殊模式匹配（如维度表、事实表）
            if self._check_special_pattern_match(query_table, candidate_table):
                rule_matches.append({
                    "table": candidate_name,
                    "score": score,
                    "reason": "Special pattern match (dimension/fact table)",
                    "method": "rule"
                })
                continue
            
            # 否则加入不确定列表
            if score > 0.6:  # 只考虑分数较高的
                uncertain.append((candidate_name, score))
        
        return rule_matches, uncertain
    
    def _calculate_column_overlap(self, table1: TableInfo, table2: TableInfo) -> float:
        """计算列重叠率"""
        cols1 = set(col.column_name.lower() for col in table1.columns)
        cols2 = set(col.column_name.lower() for col in table2.columns)
        
        if not cols1 or not cols2:
            return 0.0
        
        intersection = len(cols1 & cols2)
        union = len(cols1 | cols2)
        
        return intersection / union if union > 0 else 0.0
    
    def _has_identical_structure(self, table1: TableInfo, table2: TableInfo) -> bool:
        """检查是否有相同的表结构"""
        if len(table1.columns) != len(table2.columns):
            return False
        
        # 比较列名和类型
        cols1 = sorted([(c.column_name.lower(), self._normalize_type(c.data_type)) 
                       for c in table1.columns])
        cols2 = sorted([(c.column_name.lower(), self._normalize_type(c.data_type)) 
                       for c in table2.columns])
        
        return cols1 == cols2
    
    def _normalize_type(self, data_type: Optional[str]) -> str:
        """标准化数据类型"""
        if not data_type:
            return "unknown"
        
        type_lower = data_type.lower()
        if any(t in type_lower for t in ["int", "number", "numeric"]):
            return "numeric"
        elif any(t in type_lower for t in ["string", "text", "varchar"]):
            return "string"
        elif any(t in type_lower for t in ["date", "time"]):
            return "datetime"
        else:
            return "other"
    
    def _check_key_column_match(self, table1: TableInfo, table2: TableInfo) -> float:
        """检查关键列匹配度"""
        key_cols1 = self._extract_key_columns(table1)
        key_cols2 = self._extract_key_columns(table2)
        
        if not key_cols1 or not key_cols2:
            return 0.0
        
        # 计算关键列的匹配分数
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
            # 主键模式
            if any(pattern in col_lower for pattern in ["id", "key", "code", "_no", "_num"]):
                key_columns.append(col.column_name)
            # 外键模式
            elif col_lower.endswith("_id") or col_lower.endswith("_key"):
                key_columns.append(col.column_name)
        
        return key_columns
    
    def _is_similar_column_name(self, col1: str, col2: str) -> bool:
        """判断列名是否相似"""
        col1_lower = col1.lower()
        col2_lower = col2.lower()
        
        # 完全相同
        if col1_lower == col2_lower:
            return True
        
        # 去除常见前缀/后缀后相同
        for prefix in ["fk_", "pk_", "idx_"]:
            if col1_lower.startswith(prefix):
                col1_lower = col1_lower[len(prefix):]
            if col2_lower.startswith(prefix):
                col2_lower = col2_lower[len(prefix):]
        
        for suffix in ["_id", "_key", "_code", "_no"]:
            if col1_lower.endswith(suffix):
                col1_lower = col1_lower[:-len(suffix)]
            if col2_lower.endswith(suffix):
                col2_lower = col2_lower[:-len(suffix)]
        
        return col1_lower == col2_lower
    
    def _check_special_pattern_match(self, table1: TableInfo, table2: TableInfo) -> bool:
        """检查特殊模式匹配（维度表、事实表等）"""
        # 检查是否都是维度表
        if self._is_dimension_table(table1) and self._is_dimension_table(table2):
            # 维度表通常有相似的结构
            return self._calculate_column_overlap(table1, table2) > 0.6
        
        # 检查是否都是事实表
        if self._is_fact_table(table1) and self._is_fact_table(table2):
            # 事实表通常有多个外键
            fk_ratio1 = self._get_foreign_key_ratio(table1)
            fk_ratio2 = self._get_foreign_key_ratio(table2)
            return abs(fk_ratio1 - fk_ratio2) < 0.2
        
        return False
    
    def _is_dimension_table(self, table: TableInfo) -> bool:
        """判断是否为维度表"""
        table_lower = table.table_name.lower()
        return any(pattern in table_lower for pattern in ["dim_", "dimension_", "d_"])
    
    def _is_fact_table(self, table: TableInfo) -> bool:
        """判断是否为事实表"""
        table_lower = table.table_name.lower()
        return any(pattern in table_lower for pattern in ["fact_", "f_", "agg_"])
    
    def _get_foreign_key_ratio(self, table: TableInfo) -> float:
        """获取外键比例"""
        fk_count = sum(1 for col in table.columns 
                      if col.column_name.lower().endswith(("_id", "_key")))
        return fk_count / len(table.columns) if table.columns else 0.0
    
    async def _batch_llm_verify(
        self,
        query_table: TableInfo,
        candidates: List[Tuple[str, float]],
        table_metadata: Dict[str, TableInfo]
    ) -> List[Dict[str, Any]]:
        """批量LLM验证 - 使用优化的批量处理器"""
        from src.tools.batch_llm_processor import TableMatchingPromptBuilder
        
        # 准备批处理项目
        items = []
        for candidate_name, score in candidates:
            candidate_table = table_metadata.get(candidate_name)
            if candidate_table:
                item = {
                    "query_table": {
                        "name": query_table.table_name,
                        "columns": [
                            {
                                "name": col.column_name,
                                "type": col.data_type or "unknown"
                            }
                            for col in query_table.columns[:10]  # 限制列数
                        ]
                    },
                    "candidate_table": {
                        "name": candidate_name,
                        "columns": [
                            {
                                "name": col.column_name,
                                "type": col.data_type or "unknown"
                            }
                            for col in candidate_table.columns[:10]
                        ]
                    },
                    "initial_score": score
                }
                items.append(item)
        
        # 使用批量处理器
        results = await self.batch_processor.batch_process(
            items=items,
            prompt_builder=TableMatchingPromptBuilder.build_batch_prompt,
            response_parser=TableMatchingPromptBuilder.parse_batch_response,
            use_cache=True
        )
        
        # 合并初始分数和LLM结果
        final_results = []
        for item, result in zip(items, results):
            if result.get("match", False):
                # 如果LLM认为匹配，提升分数
                final_score = max(item["initial_score"], result.get("score", 0.5))
            else:
                # 如果LLM认为不匹配，降低分数
                final_score = min(item["initial_score"] * 0.8, result.get("score", 0.5))
            
            final_results.append({
                "table": result.get("table", item["candidate_table"]["name"]),
                "score": final_score,
                "reason": result.get("reason", ""),
                "method": "llm_batch",
                "match_type": "verified" if result.get("match") else "rejected"
            })
        
        return final_results
    
    def _build_batch_prompt(
        self,
        query_table: TableInfo,
        candidates: List[Tuple[str, float]],
        table_metadata: Dict[str, TableInfo]
    ) -> Tuple[str, List[Dict[str, Any]]]:
        """构建批量验证提示"""
        # 准备查询表信息（限制大小）
        query_info = {
            "name": query_table.table_name,
            "columns": [
                {
                    "name": col.column_name,
                    "type": col.data_type or "unknown",
                    "samples": col.sample_values[:3] if col.sample_values else []
                }
                for col in query_table.columns[:15]  # 限制列数
            ]
        }
        
        # 准备候选表信息
        candidates_info = []
        for candidate_name, score in candidates:
            candidate_table = table_metadata.get(candidate_name)
            if candidate_table:
                candidate_info = {
                    "name": candidate_name,
                    "score": score,
                    "columns": [
                        {
                            "name": col.column_name,
                            "type": col.data_type or "unknown"
                        }
                        for col in candidate_table.columns[:15]
                    ]
                }
                candidates_info.append(candidate_info)
        
        # 构建提示
        prompt = f"""Analyze whether these candidate tables can be joined or unioned with the query table.

Query Table:
{json.dumps(query_info, indent=2)}

Candidate Tables:
{json.dumps(candidates_info, indent=2)}

For each candidate, provide:
1. match_type: "join" (similar columns for joining), "union" (similar data for merging), or "none"
2. confidence: 0.0 to 1.0
3. reason: Brief explanation

Return JSON array with format:
[{{"table": "name", "match_type": "join|union|none", "confidence": 0.8, "reason": "..."}}]
"""
        
        return prompt, candidates_info
    
    async def _call_llm_with_retry(self, prompt: str, max_retries: int = 2) -> str:
        """调用LLM with重试"""
        for attempt in range(max_retries):
            try:
                response = await self.llm_client.generate(
                    prompt
                )
                return response
            except Exception as e:
                if attempt == max_retries - 1:
                    raise
                logger.warning(f"LLM调用失败，重试 {attempt + 1}: {e}")
                await asyncio.sleep(1)
    
    def _parse_batch_response(
        self,
        response: str,
        batch_info: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """解析批量响应"""
        try:
            # 提取JSON部分
            json_start = response.find('[')
            json_end = response.rfind(']') + 1
            if json_start >= 0 and json_end > json_start:
                json_str = response[json_start:json_end]
                results = json.loads(json_str)
                
                # 增强结果
                enhanced_results = []
                for result in results:
                    if result.get("match_type") != "none" and result.get("confidence", 0) > 0.6:
                        enhanced_results.append({
                            "table": result["table"],
                            "score": result["confidence"],
                            "reason": result["reason"],
                            "method": "llm",
                            "match_type": result["match_type"]
                        })
                
                return enhanced_results
            
        except Exception as e:
            logger.error(f"解析LLM响应失败: {e}")
        
        return []
    
    def _merge_results(
        self,
        rule_matches: List[Dict[str, Any]],
        llm_matches: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """合并规则和LLM结果"""
        # 使用表名作为键去重
        all_matches = {}
        
        # 先添加规则匹配（优先级高）
        for match in rule_matches:
            all_matches[match["table"]] = match
        
        # 添加LLM匹配
        for match in llm_matches:
            if match["table"] not in all_matches:
                all_matches[match["table"]] = match
        
        # 排序并返回
        results = list(all_matches.values())
        results.sort(key=lambda x: x["score"], reverse=True)
        
        return results[:10]  # 返回Top-10
    
    def _get_cache_key(
        self,
        query_table: TableInfo,
        candidates: List[Tuple[str, float]]
    ) -> str:
        """生成缓存键"""
        candidate_names = sorted([name for name, _ in candidates])
        return f"{query_table.table_name}:{':'.join(candidate_names)}"