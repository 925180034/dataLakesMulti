"""
LLMMatcherTool - Layer 3: LLM-based precise matching with parallel calls
"""
import asyncio
import time
import json
from typing import Dict, Any, List, Optional
import logging
from src.utils.llm_client_proxy import get_llm_client
from src.tools.value_similarity_tool import ValueSimilarityTool


class LLMMatcherTool:
    """
    LLM-based matcher for precise table matching
    Uses parallel LLM calls for efficiency
    """
    
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.llm_client = None
        self.value_similarity_tool = ValueSimilarityTool()
        self._initialize_llm()
        
    def _initialize_llm(self):
        """Initialize LLM client"""
        try:
            self.llm_client = get_llm_client()
            self.logger.info("LLM client initialized successfully")
        except Exception as e:
            self.logger.error(f"Failed to initialize LLM client: {e}")
            
    async def verify_match(self, query_table: Dict[str, Any], 
                          candidate_table: Dict[str, Any],
                          task_type: str = 'join',
                          existing_score: float = 0.5) -> Dict[str, Any]:
        """
        Verify if two tables match using LLM
        
        Args:
            query_table: Query table dictionary
            candidate_table: Candidate table dictionary  
            task_type: 'join' or 'union'
            existing_score: Score from previous layers
            
        Returns:
            Dictionary with match result and confidence
        """
        start_time = time.time()
        
        # Build prompt based on task type
        if task_type == 'join':
            prompt = self._build_join_prompt(query_table, candidate_table)
        else:
            prompt = self._build_union_prompt(query_table, candidate_table)
        
        try:
            # Call LLM
            response = await self.llm_client.generate(prompt)
            
            # Parse response
            result = self._parse_llm_response(response, existing_score)
            
            # Log performance
            elapsed = time.time() - start_time
            self.logger.debug(f"LLM verification took {elapsed:.2f}s")
            
            return result
            
        except Exception as e:
            self.logger.error(f"LLM verification failed: {e}")
            # Return conservative result on error
            return {
                'is_match': False,
                'confidence': existing_score * 0.5,
                'reason': f'LLM verification failed: {str(e)}'
            }
    
    def _build_join_prompt(self, query_table: Dict, candidate_table: Dict) -> str:
        """Build prompt for JOIN task verification"""
        # 提取更多细节信息
        query_cols = query_table.get('columns', [])[:15]
        candidate_cols = candidate_table.get('columns', [])[:15]
        
        # 获取列名列表用于精确比较
        query_col_names = [c.get('column_name', c.get('name', '')) for c in query_cols]
        candidate_col_names = [c.get('column_name', c.get('name', '')) for c in candidate_cols]
        
        # 计算列名重叠
        common_cols = set(query_col_names) & set(candidate_col_names)
        overlap_ratio = len(common_cols) / max(len(query_col_names), 1)
        
        # 计算值相似性
        value_similarity = self.value_similarity_tool.calculate_value_similarity(
            query_table, candidate_table, 'join'
        )
        
        # 找出最佳JOIN列
        best_join_cols = self.value_similarity_tool.find_best_join_columns(
            query_table, candidate_table
        )
        
        prompt = f"""判断这两个表是否可以进行JOIN操作。

查询表: {query_table.get('table_name')}
列数: {len(query_cols)}
列详情: {self._format_columns(query_cols)}

候选表: {candidate_table.get('table_name')}
列数: {len(candidate_cols)}
列详情: {self._format_columns(candidate_cols)}

共同列名: {list(common_cols) if common_cols else '无'}
列名重叠率: {overlap_ratio:.1%}
数据值相似度: {value_similarity:.1%}
潜在JOIN列: {[f"{q}={c} (置信度:{conf:.1%})" for q, c, conf in best_join_cols] if best_join_cols else '未发现'}

判断标准（必须满足至少一个）:
1. 存在相同的列名且数据类型兼容（最重要）
2. 数据样本值有重叠（value_similarity > 0.3）
3. 有明显的主键-外键关系（如id和xxx_id）
4. 存在语义相似的列（如user_id和userid）

重要：如果数据值相似度>0.5，很可能可以JOIN！

返回JSON格式：
{{
  "is_match": true/false,
  "confidence": 0.0-1.0,
  "join_keys": ["可用于JOIN的列对"],
  "reason": "判断理由（中文）"
}}"""
        return prompt
    
    def _build_union_prompt(self, query_table: Dict, candidate_table: Dict) -> str:
        """Build prompt for UNION task verification"""
        # 提取更多细节信息
        query_cols = query_table.get('columns', [])[:15]
        candidate_cols = candidate_table.get('columns', [])[:15]
        
        # 获取列名列表用于精确比较
        query_col_names = [c.get('column_name', c.get('name', '')) for c in query_cols]
        candidate_col_names = [c.get('column_name', c.get('name', '')) for c in candidate_cols]
        
        # 计算列名重叠
        common_cols = set(query_col_names) & set(candidate_col_names)
        overlap_ratio = len(common_cols) / max(len(set(query_col_names) | set(candidate_col_names)), 1)
        
        # 计算值分布相似性
        value_similarity = self.value_similarity_tool.calculate_value_similarity(
            query_table, candidate_table, 'union'
        )
        
        prompt = f"""判断这两个表是否可以进行UNION操作（是否有相似的模式）。

查询表: {query_table.get('table_name')}
列数: {len(query_cols)}
列详情: {self._format_columns(query_cols)}

候选表: {candidate_table.get('table_name')}
列数: {len(candidate_cols)}
列详情: {self._format_columns(candidate_cols)}

共同列名: {list(common_cols) if common_cols else '无'}
列名重叠率: {overlap_ratio:.1%}
数据分布相似度: {value_similarity:.1%}

判断标准（需要同时满足）:
1. 列数量相近（差异不超过30%）
2. 至少50%的列名相同或语义相似
3. 相同列的数据类型必须兼容
4. 数据分布相似（value_similarity > 0.5）

重要：如果数据分布相似度>0.6，很可能是相同类型的表！

返回JSON格式：
{{
  "is_match": true/false,
  "confidence": 0.0-1.0,
  "column_overlap": {overlap_ratio:.2f},
  "reason": "判断理由（中文）"
}}"""
        return prompt
    
    def _format_columns(self, columns: List[Dict]) -> str:
        """Format columns for prompt with more details"""
        formatted = []
        for col in columns:
            name = col.get('column_name', col.get('name', ''))
            dtype = col.get('data_type', col.get('type', 'unknown'))
            samples = col.get('sample_values', [])
            
            # 包含更多样本值以便更好判断
            col_str = f"{name} ({dtype})"
            if samples:
                # 显示最多5个样本值
                sample_strs = [str(s) for s in samples[:5] if s is not None]
                if sample_strs:
                    col_str += f" [{', '.join(sample_strs)}]"
            formatted.append(col_str)
            
        return '\n  '.join(formatted)  # 换行显示更清晰
    
    def _parse_llm_response(self, response: str, existing_score: float) -> Dict[str, Any]:
        """Parse LLM response"""
        try:
            # Try to parse as JSON
            if isinstance(response, str):
                # Find JSON in response
                start = response.find('{')
                end = response.rfind('}') + 1
                if start >= 0 and end > start:
                    json_str = response[start:end]
                    result = json.loads(json_str)
                else:
                    # Fallback to simple parsing
                    result = {
                        'is_match': 'true' in response.lower() or 'yes' in response.lower(),
                        'confidence': existing_score,
                        'reason': response[:200]
                    }
            else:
                result = response
                
            # Ensure required fields
            if 'is_match' not in result:
                result['is_match'] = result.get('confidence', 0) > 0.5
            if 'confidence' not in result:
                result['confidence'] = existing_score
            if 'reason' not in result:
                result['reason'] = 'LLM verification completed'
                
            return result
            
        except Exception as e:
            self.logger.error(f"Failed to parse LLM response: {e}")
            return {
                'is_match': False,
                'confidence': existing_score * 0.5,
                'reason': 'Failed to parse LLM response'
            }
    
    async def batch_verify(self, query_table: Dict[str, Any],
                          candidate_tables: List[Dict[str, Any]], 
                          task_type: str = 'join',
                          max_concurrent: int = 10,
                          existing_scores: Optional[List[float]] = None) -> List[Dict[str, Any]]:
        """
        Verify multiple candidates in parallel
        
        Args:
            query_table: Query table
            candidate_tables: List of candidate tables
            task_type: 'join' or 'union'
            max_concurrent: Maximum concurrent LLM calls
            existing_scores: Optional list of scores from previous layers
            
        Returns:
            List of verification results
        """
        # Create verification tasks
        tasks = []
        for i, candidate in enumerate(candidate_tables):
            # Get existing score if provided
            existing_score = existing_scores[i] if existing_scores and i < len(existing_scores) else 0.5
            
            task = self.verify_match(
                query_table,
                candidate,
                task_type,
                existing_score
            )
            tasks.append(task)
        
        # Execute in batches
        results = []
        for i in range(0, len(tasks), max_concurrent):
            batch = tasks[i:i + max_concurrent]
            batch_results = await asyncio.gather(*batch, return_exceptions=True)
            
            # Process results
            for j, result in enumerate(batch_results):
                if isinstance(result, Exception):
                    self.logger.error(f"Batch verification failed: {result}")
                    results.append({
                        'is_match': False,
                        'confidence': 0.0,
                        'reason': f'Verification failed: {str(result)}'
                    })
                else:
                    results.append(result)
        
        return results