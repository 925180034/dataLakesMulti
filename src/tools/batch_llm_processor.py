"""
批量LLM处理器 - 优化的批量API调用实现
通过批量处理和并发调用显著提升性能
"""

import asyncio
import logging
import time
from typing import List, Dict, Any, Optional, Callable
from collections import defaultdict
import hashlib
import json

logger = logging.getLogger(__name__)


class BatchLLMProcessor:
    """批量LLM处理器
    
    核心优化：
    1. 智能批量合并
    2. 并发调用管理
    3. 自动重试机制
    4. 结果缓存
    """
    
    def __init__(self, llm_client, max_batch_size: int = 10, max_concurrent: int = 5):
        self.llm_client = llm_client
        self.max_batch_size = max_batch_size
        self.max_concurrent = max_concurrent
        
        # 缓存系统
        self.cache = {}
        self.cache_hits = 0
        self.cache_misses = 0
        
        # 性能统计
        self.total_calls = 0
        self.total_time = 0
        self.batch_count = 0
        
    async def batch_process(
        self,
        items: List[Dict[str, Any]],
        prompt_builder: Callable,
        response_parser: Callable,
        use_cache: bool = True
    ) -> List[Dict[str, Any]]:
        """批量处理多个项目
        
        Args:
            items: 待处理的项目列表
            prompt_builder: 构建提示的函数
            response_parser: 解析响应的函数
            use_cache: 是否使用缓存
            
        Returns:
            处理结果列表
        """
        start_time = time.time()
        
        # 分组处理
        results = []
        pending_items = []
        
        # 检查缓存
        for item in items:
            if use_cache:
                cache_key = self._get_cache_key(item)
                if cache_key in self.cache:
                    self.cache_hits += 1
                else:
                    pending_items.append(item)
                    self.cache_misses += 1
            else:
                pending_items.append(item)
        
        # 批量处理未缓存的项目
        if pending_items:
            # 创建批次
            batches = self._create_batches(pending_items)
            
            # 并发处理批次
            batch_tasks = []
            semaphore = asyncio.Semaphore(self.max_concurrent)
            
            for batch in batches:
                task = self._process_batch_with_semaphore(
                    batch, prompt_builder, response_parser, semaphore
                )
                batch_tasks.append(task)
            
            # 等待所有批次完成
            batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
            
            # 合并结果
            for batch_result in batch_results:
                if isinstance(batch_result, Exception):
                    logger.error(f"批次处理失败: {batch_result}")
                    # 为失败的批次返回空结果
                    results.extend([{} for _ in range(self.max_batch_size)])
                else:
                    results.extend(batch_result)
                    
                    # 缓存成功的结果
                    if use_cache:
                        for item, result in zip(batch, batch_result):
                            cache_key = self._get_cache_key(item)
                            self.cache[cache_key] = result
        
        # 合并缓存和新结果
        final_results = []
        result_idx = 0
        
        for item in items:
            cache_key = self._get_cache_key(item) if use_cache else None
            if use_cache and cache_key in self.cache:
                # 直接从缓存获取，而不是从cached_results数组
                final_results.append(self.cache[cache_key])
            else:
                if result_idx < len(results):
                    final_results.append(results[result_idx])
                    result_idx += 1
                else:
                    final_results.append({
                        "table": "unknown",
                        "score": 0.5,
                        "match": False,
                        "reason": "No result",
                        "method": "default"
                    })
        
        # 更新统计
        self.total_time += time.time() - start_time
        self.total_calls += len(items)
        
        logger.info(f"批量处理完成: {len(items)}个项目, "
                   f"缓存命中: {self.cache_hits}/{self.cache_hits + self.cache_misses}, "
                   f"耗时: {time.time() - start_time:.2f}秒")
        
        return final_results
    
    async def _process_batch_with_semaphore(
        self,
        batch: List[Dict[str, Any]],
        prompt_builder: Callable,
        response_parser: Callable,
        semaphore: asyncio.Semaphore
    ) -> List[Dict[str, Any]]:
        """使用信号量控制并发处理批次"""
        async with semaphore:
            return await self._process_single_batch(batch, prompt_builder, response_parser)
    
    async def _process_single_batch(
        self,
        batch: List[Dict[str, Any]],
        prompt_builder: Callable,
        response_parser: Callable,
        max_retries: int = 2
    ) -> List[Dict[str, Any]]:
        """处理单个批次"""
        self.batch_count += 1
        
        # 构建批量提示
        prompt = prompt_builder(batch)
        
        # 重试机制
        for attempt in range(max_retries):
            try:
                # 调用LLM的generate_json方法以获得结构化输出
                if hasattr(self.llm_client, 'generate_json'):
                    # 优先使用generate_json方法
                    try:
                        response = await self.llm_client.generate_json(prompt)
                        # 如果返回的是字典，转换为JSON字符串供解析器处理
                        if isinstance(response, dict):
                            response = json.dumps(response)
                        elif isinstance(response, list):
                            response = json.dumps(response)
                    except Exception as e:
                        logger.warning(f"generate_json失败，回退到generate: {e}")
                        response = await self.llm_client.generate(prompt)
                else:
                    # 使用普通generate方法
                    response = await self.llm_client.generate(prompt)
                
                # 解析响应
                results = response_parser(response, batch)
                
                # 确保返回正确数量的结果
                if len(results) != len(batch):
                    logger.warning(f"结果数量不匹配: 期望{len(batch)}, 实际{len(results)}")
                    # 填充缺失的结果
                    while len(results) < len(batch):
                        results.append({
                            "table": "unknown",
                            "score": 0.5,
                            "match": False,
                            "reason": "No response",
                            "method": "default"
                        })
                
                return results
                
            except Exception as e:
                logger.error(f"批次处理失败 (尝试 {attempt + 1}/{max_retries}): {e}")
                if attempt == max_retries - 1:
                    # 返回默认结果而非空字典
                    return [{
                        "table": batch[i].get("candidate_table", {}).get("name", "unknown") if i < len(batch) else "unknown",
                        "score": 0.5,
                        "match": False,
                        "reason": f"Processing error: {str(e)[:50]}",
                        "method": "error"
                    } for i in range(len(batch))]
                    
                # 等待后重试
                await asyncio.sleep(2 ** attempt)
        
        return [{
            "table": batch[i].get("candidate_table", {}).get("name", "unknown") if i < len(batch) else "unknown",
            "score": 0.5,
            "match": False,
            "reason": "Max retries exceeded",
            "method": "error"
        } for i in range(len(batch))]
    
    def _create_batches(self, items: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
        """创建批次"""
        batches = []
        for i in range(0, len(items), self.max_batch_size):
            batch = items[i:i + self.max_batch_size]
            batches.append(batch)
        return batches
    
    def _get_cache_key(self, item: Dict[str, Any]) -> str:
        """生成缓存键"""
        # 使用item的JSON表示生成hash
        item_str = json.dumps(item, sort_keys=True)
        return hashlib.md5(item_str.encode()).hexdigest()
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取性能统计"""
        return {
            "total_calls": self.total_calls,
            "total_time": self.total_time,
            "avg_time_per_call": self.total_time / self.total_calls if self.total_calls > 0 else 0,
            "batch_count": self.batch_count,
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "cache_hit_rate": self.cache_hits / (self.cache_hits + self.cache_misses) 
                            if (self.cache_hits + self.cache_misses) > 0 else 0
        }
    
    def clear_cache(self):
        """清空缓存"""
        self.cache.clear()
        logger.info("缓存已清空")


class TableMatchingPromptBuilder:
    """表匹配提示构建器"""
    
    @staticmethod
    def build_batch_prompt(items: List[Dict[str, Any]]) -> str:
        """构建批量表匹配提示"""
        prompt = """You are a data expert. Analyze if the following table pairs are joinable.

For each candidate table, determine if it can be joined with the query table based on:
1. Common columns that could serve as join keys
2. Compatible data types
3. Semantic relationships

Query Table:
{query_info}

Candidate Tables to evaluate:
{candidates_info}

IMPORTANT: Return ONLY a valid JSON array with exactly {num_items} objects (one per candidate).
Each object must have these exact fields:
- "match": true or false (boolean)
- "confidence": 0.0 to 1.0 (number)
- "reason": brief explanation (string)

Example response format:
[
  {{"match": true, "confidence": 0.9, "reason": "Common ID columns"}},
  {{"match": false, "confidence": 0.2, "reason": "No common columns"}}
]

Response (JSON array only):
"""
        
        # 提取查询表信息（假设所有items使用相同的查询表）
        query_info = items[0].get("query_table", {}) if items else {}
        
        # 提取候选表信息
        candidates_info = []
        for i, item in enumerate(items):
            candidate = item.get("candidate_table", {})
            candidates_info.append(f"{i+1}. {candidate.get('name', 'Unknown')}: "
                                  f"{', '.join(c['name'] for c in candidate.get('columns', [])[:5])}")
        
        return prompt.format(
            query_info=json.dumps(query_info, indent=2),
            candidates_info='\n'.join(candidates_info),
            num_items=len(items)
        )
    
    @staticmethod
    def parse_batch_response(response: str, items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """解析批量响应"""
        try:
            # 如果response已经是字符串形式的JSON，解析它
            if isinstance(response, str):
                # 尝试提取JSON部分（处理可能的额外文本）
                json_start = response.find('[')
                json_end = response.rfind(']') + 1
                if json_start >= 0 and json_end > json_start:
                    json_str = response[json_start:json_end]
                    results = json.loads(json_str)
                else:
                    # 尝试整个字符串
                    results = json.loads(response)
            else:
                # 如果已经是dict或list，直接使用
                results = response
            
            if not isinstance(results, list):
                results = [results]
            
            # 确保每个item都有结果
            parsed_results = []
            for i, item in enumerate(items):
                if i < len(results):
                    result = results[i]
                    # 兼容不同的响应格式
                    if isinstance(result, dict):
                        # 尝试从多种字段名获取匹配状态
                        match_status = result.get("match", 
                                                result.get("is_match", 
                                                result.get("joinable", False)))
                        
                        # 尝试从多种字段名获取置信度
                        confidence = result.get("confidence", 
                                              result.get("score", 
                                              result.get("similarity", 0.5)))
                        
                        parsed_results.append({
                            "table": item.get("candidate_table", {}).get("name", "unknown"),
                            "score": float(confidence) if confidence else 0.5,
                            "match": bool(match_status),
                            "reason": result.get("reason", result.get("explanation", "")),
                            "method": "llm_batch"
                        })
                    else:
                        # 如果结果不是字典，创建默认结果
                        parsed_results.append({
                            "table": item.get("candidate_table", {}).get("name", "unknown"),
                            "score": 0.5,
                            "match": False,
                            "reason": "Invalid response format",
                            "method": "default"
                        })
                else:
                    # 默认结果
                    parsed_results.append({
                        "table": item.get("candidate_table", {}).get("name", "unknown"),
                        "score": 0.5,
                        "match": False,
                        "reason": "No response",
                        "method": "default"
                    })
            
            return parsed_results
            
        except (json.JSONDecodeError, Exception) as e:
            logger.error(f"响应解析失败: {e}, 原始响应: {str(response)[:200]}")
            # 返回默认结果
            return [{
                "table": item.get("candidate_table", {}).get("name", "unknown"),
                "score": 0.5,
                "match": False,
                "reason": f"Parse error: {str(e)[:50]}",
                "method": "error"
            } for item in items]