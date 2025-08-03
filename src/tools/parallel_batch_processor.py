"""
并行批量LLM处理器 - 真正的并发API调用实现
实现独立并发的API调用，大幅提升性能
"""

import asyncio
import logging
import time
from typing import List, Dict, Any, Optional, Callable, Tuple
import hashlib
import json
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class QueryTask:
    """查询任务"""
    query_table: Any
    candidate_table: Any
    task_id: int
    cache_key: str = None


class ParallelBatchProcessor:
    """并行批量处理器
    
    核心优化：
    1. 真正的并发API调用（每个查询独立）
    2. 智能负载均衡
    3. 自适应超时控制
    4. 快速失败机制
    """
    
    def __init__(
        self,
        llm_client,
        max_concurrent: int = 20,  # 提高并发数
        timeout_per_call: float = 5.0,  # 单次调用超时
        enable_fast_fail: bool = True
    ):
        self.llm_client = llm_client
        self.max_concurrent = max_concurrent
        self.timeout_per_call = timeout_per_call
        self.enable_fast_fail = enable_fast_fail
        
        # 缓存
        self.cache = {}
        self.cache_hits = 0
        self.cache_misses = 0
        
        # 性能统计
        self.call_times = []
        self.success_count = 0
        self.failure_count = 0
        
    async def process_parallel(
        self,
        tasks: List[QueryTask],
        prompt_builder: Callable,
        response_parser: Callable,
        use_cache: bool = True
    ) -> List[Dict[str, Any]]:
        """并行处理所有任务
        
        Args:
            tasks: 查询任务列表
            prompt_builder: 构建单个prompt的函数
            response_parser: 解析响应的函数
            use_cache: 是否使用缓存
            
        Returns:
            处理结果列表
        """
        start_time = time.time()
        
        # 分离缓存命中和需要处理的任务
        cached_results = {}
        pending_tasks = []
        
        for task in tasks:
            if use_cache and task.cache_key and task.cache_key in self.cache:
                cached_results[task.task_id] = self.cache[task.cache_key]
                self.cache_hits += 1
            else:
                pending_tasks.append(task)
                self.cache_misses += 1
        
        # 并发处理所有待处理任务
        if pending_tasks:
            # 创建任务协程
            coroutines = []
            for task in pending_tasks:
                coro = self._process_single_task(
                    task,
                    prompt_builder,
                    response_parser
                )
                coroutines.append(coro)
            
            # 使用信号量控制并发
            semaphore = asyncio.Semaphore(self.max_concurrent)
            
            async def limited_task(coro):
                async with semaphore:
                    return await coro
            
            # 并发执行所有任务
            limited_coroutines = [limited_task(coro) for coro in coroutines]
            results = await asyncio.gather(*limited_coroutines, return_exceptions=True)
            
            # 处理结果并更新缓存
            for task, result in zip(pending_tasks, results):
                if isinstance(result, Exception):
                    logger.warning(f"任务{task.task_id}失败: {result}")
                    self.failure_count += 1
                    # 使用默认结果
                    result = self._get_default_result(task)
                else:
                    self.success_count += 1
                    # 更新缓存
                    if use_cache and task.cache_key:
                        self.cache[task.cache_key] = result
                
                cached_results[task.task_id] = result
        
        # 按原始顺序返回结果
        final_results = []
        for task in tasks:
            final_results.append(cached_results.get(
                task.task_id,
                self._get_default_result(task)
            ))
        
        elapsed = time.time() - start_time
        logger.info(
            f"并行处理完成: {len(tasks)}个任务, "
            f"缓存命中率: {self.cache_hits/(self.cache_hits + self.cache_misses):.1%}, "
            f"成功率: {self.success_count/(self.success_count + self.failure_count):.1%}, "
            f"耗时: {elapsed:.2f}秒"
        )
        
        return final_results
    
    async def _process_single_task(
        self,
        task: QueryTask,
        prompt_builder: Callable,
        response_parser: Callable,
        max_retries: int = 1  # 减少重试次数
    ) -> Dict[str, Any]:
        """处理单个任务（带超时控制）"""
        
        for attempt in range(max_retries + 1):
            try:
                # 构建prompt
                prompt = prompt_builder(task.query_table, task.candidate_table)
                
                # 带超时的API调用
                start = time.time()
                response = await asyncio.wait_for(
                    self.llm_client.generate(prompt),
                    timeout=self.timeout_per_call
                )
                
                # 记录调用时间
                call_time = time.time() - start
                self.call_times.append(call_time)
                
                # 解析响应
                result = response_parser(response, task.candidate_table)
                
                # 快速失败检查
                if self.enable_fast_fail and call_time > self.timeout_per_call * 0.8:
                    logger.warning(f"任务{task.task_id}接近超时({call_time:.2f}s)，考虑优化")
                
                return result
                
            except asyncio.TimeoutError:
                logger.warning(f"任务{task.task_id}超时(尝试{attempt + 1})")
                if self.enable_fast_fail:
                    break  # 快速失败，不重试
                    
            except Exception as e:
                logger.error(f"任务{task.task_id}失败(尝试{attempt + 1}): {e}")
                if attempt < max_retries:
                    await asyncio.sleep(0.1 * (attempt + 1))  # 指数退避
        
        # 返回默认结果
        return self._get_default_result(task)
    
    def _get_default_result(self, task: QueryTask) -> Dict[str, Any]:
        """获取默认结果"""
        return {
            "table": task.candidate_table.table_name if hasattr(task.candidate_table, 'table_name') else "unknown",
            "score": 0.0,
            "match": False,
            "confidence": 0.0,
            "reason": "Processing failed or timeout",
            "method": "default"
        }
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """获取性能统计"""
        if not self.call_times:
            return {}
        
        return {
            "total_calls": len(self.call_times),
            "avg_time": np.mean(self.call_times),
            "median_time": np.median(self.call_times),
            "p95_time": np.percentile(self.call_times, 95),
            "p99_time": np.percentile(self.call_times, 99),
            "success_rate": self.success_count / (self.success_count + self.failure_count) if (self.success_count + self.failure_count) > 0 else 0,
            "cache_hit_rate": self.cache_hits / (self.cache_hits + self.cache_misses) if (self.cache_hits + self.cache_misses) > 0 else 0
        }


class OptimizedTableMatcher:
    """优化的表匹配器"""
    
    def __init__(self, llm_client):
        self.processor = ParallelBatchProcessor(
            llm_client=llm_client,
            max_concurrent=20,  # 高并发
            timeout_per_call=5.0,  # 5秒超时
            enable_fast_fail=True
        )
    
    async def match_tables_batch(
        self,
        query_tables: List[Any],
        candidate_tables_list: List[List[Any]]
    ) -> List[List[Dict[str, Any]]]:
        """批量匹配多个查询表
        
        Args:
            query_tables: 查询表列表
            candidate_tables_list: 每个查询表对应的候选表列表
            
        Returns:
            每个查询表的匹配结果列表
        """
        # 创建所有任务
        all_tasks = []
        task_mapping = {}  # task_id -> (query_idx, candidate_idx)
        task_id = 0
        
        for query_idx, (query_table, candidates) in enumerate(zip(query_tables, candidate_tables_list)):
            task_mapping[query_idx] = []
            for candidate_idx, candidate in enumerate(candidates[:5]):  # 限制每个查询最多5个候选
                task = QueryTask(
                    query_table=query_table,
                    candidate_table=candidate,
                    task_id=task_id,
                    cache_key=self._get_cache_key(query_table, candidate)
                )
                all_tasks.append(task)
                task_mapping[query_idx].append(task_id)
                task_id += 1
        
        # 并发处理所有任务
        all_results = await self.processor.process_parallel(
            tasks=all_tasks,
            prompt_builder=self._build_single_prompt,
            response_parser=self._parse_single_response,
            use_cache=True
        )
        
        # 重组结果
        final_results = []
        for query_idx in range(len(query_tables)):
            query_results = []
            for task_id in task_mapping.get(query_idx, []):
                if task_id < len(all_results):
                    query_results.append(all_results[task_id])
            final_results.append(query_results)
        
        return final_results
    
    def _get_cache_key(self, query_table, candidate_table) -> str:
        """生成缓存键"""
        key_str = f"{query_table.table_name}:{candidate_table.table_name}"
        return hashlib.md5(key_str.encode()).hexdigest()
    
    def _build_single_prompt(self, query_table, candidate_table) -> str:
        """构建单个匹配prompt"""
        prompt = f"""判断以下两个表是否可以连接或相似：

查询表: {query_table.table_name}
列: {', '.join([col.column_name for col in query_table.columns[:10]])}

候选表: {candidate_table.table_name}
列: {', '.join([col.column_name for col in candidate_table.columns[:10]])}

请返回JSON格式（不要包含markdown标记）：
{{"match": true/false, "confidence": 0-1, "reason": "简短原因"}}
"""
        return prompt
    
    def _parse_single_response(self, response: str, candidate_table) -> Dict[str, Any]:
        """解析单个响应"""
        try:
            # 清理响应
            cleaned = response.strip()
            if cleaned.startswith('```'):
                cleaned = cleaned.split('```')[1]
                if cleaned.startswith('json'):
                    cleaned = cleaned[4:]
            
            result = json.loads(cleaned)
            
            return {
                "table": candidate_table.table_name,
                "score": result.get("confidence", 0.5),
                "match": result.get("match", False),
                "confidence": result.get("confidence", 0.5),
                "reason": result.get("reason", ""),
                "method": "llm"
            }
        except Exception as e:
            logger.warning(f"解析响应失败: {e}")
            return {
                "table": candidate_table.table_name,
                "score": 0.0,
                "match": False,
                "confidence": 0.0,
                "reason": "Parse error",
                "method": "error"
            }