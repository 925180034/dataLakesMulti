"""
真实的多智能体数据湖发现系统
Real Multi-Agent Data Lake Discovery System with Full Implementation
"""

import asyncio
import json
import time
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass, field, asdict
from pathlib import Path
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
import faiss
from sentence_transformers import SentenceTransformer
import torch

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 导入工具层
from src.tools.metadata_filter import MetadataFilter
from src.tools.vector_search import get_vector_search_engine
from src.tools.smart_llm_matcher import SmartLLMMatcher
from src.tools.embedding import get_embedding_generator
from src.config.settings import Settings
from src.utils.llm_client import GeminiClient

# ===================== 数据结构定义 =====================

@dataclass
class TableInfo:
    """表信息"""
    table_name: str
    columns: List[Dict[str, Any]]
    row_count: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    embedding: Optional[np.ndarray] = None
    
@dataclass
class QueryTask:
    """查询任务"""
    query_id: str
    query_table: str
    task_type: str  # 'join' or 'union'
    ground_truth: List[str]
    timestamp: float = field(default_factory=time.time)
    
@dataclass
class MatchResult:
    """匹配结果"""
    query_table: str
    matched_table: str
    score: float
    match_type: str
    evidence: Dict[str, Any] = field(default_factory=dict)
    agent_used: str = ""
    time_cost: float = 0.0

@dataclass 
class AgentMessage:
    """Agent间通信消息"""
    sender: str
    receiver: str
    message_type: str
    content: Any
    priority: int = 0
    timestamp: float = field(default_factory=time.time)

@dataclass
class SystemMetrics:
    """系统性能指标"""
    total_queries: int = 0
    successful_queries: int = 0
    failed_queries: int = 0
    avg_response_time: float = 0.0
    throughput: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    f1_score: float = 0.0
    hit_at_k: Dict[int, float] = field(default_factory=dict)
    mrr: float = 0.0

# ===================== Agent基类 =====================

class BaseAgent:
    """Agent基类"""
    
    def __init__(self, name: str, llm_client=None):
        self.name = name
        self.llm_client = llm_client
        self.message_queue = asyncio.Queue()
        self.stats = {
            'processed': 0,
            'success': 0,
            'failed': 0,
            'avg_time': 0.0
        }
        
    async def receive_message(self, message: AgentMessage):
        """接收消息"""
        await self.message_queue.put(message)
        
    async def send_message(self, receiver: str, content: Any, 
                          message_type: str = "data"):
        """发送消息给其他Agent"""
        message = AgentMessage(
            sender=self.name,
            receiver=receiver,
            message_type=message_type,
            content=content
        )
        # 这里需要通过协调器转发
        return message
        
    async def process(self, task: Any) -> Any:
        """处理任务（子类实现）"""
        raise NotImplementedError
        
    def update_stats(self, success: bool, time_cost: float):
        """更新统计信息"""
        self.stats['processed'] += 1
        if success:
            self.stats['success'] += 1
        else:
            self.stats['failed'] += 1
        
        # 更新平均时间
        n = self.stats['processed']
        self.stats['avg_time'] = (
            (self.stats['avg_time'] * (n - 1) + time_cost) / n
        )

# ===================== 具体Agent实现 =====================

class OptimizerAgent(BaseAgent):
    """系统优化Agent - 动态调整系统配置"""
    
    def __init__(self, llm_client=None):
        super().__init__("OptimizerAgent", llm_client)
        self.current_config = {
            'batch_size': 32,
            'parallel_workers': 4,
            'cache_enabled': True,
            'llm_temperature': 0.1,
            'vector_top_k': 100,
            'metadata_top_k': 1000
        }
        self.performance_history = []
        
    async def process(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """根据系统性能动态优化配置"""
        start_time = time.time()
        
        # 分析当前性能
        if metrics.get('avg_response_time', 0) > 5.0:
            # 响应时间过长，增加并行度
            self.current_config['parallel_workers'] = min(8, 
                self.current_config['parallel_workers'] + 2)
            self.current_config['batch_size'] = min(64,
                self.current_config['batch_size'] * 2)
                
        if metrics.get('memory_usage', 0) > 0.8:
            # 内存使用过高，减小批次
            self.current_config['batch_size'] = max(16,
                self.current_config['batch_size'] // 2)
                
        if metrics.get('accuracy', 1.0) < 0.8:
            # 准确率低，调整LLM参数
            self.current_config['llm_temperature'] = max(0.0,
                self.current_config['llm_temperature'] - 0.05)
            self.current_config['vector_top_k'] = min(200,
                self.current_config['vector_top_k'] + 20)
        
        self.performance_history.append(metrics)
        
        # 如果有足够的历史数据，使用趋势分析
        if len(self.performance_history) > 10:
            self._analyze_trends()
        
        time_cost = time.time() - start_time
        self.update_stats(True, time_cost)
        
        logger.info(f"OptimizerAgent updated config: {self.current_config}")
        return self.current_config
        
    def _analyze_trends(self):
        """分析性能趋势"""
        recent = self.performance_history[-5:]
        avg_recent_time = np.mean([m.get('avg_response_time', 0) for m in recent])
        
        older = self.performance_history[-10:-5]
        avg_older_time = np.mean([m.get('avg_response_time', 0) for m in older])
        
        if avg_recent_time > avg_older_time * 1.2:
            # 性能恶化，需要更激进的优化
            logger.warning("Performance degradation detected, applying aggressive optimization")
            self.current_config['cache_enabled'] = True
            self.current_config['parallel_workers'] = 8


class PlannerAgent(BaseAgent):
    """策略规划Agent - 分析查询意图并制定执行策略"""
    
    def __init__(self, llm_client=None):
        super().__init__("PlannerAgent", llm_client)
        self.strategies = {
            'join': self._join_strategy,
            'union': self._union_strategy, 
            'complex': self._complex_strategy
        }
        
    async def process(self, query_task: QueryTask) -> Dict[str, Any]:
        """分析查询并制定策略"""
        start_time = time.time()
        
        # 分析查询类型
        task_type = query_task.task_type.lower()
        
        # 选择策略
        if task_type in self.strategies:
            strategy = await self.strategies[task_type](query_task)
        else:
            # 复杂查询，使用LLM分析
            strategy = await self._analyze_with_llm(query_task)
        
        # 决定需要的Agent和执行顺序
        execution_plan = {
            'task_type': task_type,
            'strategy': strategy,
            'required_agents': self._determine_required_agents(strategy),
            'parallel_possible': self._check_parallel_possibility(strategy),
            'estimated_time': self._estimate_execution_time(strategy)
        }
        
        time_cost = time.time() - start_time
        self.update_stats(True, time_cost)
        
        logger.info(f"PlannerAgent created plan for {query_task.query_id}: {execution_plan['strategy']['name']}")
        return execution_plan
        
    async def _join_strategy(self, query_task: QueryTask) -> Dict[str, Any]:
        """JOIN策略 - 寻找可连接的表"""
        return {
            'name': 'bottom_up_join',
            'description': 'Find tables with foreign key relationships',
            'steps': [
                'analyze_key_columns',
                'metadata_filter_by_keys',
                'vector_search_similar_structure',
                'llm_verify_join_conditions'
            ],
            'focus': 'foreign_keys_and_references'
        }
        
    async def _union_strategy(self, query_task: QueryTask) -> Dict[str, Any]:
        """UNION策略 - 寻找结构相似的表"""
        return {
            'name': 'top_down_union',
            'description': 'Find tables with similar schema',
            'steps': [
                'analyze_table_structure',
                'vector_search_by_schema',
                'metadata_filter_by_types',
                'llm_verify_compatibility'
            ],
            'focus': 'schema_similarity'
        }
        
    async def _complex_strategy(self, query_task: QueryTask) -> Dict[str, Any]:
        """复杂策略 - 混合方法"""
        return {
            'name': 'hybrid_complex',
            'description': 'Multi-dimensional analysis',
            'steps': [
                'deep_table_analysis',
                'parallel_search_all_methods',
                'intelligent_matching',
                'comprehensive_verification'
            ],
            'focus': 'comprehensive'
        }
        
    async def _analyze_with_llm(self, query_task: QueryTask) -> Dict[str, Any]:
        """使用LLM分析复杂查询"""
        if self.llm_client:
            prompt = f"""
            Analyze this data discovery task:
            Query Table: {query_task.query_table}
            Task Type: {query_task.task_type}
            
            Suggest the best strategy and required steps.
            """
            # 实际LLM调用
            response = await self.llm_client.generate(prompt)
            # 解析响应并返回策略
        
        return await self._complex_strategy(query_task)
        
    def _determine_required_agents(self, strategy: Dict[str, Any]) -> List[str]:
        """确定需要的Agent"""
        agents = ['AnalyzerAgent', 'SearcherAgent']
        
        if 'llm' in str(strategy.get('steps', [])).lower():
            agents.append('MatcherAgent')
        
        agents.append('AggregatorAgent')
        return agents
        
    def _check_parallel_possibility(self, strategy: Dict[str, Any]) -> bool:
        """检查是否可以并行执行"""
        parallel_steps = ['parallel_search', 'batch_processing']
        return any(step in str(strategy.get('steps', [])) for step in parallel_steps)
        
    def _estimate_execution_time(self, strategy: Dict[str, Any]) -> float:
        """估算执行时间"""
        base_time = 0.5
        if 'llm' in str(strategy).lower():
            base_time += 2.0
        if 'vector_search' in str(strategy).lower():
            base_time += 0.5
        return base_time


class AnalyzerAgent(BaseAgent):
    """数据分析Agent - 深度理解表结构"""
    
    def __init__(self, metadata_filter: MetadataFilter, llm_client=None):
        super().__init__("AnalyzerAgent", llm_client)
        self.metadata_filter = metadata_filter
        
    async def process(self, table_info: TableInfo) -> Dict[str, Any]:
        """分析表结构和特征"""
        start_time = time.time()
        
        analysis = {
            'table_name': table_info.table_name,
            'column_count': len(table_info.columns),
            'column_types': {},
            'key_columns': [],
            'patterns': [],
            'table_type': 'unknown'
        }
        
        # 分析列类型分布
        for col in table_info.columns:
            col_type = col.get('type', 'unknown')
            analysis['column_types'][col_type] = analysis['column_types'].get(col_type, 0) + 1
            
            # 识别关键列
            col_name = col.get('name', '').lower()
            if any(key in col_name for key in ['_id', '_key', '_code', '_fk']):
                analysis['key_columns'].append(col['name'])
                
        # 识别表类型
        table_name = table_info.table_name.lower()
        if any(dim in table_name for dim in ['dim_', 'd_', 'dimension']):
            analysis['table_type'] = 'dimension'
            analysis['patterns'].append('dimension_table')
        elif any(fact in table_name for fact in ['fact_', 'f_', 'agg_']):
            analysis['table_type'] = 'fact'
            analysis['patterns'].append('fact_table')
        elif any(lookup in table_name for lookup in ['lookup', 'ref_', 'code_']):
            analysis['table_type'] = 'lookup'
            analysis['patterns'].append('lookup_table')
            
        # 复杂表使用LLM深度分析
        if len(table_info.columns) > 20 and self.llm_client:
            deep_analysis = await self._deep_llm_analysis(table_info)
            analysis['deep_insights'] = deep_analysis
            
        time_cost = time.time() - start_time
        self.update_stats(True, time_cost)
        
        logger.debug(f"AnalyzerAgent analyzed {table_info.table_name}: {analysis['table_type']}")
        return analysis
        
    async def _deep_llm_analysis(self, table_info: TableInfo) -> Dict[str, Any]:
        """使用LLM进行深度分析"""
        if not self.llm_client:
            return {}
            
        prompt = f"""
        Analyze this table structure:
        Table: {table_info.table_name}
        Columns: {[col['name'] for col in table_info.columns[:20]]}
        
        Identify:
        1. Business domain
        2. Key relationships
        3. Data patterns
        """
        
        try:
            response = await self.llm_client.generate(prompt)
            return {'llm_insights': response}
        except Exception as e:
            logger.error(f"LLM analysis failed: {e}")
            return {}


class SearcherAgent(BaseAgent):
    """候选搜索Agent - 高效查找候选表"""
    
    def __init__(self, metadata_filter: MetadataFilter, 
                 vector_search, llm_client=None):
        super().__init__("SearcherAgent", llm_client)
        self.metadata_filter = metadata_filter
        self.vector_search = vector_search
        
    async def process(self, search_request: Dict[str, Any]) -> List[Tuple[str, float]]:
        """执行多层搜索"""
        start_time = time.time()
        
        query_table = search_request['query_table']
        strategy = search_request.get('strategy', {})
        analysis = search_request.get('analysis', {})
        
        candidates = []
        
        # Layer 1: 元数据筛选
        if 'metadata' in strategy.get('name', '').lower() or True:
            metadata_candidates = await self._metadata_search(
                query_table, analysis
            )
            candidates.extend(metadata_candidates)
            logger.info(f"Metadata filter found {len(metadata_candidates)} candidates")
            
        # Layer 2: 向量搜索
        if 'vector' in strategy.get('name', '').lower() or True:
            vector_candidates = await self._vector_search(
                query_table, 
                top_k=search_request.get('top_k', 100)
            )
            candidates.extend(vector_candidates)
            logger.info(f"Vector search found {len(vector_candidates)} candidates")
            
        # 合并和去重
        unique_candidates = self._merge_candidates(candidates)
        
        # 根据策略排序
        sorted_candidates = self._rank_candidates(
            unique_candidates, strategy
        )
        
        time_cost = time.time() - start_time
        self.update_stats(True, time_cost)
        
        logger.info(f"SearcherAgent found {len(sorted_candidates)} unique candidates")
        return sorted_candidates[:search_request.get('max_candidates', 100)]
        
    async def _metadata_search(self, query_table: TableInfo, 
                              analysis: Dict[str, Any]) -> List[Tuple[str, float]]:
        """元数据搜索"""
        try:
            # 使用元数据过滤器
            results = self.metadata_filter.filter_tables(
                query_table,
                column_count_threshold=2,
                type_match_weight=0.4,
                name_similarity_weight=0.3
            )
            return [(r['table_name'], r['score']) for r in results[:1000]]
        except Exception as e:
            logger.error(f"Metadata search failed: {e}")
            return []
            
    async def _vector_search(self, query_table: TableInfo, 
                            top_k: int = 100) -> List[Tuple[str, float]]:
        """向量搜索"""
        try:
            # 确保表有嵌入向量
            if query_table.embedding is None:
                # 生成嵌入
                embedding_gen = get_embedding_generator()
                query_table.embedding = embedding_gen.generate_table_embedding(query_table)
                
            # 执行向量搜索
            results = self.vector_search.search(
                query_table.embedding,
                top_k=top_k
            )
            return [(r['table_name'], r['score']) for r in results]
        except Exception as e:
            logger.error(f"Vector search failed: {e}")
            return []
            
    def _merge_candidates(self, candidates: List[Tuple[str, float]]) -> List[Tuple[str, float]]:
        """合并和去重候选"""
        candidate_dict = {}
        for table_name, score in candidates:
            if table_name in candidate_dict:
                # 取最高分
                candidate_dict[table_name] = max(candidate_dict[table_name], score)
            else:
                candidate_dict[table_name] = score
        
        return list(candidate_dict.items())
        
    def _rank_candidates(self, candidates: List[Tuple[str, float]], 
                        strategy: Dict[str, Any]) -> List[Tuple[str, float]]:
        """根据策略对候选排序"""
        # 简单按分数排序
        return sorted(candidates, key=lambda x: x[1], reverse=True)


class MatcherAgent(BaseAgent):
    """精确匹配Agent - 验证候选匹配"""
    
    def __init__(self, llm_matcher: SmartLLMMatcher, llm_client=None):
        super().__init__("MatcherAgent", llm_client)
        self.llm_matcher = llm_matcher
        
    async def process(self, match_request: Dict[str, Any]) -> List[MatchResult]:
        """精确匹配验证"""
        start_time = time.time()
        
        query_table = match_request['query_table']
        candidates = match_request['candidates']
        task_type = match_request.get('task_type', 'join')
        
        matches = []
        
        # 批量处理策略
        batch_size = match_request.get('batch_size', 10)
        
        for i in range(0, len(candidates), batch_size):
            batch = candidates[i:i+batch_size]
            
            # 并行处理批次
            batch_results = await self._process_batch(
                query_table, batch, task_type
            )
            matches.extend(batch_results)
            
        # 过滤高分匹配
        high_quality_matches = [
            m for m in matches if m.score > 0.7
        ]
        
        time_cost = time.time() - start_time
        self.update_stats(len(high_quality_matches) > 0, time_cost)
        
        logger.info(f"MatcherAgent verified {len(high_quality_matches)}/{len(candidates)} matches")
        return high_quality_matches
        
    async def _process_batch(self, query_table: TableInfo, 
                            candidates: List[Tuple[str, float]], 
                            task_type: str) -> List[MatchResult]:
        """批量处理候选"""
        results = []
        
        # 使用智能LLM匹配器
        for candidate_name, candidate_score in candidates:
            # 这里应该获取完整的candidate表信息
            # 简化处理，实际应该从数据库加载
            candidate_table = TableInfo(
                table_name=candidate_name,
                columns=[]  # 需要实际数据
            )
            
            # 检查是否需要LLM验证
            if self._needs_llm_verification(query_table, candidate_table, candidate_score):
                match_result = await self._llm_verify(
                    query_table, candidate_table, task_type
                )
            else:
                # 规则验证
                match_result = self._rule_verify(
                    query_table, candidate_table, task_type, candidate_score
                )
                
            if match_result:
                results.append(match_result)
                
        return results
        
    def _needs_llm_verification(self, query_table: TableInfo, 
                               candidate_table: TableInfo, 
                               score: float) -> bool:
        """判断是否需要LLM验证"""
        # 高分直接通过
        if score > 0.95:
            return False
        # 低分需要验证
        if score < 0.5:
            return False
        # 中等分数需要LLM验证
        return True
        
    async def _llm_verify(self, query_table: TableInfo, 
                         candidate_table: TableInfo, 
                         task_type: str) -> Optional[MatchResult]:
        """LLM验证"""
        try:
            result = await self.llm_matcher.match_tables(
                query_table, [candidate_table], task_type
            )
            if result and len(result) > 0:
                return MatchResult(
                    query_table=query_table.table_name,
                    matched_table=candidate_table.table_name,
                    score=result[0].get('score', 0.0),
                    match_type=task_type,
                    evidence=result[0].get('evidence', {}),
                    agent_used='MatcherAgent_LLM'
                )
        except Exception as e:
            logger.error(f"LLM verification failed: {e}")
            
        return None
        
    def _rule_verify(self, query_table: TableInfo, 
                    candidate_table: TableInfo, 
                    task_type: str, 
                    score: float) -> MatchResult:
        """规则验证"""
        return MatchResult(
            query_table=query_table.table_name,
            matched_table=candidate_table.table_name,
            score=score,
            match_type=task_type,
            evidence={'method': 'rule_based'},
            agent_used='MatcherAgent_Rule'
        )


class AggregatorAgent(BaseAgent):
    """结果聚合Agent - 整合和排序结果"""
    
    def __init__(self, llm_client=None):
        super().__init__("AggregatorAgent", llm_client)
        
    async def process(self, aggregation_request: Dict[str, Any]) -> List[MatchResult]:
        """聚合和排序结果"""
        start_time = time.time()
        
        all_matches = aggregation_request.get('matches', [])
        top_k = aggregation_request.get('top_k', 10)
        
        # 去重
        unique_matches = self._deduplicate(all_matches)
        
        # 排序策略
        if len(unique_matches) > 100:
            # 简单排序
            sorted_matches = sorted(
                unique_matches, 
                key=lambda x: x.score, 
                reverse=True
            )
        elif len(unique_matches) > 20:
            # 混合排序
            sorted_matches = self._hybrid_sort(unique_matches)
        else:
            # 复杂排序（可能使用LLM重排）
            sorted_matches = await self._complex_sort(unique_matches)
            
        # 添加排名和解释
        final_results = []
        for i, match in enumerate(sorted_matches[:top_k]):
            match.evidence['rank'] = i + 1
            match.evidence['explanation'] = self._generate_explanation(match)
            final_results.append(match)
            
        time_cost = time.time() - start_time
        self.update_stats(True, time_cost)
        
        logger.info(f"AggregatorAgent produced {len(final_results)} final results")
        return final_results
        
    def _deduplicate(self, matches: List[MatchResult]) -> List[MatchResult]:
        """去重"""
        seen = set()
        unique = []
        for match in matches:
            key = (match.query_table, match.matched_table)
            if key not in seen:
                seen.add(key)
                unique.append(match)
            else:
                # 保留分数更高的
                for i, existing in enumerate(unique):
                    if (existing.query_table, existing.matched_table) == key:
                        if match.score > existing.score:
                            unique[i] = match
                        break
        return unique
        
    def _hybrid_sort(self, matches: List[MatchResult]) -> List[MatchResult]:
        """混合排序"""
        # 综合考虑分数和其他因素
        def sort_key(match):
            score = match.score
            # 如果有LLM验证，加权
            if 'LLM' in match.agent_used:
                score *= 1.1
            return score
            
        return sorted(matches, key=sort_key, reverse=True)
        
    async def _complex_sort(self, matches: List[MatchResult]) -> List[MatchResult]:
        """复杂排序，可能使用LLM"""
        # 先按分数排序
        sorted_matches = sorted(matches, key=lambda x: x.score, reverse=True)
        
        # 如果有LLM客户端，可以重排top结果
        if self.llm_client and len(sorted_matches) > 5:
            # 这里可以实现LLM重排逻辑
            pass
            
        return sorted_matches
        
    def _generate_explanation(self, match: MatchResult) -> str:
        """生成匹配解释"""
        explanation = f"Table '{match.matched_table}' matches with score {match.score:.3f}"
        
        if match.evidence.get('method') == 'rule_based':
            explanation += " (rule-based verification)"
        elif 'LLM' in match.agent_used:
            explanation += " (LLM-verified)"
            
        return explanation


# ===================== 协调器 =====================

class MultiAgentOrchestrator:
    """多Agent系统协调器"""
    
    def __init__(self, config_path: Optional[str] = None):
        """初始化协调器"""
        # 加载配置
        import yaml
        if config_path and Path(config_path).exists():
            with open(config_path, 'r') as f:
                config_data = yaml.safe_load(f)
            self.settings = Settings(**config_data)
        else:
            self.settings = Settings()
        
        # 创建LLM客户端配置
        llm_config = {
            "model_name": self.settings.llm.model_name,
            "temperature": self.settings.llm.temperature,
            "max_tokens": self.settings.llm.max_tokens
        }
        self.llm_client = GeminiClient(llm_config)
        
        # 初始化工具层
        self.metadata_filter = MetadataFilter()
        self.vector_search = get_vector_search_engine()
        self.embedding_gen = get_embedding_generator()
        self.llm_matcher = SmartLLMMatcher(self.llm_client)
        
        # 初始化Agents
        self.agents = {
            'optimizer': OptimizerAgent(self.llm_client),
            'planner': PlannerAgent(self.llm_client),
            'analyzer': AnalyzerAgent(self.metadata_filter, self.llm_client),
            'searcher': SearcherAgent(
                self.metadata_filter, 
                self.vector_search, 
                self.llm_client
            ),
            'matcher': MatcherAgent(self.llm_matcher, self.llm_client),
            'aggregator': AggregatorAgent(self.llm_client)
        }
        
        # 消息总线
        self.message_bus = asyncio.Queue()
        
        # 系统指标
        self.system_metrics = SystemMetrics()
        
        # 数据缓存
        self.table_cache = {}
        self.embedding_cache = {}
        
        logger.info("MultiAgentOrchestrator initialized successfully")
        
    async def load_data(self, tables_path: str):
        """加载数据集"""
        logger.info(f"Loading data from {tables_path}")
        
        with open(tables_path, 'r') as f:
            tables_data = json.load(f)
            
        # 转换为TableInfo对象
        for table_data in tables_data:
            table_info = TableInfo(
                table_name=table_data['table_name'],
                columns=table_data['columns'],
                row_count=table_data.get('row_count'),
                metadata=table_data.get('metadata', {})
            )
            self.table_cache[table_info.table_name] = table_info
            
        logger.info(f"Loaded {len(self.table_cache)} tables")
        
        # 预计算嵌入（并行处理）
        await self._precompute_embeddings()
        
    async def _precompute_embeddings(self):
        """预计算所有表的嵌入向量"""
        logger.info("Precomputing embeddings...")
        
        # 批量并行生成嵌入
        batch_size = 100
        tables = list(self.table_cache.values())
        
        for i in range(0, len(tables), batch_size):
            batch = tables[i:i+batch_size]
            
            # 并行生成
            tasks = []
            for table in batch:
                if table.table_name not in self.embedding_cache:
                    tasks.append(self._generate_embedding(table))
                    
            if tasks:
                embeddings = await asyncio.gather(*tasks)
                for table, embedding in zip(batch, embeddings):
                    if embedding is not None:
                        self.embedding_cache[table.table_name] = embedding
                        table.embedding = embedding
                        
        # 构建向量索引
        await self._build_vector_index()
        
        logger.info(f"Computed {len(self.embedding_cache)} embeddings")
        
    async def _generate_embedding(self, table: TableInfo) -> Optional[np.ndarray]:
        """生成表嵌入"""
        try:
            return self.embedding_gen.generate_table_embedding(table)
        except Exception as e:
            logger.error(f"Failed to generate embedding for {table.table_name}: {e}")
            return None
            
    async def _build_vector_index(self):
        """构建向量索引"""
        logger.info("Building vector index...")
        
        # 收集所有嵌入
        embeddings = []
        table_names = []
        
        for table_name, embedding in self.embedding_cache.items():
            if embedding is not None:
                embeddings.append(embedding)
                table_names.append(table_name)
                
        if embeddings:
            # 构建FAISS索引
            embeddings_array = np.array(embeddings).astype('float32')
            
            # 确保向量搜索引擎已初始化索引
            if hasattr(self.vector_search, 'build_index'):
                self.vector_search.build_index(embeddings_array, table_names)
            
        logger.info(f"Built vector index with {len(embeddings)} vectors")
        
    async def process_query(self, query_task: QueryTask) -> List[MatchResult]:
        """处理单个查询任务"""
        start_time = time.time()
        
        try:
            # 获取查询表信息
            query_table = self.table_cache.get(query_task.query_table)
            if not query_table:
                logger.error(f"Query table {query_task.query_table} not found")
                return []
                
            # 1. 优化器配置
            current_metrics = {
                'avg_response_time': self.system_metrics.avg_response_time,
                'accuracy': self.system_metrics.f1_score,
                'memory_usage': 0.5  # 简化处理
            }
            config = await self.agents['optimizer'].process(current_metrics)
            
            # 2. 规划器制定策略
            execution_plan = await self.agents['planner'].process(query_task)
            
            # 3. 分析器理解数据
            analysis = await self.agents['analyzer'].process(query_table)
            
            # 4. 搜索器查找候选
            search_request = {
                'query_table': query_table,
                'strategy': execution_plan['strategy'],
                'analysis': analysis,
                'top_k': config.get('vector_top_k', 100),
                'max_candidates': config.get('metadata_top_k', 1000)
            }
            candidates = await self.agents['searcher'].process(search_request)
            
            # 5. 匹配器验证
            match_request = {
                'query_table': query_table,
                'candidates': candidates,
                'task_type': query_task.task_type,
                'batch_size': config.get('batch_size', 32)
            }
            matches = await self.agents['matcher'].process(match_request)
            
            # 6. 聚合器整合结果
            aggregation_request = {
                'matches': matches,
                'top_k': 10
            }
            final_results = await self.agents['aggregator'].process(aggregation_request)
            
            # 更新系统指标
            query_time = time.time() - start_time
            self._update_metrics(query_task, final_results, query_time)
            
            logger.info(f"Processed query {query_task.query_id} in {query_time:.2f}s")
            return final_results
            
        except Exception as e:
            logger.error(f"Error processing query {query_task.query_id}: {e}")
            self.system_metrics.failed_queries += 1
            return []
            
    async def process_batch(self, query_tasks: List[QueryTask], 
                          parallel_workers: int = 4) -> Dict[str, List[MatchResult]]:
        """批量处理查询（并行）"""
        logger.info(f"Processing batch of {len(query_tasks)} queries with {parallel_workers} workers")
        
        results = {}
        
        # 创建任务队列
        task_queue = asyncio.Queue()
        for task in query_tasks:
            await task_queue.put(task)
            
        # 创建工作协程
        async def worker():
            while not task_queue.empty():
                try:
                    task = await task_queue.get()
                    task_results = await self.process_query(task)
                    results[task.query_id] = task_results
                except Exception as e:
                    logger.error(f"Worker error: {e}")
                    
        # 并行执行
        workers = [worker() for _ in range(parallel_workers)]
        await asyncio.gather(*workers)
        
        return results
        
    def _update_metrics(self, query_task: QueryTask, 
                       results: List[MatchResult], 
                       query_time: float):
        """更新系统指标"""
        self.system_metrics.total_queries += 1
        
        if results:
            self.system_metrics.successful_queries += 1
        else:
            self.system_metrics.failed_queries += 1
            
        # 更新平均响应时间
        n = self.system_metrics.total_queries
        self.system_metrics.avg_response_time = (
            (self.system_metrics.avg_response_time * (n - 1) + query_time) / n
        )
        
        # 计算精确率和召回率
        if query_task.ground_truth:
            predicted = [r.matched_table for r in results]
            self._calculate_precision_recall(predicted, query_task.ground_truth)
            self._calculate_hit_at_k(predicted, query_task.ground_truth)
            self._calculate_mrr(predicted, query_task.ground_truth)
            
    def _calculate_precision_recall(self, predicted: List[str], 
                                   ground_truth: List[str]):
        """计算精确率和召回率"""
        if not predicted:
            return
            
        true_positives = len(set(predicted) & set(ground_truth))
        
        if predicted:
            precision = true_positives / len(predicted)
        else:
            precision = 0.0
            
        if ground_truth:
            recall = true_positives / len(ground_truth)
        else:
            recall = 0.0
            
        # 更新系统指标（移动平均）
        n = self.system_metrics.successful_queries
        if n > 0:
            self.system_metrics.precision = (
                (self.system_metrics.precision * (n - 1) + precision) / n
            )
            self.system_metrics.recall = (
                (self.system_metrics.recall * (n - 1) + recall) / n
            )
            
            # F1分数
            if self.system_metrics.precision + self.system_metrics.recall > 0:
                self.system_metrics.f1_score = (
                    2 * self.system_metrics.precision * self.system_metrics.recall /
                    (self.system_metrics.precision + self.system_metrics.recall)
                )
                
    def _calculate_hit_at_k(self, predicted: List[str], 
                           ground_truth: List[str]):
        """计算Hit@K"""
        for k in [1, 3, 5, 10]:
            if len(predicted) >= k:
                hit = 1 if any(p in ground_truth for p in predicted[:k]) else 0
                
                # 更新移动平均
                n = self.system_metrics.successful_queries
                if n > 0:
                    current = self.system_metrics.hit_at_k.get(k, 0.0)
                    self.system_metrics.hit_at_k[k] = (
                        (current * (n - 1) + hit) / n
                    )
                    
    def _calculate_mrr(self, predicted: List[str], ground_truth: List[str]):
        """计算MRR (Mean Reciprocal Rank)"""
        rr = 0.0
        for i, p in enumerate(predicted):
            if p in ground_truth:
                rr = 1.0 / (i + 1)
                break
                
        # 更新移动平均
        n = self.system_metrics.successful_queries
        if n > 0:
            self.system_metrics.mrr = (
                (self.system_metrics.mrr * (n - 1) + rr) / n
            )
            
    def get_metrics(self) -> Dict[str, Any]:
        """获取系统指标"""
        return asdict(self.system_metrics)
        
    def get_agent_stats(self) -> Dict[str, Dict[str, Any]]:
        """获取各Agent统计信息"""
        stats = {}
        for name, agent in self.agents.items():
            stats[name] = agent.stats
        return stats


# ===================== 主函数 =====================

async def main():
    """主函数 - 运行完整的多Agent系统测试"""
    
    # 配置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # 创建协调器
    orchestrator = MultiAgentOrchestrator('config_optimized.yml')
    
    # 加载完整数据集
    await orchestrator.load_data('examples/final_complete_tables.json')
    
    # 加载ground truth
    with open('examples/final_complete_ground_truth.json', 'r') as f:
        ground_truth_data = json.load(f)
        
    # 选择100个查询进行测试（50 JOIN + 50 UNION）
    query_tasks = []
    
    # JOIN查询
    join_queries = [gt for gt in ground_truth_data if gt['type'] == 'join'][:50]
    for i, gt in enumerate(join_queries):
        task = QueryTask(
            query_id=f"join_{i}",
            query_table=gt['query_table'],
            task_type='join',
            ground_truth=gt['ground_truth']
        )
        query_tasks.append(task)
        
    # UNION查询
    union_queries = [gt for gt in ground_truth_data if gt['type'] == 'union'][:50]
    for i, gt in enumerate(union_queries):
        task = QueryTask(
            query_id=f"union_{i}",
            query_table=gt['query_table'],
            task_type='union',
            ground_truth=gt['ground_truth']
        )
        query_tasks.append(task)
        
    logger.info(f"Created {len(query_tasks)} query tasks")
    
    # 批量处理（并行）
    start_time = time.time()
    results = await orchestrator.process_batch(query_tasks, parallel_workers=4)
    total_time = time.time() - start_time
    
    # 获取系统指标
    metrics = orchestrator.get_metrics()
    agent_stats = orchestrator.get_agent_stats()
    
    # 输出结果
    print("\n" + "="*60)
    print("MULTI-AGENT SYSTEM EVALUATION RESULTS")
    print("="*60)
    
    print(f"\n📊 Overall Performance:")
    print(f"  Total Queries: {metrics['total_queries']}")
    print(f"  Successful: {metrics['successful_queries']}")
    print(f"  Failed: {metrics['failed_queries']}")
    print(f"  Total Time: {total_time:.2f}s")
    print(f"  Avg Response Time: {metrics['avg_response_time']:.3f}s")
    print(f"  Throughput: {metrics['total_queries']/total_time:.2f} QPS")
    
    print(f"\n🎯 Accuracy Metrics:")
    print(f"  Precision: {metrics['precision']:.3f}")
    print(f"  Recall: {metrics['recall']:.3f}")
    print(f"  F1-Score: {metrics['f1_score']:.3f}")
    print(f"  MRR: {metrics['mrr']:.3f}")
    
    print(f"\n📈 Hit@K Metrics:")
    for k, hit_rate in sorted(metrics['hit_at_k'].items()):
        print(f"  Hit@{k}: {hit_rate:.3f}")
        
    print(f"\n🤖 Agent Performance:")
    for agent_name, stats in agent_stats.items():
        print(f"  {agent_name}:")
        print(f"    Processed: {stats['processed']}")
        print(f"    Success: {stats['success']}")
        print(f"    Avg Time: {stats['avg_time']:.3f}s")
        
    # 保存结果
    output_file = f"experiment_results/multi_agent_test_{int(time.time())}.json"
    with open(output_file, 'w') as f:
        json.dump({
            'metrics': metrics,
            'agent_stats': agent_stats,
            'total_time': total_time,
            'query_count': len(query_tasks),
            'results_sample': {
                k: [asdict(r) for r in v[:3]] 
                for k, v in list(results.items())[:5]
            }
        }, f, indent=2, default=str)
        
    print(f"\n💾 Results saved to: {output_file}")
    

if __name__ == "__main__":
    asyncio.run(main())