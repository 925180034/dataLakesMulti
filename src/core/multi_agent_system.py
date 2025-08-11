"""
真正的多智能体系统实现
集成三层加速架构的多Agent协同系统
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
from abc import ABC, abstractmethod

from src.core.models import TableInfo, ColumnInfo, AgentState, TaskStrategy
from src.tools.metadata_filter import MetadataFilter
from src.tools.batch_vector_search import BatchVectorSearch
from src.tools.smart_llm_matcher import SmartLLMMatcher
from src.utils.llm_client import create_llm_client

logger = logging.getLogger(__name__)


class AgentRole(Enum):
    """Agent角色定义"""
    PLANNER = "planner"          # 规划者：分析任务，制定策略
    ANALYZER = "analyzer"        # 分析者：理解数据结构
    SEARCHER = "searcher"        # 搜索者：查找候选
    MATCHER = "matcher"          # 匹配者：精确匹配
    AGGREGATOR = "aggregator"    # 聚合者：整合结果
    OPTIMIZER = "optimizer"      # 优化者：性能优化


@dataclass
class AgentMessage:
    """Agent间通信消息"""
    sender: str
    receiver: str
    message_type: str
    content: Any
    priority: int = 0


class IntelligentAgent(ABC):
    """智能Agent基类 - 集成三层加速能力"""
    
    def __init__(self, name: str, role: AgentRole):
        self.name = name
        self.role = role
        self.llm_client = None
        
        # 三层加速工具（可选使用）
        self.metadata_filter = None    # Layer 1: 快速规则筛选
        self.vector_search = None      # Layer 2: 向量相似度
        self.llm_matcher = None        # Layer 3: 智能LLM匹配
        
        # Agent能力配置
        self.use_acceleration = True   # 是否使用三层加速
        self.can_call_llm = True      # 是否可以调用LLM
        self.memory = {}              # Agent记忆
        self.message_queue = []       # 消息队列
        
    async def initialize(self, tools: Optional[Dict] = None):
        """初始化Agent和工具"""
        # 初始化LLM客户端
        if self.can_call_llm:
            self.llm_client = create_llm_client()
        
        # 初始化三层加速工具
        if self.use_acceleration and tools:
            self.metadata_filter = tools.get('metadata_filter')
            self.vector_search = tools.get('vector_search')
            self.llm_matcher = tools.get('llm_matcher')
        
        logger.info(f"Agent {self.name} ({self.role.value}) initialized")
    
    @abstractmethod
    async def think(self, task: Any) -> Dict:
        """Agent思考：分析任务，制定计划"""
        pass
    
    @abstractmethod
    async def act(self, plan: Dict) -> Any:
        """Agent行动：执行计划"""
        pass
    
    async def communicate(self, message: AgentMessage):
        """Agent通信：发送/接收消息"""
        self.message_queue.append(message)
    
    async def collaborate(self, other_agents: List['IntelligentAgent'], task: Any) -> Any:
        """Agent协作：与其他Agent合作完成任务"""
        # 思考阶段
        plan = await self.think(task)
        
        # 是否需要其他Agent帮助
        if self._needs_help(plan):
            helpers = self._select_helpers(other_agents, plan)
            for helper in helpers:
                msg = AgentMessage(
                    sender=self.name,
                    receiver=helper.name,
                    message_type="help_request",
                    content=plan
                )
                await helper.communicate(msg)
        
        # 执行阶段
        result = await self.act(plan)
        
        return result
    
    def _needs_help(self, plan: Dict) -> bool:
        """判断是否需要其他Agent帮助"""
        return plan.get('complexity', 0) > 0.7 or plan.get('requires_collaboration', False)
    
    def _select_helpers(self, agents: List['IntelligentAgent'], plan: Dict) -> List['IntelligentAgent']:
        """选择合适的协助Agent"""
        helpers = []
        required_roles = plan.get('required_roles', [])
        
        for agent in agents:
            if agent.role in required_roles and agent != self:
                helpers.append(agent)
        
        return helpers


class PlannerAgent(IntelligentAgent):
    """规划Agent - 负责任务分析和策略制定"""
    
    def __init__(self):
        super().__init__("PlannerAgent", AgentRole.PLANNER)
        self.strategies = {
            "join": self._plan_join_task,
            "union": self._plan_union_task,
            "general": self._plan_general_task
        }
    
    async def think(self, task: Dict) -> Dict:
        """分析任务，确定策略"""
        query = task.get('query', '')
        task_type = self._identify_task_type(query)
        
        # 可以选择：使用LLM分析或使用规则
        if self.can_call_llm and self._is_complex_query(query):
            # 复杂查询使用LLM
            analysis = await self._llm_analyze(query)
            strategy = analysis.get('strategy', 'general')
        else:
            # 简单查询使用规则
            strategy = task_type
        
        plan = {
            'task_type': task_type,
            'strategy': strategy,
            'complexity': self._estimate_complexity(query),
            'required_roles': self._determine_required_agents(task_type),
            'use_acceleration': True  # 建议使用三层加速
        }
        
        return plan
    
    async def act(self, plan: Dict) -> Dict:
        """执行规划"""
        strategy = plan.get('strategy', 'general')
        
        if strategy in self.strategies:
            detailed_plan = await self.strategies[strategy](plan)
        else:
            detailed_plan = await self._plan_general_task(plan)
        
        return detailed_plan
    
    def _identify_task_type(self, query: str) -> str:
        """识别任务类型"""
        query_lower = query.lower()
        if 'join' in query_lower or 'joinable' in query_lower:
            return 'join'
        elif 'union' in query_lower or 'similar' in query_lower:
            return 'union'
        else:
            return 'general'
    
    def _is_complex_query(self, query: str) -> bool:
        """判断查询是否复杂"""
        return len(query.split()) > 10 or 'complex' in query
    
    def _estimate_complexity(self, query: str) -> float:
        """估计任务复杂度"""
        factors = {
            'join': 0.6,
            'union': 0.5,
            'multiple': 0.8,
            'complex': 0.9
        }
        
        complexity = 0.3  # 基础复杂度
        for keyword, weight in factors.items():
            if keyword in query.lower():
                complexity = max(complexity, weight)
        
        return complexity
    
    def _determine_required_agents(self, task_type: str) -> List[AgentRole]:
        """确定需要的Agent"""
        if task_type == 'join':
            return [AgentRole.ANALYZER, AgentRole.SEARCHER, AgentRole.MATCHER]
        elif task_type == 'union':
            return [AgentRole.SEARCHER, AgentRole.MATCHER, AgentRole.AGGREGATOR]
        else:
            return [AgentRole.ANALYZER, AgentRole.SEARCHER]
    
    async def _llm_analyze(self, query: str) -> Dict:
        """使用LLM分析查询"""
        if not self.llm_client:
            return {'strategy': 'general'}
        
        prompt = f"""
        Analyze this data lake query and determine the best strategy:
        Query: {query}
        
        Return a JSON with:
        - strategy: 'join', 'union', or 'general'
        - complexity: 0.0 to 1.0
        - key_tables: list of important table names mentioned
        """
        
        try:
            response = await self.llm_client.generate(prompt)
            # 简单解析响应
            if 'join' in response.lower():
                return {'strategy': 'join'}
            elif 'union' in response.lower():
                return {'strategy': 'union'}
            else:
                return {'strategy': 'general'}
        except:
            return {'strategy': 'general'}
    
    async def _plan_join_task(self, initial_plan: Dict) -> Dict:
        """规划JOIN任务"""
        return {
            **initial_plan,
            'steps': [
                {'agent': 'AnalyzerAgent', 'action': 'analyze_schemas'},
                {'agent': 'SearcherAgent', 'action': 'find_join_candidates'},
                {'agent': 'MatcherAgent', 'action': 'verify_joins'},
                {'agent': 'AggregatorAgent', 'action': 'rank_results'}
            ],
            'optimization': 'use_column_index'
        }
    
    async def _plan_union_task(self, initial_plan: Dict) -> Dict:
        """规划UNION任务"""
        return {
            **initial_plan,
            'steps': [
                {'agent': 'SearcherAgent', 'action': 'find_similar_tables'},
                {'agent': 'MatcherAgent', 'action': 'verify_similarity'},
                {'agent': 'AggregatorAgent', 'action': 'merge_results'}
            ],
            'optimization': 'use_vector_search'
        }
    
    async def _plan_general_task(self, initial_plan: Dict) -> Dict:
        """规划通用任务"""
        return {
            **initial_plan,
            'steps': [
                {'agent': 'AnalyzerAgent', 'action': 'understand_query'},
                {'agent': 'SearcherAgent', 'action': 'broad_search'},
                {'agent': 'AggregatorAgent', 'action': 'compile_results'}
            ],
            'optimization': 'balanced'
        }


class SearcherAgent(IntelligentAgent):
    """搜索Agent - 负责查找候选表"""
    
    def __init__(self):
        super().__init__("SearcherAgent", AgentRole.SEARCHER)
        self.search_methods = {
            'metadata': self._search_by_metadata,
            'vector': self._search_by_vector,
            'hybrid': self._hybrid_search
        }
    
    async def think(self, task: Dict) -> Dict:
        """决定搜索策略"""
        # 根据任务决定搜索方法
        if self.use_acceleration:
            # 使用三层加速
            method = 'hybrid'
        elif task.get('optimization') == 'use_vector_search':
            method = 'vector'
        else:
            method = 'metadata'
        
        return {
            'search_method': method,
            'max_candidates': task.get('max_candidates', 100),
            'use_cache': True
        }
    
    async def act(self, plan: Dict) -> List[Tuple[str, float]]:
        """执行搜索"""
        method = plan.get('search_method', 'hybrid')
        
        if method in self.search_methods:
            candidates = await self.search_methods[method](plan)
        else:
            candidates = await self._hybrid_search(plan)
        
        return candidates
    
    async def _search_by_metadata(self, plan: Dict) -> List[Tuple[str, float]]:
        """使用元数据筛选（Layer 1）"""
        if not self.metadata_filter:
            return []
        
        # 使用MetadataFilter快速筛选
        query_table = plan.get('query_table')
        if query_table:
            candidates = await self.metadata_filter.filter_tables(
                query_table,
                max_candidates=plan.get('max_candidates', 100)
            )
            return candidates
        
        return []
    
    async def _search_by_vector(self, plan: Dict) -> List[Tuple[str, float]]:
        """使用向量搜索（Layer 2）"""
        if not self.vector_search:
            return []
        
        # 使用向量相似度搜索
        query_embedding = plan.get('query_embedding')
        if query_embedding:
            candidates = await self.vector_search.search(
                query_embedding,
                top_k=plan.get('max_candidates', 100)
            )
            return candidates
        
        return []
    
    async def _hybrid_search(self, plan: Dict) -> List[Tuple[str, float]]:
        """混合搜索（Layer 1 + Layer 2）"""
        candidates = []
        
        # 先用元数据筛选
        if self.metadata_filter:
            metadata_candidates = await self._search_by_metadata(plan)
            candidates.extend(metadata_candidates)
        
        # 再用向量搜索补充
        if self.vector_search and len(candidates) < plan.get('max_candidates', 100):
            vector_candidates = await self._search_by_vector(plan)
            # 合并结果，去重
            seen = {c[0] for c in candidates}
            for table, score in vector_candidates:
                if table not in seen:
                    candidates.append((table, score))
        
        # 排序并截断
        candidates.sort(key=lambda x: x[1], reverse=True)
        return candidates[:plan.get('max_candidates', 100)]


class MatcherAgent(IntelligentAgent):
    """匹配Agent - 负责精确匹配验证"""
    
    def __init__(self):
        super().__init__("MatcherAgent", AgentRole.MATCHER)
    
    async def think(self, task: Dict) -> Dict:
        """决定匹配策略"""
        candidates = task.get('candidates', [])
        
        # 根据候选数量决定策略
        if len(candidates) > 50 and self.use_acceleration:
            strategy = 'smart_llm'  # 使用SmartLLMMatcher（Layer 3）
        elif len(candidates) > 20:
            strategy = 'batch_llm'  # 批量LLM
        else:
            strategy = 'detailed_llm'  # 详细LLM分析
        
        return {
            'match_strategy': strategy,
            'batch_size': 10,
            'threshold': 0.7
        }
    
    async def act(self, plan: Dict) -> List[Dict]:
        """执行匹配验证"""
        strategy = plan.get('match_strategy', 'smart_llm')
        
        if strategy == 'smart_llm' and self.llm_matcher:
            # 使用SmartLLMMatcher（三层加速的Layer 3）
            matches = await self._smart_llm_match(plan)
        elif strategy == 'batch_llm' and self.llm_client:
            # 批量调用LLM
            matches = await self._batch_llm_match(plan)
        else:
            # 基于规则的匹配
            matches = await self._rule_based_match(plan)
        
        return matches
    
    async def _smart_llm_match(self, plan: Dict) -> List[Dict]:
        """使用智能LLM匹配器"""
        if not self.llm_matcher:
            return []
        
        candidates = plan.get('candidates', [])
        query_table = plan.get('query_table')
        
        # 调用SmartLLMMatcher
        matches = await self.llm_matcher.match_tables(
            [query_table],
            {'query': candidates},
            plan.get('table_metadata', {})
        )
        
        return matches.get('query', [])
    
    async def _batch_llm_match(self, plan: Dict) -> List[Dict]:
        """批量LLM匹配"""
        if not self.llm_client:
            return []
        
        # 实现批量LLM调用逻辑
        # ...
        
        return []
    
    async def _rule_based_match(self, plan: Dict) -> List[Dict]:
        """基于规则的匹配"""
        candidates = plan.get('candidates', [])
        threshold = plan.get('threshold', 0.7)
        
        matches = []
        for table, score in candidates:
            if score > threshold:
                matches.append({
                    'table': table,
                    'score': score,
                    'method': 'rule_based'
                })
        
        return matches


class MultiAgentOrchestrator:
    """多智能体系统协调器"""
    
    def __init__(self):
        self.agents = {}
        self.message_bus = []  # 消息总线
        
        # 三层加速工具（共享给所有Agent）
        self.shared_tools = {
            'metadata_filter': None,
            'vector_search': None,
            'llm_matcher': None
        }
        
    async def initialize(self, all_tables: List[TableInfo]):
        """初始化多Agent系统"""
        logger.info("Initializing Multi-Agent System...")
        
        # 初始化三层加速工具
        await self._initialize_acceleration_layers(all_tables)
        
        # 创建各种Agent
        self.agents['planner'] = PlannerAgent()
        self.agents['searcher'] = SearcherAgent()
        self.agents['matcher'] = MatcherAgent()
        
        # 初始化所有Agent
        for agent in self.agents.values():
            await agent.initialize(self.shared_tools)
        
        logger.info(f"Multi-Agent System initialized with {len(self.agents)} agents")
    
    async def _initialize_acceleration_layers(self, all_tables: List[TableInfo]):
        """初始化三层加速架构"""
        # Layer 1: 元数据筛选
        self.shared_tools['metadata_filter'] = MetadataFilter()
        self.shared_tools['metadata_filter'].build_index(all_tables)
        
        # Layer 2: 向量搜索
        from src.tools.vector_search import get_vector_search_engine
        from src.tools.embedding import get_embedding_generator
        
        vector_engine = get_vector_search_engine()
        embedding_gen = get_embedding_generator()
        
        # 为所有表创建向量索引
        for table in all_tables:
            # 创建表的文本表示
            table_text = f"{table.table_name} columns: {', '.join([col.column_name for col in table.columns[:10]])}"
            # 生成向量嵌入
            table_embedding = await embedding_gen.generate_text_embedding(table_text)
            await vector_engine.add_table_vector(table, table_embedding)
        
        self.shared_tools['vector_search'] = BatchVectorSearch(vector_engine)
        
        # Layer 3: 智能LLM匹配
        llm_client = create_llm_client()
        self.shared_tools['llm_matcher'] = SmartLLMMatcher(llm_client)
        
        logger.info("Three-layer acceleration initialized")
    
    async def process_query(self, query: str, query_table: TableInfo) -> List[Dict]:
        """处理查询 - 多Agent协同工作"""
        logger.info(f"Processing query: {query}")
        
        # Step 1: 规划者分析任务
        planner = self.agents['planner']
        task = {'query': query, 'query_table': query_table}
        plan = await planner.think(task)
        detailed_plan = await planner.act(plan)
        
        logger.info(f"Plan created: {detailed_plan.get('strategy')}")
        
        # Step 2: 搜索者查找候选
        searcher = self.agents['searcher']
        search_task = {
            **detailed_plan,
            'query_table': query_table
        }
        search_plan = await searcher.think(search_task)
        candidates = await searcher.act(search_plan)
        
        logger.info(f"Found {len(candidates)} candidates")
        
        # Step 3: 匹配者验证
        matcher = self.agents['matcher']
        match_task = {
            'candidates': candidates,
            'query_table': query_table,
            'table_metadata': self.shared_tools.get('table_metadata', {})
        }
        match_plan = await matcher.think(match_task)
        matches = await matcher.act(match_plan)
        
        logger.info(f"Verified {len(matches)} matches")
        
        # Step 4: Agent间通信和协作
        # 这里可以实现更复杂的Agent间交互
        
        return matches
    
    async def broadcast_message(self, message: AgentMessage):
        """广播消息给所有Agent"""
        self.message_bus.append(message)
        
        for agent in self.agents.values():
            if agent.name != message.sender:
                await agent.communicate(message)
    
    def get_system_status(self) -> Dict:
        """获取系统状态"""
        return {
            'num_agents': len(self.agents),
            'agents': list(self.agents.keys()),
            'acceleration_enabled': all(self.shared_tools.values()),
            'message_count': len(self.message_bus)
        }


async def demo_multi_agent_system():
    """演示多Agent系统"""
    print("="*60)
    print("🤖 多智能体系统演示")
    print("="*60)
    
    # 创建测试数据
    tables = [
        TableInfo(
            table_name="users",
            columns=[
                ColumnInfo(table_name="users", column_name="id", data_type="int"),
                ColumnInfo(table_name="users", column_name="name", data_type="string")
            ]
        ),
        TableInfo(
            table_name="orders",
            columns=[
                ColumnInfo(table_name="orders", column_name="order_id", data_type="int"),
                ColumnInfo(table_name="orders", column_name="user_id", data_type="int")
            ]
        )
    ]
    
    # 创建协调器
    orchestrator = MultiAgentOrchestrator()
    
    # 初始化系统
    print("\n📦 初始化多Agent系统...")
    await orchestrator.initialize(tables)
    
    # 显示系统状态
    status = orchestrator.get_system_status()
    print(f"\n✅ 系统状态:")
    print(f"  - Agent数量: {status['num_agents']}")
    print(f"  - Agents: {status['agents']}")
    print(f"  - 三层加速: {'启用' if status['acceleration_enabled'] else '禁用'}")
    
    # 处理查询
    query = "Find tables that can be joined with users table"
    print(f"\n🔍 处理查询: {query}")
    
    results = await orchestrator.process_query(query, tables[0])
    
    print(f"\n📊 结果: 找到{len(results)}个匹配")
    for result in results[:5]:
        print(f"  - {result}")
    
    print("\n" + "="*60)
    print("✅ 多Agent系统演示完成！")
    print("="*60)


if __name__ == "__main__":
    import asyncio
    asyncio.run(demo_multi_agent_system())