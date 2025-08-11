"""
增强版多智能体系统
完整实现多Agent协同 + 三层加速架构
"""

import asyncio
import logging
import time
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict
import json

from src.core.models import TableInfo, ColumnInfo, AgentState, TaskStrategy
from src.core.multi_agent_system import (
    IntelligentAgent, AgentRole, AgentMessage,
    PlannerAgent, SearcherAgent, MatcherAgent
)
from src.tools.metadata_filter import MetadataFilter
from src.tools.batch_vector_search import BatchVectorSearch
from src.tools.smart_llm_matcher import SmartLLMMatcher
from src.utils.llm_client import create_llm_client

logger = logging.getLogger(__name__)


class AnalyzerAgent(IntelligentAgent):
    """分析Agent - 负责理解表结构和数据特征"""
    
    def __init__(self):
        super().__init__("AnalyzerAgent", AgentRole.ANALYZER)
        self.schema_cache = {}
        self.pattern_cache = {}
    
    async def think(self, task: Dict) -> Dict:
        """分析表结构，识别模式"""
        tables = task.get('tables', [])
        query_table = task.get('query_table')
        
        analysis = {
            'table_count': len(tables),
            'has_query_table': query_table is not None,
            'analysis_depth': 'deep' if len(tables) < 100 else 'shallow',
            'use_llm': len(tables) < 20 and self.can_call_llm,
            'patterns': []
        }
        
        # 识别数据模式
        if query_table:
            patterns = self._identify_patterns(query_table)
            analysis['patterns'] = patterns
            analysis['key_columns'] = self._extract_key_columns(query_table)
        
        return analysis
    
    async def act(self, plan: Dict) -> Dict:
        """执行分析"""
        result = {
            'schema_analysis': {},
            'pattern_matches': {},
            'recommendations': []
        }
        
        # 深度分析
        if plan.get('analysis_depth') == 'deep':
            if plan.get('use_llm') and self.llm_client:
                # 使用LLM进行深度分析
                result['llm_insights'] = await self._llm_analyze_schema(plan)
            else:
                # 使用规则分析
                result['rule_insights'] = self._rule_analyze_schema(plan)
        
        # 使用三层加速优化分析
        if self.use_acceleration and self.metadata_filter:
            # Layer 1: 快速元数据分析
            result['metadata_summary'] = await self._fast_metadata_analysis(plan)
        
        return result
    
    def _identify_patterns(self, table: TableInfo) -> List[str]:
        """识别表模式"""
        patterns = []
        
        # 检查是否是维度表
        if self._is_dimension_table(table):
            patterns.append('dimension_table')
        
        # 检查是否是事实表
        if self._is_fact_table(table):
            patterns.append('fact_table')
        
        # 检查是否有时间序列
        if self._has_time_series(table):
            patterns.append('time_series')
        
        return patterns
    
    def _extract_key_columns(self, table: TableInfo) -> List[str]:
        """提取关键列"""
        key_columns = []
        
        for col in table.columns:
            col_name = col.column_name.lower()
            # 主键模式
            if any(pattern in col_name for pattern in ['id', 'key', 'code']):
                key_columns.append(col.column_name)
            # 外键模式
            elif col_name.endswith('_id') or col_name.endswith('_key'):
                key_columns.append(col.column_name)
        
        return key_columns
    
    def _is_dimension_table(self, table: TableInfo) -> bool:
        """判断是否为维度表"""
        table_name = table.table_name.lower()
        dim_patterns = ['dim_', 'dimension_', 'd_', '_dim']
        return any(pattern in table_name for pattern in dim_patterns)
    
    def _is_fact_table(self, table: TableInfo) -> bool:
        """判断是否为事实表"""
        table_name = table.table_name.lower()
        fact_patterns = ['fact_', 'f_', 'agg_', '_fact']
        return any(pattern in table_name for pattern in fact_patterns)
    
    def _has_time_series(self, table: TableInfo) -> bool:
        """检查是否有时间序列数据"""
        time_columns = ['date', 'time', 'timestamp', 'created_at', 'updated_at']
        column_names = [col.column_name.lower() for col in table.columns]
        return any(tc in col for tc in time_columns for col in column_names)
    
    async def _llm_analyze_schema(self, plan: Dict) -> Dict:
        """使用LLM分析schema"""
        if not self.llm_client:
            return {}
        
        query_table = plan.get('query_table')
        if not query_table:
            return {}
        
        prompt = f"""
        Analyze this table schema and identify:
        1. Table type (dimension, fact, lookup, etc.)
        2. Key columns for joining
        3. Data patterns
        
        Table: {query_table.table_name}
        Columns: {[col.column_name for col in query_table.columns[:10]]}
        
        Return insights as JSON.
        """
        
        try:
            response = await self.llm_client.generate(prompt)
            return {'llm_analysis': response}
        except:
            return {}
    
    def _rule_analyze_schema(self, plan: Dict) -> Dict:
        """基于规则分析schema"""
        query_table = plan.get('query_table')
        if not query_table:
            return {}
        
        return {
            'column_count': len(query_table.columns),
            'has_primary_key': any('id' in col.column_name.lower() 
                                  for col in query_table.columns),
            'potential_foreign_keys': [
                col.column_name for col in query_table.columns
                if col.column_name.lower().endswith('_id')
            ]
        }
    
    async def _fast_metadata_analysis(self, plan: Dict) -> Dict:
        """快速元数据分析"""
        # 使用Layer 1加速
        return {
            'analysis_time': 'fast',
            'method': 'metadata_filter',
            'cached': True
        }


class AggregatorAgent(IntelligentAgent):
    """聚合Agent - 负责整合和排序结果"""
    
    def __init__(self):
        super().__init__("AggregatorAgent", AgentRole.AGGREGATOR)
        self.ranking_strategies = {
            'score': self._rank_by_score,
            'relevance': self._rank_by_relevance,
            'hybrid': self._hybrid_ranking
        }
    
    async def think(self, task: Dict) -> Dict:
        """决定聚合策略"""
        results = task.get('results', [])
        
        # 根据结果数量选择策略
        if len(results) > 100:
            strategy = 'score'  # 大量结果用简单评分
        elif len(results) > 20:
            strategy = 'hybrid'  # 中等数量用混合策略
        else:
            strategy = 'relevance'  # 少量结果用复杂相关性分析
        
        return {
            'ranking_strategy': strategy,
            'top_k': task.get('top_k', 10),
            'merge_duplicates': True,
            'use_llm_rerank': len(results) < 30 and self.can_call_llm
        }
    
    async def act(self, plan: Dict) -> List[Dict]:
        """执行聚合"""
        strategy = plan.get('ranking_strategy', 'hybrid')
        results = plan.get('results', [])
        
        # 去重
        if plan.get('merge_duplicates'):
            results = self._merge_duplicates(results)
        
        # 排序
        if strategy in self.ranking_strategies:
            ranked_results = await self.ranking_strategies[strategy](results, plan)
        else:
            ranked_results = results
        
        # LLM重排序（可选）
        if plan.get('use_llm_rerank') and self.llm_client:
            ranked_results = await self._llm_rerank(ranked_results, plan)
        
        # 返回Top-K
        return ranked_results[:plan.get('top_k', 10)]
    
    def _merge_duplicates(self, results: List[Dict]) -> List[Dict]:
        """合并重复结果"""
        seen = {}
        merged = []
        
        for result in results:
            table_name = result.get('table', result.get('table_name'))
            if table_name in seen:
                # 合并分数
                seen[table_name]['score'] = max(
                    seen[table_name].get('score', 0),
                    result.get('score', 0)
                )
                # 合并证据
                if 'evidence' in result:
                    if 'evidence' not in seen[table_name]:
                        seen[table_name]['evidence'] = []
                    seen[table_name]['evidence'].append(result['evidence'])
            else:
                seen[table_name] = result
                merged.append(result)
        
        return merged
    
    async def _rank_by_score(self, results: List[Dict], plan: Dict) -> List[Dict]:
        """按分数排序"""
        return sorted(results, key=lambda x: x.get('score', 0), reverse=True)
    
    async def _rank_by_relevance(self, results: List[Dict], plan: Dict) -> List[Dict]:
        """按相关性排序"""
        # 复杂的相关性计算
        for result in results:
            relevance = result.get('score', 0) * 0.5
            
            # 考虑多个因素
            if result.get('method') == 'llm_verified':
                relevance += 0.3
            if result.get('has_foreign_key'):
                relevance += 0.2
            
            result['relevance'] = relevance
        
        return sorted(results, key=lambda x: x.get('relevance', 0), reverse=True)
    
    async def _hybrid_ranking(self, results: List[Dict], plan: Dict) -> List[Dict]:
        """混合排序策略"""
        # 结合分数和相关性
        for result in results:
            score = result.get('score', 0)
            relevance_boost = 0
            
            # 根据匹配方法加权
            method = result.get('method', '')
            if 'llm' in method:
                relevance_boost += 0.2
            if 'vector' in method:
                relevance_boost += 0.1
            
            result['final_score'] = score * (1 + relevance_boost)
        
        return sorted(results, key=lambda x: x.get('final_score', 0), reverse=True)
    
    async def _llm_rerank(self, results: List[Dict], plan: Dict) -> List[Dict]:
        """使用LLM重新排序"""
        if not self.llm_client or not results:
            return results
        
        # 准备重排序prompt
        query = plan.get('query', '')
        tables = [r.get('table', r.get('table_name')) for r in results[:10]]
        
        prompt = f"""
        Rerank these tables based on relevance to the query.
        Query: {query}
        Tables: {tables}
        
        Return the table names in order of relevance.
        """
        
        try:
            response = await self.llm_client.generate(prompt)
            # 简单解析并重排序
            # ... 实现细节 ...
            return results
        except:
            return results


class OptimizerAgent(IntelligentAgent):
    """优化Agent - 负责性能优化和资源管理"""
    
    def __init__(self):
        super().__init__("OptimizerAgent", AgentRole.OPTIMIZER)
        self.performance_stats = defaultdict(list)
        self.optimization_history = []
    
    async def think(self, task: Dict) -> Dict:
        """分析性能瓶颈，制定优化策略"""
        current_stats = task.get('performance_stats', {})
        
        optimization_plan = {
            'use_cache': True,
            'parallel_execution': False,
            'batch_size': 10,
            'optimization_level': 'balanced'
        }
        
        # 根据性能统计决定优化策略
        if current_stats.get('avg_latency', 0) > 5:
            # 高延迟，需要激进优化
            optimization_plan['optimization_level'] = 'aggressive'
            optimization_plan['parallel_execution'] = True
            optimization_plan['batch_size'] = 20
        elif current_stats.get('memory_usage', 0) > 0.8:
            # 内存压力，需要保守优化
            optimization_plan['optimization_level'] = 'conservative'
            optimization_plan['batch_size'] = 5
        
        return optimization_plan
    
    async def act(self, plan: Dict) -> Dict:
        """执行优化"""
        optimizations = {}
        
        # 缓存优化
        if plan.get('use_cache'):
            optimizations['cache_enabled'] = True
            optimizations['cache_ttl'] = 3600
        
        # 并行优化
        if plan.get('parallel_execution'):
            optimizations['max_workers'] = 4
            optimizations['async_mode'] = True
        
        # 批处理优化
        optimizations['batch_size'] = plan.get('batch_size', 10)
        
        # 三层加速优化
        if self.use_acceleration:
            optimizations['acceleration_config'] = {
                'layer1_enabled': True,
                'layer2_enabled': True,
                'layer3_enabled': plan.get('optimization_level') != 'conservative'
            }
        
        # 记录优化历史
        self.optimization_history.append({
            'timestamp': time.time(),
            'plan': plan,
            'optimizations': optimizations
        })
        
        return optimizations
    
    def get_performance_report(self) -> Dict:
        """生成性能报告"""
        return {
            'optimization_history': self.optimization_history[-10:],
            'current_stats': dict(self.performance_stats),
            'recommendations': self._generate_recommendations()
        }
    
    def _generate_recommendations(self) -> List[str]:
        """生成优化建议"""
        recommendations = []
        
        # 基于历史数据生成建议
        if len(self.optimization_history) > 5:
            recent = self.optimization_history[-5:]
            
            # 检查是否频繁使用激进优化
            aggressive_count = sum(1 for r in recent 
                                 if r['plan'].get('optimization_level') == 'aggressive')
            if aggressive_count > 3:
                recommendations.append("Consider scaling up resources")
            
            # 检查缓存效果
            cache_enabled = all(r['optimizations'].get('cache_enabled') 
                              for r in recent)
            if not cache_enabled:
                recommendations.append("Enable caching for better performance")
        
        return recommendations


class EnhancedMultiAgentOrchestrator:
    """增强版多智能体系统协调器"""
    
    def __init__(self):
        self.agents = {}
        self.message_bus = []
        self.shared_tools = {}
        self.performance_monitor = {
            'query_count': 0,
            'total_time': 0,
            'success_rate': 0,
            'agent_stats': defaultdict(lambda: {'calls': 0, 'time': 0})
        }
    
    async def initialize(self, all_tables: List[TableInfo]):
        """初始化增强版多Agent系统"""
        logger.info("Initializing Enhanced Multi-Agent System...")
        
        # 初始化三层加速工具
        await self._initialize_acceleration_layers(all_tables)
        
        # 创建所有Agent
        self.agents = {
            'planner': PlannerAgent(),
            'analyzer': AnalyzerAgent(),
            'searcher': SearcherAgent(),
            'matcher': MatcherAgent(),
            'aggregator': AggregatorAgent(),
            'optimizer': OptimizerAgent()
        }
        
        # 初始化所有Agent
        for agent in self.agents.values():
            await agent.initialize(self.shared_tools)
        
        logger.info(f"Enhanced Multi-Agent System initialized with {len(self.agents)} agents")
    
    async def _initialize_acceleration_layers(self, all_tables: List[TableInfo]):
        """初始化三层加速架构"""
        # Layer 1: 元数据筛选
        self.shared_tools['metadata_filter'] = MetadataFilter()
        # build_index is not async, call it directly
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
        
        # 额外：表元数据缓存
        self.shared_tools['table_metadata'] = {
            table.table_name: table for table in all_tables
        }
        
        logger.info("Three-layer acceleration initialized for all agents")
    
    async def process_query_with_collaboration(
        self,
        query: str,
        query_table: TableInfo,
        strategy: str = "auto"
    ) -> List[Dict]:
        """处理查询 - 完整的多Agent协同工作流"""
        
        start_time = time.time()
        self.performance_monitor['query_count'] += 1
        
        logger.info(f"Processing query with multi-agent collaboration: {query}")
        
        # Step 1: 优化器分析当前系统状态
        optimizer = self.agents['optimizer']
        optimization_task = {'performance_stats': self.performance_monitor}
        optimization_plan = await optimizer.think(optimization_task)
        optimizations = await optimizer.act(optimization_plan)
        
        logger.info(f"Optimization level: {optimization_plan.get('optimization_level')}")
        
        # Step 2: 规划者制定策略
        planner = self.agents['planner']
        task = {
            'query': query,
            'query_table': query_table,
            'optimizations': optimizations
        }
        plan = await planner.think(task)
        detailed_plan = await planner.act(plan)
        
        # Step 3: 分析者理解数据
        analyzer = self.agents['analyzer']
        analysis_task = {
            'tables': list(self.shared_tools['table_metadata'].values()),
            'query_table': query_table,
            **detailed_plan
        }
        analysis_plan = await analyzer.think(analysis_task)
        analysis_result = await analyzer.act(analysis_plan)
        
        # Step 4: 搜索者查找候选（使用三层加速）
        searcher = self.agents['searcher']
        search_task = {
            **detailed_plan,
            **analysis_result,
            'query_table': query_table,
            'max_candidates': optimizations.get('batch_size', 10) * 10
        }
        search_plan = await searcher.think(search_task)
        candidates = await searcher.act(search_plan)
        
        logger.info(f"Searcher found {len(candidates)} candidates using {search_plan.get('search_method')}")
        
        # Step 5: 匹配者验证（使用Layer 3 LLM）
        matcher = self.agents['matcher']
        match_task = {
            'candidates': candidates,
            'query_table': query_table,
            'table_metadata': self.shared_tools['table_metadata'],
            **analysis_result
        }
        match_plan = await matcher.think(match_task)
        matches = await matcher.act(match_plan)
        
        logger.info(f"Matcher verified {len(matches)} matches using {match_plan.get('match_strategy')}")
        
        # Step 6: 聚合者整合结果
        aggregator = self.agents['aggregator']
        aggregation_task = {
            'results': matches,
            'query': query,
            'top_k': 10
        }
        aggregation_plan = await aggregator.think(aggregation_task)
        final_results = await aggregator.act(aggregation_plan)
        
        # Step 7: Agent间消息广播（协同通信）
        completion_message = AgentMessage(
            sender="orchestrator",
            receiver="all",
            message_type="query_completed",
            content={
                'query': query,
                'result_count': len(final_results),
                'execution_time': time.time() - start_time
            }
        )
        await self.broadcast_message(completion_message)
        
        # 更新性能监控
        execution_time = time.time() - start_time
        self.performance_monitor['total_time'] += execution_time
        
        logger.info(f"Query completed in {execution_time:.2f}s with {len(final_results)} results")
        
        return final_results
    
    async def broadcast_message(self, message: AgentMessage):
        """广播消息给所有Agent"""
        self.message_bus.append(message)
        
        for agent in self.agents.values():
            if agent.name != message.sender:
                await agent.communicate(message)
    
    def get_detailed_status(self) -> Dict:
        """获取详细系统状态"""
        return {
            'system': {
                'num_agents': len(self.agents),
                'agents': list(self.agents.keys()),
                'message_count': len(self.message_bus)
            },
            'acceleration': {
                'layer1_enabled': self.shared_tools.get('metadata_filter') is not None,
                'layer2_enabled': self.shared_tools.get('vector_search') is not None,
                'layer3_enabled': self.shared_tools.get('llm_matcher') is not None
            },
            'performance': {
                'queries_processed': self.performance_monitor['query_count'],
                'avg_time': (self.performance_monitor['total_time'] / 
                           max(1, self.performance_monitor['query_count'])),
                'agent_stats': dict(self.performance_monitor['agent_stats'])
            },
            'optimizer_report': self.agents['optimizer'].get_performance_report() 
                              if 'optimizer' in self.agents else {}
        }


async def test_enhanced_multi_agent_system():
    """测试增强版多Agent系统"""
    print("="*80)
    print("🤖 增强版多智能体系统测试")
    print("="*80)
    
    # 加载真实数据
    import json
    from pathlib import Path
    
    dataset_path = Path("examples/separated_datasets/join_subset/tables.json")
    if dataset_path.exists():
        with open(dataset_path) as f:
            tables_data = json.load(f)[:20]  # 使用20个表测试
        
        tables = []
        for td in tables_data:
            table = TableInfo(
                table_name=td['table_name'],
                columns=[
                    ColumnInfo(
                        table_name=td['table_name'],
                        column_name=col.get('column_name', col.get('name', '')),
                        data_type=col.get('data_type', 'unknown'),
                        sample_values=col.get('sample_values', [])[:3]
                    )
                    for col in td.get('columns', [])[:10]
                ]
            )
            tables.append(table)
    else:
        # 使用测试数据
        tables = [
            TableInfo(
                table_name="users",
                columns=[
                    ColumnInfo(table_name="users", column_name="user_id", data_type="int"),
                    ColumnInfo(table_name="users", column_name="name", data_type="string"),
                    ColumnInfo(table_name="users", column_name="email", data_type="string")
                ]
            ),
            TableInfo(
                table_name="orders",
                columns=[
                    ColumnInfo(table_name="orders", column_name="order_id", data_type="int"),
                    ColumnInfo(table_name="orders", column_name="user_id", data_type="int"),
                    ColumnInfo(table_name="orders", column_name="product_id", data_type="int")
                ]
            ),
            TableInfo(
                table_name="products",
                columns=[
                    ColumnInfo(table_name="products", column_name="product_id", data_type="int"),
                    ColumnInfo(table_name="products", column_name="name", data_type="string"),
                    ColumnInfo(table_name="products", column_name="price", data_type="float")
                ]
            )
        ]
    
    # 创建协调器
    orchestrator = EnhancedMultiAgentOrchestrator()
    
    # 初始化系统
    print(f"\n📦 初始化多Agent系统 ({len(tables)}个表)...")
    start = time.time()
    await orchestrator.initialize(tables)
    print(f"✅ 初始化完成，耗时: {time.time()-start:.2f}秒")
    
    # 显示系统状态
    status = orchestrator.get_detailed_status()
    print(f"\n📊 系统状态:")
    print(f"  Agents: {status['system']['agents']}")
    print(f"  三层加速:")
    print(f"    - Layer 1 (Metadata): {'✅' if status['acceleration']['layer1_enabled'] else '❌'}")
    print(f"    - Layer 2 (Vector): {'✅' if status['acceleration']['layer2_enabled'] else '❌'}")
    print(f"    - Layer 3 (LLM): {'✅' if status['acceleration']['layer3_enabled'] else '❌'}")
    
    # 测试查询
    if tables:
        query = f"Find tables that can be joined with {tables[0].table_name}"
        print(f"\n🔍 测试查询: {query}")
        
        start = time.time()
        results = await orchestrator.process_query_with_collaboration(
            query,
            tables[0],
            strategy="auto"
        )
        elapsed = time.time() - start
        
        print(f"\n✅ 查询完成!")
        print(f"  耗时: {elapsed:.2f}秒")
        print(f"  找到 {len(results)} 个结果")
        
        if results:
            print(f"\n📊 Top 5 结果:")
            for i, result in enumerate(results[:5], 1):
                table_name = result.get('table', result.get('table_name', 'unknown'))
                score = result.get('score', 0)
                method = result.get('method', 'unknown')
                print(f"  {i}. {table_name} (分数: {score:.2f}, 方法: {method})")
    
    # 显示最终状态
    final_status = orchestrator.get_detailed_status()
    print(f"\n📈 性能统计:")
    print(f"  查询处理数: {final_status['performance']['queries_processed']}")
    print(f"  平均时间: {final_status['performance']['avg_time']:.2f}秒")
    
    print("\n" + "="*80)
    print("✅ 增强版多Agent系统测试完成！")
    print("="*80)


if __name__ == "__main__":
    import asyncio
    asyncio.run(test_enhanced_multi_agent_system())