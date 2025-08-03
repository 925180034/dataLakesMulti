from typing import Dict, Any, List
import logging
from langgraph.graph import StateGraph, END
from src.core.models import AgentState, TaskStrategy
from src.agents.planner import PlannerAgent
from src.agents.column_discovery import ColumnDiscoveryAgent
from src.agents.table_aggregation import TableAggregationAgent
from src.agents.table_discovery import TableDiscoveryAgent
from src.agents.table_matching import TableMatchingAgent

logger = logging.getLogger(__name__)


class DataLakesWorkflow:
    """数据湖多智能体工作流"""
    
    def __init__(self):
        self.planner = PlannerAgent()
        self.column_discovery = ColumnDiscoveryAgent()
        self.table_aggregation = TableAggregationAgent()
        self.table_discovery = TableDiscoveryAgent()
        self.table_matching = TableMatchingAgent()
        
        # 注释掉索引加载，在初始化时会导致事件循环问题
        # 索引会在实际使用时按需加载
        # self._load_existing_indices()
        
        # 标记，确保只初始化一次向量搜索引擎
        self._vector_search_initialized = False
        
        self.workflow = self._build_workflow()
        self.app = self.workflow.compile()
    
    def _load_existing_indices(self):
        """加载已存在的索引"""
        try:
            from src.config.settings import settings
            import asyncio
            import nest_asyncio
            
            # 允许嵌套的事件循环
            nest_asyncio.apply()
            
            # 异步加载索引
            async def load_indices():
                index_path = settings.vector_db.db_path
                
                # 检查并加载表发现的索引
                if self.table_discovery.vector_search is not None:
                    await self.table_discovery.vector_search.load_index(index_path)
                else:
                    logger.warning("表发现的向量搜索引擎未初始化，跳过索引加载")
                
                # 检查并加载列发现的索引
                if self.column_discovery.vector_search is not None:
                    await self.column_discovery.vector_search.load_index(index_path)
                else:
                    logger.warning("列发现的向量搜索引擎未初始化，跳过索引加载")
                
                logger.info("索引加载任务完成")
            
            # 使用 asyncio.run 来运行异步任务
            asyncio.run(load_indices())
                
        except Exception as e:
            logger.warning(f"加载索引失败（这是正常的，如果索引文件不存在）: {e}")
    
    def _build_workflow(self) -> StateGraph:
        """构建工作流图"""
        workflow = StateGraph(AgentState)
        
        # 添加所有节点
        workflow.add_node("planner", self._planner_node)
        workflow.add_node("column_discovery", self._column_discovery_node)
        workflow.add_node("table_aggregation", self._table_aggregation_node)
        workflow.add_node("table_discovery", self._table_discovery_node)
        workflow.add_node("table_matching", self._table_matching_node)
        workflow.add_node("finalization", self._finalization_node)
        
        # 设置入口点
        workflow.set_entry_point("planner")
        
        # 添加条件路由
        workflow.add_conditional_edges(
            "planner",
            self._route_from_planner,
            {
                "column_discovery": "column_discovery",  # Bottom-Up策略
                "table_discovery": "table_discovery",    # Top-Down策略
                "finalization": "finalization",          # 直接生成报告
                "end": END                                # 结束
            }
        )
        
        # Bottom-Up路径
        workflow.add_conditional_edges(
            "column_discovery",
            self._route_from_column_discovery,
            {
                "table_aggregation": "table_aggregation",
                "finalization": "finalization",
                "end": END
            }
        )
        
        workflow.add_conditional_edges(
            "table_aggregation",
            self._route_from_table_aggregation,
            {
                "finalization": "finalization",
                "end": END
            }
        )
        
        # Top-Down路径
        workflow.add_conditional_edges(
            "table_discovery",
            self._route_from_table_discovery,
            {
                "table_matching": "table_matching",
                "finalization": "finalization",
                "end": END
            }
        )
        
        workflow.add_conditional_edges(
            "table_matching",
            self._route_from_table_matching,
            {
                "finalization": "finalization",
                "end": END
            }
        )
        
        # 最终处理
        workflow.add_edge("finalization", END)
        
        return workflow
    
    async def _planner_node(self, state: AgentState) -> AgentState:
        """规划器节点"""
        try:
            logger.info("执行规划器节点")
            return await self.planner.process(state)
        except Exception as e:
            logger.error(f"规划器节点执行失败: {e}")
            state.add_error(f"规划器执行失败: {e}")
            return state
    
    async def _column_discovery_node(self, state: AgentState) -> AgentState:
        """列发现节点"""
        try:
            logger.info("执行列发现节点")
            return await self.column_discovery.process(state)
        except Exception as e:
            logger.error(f"列发现节点执行失败: {e}")
            state.add_error(f"列发现执行失败: {e}")
            return state
    
    async def _table_aggregation_node(self, state: AgentState) -> AgentState:
        """表聚合节点"""
        try:
            logger.info("执行表聚合节点")
            return await self.table_aggregation.process(state)
        except Exception as e:
            logger.error(f"表聚合节点执行失败: {e}")
            state.add_error(f"表聚合执行失败: {e}")
            return state
    
    async def _table_discovery_node(self, state: AgentState) -> AgentState:
        """表发现节点"""
        try:
            logger.info("执行表发现节点")
            return await self.table_discovery.process(state)
        except Exception as e:
            logger.error(f"表发现节点执行失败: {e}")
            state.add_error(f"表发现执行失败: {e}")
            return state
    
    async def _table_matching_node(self, state: AgentState) -> AgentState:
        """表匹配节点"""
        try:
            logger.info("执行表匹配节点")
            return await self.table_matching.process(state)
        except Exception as e:
            logger.error(f"表匹配节点执行失败: {e}")
            state.add_error(f"表匹配执行失败: {e}")
            return state
    
    async def _finalization_node(self, state: AgentState) -> AgentState:
        """最终处理节点"""
        try:
            logger.info("执行最终处理节点")
            # 如果规划器还没有生成最终报告，让它生成
            if not state.final_report and state.current_step == "finalization":
                state = await self.planner.process(state)
            
            # 设置完成状态
            state.current_step = "completed"
            return state
        except Exception as e:
            logger.error(f"最终处理节点执行失败: {e}")
            state.add_error(f"最终处理执行失败: {e}")
            return state
    
    def _route_from_planner(self, state: AgentState) -> str:
        """从规划器节点的路由逻辑"""
        if state.error_messages:
            logger.warning("检测到错误，直接结束")
            return "end"
        
        if state.current_step == "column_discovery":
            return "column_discovery"
        elif state.current_step == "table_discovery":
            return "table_discovery"
        elif state.current_step == "finalization":
            return "finalization"
        else:
            logger.warning(f"未知的处理步骤: {state.current_step}")
            return "end"
    
    def _route_from_column_discovery(self, state: AgentState) -> str:
        """从列发现节点的路由逻辑"""
        if state.error_messages:
            return "finalization"  # 有错误时也尝试生成报告
        
        if state.current_step == "table_aggregation":
            return "table_aggregation"
        elif state.current_step == "finalization":
            return "finalization"
        else:
            return "end"
    
    def _route_from_table_aggregation(self, state: AgentState) -> str:
        """从表聚合节点的路由逻辑"""
        if state.current_step == "finalization":
            return "finalization"
        else:
            return "end"
    
    def _route_from_table_discovery(self, state: AgentState) -> str:
        """从表发现节点的路由逻辑"""
        if state.error_messages:
            return "finalization"
        
        if state.current_step == "table_matching":
            return "table_matching"
        elif state.current_step == "finalization":
            return "finalization"
        else:
            return "end"
    
    def _route_from_table_matching(self, state: AgentState) -> str:
        """从表匹配节点的路由逻辑"""
        if state.current_step == "finalization":
            return "finalization"
        else:
            return "end"
    
    async def run(self, initial_state: AgentState) -> AgentState:
        """运行工作流"""
        try:
            logger.info("开始执行数据湖工作流")
            logger.info(f"初始状态: 查询={initial_state.user_query}, 表数={len(initial_state.query_tables)}, 列数={len(initial_state.query_columns)}")
            
            # 验证初始状态
            self._validate_initial_state(initial_state)
            logger.info("初始状态验证完成")
            
            # 执行工作流
            logger.info("开始调用工作流应用...")
            result = await self.app.ainvoke(initial_state)
            
            # 确保返回的是 AgentState 对象
            if isinstance(result, dict):
                logger.info("将字典结果转换为 AgentState 对象")
                final_state = AgentState(**result)
            else:
                final_state = result
            
            logger.info("数据湖工作流执行完成")
            return final_state
            
        except Exception as e:
            logger.error(f"工作流执行失败: {e}")
            import traceback
            logger.error(f"详细错误信息: {traceback.format_exc()}")
            initial_state.add_error(f"工作流执行失败: {e}")
            return initial_state
    
    def _validate_initial_state(self, state: AgentState) -> None:
        """验证初始状态"""
        if not state.user_query:
            raise ValueError("用户查询不能为空")
        
        if not state.query_tables and not state.query_columns:
            raise ValueError("必须提供查询表或查询列信息")
        
        logger.info(f"初始状态验证通过: 查询={state.user_query}, "
                   f"表数量={len(state.query_tables)}, 列数量={len(state.query_columns)}")
    
    async def run_with_streaming(self, initial_state: AgentState):
        """流式运行工作流（生成器模式）"""
        try:
            logger.info("开始流式执行数据湖工作流")
            
            self._validate_initial_state(initial_state)
            
            # 使用流式API
            async for state in self.app.astream(initial_state):
                yield state
                
        except Exception as e:
            logger.error(f"流式工作流执行失败: {e}")
            initial_state.add_error(f"流式工作流执行失败: {e}")
            yield initial_state
    
    def get_workflow_info(self) -> Dict[str, Any]:
        """获取工作流信息"""
        return {
            "nodes": [
                "planner", "column_discovery", "table_aggregation",
                "table_discovery", "table_matching", "finalization"
            ],
            "strategies": {
                "bottom_up": ["planner", "column_discovery", "table_aggregation", "finalization"],
                "top_down": ["planner", "table_discovery", "table_matching", "finalization"]
            },
            "entry_point": "planner",
            "description": "基于LangGraph的数据湖模式匹配与数据发现工作流"
        }


# 工作流工厂函数
def create_workflow(use_optimized: bool = True, use_ultra: bool = False) -> DataLakesWorkflow:
    """创建数据湖工作流实例
    
    Args:
        use_optimized: 是否使用优化的工作流，默认为True
        use_ultra: 是否使用超优化工作流，默认为False
        
    Returns:
        DataLakesWorkflow, OptimizedDataLakesWorkflow 或 UltraOptimizedWorkflow 实例
    """
    if use_ultra:
        try:
            from src.core.ultra_optimized_workflow import create_ultra_optimized_workflow
            return create_ultra_optimized_workflow()
        except ImportError as e:
            logger.warning(f"无法导入超优化工作流，降级到优化版本: {e}")
            use_optimized = True
    
    if use_optimized:
        try:
            from src.core.optimized_workflow import create_optimized_workflow
            return create_optimized_workflow()
        except ImportError as e:
            logger.warning(f"无法导入优化工作流，使用基础版本: {e}")
            return DataLakesWorkflow()
    else:
        return DataLakesWorkflow()


# 便捷函数
async def discover_data(
    user_query: str,
    query_tables: List[Dict[str, Any]] = None,
    query_columns: List[Dict[str, Any]] = None,
    all_tables_data: List[Dict[str, Any]] = None,
    use_optimized: bool = True,
    use_ultra: bool = False
) -> AgentState:
    """便捷的数据发现函数
    
    Args:
        user_query: 用户查询
        query_tables: 查询表的数据
        query_columns: 查询列的数据
        all_tables_data: 所有表的数据（用于初始化优化工作流）
        use_optimized: 是否使用优化工作流
        use_ultra: 是否使用超优化工作流
    """
    from src.core.models import TableInfo, ColumnInfo
    
    # 转换输入数据
    parsed_tables = []
    if query_tables:
        for table_data in query_tables:
            table_info = TableInfo(**table_data)
            parsed_tables.append(table_info)
    
    parsed_columns = []
    if query_columns:
        for col_data in query_columns:
            col_info = ColumnInfo(**col_data)
            parsed_columns.append(col_info)
    
    # 创建初始状态
    initial_state = AgentState(
        user_query=user_query,
        query_tables=parsed_tables,
        query_columns=parsed_columns
    )
    
    # 创建工作流
    workflow = create_workflow(use_optimized=use_optimized, use_ultra=use_ultra)
    
    # 如果使用优化工作流且提供了所有表数据，则进行初始化
    if (use_optimized or use_ultra) and all_tables_data:
        from src.core.optimized_workflow import OptimizedDataLakesWorkflow
        from src.core.ultra_optimized_workflow import UltraOptimizedWorkflow
        
        if isinstance(workflow, (OptimizedDataLakesWorkflow, UltraOptimizedWorkflow)):
            logger.info(f"初始化{'超' if use_ultra else ''}优化工作流...")
            from src.utils.data_parser import parse_tables_data
            all_tables = parse_tables_data(all_tables_data)
            await workflow.initialize(all_tables)
            
            # 对于优化工作流，使用 run_optimized 方法
            all_table_names = [table_data.get('table_name', '') for table_data in all_tables_data]
            return await workflow.run_optimized(initial_state, all_table_names)
    
    # 运行基础工作流
    return await workflow.run(initial_state)