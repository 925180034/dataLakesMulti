"""测试数据模型"""
import pytest
import sys
import os

# 添加项目根目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.core.models import TableInfo, ColumnInfo, AgentState, TaskStrategy


def test_column_info_creation():
    """测试列信息创建"""
    col = ColumnInfo(
        table_name='test_table',
        column_name='test_column',
        data_type='string',
        sample_values=['value1', 'value2']
    )
    
    assert col.table_name == 'test_table'
    assert col.column_name == 'test_column'
    assert col.data_type == 'string'
    assert len(col.sample_values) == 2
    assert col.full_name == 'test_table.test_column'


def test_table_info_creation():
    """测试表信息创建"""
    col1 = ColumnInfo(
        table_name='users',
        column_name='id',
        data_type='int',
        sample_values=['1', '2', '3']
    )
    
    col2 = ColumnInfo(
        table_name='users',
        column_name='name',
        data_type='string',
        sample_values=['Alice', 'Bob']
    )
    
    table = TableInfo(
        table_name='users',
        columns=[col1, col2]
    )
    
    assert table.table_name == 'users'
    assert len(table.columns) == 2
    
    # 测试获取列
    id_col = table.get_column('id')
    assert id_col is not None
    assert id_col.column_name == 'id'
    
    # 测试获取不存在的列
    missing_col = table.get_column('missing')
    assert missing_col is None


def test_agent_state_creation():
    """测试智能体状态创建"""
    state = AgentState(
        user_query='find similar tables',
        strategy=TaskStrategy.TOP_DOWN
    )
    
    assert state.user_query == 'find similar tables'
    assert state.strategy == TaskStrategy.TOP_DOWN
    assert state.current_step == 'planning'
    assert len(state.processing_log) == 0
    assert len(state.error_messages) == 0


def test_agent_state_logging():
    """测试智能体状态日志功能"""
    state = AgentState(user_query='test')
    
    # 测试添加日志
    state.add_log('Processing started')
    assert len(state.processing_log) == 1
    assert state.processing_log[0] == 'Processing started'
    
    # 测试添加错误
    state.add_error('Test error')
    assert len(state.error_messages) == 1
    assert state.error_messages[0] == 'Test error'
    
    # 测试清除中间结果
    state.column_matches = ['dummy']
    state.table_candidates = ['dummy']
    state.clear_intermediates()
    assert len(state.column_matches) == 0
    assert len(state.table_candidates) == 0


def test_task_strategy_enum():
    """测试任务策略枚举"""
    assert TaskStrategy.BOTTOM_UP == "BOTTOM_UP"
    assert TaskStrategy.TOP_DOWN == "TOP_DOWN"