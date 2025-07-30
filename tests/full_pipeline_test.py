#!/usr/bin/env python3
"""
完整流程测试 - 确保调用Gemini API
"""

import asyncio
import json
import sys
import logging
from pathlib import Path

# 添加项目根目录到Python路径
sys.path.insert(0, str(Path(__file__).parent.absolute()))

# 配置日志以查看API调用
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from src.core.workflow import discover_data
from src.utils.llm_client import llm_client
from src.core.models import ColumnInfo, TableInfo


async def test_gemini_api():
    """测试Gemini API连接"""
    print("=== 测试Gemini API连接 ===")
    
    try:
        response = await llm_client.generate(
            "Hello! Please respond with 'API connection successful'",
            "You are a helpful assistant."
        )
        print(f"✅ Gemini API响应: {response}")
        return True
    except Exception as e:
        print(f"❌ Gemini API连接失败: {e}")
        return False


async def test_full_pipeline():
    """测试完整的数据发现流程"""
    print("\n=== 完整流程测试 ===")
    
    # 创建简单测试数据
    test_query = "find columns similar to player name"
    
    # 创建查询列
    query_columns = [{
        "table_name": "players",
        "column_name": "player_name",
        "data_type": "string",
        "sample_values": ["John Doe", "Jane Smith", "Mike Johnson"]
    }]
    
    # 创建候选表（使用WebTable的部分数据）
    try:
        with open("examples/webtable_join_tables.json", 'r', encoding='utf-8') as f:
            tables_data = json.load(f)
        
        # 只使用前3个表进行快速测试
        candidate_tables = tables_data[:3]
        
        print(f"📊 测试数据准备:")
        print(f"   查询: {test_query}")
        print(f"   查询列: {query_columns[0]['table_name']}.{query_columns[0]['column_name']}")
        print(f"   候选表数量: {len(candidate_tables)}")
        
        # 执行发现流程
        print("\n🔍 开始执行发现流程...")
        logger.info("开始调用discover_data函数")
        
        result = await discover_data(
            user_query=test_query,
            query_tables=None,
            query_columns=query_columns
        )
        
        print("✅ 发现流程执行完成")
        
        # 显示结果
        print(f"\n📊 流程执行结果:")
        print(f"   策略: {result.strategy}")
        print(f"   处理步骤: {len(result.processing_steps)}")
        print(f"   列匹配数量: {len(result.column_matches)}")
        print(f"   表匹配数量: {len(result.table_matches)}")
        
        if result.processing_steps:
            print(f"\n🔄 执行步骤:")
            for i, step in enumerate(result.processing_steps):
                print(f"   {i+1}. {step}")
        
        if result.column_matches:
            print(f"\n🎯 列匹配结果:")
            for i, match in enumerate(result.column_matches[:3]):
                print(f"   {i+1}. {match.target_column} (置信度: {match.confidence:.3f})")
                print(f"      匹配类型: {match.match_type}")
                print(f"      原因: {match.reason[:100]}...")
        
        if result.table_matches:
            print(f"\n📋 表匹配结果:")
            for i, match in enumerate(result.table_matches[:3]):
                print(f"   {i+1}. {match.table_name} (置信度: {match.confidence:.3f})")
        
        # 检查是否有错误
        if result.error_messages:
            print(f"\n⚠️ 执行过程中的错误:")
            for error in result.error_messages:
                print(f"   - {error}")
        
        # 显示最终报告
        if hasattr(result, 'final_report') and result.final_report:
            print(f"\n📋 最终报告:\n{result.final_report}")
        
        return True
        
    except Exception as e:
        print(f"❌ 完整流程测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_specific_agent():
    """测试特定智能体的LLM调用"""
    print("\n=== 测试智能体LLM调用 ===")
    
    try:
        from src.agents.planner import PlannerAgent
        
        planner = PlannerAgent()
        
        # 创建简单的状态用于测试
        from src.core.models import AgentState, TaskStrategy
        
        state = AgentState(
            user_query="find similar tables for joining",
            query_columns=[],
            candidate_tables=[]
        )
        
        print("🧠 测试规划器智能体...")
        result_state = await planner.process(state)
        
        print(f"✅ 规划器执行完成")
        print(f"   选择策略: {result_state.strategy}")
        print(f"   处理步骤: {len(result_state.processing_steps)}")
        
        if result_state.processing_steps:
            for step in result_state.processing_steps:
                print(f"   - {step}")
        
        return True
        
    except Exception as e:
        print(f"❌ 智能体测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


async def main():
    """主测试函数"""
    print("🚀 开始完整的API和流程测试")
    
    # 测试1: API连接
    api_ok = await test_gemini_api()
    
    if not api_ok:
        print("❌ API连接失败，无法继续测试")
        return
    
    # 测试2: 智能体调用
    agent_ok = await test_specific_agent()
    
    # 测试3: 完整流程
    pipeline_ok = await test_full_pipeline()
    
    # 总结
    print(f"\n{'='*50}")
    print("🎯 测试总结:")
    print(f"   API连接: {'✅' if api_ok else '❌'}")
    print(f"   智能体调用: {'✅' if agent_ok else '❌'}")
    print(f"   完整流程: {'✅' if pipeline_ok else '❌'}")
    
    if api_ok and agent_ok and pipeline_ok:
        print("\n🎉 所有测试通过！系统正常工作，Gemini API调用成功！")
    else:
        print("\n⚠️ 部分测试未通过，需要进一步调试")


if __name__ == "__main__":
    asyncio.run(main())