#!/usr/bin/env python3
"""
简化演示：多智能体系统 + 三层加速
展示真正的多Agent协同工作原理
"""

import asyncio
import time
from typing import List, Dict, Any

# 模拟数据
from src.core.models import TableInfo, ColumnInfo

print("="*80)
print("🤖 多智能体系统架构演示")
print("="*80)

# 模拟Agent类
class DemoAgent:
    """演示用Agent"""
    def __init__(self, name: str, role: str):
        self.name = name
        self.role = role
        self.can_call_llm = True
        self.use_acceleration = True
        
    async def think(self, task: Dict) -> Dict:
        """Agent思考过程"""
        print(f"\n💭 {self.name} 正在思考...")
        print(f"   输入任务: {task.get('description', 'unknown')}")
        
        # 模拟决策过程
        if self.can_call_llm and "complex" in str(task).lower():
            decision = f"使用LLM分析复杂任务"
        else:
            decision = f"使用规则处理简单任务"
        
        print(f"   决策: {decision}")
        
        return {
            "strategy": decision,
            "complexity": 0.7 if "complex" in str(task).lower() else 0.3,
            "use_acceleration": self.use_acceleration
        }
    
    async def act(self, plan: Dict) -> Any:
        """Agent执行动作"""
        print(f"\n🎯 {self.name} 开始执行...")
        print(f"   策略: {plan.get('strategy')}")
        
        # 模拟执行
        await asyncio.sleep(0.1)  # 模拟处理时间
        
        result = {
            "agent": self.name,
            "result": f"{self.role}完成",
            "used_llm": "LLM" in plan.get('strategy', ''),
            "used_acceleration": plan.get('use_acceleration', False)
        }
        
        print(f"   ✅ 完成: {result['result']}")
        return result


class MultiAgentDemo:
    """多智能体系统演示"""
    
    def __init__(self):
        self.agents = {}
        self.acceleration_layers = {
            "Layer1_MetadataFilter": "规则筛选(10ms)",
            "Layer2_VectorSearch": "向量搜索(50ms)",
            "Layer3_LLMMatcher": "LLM验证(1-3s)"
        }
        
    async def initialize(self):
        """初始化系统"""
        print("\n📦 初始化多智能体系统...")
        
        # 创建6个专门的Agent
        self.agents = {
            "PlannerAgent": DemoAgent("PlannerAgent", "规划任务"),
            "AnalyzerAgent": DemoAgent("AnalyzerAgent", "分析数据"),
            "SearcherAgent": DemoAgent("SearcherAgent", "搜索候选"),
            "MatcherAgent": DemoAgent("MatcherAgent", "匹配验证"),
            "AggregatorAgent": DemoAgent("AggregatorAgent", "聚合结果"),
            "OptimizerAgent": DemoAgent("OptimizerAgent", "优化性能")
        }
        
        print(f"✅ 初始化完成: {len(self.agents)}个Agent")
        print(f"   Agents: {list(self.agents.keys())}")
        print(f"   三层加速: {list(self.acceleration_layers.keys())}")
        
    async def process_query(self, query: str, complexity: str = "simple"):
        """处理查询 - 展示Agent协同"""
        print(f"\n{'='*60}")
        print(f"🔍 处理查询: {query}")
        print(f"   复杂度: {complexity}")
        print(f"{'='*60}")
        
        start_time = time.time()
        
        # Step 1: OptimizerAgent 分析系统状态
        print("\n📊 Step 1: 系统优化分析")
        optimizer = self.agents["OptimizerAgent"]
        opt_task = {"description": "分析系统状态", "complexity": complexity}
        opt_plan = await optimizer.think(opt_task)
        opt_result = await optimizer.act(opt_plan)
        
        # Step 2: PlannerAgent 制定策略
        print("\n📋 Step 2: 任务规划")
        planner = self.agents["PlannerAgent"]
        plan_task = {"description": f"规划{query}", "complexity": complexity}
        plan = await planner.think(plan_task)
        plan_result = await planner.act(plan)
        
        # Step 3: AnalyzerAgent 分析数据
        print("\n🔬 Step 3: 数据分析")
        analyzer = self.agents["AnalyzerAgent"]
        analyze_task = {"description": "理解表结构", "complexity": complexity}
        analyze_plan = await analyzer.think(analyze_task)
        analyze_result = await analyzer.act(analyze_plan)
        
        # 决定是否使用三层加速
        if analyze_plan.get('use_acceleration'):
            print("\n⚡ 使用三层加速架构:")
            for layer, desc in self.acceleration_layers.items():
                print(f"   • {layer}: {desc}")
        
        # Step 4: SearcherAgent 查找候选
        print("\n🔎 Step 4: 搜索候选")
        searcher = self.agents["SearcherAgent"]
        search_task = {"description": "查找相关表", "complexity": complexity}
        search_plan = await searcher.think(search_task)
        
        # SearcherAgent 决定使用哪些加速层
        if search_plan.get('use_acceleration'):
            if plan.get('complexity', 0) > 0.5:
                print("   使用Layer1+Layer2混合搜索")
            else:
                print("   仅使用Layer1快速筛选")
        
        search_result = await searcher.act(search_plan)
        
        # Step 5: MatcherAgent 验证匹配
        print("\n✔️ Step 5: 匹配验证")
        matcher = self.agents["MatcherAgent"]
        match_task = {"description": "验证候选", "complexity": complexity}
        match_plan = await matcher.think(match_task)
        
        # MatcherAgent 决定是否调用Layer3 LLM
        if match_plan.get('use_acceleration') and plan.get('complexity', 0) > 0.6:
            print("   调用Layer3 LLM验证高复杂度匹配")
        else:
            print("   使用规则验证简单匹配")
        
        match_result = await matcher.act(match_plan)
        
        # Step 6: AggregatorAgent 整合结果
        print("\n📈 Step 6: 结果聚合")
        aggregator = self.agents["AggregatorAgent"]
        agg_task = {"description": "整合排序结果", "complexity": complexity}
        agg_plan = await aggregator.think(agg_task)
        
        # AggregatorAgent 可选择使用LLM重排序
        if agg_plan.get('strategy') and "LLM" in agg_plan['strategy']:
            print("   使用LLM重新排序Top结果")
        
        agg_result = await aggregator.act(agg_plan)
        
        # 统计
        elapsed = time.time() - start_time
        
        print(f"\n{'='*60}")
        print(f"✅ 查询处理完成!")
        print(f"   总耗时: {elapsed:.2f}秒")
        print(f"   参与Agents: {len(self.agents)}个")
        
        # 显示哪些Agent使用了LLM
        llm_agents = []
        acceleration_agents = []
        
        for result in [opt_result, plan_result, analyze_result, search_result, match_result, agg_result]:
            if result.get('used_llm'):
                llm_agents.append(result['agent'])
            if result.get('used_acceleration'):
                acceleration_agents.append(result['agent'])
        
        print(f"   使用LLM的Agents: {llm_agents if llm_agents else '无'}")
        print(f"   使用三层加速的Agents: {acceleration_agents[:3]}...")
        print(f"{'='*60}")
        
        return {
            "query": query,
            "elapsed": elapsed,
            "agents_used": len(self.agents),
            "llm_calls": len(llm_agents),
            "acceleration_used": len(acceleration_agents) > 0
        }


async def main():
    """主演示流程"""
    
    # 创建系统
    system = MultiAgentDemo()
    await system.initialize()
    
    # 测试不同复杂度的查询
    test_cases = [
        ("查找可以与users表JOIN的表", "simple"),
        ("分析复杂的多表关联关系", "complex"),
        ("优化大规模数据湖查询性能", "complex")
    ]
    
    print("\n" + "="*80)
    print("🚀 开始演示多Agent协同处理")
    print("="*80)
    
    for query, complexity in test_cases:
        result = await system.process_query(query, complexity)
        await asyncio.sleep(0.5)  # 演示间隔
    
    # 架构对比
    print("\n" + "="*80)
    print("📊 架构对比总结")
    print("="*80)
    
    print("\n1️⃣ 纯三层加速架构:")
    print("   Layer1 → Layer2 → Layer3 (固定流程)")
    print("   • 优点: 简单、快速、可预测")
    print("   • 缺点: 不灵活、每层固定调用")
    
    print("\n2️⃣ 多智能体系统 + 三层加速:")
    print("   Agents协同 + 选择性使用加速层")
    print("   • 优点: ")
    print("     - 每个Agent独立决策，灵活选择是否用LLM")
    print("     - Agent间协同优化，共享信息")
    print("     - 三层加速作为工具，按需调用")
    print("     - 支持复杂任务的智能处理")
    print("   • 特点:")
    print("     - PlannerAgent: 复杂查询用LLM，简单用规则")
    print("     - SearcherAgent: 灵活选择Layer1/2/混合")
    print("     - MatcherAgent: 智能决定是否调用Layer3")
    print("     - OptimizerAgent: 动态调整系统配置")
    
    print("\n✨ 核心区别:")
    print("   纯三层: 数据流过固定管道")
    print("   多Agent: 智能体协作决策 + 按需加速")
    
    print("\n" + "="*80)
    print("✅ 多智能体系统演示完成!")
    print("="*80)


if __name__ == "__main__":
    asyncio.run(main())