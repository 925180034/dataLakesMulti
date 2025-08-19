#!/usr/bin/env python3
"""
验证优化参数和性能
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from src.agents.optimizer_agent import OptimizerAgent
from types import SimpleNamespace

def verify_new_parameters():
    """验证新的优化参数"""
    print("="*80)
    print("🔧 验证新的优化参数")
    print("="*80)
    
    optimizer = OptimizerAgent()
    
    # 测试JOIN参数
    print("\n📌 JOIN任务参数（优化后）:")
    join_task = SimpleNamespace(task_type='join')
    join_state = {
        'query_task': join_task,
        'all_tables': [{'name': f'table_{i}'} for i in range(100)]
    }
    join_result = optimizer.process(join_state)
    config = join_result['optimization_config']
    
    print(f"  置信度阈值: {config.llm_confidence_threshold} (目标: 0.10)")
    print(f"  最小分数: {config.aggregator_min_score} (目标: 0.01)")
    print(f"  最大结果: {config.aggregator_max_results} (目标: 500)")
    print(f"  向量TopK: {config.vector_top_k} (目标: 600)")
    
    # 测试UNION参数
    print("\n📌 UNION任务参数（优化后）:")
    union_task = SimpleNamespace(task_type='union')
    union_state = {
        'query_task': union_task,
        'all_tables': [{'name': f'table_{i}'} for i in range(100)]
    }
    union_result = optimizer.process(union_state)
    config = union_result['optimization_config']
    
    print(f"  置信度阈值: {config.llm_confidence_threshold} (目标: 0.15)")
    print(f"  最小分数: {config.aggregator_min_score} (目标: 0.03)")
    print(f"  最大结果: {config.aggregator_max_results} (目标: 200)")
    print(f"  向量TopK: {config.vector_top_k} (目标: 350)")
    
    print("\n" + "="*80)
    print("📊 性能优化建议")
    print("="*80)
    
    import multiprocessing
    cpu_count = multiprocessing.cpu_count()
    
    print(f"\n💻 CPU核心数: {cpu_count}")
    print(f"当前使用: 16个进程")
    print(f"推荐使用: {min(48, cpu_count//2)}个进程")
    print(f"最大可用: {min(64, cpu_count*3//4)}个进程")
    
    print("\n🚀 推荐命令:")
    print(f"""
# 快速测试（10个查询，32进程）
python three_layer_ablation_optimized.py \\
    --task join \\
    --dataset subset \\
    --max-queries 10 \\
    --workers 32

# 中等测试（30个查询，48进程）
python three_layer_ablation_optimized.py \\
    --task both \\
    --dataset subset \\
    --max-queries 30 \\
    --workers 48

# 完整测试（所有查询，48进程）
python three_layer_ablation_optimized.py \\
    --task both \\
    --dataset subset \\
    --max-queries all \\
    --workers 48
    """)
    
    print("\n📈 预期改进（激进参数）:")
    print("  JOIN F1-Score: 11.8% → 25-35% (极低阈值)")
    print("  UNION F1-Score: 30.9% → 40-50% (平衡优化)")
    print("  处理速度: 3-4倍提升（使用48进程）")
    print("\n⚠️ 注意: 极低阈值会增加API调用量，但能显著提升召回率")

if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.WARNING)
    verify_new_parameters()