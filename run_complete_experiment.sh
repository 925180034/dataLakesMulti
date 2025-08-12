#!/bin/bash
# 多智能体系统完整实验脚本

echo "=========================================="
echo "🚀 Multi-Agent System Complete Experiments"
echo "=========================================="

# 1. Subset数据集测试 (小规模验证)
echo -e "\n📝 Test 1: Subset Dataset (100 tables)"
echo "Running with 20 queries..."
python run_multi_agent_llm_enabled.py --dataset subset --queries 20 --workers 4
echo "✅ Subset test completed"

# 2. Complete数据集测试 (大规模)
echo -e "\n📝 Test 2: Complete Dataset (1,534 tables)"
echo "Running with 50 queries..."
python run_multi_agent_llm_enabled.py --dataset complete --queries 50 --workers 8
echo "✅ Complete test completed"

# 3. 高负载测试 (可选，取消注释以启用)
# echo -e "\n📝 Test 3: High Load Test (1,534 tables, 100 queries)"
# python run_multi_agent_llm_enabled.py --dataset complete --queries 100 --workers 8
# echo "✅ High load test completed"

echo -e "\n=========================================="
echo "✅ All experiments completed!"
echo "Results saved in: experiment_results/multi_agent_llm/"
echo "=========================================="

# 显示结果文件
echo -e "\nGenerated result files:"
ls -lht experiment_results/multi_agent_llm/*.json | head -5