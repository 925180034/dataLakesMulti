#\!/bin/bash
# 多智能体系统完整实验脚本

echo "=========================================="
echo "🚀 Multi-Agent System Complete Experiments"
echo "=========================================="

# 1. 小规模测试 (验证系统)
echo -e "\n📝 Test 1: Small Scale (subset, 10 queries)"
python run_final_multi_agent_test.py --dataset subset --join-queries 5 --union-queries 5 --workers 2

# 2. 中等规模测试 (subset完整)
echo -e "\n📝 Test 2: Medium Scale (subset, 50 queries)"
python run_final_multi_agent_test.py --dataset subset --join-queries 25 --union-queries 25 --workers 4

# 3. 完整数据集测试 (complete)
echo -e "\n📝 Test 3: Full Scale (complete, 100 queries)"
python run_final_multi_agent_test.py --dataset complete --join-queries 50 --union-queries 50 --workers 4

# 4. 大规模测试 (可选)
# echo -e "\n📝 Test 4: Large Scale (complete, 200 queries)"
# python run_final_multi_agent_test.py --dataset complete --join-queries 100 --union-queries 100 --workers 8

echo -e "\n✅ All experiments completed\!"
echo "Check experiment_results/ directory for results"
