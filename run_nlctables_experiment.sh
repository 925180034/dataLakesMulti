#!/bin/bash

echo "================================================"
echo "NLCTables消融实验运行脚本"
echo "================================================"

# 检查数据是否存在，不存在则提取
if [ ! -d "examples/nlctables/union_subset" ]; then
    echo "📊 数据不存在，正在提取..."
    python extract_nlctables_full.py --task union --subset-size 100 --complete-size 3000
fi

echo ""
echo "1. 运行快速测试（10个查询，subset数据集）..."
echo "================================================"
python nlctables_ablation_optimized.py \
    --task union \
    --dataset nlctables \
    --dataset-type subset \
    --max-queries 10 \
    --workers 4 \
    --output experiment_results/nlctables

echo ""
echo "2. 运行标准测试（100个查询，subset数据集）..."
echo "================================================"
echo "python nlctables_ablation_optimized.py --task union --dataset nlctables --dataset-type subset --workers 8"

echo ""
echo "3. 运行完整测试（所有查询，complete数据集）..."
echo "================================================"
echo "提示: 使用 --dataset-type complete 运行完整数据集"
echo "python nlctables_ablation_optimized.py --task union --dataset nlctables --dataset-type complete"

echo ""
echo "4. 查看最新结果..."
echo "================================================"
latest_result=$(ls -t experiment_results/nlctables/nlctables_ablation_*.json 2>/dev/null | head -1)
if [ -n "$latest_result" ]; then
    echo "最新结果文件: $latest_result"
    python -c "
import json
with open('$latest_result', 'r') as f:
    data = json.load(f)
    # 处理新格式（任务名作为key）
    for task_name, task_results in data.items():
        print(f\"任务: {task_name}\")
        for layer, layer_data in task_results.items():
            metrics = layer_data.get('metrics', {})
            print(f\"  {layer}: F1={metrics.get('f1_score', 0):.3f}, Hit@1={metrics.get('hit@1', 0):.3f}, Time={layer_data.get('avg_time', 0):.2f}s\")
    "
else
    echo "没有找到结果文件"
fi

echo ""
echo "✅ 实验完成！"
echo ""
echo "更多选项:"
echo "  --task: union, join, both"
echo "  --dataset-type: subset, complete"
echo "  --max-queries: 数字或留空使用全部"
echo "  --challenging: 使用挑战性查询"
echo "  --simple: 使用简单查询"
echo "  --workers: 并行进程数"
echo "  --output: 输出目录"