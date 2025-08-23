#!/bin/bash

echo "========================================="
echo "📊 完整数据集实验（优化版）"
echo "========================================="

# 创建必要目录
mkdir -p experiment_results
mkdir -p cache/ablation_examples/{opendata,webtable}

# 获取时间戳
timestamp=$(date +%Y%m%d_%H%M%S)

# 1. OpenData Subset实验
echo -e "\n[1/6] 🚀 OpenData Subset (JOIN & UNION, 各20个查询)..."
python three_layer_ablation_optimized.py \
    --task both \
    --dataset opendata \
    --dataset-type subset \
    --max-queries 20 \
    --simple \
    --output experiment_results/opendata_subset_${timestamp}.json

# 2. WebTable Subset实验
echo -e "\n[2/6] 🚀 WebTable Subset (JOIN & UNION, 各20个查询)..."
python three_layer_ablation_optimized.py \
    --task both \
    --dataset webtable \
    --dataset-type subset \
    --max-queries 20 \
    --simple \
    --output experiment_results/webtable_subset_${timestamp}.json

# 3. OpenData Complete JOIN
echo -e "\n[3/6] 🚀 OpenData Complete JOIN (50个查询)..."
python three_layer_ablation_optimized.py \
    --task join \
    --dataset opendata \
    --dataset-type complete \
    --max-queries 50 \
    --simple \
    --output experiment_results/opendata_complete_join_${timestamp}.json

# 4. OpenData Complete UNION
echo -e "\n[4/6] 🚀 OpenData Complete UNION (100个查询)..."
python three_layer_ablation_optimized.py \
    --task union \
    --dataset opendata \
    --dataset-type complete \
    --max-queries 100 \
    --simple \
    --output experiment_results/opendata_complete_union_${timestamp}.json

# 5. WebTable Complete JOIN
echo -e "\n[5/6] 🚀 WebTable Complete JOIN (50个查询)..."
python three_layer_ablation_optimized.py \
    --task join \
    --dataset webtable \
    --dataset-type complete \
    --max-queries 50 \
    --simple \
    --output experiment_results/webtable_complete_join_${timestamp}.json

# 6. WebTable Complete UNION
echo -e "\n[6/6] 🚀 WebTable Complete UNION (100个查询)..."
python three_layer_ablation_optimized.py \
    --task union \
    --dataset webtable \
    --dataset-type complete \
    --max-queries 100 \
    --simple \
    --output experiment_results/webtable_complete_union_${timestamp}.json

echo -e "\n========================================="
echo "✅ 所有实验完成！"
echo "结果保存在 experiment_results/ 目录"
echo "========================================="

# 汇总结果
echo -e "\n📊 实验结果汇总："
for file in experiment_results/*_${timestamp}.json; do
    if [ -f "$file" ]; then
        echo -e "\n$(basename $file):"
        python -c "
import json
with open('$file', 'r') as f:
    data = json.load(f)
    for task in data:
        if 'L1_L2_L3' in data[task]:
            metrics = data[task]['L1_L2_L3']['metrics']
            print(f'  {task.upper()}: F1={metrics[\"f1_score\"]:.3f}, Hit@1={metrics[\"hit@1\"]:.3f}')
"
    fi
done