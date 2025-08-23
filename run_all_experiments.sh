#!/bin/bash

echo "========================================="
echo "完整数据集实验批量运行"
echo "========================================="

# 创建必要目录
mkdir -p experiment_results
mkdir -p cache/ablation_examples/{opendata,webtable}

# 1. OpenData Subset
echo -e "\n[1/4] OpenData Subset (JOIN & UNION, 20 queries each)..."
python three_layer_ablation_optimized.py \
    --task both \
    --dataset examples/opendata \
    --max-queries 20 \
    --simple \
    --output experiment_results/opendata_subset_$(date +%Y%m%d_%H%M%S).json

# 2. WebTable Subset  
echo -e "\n[2/4] WebTable Subset (JOIN & UNION, 20 queries each)..."
python three_layer_ablation_optimized.py \
    --task both \
    --dataset examples/webtable \
    --max-queries 20 \
    --simple \
    --output experiment_results/webtable_subset_$(date +%Y%m%d_%H%M%S).json

# 3. OpenData Complete (分开运行因为数据量大)
echo -e "\n[3/4] OpenData Complete JOIN (50 queries)..."
python three_layer_ablation_optimized.py \
    --task join \
    --dataset examples/opendata/join_complete \
    --max-queries 50 \
    --simple \
    --output experiment_results/opendata_join_complete_$(date +%Y%m%d_%H%M%S).json

echo -e "\n[4/4] OpenData Complete UNION (100 queries)..."
python three_layer_ablation_optimized.py \
    --task union \
    --dataset examples/opendata/union_complete \
    --max-queries 100 \
    --simple \
    --output experiment_results/opendata_union_complete_$(date +%Y%m%d_%H%M%S).json

echo -e "\n========================================="
echo "✅ 所有实验完成！"
echo "结果保存在 experiment_results/ 目录"
echo "========================================="