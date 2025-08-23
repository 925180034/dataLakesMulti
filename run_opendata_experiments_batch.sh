#!/bin/bash

# OpenData批量实验脚本

echo "========================================="
echo "OpenData 数据集实验"
echo "========================================="

# 创建结果目录
mkdir -p experiment_results

# 1. Subset实验（快速验证）
echo -e "\n[1/4] OpenData JOIN Subset (前20个查询)..."
python three_layer_ablation_optimized.py \
    --task join \
    --dataset examples/opendata/join_subset \
    --max-queries 20 \
    --simple \
    --output experiment_results/opendata_join_subset_$(date +%Y%m%d_%H%M%S).json

echo -e "\n[2/4] OpenData UNION Subset (前20个查询)..."
python three_layer_ablation_optimized.py \
    --task union \
    --dataset examples/opendata/union_subset \
    --max-queries 20 \
    --simple \
    --output experiment_results/opendata_union_subset_$(date +%Y%m%d_%H%M%S).json

# 2. Complete实验（完整评估）
echo -e "\n[3/4] OpenData JOIN Complete (前50个查询)..."
python three_layer_ablation_optimized.py \
    --task join \
    --dataset examples/opendata/join_complete \
    --max-queries 50 \
    --simple \
    --output experiment_results/opendata_join_complete_$(date +%Y%m%d_%H%M%S).json

echo -e "\n[4/4] OpenData UNION Complete (前100个查询)..."
python three_layer_ablation_optimized.py \
    --task union \
    --dataset examples/opendata/union_complete \
    --max-queries 100 \
    --simple \
    --output experiment_results/opendata_union_complete_$(date +%Y%m%d_%H%M%S).json

echo -e "\n========================================="
echo "✅ 实验完成！"
echo "结果保存在 experiment_results/ 目录"
echo "========================================="