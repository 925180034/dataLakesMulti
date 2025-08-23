#!/bin/bash

# OpenData数据集实验脚本

echo "========================================="
echo "OpenData 数据集三层消融实验"
echo "========================================="

# 1. Subset实验（快速）
echo -e "\n[1/4] 测试 OpenData JOIN Subset..."
python three_layer_ablation_optimized.py \
    --task join \
    --dataset examples/opendata/join_subset \
    --max-queries 20 \
    --output experiment_results/opendata_join_subset_$(date +%Y%m%d_%H%M%S).json

echo -e "\n[2/4] 测试 OpenData UNION Subset..."
python three_layer_ablation_optimized.py \
    --task union \
    --dataset examples/opendata/union_subset \
    --max-queries 20 \
    --output experiment_results/opendata_union_subset_$(date +%Y%m%d_%H%M%S).json

# 2. Complete实验（完整）
echo -e "\n[3/4] 测试 OpenData JOIN Complete..."
python three_layer_ablation_optimized.py \
    --task join \
    --dataset examples/opendata/join_complete \
    --max-queries 50 \
    --output experiment_results/opendata_join_complete_$(date +%Y%m%d_%H%M%S).json

echo -e "\n[4/4] 测试 OpenData UNION Complete..."
python three_layer_ablation_optimized.py \
    --task union \
    --dataset examples/opendata/union_complete \
    --max-queries 100 \
    --output experiment_results/opendata_union_complete_$(date +%Y%m%d_%H%M%S).json

echo -e "\n========================================="
echo "实验完成！结果保存在 experiment_results/ 目录"
echo "========================================="