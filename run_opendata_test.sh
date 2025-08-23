#!/bin/bash

echo "========================================="
echo "OpenData 数据集快速测试"
echo "========================================="

# 确保目录存在
mkdir -p experiment_results
mkdir -p cache/ablation_examples/opendata

# 1. Subset版本（同时运行JOIN和UNION）
echo -e "\n[测试 OpenData Subset - JOIN & UNION]"
echo "使用基础路径，自动检测join_subset和union_subset..."

python three_layer_ablation_optimized.py \
    --task both \
    --dataset examples/opendata \
    --max-queries 10 \
    --simple \
    --output experiment_results/opendata_subset_both_$(date +%Y%m%d_%H%M%S).json

echo -e "\n========================================="
echo "测试完成！"