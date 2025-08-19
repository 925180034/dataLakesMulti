#!/bin/bash

echo "=========================================="
echo "🚀 性能优化测试脚本"
echo "=========================================="
echo ""
echo "您的系统有128个CPU核心，现在测试优化后的性能"
echo ""

# 颜色定义
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${GREEN}📊 第1步: 快速验证（5个查询，32进程）${NC}"
echo "测试优化的参数是否有效提升性能..."
echo ""

CMD1="python three_layer_ablation_optimized.py --task join --dataset subset --max-queries 5 --workers 32"
echo -e "${YELLOW}执行: $CMD1${NC}"
echo ""
echo "按回车开始测试（预计2-3分钟）..."
read

time $CMD1

echo ""
echo -e "${GREEN}📊 第2步: 中等规模测试（15个查询，48进程）${NC}"
echo "如果上面的测试显示F1有改善，可以运行这个测试..."
echo ""

CMD2="python three_layer_ablation_optimized.py --task both --dataset subset --max-queries 15 --workers 48"
echo -e "${YELLOW}执行: $CMD2${NC}"
echo ""
echo "按回车继续（预计5-8分钟）..."
read

time $CMD2

echo ""
echo -e "${GREEN}📊 第3步: 完整subset测试（50个查询，48进程）${NC}"
echo "完整测试subset数据集的所有查询..."
echo ""

CMD3="python three_layer_ablation_optimized.py --task both --dataset subset --max-queries all --workers 48"
echo -e "${YELLOW}执行: $CMD3${NC}"
echo ""
echo "按回车继续（预计15-20分钟）..."
read

time $CMD3

echo ""
echo "=========================================="
echo "✅ 测试完成！"
echo "=========================================="
echo ""
echo "📊 关键指标对比:"
echo "之前的结果:"
echo "  - JOIN F1: 11.8% (Precision: 10.0%, Recall: 14.6%)"
echo "  - UNION F1: 30.9% (Precision: 81.8%, Recall: 20.4%)"
echo ""
echo "优化后的目标:"
echo "  - JOIN F1: 25-35% (极低阈值提升召回率)"
echo "  - UNION F1: 40-50% (平衡precision和recall)"
echo ""
echo "💡 如果性能改善明显，可以考虑测试complete数据集"
echo "   python three_layer_ablation_optimized.py --task join --dataset complete --max-queries 100 --workers 48"