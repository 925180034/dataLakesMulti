#!/bin/bash
# 完整测试脚本 - 包含索引构建和查询测试

echo "============================================"
echo "数据湖多智能体系统 - 完整测试流程"
echo "============================================"

# 1. 清理旧索引
echo ""
echo "Step 1: 清理旧索引..."
rm -rf ./data/vector_db/
rm -rf ./data/index_db/
rm -rf ./cache/
echo "✅ 清理完成"

# 2. 构建新索引
echo ""
echo "Step 2: 构建索引..."
echo "使用数据集: examples/final_subset_tables.json (100个表)"
python run_cli.py index-tables --tables examples/final_subset_tables.json
echo "✅ 索引构建完成"

# 3. 运行测试查询
echo ""
echo "Step 3: 运行测试查询..."
echo "查询: 寻找可连接的表"
python run_cli.py discover -q "find joinable tables for csvData6444295__5" -t examples/final_subset_tables.json --all-tables examples/final_subset_tables.json -f json > query_result.json
echo "✅ 查询完成，结果保存到 query_result.json"

# 4. 显示查询结果
echo ""
echo "Step 4: 查询结果预览:"
python -c "
import json
with open('query_result.json') as f:
    result = json.load(f)
    # 兼容不同的字段名
    if 'results' in result:
        print(f'✅ 找到 {len(result[\"results\"])} 个匹配表:')
        for i, match in enumerate(result['results'][:5], 1):
            target = match.get('target_table', match.get('table', 'unknown'))
            score = match.get('score', 0)
            print(f'  {i}. {target} - 分数: {score:.2f}')
    elif 'matches' in result:
        print(f'✅ 找到 {len(result[\"matches\"])} 个匹配表:')
        for i, match in enumerate(result['matches'][:5], 1):
            print(f'  {i}. {match[\"table\"]} - 分数: {match[\"score\"]}')
    else:
        print('❌ 未找到匹配结果')
        print(f'JSON结构: {list(result.keys())}')
"

# 5. 运行性能测试
echo ""
echo "Step 5: 运行性能测试..."
python test_quick_performance.py

# 6. 显示性能结果
echo ""
echo "Step 6: 性能测试结果:"
python -c "
import json
try:
    with open('quick_performance_results.json') as f:
        results = json.load(f)
        print(f'首次查询: {results[\"first_run\"]:.2f}秒')
        print(f'缓存查询: {results[\"second_run\"]:.2f}秒')
        if results['target_met']:
            print('✅ 性能达标（3-8秒目标）')
        else:
            print('⚠️ 性能未达标')
except:
    print('性能结果文件未找到')
"

echo ""
echo "============================================"
echo "测试完成！"
echo "============================================"
echo ""
echo "后续步骤:"
echo "1. 查看详细结果: cat query_result.json"
echo "2. 运行更多查询: python run_cli.py discover -q \"your query\" -t examples/final_subset_tables.json"
echo "3. 查看性能报告: cat PERFORMANCE_OPTIMIZATION_REPORT.md"