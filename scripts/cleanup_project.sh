#!/bin/bash
# 项目清理脚本 - 清理临时文件和缓存

echo "🧹 开始清理项目..."

# 1. 清理Python缓存
echo "清理Python缓存..."
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null
find . -type f -name "*.pyc" -delete 2>/dev/null
find . -type f -name "*.pyo" -delete 2>/dev/null
find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null

# 2. 清理临时输出文件
echo "清理临时输出文件..."
rm -f *_result.json 2>/dev/null
rm -f *_results.json 2>/dev/null
rm -f test_*.json 2>/dev/null

# 3. 清理日志文件（保留目录）
echo "清理日志文件..."
rm -f logs/*.log 2>/dev/null

# 4. 清理缓存目录（可选）
read -p "是否清理缓存目录? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "清理缓存..."
    rm -rf cache/* 2>/dev/null
fi

# 5. 清理实验结果（可选）
read -p "是否清理实验结果? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "清理实验结果..."
    rm -rf experiment_results/* 2>/dev/null
fi

# 6. 显示清理结果
echo ""
echo "✅ 清理完成！"
echo ""
echo "当前项目结构:"
echo "==============="
ls -la | grep -E "^d" | grep -v "^\." | awk '{print "📁 " $NF}'
echo ""
echo "根目录文件数: $(ls -1 | wc -l)"
echo "==============="