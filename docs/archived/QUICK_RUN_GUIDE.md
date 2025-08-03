# 快速运行指南 - 数据湖多智能体系统

## 步骤 1: 构建索引（首次运行必需）

```bash
# 构建向量索引和元数据索引
python run_cli.py index-tables --tables examples/final_subset_tables.json

# 或者使用完整数据集（1534个表）
python run_cli.py index-tables --tables examples/final_complete_tables.json
```

**说明**: 
- 这会构建HNSW向量索引和元数据索引
- 索引文件保存在 `./data/vector_db/` 目录
- 首次构建需要1-2分钟

## 步骤 2: 运行查询

### 方法 1: CLI命令行查询（推荐）
```bash
# 单个查询（需要指定--all-tables参数用于优化工作流）
python run_cli.py discover -q "find joinable tables for users" -t examples/final_subset_tables.json --all-tables examples/final_subset_tables.json -f json

# 指定具体表名查询
python run_cli.py discover -q "find tables similar to csvData6444295__5" -t examples/final_subset_tables.json --all-tables examples/final_subset_tables.json -f markdown

# 如果不想使用优化版本，可以添加 --no-optimize 参数
python run_cli.py discover -q "find joinable tables" -t examples/final_subset_tables.json --no-optimize -f json
```

### 方法 2: Python脚本查询
```python
import asyncio
import json
from src.core.workflow import discover_data

async def run_query():
    # 加载数据
    with open('examples/final_subset_tables.json') as f:
        all_tables = json.load(f)
    
    # 选择查询表
    query_table = all_tables[0]
    
    # 执行查询（使用优化工作流）
    result = await discover_data(
        user_query=f"Find joinable tables for {query_table['table_name']}",
        query_tables=[query_table],
        all_tables_data=all_tables,
        use_optimized=True  # 使用优化版本
    )
    
    # 打印结果
    if result.table_matches:
        for match in result.table_matches[:5]:
            print(f"{match.target_table}: {match.score:.2f}")

asyncio.run(run_query())
```

### 方法 3: 批量评估
```bash
# 运行完整评估（包含性能指标）
python evaluate_with_metrics.py

# 查看结果
cat evaluation_results.json
```

## 步骤 3: 性能测试

```bash
# 快速性能测试（单个查询）
python test_quick_performance.py

# 批量性能测试（5个查询）
python test_optimized_performance.py
```

## 注意事项

### ⚠️ 首次运行
1. **必须先构建索引** - 否则查询会很慢
2. **初始化需要时间** - 首次查询包含模型加载（约10秒）
3. **后续查询很快** - 利用缓存后只需0.07-3秒

### 🚀 性能优化提示
1. **使用优化工作流**: 设置 `use_optimized=True`
2. **批量查询**: 一次处理多个查询以复用初始化
3. **缓存利用**: 相似查询会自动使用缓存

### 📊 预期性能
- **首次查询**: 10-15秒（包含初始化）
- **后续查询**: 0.07-3秒（使用缓存）
- **批量查询**: 平均1-2秒/查询

## 示例命令序列

```bash
# 1. 激活环境
conda activate data_lakes_multi

# 2. 构建索引（只需运行一次）
python run_cli.py index-tables --tables examples/final_subset_tables.json

# 3. 运行查询测试（使用优化版本）
python run_cli.py discover -q "find joinable tables" -t examples/final_subset_tables.json --all-tables examples/final_subset_tables.json -f json

# 4. 查看性能
python test_quick_performance.py
```

## 故障排除

### 问题: "索引未找到"
```bash
# 重新构建索引
rm -rf ./data/vector_db/
python run_cli.py index-tables -t examples/final_subset_tables.json
```

### 问题: "查询很慢"
```bash
# 确认使用优化版本
# 在代码中设置 use_optimized=True
# 或使用 config.yml 中的 use_optimized_workflow: true
```

### 问题: "内存不足"
```bash
# 减少批处理大小
# 编辑 config.yml
# batch_size: 5  # 从10减少到5
```

---

**准备就绪！** 按照上述步骤即可快速运行系统。