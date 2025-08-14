# 系统脚本说明

## 核心脚本

### 主入口
- **`run_cli.py`** - 命令行接口主入口
  ```bash
  python run_cli.py discover -q "find joinable tables" -t examples/final_subset_tables.json
  ```

### 实验运行
- **`run_cached_experiments.py`** - 带缓存的批量实验（推荐使用）
  ```bash
  python run_cached_experiments.py --task join --dataset subset --max-queries 10
  ```

- **`final_test.py`** - 最终系统测试脚本
  ```bash
  python final_test.py  # 运行5个测试查询验证系统
  ```

- **`demo_system.py`** - 系统演示脚本
  ```bash
  python demo_system.py  # 快速演示系统功能
  ```

### 工具脚本
- **`evaluate_with_metrics.py`** - 评估指标计算模块（被其他脚本调用）
- **`fix_evaluation.py`** - 修复ground truth格式和重新计算指标
- **`precompute_embeddings.py`** - 预计算表嵌入向量

### LangGraph系统
- **`run_langgraph_system.py`** - LangGraph多智能体系统主脚本
  ```bash
  python run_langgraph_system.py
  ```

## 使用建议

1. **快速测试系统**：
   ```bash
   python final_test.py
   ```

2. **运行完整实验**：
   ```bash
   python run_cached_experiments.py --task join --dataset subset --max-queries 20
   ```

3. **单个查询测试**：
   ```bash
   python run_cli.py discover -q "find joinable tables for table_name" -t examples/final_subset_tables.json
   ```

## 已删除的临时脚本
- test_caching.py
- quick_cache_test.py
- test_final_system.py
- run_complete_system.py
- quick_system_test.py
- run_complete_experiments.py
- run_experiments.py
- enhanced_ablation_experiment.py
- run_optimized_ablation.py

这些脚本是开发过程中的临时版本，其功能已被最终版本脚本完全覆盖。