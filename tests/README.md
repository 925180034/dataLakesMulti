# 测试文档

本目录包含数据湖多智能体系统的所有测试文件。

## 测试文件说明

### 单元测试
- `test_models.py` - 核心数据模型的单元测试

### 功能测试
- `archived/test_hybrid_similarity.py` - 混合相似度算法测试（已归档）
- `archived/test_integration.py` - 系统集成测试（已归档）
- `archived/test_scalable_search.py` - 可扩展搜索功能测试（已归档）
- `test_union_scenario.py` - Union场景测试
- `test_webtable_dataset.py` - WebTable数据集测试

### 系统测试
- `full_pipeline_test.py` - 完整管道端到端测试
- `simple_webtable_test.py` - WebTable简单功能测试
- `test_simple_fixed.py` - 修复后的简化系统测试
- `test_fixed_workflow.py` - 修复后的完整工作流测试

### Phase 2 优化测试
- `test_phase2_optimized.py` - 阶段二优化性能测试
- `test_webtable_phase2_optimized.py` - WebTable阶段二优化测试
- `test_phase1_performance.py` - 阶段一性能基准测试
- `test_phase2_performance.py` - 阶段二性能验证测试
- `test_phase2_optimization_validation.py` - 阶段二优化验证测试

### 测试结果
- `results/` - 测试结果文件存储目录

## 运行测试

### 运行所有测试
```bash
python -m pytest tests/
```

### 运行特定测试
```bash
python -m pytest tests/test_models.py -v
```

### 运行集成测试
```bash
python tests/full_pipeline_test.py
```

## 测试数据

测试使用的示例数据位于 `examples/` 目录中：
- `sample_tables.json` - 示例表数据
- `sample_columns.json` - 示例列数据
- `webtable_*` - WebTable数据集相关文件

## 环境要求

确保已安装所有依赖：
```bash
pip install -r requirements.txt
```

并配置好API密钥（通常是GEMINI_API_KEY）。