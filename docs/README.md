# 📚 数据湖多智能体系统文档中心

欢迎来到数据湖多智能体系统的文档中心。本目录包含了系统的所有技术文档、设计文档和使用指南。

## 🗂️ 文档索引

### 核心文档

1. **[系统架构与实施计划](SYSTEM_ARCHITECTURE_AND_PLAN.md)** ⭐
   - 三层加速架构设计
   - 支持10,000+表的实用方案
   - 环境配置指南
   - 性能优化策略
   - *推荐首先阅读*

2. **[实用架构详细设计](PRACTICAL_ARCHITECTURE_PLAN.md)** 🆕
   - 详细的三层架构实现
   - 并行处理框架
   - 多级缓存策略
   - 性能分析和优化

2. **[快速开始指南](QUICK_START.md)**
   - 环境搭建步骤
   - 基本使用示例
   - 常见问题解答

3. **[项目设计文档](Project-Design-Document.md)**
   - 原始设计理念
   - 多智能体协作框架
   - 数据格式说明

### 技术分析文档

4. **[LakeBench项目分析](lakebench_analysis.md)**
   - LakeBench论文深度解析
   - 11种数据发现算法对比
   - 性能优化insights

5. **[架构图解](architecture_diagram.md)**
   - 系统组件关系图
   - 数据流程图
   - 智能体交互图

6. **[系统架构详细图表](SYSTEM_ARCHITECTURE_DIAGRAMS.md)** 🆕⭐
   - 系统整体架构图
   - 数据流图
   - 查询处理流程图
   - 多智能体协作流程图
   - 性能优化组件图
   - 三层加速架构详细图
   - 系统部署架构图

### 测试与评估

7. **[WebTable测试报告](WEBTABLE_TEST_REPORT.md)**
   - WebTable数据集测试结果
   - 性能基准测试
   - 问题分析与解决方案

7. **[环境需求说明](environment_requirements.md)**
   - 硬件要求
   - 软件依赖
   - 配置建议

## 🚀 快速导航

### 新用户入门
1. 阅读 [快速开始指南](QUICK_START.md)
2. 查看 [系统架构与实施计划](SYSTEM_ARCHITECTURE_AND_PLAN.md) 的环境配置部分
3. 运行第一个实验

### 开发者指南
1. 深入了解 [项目设计文档](Project-Design-Document.md)
2. 研究 [LakeBench项目分析](lakebench_analysis.md)
3. 查看架构实现细节

### 实验与评估
1. 使用统一实验脚本：
   ```bash
   python unified_experiment.py 50 subset 30
   ```
2. 查看实验结果：`experiment_results/`目录
3. 分析评价指标

## 📊 实验运行指南

### 推荐实验流程

1. **环境验证**
   ```bash
   ./run_experiment.sh config
   ```

2. **快速测试（3个查询）**
   ```bash
   python unified_experiment.py 3 subset 30
   ```

3. **标准评估（50个查询）**
   ```bash
   python unified_experiment.py 50 subset 30
   ```

4. **完整评估（100个查询）**
   ```bash
   python unified_experiment.py 100 complete 60
   ```

### 评价指标说明

系统自动计算以下指标：
- **Precision（精确率）**：返回结果的准确性
- **Recall（召回率）**：找到所有正确答案的能力
- **F1-Score**：精确率和召回率的综合指标
- **成功率**：查询完成的比例
- **响应时间**：平均处理时间

所有结果保存在 `experiment_results/` 目录中。

## 🔧 配置优化

### 性能优化配置
使用优化配置以获得更好的性能：
```bash
cp config_optimized.yml config.yml
```

### 环境变量配置
创建 `.env` 文件：
```bash
GEMINI_API_KEY=your_api_key
CACHE_ENABLED=true
DEBUG=false
```

## 📈 最新进展

- ✅ **Phase 1 完成**：核心功能实现
- 🔄 **Phase 2 进行中**：性能优化
- 📋 **Phase 3 计划中**：完整部署

## 🤝 贡献指南

欢迎贡献代码和文档！请：
1. 遵循现有的文档格式
2. 更新相关的索引和链接
3. 运行测试确保功能正常

## 📞 获取帮助

- 查看 [快速开始指南](QUICK_START.md) 的故障排除部分
- 检查 [系统架构文档](SYSTEM_ARCHITECTURE_AND_PLAN.md) 的性能优化建议
- 查看实验日志：`logs/` 目录

---

**文档更新日期**: 2024年7月30日  
**版本**: v3.0